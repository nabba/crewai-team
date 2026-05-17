"""Tests for app.substrate.migration live mode — WP D Phase 3.

The live path actually shells out to terraform / gcloud / kubectl. Tests
exercise the orchestrator's logic WITHOUT real cloud invocation:

  * ``_shell`` has a hard execute-gate: it requires either
    ``execute=True`` arg or ``BOTARMY_MIGRATE_LIVE_EXECUTE=1`` env var.
    Tests set neither, so _shell returns ``<dry: ...>`` placeholders.
  * The orchestrator still records steps, evaluates gates, emits
    ledger events, writes reports — every code path is exercised
    EXCEPT actual cloud calls.

What's pinned:
  * Every gate enforces what it claims (typed phrase, project,
    budget cap, bundle freshness, cloud_doctor=OK).
  * Live orchestrator halts on first step failure (vs dry-run which
    continues for report completeness).
  * Each successful step emits a continuity-ledger landmark.
  * Report file persisted with dry_run=False.
"""
import io
import json
import os
import sys
import tarfile
import time
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Shared fixtures ─────────────────────────────────────────────────


@pytest.fixture
def isolated_workspace(tmp_path, monkeypatch):
    from app import paths as _paths
    monkeypatch.setattr(_paths, "WORKSPACE_ROOT", tmp_path)
    return tmp_path


@pytest.fixture
def fresh_bundle(isolated_workspace):
    """Drop a fresh DR tarball into workspace/backups/dr/."""
    backup_dir = isolated_workspace / "backups" / "dr"
    backup_dir.mkdir(parents=True, exist_ok=True)
    tarball = backup_dir / "dr_test.tar.gz"
    manifest = {
        "ok": True,
        "started_at": "2026-05-17T00:00:00+00:00",
        "subia_integrity_at_export": {"ok": True, "n_files": 164},
        "chromadb": [], "postgres": [], "ledgers": [],
        "total_rows_chromadb": 0, "total_rows_postgres": 0, "total_bytes": 0,
    }
    with tarfile.open(tarball, "w:gz") as tf:
        data = json.dumps(manifest).encode()
        info = tarfile.TarInfo(name="manifest.json")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    return tarball


@pytest.fixture
def mock_cloud_doctor_ok(monkeypatch):
    """cloud_doctor.check_readiness returns overall=OK."""
    from app.substrate import cloud_doctor as cd

    class _FakeReadiness:
        target = "gcp"
        timestamp = "2026-05-17T00:00:00+00:00"
        overall = "OK"
        probes = []

    monkeypatch.setattr(cd, "check_readiness", lambda target="gcp": _FakeReadiness())


@pytest.fixture
def mock_cloud_doctor_degraded(monkeypatch):
    """cloud_doctor.check_readiness returns overall=DEGRADED."""
    from app.substrate import cloud_doctor as cd

    class _FakeReadiness:
        target = "gcp"
        timestamp = "2026-05-17T00:00:00+00:00"
        overall = "DEGRADED"
        probes = []

    monkeypatch.setattr(cd, "check_readiness", lambda target="gcp": _FakeReadiness())


# ── Gate evaluation (pure-function tests) ───────────────────────────


class TestGateEvaluation:
    def _all_args(self, **overrides):
        defaults = dict(
            target="gcp",
            project_id="my-proj",
            confirm_phrase="MIGRATE TO GCP",
            budget_cap_usd=300.0,
            cost_estimate_monthly_usd=170.0,
            bundle_age_days=0.5,
            cloud_doctor_overall="OK",
        )
        defaults.update(overrides)
        return defaults

    def test_all_gates_pass_with_defaults(self):
        from app.substrate.migration import evaluate_live_gates
        gates = evaluate_live_gates(**self._all_args())
        assert all(g.passed for g in gates), [g.detail for g in gates if not g.passed]

    def test_wrong_typed_phrase_fails(self):
        from app.substrate.migration import evaluate_live_gates
        gates = evaluate_live_gates(**self._all_args(confirm_phrase="migrate to gcp"))
        tp = next(g for g in gates if g.name == "typed_phrase")
        assert not tp.passed

    def test_empty_typed_phrase_fails(self):
        from app.substrate.migration import evaluate_live_gates
        gates = evaluate_live_gates(**self._all_args(confirm_phrase=""))
        tp = next(g for g in gates if g.name == "typed_phrase")
        assert not tp.passed

    def test_missing_project_fails(self):
        from app.substrate.migration import evaluate_live_gates
        gates = evaluate_live_gates(**self._all_args(project_id=None))
        pid = next(g for g in gates if g.name == "project_id")
        assert not pid.passed

    def test_budget_cap_exceeded_fails(self):
        from app.substrate.migration import evaluate_live_gates
        gates = evaluate_live_gates(**self._all_args(
            cost_estimate_monthly_usd=500.0, budget_cap_usd=300.0,
        ))
        bc = next(g for g in gates if g.name == "budget_cap")
        assert not bc.passed

    def test_budget_cap_exact_match_passes(self):
        from app.substrate.migration import evaluate_live_gates
        gates = evaluate_live_gates(**self._all_args(
            cost_estimate_monthly_usd=300.0, budget_cap_usd=300.0,
        ))
        bc = next(g for g in gates if g.name == "budget_cap")
        assert bc.passed

    def test_stale_bundle_fails_live_gate(self):
        from app.substrate.migration import evaluate_live_gates
        gates = evaluate_live_gates(**self._all_args(bundle_age_days=2.0))
        bf = next(g for g in gates if g.name == "bundle_freshness")
        assert not bf.passed

    def test_missing_bundle_fails(self):
        from app.substrate.migration import evaluate_live_gates
        gates = evaluate_live_gates(**self._all_args(bundle_age_days=None))
        bf = next(g for g in gates if g.name == "bundle_freshness")
        assert not bf.passed

    def test_degraded_cloud_doctor_fails(self):
        from app.substrate.migration import evaluate_live_gates
        gates = evaluate_live_gates(**self._all_args(cloud_doctor_overall="DEGRADED"))
        cd = next(g for g in gates if g.name == "cloud_doctor")
        assert not cd.passed


# ── Live orchestrator: refusal paths ────────────────────────────────


class TestLiveRefusalPaths:
    def test_no_typed_phrase_raises_gate_failure(self, fresh_bundle, mock_cloud_doctor_ok):
        from app.substrate.migration import run_migration_live, GateFailure
        with pytest.raises(GateFailure, match="typed_phrase"):
            run_migration_live(
                target="gcp", project_id="p",
                confirm_phrase="",  # wrong
                budget_cap_usd=300.0,
            )

    def test_wrong_typed_phrase_raises(self, fresh_bundle, mock_cloud_doctor_ok):
        from app.substrate.migration import run_migration_live, GateFailure
        with pytest.raises(GateFailure, match="typed_phrase"):
            run_migration_live(
                target="gcp", project_id="p",
                confirm_phrase="please migrate",
                budget_cap_usd=300.0,
            )

    def test_no_project_raises(self, fresh_bundle, mock_cloud_doctor_ok):
        from app.substrate.migration import run_migration_live, GateFailure
        with pytest.raises(GateFailure, match="project_id"):
            run_migration_live(
                target="gcp", project_id=None,
                confirm_phrase="MIGRATE TO GCP",
                budget_cap_usd=300.0,
            )

    def test_low_budget_cap_raises(self, fresh_bundle, mock_cloud_doctor_ok):
        from app.substrate.migration import run_migration_live, GateFailure
        with pytest.raises(GateFailure, match="budget_cap"):
            run_migration_live(
                target="gcp", project_id="p",
                confirm_phrase="MIGRATE TO GCP",
                budget_cap_usd=50.0,   # below cheapest tier estimate
            )

    def test_no_bundle_raises(self, isolated_workspace, mock_cloud_doctor_ok):
        # No fresh_bundle fixture used → no DR tarball exists
        from app.substrate.migration import run_migration_live, GateFailure
        with pytest.raises(GateFailure, match="bundle_freshness"):
            run_migration_live(
                target="gcp", project_id="p",
                confirm_phrase="MIGRATE TO GCP",
                budget_cap_usd=300.0,
            )

    def test_stale_bundle_raises(self, isolated_workspace, mock_cloud_doctor_ok):
        # Create a stale bundle (2 days old)
        backup_dir = isolated_workspace / "backups" / "dr"
        backup_dir.mkdir(parents=True)
        tar = backup_dir / "dr_old.tar.gz"
        tar.write_bytes(b"x")
        old = time.time() - (2 * 86400)
        os.utime(tar, (old, old))

        from app.substrate.migration import run_migration_live, GateFailure
        with pytest.raises(GateFailure, match="bundle_freshness"):
            run_migration_live(
                target="gcp", project_id="p",
                confirm_phrase="MIGRATE TO GCP",
                budget_cap_usd=300.0,
            )

    def test_degraded_cloud_doctor_raises(self, fresh_bundle, mock_cloud_doctor_degraded):
        from app.substrate.migration import run_migration_live, GateFailure
        with pytest.raises(GateFailure, match="cloud_doctor"):
            run_migration_live(
                target="gcp", project_id="p",
                confirm_phrase="MIGRATE TO GCP",
                budget_cap_usd=300.0,
            )


# ── Live orchestrator: happy path with mocked shell ─────────────────


class TestLiveHappyPath:
    """Even with all gates green, _shell stays in dry-mode unless the
    operator explicitly sets BOTARMY_MIGRATE_LIVE_EXECUTE=1 or passes
    execute=True. Tests verify the orchestrator's logic without firing
    real terraform.
    """

    def test_live_run_with_all_gates_green(self, fresh_bundle, mock_cloud_doctor_ok, monkeypatch):
        # Make sure env-var guard is OFF
        monkeypatch.delenv("BOTARMY_MIGRATE_LIVE_EXECUTE", raising=False)

        from app.substrate.migration import run_migration_live
        run = run_migration_live(
            target="gcp", project_id="p",
            confirm_phrase="MIGRATE TO GCP",
            budget_cap_usd=300.0,
        )
        # Run completed (gates passed; steps ran in dry-shell mode)
        assert run.dry_run is False
        assert run.target == "gcp"
        # Steps recorded — verify the production sequence: provision,
        # transfer, restore, verify.
        names = [s.name for s in run.steps]
        # In dry-shell mode, all subprocess returns rc=0, so all steps
        # report ok EXCEPT verify which expects JSON output (gets "<dry: ...>")
        assert "provision" in names
        assert "transfer" in names

    def test_live_halts_on_first_failure(self, fresh_bundle, mock_cloud_doctor_ok, monkeypatch):
        """If provision step fails, transfer/restore/verify must NOT run."""
        from app.substrate import migration as m

        original_shell = m._shell

        def _fail_provision_shell(argv, *, timeout, execute=False, cwd=None):
            # Any bash invocation of install.sh fails
            if argv and argv[0] == "bash" and "install" in " ".join(argv):
                return (1, "", "simulated provision failure")
            return original_shell(argv, timeout=timeout, execute=execute, cwd=cwd)

        monkeypatch.setattr(m, "_shell", _fail_provision_shell)

        from app.substrate.migration import run_migration_live
        run = run_migration_live(
            target="gcp", project_id="p",
            confirm_phrase="MIGRATE TO GCP",
            budget_cap_usd=300.0,
        )
        # Provision should fail; later steps should NOT have run
        names = [s.name for s in run.steps]
        assert "provision" in names
        provision = next(s for s in run.steps if s.name == "provision")
        assert provision.status == "fail"
        # Halted — no transfer/restore/verify
        assert "transfer" not in names
        assert "restore" not in names
        assert "verify" not in names
        # Run is not ready
        assert run.ready_for_live is False

    def test_report_persisted_with_dry_run_false(self, fresh_bundle, mock_cloud_doctor_ok, isolated_workspace):
        from app.substrate.migration import run_migration_live
        run = run_migration_live(
            target="gcp", project_id="p",
            confirm_phrase="MIGRATE TO GCP",
            budget_cap_usd=300.0,
        )
        report = isolated_workspace / "migrations" / run.run_id / "report.json"
        assert report.exists()
        data = json.loads(report.read_text())
        assert data["dry_run"] is False
        assert data["run_id"] == run.run_id


# ── Shell execute-gate (the double-belt safety) ─────────────────────


class TestShellExecuteGate:
    """The _shell function MUST refuse to actually execute unless
    either execute=True passed OR BOTARMY_MIGRATE_LIVE_EXECUTE=1 set.
    """

    def test_neither_flag_returns_dry_placeholder(self, monkeypatch):
        from app.substrate.migration import _shell
        monkeypatch.delenv("BOTARMY_MIGRATE_LIVE_EXECUTE", raising=False)
        rc, out, err = _shell(["echo", "hello"], timeout=5.0)
        assert rc == 0
        assert "<dry:" in out
        assert "hello" in out
        assert err == ""

    def test_execute_true_actually_runs(self, monkeypatch):
        from app.substrate.migration import _shell
        monkeypatch.delenv("BOTARMY_MIGRATE_LIVE_EXECUTE", raising=False)
        rc, out, err = _shell(["echo", "real"], timeout=5.0, execute=True)
        assert rc == 0
        assert "real" in out
        assert "<dry:" not in out

    def test_env_var_actually_runs(self, monkeypatch):
        from app.substrate.migration import _shell
        monkeypatch.setenv("BOTARMY_MIGRATE_LIVE_EXECUTE", "1")
        rc, out, err = _shell(["echo", "envgo"], timeout=5.0)
        assert rc == 0
        assert "envgo" in out
        assert "<dry:" not in out

    def test_nonexistent_command_returns_127(self, monkeypatch):
        from app.substrate.migration import _shell
        monkeypatch.setenv("BOTARMY_MIGRATE_LIVE_EXECUTE", "1")
        rc, out, err = _shell(["this-command-definitely-does-not-exist"], timeout=5.0)
        assert rc == 127
        assert "command not found" in err

    def test_never_raises_on_garbage_input(self, monkeypatch):
        from app.substrate.migration import _shell
        monkeypatch.setenv("BOTARMY_MIGRATE_LIVE_EXECUTE", "1")
        # Empty argv would normally raise; _shell catches everything
        rc, out, err = _shell([], timeout=5.0)
        assert rc != 0  # FileNotFoundError or similar


# ── Identity-ledger emissions for live run ──────────────────────────


class TestTfvarsSeeding:
    """Gap-1 fix (2026-05-17): _step_provision_live writes a per-run
    terraform.tfvars and passes it via --config. Without this, gcp.sh
    with --non-interactive exits 1 because it can't find a tfvars
    file.
    """

    def test_per_run_tfvars_written(self, fresh_bundle, mock_cloud_doctor_ok, isolated_workspace):
        from app.substrate.migration import run_migration_live
        run = run_migration_live(
            target="gcp", project_id="botarmy-495107",
            confirm_phrase="MIGRATE TO GCP",
            budget_cap_usd=300.0,
        )
        tfvars_path = isolated_workspace / "migrations" / run.run_id / "terraform.tfvars"
        assert tfvars_path.exists(), "per-run terraform.tfvars must be written"
        content = tfvars_path.read_text()
        assert 'project_id        = "botarmy-495107"' in content
        assert 'region            = "europe-north1"' in content
        assert 'tier              = "cheapest"' in content
        # Conservative defaults — no public ingress, no monitoring, no keys
        assert "enable_monitoring = false" in content
        assert 'domain            = ""' in content
        assert "extra_env         = {}" in content

    def test_install_argv_includes_config_flag(self, fresh_bundle, mock_cloud_doctor_ok, monkeypatch):
        """The provision step must pass --config <tfvars_path> to install.sh."""
        from app.substrate import migration as m

        captured: list[list[str]] = []
        original_shell = m._shell

        def _spy_shell(argv, *, timeout, execute=False, cwd=None):
            captured.append(list(argv))
            return original_shell(argv, timeout=timeout, execute=execute, cwd=cwd)

        monkeypatch.setattr(m, "_shell", _spy_shell)

        from app.substrate.migration import run_migration_live
        run_migration_live(
            target="gcp", project_id="botarmy-495107",
            confirm_phrase="MIGRATE TO GCP",
            budget_cap_usd=300.0,
        )
        # First shell call is the install.sh invocation
        install_call = next(
            (c for c in captured if c[0] == "bash" and "install" in " ".join(c)),
            None,
        )
        assert install_call is not None, captured
        assert "--config" in install_call
        config_idx = install_call.index("--config")
        config_path = install_call[config_idx + 1]
        assert config_path.endswith("terraform.tfvars")
        assert "migrations" in config_path  # per-run scope

    def test_no_clobber_of_operator_tfvars(self, fresh_bundle, mock_cloud_doctor_ok, isolated_workspace):
        """The auto-generated tfvars MUST go to workspace/migrations/<run_id>/,
        NOT to deploy/terraform/gcp/terraform.tfvars (which may belong to
        the operator).
        """
        from app.substrate.migration import run_migration_live
        run = run_migration_live(
            target="gcp", project_id="botarmy-495107",
            confirm_phrase="MIGRATE TO GCP",
            budget_cap_usd=300.0,
        )
        # The per-run tfvars MUST live under workspace/, not in deploy/
        per_run = isolated_workspace / "migrations" / run.run_id / "terraform.tfvars"
        assert per_run.exists()
        # And the operator's tfvars location should NOT have been touched
        from app.substrate.migration import _repo_root
        operator_tfvars = _repo_root() / "deploy" / "terraform" / "gcp" / "terraform.tfvars"
        # Either absent (we never created it) OR pre-existing and not modified
        # by us; we don't make a stricter assertion because the operator
        # may already have a file there.
        # The strict assertion: it does NOT equal our generated content.
        if operator_tfvars.exists():
            assert run.run_id not in operator_tfvars.read_text(), (
                "we should not have written our run-id into the operator's tfvars"
            )


class TestLiveLedgerEmissions:
    def test_live_run_emits_multiple_landmarks(self, fresh_bundle, mock_cloud_doctor_ok, isolated_workspace, monkeypatch):
        """live_started + per-step success + final landmark."""
        ledger_path = isolated_workspace / "identity" / "continuity_ledger.jsonl"

        from app.identity import continuity_ledger as cl
        monkeypatch.setenv("IDENTITY_LEDGER_ENABLED", "true")
        monkeypatch.setattr(cl, "_resolve_path", lambda: ledger_path)

        from app.substrate.migration import run_migration_live
        run_migration_live(
            target="gcp", project_id="p",
            confirm_phrase="MIGRATE TO GCP",
            budget_cap_usd=300.0,
        )

        assert ledger_path.exists()
        events = [json.loads(line) for line in ledger_path.read_text().splitlines()]
        cm = [e for e in events if e["kind"] == "cloud_migration"]
        phases = [e["detail"]["phase"] for e in cm]
        # At minimum: live_started + a final landmark
        assert "live_started" in phases
        # All emitted events must agree on the run_id
        run_ids = {e["detail"]["run_id"] for e in cm}
        assert len(run_ids) == 1
