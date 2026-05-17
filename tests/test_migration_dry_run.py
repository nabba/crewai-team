"""Tests for app.substrate.migration — WP D Phase 2 (dry-run).

The orchestrator is the operator-visible safety gate before any live
migrate. Tests pin:
  * The pipeline never raises, regardless of probe failures.
  * Blockers correctly classify (fail → blocker, warn → warning).
  * Report file is persisted to workspace/migrations/<run_id>/.
  * Identity ledger receives ONE cloud_migration event per run.
  * No side effects: no subprocess, no cloud API, no terraform apply.
"""
import json
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Common fixtures ─────────────────────────────────────────────────


@pytest.fixture
def isolated_workspace(tmp_path, monkeypatch):
    """Point WORKSPACE_ROOT at a fresh tmp dir for the duration of one test."""
    from app import paths as _paths
    monkeypatch.setattr(_paths, "WORKSPACE_ROOT", tmp_path)
    return tmp_path


@pytest.fixture
def mock_cloud_doctor_ok(monkeypatch):
    """Make cloud_doctor.check_readiness return overall=OK."""
    from app.substrate import cloud_doctor as cd

    class _FakeReadiness:
        target = "gcp"
        timestamp = "2026-05-17T00:00:00+00:00"
        overall = "OK"

        class _P:
            def __init__(self, name, status="OK", required=True, detail=""):
                self.name = name
                self.status = status
                self.required = required
                self.detail = detail

        probes = [
            _P("terraform"),
            _P("kubectl"),
            _P("helm"),
            _P("docker"),
            _P("gcloud"),
            _P("gcloud auth"),
            _P("continuity bundle"),
        ]

    monkeypatch.setattr(cd, "check_readiness", lambda target="gcp": _FakeReadiness())


@pytest.fixture
def mock_cloud_doctor_missing(monkeypatch):
    """Make cloud_doctor.check_readiness return overall=MISSING."""
    from app.substrate import cloud_doctor as cd

    class _FakeReadiness:
        target = "gcp"
        timestamp = "2026-05-17T00:00:00+00:00"
        overall = "MISSING"

        class _P:
            def __init__(self, name, status, required=True, detail=""):
                self.name = name
                self.status = status
                self.required = required
                self.detail = detail

        probes = [
            _P("terraform", "MISSING", True, "terraform not on PATH"),
        ]

    monkeypatch.setattr(cd, "check_readiness", lambda target="gcp": _FakeReadiness())


def _make_fake_bundle(tmp_path: Path, age_days: float = 0.0) -> Path:
    """Create a fake DR tarball at workspace/backups/dr/.

    Embeds a minimal manifest.json so the bundle_metadata step can peek.
    """
    import tarfile
    import io

    backup_dir = tmp_path / "backups" / "dr"
    backup_dir.mkdir(parents=True, exist_ok=True)
    tarball = backup_dir / "dr_test.tar.gz"

    manifest = {
        "program": "test",
        "started_at": "2026-05-17T00:00:00+00:00",
        "completed_at": "2026-05-17T00:01:00+00:00",
        "duration_s": 60,
        "workspace_root": str(tmp_path),
        "chromadb": [{"kb": "memory", "collection": "test_col", "rows": 1, "bytes": 100}],
        "postgres": [{"table": "control_plane.audit_log", "rows": 10, "bytes": 1000}],
        "ledgers": [{"rel_path": "identity/continuity_ledger.jsonl", "bytes": 500, "sha256": "x"}],
        "excluded_secret_paths": [],
        "total_rows_chromadb": 1,
        "total_rows_postgres": 10,
        "total_bytes": 1500,
        "subia_integrity_at_export": {
            "ok": True, "has_drift": False, "n_files": 164,
            "n_mismatched": 0, "n_extra": 0, "n_missing": 0,
        },
        "ok": True,
        "errors": [],
    }
    with tarfile.open(tarball, "w:gz") as tf:
        data = json.dumps(manifest, indent=2).encode("utf-8")
        info = tarfile.TarInfo(name="manifest.json")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))

    if age_days > 0:
        old = time.time() - (age_days * 86400)
        os.utime(tarball, (old, old))
    return tarball


# ── Public shape ─────────────────────────────────────────────────────


class TestPublicShape:
    def test_run_returns_migration_run(self, isolated_workspace, mock_cloud_doctor_ok):
        _make_fake_bundle(isolated_workspace)
        from app.substrate.migration import run_migration_dry_run, MigrationRun
        run = run_migration_dry_run(target="gcp", tier="cheapest", project_id="p")
        assert isinstance(run, MigrationRun)
        assert run.dry_run is True
        assert run.run_id  # auto-generated
        assert run.target == "gcp"
        assert run.region == "europe-north1"   # auto-default
        assert len(run.steps) == 6   # pipeline shape pinned

    def test_step_names_are_stable(self, isolated_workspace, mock_cloud_doctor_ok):
        _make_fake_bundle(isolated_workspace)
        from app.substrate.migration import run_migration_dry_run
        run = run_migration_dry_run(target="gcp", project_id="p")
        names = [s.name for s in run.steps]
        assert names == [
            "preflight",
            "cost_estimate",
            "bundle_metadata",
            "transfer_plan",
            "restore_plan",
            "verify_plan",
        ]

    def test_explicit_run_id_honored(self, isolated_workspace, mock_cloud_doctor_ok):
        _make_fake_bundle(isolated_workspace)
        from app.substrate.migration import run_migration_dry_run
        run = run_migration_dry_run(target="gcp", project_id="p", run_id="custom-id-123")
        assert run.run_id == "custom-id-123"


# ── Roll-up correctness ──────────────────────────────────────────────


class TestRollup:
    def test_ready_for_live_when_clean(self, isolated_workspace, mock_cloud_doctor_ok):
        _make_fake_bundle(isolated_workspace)
        from app.substrate.migration import run_migration_dry_run
        run = run_migration_dry_run(target="gcp", project_id="p")
        assert run.ready_for_live is True
        assert run.blockers == []

    def test_not_ready_when_preflight_missing(self, isolated_workspace, mock_cloud_doctor_missing):
        _make_fake_bundle(isolated_workspace)
        from app.substrate.migration import run_migration_dry_run
        run = run_migration_dry_run(target="gcp", project_id="p")
        assert run.ready_for_live is False
        assert any("preflight" in b for b in run.blockers)

    def test_not_ready_without_bundle(self, isolated_workspace, mock_cloud_doctor_ok):
        # No bundle created
        from app.substrate.migration import run_migration_dry_run
        run = run_migration_dry_run(target="gcp", project_id="p")
        assert run.ready_for_live is False
        assert any("bundle_metadata" in b for b in run.blockers)

    def test_stale_bundle_is_warning_not_blocker(self, isolated_workspace, mock_cloud_doctor_ok):
        _make_fake_bundle(isolated_workspace, age_days=15.0)
        from app.substrate.migration import run_migration_dry_run
        run = run_migration_dry_run(target="gcp", project_id="p")
        # Stale = warn, not fail, so live migrate is still possible
        assert run.ready_for_live is True
        assert any("bundle_metadata" in w for w in run.warnings)

    def test_budget_cap_blocks_when_exceeded(self, isolated_workspace, mock_cloud_doctor_ok):
        _make_fake_bundle(isolated_workspace)
        from app.substrate.migration import run_migration_dry_run
        # prod is ~$734/mo on europe-north1; cap of $200 will trigger
        run = run_migration_dry_run(
            target="gcp", tier="prod", project_id="p", budget_cap_usd=200.0,
        )
        assert run.ready_for_live is False
        assert any("cost_estimate" in b for b in run.blockers)

    def test_budget_cap_passes_when_under(self, isolated_workspace, mock_cloud_doctor_ok):
        _make_fake_bundle(isolated_workspace)
        from app.substrate.migration import run_migration_dry_run
        # cheapest is ~$170/mo; cap of $500 should pass
        run = run_migration_dry_run(
            target="gcp", tier="cheapest", project_id="p", budget_cap_usd=500.0,
        )
        assert run.ready_for_live is True
        cost_step = next(s for s in run.steps if s.name == "cost_estimate")
        assert cost_step.status == "ok"


# ── Report persistence ──────────────────────────────────────────────


class TestReportPersistence:
    def test_report_written_to_workspace(self, isolated_workspace, mock_cloud_doctor_ok):
        _make_fake_bundle(isolated_workspace)
        from app.substrate.migration import run_migration_dry_run
        run = run_migration_dry_run(target="gcp", project_id="p")
        report = isolated_workspace / "migrations" / run.run_id / "report.json"
        assert report.exists()
        data = json.loads(report.read_text())
        assert data["run_id"] == run.run_id
        assert data["target"] == "gcp"
        assert data["dry_run"] is True
        assert "steps" in data
        assert "blockers" in data


# ── Identity-ledger emission ────────────────────────────────────────


class TestLedgerEmission:
    def test_one_cloud_migration_event_per_run(self, isolated_workspace, mock_cloud_doctor_ok, monkeypatch):
        _make_fake_bundle(isolated_workspace)
        ledger_path = isolated_workspace / "identity" / "continuity_ledger.jsonl"

        from app.identity import continuity_ledger as cl
        # Force IDENTITY_LEDGER_ENABLED on
        monkeypatch.setenv("IDENTITY_LEDGER_ENABLED", "true")

        # Drive the ledger to write into the isolated workspace
        def _resolve(): return ledger_path
        monkeypatch.setattr(cl, "_resolve_path", _resolve)

        from app.substrate.migration import run_migration_dry_run
        run_migration_dry_run(target="gcp", project_id="p")

        assert ledger_path.exists()
        lines = ledger_path.read_text().splitlines()
        # At least our event should be there
        events = [json.loads(line) for line in lines]
        cm = [e for e in events if e["kind"] == "cloud_migration"]
        assert len(cm) == 1
        assert cm[0]["actor"] == "botarmy_migrate"
        assert cm[0]["detail"]["phase"] == "dry_run"


# ── Failure isolation ───────────────────────────────────────────────


class TestFailureIsolation:
    def test_step_crash_doesnt_kill_run(self, isolated_workspace, mock_cloud_doctor_ok, monkeypatch):
        _make_fake_bundle(isolated_workspace)
        # Make the cost estimate raise mid-pipeline
        from app.substrate import migration as m

        def _boom(*a, **kw):
            raise RuntimeError("simulated cost estimator crash")

        monkeypatch.setattr(m, "_step_cost_estimate", _boom)

        from app.substrate.migration import run_migration_dry_run
        run = run_migration_dry_run(target="gcp", project_id="p")
        # Still has all 6 steps
        assert len(run.steps) == 6
        # The cost step recorded the crash as a failure
        cost = next(s for s in run.steps if s.name == "cost_estimate")
        assert cost.status == "fail"
        assert "RuntimeError" in cost.detail
        # Other steps still ran
        assert any(s.name == "bundle_metadata" and s.status == "ok" for s in run.steps)
        # Run is not ready
        assert run.ready_for_live is False


# ── Transfer plan ────────────────────────────────────────────────────


class TestTransferPlan:
    def test_gcp_includes_project_in_bucket(self, isolated_workspace, mock_cloud_doctor_ok):
        _make_fake_bundle(isolated_workspace)
        from app.substrate.migration import run_migration_dry_run
        run = run_migration_dry_run(target="gcp", project_id="botarmy-495107")
        tp = next(s for s in run.steps if s.name == "transfer_plan")
        assert tp.status == "ok"
        assert "botarmy-495107" in tp.output["bucket"]
        assert run.run_id in tp.output["destination"]

    def test_gcp_missing_project_fails(self, isolated_workspace, mock_cloud_doctor_ok):
        _make_fake_bundle(isolated_workspace)
        from app.substrate.migration import run_migration_dry_run
        run = run_migration_dry_run(target="gcp", project_id=None)
        tp = next(s for s in run.steps if s.name == "transfer_plan")
        assert tp.status == "fail"
        assert "--project" in tp.detail


# ── Format ──────────────────────────────────────────────────────────


class TestFormatRun:
    def test_format_includes_run_id_target_steps(self, isolated_workspace, mock_cloud_doctor_ok):
        _make_fake_bundle(isolated_workspace)
        from app.substrate.migration import run_migration_dry_run, format_run
        run = run_migration_dry_run(target="gcp", project_id="p")
        out = format_run(run)
        assert run.run_id[:12] in out
        assert "gcp" in out
        assert "europe-north1" in out
        # Every step should appear by name
        for s in run.steps:
            assert s.name in out


# ── No-side-effect guarantee ────────────────────────────────────────


class TestNoSideEffects:
    def test_does_not_call_subprocess_for_cloud_apis(self, isolated_workspace, mock_cloud_doctor_ok, monkeypatch):
        """Dry-run must not shell out to terraform / gcloud / kubectl /
        docker beyond what cloud_doctor does. We mocked cloud_doctor —
        any other subprocess invocation is a regression.
        """
        import subprocess
        calls: list[list[str]] = []
        real_run = subprocess.run

        def _spy(argv, *a, **kw):
            calls.append(list(argv) if isinstance(argv, (list, tuple)) else [str(argv)])
            return real_run(argv, *a, **kw)

        monkeypatch.setattr(subprocess, "run", _spy)

        _make_fake_bundle(isolated_workspace)
        from app.substrate.migration import run_migration_dry_run
        run_migration_dry_run(target="gcp", project_id="p")

        # cloud_doctor is mocked, so NO subprocess.run should have fired.
        assert calls == [], f"unexpected subprocess invocations: {calls}"
