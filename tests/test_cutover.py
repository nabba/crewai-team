"""Tests for app.substrate.cutover — WP D Phase 4.

Cutover is the operator-decided commit of a verified cloud migration.
Tests pin:
  * Every gate refuses what it claims to refuse (typed phrase, missing
    report, dry-run-only report, unsuccessful report, stale, mismatch).
  * Cutover halts on first step failure (cloud_health fail → no demote).
  * demote_local actually flips warm_spare state.
  * Identity-ledger landmarks (cutover_started + completed/failed).
  * Report persisted at workspace/migrations/<run_id>/cutover.json.
  * Subprocess execute-gate honored (test runs never fire real kubectl).
"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Shared fixtures ─────────────────────────────────────────────────


@pytest.fixture
def isolated_workspace(tmp_path, monkeypatch):
    from app import paths as _paths
    monkeypatch.setattr(_paths, "WORKSPACE_ROOT", tmp_path)
    return tmp_path


def _write_migration_report(
    workspace: Path,
    run_id: str = "test-run-1",
    *,
    dry_run: bool = False,
    ready_for_live: bool = True,
    target: str = "gcp",
    started_at: str | None = None,
    age_days: float = 1.0,
) -> Path:
    """Drop a synthetic migration report at workspace/migrations/<id>/report.json."""
    out_dir = workspace / "migrations" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    if started_at is None:
        started_at = (datetime.now(timezone.utc) - timedelta(days=age_days)).isoformat()
    report = {
        "run_id": run_id,
        "started_at": started_at,
        "completed_at": started_at,
        "target": target,
        "tier": "cheapest",
        "region": "europe-north1",
        "project_id": "my-project",
        "dry_run": dry_run,
        "ready_for_live": ready_for_live,
        "steps": [],
        "blockers": [],
        "warnings": [],
    }
    path = out_dir / "report.json"
    path.write_text(json.dumps(report, indent=2))
    return path


# ── Gate evaluation ─────────────────────────────────────────────────


class TestGateEvaluation:
    def test_all_gates_pass_with_valid_report(self, isolated_workspace):
        _write_migration_report(isolated_workspace, run_id="r1")
        from app.substrate.cutover import evaluate_cutover_gates
        gates = evaluate_cutover_gates(
            target="gcp",
            migration_run_id="r1",
            confirm_phrase="CUTOVER TO GCP",
        )
        failed = [g for g in gates if not g.passed]
        assert not failed, [g.detail for g in failed]

    def test_wrong_typed_phrase_fails(self, isolated_workspace):
        _write_migration_report(isolated_workspace, run_id="r1")
        from app.substrate.cutover import evaluate_cutover_gates
        gates = evaluate_cutover_gates(
            target="gcp", migration_run_id="r1", confirm_phrase="cutover",
        )
        tp = next(g for g in gates if g.name == "typed_phrase")
        assert not tp.passed

    def test_empty_run_id_fails(self, isolated_workspace):
        from app.substrate.cutover import evaluate_cutover_gates
        gates = evaluate_cutover_gates(
            target="gcp", migration_run_id="", confirm_phrase="CUTOVER TO GCP",
        )
        rid = next(g for g in gates if g.name == "migration_run_id")
        assert not rid.passed

    def test_missing_report_fails(self, isolated_workspace):
        from app.substrate.cutover import evaluate_cutover_gates
        gates = evaluate_cutover_gates(
            target="gcp", migration_run_id="nonexistent",
            confirm_phrase="CUTOVER TO GCP",
        )
        rid = next(g for g in gates if g.name == "migration_run_id")
        assert not rid.passed
        assert "no migration report" in rid.detail

    def test_dry_run_report_blocks_cutover(self, isolated_workspace):
        _write_migration_report(isolated_workspace, run_id="r1", dry_run=True)
        from app.substrate.cutover import evaluate_cutover_gates
        gates = evaluate_cutover_gates(
            target="gcp", migration_run_id="r1",
            confirm_phrase="CUTOVER TO GCP",
        )
        live = next(g for g in gates if g.name == "migration_was_live")
        assert not live.passed

    def test_unsuccessful_migration_blocks_cutover(self, isolated_workspace):
        _write_migration_report(isolated_workspace, run_id="r1", ready_for_live=False)
        from app.substrate.cutover import evaluate_cutover_gates
        gates = evaluate_cutover_gates(
            target="gcp", migration_run_id="r1",
            confirm_phrase="CUTOVER TO GCP",
        )
        ok = next(g for g in gates if g.name == "migration_succeeded")
        assert not ok.passed

    def test_stale_migration_blocks_cutover(self, isolated_workspace):
        _write_migration_report(isolated_workspace, run_id="r1", age_days=45.0)
        from app.substrate.cutover import evaluate_cutover_gates
        gates = evaluate_cutover_gates(
            target="gcp", migration_run_id="r1",
            confirm_phrase="CUTOVER TO GCP",
        )
        fresh = next(g for g in gates if g.name == "migration_freshness")
        assert not fresh.passed

    def test_target_mismatch_blocks(self, isolated_workspace):
        _write_migration_report(isolated_workspace, run_id="r1", target="aws")
        from app.substrate.cutover import evaluate_cutover_gates
        gates = evaluate_cutover_gates(
            target="gcp",   # requested gcp but report is aws
            migration_run_id="r1", confirm_phrase="CUTOVER TO GCP",
        )
        tm = next(g for g in gates if g.name == "target_match")
        assert not tm.passed


# ── Public orchestrator: refusal paths ──────────────────────────────


class TestCutoverRefusalPaths:
    def test_raises_gate_failure_on_wrong_phrase(self, isolated_workspace):
        _write_migration_report(isolated_workspace, run_id="r1")
        from app.substrate.cutover import run_cutover, CutoverGateFailure
        with pytest.raises(CutoverGateFailure, match="typed_phrase"):
            run_cutover(
                migration_run_id="r1",
                confirm_phrase="",
            )

    def test_raises_on_missing_report(self, isolated_workspace):
        from app.substrate.cutover import run_cutover, CutoverGateFailure
        with pytest.raises(CutoverGateFailure, match="migration_run_id"):
            run_cutover(
                migration_run_id="ghost",
                confirm_phrase="CUTOVER TO GCP",
            )

    def test_raises_on_dry_run_report(self, isolated_workspace):
        _write_migration_report(isolated_workspace, run_id="r1", dry_run=True)
        from app.substrate.cutover import run_cutover, CutoverGateFailure
        with pytest.raises(CutoverGateFailure, match="migration_was_live"):
            run_cutover(
                migration_run_id="r1",
                confirm_phrase="CUTOVER TO GCP",
            )


# ── Happy path ──────────────────────────────────────────────────────


class TestCutoverHappyPath:
    def test_all_gates_pass_runs_pipeline(self, isolated_workspace, monkeypatch):
        _write_migration_report(isolated_workspace, run_id="r1")
        # Force env-var gate OFF so subprocess stays dry
        monkeypatch.delenv("BOTARMY_MIGRATE_LIVE_EXECUTE", raising=False)

        from app.substrate.cutover import run_cutover
        run = run_cutover(
            migration_run_id="r1",
            confirm_phrase="CUTOVER TO GCP",
        )
        # All 3 steps recorded
        names = [s.name for s in run.steps]
        assert names == ["cloud_health", "demote_local", "claim_canonical_cloud"]

    def test_succeeded_field_set(self, isolated_workspace, monkeypatch):
        _write_migration_report(isolated_workspace, run_id="r1")
        monkeypatch.delenv("BOTARMY_MIGRATE_LIVE_EXECUTE", raising=False)
        from app.substrate.cutover import run_cutover
        run = run_cutover(
            migration_run_id="r1",
            confirm_phrase="CUTOVER TO GCP",
        )
        # In dry-shell mode, cloud_health probe returns warn (unparseable
        # output) and claim_canonical_cloud returns warn — but the demote
        # step DID run cleanly. No fails → succeeded=True.
        assert run.succeeded is True
        # warnings present from the dry-shell parse failures
        assert len(run.warnings) >= 1

    def test_demote_local_actually_changes_warm_spare_state(self, isolated_workspace, monkeypatch):
        _write_migration_report(isolated_workspace, run_id="r1")
        from app.warm_spare.failover import current_state
        from app.substrate.cutover import run_cutover

        prior = current_state()
        run_cutover(
            migration_run_id="r1",
            confirm_phrase="CUTOVER TO GCP",
        )
        after = current_state()
        # State should now be DEMOTED (regardless of what it was before)
        assert after.get("state") == "demoted", after

    def test_project_id_resolved_from_report_when_omitted(self, isolated_workspace):
        _write_migration_report(isolated_workspace, run_id="r1")
        from app.substrate.cutover import run_cutover
        run = run_cutover(
            migration_run_id="r1",
            confirm_phrase="CUTOVER TO GCP",
            project_id=None,
        )
        # The synthetic report has project_id="my-project"
        assert run.project_id == "my-project"


# ── Halt-on-fail ────────────────────────────────────────────────────


class TestHaltOnFail:
    def test_cloud_health_failure_skips_demote_and_claim(self, isolated_workspace, monkeypatch):
        _write_migration_report(isolated_workspace, run_id="r1")
        from app.substrate import cutover as co

        # Replace _shell so the cloud_health kubectl probe fails hard
        original_shell = co._shell

        def _fail_shell(argv, *, timeout, execute=False):
            if argv and "kubectl" in argv[0]:
                return (1, "", "kubectl: connection refused")
            return original_shell(argv, timeout=timeout, execute=execute)

        monkeypatch.setattr(co, "_shell", _fail_shell)

        from app.warm_spare.failover import current_state
        prior_state = current_state().get("state")

        run = co.run_cutover(
            migration_run_id="r1",
            confirm_phrase="CUTOVER TO GCP",
        )
        # Halted after cloud_health
        names = [s.name for s in run.steps]
        assert "cloud_health" in names
        assert names == ["cloud_health"]
        # Demote did NOT run — local state unchanged
        after_state = current_state().get("state")
        assert after_state == prior_state
        # Run failed
        assert run.succeeded is False


# ── Report persistence ──────────────────────────────────────────────


class TestReportPersistence:
    def test_cutover_json_written_alongside_migration_report(self, isolated_workspace):
        _write_migration_report(isolated_workspace, run_id="r1")
        from app.substrate.cutover import run_cutover
        run_cutover(
            migration_run_id="r1",
            confirm_phrase="CUTOVER TO GCP",
        )
        cutover_json = isolated_workspace / "migrations" / "r1" / "cutover.json"
        assert cutover_json.exists()
        data = json.loads(cutover_json.read_text())
        assert data["migration_run_id"] == "r1"
        assert "steps" in data
        assert "succeeded" in data


# ── Identity-ledger emissions ───────────────────────────────────────


class TestLedgerEmissions:
    def test_emits_started_and_terminal_landmark(self, isolated_workspace, monkeypatch):
        _write_migration_report(isolated_workspace, run_id="r1")
        ledger_path = isolated_workspace / "identity" / "continuity_ledger.jsonl"

        from app.identity import continuity_ledger as cl
        monkeypatch.setenv("IDENTITY_LEDGER_ENABLED", "true")
        monkeypatch.setattr(cl, "_resolve_path", lambda: ledger_path)

        from app.substrate.cutover import run_cutover
        run_cutover(
            migration_run_id="r1",
            confirm_phrase="CUTOVER TO GCP",
        )

        assert ledger_path.exists()
        events = [json.loads(line) for line in ledger_path.read_text().splitlines()]
        cm = [e for e in events if e["kind"] == "cloud_migration"]
        phases = [e["detail"]["phase"] for e in cm]
        assert "cutover_started" in phases
        # In dry-shell mode the run succeeds (no fail steps) → completed
        assert "cutover_completed" in phases


# ── Format checklist ────────────────────────────────────────────────


class TestFormatCheckList:
    def test_success_output_includes_signal_handoff(self, isolated_workspace):
        _write_migration_report(isolated_workspace, run_id="r1")
        from app.substrate.cutover import run_cutover, format_cutover_run
        run = run_cutover(
            migration_run_id="r1",
            confirm_phrase="CUTOVER TO GCP",
        )
        out = format_cutover_run(run)
        assert "CUTOVER COMPLETE" in out
        # The post-cutover checklist names the manual Signal step
        assert "signal-cli" in out
        assert "STOP the local gateway" in out

    def test_failure_output_includes_recovery_hint(self, isolated_workspace, monkeypatch):
        _write_migration_report(isolated_workspace, run_id="r1")
        from app.substrate import cutover as co

        def _fail_shell(argv, *, timeout, execute=False):
            return (1, "", "boom")

        monkeypatch.setattr(co, "_shell", _fail_shell)

        run = co.run_cutover(
            migration_run_id="r1",
            confirm_phrase="CUTOVER TO GCP",
        )
        out = co.format_cutover_run(run)
        assert "CUTOVER FAILED" in out
        assert "cutover.json" in out


# ── Subprocess execute-gate ─────────────────────────────────────────


class TestShellExecuteGate:
    def test_no_subprocess_run_without_opt_in(self, isolated_workspace, monkeypatch):
        """Cutover must NOT shell out for real in the default test config."""
        _write_migration_report(isolated_workspace, run_id="r1")
        import subprocess
        calls: list[list[str]] = []
        real_run = subprocess.run

        def _spy(argv, *a, **kw):
            calls.append(list(argv) if isinstance(argv, (list, tuple)) else [str(argv)])
            return real_run(argv, *a, **kw)

        monkeypatch.setattr(subprocess, "run", _spy)
        monkeypatch.delenv("BOTARMY_MIGRATE_LIVE_EXECUTE", raising=False)

        from app.substrate.cutover import run_cutover
        run_cutover(
            migration_run_id="r1",
            confirm_phrase="CUTOVER TO GCP",
        )
        # No subprocess.run calls — _shell short-circuited to dry mode
        assert calls == [], f"unexpected subprocess invocations: {calls}"
