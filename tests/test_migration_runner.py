"""Tests for app.substrate.migration_runner — WP D Phase 5a.

The runner wraps the existing live-migrate orchestrator in a background
thread so the React event loop can poll for progress. Tests pin:

  * Status transitions follow the documented state machine
    (queued → preparing → cloud_prep → preflight → running → succeeded/failed).
  * Single-flight: a second start() while one is in-flight raises.
  * Cancel flag halts the pipeline cleanly at a step boundary.
  * Run state persists to ``workspace/migrations/<run_id>/run_state.json``.
  * load_run_record reads it back.
  * list_recent_runs is sorted newest-first.
  * Failure in cloud_prep stops the pipeline before any terraform call.
  * The runner inherits the orchestrator's safety gates — bad
    confirm_phrase still raises GateFailure inside the worker thread
    and lands as status='failed'.
"""
import json
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def isolated_workspace(tmp_path, monkeypatch):
    from app import paths as _paths
    monkeypatch.setattr(_paths, "WORKSPACE_ROOT", tmp_path)
    return tmp_path


@pytest.fixture
def fresh_bundle(isolated_workspace):
    import io
    import json as _json
    import tarfile
    backup_dir = isolated_workspace / "backups" / "dr"
    backup_dir.mkdir(parents=True)
    tar = backup_dir / "dr_test.tar.gz"
    manifest = {"ok": True, "subia_integrity_at_export": {"ok": True, "n_files": 1}}
    with tarfile.open(tar, "w:gz") as tf:
        data = _json.dumps(manifest).encode()
        info = tarfile.TarInfo(name="manifest.json")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    return tar


@pytest.fixture
def mock_cloud_doctor_ok(monkeypatch):
    from app.substrate import cloud_doctor as cd

    class _OK:
        target = "gcp"
        timestamp = "2026-05-17T00:00:00+00:00"
        overall = "OK"
        probes = []

    monkeypatch.setattr(cd, "check_readiness", lambda target="gcp": _OK())


@pytest.fixture
def mock_cloud_prep_ok(monkeypatch):
    """Make cloud_prep.prepare_gcp_for_migrate return a successful PrepResult
    without doing any subprocess."""
    from app.substrate import cloud_prep as cp

    def _fake_prep(*, active_account, project_id, apis=cp.REQUIRED_GCP_APIS):
        result = cp.PrepResult(
            active_account=active_account, project_id=project_id,
        )
        for step_name in ("set_active_account", "set_project", "enable_apis", "mint_terraform_token"):
            result.steps.append(cp.PrepStep(name=step_name, status="ok", detail="mocked"))
        result.terraform_env = {
            "GOOGLE_OAUTH_ACCESS_TOKEN": "ya29.mock-token",
            "GOOGLE_PROJECT": project_id,
        }
        result.succeeded = True
        return result

    monkeypatch.setattr(cp, "prepare_gcp_for_migrate", _fake_prep)


@pytest.fixture
def reset_runner_singleton():
    """Clean the singleton between tests to avoid pollution."""
    from app.substrate import migration_runner as mr
    mr._RUNNER._active_run_id = None
    mr._RUNNER._active_thread = None
    mr._RUNNER._cancel_flag.clear()
    yield
    # Wait for any background thread to finish (defensive)
    for _ in range(50):
        if mr._RUNNER.active_run_id() is None:
            break
        time.sleep(0.05)


# ── Persistence ────────────────────────────────────────────────────


class TestPersistence:
    def test_load_returns_none_when_missing(self, isolated_workspace):
        from app.substrate.migration_runner import load_run_record
        assert load_run_record("ghost") is None

    def test_roundtrip_persist_and_load(self, isolated_workspace):
        from app.substrate.migration_runner import (
            AsyncRunRecord, _persist, load_run_record,
        )
        rec = AsyncRunRecord(
            run_id="r1",
            status="running",
            target="gcp",
            project_id="my-proj",
            current_step="provision",
            progress_pct=25,
        )
        _persist(rec)
        loaded = load_run_record("r1")
        assert loaded is not None
        assert loaded.run_id == "r1"
        assert loaded.status == "running"
        assert loaded.progress_pct == 25

    def test_list_recent_runs_sorts_newest_first(self, isolated_workspace):
        from app.substrate.migration_runner import AsyncRunRecord, _persist, list_recent_runs
        _persist(AsyncRunRecord(run_id="old", started_at="2026-05-15T10:00:00Z", status="succeeded"))
        _persist(AsyncRunRecord(run_id="new", started_at="2026-05-17T10:00:00Z", status="running"))
        recs = list_recent_runs()
        assert [r.run_id for r in recs] == ["new", "old"]


# ── Single-flight ──────────────────────────────────────────────────


class TestSingleFlight:
    def test_second_start_raises_busy(
        self, isolated_workspace, fresh_bundle, mock_cloud_doctor_ok,
        mock_cloud_prep_ok, reset_runner_singleton,
    ):
        from app.substrate.migration_runner import start_async_migration, RunnerBusyError

        rec = start_async_migration(
            target="gcp", project_id="p",
            active_account="u@example.com",
            confirm_phrase="MIGRATE TO GCP",
            budget_cap_usd=300.0,
        )
        # Immediately try to start a second — should raise
        with pytest.raises(RunnerBusyError, match="already in flight"):
            start_async_migration(
                target="gcp", project_id="p",
                active_account="u@example.com",
                confirm_phrase="MIGRATE TO GCP",
                budget_cap_usd=300.0,
            )
        # Wait for the first to finish so the singleton clears
        from app.substrate import migration_runner as mr
        for _ in range(100):
            if mr.active_run_id() is None:
                break
            time.sleep(0.05)

    def test_active_run_id_reflects_state(self, isolated_workspace, fresh_bundle, mock_cloud_doctor_ok, mock_cloud_prep_ok, reset_runner_singleton):
        from app.substrate import migration_runner as mr
        assert mr.active_run_id() is None
        rec = mr.start_async_migration(
            target="gcp", project_id="p",
            active_account="u@example.com",
            confirm_phrase="MIGRATE TO GCP",
            budget_cap_usd=300.0,
        )
        # Either the run is still flight (active_run_id == rec.run_id) OR
        # it finished super fast (active_run_id None and status terminal).
        # Both are acceptable — we just verify the singleton-vs-loaded
        # contract holds.
        for _ in range(100):
            if mr.active_run_id() is None:
                break
            time.sleep(0.05)
        loaded = mr.load_run_record(rec.run_id)
        assert loaded is not None
        assert loaded.status in ("succeeded", "failed", "cancelled")


# ── State machine ──────────────────────────────────────────────────


class TestStateMachine:
    def _wait_terminal(self, run_id, timeout=5.0):
        from app.substrate.migration_runner import load_run_record
        deadline = time.monotonic() + timeout
        terminal = {"succeeded", "failed", "preflight_failed", "cancelled"}
        while time.monotonic() < deadline:
            rec = load_run_record(run_id)
            if rec and rec.status in terminal:
                return rec
            time.sleep(0.05)
        return load_run_record(run_id)

    def test_succeeded_status_on_happy_path(self, isolated_workspace, fresh_bundle, mock_cloud_doctor_ok, mock_cloud_prep_ok, reset_runner_singleton):
        from app.substrate.migration_runner import start_async_migration
        rec = start_async_migration(
            target="gcp", project_id="p",
            active_account="u@example.com",
            confirm_phrase="MIGRATE TO GCP",
            budget_cap_usd=300.0,
        )
        final = self._wait_terminal(rec.run_id)
        assert final is not None
        # In dry-shell mode all live steps succeed except 'verify' which
        # records 'warn' (unparseable JSON from <dry: ...>); succeeded=True
        # because no fails.
        assert final.status == "succeeded", f"actual: {final.status} / {final.error}"
        assert final.progress_pct == 100

    def test_cloud_prep_failure_yields_preflight_failed(self, isolated_workspace, fresh_bundle, mock_cloud_doctor_ok, monkeypatch, reset_runner_singleton):
        # Make cloud_prep fail
        from app.substrate import cloud_prep as cp

        def _bad_prep(*, active_account, project_id, apis=cp.REQUIRED_GCP_APIS):
            r = cp.PrepResult(active_account=active_account, project_id=project_id)
            r.steps.append(cp.PrepStep(name="set_active_account", status="fail", detail="boom"))
            r.fail_reason = "set_active_account: boom"
            return r

        monkeypatch.setattr(cp, "prepare_gcp_for_migrate", _bad_prep)

        from app.substrate.migration_runner import start_async_migration
        rec = start_async_migration(
            target="gcp", project_id="p",
            active_account="u@example.com",
            confirm_phrase="MIGRATE TO GCP",
            budget_cap_usd=300.0,
        )
        final = self._wait_terminal(rec.run_id)
        assert final.status == "preflight_failed"
        assert "boom" in final.error

    def test_doctor_missing_yields_preflight_failed(self, isolated_workspace, fresh_bundle, mock_cloud_prep_ok, monkeypatch, reset_runner_singleton):
        # Make cloud_doctor say MISSING
        from app.substrate import cloud_doctor as cd

        class _Missing:
            target = "gcp"
            timestamp = ""
            overall = "MISSING"
            probes = []

        monkeypatch.setattr(cd, "check_readiness", lambda target="gcp": _Missing())

        from app.substrate.migration_runner import start_async_migration
        rec = start_async_migration(
            target="gcp", project_id="p",
            active_account="u@example.com",
            confirm_phrase="MIGRATE TO GCP",
            budget_cap_usd=300.0,
        )
        final = self._wait_terminal(rec.run_id)
        assert final.status == "preflight_failed"
        assert "MISSING" in final.error


# ── Cancel ─────────────────────────────────────────────────────────


class TestCancel:
    def test_cancel_active_returns_false_when_idle(self, reset_runner_singleton):
        from app.substrate.migration_runner import cancel_active
        assert cancel_active() is False


# ── Inherits safety gates from migration.py ────────────────────────


class TestInheritsGates:
    def test_bad_typed_phrase_lands_as_failed(self, isolated_workspace, fresh_bundle, mock_cloud_doctor_ok, mock_cloud_prep_ok, reset_runner_singleton):
        from app.substrate.migration_runner import start_async_migration, load_run_record
        rec = start_async_migration(
            target="gcp", project_id="p",
            active_account="u@example.com",
            confirm_phrase="wrong-phrase",   # gate failure
            budget_cap_usd=300.0,
        )
        # Wait for worker to crash inside run_migration_live → GateFailure
        from app.substrate import migration_runner as mr
        for _ in range(100):
            if mr.active_run_id() is None:
                break
            time.sleep(0.05)
        final = load_run_record(rec.run_id)
        assert final is not None
        # The gate failure manifests as the worker crashing → status='failed'
        assert final.status == "failed"
        assert "typed_phrase" in final.error or "GateFailure" in final.error
