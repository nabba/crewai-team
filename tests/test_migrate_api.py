"""Tests for app.control_plane.migrate_api — WP D Phase 5a.

Uses FastAPI's TestClient to hit each endpoint. The endpoints
delegate to cloud_prep / cloud_doctor / cloud_cost / migration_runner,
all of which have their own unit tests; this file pins the REST contract
(status codes, request validation, auth wiring, error mapping).
"""
import json
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def isolated_workspace(tmp_path, monkeypatch):
    from app import paths as _paths
    monkeypatch.setattr(_paths, "WORKSPACE_ROOT", tmp_path)
    return tmp_path


@pytest.fixture
def reset_runner_singleton():
    from app.substrate import migration_runner as mr
    mr._RUNNER._active_run_id = None
    mr._RUNNER._active_thread = None
    mr._RUNNER._cancel_flag.clear()
    yield
    for _ in range(50):
        if mr._RUNNER.active_run_id() is None:
            break
        time.sleep(0.05)


@pytest.fixture
def client(isolated_workspace):
    """Spin up a minimal FastAPI app with just the migrate router."""
    from app.control_plane.migrate_api import router as migrate_router

    app = FastAPI()
    app.include_router(migrate_router)
    # GATEWAY_AUTH_REQUIRED unset → auth_dep passes through (dev mode)
    return TestClient(app)


# ── /accounts ──────────────────────────────────────────────────────


class TestAccountsEndpoint:
    def test_returns_account_list(self, client, monkeypatch):
        from app.substrate import cloud_prep as cp
        monkeypatch.setattr(
            cp, "list_authenticated_accounts",
            lambda: [
                {"account": "user@example.com", "type": "user", "active": "yes"},
                {"account": "sa@p.iam.gserviceaccount.com", "type": "service_account", "active": "no"},
            ],
        )
        r = client.get("/api/cp/migrate/accounts")
        assert r.status_code == 200
        assert len(r.json()["accounts"]) == 2

    def test_empty_when_no_accounts(self, client, monkeypatch):
        from app.substrate import cloud_prep as cp
        monkeypatch.setattr(cp, "list_authenticated_accounts", lambda: [])
        r = client.get("/api/cp/migrate/accounts")
        assert r.status_code == 200
        assert r.json()["accounts"] == []


# ── /preflight ─────────────────────────────────────────────────────


class TestPreflightEndpoint:
    def test_returns_readiness_dict(self, client, monkeypatch):
        from app.substrate import cloud_doctor as cd

        class _OK:
            target = "gcp"
            timestamp = "2026-05-17T00:00:00+00:00"
            overall = "OK"
            probes = []

            def to_dict(self):
                return {
                    "target": "gcp",
                    "timestamp": "2026-05-17T00:00:00+00:00",
                    "overall": "OK",
                    "probes": [],
                }

        monkeypatch.setattr(cd, "check_readiness", lambda target="gcp": _OK())
        r = client.get("/api/cp/migrate/preflight")
        assert r.status_code == 200
        body = r.json()
        assert body["overall"] == "OK"

    def test_target_query_param_passed_through(self, client, monkeypatch):
        from app.substrate import cloud_doctor as cd
        seen: dict = {}

        class _R:
            def to_dict(self): return {"overall": "OK", "target": seen["target"]}

        def _spy(target="gcp"):
            seen["target"] = target
            return _R()

        monkeypatch.setattr(cd, "check_readiness", _spy)
        r = client.get("/api/cp/migrate/preflight?target=aws")
        assert r.status_code == 200
        assert seen["target"] == "aws"


# ── /cost ──────────────────────────────────────────────────────────


class TestCostEndpoint:
    def test_returns_breakdown(self, client):
        r = client.post("/api/cp/migrate/cost", json={
            "target": "gcp", "tier": "cheapest",
            "enable_monitoring": True, "has_domain": False,
        })
        assert r.status_code == 200
        body = r.json()
        assert "total_monthly_usd" in body
        assert "line_items" in body
        # Cheapest GCP @ europe-north1 sanity range (same as cost tests)
        assert 100 <= body["total_monthly_usd"] <= 250

    def test_bad_target_returns_422(self, client):
        r = client.post("/api/cp/migrate/cost", json={"target": "azure"})
        assert r.status_code == 422


# ── /start ─────────────────────────────────────────────────────────


class TestStartEndpoint:
    @pytest.fixture(autouse=True)
    def _no_real_pipeline(self, monkeypatch):
        # Stub out the orchestrator entirely so /start doesn't kick off
        # an actual thread that would touch real cloud APIs.
        from app.substrate import migration_runner as mr

        def _fake_start(**kw):
            rec = mr.AsyncRunRecord(
                run_id="test-run-1",
                status="queued",
                target=kw.get("target", "gcp"),
                tier=kw.get("tier", "cheapest"),
                region=kw.get("region") or "europe-north1",
                project_id=kw.get("project_id", ""),
                active_account=kw.get("active_account", ""),
            )
            return rec

        monkeypatch.setattr(mr, "start_async_migration", _fake_start)

    def test_happy_path_returns_202(self, client):
        r = client.post("/api/cp/migrate/start", json={
            "target": "gcp", "tier": "cheapest",
            "project_id": "my-proj",
            "active_account": "user@example.com",
            "confirm_phrase": "MIGRATE TO GCP",
            "budget_cap_usd": 300.0,
        })
        assert r.status_code == 202
        body = r.json()
        assert body["run_id"] == "test-run-1"
        assert body["status"] == "queued"
        # dry_shell_mode mirrors the env-var state
        assert "dry_shell_mode" in body

    def test_wrong_phrase_400(self, client):
        r = client.post("/api/cp/migrate/start", json={
            "target": "gcp", "tier": "cheapest",
            "project_id": "my-proj",
            "active_account": "user@example.com",
            "confirm_phrase": "please migrate",
            "budget_cap_usd": 300.0,
        })
        assert r.status_code == 400
        assert "MIGRATE TO GCP" in r.json()["detail"]

    def test_budget_over_500_returns_422(self, client):
        """API hard-ceiling: budget_cap_usd <= 500."""
        r = client.post("/api/cp/migrate/start", json={
            "target": "gcp", "tier": "cheapest",
            "project_id": "my-proj",
            "active_account": "user@example.com",
            "confirm_phrase": "MIGRATE TO GCP",
            "budget_cap_usd": 1500.0,   # over the API ceiling
        })
        assert r.status_code == 422

    def test_zero_budget_returns_422(self, client):
        r = client.post("/api/cp/migrate/start", json={
            "target": "gcp", "tier": "cheapest",
            "project_id": "my-proj",
            "active_account": "user@example.com",
            "confirm_phrase": "MIGRATE TO GCP",
            "budget_cap_usd": 0,
        })
        assert r.status_code == 422

    def test_missing_project_id_returns_422(self, client):
        r = client.post("/api/cp/migrate/start", json={
            "target": "gcp", "tier": "cheapest",
            "active_account": "user@example.com",
            "confirm_phrase": "MIGRATE TO GCP",
            "budget_cap_usd": 300.0,
        })
        assert r.status_code == 422

    def test_busy_runner_returns_409(self, client, monkeypatch):
        from app.substrate import migration_runner as mr

        def _busy(**kw):
            raise mr.RunnerBusyError("another migration (run-xyz) is already in flight")

        monkeypatch.setattr(mr, "start_async_migration", _busy)

        r = client.post("/api/cp/migrate/start", json={
            "target": "gcp", "tier": "cheapest",
            "project_id": "p",
            "active_account": "u@example.com",
            "confirm_phrase": "MIGRATE TO GCP",
            "budget_cap_usd": 300.0,
        })
        assert r.status_code == 409
        assert "already in flight" in r.json()["detail"]


# ── /runs + /runs/<id> ─────────────────────────────────────────────


class TestRunsEndpoints:
    def test_runs_returns_active_id_and_list(self, client, isolated_workspace):
        # Seed a couple of runs to disk
        from app.substrate.migration_runner import AsyncRunRecord, _persist
        _persist(AsyncRunRecord(run_id="a", started_at="2026-05-15T10:00:00Z", status="succeeded"))
        _persist(AsyncRunRecord(run_id="b", started_at="2026-05-17T10:00:00Z", status="failed"))

        r = client.get("/api/cp/migrate/runs")
        assert r.status_code == 200
        body = r.json()
        assert "active_run_id" in body
        assert "runs" in body
        # Newest first
        assert body["runs"][0]["run_id"] == "b"

    def test_get_single_run_404_when_missing(self, client, isolated_workspace):
        r = client.get("/api/cp/migrate/runs/ghost")
        assert r.status_code == 404

    def test_get_single_run_returns_record(self, client, isolated_workspace):
        from app.substrate.migration_runner import AsyncRunRecord, _persist
        _persist(AsyncRunRecord(
            run_id="r1", status="running", current_step="provision",
            progress_pct=25, started_at="2026-05-17T10:00:00Z",
        ))
        r = client.get("/api/cp/migrate/runs/r1")
        assert r.status_code == 200
        body = r.json()
        assert body["run_id"] == "r1"
        assert body["status"] == "running"
        assert body["progress_pct"] == 25


# ── /runs/<id>/cancel ──────────────────────────────────────────────


class TestCancelEndpoint:
    def test_idle_runner_returns_409(self, client, reset_runner_singleton):
        r = client.post("/api/cp/migrate/runs/anything/cancel")
        assert r.status_code == 409

    def test_wrong_run_id_returns_409(self, client, reset_runner_singleton, monkeypatch):
        from app.substrate import migration_runner as mr
        monkeypatch.setattr(mr, "active_run_id", lambda: "actual-id")
        r = client.post("/api/cp/migrate/runs/different-id/cancel")
        assert r.status_code == 409
        assert "not the active run" in r.json()["detail"]

    def test_matching_run_id_cancels(self, client, reset_runner_singleton, monkeypatch):
        from app.substrate import migration_runner as mr
        calls = {}
        monkeypatch.setattr(mr, "active_run_id", lambda: "run-1")
        monkeypatch.setattr(mr, "cancel_active", lambda: calls.update({"cancelled": True}) or True)
        r = client.post("/api/cp/migrate/runs/run-1/cancel")
        assert r.status_code == 200
        assert r.json()["cancel_requested"] is True
        assert calls.get("cancelled") is True
