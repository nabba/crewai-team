"""Tests for /api/cp/coding-sessions — Phase 5.4-f.

The endpoints are read-only — there are no POSTs. We exercise:

  * empty list
  * status filter (valid + invalid)
  * agent filter
  * detail fetch (200 + 404)
  * is_active / is_terminal derived fields surface in the response

Auth is set off via GATEWAY_AUTH_REQUIRED=0 — same dance as
test_change_requests.py uses for its API surface.
"""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A FastAPI TestClient with the coding-sessions store redirected
    to a tmp_path. Auth is off — required for the test client to call
    the protected /cp/ routes."""
    monkeypatch.setenv("GATEWAY_AUTH_REQUIRED", "0")
    monkeypatch.setenv("CREWAI_TELEMETRY_OPT_OUT", "true")

    from app.coding_session import store as cs_store

    monkeypatch.setattr(cs_store, "_STORE_DIR", tmp_path)
    monkeypatch.setattr(
        cs_store, "_AUDIT_LOG", tmp_path / "audit.jsonl",
    )
    cs_store.reset_for_tests()

    from app.main import app
    from fastapi.testclient import TestClient

    return TestClient(app)


def _make_session(
    id: str = "t1",
    *,
    status_str: str = "active",
    agent_id: str = "coder",
):
    """Helper to drop a session directly into the store. Bypasses the
    manager so we don't need a real backend."""
    from app.coding_session import CodingSession, Status, store

    cs = CodingSession(
        id=id,
        agent_id=agent_id,
        purpose=f"test session {id}",
        created_at=f"2026-05-0{int(id[-1]) if id[-1].isdigit() else 1}T00:00:00+00:00",
        base="main",
        base_sha="a" * 40,
        worktree_path=f"/tmp/agent-sessions/{id}",
        expires_at="2026-05-04T01:00:00+00:00",
        last_activity_at=f"2026-05-0{int(id[-1]) if id[-1].isdigit() else 1}T00:00:00+00:00",
        status=Status(status_str),
    )
    store.save(cs)
    return cs


# ── List endpoint ───────────────────────────────────────────────────


class TestList:

    def test_empty(self, client) -> None:
        r = client.get("/api/cp/coding-sessions")
        assert r.status_code == 200
        body = r.json()
        assert body == {"count": 0, "sessions": []}

    def test_returns_session(self, client) -> None:
        _make_session("t1")
        r = client.get("/api/cp/coding-sessions")
        body = r.json()
        assert body["count"] == 1
        assert body["sessions"][0]["id"] == "t1"
        # Derived fields surface in the response
        assert body["sessions"][0]["is_active"] is True
        assert body["sessions"][0]["is_terminal"] is False

    def test_status_filter(self, client) -> None:
        _make_session("t1", status_str="active")
        _make_session("t2", status_str="submitted")
        _make_session("t3", status_str="discarded")

        r = client.get("/api/cp/coding-sessions?status=active")
        body = r.json()
        assert body["count"] == 1
        assert body["sessions"][0]["id"] == "t1"

        r = client.get("/api/cp/coding-sessions?status=submitted")
        body = r.json()
        assert body["count"] == 1
        assert body["sessions"][0]["id"] == "t2"

    def test_invalid_status_returns_400(self, client) -> None:
        r = client.get("/api/cp/coding-sessions?status=nonsense")
        assert r.status_code == 400
        assert "invalid status" in r.json()["detail"]

    def test_agent_id_filter(self, client) -> None:
        _make_session("t1", agent_id="coder")
        _make_session("t2", agent_id="researcher")
        _make_session("t3", agent_id="coder")

        r = client.get("/api/cp/coding-sessions?agent_id=coder")
        body = r.json()
        assert body["count"] == 2
        assert {s["id"] for s in body["sessions"]} == {"t1", "t3"}

    def test_combined_filters(self, client) -> None:
        _make_session("t1", agent_id="coder", status_str="active")
        _make_session("t2", agent_id="coder", status_str="submitted")
        _make_session("t3", agent_id="researcher", status_str="active")

        r = client.get(
            "/api/cp/coding-sessions?agent_id=coder&status=active",
        )
        body = r.json()
        assert body["count"] == 1
        assert body["sessions"][0]["id"] == "t1"

    def test_limit(self, client) -> None:
        for i in range(5):
            _make_session(f"t{i}")
        r = client.get("/api/cp/coding-sessions?limit=2")
        assert r.json()["count"] == 2

    def test_invalid_limit_returns_422(self, client) -> None:
        """FastAPI's Query validation rejects ge=1 violations as 422."""
        r = client.get("/api/cp/coding-sessions?limit=0")
        assert r.status_code == 422


# ── Detail endpoint ─────────────────────────────────────────────────


class TestDetail:

    def test_existing_session(self, client) -> None:
        cs = _make_session("t1")
        r = client.get(f"/api/cp/coding-sessions/{cs.id}")
        assert r.status_code == 200
        body = r.json()
        assert body["id"] == cs.id
        assert body["base"] == "main"
        assert body["agent_id"] == "coder"
        assert body["is_active"] is True
        assert body["is_terminal"] is False

    def test_unknown_returns_404(self, client) -> None:
        r = client.get("/api/cp/coding-sessions/does-not-exist")
        assert r.status_code == 404
        assert "not found" in r.json()["detail"]

    def test_terminal_session_predicates(self, client) -> None:
        _make_session("t1", status_str="discarded")
        r = client.get("/api/cp/coding-sessions/t1")
        body = r.json()
        assert body["is_active"] is False
        assert body["is_terminal"] is True


# ── No-write contract ───────────────────────────────────────────────


class TestNoWriteEndpoints:
    """Phase 5.4-f is read-only by design. Confirm the obvious
    write-style routes are NOT mounted."""

    @pytest.mark.parametrize("method,path", [
        ("POST", "/api/cp/coding-sessions/t1/discard"),
        ("POST", "/api/cp/coding-sessions/t1/expire"),
        ("DELETE", "/api/cp/coding-sessions/t1"),
        ("PUT", "/api/cp/coding-sessions/t1"),
    ])
    def test_write_method_returns_405_or_404(
        self, client, method: str, path: str,
    ) -> None:
        r = client.request(method, path)
        # FastAPI returns 405 if path matches but method doesn't,
        # 404 if path itself isn't mounted. Either is fine for our
        # contract (the route doesn't exist).
        assert r.status_code in (404, 405)
