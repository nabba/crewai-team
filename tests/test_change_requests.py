"""Tests for app.change_requests — Phase 5.3a backend.

Coverage:
  * **Validator** — TIER_IMMUTABLE rejection (with the
    is_tier_immutable flag), outside-roots, path traversal,
    blocked patterns, content size cap, normalization.
  * **Models** — to_dict/from_dict round-trip, status enum,
    is_terminal / is_rollbackable predicates.
  * **Store** — save/get/list_all, find_by_signal_ts, audit log
    chain integrity.
  * **Lifecycle** — create_request happy path; TIER_IMMUTABLE
    refusal at request time; approve/reject transitions; illegal
    transitions raise; idempotency.
  * **API endpoints** — list/get/approve/reject/rollback/retry-apply.
    TIER_IMMUTABLE → 403; not-found → 404; wrong-state → 409.
  * **Signal integration** — build_ask_body shape; reaction handler
    dispatch (mocked Signal client).

Tests use ``tmp_path`` + ``monkeypatch`` to redirect the store
directory so they don't pollute real workspace state.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


# ── Validator ────────────────────────────────────────────────────────


class TestValidator:

    def test_allowed_path_passes(self):
        from app.change_requests import validate
        result = validate(path="app/agents/pim_agent.py", new_content="x = 1")
        assert result.ok
        assert result.reason is None

    def test_tier_immutable_rejected_with_flag(self):
        """TIER_IMMUTABLE files MUST be rejected with is_tier_immutable=True
        so the caller can persist as TIER_IMMUTABLE_REFUSED status."""
        from app.change_requests import validate
        result = validate(path="app/auto_deployer.py", new_content="x = 1")
        assert not result.ok
        assert result.is_tier_immutable is True
        assert "TIER_IMMUTABLE" in result.reason

    def test_souls_loader_tier_immutable(self):
        from app.change_requests import validate
        result = validate(path="app/souls/loader.py", new_content="x = 1")
        assert not result.ok
        assert result.is_tier_immutable is True

    def test_capabilities_tier_immutable(self):
        """tool_registry/capabilities.py is in TIER_IMMUTABLE per
        the Phase 1a governance rule."""
        from app.change_requests import validate
        result = validate(
            path="app/tool_registry/capabilities.py",
            new_content="CAPABILITIES = {}",
        )
        assert not result.ok
        assert result.is_tier_immutable is True

    def test_outside_roots_rejected(self):
        """Paths outside the allowed roots — like workspace/ — get a
        plain rejection (NOT tier_immutable; different reason path)."""
        from app.change_requests import validate
        result = validate(path="workspace/foo.py", new_content="x = 1")
        assert not result.ok
        assert result.is_tier_immutable is False
        assert "outside the repo's allowed roots" in result.reason

    def test_path_traversal_rejected(self):
        from app.change_requests import validate
        result = validate(path="../etc/passwd", new_content="x = 1")
        assert not result.ok
        assert "traversal" in result.reason

    def test_absolute_path_rejected(self):
        from app.change_requests import validate
        result = validate(path="/app/agents/pim_agent.py", new_content="x = 1")
        assert not result.ok

    def test_blocked_pattern_env(self):
        from app.change_requests import validate
        result = validate(path="app/.env", new_content="SECRET=foo")
        assert not result.ok
        assert "blocked pattern" in result.reason

    def test_content_too_large(self):
        """Content > 1 MB is rejected to keep diff sizes manageable."""
        from app.change_requests import validate
        big = "x = 1\n" * 200_000  # ~1.2 MB
        result = validate(path="app/foo.py", new_content=big)
        assert not result.ok
        assert "exceeds" in result.reason

    def test_empty_path_rejected(self):
        from app.change_requests import validate
        result = validate(path="", new_content="x")
        assert not result.ok

    def test_is_protected(self):
        from app.change_requests import is_protected
        assert is_protected("app/auto_deployer.py") is True
        assert is_protected("app/agents/pim_agent.py") is False


# ── Models ──────────────────────────────────────────────────────────


class TestModels:

    def test_to_dict_from_dict_roundtrip(self):
        from app.change_requests import ChangeRequest, DecisionSource, Status
        cr = ChangeRequest(
            id="abc123",
            created_at="2026-05-04T16:00:00Z",
            requestor="coder",
            path="app/agents/pim_agent.py",
            new_content="new",
            old_content="old",
            reason="fix bug",
            diff="--- a/...\n+++ b/...",
            status=Status.PENDING,
        )
        d = cr.to_dict()
        cr2 = ChangeRequest.from_dict(d)
        assert cr2.id == cr.id
        assert cr2.status == Status.PENDING

    def test_status_predicates(self):
        from app.change_requests import ChangeRequest, Status
        cr = ChangeRequest(
            id="x", created_at="t", requestor="r", path="app/x.py",
            new_content="", old_content="", reason="", diff="",
            status=Status.APPLIED,
        )
        assert cr.is_rollbackable
        assert not cr.is_terminal

        cr.status = Status.ROLLED_BACK
        assert not cr.is_rollbackable
        assert cr.is_terminal

        cr.status = Status.PENDING
        assert not cr.is_rollbackable
        assert not cr.is_terminal


# ── Store ───────────────────────────────────────────────────────────


class TestStore:

    @pytest.fixture
    def store_dir(self, tmp_path, monkeypatch):
        from app.change_requests import store
        monkeypatch.setattr(store, "_STORE_DIR", tmp_path)
        monkeypatch.setattr(store, "_AUDIT_LOG", tmp_path / "audit.jsonl")
        store.reset_for_tests()
        return tmp_path

    def test_save_get_roundtrip(self, store_dir):
        from app.change_requests import ChangeRequest, Status, store
        cr = ChangeRequest(
            id="t1", created_at="2026-05-04T16:00:00Z",
            requestor="coder",
            path="app/agents/pim_agent.py",
            new_content="x", old_content="y",
            reason="r", diff="d",
            status=Status.PENDING,
        )
        store.save(cr, audit_event="created")
        loaded = store.get("t1")
        assert loaded.id == "t1"
        assert loaded.requestor == "coder"

    def test_audit_log_hash_chain(self, store_dir):
        from app.change_requests import ChangeRequest, Status, store
        for i in range(3):
            cr = ChangeRequest(
                id=f"t{i}", created_at="t", requestor="r",
                path=f"app/x{i}.py",
                new_content="", old_content="", reason="", diff="",
                status=Status.PENDING,
            )
            store.save(cr, audit_event="created")
        # Audit log: 3 entries, each linked to the previous via prev_hash
        log_path = store_dir / "audit.jsonl"
        assert log_path.exists()
        entries = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
        assert len(entries) == 3
        # First entry has empty prev_hash; subsequent ones link to predecessor
        assert entries[0]["prev_hash"] == ""
        assert entries[1]["prev_hash"] == entries[0]["entry_hash"]
        assert entries[2]["prev_hash"] == entries[1]["entry_hash"]

    def test_list_all_filtered_by_status(self, store_dir):
        from app.change_requests import ChangeRequest, Status, store
        for i, status in enumerate([Status.PENDING, Status.APPROVED, Status.REJECTED]):
            cr = ChangeRequest(
                id=f"t{i}", created_at=f"2026-05-0{i+1}T00:00:00Z",
                requestor="r", path=f"app/x{i}.py",
                new_content="", old_content="", reason="", diff="",
                status=status,
            )
            store.save(cr)
        pending = store.list_all(status=Status.PENDING)
        assert len(pending) == 1
        assert pending[0].id == "t0"

    def test_find_by_signal_ts(self, store_dir):
        from app.change_requests import ChangeRequest, Status, store
        cr = ChangeRequest(
            id="t1", created_at="t", requestor="r",
            path="app/x.py", new_content="", old_content="",
            reason="", diff="", status=Status.PENDING,
            signal_message_ts=1234567890,
        )
        store.save(cr)
        assert store.find_by_signal_ts(1234567890) == "t1"
        assert store.find_by_signal_ts(0) is None
        assert store.find_by_signal_ts(99999) is None


# ── Lifecycle ───────────────────────────────────────────────────────


class TestLifecycle:

    @pytest.fixture(autouse=True)
    def _store_dir(self, tmp_path, monkeypatch):
        from app.change_requests import store
        monkeypatch.setattr(store, "_STORE_DIR", tmp_path)
        monkeypatch.setattr(store, "_AUDIT_LOG", tmp_path / "audit.jsonl")
        store.reset_for_tests()

    def test_create_request_happy_path(self):
        from app.change_requests import Status, create_request
        cr = create_request(
            requestor="coder",
            path="app/agents/pim_agent.py",
            new_content="x = 1",
            old_content="",
            reason="add import",
        )
        assert cr.status == Status.PENDING
        assert cr.diff != ""

    def test_create_request_tier_immutable(self):
        from app.change_requests import Status, create_request
        cr = create_request(
            requestor="coder",
            path="app/auto_deployer.py",
            new_content="x = 1",
            old_content="",
            reason="something",
        )
        assert cr.status == Status.TIER_IMMUTABLE_REFUSED
        assert "TIER_IMMUTABLE" in cr.decision_reason

    def test_create_request_validation_failure(self):
        """Outside-roots paths get REJECTED status (NOT tier_immutable_refused)."""
        from app.change_requests import Status, create_request
        cr = create_request(
            requestor="coder",
            path="workspace/foo.py",
            new_content="x = 1",
            old_content="",
            reason="x",
        )
        assert cr.status == Status.REJECTED
        assert "outside" in cr.decision_reason

    def test_approve_pending_to_approved(self):
        from app.change_requests import (
            DecisionSource, Status, approve, create_request,
        )
        cr = create_request(
            requestor="coder", path="app/x.py",
            new_content="x", old_content="", reason="r",
        )
        updated = approve(cr.id, source=DecisionSource.SIGNAL_THUMBS_UP)
        assert updated.status == Status.APPROVED
        assert updated.decided_by == DecisionSource.SIGNAL_THUMBS_UP
        assert updated.decided_at is not None

    def test_approve_idempotent(self):
        from app.change_requests import (
            DecisionSource, approve, create_request,
        )
        cr = create_request(
            requestor="coder", path="app/x.py",
            new_content="x", old_content="", reason="r",
        )
        approve(cr.id, source=DecisionSource.SIGNAL_THUMBS_UP)
        again = approve(cr.id, source=DecisionSource.SIGNAL_THUMBS_UP)
        assert again.id == cr.id  # no error

    def test_reject_pending_to_rejected(self):
        from app.change_requests import (
            DecisionSource, Status, create_request, reject,
        )
        cr = create_request(
            requestor="coder", path="app/x.py",
            new_content="x", old_content="", reason="r",
        )
        updated = reject(
            cr.id, source=DecisionSource.SIGNAL_THUMBS_DOWN,
            decision_reason="user said no",
        )
        assert updated.status == Status.REJECTED

    def test_illegal_transition_raises(self):
        """Cannot approve a TIER_IMMUTABLE_REFUSED request."""
        from app.change_requests import DecisionSource, approve, create_request
        cr = create_request(
            requestor="coder", path="app/auto_deployer.py",
            new_content="x", old_content="", reason="r",
        )
        with pytest.raises(ValueError, match="cannot approve"):
            approve(cr.id, source=DecisionSource.REACT_APPROVE)


# ── HTTP endpoints ───────────────────────────────────────────────────


class TestAPI:

    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GATEWAY_AUTH_REQUIRED", "0")
        monkeypatch.setenv("CREWAI_TELEMETRY_OPT_OUT", "true")
        from app.change_requests import store
        monkeypatch.setattr(store, "_STORE_DIR", tmp_path)
        monkeypatch.setattr(store, "_AUDIT_LOG", tmp_path / "audit.jsonl")
        store.reset_for_tests()
        from app.main import app
        from fastapi.testclient import TestClient
        return TestClient(app)

    def _make_pending(self):
        from app.change_requests import create_request
        return create_request(
            requestor="coder",
            path="app/agents/pim_agent.py",
            new_content="x = 1",
            old_content="",
            reason="test",
        )

    def test_list_empty(self, client):
        r = client.get("/api/cp/changes")
        assert r.status_code == 200
        assert r.json() == {"count": 0, "changes": []}

    def test_list_with_status_filter(self, client):
        cr = self._make_pending()
        r = client.get("/api/cp/changes?status=pending")
        data = r.json()
        assert data["count"] == 1
        assert data["changes"][0]["id"] == cr.id

    def test_invalid_status_filter(self, client):
        r = client.get("/api/cp/changes?status=bogus")
        assert r.status_code == 400

    def test_get_detail(self, client):
        cr = self._make_pending()
        r = client.get(f"/api/cp/changes/{cr.id}")
        assert r.status_code == 200
        assert r.json()["id"] == cr.id

    def test_get_404(self, client):
        r = client.get("/api/cp/changes/nope")
        assert r.status_code == 404

    def test_reject_via_react(self, client):
        cr = self._make_pending()
        r = client.post(
            f"/api/cp/changes/{cr.id}/reject",
            json={"operator": "andrus", "reason": "no"},
        )
        assert r.status_code == 200
        assert r.json()["change"]["status"] == "rejected"

    def test_reject_already_rejected_returns_409(self, client):
        cr = self._make_pending()
        client.post(f"/api/cp/changes/{cr.id}/reject", json={})
        r = client.post(f"/api/cp/changes/{cr.id}/reject", json={})
        assert r.status_code == 409

    def test_approve_tier_immutable_returns_403(self, client):
        """Attempt to approve a TIER_IMMUTABLE_REFUSED change → 403."""
        from app.change_requests import create_request
        cr = create_request(
            requestor="coder", path="app/auto_deployer.py",
            new_content="x", old_content="", reason="r",
        )
        r = client.post(f"/api/cp/changes/{cr.id}/approve", json={"operator": "andrus"})
        assert r.status_code == 403
        assert "TIER_IMMUTABLE" in r.json()["detail"]

    def test_rollback_pending_returns_409(self, client):
        cr = self._make_pending()
        r = client.post(f"/api/cp/changes/{cr.id}/rollback", json={"operator": "andrus"})
        assert r.status_code == 409


# ── Signal integration ──────────────────────────────────────────────


class TestSignalIntegration:

    def test_build_ask_body_shape(self):
        from app.change_requests import build_ask_body
        body = build_ask_body(
            request_id="abc123",
            requestor="coder",
            path="app/agents/pim_agent.py",
            reason="add missing import",
            diff="--- a/...\n+++ b/...\n@@ -1,3 +1,4 @@\n+from x import y\n",
        )
        assert "CHANGE REQUEST" in body
        assert "app/agents/pim_agent.py" in body
        assert "add missing import" in body
        assert "👍" in body
        assert "👎" in body
        assert "abc123" in body

    def test_build_ask_body_truncates_long_diff(self):
        from app.change_requests import build_ask_body
        big_diff = "x" * 5000
        body = build_ask_body(
            request_id="x", requestor="coder",
            path="app/x.py", reason="r", diff=big_diff,
        )
        assert "diff truncated" in body
        assert "/api/cp/changes/x" in body  # pointer to full diff


# ── Agent tool ──────────────────────────────────────────────────────


class TestAgentTool:

    def setup_method(self) -> None:
        from app.tool_registry import ToolRegistry
        from app.tool_registry.boot import boot_registry
        ToolRegistry.reset_for_tests()
        boot_registry(snapshot_to_postgres=False, index_to_chromadb=False, sync_forge=False)

    def test_tool_registered(self):
        from app.tool_registry import ToolRegistry
        spec = ToolRegistry.instance().get("request_restricted_write")
        assert spec is not None

    def test_factory_returns_one_tool(self):
        from app.tools.restricted_write_tool import create_restricted_write_tools
        tools = create_restricted_write_tools()
        assert len(tools) == 1
        assert tools[0].name == "request_restricted_write"

    def test_tool_run_tier_immutable_refused(self, tmp_path, monkeypatch):
        """Calling the tool with a TIER_IMMUTABLE path returns the
        REFUSED message without sending Signal."""
        from app.change_requests import store
        monkeypatch.setattr(store, "_STORE_DIR", tmp_path)
        monkeypatch.setattr(store, "_AUDIT_LOG", tmp_path / "audit.jsonl")
        store.reset_for_tests()

        from app.tools.restricted_write_tool import create_restricted_write_tools
        [tool] = create_restricted_write_tools()
        out = tool._run(
            path="app/auto_deployer.py",
            new_content="x = 1",
            old_content="",
            reason="naughty",
        )
        assert "REFUSED" in out
        assert "TIER_IMMUTABLE" in out

    def test_tool_run_validation_rejected(self, tmp_path, monkeypatch):
        from app.change_requests import store
        monkeypatch.setattr(store, "_STORE_DIR", tmp_path)
        monkeypatch.setattr(store, "_AUDIT_LOG", tmp_path / "audit.jsonl")
        store.reset_for_tests()

        from app.tools.restricted_write_tool import create_restricted_write_tools
        [tool] = create_restricted_write_tools()
        out = tool._run(
            path="workspace/foo.py",  # outside allowed roots
            new_content="x",
            old_content="",
            reason="r",
        )
        assert "REJECTED" in out

    def test_tool_run_pending_when_signal_unavailable(self, tmp_path, monkeypatch):
        """When Signal owner is not configured, the tool returns
        PENDING status with a pointer to React."""
        from app.change_requests import store
        monkeypatch.setattr(store, "_STORE_DIR", tmp_path)
        monkeypatch.setattr(store, "_AUDIT_LOG", tmp_path / "audit.jsonl")
        store.reset_for_tests()

        # Mock Signal: signal_client.send_message_blocking returns None (failure).
        # signal.py imports send_message_blocking lazily inside send_ask(), so
        # patch the source module not the consumer.
        with patch("app.signal_client.send_message_blocking", return_value=None):
            from app.tools.restricted_write_tool import create_restricted_write_tools
            [tool] = create_restricted_write_tools()
            out = tool._run(
                path="app/agents/pim_agent.py",
                new_content="x = 1",
                old_content="",
                reason="bug fix",
            )
        assert "PENDING" in out
        assert "/api/cp/changes/" in out
