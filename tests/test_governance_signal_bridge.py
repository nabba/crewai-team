"""Tests for app/governance_signal_bridge.py.

Verifies the JSON-sidecar map from Signal timestamps to governance
request IDs. Covers register/find/unregister/purge round-trips and
the prefix-match helper used by the text-command fallback.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _import_bridge_with_workspace(tmp_path, monkeypatch):
    """Re-import the bridge with WORKSPACE_ROOT pointing at tmp_path."""
    import importlib
    import app.paths as paths_mod
    monkeypatch.setattr(paths_mod, "WORKSPACE_ROOT", tmp_path, raising=True)
    import app.governance_signal_bridge as bridge
    importlib.reload(bridge)
    return bridge


class TestRegisterAndFind:
    def test_register_then_find_returns_request_id(self, tmp_path, monkeypatch):
        bridge = _import_bridge_with_workspace(tmp_path, monkeypatch)
        bridge.register(1700000000123, "abc-123-uuid")
        assert bridge.find_request_id(1700000000123) == "abc-123-uuid"

    def test_find_for_unknown_ts_returns_none(self, tmp_path, monkeypatch):
        bridge = _import_bridge_with_workspace(tmp_path, monkeypatch)
        assert bridge.find_request_id(99999) is None

    def test_find_with_zero_ts_returns_none(self, tmp_path, monkeypatch):
        bridge = _import_bridge_with_workspace(tmp_path, monkeypatch)
        bridge.register(0, "should-not-store")
        assert bridge.find_request_id(0) is None

    def test_register_with_empty_id_is_a_noop(self, tmp_path, monkeypatch):
        bridge = _import_bridge_with_workspace(tmp_path, monkeypatch)
        bridge.register(1700000001, "")
        assert bridge.find_request_id(1700000001) is None

    def test_multiple_entries_are_all_resolvable(self, tmp_path, monkeypatch):
        bridge = _import_bridge_with_workspace(tmp_path, monkeypatch)
        bridge.register(1, "id-1")
        bridge.register(2, "id-2")
        bridge.register(3, "id-3")
        assert bridge.find_request_id(1) == "id-1"
        assert bridge.find_request_id(2) == "id-2"
        assert bridge.find_request_id(3) == "id-3"


class TestUnregister:
    def test_unregister_drops_only_matching_request_id(self, tmp_path, monkeypatch):
        bridge = _import_bridge_with_workspace(tmp_path, monkeypatch)
        bridge.register(10, "uuid-keep")
        bridge.register(20, "uuid-drop")
        bridge.unregister("uuid-drop")
        assert bridge.find_request_id(10) == "uuid-keep"
        assert bridge.find_request_id(20) is None

    def test_unregister_empty_string_is_safe(self, tmp_path, monkeypatch):
        bridge = _import_bridge_with_workspace(tmp_path, monkeypatch)
        bridge.register(11, "uuid-still-here")
        bridge.unregister("")
        assert bridge.find_request_id(11) == "uuid-still-here"


class TestExpiry:
    def test_entries_older_than_25h_are_purged_on_read(self, tmp_path, monkeypatch):
        bridge = _import_bridge_with_workspace(tmp_path, monkeypatch)
        # Seed an entry with an artificially old timestamp.
        old_epoch = datetime.now(timezone.utc).timestamp() - (26 * 3600)
        seed = {
            "42": {
                "request_id": "ancient-uuid",
                "created_at_epoch": old_epoch,
                "created_at": "(synthetic)",
            }
        }
        (tmp_path / "governance_signal_bridge.json").write_text(json.dumps(seed))
        assert bridge.find_request_id(42) is None
        # Read should have also persisted the purged file.
        data = json.loads((tmp_path / "governance_signal_bridge.json").read_text())
        assert "42" not in data

    def test_fresh_entries_survive_purge(self, tmp_path, monkeypatch):
        bridge = _import_bridge_with_workspace(tmp_path, monkeypatch)
        bridge.register(123, "fresh-uuid")
        # Trigger another read (which purges)
        bridge.register(124, "also-fresh")
        assert bridge.find_request_id(123) == "fresh-uuid"
        assert bridge.find_request_id(124) == "also-fresh"


class _StubGovernanceModule:
    """A drop-in replacement for app.control_plane.governance for tests
    that don't have psycopg2 installed (which the real module imports
    transitively via app.control_plane.db)."""

    def __init__(self, pending):
        self._pending = pending

    def get_governance(self):
        outer = self

        class _Gate:
            def get_pending(self):
                return outer._pending

        return _Gate()


def _inject_stub_governance(monkeypatch, pending):
    """Replace app.control_plane.governance in sys.modules with a stub
    that returns the given pending rows from get_pending()."""
    import types
    stub = types.ModuleType("app.control_plane.governance")
    stub_state = _StubGovernanceModule(pending)
    stub.get_governance = stub_state.get_governance  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "app.control_plane.governance", stub)


class TestPrefixHelper:
    def test_prefix_match_returns_unique_pending(self, tmp_path, monkeypatch):
        bridge = _import_bridge_with_workspace(tmp_path, monkeypatch)
        _inject_stub_governance(monkeypatch, [
            {"id": "73483ad2-aaaa-bbbb-cccc-deadbeef0001",
             "request_type": "code_change", "status": "pending"},
            {"id": "11112222-3333-4444-5555-666677778888",
             "request_type": "budget_override", "status": "pending"},
        ])
        row = bridge.find_pending_by_id_prefix("73483ad2")
        assert row is not None
        assert row["id"].startswith("73483ad2")

    def test_prefix_no_match_returns_none(self, tmp_path, monkeypatch):
        bridge = _import_bridge_with_workspace(tmp_path, monkeypatch)
        _inject_stub_governance(monkeypatch, [
            {"id": "abcd-1111", "request_type": "code_change", "status": "pending"},
        ])
        assert bridge.find_pending_by_id_prefix("99999999") is None

    def test_ambiguous_prefix_returns_none(self, tmp_path, monkeypatch):
        """Two rows with the same prefix → caller should fall back to
        the React dashboard rather than silently picking one."""
        bridge = _import_bridge_with_workspace(tmp_path, monkeypatch)
        _inject_stub_governance(monkeypatch, [
            {"id": "73483ad2-aaaa", "request_type": "code_change", "status": "pending"},
            {"id": "73483ad2-bbbb", "request_type": "code_change", "status": "pending"},
        ])
        assert bridge.find_pending_by_id_prefix("73483ad2") is None

    def test_empty_prefix_returns_none(self, tmp_path, monkeypatch):
        bridge = _import_bridge_with_workspace(tmp_path, monkeypatch)
        assert bridge.find_pending_by_id_prefix("") is None


class TestFailureIsolation:
    def test_corrupt_file_is_treated_as_empty(self, tmp_path, monkeypatch):
        bridge = _import_bridge_with_workspace(tmp_path, monkeypatch)
        (tmp_path / "governance_signal_bridge.json").write_text("{not valid json")
        # Should not raise; just behaves like an empty registry.
        assert bridge.find_request_id(1) is None
        # Subsequent writes should still work.
        bridge.register(2, "ok-uuid")
        assert bridge.find_request_id(2) == "ok-uuid"
