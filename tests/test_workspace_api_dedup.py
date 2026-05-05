"""Regression test for the workspaces-listing dedup bug.

The bug: the in-memory ``_gates`` registry (``app.subia.scene.buffer``)
ends up with multiple snapshots that resolve to the same display name
when a project is referenced by both UUID and legacy name in the same
gateway lifetime. The React UI rendered every snapshot as a separate
tab — confusing duplicates labeled "PLG 4/4" stacked on top of each
other.

The fix in ``app/api/workspace_api.py`` collapses by resolved
display name, keeping the snapshot with the most signal (cycle →
active_count → capacity tiebreak). This test verifies that:

  1. A fake registry with duplicate-resolving gates is collapsed
     to one entry per display.
  2. The "winner" is the snapshot with cycle > 0 (the active gate),
     not the empty reconcile gate.
  3. A clean registry passes through unchanged.

Also covers the ``POST /api/workspaces/_dedup`` operator GC endpoint
that prunes stale gates from the underlying ``_gates`` dict.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest


class _FakeGate:
    """Stands in for ``CompetitiveGate``. Returns a fixed snapshot."""

    def __init__(
        self,
        *,
        cycle: int = 0,
        capacity: int = 3,
        active_count: int = 0,
        peripheral_count: int = 0,
    ) -> None:
        self._snap = {
            "cycle": cycle,
            "capacity": capacity,
            "active_count": active_count,
            "peripheral_count": peripheral_count,
            "active_items": [],
            "salience_distribution": {},
        }

    def get_snapshot(self) -> dict:
        return dict(self._snap)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    """A FastAPI TestClient with auth disabled."""
    monkeypatch.setenv("GATEWAY_AUTH_REQUIRED", "0")
    monkeypatch.setenv("CREWAI_TELEMETRY_OPT_OUT", "true")

    from app.main import app
    from fastapi.testclient import TestClient

    return TestClient(app)


# ── List dedup ──────────────────────────────────────────────────────


class TestListDedup:
    """Verifies the GET /api/workspaces serializer collapses duplicates."""

    def _fake_projects(self, projects: list[dict]):
        """Patch get_projects().list_all() + .get_by_id() in one shot."""

        class _FakeProjects:
            def list_all(self_inner):
                return projects

            def get_by_id(self_inner, pid: str):
                for p in projects:
                    if p.get("id") == pid:
                        return p
                return None

        return _FakeProjects()

    def test_collision_collapses_to_one_per_display(
        self, client, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Two gates resolving to the same display ('default') — one
        active (cycle 1), one empty (cycle 0). Result has one entry,
        the active one wins."""

        # Two CP projects: one default (UUID) + one PLG (UUID)
        projects = [
            {"id": "676a8f70-d369-44bd-b3a4-76ff0635657a", "name": "default"},
            {"id": "969f4a48-866c-42dc-b0a3-6d17b5dd304d", "name": "PLG"},
        ]

        # _gates with FOUR entries: 2 per project (active + empty)
        # The active ones are keyed by the legacy NAME (pre-migration shape);
        # the empty ones are keyed by the new UUID (current reconcile output).
        # Both resolve to the same display via _project_display_name:
        #   pid='default' (8 chars, not UUID-shape) → passthrough → 'default'
        #   pid='676a8f70-...' (36 chars UUID) → CP lookup → 'default'
        fake_gates = {
            "default": _FakeGate(cycle=1, capacity=4, active_count=1),
            "676a8f70-d369-44bd-b3a4-76ff0635657a": _FakeGate(
                cycle=0, capacity=3, active_count=0,
            ),
            "PLG": _FakeGate(cycle=6, capacity=4, active_count=4),
            "969f4a48-866c-42dc-b0a3-6d17b5dd304d": _FakeGate(
                cycle=0, capacity=3, active_count=0,
            ),
        }

        with (
            patch(
                "app.subia.scene.buffer.list_workspaces",
                lambda: {pid: gate.get_snapshot()
                         for pid, gate in fake_gates.items()},
            ),
            patch(
                "app.control_plane.projects.get_projects",
                return_value=self._fake_projects(projects),
            ),
        ):
            r = client.get("/api/workspaces")

        body = r.json()
        # 4 input snapshots, 2 unique displays after dedup
        assert body["count"] == 2
        # Both kept entries are the high-signal ones (cycle > 0)
        for ws in body["workspaces"]:
            assert ws["cycle"] > 0, (
                f"expected high-signal entry, got {ws}"
            )
        displays = {ws["display_name"] for ws in body["workspaces"]}
        assert displays == {"default", "PLG"}

    def test_no_duplicates_passes_through_unchanged(
        self, client, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Clean registry (one gate per project) returns all entries."""


        projects = [
            {"id": "676a8f70-d369-44bd-b3a4-76ff0635657a", "name": "default"},
            {"id": "55c7c5b2-d9d0-4fda-8ed1-e021c6fd9211", "name": "Archibal"},
        ]
        fake_gates = {
            "676a8f70-d369-44bd-b3a4-76ff0635657a": _FakeGate(
                cycle=2, capacity=4, active_count=2,
            ),
            "55c7c5b2-d9d0-4fda-8ed1-e021c6fd9211": _FakeGate(
                cycle=0, capacity=3, active_count=0,
            ),
        }
        with (
            patch(
                "app.subia.scene.buffer.list_workspaces",
                lambda: {pid: gate.get_snapshot()
                         for pid, gate in fake_gates.items()},
            ),
            patch(
                "app.control_plane.projects.get_projects",
                return_value=self._fake_projects(projects),
            ),
        ):
            r = client.get("/api/workspaces")
        body = r.json()
        assert body["count"] == 2
        displays = {ws["display_name"] for ws in body["workspaces"]}
        assert displays == {"default", "Archibal"}

    def test_active_winner_when_both_have_same_cycle(
        self, client, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Cycle tie → break by active_count, then capacity."""


        projects = [
            {"id": "676a8f70-d369-44bd-b3a4-76ff0635657a", "name": "default"},
        ]
        fake_gates = {
            "default": _FakeGate(cycle=0, capacity=3, active_count=0),
            "676a8f70-d369-44bd-b3a4-76ff0635657a": _FakeGate(
                cycle=0, capacity=4, active_count=1,
            ),
        }
        with (
            patch(
                "app.subia.scene.buffer.list_workspaces",
                lambda: {pid: gate.get_snapshot()
                         for pid, gate in fake_gates.items()},
            ),
            patch(
                "app.control_plane.projects.get_projects",
                return_value=self._fake_projects(projects),
            ),
        ):
            r = client.get("/api/workspaces")
        body = r.json()
        assert body["count"] == 1
        ws = body["workspaces"][0]
        # Picked the one with active_count=1 (vs 0)
        assert ws["active_count"] == 1
        assert ws["capacity"] == 4
        assert ws["display_name"] == "default"


# ── Operator GC endpoint ────────────────────────────────────────────


class TestDedupOperatorEndpoint:
    """POST /api/workspaces/_dedup actually mutates _gates."""

    def test_dedup_endpoint_removes_stale_gates(
        self, client, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from app.subia.scene import buffer as buffer_mod

        # Snapshot the real _gates so we don't pollute it permanently
        original_gates = dict(buffer_mod._gates)
        try:
            # Inject a synthetic duplicate state
            buffer_mod._gates.clear()
            buffer_mod._gates["default"] = _FakeGate(
                cycle=1, capacity=4, active_count=1,
            )
            buffer_mod._gates["676a8f70-d369-44bd-b3a4-76ff0635657a"] = (
                _FakeGate(cycle=0, capacity=3, active_count=0)
            )

            # Patch CP projects so display resolution works
            class _FP:
                def list_all(self):
                    return [{"id": "676a8f70-d369-44bd-b3a4-76ff0635657a",
                             "name": "default"}]

                def get_by_id(self, pid):
                    if pid == "676a8f70-d369-44bd-b3a4-76ff0635657a":
                        return {"id": pid, "name": "default"}
                    return None

            with patch(
                "app.control_plane.projects.get_projects",
                return_value=_FP(),
            ):
                r = client.post("/api/workspaces/_dedup")

            assert r.status_code == 200
            body = r.json()
            assert body["removed_count"] == 1
            # The kept gate is the higher-signal one. The stale one should
            # have been removed from _gates.
            remaining_keys = set(buffer_mod._gates.keys())
            # Either the legacy 'default' key was kept (active gate) and
            # the UUID key removed, depending on which had higher signal.
            # In our setup, "default" has cycle=1 (higher), so the UUID
            # gate is dropped.
            assert "default" in remaining_keys
            assert "676a8f70-d369-44bd-b3a4-76ff0635657a" not in remaining_keys
        finally:
            buffer_mod._gates.clear()
            buffer_mod._gates.update(original_gates)

    def test_dedup_endpoint_idempotent_on_clean_state(
        self, client, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from app.subia.scene import buffer as buffer_mod

        original_gates = dict(buffer_mod._gates)
        try:
            buffer_mod._gates.clear()
            buffer_mod._gates["676a8f70-d369-44bd-b3a4-76ff0635657a"] = (
                _FakeGate(cycle=0, capacity=3, active_count=0)
            )

            class _FP:
                def list_all(self):
                    return [{"id": "676a8f70-d369-44bd-b3a4-76ff0635657a",
                             "name": "default"}]

                def get_by_id(self, pid):
                    return {"id": pid, "name": "default"}

            with patch(
                "app.control_plane.projects.get_projects",
                return_value=_FP(),
            ):
                r = client.post("/api/workspaces/_dedup")
            assert r.status_code == 200
            assert r.json()["removed_count"] == 0
            assert "676a8f70-d369-44bd-b3a4-76ff0635657a" in buffer_mod._gates
        finally:
            buffer_mod._gates.clear()
            buffer_mod._gates.update(original_gates)
