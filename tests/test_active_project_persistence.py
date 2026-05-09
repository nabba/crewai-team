"""Tests for active-project persistence across process boundaries.

Pre-fix behaviour:
    explicit `switch workspace to eesti mets` → in-memory only;
    gateway restart wiped it; next ticket landed under default/PLG.

Post-fix behaviour (this test file):
    `switch()` writes through to ``workspace/control_plane/active_project.json``;
    on first ``get_active_project_id()`` call after process start, the
    file is read back and the in-memory state restored.

Live reproduction that motivated the fix:
    - 1:26 AM "Switch to workspace eesti mets" → "Switched to project: Eesti mets"
    - (gateway restart overnight)
    - 8:18 AM new task → routed to default/PLG
    - 8:22 AM user "what is the current workspace?" → "PLG"

These tests reset the ProjectManager class state between cases (it's
class-level, not instance-level) and redirect the JsonStore to a
tmp_path so the real on-disk file isn't polluted.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def isolated_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect the JsonStore + reset class state between tests.

    The store path is module-level (constructed at import); we patch
    its ``path`` attribute directly.
    """
    from app.control_plane import projects as projects_mod

    # Redirect the store
    store = projects_mod._ACTIVE_PROJECT_STORE
    store.path = tmp_path / "active_project.json"

    # Reset class-level state — pytest doesn't isolate module globals
    projects_mod.ProjectManager._active_project_id = None
    projects_mod.ProjectManager._active_project_source = None
    projects_mod.ProjectManager._persisted_loaded = False

    yield store

    # Tear down — leave clean state for the next test
    projects_mod.ProjectManager._active_project_id = None
    projects_mod.ProjectManager._active_project_source = None
    projects_mod.ProjectManager._persisted_loaded = False


# ── Read/write helpers ──────────────────────────────────────────────


class TestPersistedReadWrite:
    """Direct unit tests on the module-level helpers, without
    constructing a ProjectManager."""

    def test_empty_store_returns_none(self, isolated_store) -> None:
        from app.control_plane.projects import _read_persisted_active

        pid, source = _read_persisted_active()
        assert pid is None and source is None

    def test_round_trip(self, isolated_store) -> None:
        from app.control_plane.projects import (
            _read_persisted_active,
            _write_persisted_active,
        )

        _write_persisted_active("abc-123", "user")
        pid, source = _read_persisted_active()
        assert pid == "abc-123"
        assert source == "user"

    def test_write_none_clears(self, isolated_store) -> None:
        from app.control_plane.projects import (
            _read_persisted_active,
            _write_persisted_active,
        )

        _write_persisted_active("abc-123", "user")
        assert _read_persisted_active() == ("abc-123", "user")

        _write_persisted_active(None, None)
        pid, source = _read_persisted_active()
        assert pid is None and source is None

    def test_partial_data_is_safe(
        self, isolated_store, tmp_path: Path,
    ) -> None:
        """A malformed file (missing project_id, wrong types) must not
        crash; we return (None, None) and let the caller fall back to
        the default project."""
        from app.control_plane.projects import _read_persisted_active

        # Hand-craft a junk file
        store_path = isolated_store.path
        store_path.parent.mkdir(parents=True, exist_ok=True)
        store_path.write_text('{"unrelated_key": 42}')
        pid, source = _read_persisted_active()
        assert pid is None and source is None

        # Project_id present but not a string → reject
        store_path.write_text('{"project_id": 12345, "source": "user"}')
        pid, source = _read_persisted_active()
        assert pid is None


# ── Manager: lazy-restore on first read ─────────────────────────────


class TestLazyRestore:

    def test_no_persisted_state_falls_back_to_default(
        self, isolated_store,
    ) -> None:
        """No file on disk → behaves like pre-fix (default-project
        fallback)."""
        from app.control_plane.projects import ProjectManager

        pm = ProjectManager()
        with patch.object(pm, "get_default_project_id", return_value="default-uuid"):
            pid = pm.get_active_project_id()
        assert pid == "default-uuid"

    def test_persisted_state_restored_on_first_read(
        self, isolated_store,
    ) -> None:
        """File on disk → first ``get_active_project_id()`` reads it
        and the manager remembers."""
        from app.control_plane.projects import (
            ProjectManager,
            _write_persisted_active,
        )

        # Simulate previous-run state
        _write_persisted_active("eesti-mets-uuid", "user")

        pm = ProjectManager()
        # The lazy-load needs to verify the project exists; mock that.
        with patch.object(
            pm, "get_by_id",
            return_value={"id": "eesti-mets-uuid", "name": "Eesti mets"},
        ):
            pid = pm.get_active_project_id()
        assert pid == "eesti-mets-uuid"
        # Source restored too — needed for sticky-user-pick guard
        assert ProjectManager._active_project_source == "user"

    def test_stale_persisted_state_cleared(
        self, isolated_store,
    ) -> None:
        """If a project was deleted while the gateway was down, the
        stale file is dropped and the manager falls back to default."""
        from app.control_plane.projects import (
            ProjectManager,
            _read_persisted_active,
            _write_persisted_active,
        )

        _write_persisted_active("ghost-uuid", "user")

        pm = ProjectManager()
        with (
            patch.object(pm, "get_by_id", return_value=None),
            patch.object(
                pm, "get_default_project_id",
                return_value="default-uuid",
            ),
        ):
            pid = pm.get_active_project_id()
        assert pid == "default-uuid"
        # Stale file cleared
        cleaned_pid, _ = _read_persisted_active()
        assert cleaned_pid is None

    def test_lazy_load_only_runs_once(
        self, isolated_store,
    ) -> None:
        """Subsequent ``get_active_project_id()`` calls don't re-hit
        the disk — the in-memory copy is the source of truth after
        first load."""
        from app.control_plane.projects import (
            ProjectManager,
            _write_persisted_active,
        )

        _write_persisted_active("uuid-1", "user")

        pm = ProjectManager()

        get_by_id_mock = MagicMock(return_value={"id": "uuid-1", "name": "p1"})
        with patch.object(pm, "get_by_id", get_by_id_mock):
            pm.get_active_project_id()
            pm.get_active_project_id()
            pm.get_active_project_id()

        # The DB lookup runs exactly once (the lazy-load); subsequent
        # calls hit the in-memory state directly
        assert get_by_id_mock.call_count == 1


# ── Manager: switch() write-through ─────────────────────────────────


class TestSwitchWriteThrough:

    def test_switch_persists_to_disk(self, isolated_store) -> None:
        """A successful ``switch()`` call writes the new state to the
        on-disk store."""
        from app.control_plane.projects import (
            ProjectManager,
            _read_persisted_active,
        )

        pm = ProjectManager()
        target = {"id": "eesti-uuid", "name": "Eesti mets"}
        with patch.object(pm, "get_by_name", return_value=target):
            result = pm.switch("eesti mets", source="user")

        assert result == target
        pid, source = _read_persisted_active()
        assert pid == "eesti-uuid"
        assert source == "user"

    def test_auto_switch_persists_too(self, isolated_store) -> None:
        """Auto-detector picks also persist (so the sticky-user-pick
        guard survives restart)."""
        from app.control_plane.projects import (
            ProjectManager,
            _read_persisted_active,
        )

        pm = ProjectManager()
        target = {"id": "plg-uuid", "name": "PLG"}
        with patch.object(pm, "get_by_name", return_value=target):
            pm.switch("PLG", source="auto")

        pid, source = _read_persisted_active()
        assert pid == "plg-uuid"
        assert source == "auto"

    def test_sticky_user_pick_blocks_auto_persists_neither(
        self, isolated_store,
    ) -> None:
        """Pre-existing user pick + later auto-suggestion → auto is
        ignored AND the disk state is unchanged."""
        from app.control_plane.projects import (
            ProjectManager,
            _read_persisted_active,
        )

        pm = ProjectManager()

        # First: user picks Eesti mets
        with patch.object(
            pm, "get_by_name",
            return_value={"id": "eesti-uuid", "name": "Eesti mets"},
        ):
            pm.switch("Eesti mets", source="user")

        # Then: auto-detector tries to switch to PLG
        with patch.object(
            pm, "get_by_name",
            return_value={"id": "plg-uuid", "name": "PLG"},
        ):
            pm.switch("PLG", source="auto")

        # Disk state still reflects the user pick
        pid, source = _read_persisted_active()
        assert pid == "eesti-uuid"
        assert source == "user"

    def test_switch_unknown_project_does_not_persist(
        self, isolated_store,
    ) -> None:
        """``switch("nonexistent")`` returns None and the disk state
        is unchanged."""
        from app.control_plane.projects import (
            ProjectManager,
            _read_persisted_active,
            _write_persisted_active,
        )

        # Pre-state: user is on Eesti mets
        _write_persisted_active("eesti-uuid", "user")

        pm = ProjectManager()
        with patch.object(pm, "get_by_name", return_value=None):
            result = pm.switch("Atlantis")
        assert result is None

        # Disk state preserved
        pid, source = _read_persisted_active()
        assert pid == "eesti-uuid"
        assert source == "user"


# ── End-to-end: simulate gateway restart ────────────────────────────


class TestRestartScenario:
    """Reproduces the live failure shape — explicit switch at 1:26 AM,
    gateway restart overnight, next task at 8:18 AM should still see
    the chosen project."""

    def test_switch_then_simulated_restart(self, isolated_store) -> None:
        from app.control_plane import projects as projects_mod

        # ── Process A: user switches to Eesti mets ─────────────────
        pm_a = projects_mod.ProjectManager()
        target = {"id": "eesti-uuid", "name": "Eesti mets"}
        with patch.object(pm_a, "get_by_name", return_value=target):
            pm_a.switch("Eesti mets", source="user")

        # ── Simulated gateway restart: reset the class state ───────
        projects_mod.ProjectManager._active_project_id = None
        projects_mod.ProjectManager._active_project_source = None
        projects_mod.ProjectManager._persisted_loaded = False

        # ── Process B: fresh manager — should restore from disk ────
        pm_b = projects_mod.ProjectManager()
        with patch.object(
            pm_b, "get_by_id",
            return_value={"id": "eesti-uuid", "name": "Eesti mets"},
        ):
            pid = pm_b.get_active_project_id()
        # The user's choice survived the simulated restart
        assert pid == "eesti-uuid"
        assert (
            projects_mod.ProjectManager._active_project_source == "user"
        )
