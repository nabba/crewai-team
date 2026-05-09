"""Tests for TicketManager.move_ticket — Postgres-backed cross-project
ticket move added 2026-05-09.

Why this lives in its own file
------------------------------
The existing ``tests/test_control_plane.py`` has a heavy
module-level setup that stubs psycopg2 and ten other third-party
modules.  Adding 5 more tests there would dilute the test name
listing and make targeted reruns awkward (``pytest -k move`` would
trigger the whole module-level import dance).  The new file mirrors
the same stub pattern so it runs cleanly on the dev host.

All Postgres calls are mocked — no real database required.
"""
from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock, patch


# ── Stub heavy dependencies the dev host may not have ──────────────
_STUBS_INSERTED: list[str] = []
_PSYCOPG2_INSERTED: list[str] = []

_CANDIDATES = [
    "crewai", "crewai.tools", "langchain_anthropic", "docker",
    "chromadb", "sentence_transformers", "trafilatura",
    "youtube_transcript_api", "brave_search", "apscheduler",
    "firebase_admin", "pypdf", "docx", "openpyxl", "PIL",
    "litellm", "bs4", "neo4j", "mem0",
]
for _mod in _CANDIDATES:
    if _mod in sys.modules:
        continue
    try:
        __import__(_mod)
        continue
    except Exception:
        pass
    m = types.ModuleType(_mod)
    if _mod == "crewai.tools":
        m.tool = lambda name: (lambda fn: fn)
    sys.modules[_mod] = m
    _STUBS_INSERTED.append(_mod)


_mock_psycopg2 = MagicMock()
_mock_psycopg2.InterfaceError = type("InterfaceError", (Exception,), {})
_mock_psycopg2.OperationalError = type("OperationalError", (Exception,), {})
if "psycopg2" not in sys.modules:
    sys.modules["psycopg2"] = _mock_psycopg2
    _PSYCOPG2_INSERTED.append("psycopg2")
if "psycopg2.pool" not in sys.modules:
    sys.modules["psycopg2.pool"] = MagicMock()
    _PSYCOPG2_INSERTED.append("psycopg2.pool")


def teardown_module(module):
    for name in _STUBS_INSERTED + _PSYCOPG2_INSERTED:
        sys.modules.pop(name, None)


# ============================================================================
# TicketManager.move_ticket
# ============================================================================


class TestTicketManagerMove(unittest.TestCase):
    """Tests for the new ticket move pipeline."""

    def setUp(self):
        from app.control_plane.tickets import TicketManager
        self.tm = TicketManager()
        self.tm._audit = MagicMock()

    @patch("app.control_plane.projects.get_projects")
    @patch("app.control_plane.tickets.execute_one")
    def test_move_happy_path(self, mock_exec_one, mock_get_projects):
        """Known ticket + known target project → UPDATE + audit + row returned."""
        # ProjectManager.get_by_name resolves the target project.
        mock_pm = MagicMock()
        mock_pm.get_by_name.return_value = {"id": "proj-target", "name": "Eesti Mets"}
        mock_get_projects.return_value = mock_pm

        # Two execute_one calls: SELECT existing ticket, then UPDATE...RETURNING.
        mock_exec_one.side_effect = [
            # SELECT current row
            {"id": "tick-1", "project_id": "proj-source"},
            # UPDATE...RETURNING the new state
            {
                "id": "tick-1", "title": "Forest age", "status": "todo",
                "project_id": "proj-target", "priority": 5,
                "assigned_crew": None, "assigned_agent": None,
                "created_at": None, "updated_at": None,
            },
        ]

        result = self.tm.move_ticket("tick-1", "eesti mets")

        assert result is not None
        assert result["id"] == "tick-1"
        assert result["project_id"] == "proj-target"

        # The lookup is case-insensitive (delegated to get_by_name).
        mock_pm.get_by_name.assert_called_once_with("eesti mets")

        # Audit row was written with the right shape.
        self.tm._audit.log.assert_called_once()
        kwargs = self.tm._audit.log.call_args[1]
        assert kwargs["action"] == "ticket.moved"
        assert kwargs["resource_type"] == "ticket"
        assert kwargs["resource_id"] == "tick-1"
        assert kwargs["project_id"] == "proj-target"
        assert kwargs["detail"]["from_project_id"] == "proj-source"
        assert kwargs["detail"]["to_project_id"] == "proj-target"
        assert kwargs["detail"]["to_project_name"] == "Eesti Mets"

        # The UPDATE...RETURNING was the second call.
        update_sql = mock_exec_one.call_args_list[1][0][0]
        assert "UPDATE control_plane.tickets" in update_sql
        assert "SET project_id" in update_sql
        update_params = mock_exec_one.call_args_list[1][0][1]
        assert update_params == ("proj-target", "tick-1")

    @patch("app.control_plane.projects.get_projects")
    @patch("app.control_plane.tickets.execute_one")
    def test_move_unknown_ticket(self, mock_exec_one, mock_get_projects):
        """Target project resolves but ticket id is not found → None, no UPDATE, no audit."""
        mock_pm = MagicMock()
        mock_pm.get_by_name.return_value = {"id": "proj-target", "name": "PLG"}
        mock_get_projects.return_value = mock_pm

        # Only one execute_one call: SELECT returns None.
        mock_exec_one.return_value = None

        result = self.tm.move_ticket("nonexistent-uuid", "PLG")

        assert result is None
        # SELECT only — no UPDATE.
        assert mock_exec_one.call_count == 1
        select_sql = mock_exec_one.call_args[0][0]
        assert select_sql.startswith("SELECT") or "SELECT" in select_sql
        # No audit log.
        self.tm._audit.log.assert_not_called()

    @patch("app.control_plane.projects.get_projects")
    @patch("app.control_plane.tickets.execute_one")
    def test_move_unknown_target_project(self, mock_exec_one, mock_get_projects):
        """Target project doesn't exist → None, no SELECT/UPDATE, no audit."""
        mock_pm = MagicMock()
        mock_pm.get_by_name.return_value = None
        mock_get_projects.return_value = mock_pm

        result = self.tm.move_ticket("tick-1", "no-such-project")

        assert result is None
        # Bails out before touching the tickets table at all.
        mock_exec_one.assert_not_called()
        self.tm._audit.log.assert_not_called()

    @patch("app.control_plane.projects.get_projects")
    @patch("app.control_plane.tickets.execute_one")
    def test_move_writes_audit_entry_with_project_names(
        self, mock_exec_one, mock_get_projects,
    ):
        """The audit detail includes both from_project_id and the
        canonical target name (case-corrected from the input)."""
        mock_pm = MagicMock()
        mock_pm.get_by_name.return_value = {"id": "proj-b", "name": "PLG"}
        mock_get_projects.return_value = mock_pm

        mock_exec_one.side_effect = [
            {"id": "tick-x", "project_id": "proj-a"},
            {
                "id": "tick-x", "title": "Essay", "status": "in_progress",
                "project_id": "proj-b", "priority": 3,
                "assigned_crew": None, "assigned_agent": None,
                "created_at": None, "updated_at": None,
            },
        ]

        # Lower-case input — exercises case-insensitive name resolution.
        self.tm.move_ticket("tick-x", "plg")

        self.tm._audit.log.assert_called_once()
        detail = self.tm._audit.log.call_args[1]["detail"]
        assert detail["from_project_id"] == "proj-a"
        assert detail["to_project_id"] == "proj-b"
        # Canonical-cased name from the DB row, not the user-supplied "plg".
        assert detail["to_project_name"] == "PLG"

    @patch("app.control_plane.projects.get_projects")
    @patch("app.control_plane.tickets.execute_one")
    def test_move_idempotent_remove(self, mock_exec_one, mock_get_projects):
        """Re-moving a ticket to the project it already belongs to
        succeeds without raising and still emits an audit entry — the
        UPDATE writes the same value, but the operation is observable
        for traceability."""
        mock_pm = MagicMock()
        mock_pm.get_by_name.return_value = {"id": "proj-same", "name": "PLG"}
        mock_get_projects.return_value = mock_pm

        mock_exec_one.side_effect = [
            # Current project_id == target project_id.
            {"id": "tick-y", "project_id": "proj-same"},
            {
                "id": "tick-y", "title": "Already here", "status": "todo",
                "project_id": "proj-same", "priority": 5,
                "assigned_crew": None, "assigned_agent": None,
                "created_at": None, "updated_at": None,
            },
        ]

        result = self.tm.move_ticket("tick-y", "PLG")

        assert result is not None
        assert result["project_id"] == "proj-same"
        # Audit fires every time — re-move is observable.
        self.tm._audit.log.assert_called_once()
        detail = self.tm._audit.log.call_args[1]["detail"]
        assert detail["from_project_id"] == "proj-same"
        assert detail["to_project_id"] == "proj-same"


if __name__ == "__main__":
    unittest.main()
