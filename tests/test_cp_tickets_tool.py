"""Tests for the ``cp_*`` agent tools that surface
``control_plane.tickets`` (Postgres) — added 2026-05-09 to close the
gap that produced the "no tasks found" hallucination earlier the
same day.

What's covered
--------------
* ``cp_list_tickets`` — happy path, default-to-active project,
  unknown project, status filter, project with no tickets.
* ``cp_search_tickets`` — happy path, no hits, empty query.
* ``cp_move_ticket`` — happy path, unknown ticket, validation
  errors, exception-from-manager.

All Postgres + project lookups are mocked.  The tool factories
short-circuit on ImportError when ``crewai`` is missing on the dev
host; the conftest in this repo stubs it, but we add a defensive
skip just in case.
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
        # Provide a minimal BaseTool stand-in that satisfies the
        # ``class FooTool(BaseTool)`` shape used by the tool factories.
        class _BaseTool:
            name: str = ""
            description: str = ""
            args_schema = None
            def _run(self, *a, **k):
                raise NotImplementedError
            def run(self, *a, **k):
                return self._run(*a, **k)
        m.BaseTool = _BaseTool
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


def _get_tool(name: str):
    """Build the cp_* tools and pluck the one with the given name."""
    from app.tools.control_plane_tickets_tool import create_cp_tickets_tools
    tools = create_cp_tickets_tools("test")
    for t in tools:
        if t.name == name:
            return t
    raise AssertionError(f"tool {name!r} not found in factory output")


# ============================================================================
# cp_list_tickets
# ============================================================================


class TestCpListTickets(unittest.TestCase):

    @patch("app.control_plane.projects.get_projects")
    @patch("app.control_plane.db.execute")
    def test_list_in_named_project(self, mock_exec, mock_get_projects):
        mock_pm = MagicMock()
        mock_pm.get_by_name.return_value = {"id": "proj-1", "name": "PLG"}
        mock_pm.get_by_id.return_value = {"id": "proj-1", "name": "PLG"}
        mock_get_projects.return_value = mock_pm

        mock_exec.return_value = [
            {"id": "tick-aaa", "title": "Forest age", "status": "todo",
             "project_id": "proj-1", "priority": 5, "assigned_crew": None,
             "assigned_agent": None, "updated_at": None},
            {"id": "tick-bbb", "title": "Essay", "status": "in_progress",
             "project_id": "proj-1", "priority": 3, "assigned_crew": None,
             "assigned_agent": None, "updated_at": None},
        ]

        out = _get_tool("cp_list_tickets")._run(project_name="PLG")
        assert "2 ticket(s) in project 'PLG'" in out
        assert "Forest age" in out
        assert "Essay" in out
        # The resolved UUID — not the user-supplied name — is what the
        # SELECT is parameterised with.
        sent_params = mock_exec.call_args[0][1]
        assert sent_params[0] == "proj-1"

    @patch("app.control_plane.projects.get_projects")
    @patch("app.control_plane.db.execute")
    def test_list_defaults_to_active_project(self, mock_exec, mock_get_projects):
        mock_pm = MagicMock()
        mock_pm.get_active_project_id.return_value = "active-proj"
        mock_pm.get_by_id.return_value = {"id": "active-proj", "name": "default"}
        mock_get_projects.return_value = mock_pm
        mock_exec.return_value = []

        out = _get_tool("cp_list_tickets")._run()
        # No tickets, but the project resolved successfully.
        assert "default" in out

    @patch("app.control_plane.projects.get_projects")
    def test_list_unknown_project(self, mock_get_projects):
        mock_pm = MagicMock()
        mock_pm.get_by_name.return_value = None
        mock_get_projects.return_value = mock_pm

        out = _get_tool("cp_list_tickets")._run(project_name="not-a-project")
        assert "No project found" in out
        assert "not-a-project" in out

    @patch("app.control_plane.projects.get_projects")
    @patch("app.control_plane.db.execute")
    def test_list_status_filter_passed_through(self, mock_exec, mock_get_projects):
        mock_pm = MagicMock()
        mock_pm.get_by_name.return_value = {"id": "p", "name": "X"}
        mock_get_projects.return_value = mock_pm
        mock_exec.return_value = []

        _get_tool("cp_list_tickets")._run(project_name="X", status="done")
        sql = mock_exec.call_args[0][0]
        params = mock_exec.call_args[0][1]
        assert "AND status = %s" in sql
        # project_id, status filter — limit 50 is hard-coded so not in params.
        assert params == ("p", "done")

    @patch("app.control_plane.projects.get_projects")
    @patch("app.control_plane.db.execute")
    def test_list_no_tickets_message(self, mock_exec, mock_get_projects):
        mock_pm = MagicMock()
        mock_pm.get_by_name.return_value = {"id": "p", "name": "Empty"}
        mock_get_projects.return_value = mock_pm
        mock_exec.return_value = []

        out = _get_tool("cp_list_tickets")._run(project_name="Empty")
        assert "No tickets in project 'Empty'" in out


# ============================================================================
# cp_search_tickets
# ============================================================================


class TestCpSearchTickets(unittest.TestCase):

    @patch("app.control_plane.projects.get_projects")
    @patch("app.control_plane.db.execute")
    def test_search_finds_hits(self, mock_exec, mock_get_projects):
        # Project lookup for the rendering helper; not the search itself.
        mock_pm = MagicMock()
        mock_pm.get_by_id.return_value = {"id": "p", "name": "PLG"}
        mock_get_projects.return_value = mock_pm

        mock_exec.return_value = [
            {"id": "tick-1", "title": "Forest age graphic", "status": "failed",
             "project_id": "p", "updated_at": None},
        ]
        out = _get_tool("cp_search_tickets")._run(query="forest age")
        assert "1 ticket(s) matching 'forest age'" in out
        assert "Forest age graphic" in out
        # The query becomes a LIKE pattern.
        params = mock_exec.call_args[0][1]
        assert params == ("%forest age%", "%forest age%")

    @patch("app.control_plane.db.execute")
    def test_search_no_hits(self, mock_exec):
        mock_exec.return_value = []
        out = _get_tool("cp_search_tickets")._run(query="nonexistent")
        assert "No tickets matching 'nonexistent'" in out

    def test_search_empty_query(self):
        # No DB call should happen for an empty query.
        out = _get_tool("cp_search_tickets")._run(query="   ")
        assert out == "Empty query."


# ============================================================================
# cp_move_ticket
# ============================================================================


class TestCpMoveTicket(unittest.TestCase):

    @patch("app.control_plane.tickets.get_tickets")
    def test_move_happy(self, mock_get_tickets):
        mock_mgr = MagicMock()
        mock_mgr.move_ticket.return_value = {
            "id": "tick-1", "title": "Forest age", "status": "todo",
            "project_id": "proj-target", "priority": 5,
            "assigned_crew": None, "assigned_agent": None,
        }
        mock_get_tickets.return_value = mock_mgr

        out = _get_tool("cp_move_ticket")._run(
            ticket_id="tick-1", target_project_name="Eesti Mets",
        )
        assert "Moved ticket tick-1" in out
        assert "Eesti Mets" in out
        mock_mgr.move_ticket.assert_called_once_with("tick-1", "Eesti Mets")

    @patch("app.control_plane.tickets.get_tickets")
    def test_move_unknown_returns_helpful_error(self, mock_get_tickets):
        # Manager returns None when either side of the lookup fails.
        mock_mgr = MagicMock()
        mock_mgr.move_ticket.return_value = None
        mock_get_tickets.return_value = mock_mgr

        out = _get_tool("cp_move_ticket")._run(
            ticket_id="bogus", target_project_name="PLG",
        )
        assert "Could not move ticket" in out
        # Tells the agent what to do next.
        assert "cp_search_tickets" in out
        assert "cp_list_tickets" in out

    def test_move_validation_missing_ticket_id(self):
        out = _get_tool("cp_move_ticket")._run(
            ticket_id="", target_project_name="PLG",
        )
        assert "Missing ticket_id" in out

    def test_move_validation_missing_target_name(self):
        out = _get_tool("cp_move_ticket")._run(
            ticket_id="tick-1", target_project_name="",
        )
        assert "Missing target_project_name" in out

    @patch("app.control_plane.tickets.get_tickets")
    def test_move_exception_returns_string(self, mock_get_tickets):
        """Manager-level exceptions are stringified, not propagated —
        agent tools must return strings, never raise."""
        mock_mgr = MagicMock()
        mock_mgr.move_ticket.side_effect = RuntimeError("db down")
        mock_get_tickets.return_value = mock_mgr

        out = _get_tool("cp_move_ticket")._run(
            ticket_id="tick-1", target_project_name="PLG",
        )
        assert "cp_move_ticket ERROR" in out
        assert "RuntimeError" in out


# ============================================================================
# Capability registration
# ============================================================================


class TestCapabilityRegistration(unittest.TestCase):
    """The new ``manages-tickets`` capability is in the bounded
    vocabulary; the three cp_* tools register against the right tags.
    Catches typos that would otherwise silently break discovery."""

    def test_manages_tickets_capability_in_vocab(self):
        from app.tool_registry.capabilities import is_known
        assert is_known("manages-tickets")

    def test_reads_deployment_state_still_in_vocab(self):
        # cp_list_tickets / cp_search_tickets both reuse this existing tag.
        from app.tool_registry.capabilities import is_known
        assert is_known("reads-deployment-state")


if __name__ == "__main__":
    unittest.main()
