"""Tests for app/control_plane/ — organizational layer.

Covers: db, audit, tickets, budgets, governance, projects, org_chart,
heartbeats, cost_tracker, and dashboard_api wiring.

All PostgreSQL calls are mocked — no real database required.
"""
import json
import sys
import threading
import time
import types
import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, call

# ── Stub heavy dependencies that aren't on the dev host ─────────────────
# Capture which entries we insert so teardown_module below can clean up
# afterward — without that, the stubs bleed into every later test file
# (e.g. crewai/Agent imports start failing).
_STUBS_INSERTED: list[str] = []
_PSYCOPG2_INSERTED: list[str] = []

# Skip stubbing for modules that are installed in the venv. Those imports
# are real on this host and stubbing them creates phantom failures in
# later tests (test_capability_routing.py etc. need real `from crewai
# import Agent`).
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
        # Real module is importable — leave it alone.
        continue
    except Exception:
        pass
    m = types.ModuleType(_mod)
    if _mod == "crewai.tools":
        m.tool = lambda name: (lambda fn: fn)
    sys.modules[_mod] = m
    _STUBS_INSERTED.append(_mod)


# ── Mock psycopg2 at module level so db.py imports cleanly ──────────────
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
    """Remove any stubs we inserted so subsequent test files that need
    the real modules (crewai.Agent, real psycopg2.pool) don't see our
    placeholders."""
    for name in _STUBS_INSERTED + _PSYCOPG2_INSERTED:
        sys.modules.pop(name, None)


# ============================================================================
# Helpers
# ============================================================================

def _fake_execute(return_rows=None, scalar=None):
    """Build a mock execute() that returns canned rows."""
    def _exec(query, params=(), fetch=False):
        if fetch:
            return return_rows or []
        return []
    return _exec


def _fake_execute_one(row: dict | None = None):
    def _exec(query, params=()):
        return row
    return _exec


def _fake_execute_scalar(value=None):
    def _exec(query, params=()):
        return value
    return _exec


# ============================================================================
# AuditTrail
# ============================================================================

class TestAuditTrail(unittest.TestCase):
    """Tests for app.control_plane.audit."""

    def setUp(self):
        from app.control_plane.audit import AuditTrail
        self.audit = AuditTrail()

    # PR 3 (2026-05-16): AuditTrail.log now calls execute_required.
    @patch("app.control_plane.audit.execute_required")
    def test_log_inserts_row(self, mock_exec):
        self.audit.log(actor="user", action="test.action", project_id="p1",
                       resource_type="ticket", resource_id="t1",
                       detail={"key": "value"}, cost_usd=0.01, tokens=100)
        mock_exec.assert_called_once()
        args = mock_exec.call_args
        assert "INSERT INTO control_plane.audit_log" in args[0][0]
        params = args[0][1]
        assert params[0] == "p1"      # project_id
        assert params[1] == "user"    # actor
        assert params[2] == "test.action"

    @patch("app.control_plane.audit.execute_required")
    def test_log_never_raises(self, mock_exec):
        """Audit log is fire-and-forget — errors must not propagate."""
        mock_exec.side_effect = RuntimeError("db down")
        # Should not raise
        self.audit.log(actor="user", action="crash.test")

    @patch("app.control_plane.audit.execute")
    def test_query_filters(self, mock_exec):
        mock_exec.return_value = [{"id": 1, "actor": "user", "action": "x"}]
        rows = self.audit.query(project_id="p1", actor="user",
                                action_prefix="ticket.", limit=10)
        sql = mock_exec.call_args[0][0]
        assert "project_id = %s" in sql
        assert "actor = %s" in sql
        assert "action LIKE %s" in sql

    @patch("app.control_plane.audit.execute")
    def test_cost_summary_aggregates(self, mock_exec):
        mock_exec.return_value = [
            {"actor": "commander", "calls": 5, "total_cost": 1.5, "total_tokens": 10000},
            {"actor": "researcher", "calls": 3, "total_cost": 0.5, "total_tokens": 5000},
        ]
        result = self.audit.cost_summary(project_id="p1")
        assert result["total_cost"] == 2.0
        assert len(result["by_actor"]) == 2


# ============================================================================
# TicketManager
# ============================================================================

class TestTicketManager(unittest.TestCase):
    """Tests for app.control_plane.tickets."""

    def setUp(self):
        from app.control_plane.tickets import TicketManager
        self.tm = TicketManager()
        self.tm._audit = MagicMock()

    # PR 3 (2026-05-16): TicketManager INSERT/UPDATE writes converted
    # to execute_required / execute_one_required so silent DB failures
    # can no longer masquerade as empty result sets. Tests below patch
    # both the new ``*_required`` symbol AND the old ``execute_one``
    # used by the dedup-read in create_from_signal.
    @patch("app.control_plane.tickets.execute_one_required")
    @patch("app.control_plane.tickets.execute_one")
    def test_create_from_signal(self, mock_dedup, mock_insert):
        mock_dedup.return_value = None  # no existing duplicate
        mock_insert.return_value = {
            "id": "abc-123", "title": "Test task", "status": "todo",
            "created_at": "2026-04-16",
        }
        result = self.tm.create_from_signal(
            "Test task description", sender="user1",
            project_id="p1", difficulty=5, priority=7,
        )
        assert result["id"] == "abc-123"
        assert result["status"] == "todo"
        self.tm._audit.log.assert_called_once()
        audit_kwargs = self.tm._audit.log.call_args[1]
        assert audit_kwargs["action"] == "ticket.created"

    @patch("app.control_plane.tickets.execute_one_required")
    @patch("app.control_plane.tickets.execute_one")
    def test_create_from_signal_truncates_title(self, mock_dedup, mock_insert):
        """Title is truncated to 200 chars."""
        mock_dedup.return_value = None
        mock_insert.return_value = {"id": "x", "title": "t", "status": "todo", "created_at": ""}
        long_message = "A" * 500
        self.tm.create_from_signal(long_message, "u", "p1")
        # Either the dedup-read or the INSERT carries the truncated title;
        # the INSERT is the canonical write so we read its params.
        params = mock_insert.call_args[0][1]
        assert len(params[1]) <= 200  # title param

    @patch("app.control_plane.tickets.execute_one_required")
    def test_create_manual(self, mock_insert):
        mock_insert.return_value = {"id": "m1", "title": "Manual", "status": "todo", "created_at": ""}
        result = self.tm.create_manual("Manual ticket", "p1", priority=3)
        assert result["id"] == "m1"

    @patch("app.control_plane.tickets.execute_required")
    def test_assign_to_crew(self, mock_exec):
        self.tm.assign_to_crew("t1", "research_crew", "researcher")
        sql = mock_exec.call_args[0][0]
        assert "assigned_crew = %s" in sql
        assert "status = 'in_progress'" in sql
        self.tm._audit.log.assert_called_once()

    @patch("app.control_plane.tickets.execute_required")
    @patch("app.control_plane.tickets.execute_one")
    def test_complete(self, mock_pid_lookup, mock_exec):
        # complete() does a project_id lookup via execute_one (read,
        # not required) before the UPDATE via execute_required.
        mock_pid_lookup.return_value = {"project_id": "p1"}
        self.tm.complete("t1", "All done", cost_usd=0.05, tokens=500)
        sql = mock_exec.call_args[0][0]
        assert "status = 'done'" in sql
        self.tm._audit.log.assert_called_once()

    @patch("app.control_plane.tickets.execute_required")
    def test_fail(self, mock_exec):
        self.tm.fail("t1", "Something broke")
        sql = mock_exec.call_args[0][0]
        assert "status = 'failed'" in sql
        self.tm._audit.log.assert_called_once()

    @patch("app.control_plane.tickets.execute")
    def test_add_comment(self, mock_exec):
        self.tm.add_comment("t1", "researcher", "Found 3 sources", {"urls": 3})
        sql = mock_exec.call_args[0][0]
        assert "ticket_comments" in sql

    @patch("app.control_plane.tickets.execute")
    @patch("app.control_plane.tickets.execute_one")
    def test_get_with_comments(self, mock_one, mock_exec):
        mock_one.return_value = {"id": "t1", "title": "Test", "status": "done"}
        mock_exec.return_value = [
            {"id": "c1", "author": "user", "content": "looks good"},
        ]
        ticket = self.tm.get("t1")
        assert ticket["id"] == "t1"
        assert len(ticket["comments"]) == 1

    @patch("app.control_plane.tickets.execute_one")
    def test_get_nonexistent(self, mock_one):
        mock_one.return_value = None
        assert self.tm.get("no-such-id") is None

    @patch("app.control_plane.tickets.execute")
    def test_get_board_groups_by_status(self, mock_exec):
        mock_exec.return_value = [
            {"id": "1", "status": "todo", "title": "A", "priority": 5,
             "assigned_agent": None, "assigned_crew": None, "difficulty": 3,
             "cost_usd": 0, "tokens_used": 0, "created_at": "", "started_at": None,
             "completed_at": None},
            {"id": "2", "status": "done", "title": "B", "priority": 5,
             "assigned_agent": None, "assigned_crew": None, "difficulty": 3,
             "cost_usd": 0, "tokens_used": 0, "created_at": "", "started_at": None,
             "completed_at": None},
        ]
        board = self.tm.get_board("p1")
        assert board["counts"]["todo"] == 1
        assert board["counts"]["done"] == 1
        assert board["total"] == 2


# ============================================================================
# BudgetEnforcer
# ============================================================================

class TestBudgetEnforcer(unittest.TestCase):
    """Tests for app.control_plane.budgets."""

    def setUp(self):
        from app.control_plane.budgets import BudgetEnforcer
        self.be = BudgetEnforcer()
        self.be._audit = MagicMock()

    @patch("app.control_plane.budgets.execute_scalar")
    def test_check_and_record_allowed(self, mock_scalar):
        mock_scalar.return_value = True
        allowed, reason = self.be.check_and_record("p1", "researcher", 0.05, 500)
        assert allowed is True
        assert reason is None

    @patch("app.control_plane.budgets.execute_scalar")
    def test_check_and_record_exceeded(self, mock_scalar):
        mock_scalar.return_value = False
        allowed, reason = self.be.check_and_record("p1", "researcher", 0.05, 500)
        assert allowed is False
        assert "exceeded" in reason.lower()
        self.be._audit.log.assert_called_once()

    @patch("app.control_plane.budgets.execute_scalar")
    def test_check_and_record_fails_open(self, mock_scalar):
        """If budget system is down, allow the call (fail-open)."""
        mock_scalar.side_effect = RuntimeError("db down")
        allowed, reason = self.be.check_and_record("p1", "researcher", 0.05)
        assert allowed is True

    @patch("app.control_plane.budgets.execute")
    def test_set_budget_upserts(self, mock_exec):
        self.be.set_budget("p1", "coder", 100.0, limit_tokens=1000000)
        sql = mock_exec.call_args[0][0]
        assert "ON CONFLICT" in sql
        self.be._audit.log.assert_called_once()

    @patch("app.control_plane.budgets.execute")
    def test_override_budget(self, mock_exec):
        self.be.override_budget("p1", "coder", 200.0, approver="admin")
        sql = mock_exec.call_args[0][0]
        assert "is_paused = FALSE" in sql
        self.be._audit.log.assert_called_once()

    @patch("app.control_plane.budgets.execute")
    def test_get_status(self, mock_exec):
        mock_exec.return_value = [
            {"agent_role": "commander", "period": "2026-04", "limit_usd": 50,
             "spent_usd": 10, "limit_tokens": None, "spent_tokens": 0,
             "is_paused": False, "pct_used": 20},
        ]
        rows = self.be.get_status("p1")
        assert len(rows) == 1
        assert rows[0]["agent_role"] == "commander"

    @patch("app.control_plane.budgets.execute")
    @patch("app.control_plane.budgets.execute_scalar")
    def test_ensure_default_budgets(self, mock_scalar, mock_exec):
        """Creates budget rows for agents that don't have one."""
        mock_exec.return_value = [
            {"agent_role": "commander"},
            {"agent_role": "researcher"},
        ]
        mock_scalar.return_value = None  # no existing budget
        self.be.ensure_default_budgets("p1", default_limit=25.0)
        # Should have called execute for the SELECT + 2 INSERTs
        insert_calls = [c for c in mock_exec.call_args_list
                        if "INSERT INTO control_plane.budgets" in str(c)]
        assert len(insert_calls) == 2

    @patch("app.control_plane.budgets.execute")
    def test_format_status_empty(self, mock_exec):
        mock_exec.return_value = []
        text = self.be.format_status("p1")
        assert "No budgets" in text

    @patch("app.control_plane.budgets.execute")
    def test_format_status_with_data(self, mock_exec):
        mock_exec.return_value = [
            {"agent_role": "commander", "limit_usd": 50, "spent_usd": 25,
             "pct_used": 50, "is_paused": False},
        ]
        text = self.be.format_status("p1")
        assert "commander" in text
        assert "$25.00/$50.00" in text


# ============================================================================
# GovernanceGate
# ============================================================================

class TestGovernanceGate(unittest.TestCase):
    """Tests for app.control_plane.governance."""

    def setUp(self):
        from app.control_plane.governance import GovernanceGate, REQUIRES_APPROVAL
        self.gate = GovernanceGate()
        self.gate._audit = MagicMock()
        self.REQUIRES_APPROVAL = REQUIRES_APPROVAL

    def test_needs_approval_sensitive_ops(self):
        assert self.gate.needs_approval("evolution_deploy") is True
        assert self.gate.needs_approval("budget_override") is True
        assert self.gate.needs_approval("code_change") is True
        assert self.gate.needs_approval("agent_config") is True

    def test_needs_approval_autonomous_ops(self):
        assert self.gate.needs_approval("evolution_experiment") is False
        assert self.gate.needs_approval("skill_creation") is False
        assert self.gate.needs_approval("learning") is False
        assert self.gate.needs_approval("ticket execution") is False

    @patch("app.control_plane.governance.execute_one_required")
    def test_request_approval(self, mock_one):
        mock_one.return_value = {
            "id": "gov-1", "request_type": "code_change",
            "title": "Deploy v2", "status": "pending", "created_at": "",
        }
        result = self.gate.request_approval(
            "p1", "code_change", "auto_deployer", "Deploy v2",
            detail={"sha": "abc123"}, expires_hours=12,
        )
        assert result["id"] == "gov-1"
        assert result["status"] == "pending"
        self.gate._audit.log.assert_called_once()

    @patch("app.control_plane.governance.execute_one_required")
    def test_approve(self, mock_one):
        mock_one.return_value = {"id": "gov-1", "request_type": "code_change", "title": "Deploy"}
        assert self.gate.approve("gov-1", reviewer="admin") is True
        sql = mock_one.call_args[0][0]
        assert "status = 'approved'" in sql
        self.gate._audit.log.assert_called_once()

    @patch("app.control_plane.governance.execute_one_required")
    def test_approve_already_resolved(self, mock_one):
        mock_one.return_value = None  # no matching pending row
        assert self.gate.approve("gov-1") is False

    @patch("app.control_plane.governance.execute_one_required")
    def test_reject(self, mock_one):
        mock_one.return_value = {"id": "gov-1", "request_type": "code_change", "title": "Deploy"}
        assert self.gate.reject("gov-1", reason="not ready") is True
        sql = mock_one.call_args[0][0]
        assert "status = 'rejected'" in sql

    @patch("app.control_plane.governance.execute")
    def test_expire_old(self, mock_exec):
        mock_exec.return_value = [{"id": "g1"}, {"id": "g2"}]
        count = self.gate.expire_old()
        assert count == 2
        sql = mock_exec.call_args[0][0]
        assert "status = 'expired'" in sql
        assert "expires_at < NOW()" in sql

    @patch("app.control_plane.governance.execute")
    def test_get_pending(self, mock_exec):
        mock_exec.return_value = [{"id": "g1", "status": "pending"}]
        pending = self.gate.get_pending("p1")
        assert len(pending) == 1

    @patch("app.control_plane.governance.execute_scalar")
    def test_pending_count(self, mock_scalar):
        mock_scalar.return_value = 3
        assert self.gate.pending_count() == 3

    @patch("app.control_plane.governance.execute")
    def test_format_pending_empty(self, mock_exec):
        mock_exec.return_value = []
        text = self.gate.format_pending()
        assert "No pending" in text

    @patch("app.control_plane.governance.execute")
    def test_format_pending_with_items(self, mock_exec):
        mock_exec.return_value = [
            {"id": "abcd1234-5678", "request_type": "code_change",
             "title": "Deploy new model", "requested_by": "auto_deployer",
             "created_at": "2026-04-16"},
        ]
        text = self.gate.format_pending()
        assert "Pending Approvals" in text
        assert "code_change" in text


# ============================================================================
# ProjectManager
# ============================================================================

class TestProjectManager(unittest.TestCase):
    """Tests for app.control_plane.projects."""

    def setUp(self):
        from app.control_plane.projects import ProjectManager
        self.pm = ProjectManager()
        # Reset class-level active project
        ProjectManager._active_project_id = None

    @patch("app.control_plane.projects.execute_one")
    def test_create(self, mock_one):
        mock_one.return_value = {
            "id": "p-new", "name": "TestProject", "mission": "Test",
            "is_active": True, "created_at": "",
        }
        with patch("app.control_plane.audit.get_audit") as mock_audit:
            mock_audit.return_value = MagicMock()
            result = self.pm.create("TestProject", mission="Test")
        assert result["name"] == "TestProject"

    @patch("app.control_plane.projects.execute")
    def test_list_all(self, mock_exec):
        mock_exec.return_value = [
            {"id": "p1", "name": "default", "description": "", "mission": "",
             "is_active": True, "created_at": ""},
        ]
        projects = self.pm.list_all()
        assert len(projects) == 1

    @patch("app.control_plane.projects.execute_one")
    def test_get_by_name(self, mock_one):
        mock_one.return_value = {"id": "p1", "name": "default"}
        assert self.pm.get_by_name("default")["id"] == "p1"

    @patch("app.control_plane.projects.execute_scalar")
    def test_get_default_project_id(self, mock_scalar):
        mock_scalar.return_value = "p-default"
        assert self.pm.get_default_project_id() == "p-default"

    @patch("app.control_plane.projects.execute_scalar")
    def test_get_active_project_id_fallback(self, mock_scalar):
        """Falls back to default when no project explicitly activated."""
        mock_scalar.return_value = "p-default"
        assert self.pm.get_active_project_id() == "p-default"

    @patch("app.control_plane.projects.execute_one")
    def test_switch(self, mock_one):
        mock_one.return_value = {"id": "p-plg", "name": "PLG"}
        with patch("app.project_isolation.get_manager") as mock_iso:
            mock_iso.return_value = MagicMock()
            result = self.pm.switch("PLG")
        assert result["id"] == "p-plg"
        assert self.pm._active_project_id == "p-plg"

    @patch("app.control_plane.projects.execute_one")
    def test_switch_nonexistent(self, mock_one):
        mock_one.return_value = None
        assert self.pm.switch("NoSuchProject") is None

    @patch("app.control_plane.projects.execute")
    @patch("app.control_plane.projects.execute_scalar")
    def test_format_list(self, mock_scalar, mock_exec):
        mock_scalar.return_value = "p1"
        mock_exec.return_value = [
            {"id": "p1", "name": "default", "mission": "General ops"},
            {"id": "p2", "name": "PLG", "mission": "Ticketing platform"},
        ]
        text = self.pm.format_list()
        assert "default" in text
        assert "PLG" in text
        assert "active" in text  # default is active


# ============================================================================
# OrgChart
# ============================================================================

class TestOrgChart(unittest.TestCase):
    """Tests for app.control_plane.org_chart."""

    @patch("app.control_plane.org_chart.execute")
    def test_get_org_chart(self, mock_exec):
        mock_exec.return_value = [
            {"agent_role": "commander", "display_name": "Commander",
             "reports_to": None, "job_description": "Routes requests",
             "soul_file": "souls/commander.md", "default_model": None, "sort_order": 0},
        ]
        from app.control_plane.org_chart import get_org_chart
        chart = get_org_chart()
        assert len(chart) == 1
        assert chart[0]["agent_role"] == "commander"

    @patch("app.control_plane.org_chart.execute")
    def test_get_reports(self, mock_exec):
        mock_exec.return_value = [
            {"agent_role": "researcher", "display_name": "Researcher",
             "job_description": "Web research"},
        ]
        from app.control_plane.org_chart import get_reports
        reports = get_reports("commander")
        assert len(reports) == 1

    @patch("app.control_plane.org_chart.execute")
    def test_format_org_chart_empty(self, mock_exec):
        mock_exec.return_value = []
        from app.control_plane.org_chart import format_org_chart
        assert "No agents" in format_org_chart()

    @patch("app.control_plane.org_chart.execute")
    def test_format_org_chart_hierarchy(self, mock_exec):
        mock_exec.return_value = [
            {"agent_role": "commander", "display_name": "Commander",
             "reports_to": None, "job_description": "Routes"},
            {"agent_role": "researcher", "display_name": "Researcher",
             "reports_to": "commander", "job_description": "Researches"},
        ]
        from app.control_plane.org_chart import format_org_chart
        text = format_org_chart()
        assert "(CEO)" in text
        assert "reports to commander" in text


# ============================================================================
# CostTracker
# ============================================================================

class TestCostTracker(unittest.TestCase):
    """Tests for app.control_plane.cost_tracker."""

    def test_estimate_tokens(self):
        from app.control_plane.cost_tracker import estimate_tokens
        inp, out = estimate_tokens("Hello world!", max_output_tokens=512)
        assert inp >= 1
        assert out == 512

    def test_estimate_cost_local_model_free(self):
        from app.control_plane.cost_tracker import estimate_cost
        with patch("app.llm_catalog.get_model", return_value={"tier": "local", "cost_input_per_m": 0, "cost_output_per_m": 0}):
            cost = estimate_cost("ollama/llama3", input_tokens=1000, output_tokens=500)
        assert cost == 0.0

    def test_estimate_cost_unknown_model(self):
        from app.control_plane.cost_tracker import estimate_cost
        with patch("app.llm_catalog.get_model", return_value=None):
            cost = estimate_cost("unknown-model", input_tokens=1000, output_tokens=1000)
        assert cost > 0  # conservative estimate

    def test_estimate_cost_from_prompt(self):
        from app.control_plane.cost_tracker import estimate_cost
        with patch("app.llm_catalog.get_model", return_value={"tier": "cloud", "cost_input_per_m": 3.0, "cost_output_per_m": 15.0}):
            cost = estimate_cost("claude-sonnet", prompt="a" * 4000, output_tokens=1024)
        assert cost > 0


# ============================================================================
# HeartbeatScheduler
# ============================================================================

class TestHeartbeatScheduler(unittest.TestCase):
    """Tests for app.control_plane.heartbeats."""

    def setUp(self):
        from app.control_plane.heartbeats import HeartbeatScheduler
        self.hb = HeartbeatScheduler()

    def test_default_intervals(self):
        assert self.hb._intervals["commander"] == 300
        assert self.hb._intervals["self_improver"] == 1800

    def test_configure(self):
        self.hb.configure("commander", 120)
        assert self.hb._intervals["commander"] == 120

    def test_should_beat_first_time(self):
        """First beat should always be allowed (no last_beat recorded)."""
        assert self.hb.should_beat("commander") is True

    def test_should_beat_respects_interval(self):
        self.hb._last_beat["commander"] = time.monotonic()
        assert self.hb.should_beat("commander") is False

    def test_trigger_wake(self):
        self.hb.trigger_wake("researcher", "ticket_assigned", ticket_id="t1")
        wakes = self.hb.get_pending_wakes("researcher")
        assert len(wakes) == 1
        assert wakes[0]["reason"] == "ticket_assigned"
        assert wakes[0]["ticket_id"] == "t1"

    def test_get_pending_wakes_clears(self):
        self.hb.trigger_wake("coder", "mention")
        self.hb.get_pending_wakes("coder")
        # Second call should return empty
        assert self.hb.get_pending_wakes("coder") == []

    def test_get_pending_wakes_agent_isolation(self):
        """Wakes for one agent don't appear for another."""
        self.hb.trigger_wake("coder", "mention")
        assert self.hb.get_pending_wakes("researcher") == []
        assert len(self.hb.get_pending_wakes("coder")) == 1

    @patch("app.control_plane.heartbeats.execute")
    def test_record_beat(self, mock_exec):
        self.hb.record_beat("commander", "p1", "scheduled", tickets_processed=2)
        sql = mock_exec.call_args[0][0]
        assert "INSERT INTO control_plane.heartbeats" in sql

    @patch("app.control_plane.heartbeats.execute")
    def test_record_beat_db_failure_silent(self, mock_exec):
        """DB failure during beat logging should not raise."""
        mock_exec.side_effect = RuntimeError("db down")
        self.hb.record_beat("commander")  # should not raise

    @patch("app.control_plane.heartbeats.execute")
    def test_run_active_heartbeat_idle(self, mock_exec):
        """No tickets and no wakes → idle heartbeat (active path)."""
        mock_exec.return_value = []
        result = self.hb.run_active_heartbeat("researcher", "p1")
        assert result["status"] == "idle"
        assert result["tickets_processed"] == 0

    @patch("app.control_plane.heartbeats.execute")
    def test_run_active_heartbeat_with_todo_ticket(self, mock_exec):
        """A todo ticket should be picked up and processed."""
        mock_exec.return_value = [
            {"id": "t1", "title": "Research AI safety", "status": "todo", "difficulty": 4},
        ]
        mock_tm = MagicMock()
        mock_be = MagicMock()
        mock_be.check_and_record.return_value = (True, None)
        mock_commander_cls = MagicMock()
        mock_commander_cls.return_value._run_crew.return_value = "Research complete"

        with patch("app.control_plane.tickets.get_tickets", return_value=mock_tm), \
             patch("app.control_plane.budgets.get_budget_enforcer", return_value=mock_be), \
             patch.dict("sys.modules", {"app.agents.commander.orchestrator": MagicMock(Commander=mock_commander_cls)}):
            result = self.hb.run_active_heartbeat("researcher", "p1")

        assert result["status"] == "active"
        assert result["tickets_processed"] == 1
        mock_tm.assign_to_crew.assert_called_once()
        mock_tm.complete.assert_called_once()

    @patch("app.control_plane.heartbeats.execute")
    def test_run_active_heartbeat_budget_exceeded(self, mock_exec):
        """Should stop processing when budget is exceeded."""
        mock_exec.return_value = [
            {"id": "t1", "title": "Task", "status": "todo", "difficulty": 3},
        ]
        mock_be = MagicMock()
        mock_be.check_and_record.return_value = (False, "Budget exceeded")

        with patch("app.control_plane.budgets.get_budget_enforcer", return_value=mock_be), \
             patch("app.control_plane.tickets.get_tickets", return_value=MagicMock()):
            result = self.hb.run_active_heartbeat("coder", "p1")

        assert result["status"] == "budget_exceeded"

    @patch("app.control_plane.heartbeats.execute")
    def test_run_active_heartbeat_ticket_failure(self, mock_exec):
        """Failed ticket execution marks ticket as failed."""
        mock_exec.return_value = [
            {"id": "t1", "title": "Hard task", "status": "todo", "difficulty": 8},
        ]
        mock_tm = MagicMock()
        mock_be = MagicMock()
        mock_be.check_and_record.return_value = (True, None)
        mock_commander_cls = MagicMock()
        mock_commander_cls.return_value._run_crew.side_effect = RuntimeError("crew crashed")

        with patch("app.control_plane.tickets.get_tickets", return_value=mock_tm), \
             patch("app.control_plane.budgets.get_budget_enforcer", return_value=mock_be), \
             patch.dict("sys.modules", {"app.agents.commander.orchestrator": MagicMock(Commander=mock_commander_cls)}):
            result = self.hb.run_active_heartbeat("coder", "p1")

        mock_tm.fail.assert_called_once()
        assert "crew crashed" in mock_tm.fail.call_args[0][1]

    def test_get_schedule(self):
        schedule = self.hb.get_schedule()
        roles = [s["agent_role"] for s in schedule]
        assert "commander" in roles
        assert "self_improver" in roles
        for s in schedule:
            assert "interval_seconds" in s
            assert "next_in_seconds" in s

    def test_thread_safety(self):
        """Concurrent trigger_wake calls should not lose events."""
        errors = []

        def _trigger(n):
            try:
                for i in range(50):
                    self.hb.trigger_wake("commander", f"test-{n}-{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_trigger, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        wakes = self.hb.get_pending_wakes("commander")
        assert len(wakes) == 200  # 4 threads * 50 events


# ============================================================================
# Singleton factories
# ============================================================================

class TestSingletons(unittest.TestCase):
    """Verify singleton getters return the same instance."""

    def test_get_audit_singleton(self):
        from app.control_plane.audit import get_audit
        a1 = get_audit()
        a2 = get_audit()
        assert a1 is a2

    def test_get_tickets_singleton(self):
        from app.control_plane.tickets import get_tickets
        t1 = get_tickets()
        t2 = get_tickets()
        assert t1 is t2

    def test_get_budget_enforcer_singleton(self):
        from app.control_plane.budgets import get_budget_enforcer
        b1 = get_budget_enforcer()
        b2 = get_budget_enforcer()
        assert b1 is b2

    def test_get_governance_singleton(self):
        from app.control_plane.governance import get_governance
        g1 = get_governance()
        g2 = get_governance()
        assert g1 is g2

    def test_get_projects_singleton(self):
        from app.control_plane.projects import get_projects
        p1 = get_projects()
        p2 = get_projects()
        assert p1 is p2

    def test_get_heartbeat_scheduler_singleton(self):
        from app.control_plane.heartbeats import get_heartbeat_scheduler
        h1 = get_heartbeat_scheduler()
        h2 = get_heartbeat_scheduler()
        assert h1 is h2


# ============================================================================
# __init__.py exports
# ============================================================================

class TestPackageExports(unittest.TestCase):
    """Verify __init__.py exposes all public symbols."""

    def test_all_exports(self):
        import app.control_plane as cp
        expected = [
            "get_pool", "AuditTrail", "get_audit",
            "TicketManager", "get_tickets",
            "BudgetEnforcer", "get_budget_enforcer",
            "ProjectManager", "get_projects",
            "GovernanceGate", "get_governance",
            "get_org_chart", "estimate_cost",
            "HeartbeatScheduler", "get_heartbeat_scheduler",
        ]
        for name in expected:
            assert hasattr(cp, name), f"Missing export: {name}"
            assert name in cp.__all__, f"Not in __all__: {name}"


# ============================================================================
# Dashboard API routes
# ============================================================================

class TestDashboardAPI(unittest.TestCase):
    """Tests for app.control_plane.dashboard_api route definitions."""

    @classmethod
    def setUpClass(cls):
        """Ensure FastAPI is importable (it's a real dependency)."""
        try:
            import fastapi  # noqa
            cls._fastapi_available = True
        except ImportError:
            cls._fastapi_available = False

    def setUp(self):
        if not self._fastapi_available:
            self.skipTest("fastapi not installed")

    def test_router_exists_with_prefix(self):
        from app.control_plane.dashboard_api import router
        assert router.prefix == "/api/cp"

    def test_route_paths(self):
        """Verify all expected API routes are registered."""
        from app.control_plane.dashboard_api import router
        paths = [r.path for r in router.routes]
        expected_paths = [
            "/projects", "/projects/{project_id}",
            "/projects/{project_id}/status",
            "/tickets", "/tickets/board", "/tickets/{ticket_id}",
            "/tickets/{ticket_id}/comments",
            "/budgets", "/budgets/override",
            "/audit", "/audit/costs",
            "/governance/pending",
            "/governance/{request_id}/approve",
            "/governance/{request_id}/reject",
            "/org-chart",
            "/health",
            "/costs/by-agent", "/costs/daily",
        ]
        for ep in expected_paths:
            assert ep in paths, f"Missing route: {ep}"

    def test_request_models(self):
        from app.control_plane.dashboard_api import (
            ProjectCreate, TicketCreate, TicketUpdate,
            CommentCreate, BudgetOverride, BudgetSet,
        )
        # Verify pydantic models instantiate correctly
        pc = ProjectCreate(name="test", mission="m")
        assert pc.name == "test"

        tc = TicketCreate(title="bug fix")
        assert tc.priority == 5  # default

        tu = TicketUpdate(status="done")
        assert tu.status == "done"

        cc = CommentCreate(content="looks good")
        assert cc.author == "user"  # default

        bo = BudgetOverride(project_id="p1", agent_role="coder", new_limit=100.0)
        assert bo.approver == "user"  # default

        bs = BudgetSet(project_id="p1", agent_role="coder", limit_usd=50.0)
        assert bs.limit_tokens is None  # default


# ============================================================================
# DB connection pool
# ============================================================================

class TestDBPool(unittest.TestCase):
    """Tests for app.control_plane.db — connection pool and query helpers."""

    def test_get_pool_no_config(self):
        """Returns None when no postgres URL is configured."""
        from app.control_plane import db
        old_pool = db._pool
        db._pool = None  # reset
        try:
            mock_settings = MagicMock()
            mock_settings.return_value = MagicMock(mem0_postgres_url="")
            with patch.dict("sys.modules", {"app.config": MagicMock(get_settings=mock_settings)}):
                pool = db.get_pool()
            assert pool is None
        finally:
            db._pool = old_pool  # restore

    def test_execute_no_pool_returns_none(self):
        from app.control_plane import db
        with patch.object(db, "get_pool", return_value=None):
            result = db.execute("SELECT 1", fetch=True)
        assert result is None

    def test_execute_one_no_rows(self):
        from app.control_plane import db
        with patch.object(db, "execute", return_value=[]):
            result = db.execute_one("SELECT 1")
        assert result is None

    def test_execute_one_returns_first(self):
        from app.control_plane import db
        with patch.object(db, "execute", return_value=[{"id": 1}, {"id": 2}]):
            result = db.execute_one("SELECT 1")
        assert result == {"id": 1}

    def test_execute_scalar_no_pool(self):
        from app.control_plane import db
        with patch.object(db, "get_pool", return_value=None):
            result = db.execute_scalar("SELECT 1")
        assert result is None


# ============================================================================
# Integration: ticket lifecycle
# ============================================================================

class TestTicketLifecycle(unittest.TestCase):
    """End-to-end ticket lifecycle: create → assign → comment → complete."""

    def setUp(self):
        from app.control_plane.tickets import TicketManager
        self.tm = TicketManager()
        self.tm._audit = MagicMock()
        self._exec_calls = []

    # PR 3 (2026-05-16): both lifecycle tests now patch the four
    # symbols that the converted code actually uses:
    #   * execute_one         — dedup-read in create_from_signal and
    #                            project_id lookup in complete (optional)
    #   * execute_one_required — INSERT in create_from_signal (required)
    #   * execute_required     — UPDATE in assign_to_crew / complete /
    #                            fail / close (required)
    #   * execute              — INSERT in add_comment (still optional;
    #                            comments are observational)
    @patch("app.control_plane.tickets.execute")
    @patch("app.control_plane.tickets.execute_required")
    @patch("app.control_plane.tickets.execute_one_required")
    @patch("app.control_plane.tickets.execute_one")
    def test_full_lifecycle(self, mock_one, mock_one_req, mock_req, mock_exec):
        mock_one.return_value = None  # no dedup match
        mock_one_req.return_value = {
            "id": "t-lifecycle", "title": "Integration test",
            "status": "todo", "created_at": "2026-04-16",
        }

        # Create
        ticket = self.tm.create_from_signal("Integration test", "user", "p1")
        assert ticket["status"] == "todo"

        # Assign
        self.tm.assign_to_crew("t-lifecycle", "research_crew", "researcher")
        assign_sql = mock_req.call_args[0][0]
        assert "in_progress" in assign_sql

        # Comment
        self.tm.add_comment("t-lifecycle", "researcher", "Found 5 sources")

        # Complete — overrides project_id lookup to None to skip audit project_id
        mock_one.return_value = None
        self.tm.complete("t-lifecycle", "Done: 5 sources analyzed", cost_usd=0.03, tokens=2000)
        complete_sql = mock_req.call_args[0][0]
        assert "done" in complete_sql

        # Audit trail should have 3 entries: create, assign, complete
        assert self.tm._audit.log.call_count == 3

    @patch("app.control_plane.tickets.execute_required")
    @patch("app.control_plane.tickets.execute_one_required")
    @patch("app.control_plane.tickets.execute_one")
    def test_failure_lifecycle(self, mock_one, mock_one_req, mock_req):
        mock_one.return_value = None  # no dedup match
        mock_one_req.return_value = {
            "id": "t-fail", "title": "Failing task",
            "status": "todo", "created_at": "",
        }

        ticket = self.tm.create_from_signal("Failing task", "user", "p1")
        self.tm.assign_to_crew("t-fail", "coding_crew", "coder")
        self.tm.fail("t-fail", "LLM timeout after 3 retries")

        fail_sql = mock_req.call_args[0][0]
        assert "failed" in fail_sql
        # Audit: create + assign + fail = 3
        assert self.tm._audit.log.call_count == 3


if __name__ == "__main__":
    unittest.main()
