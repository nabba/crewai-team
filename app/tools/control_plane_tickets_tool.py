"""control_plane_tickets_tool.py — agent-callable interface to the
PostgreSQL ``control_plane.tickets`` table (the React Kanban board /
Signal-message tickets system).

Why this exists
---------------
There are TWO ticket systems in the codebase:

  1. ``/app/workspace/tasks.db`` (SQLite) — the PIM-style local-tasks
     system.  Surfaced via ``app/tools/task_tools.py``
     (``create_task``, ``list_tasks``, ``complete_task``, ...).  Has
     no ``project_id`` column.

  2. ``control_plane.tickets`` (PostgreSQL) — the React Kanban /
     Signal-message tickets.  Has ``project_id`` and is what the
     dashboard renders.  Pre-this-tool, NO agent had a tool for it.

The 2026-05-09 incident — user asked the agent to move a ticket from
PLG to a different workspace; agent searched the SQLite tasks.db,
found nothing, and confidently reported "the database returned no
tasks" — was caused by exactly that gap.  This module closes it.

Tool surface
------------
* ``cp_list_tickets(project_name="", status="")`` — list tickets in
  the named project (default: currently active project).
* ``cp_search_tickets(query)`` — title/description LIKE search,
  capped at 20 results, returns id + title + project name + status.
* ``cp_move_ticket(ticket_id, target_project_name)`` — move a ticket
  to a different project.  Mutating; audit-logged as
  ``ticket.moved``.

All three hit ``control_plane.tickets`` directly.  They are not a
replacement for the SQLite tools — they coexist.  An agent should
pick by what the user is talking about: ticket on the Kanban board
or a personal todo from the PIM agent.
"""
from __future__ import annotations

import logging
from typing import Type

logger = logging.getLogger(__name__)


# ── Helpers ─────────────────────────────────────────────────────────

def _resolve_project_id(project_name: str) -> tuple[str | None, str | None]:
    """Resolve a (possibly empty) project name to ``(project_id, canonical_name)``.

    Empty / whitespace-only ``project_name`` → use the currently
    active project.  Returns ``(None, None)`` when the project can't
    be found, which the callers translate to an LLM-friendly error
    message.
    """
    from app.control_plane.projects import get_projects
    pm = get_projects()
    name = (project_name or "").strip()
    if not name:
        active_id = pm.get_active_project_id()
        if not active_id:
            return None, None
        row = pm.get_by_id(active_id)
        if not row:
            return None, None
        return str(row["id"]), row.get("name") or ""
    row = pm.get_by_name(name)
    if not row:
        return None, None
    return str(row["id"]), row.get("name") or name


def _project_name_for(project_id: str) -> str:
    """Best-effort lookup of project name by id; returns id on miss."""
    if not project_id:
        return ""
    try:
        from app.control_plane.projects import get_projects
        row = get_projects().get_by_id(project_id)
        if row and row.get("name"):
            return str(row["name"])
    except Exception:
        pass
    return str(project_id)


def _format_ticket_line(row: dict) -> str:
    """Render a ticket row as a single readable line."""
    tid = str(row.get("id", "?"))
    title = str(row.get("title", "")).strip() or "(no title)"
    status = str(row.get("status", "?"))
    pid = row.get("project_id")
    pname = _project_name_for(str(pid)) if pid else "—"
    return f"  #{tid[:8]}  [{status}]  ({pname})  {title[:120]}"


# ── CrewAI tool factory ─────────────────────────────────────────────

def create_cp_tickets_tools(agent_id: str = "default") -> list:
    """Create control-plane ticket tools.  Returns ``[]`` if crewai is
    not installed (mirrors the pattern in every other ``create_*_tools``
    factory in this directory)."""
    try:
        from crewai.tools import BaseTool
        from pydantic import BaseModel, Field
    except ImportError:
        logger.debug("control_plane_tickets_tool: crewai not installed")
        return []

    class _ListInput(BaseModel):
        project_name: str = Field(
            default="",
            description=(
                "Project name (case-insensitive).  Leave empty to list "
                "tickets in the currently active project."
            ),
        )
        status: str = Field(
            default="",
            description=(
                "Filter by status: todo, in_progress, review, done, "
                "failed, blocked.  Empty = all statuses."
            ),
        )

    class CpListTicketsTool(BaseTool):
        name: str = "cp_list_tickets"
        description: str = (
            "List Kanban tickets from control_plane.tickets (Postgres) — "
            "this is the React dashboard's ticket system, NOT the SQLite "
            "tasks.db that create_task / list_tasks use.  USE THIS when "
            "the user mentions tickets, the Kanban board, or moving "
            "items between workspaces / projects.\n\n"
            "Returns up to 50 tickets ordered by most recently updated. "
            "Each line shows the short ticket id, status, project name, "
            "and title."
        )
        args_schema: Type[BaseModel] = _ListInput

        def _run(self, project_name: str = "", status: str = "") -> str:
            from app.control_plane.db import execute
            project_id, canonical = _resolve_project_id(project_name)
            if project_id is None:
                want = (project_name or "").strip() or "(active project)"
                return f"No project found matching {want!r}."
            params: list = [project_id]
            sql = (
                "SELECT id, title, status, project_id, priority, "
                "assigned_crew, assigned_agent, updated_at "
                "FROM control_plane.tickets WHERE project_id = %s"
            )
            if status:
                sql += " AND status = %s"
                params.append(status)
            sql += " ORDER BY updated_at DESC LIMIT 50"
            rows = execute(sql, tuple(params), fetch=True) or []
            if not rows:
                return f"No tickets in project {canonical!r}."
            header = (
                f"{len(rows)} ticket(s) in project {canonical!r}"
                + (f" with status={status!r}" if status else "")
                + ":"
            )
            return header + "\n" + "\n".join(_format_ticket_line(r) for r in rows)

    class _SearchInput(BaseModel):
        query: str = Field(
            description=(
                "Substring to match against ticket title or description "
                "(case-insensitive)."
            ),
        )

    class CpSearchTicketsTool(BaseTool):
        name: str = "cp_search_tickets"
        description: str = (
            "Search Kanban tickets in control_plane.tickets (Postgres) "
            "by title or description LIKE %query%.  Returns at most 20 "
            "results with short id, title, status, and project name. "
            "USE THIS when the user references a ticket by content "
            "(\"the forest age task\", \"that essay ticket\") and you "
            "need to find its id before moving / inspecting it.\n\n"
            "Searches across ALL projects — pair with cp_list_tickets "
            "if you need a project-scoped view."
        )
        args_schema: Type[BaseModel] = _SearchInput

        def _run(self, query: str) -> str:
            from app.control_plane.db import execute
            q = (query or "").strip()
            if not q:
                return "Empty query."
            like = f"%{q}%"
            rows = execute(
                """SELECT id, title, status, project_id, updated_at
                     FROM control_plane.tickets
                    WHERE title ILIKE %s OR description ILIKE %s
                    ORDER BY updated_at DESC
                    LIMIT 20""",
                (like, like),
                fetch=True,
            ) or []
            if not rows:
                return f"No tickets matching {q!r}."
            header = f"{len(rows)} ticket(s) matching {q!r}:"
            return header + "\n" + "\n".join(_format_ticket_line(r) for r in rows)

    class _MoveInput(BaseModel):
        ticket_id: str = Field(
            description=(
                "Full UUID of the ticket to move.  Get this from "
                "cp_list_tickets or cp_search_tickets."
            ),
        )
        target_project_name: str = Field(
            description=(
                "Name of the project to move the ticket into "
                "(case-insensitive)."
            ),
        )

    class CpMoveTicketTool(BaseTool):
        name: str = "cp_move_ticket"
        description: str = (
            "Move a Kanban ticket to a different project.  Mutates "
            "control_plane.tickets.project_id and audit-logs the move "
            "as ticket.moved.\n\n"
            "USE THIS when the user asks to move a task / ticket to a "
            "different workspace / project (e.g. \"move the forest "
            "age task out of PLG\").  Look up the ticket id first with "
            "cp_search_tickets, then call this with the full UUID and "
            "the target project name.\n\n"
            "Returns the updated ticket on success, or an error string "
            "if the ticket id or project name was not found."
        )
        args_schema: Type[BaseModel] = _MoveInput

        def _run(self, ticket_id: str, target_project_name: str) -> str:
            from app.control_plane.tickets import get_tickets
            tid = (ticket_id or "").strip()
            tname = (target_project_name or "").strip()
            if not tid:
                return "Missing ticket_id."
            if not tname:
                return "Missing target_project_name."
            try:
                row = get_tickets().move_ticket(tid, tname)
            except Exception as exc:  # noqa: BLE001
                return f"cp_move_ticket ERROR: {type(exc).__name__}: {exc}"
            if not row:
                return (
                    f"Could not move ticket {tid!r} to {tname!r}: "
                    "either the ticket id or the target project name "
                    "was not found.  Use cp_search_tickets to look up "
                    "the ticket id, and cp_list_tickets to confirm the "
                    "project name."
                )
            return (
                f"Moved ticket {tid} to project {tname!r}.\n"
                + _format_ticket_line(row)
            )

    return [
        CpListTicketsTool(),
        CpSearchTicketsTool(),
        CpMoveTicketTool(),
    ]


# ── Tool registry annotations ───────────────────────────────────────
# Mirrors the pattern in currency_tools.py / system_state_tool.py:
# the @register_tool factories sit at module bottom, are passive on
# import (no side effects on factory state), and reuse the same
# create_cp_tickets_tools() factory.  Wrapping in try/ImportError so
# legacy paths that haven't loaded the registry still import cleanly.
try:
    from app.tool_registry import Lifecycle, Tier, register_tool

    @register_tool(
        name="cp_list_tickets",
        capabilities=["reads-deployment-state"],
        description=(
            "List Kanban tickets from control_plane.tickets (Postgres). "
            "Distinct from list_tasks (SQLite tasks.db).  Use when the "
            "user is talking about tickets / the Kanban board / moving "
            "items between workspaces."
        ),
        tier=Tier.PRODUCTION,
        lifecycle=Lifecycle.SINGLETON,
    )
    def _cp_list_tickets_factory(agent_id: str = "default"):
        for t in create_cp_tickets_tools(agent_id=agent_id):
            if t.name == "cp_list_tickets":
                return t
        raise RuntimeError("cp_list_tickets factory could not find tool")

    @register_tool(
        name="cp_search_tickets",
        capabilities=["reads-deployment-state"],
        description=(
            "Search Kanban tickets by title/description.  Returns at "
            "most 20 hits with id + title + project.  Use this before "
            "cp_move_ticket to look up the ticket UUID."
        ),
        tier=Tier.PRODUCTION,
        lifecycle=Lifecycle.SINGLETON,
    )
    def _cp_search_tickets_factory(agent_id: str = "default"):
        for t in create_cp_tickets_tools(agent_id=agent_id):
            if t.name == "cp_search_tickets":
                return t
        raise RuntimeError("cp_search_tickets factory could not find tool")

    @register_tool(
        name="cp_move_ticket",
        capabilities=["manages-tickets"],
        description=(
            "Move a Kanban ticket between projects.  Mutates "
            "control_plane.tickets.project_id and audit-logs the move "
            "as ticket.moved."
        ),
        tier=Tier.PRODUCTION,
        lifecycle=Lifecycle.SINGLETON,
    )
    def _cp_move_ticket_factory(agent_id: str = "default"):
        for t in create_cp_tickets_tools(agent_id=agent_id):
            if t.name == "cp_move_ticket":
                return t
        raise RuntimeError("cp_move_ticket factory could not find tool")
except ImportError:
    pass
