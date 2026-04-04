"""Ticket system — persistent task tracking replacing transient Signal messages.

Every user request becomes a ticket. Every crew execution is logged against it.
Ticket lifecycle: todo → in_progress → review → done/failed/blocked
"""
import json
import logging
import threading
from datetime import datetime, timezone
from typing import Optional

from app.control_plane.db import execute, execute_one, execute_scalar

logger = logging.getLogger(__name__)


class TicketManager:
    """Persistent task tracking with threaded comments."""

    def __init__(self):
        self._audit = None

    @property
    def audit(self):
        if self._audit is None:
            from app.control_plane.audit import get_audit
            self._audit = get_audit()
        return self._audit

    def create_from_signal(
        self,
        message: str,
        sender: str,
        project_id: str,
        difficulty: int = None,
        priority: int = 5,
    ) -> dict:
        """Create a ticket from an incoming Signal message."""
        title = message[:200].strip()
        row = execute_one(
            """INSERT INTO control_plane.tickets
               (project_id, title, description, source, difficulty, priority)
               VALUES (%s, %s, %s, 'signal', %s, %s)
               RETURNING id, title, status, created_at""",
            (project_id, title, message[:4000], difficulty, priority),
        )
        if row:
            self.audit.log(
                actor="user", action="ticket.created",
                project_id=project_id,
                resource_type="ticket", resource_id=str(row["id"]),
                detail={"title": title[:100], "source": "signal"},
            )
        return row or {}

    def create_manual(
        self,
        title: str,
        project_id: str,
        description: str = None,
        priority: int = 5,
        source: str = "dashboard",
    ) -> dict:
        """Create a ticket manually (from dashboard or command)."""
        row = execute_one(
            """INSERT INTO control_plane.tickets
               (project_id, title, description, source, priority)
               VALUES (%s, %s, %s, %s, %s)
               RETURNING id, title, status, created_at""",
            (project_id, title, description, source, priority),
        )
        if row:
            self.audit.log(
                actor="user", action="ticket.created",
                project_id=project_id,
                resource_type="ticket", resource_id=str(row["id"]),
                detail={"title": title[:100], "source": source},
            )
        return row or {}

    def assign_to_crew(self, ticket_id: str, crew: str, agent: str) -> None:
        """Assign ticket to a crew/agent and transition to in_progress."""
        execute(
            """UPDATE control_plane.tickets
               SET assigned_crew = %s, assigned_agent = %s,
                   status = 'in_progress', started_at = NOW(), updated_at = NOW()
               WHERE id = %s""",
            (crew, agent, ticket_id),
        )
        self.audit.log(
            actor="commander", action="ticket.assigned",
            resource_type="ticket", resource_id=str(ticket_id),
            detail={"crew": crew, "agent": agent},
        )

    def add_comment(self, ticket_id: str, author: str, content: str,
                    metadata: dict = None) -> None:
        """Thread a comment onto a ticket (agent output, user reply, etc.)."""
        execute(
            """INSERT INTO control_plane.ticket_comments
               (ticket_id, author, content, metadata_json)
               VALUES (%s, %s, %s, %s)""",
            (ticket_id, author, content[:10000], json.dumps(metadata or {})),
        )

    def complete(self, ticket_id: str, result_summary: str,
                 cost_usd: float = 0, tokens: int = 0) -> None:
        """Mark ticket done with cost and result summary."""
        execute(
            """UPDATE control_plane.tickets
               SET status = 'done', result_summary = %s,
                   cost_usd = %s, tokens_used = %s,
                   completed_at = NOW(), updated_at = NOW()
               WHERE id = %s""",
            (result_summary[:4000], cost_usd, tokens, ticket_id),
        )
        self.audit.log(
            actor="system", action="ticket.completed",
            resource_type="ticket", resource_id=str(ticket_id),
            cost_usd=cost_usd, tokens=tokens,
            detail={"result_preview": result_summary[:200]},
        )

    def fail(self, ticket_id: str, error: str) -> None:
        """Mark ticket as failed."""
        execute(
            """UPDATE control_plane.tickets
               SET status = 'failed', result_summary = %s,
                   completed_at = NOW(), updated_at = NOW()
               WHERE id = %s""",
            (error[:4000], ticket_id),
        )
        self.audit.log(
            actor="system", action="ticket.failed",
            resource_type="ticket", resource_id=str(ticket_id),
            detail={"error": error[:200]},
        )

    def close(self, ticket_id: str) -> None:
        """Mark ticket done (manual close by user)."""
        execute(
            """UPDATE control_plane.tickets
               SET status = 'done', completed_at = NOW(), updated_at = NOW()
               WHERE id = %s""",
            (ticket_id,),
        )

    def get(self, ticket_id: str) -> dict | None:
        """Get a single ticket with its comments."""
        ticket = execute_one(
            """SELECT * FROM control_plane.tickets WHERE id = %s""",
            (ticket_id,),
        )
        if not ticket:
            return None
        comments = execute(
            """SELECT * FROM control_plane.ticket_comments
               WHERE ticket_id = %s ORDER BY created_at""",
            (ticket_id,), fetch=True,
        )
        ticket["comments"] = comments or []
        return ticket

    def get_board(self, project_id: str = None, limit: int = 50) -> dict:
        """Kanban-style board: tickets grouped by status."""
        cond = "WHERE project_id = %s" if project_id else ""
        params = (project_id, limit) if project_id else (limit,)
        rows = execute(
            f"""SELECT id, title, status, priority, assigned_agent,
                       assigned_crew, difficulty, cost_usd, tokens_used,
                       created_at, started_at, completed_at
                FROM control_plane.tickets
                {cond}
                ORDER BY updated_at DESC
                LIMIT %s""",
            params, fetch=True,
        )
        board = {"todo": [], "in_progress": [], "review": [], "done": [], "failed": [], "blocked": []}
        for r in (rows or []):
            board.setdefault(r.get("status", "todo"), []).append(r)
        # Count totals
        counts = {k: len(v) for k, v in board.items()}
        return {"board": board, "counts": counts, "total": sum(counts.values())}

    def get_recent(self, project_id: str = None, limit: int = 20) -> list[dict]:
        """Get recent tickets (all statuses)."""
        if project_id:
            return execute(
                """SELECT * FROM control_plane.tickets
                   WHERE project_id = %s
                   ORDER BY created_at DESC LIMIT %s""",
                (project_id, limit), fetch=True,
            ) or []
        return execute(
            """SELECT * FROM control_plane.tickets
               ORDER BY created_at DESC LIMIT %s""",
            (limit,), fetch=True,
        ) or []


# ── Singleton ────────────────────────────────────────────────────────────────

_tickets: Optional[TicketManager] = None
_lock = threading.Lock()


def get_tickets() -> TicketManager:
    global _tickets
    if _tickets is None:
        with _lock:
            if _tickets is None:
                _tickets = TicketManager()
    return _tickets
