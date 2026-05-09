"""Ticket system — persistent task tracking replacing transient Signal messages.

Every user request becomes a ticket. Every crew execution is logged against it.
Ticket lifecycle: todo → in_progress → review → done/failed/blocked
"""
import json
import logging
import threading
from datetime import datetime, timezone

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
        """Create a ticket from an incoming Signal message.

        Idempotent w.r.t. the inbound_queue replay path: if an active ticket
        with the same title already exists for this project within the last
        60 minutes, reuse it instead of spawning a duplicate.  Without this,
        a crash-loop where the crew fails to finalize (cost stays $0,
        ticket stays in_progress) produces N new tickets for the same
        message every time the inbound queue replays.
        """
        title = message[:200].strip()

        # ── Replay idempotency ──────────────────────────────────────────
        # We key on (project_id, title, source='signal', active-status,
        # recent) because the tickets table has no sender/signal_ts
        # columns.  Title is message[:200] so exact-duplicate messages
        # from the same queue row always match.  60 min window is long
        # enough to cover inbound-queue retry spacing but short enough
        # that an intentional same-text resend an hour later creates a
        # fresh ticket.
        existing = execute_one(
            """SELECT id, title, status, created_at
                 FROM control_plane.tickets
                WHERE project_id = %s
                  AND title = %s
                  AND source = 'signal'
                  AND status IN ('todo', 'in_progress')
                  AND created_at >= NOW() - interval '60 minutes'
                ORDER BY created_at DESC
                LIMIT 1""",
            (project_id, title),
        )
        if existing:
            self.audit.log(
                actor="system", action="ticket.reused",
                project_id=project_id,
                resource_type="ticket", resource_id=str(existing["id"]),
                detail={"title": title[:100], "source": "signal"},
            )
            logger.info(
                "tickets: reused existing active ticket id=%s title=%r (skip duplicate on replay)",
                existing["id"], title[:60],
            )
            return existing

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
        from app.control_plane.db import execute_one
        project_id = None
        try:
            row = execute_one(
                "SELECT project_id FROM control_plane.tickets WHERE id = %s",
                (ticket_id,),
            )
            if row:
                pid = row.get("project_id") if isinstance(row, dict) else row[0]
                project_id = str(pid) if pid is not None else None
        except Exception:
            pass
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
            project_id=project_id,
            resource_type="ticket", resource_id=str(ticket_id),
            cost_usd=cost_usd, tokens=tokens,
            detail={"result_preview": result_summary[:200]},
        )

    def move_ticket(
        self, ticket_id: str, target_project_name: str
    ) -> dict | None:
        """Move a ticket to a different project (by project name).

        Resolves ``target_project_name`` via
        ``ProjectManager.get_by_name`` (case-insensitive).  Returns the
        updated ticket row, or ``None`` if either the ticket or the
        target project is unknown.

        Idempotent: re-moving a ticket to the project it already
        belongs to writes the same UPDATE and re-emits an audit entry,
        but produces no observable state change.
        """
        from app.control_plane.projects import get_projects

        target_project = get_projects().get_by_name(target_project_name)
        if not target_project:
            return None

        target_project_id = str(target_project["id"])
        target_canonical_name = target_project.get("name") or target_project_name

        existing = execute_one(
            "SELECT id, project_id FROM control_plane.tickets WHERE id = %s",
            (ticket_id,),
        )
        if not existing:
            return None

        from_project_id = existing.get("project_id") if isinstance(existing, dict) else existing[1]
        from_project_id_str = str(from_project_id) if from_project_id is not None else None

        row = execute_one(
            """UPDATE control_plane.tickets
                  SET project_id = %s, updated_at = NOW()
                WHERE id = %s
            RETURNING id, title, status, project_id, priority,
                      assigned_crew, assigned_agent, created_at, updated_at""",
            (target_project_id, ticket_id),
        )
        if row:
            self.audit.log(
                actor="user", action="ticket.moved",
                project_id=target_project_id,
                resource_type="ticket", resource_id=str(ticket_id),
                detail={
                    "from_project_id": from_project_id_str,
                    "to_project_id": target_project_id,
                    "to_project_name": target_canonical_name,
                },
            )
        return row

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

    def fail_stuck_in_progress(
        self,
        max_age_minutes: int = 15,
        only_zero_cost: bool = True,
    ) -> list[str]:
        """Janitor: mark any in_progress ticket that has been running for
        longer than ``max_age_minutes`` without accruing cost as failed.

        Handles the case where the orchestrator gets trapped in a retry
        loop deep inside the auditor or a crew — the thread keeps running
        past the 600s asyncio.wait_for timeout (Python can't cancel
        threads), so the outer safety net in handle_task() never fires
        and the ticket stays ``in_progress`` indefinitely at $0.  This
        janitor runs every 5 min on the scheduler and is a last-resort
        sweeper.

        Args:
            max_age_minutes: tickets with started_at older than this are
                considered stuck.
            only_zero_cost: when True (default), only fail tickets that
                have recorded $0 of cost — a ticket actively spending on
                LLM calls is clearly making progress and should be left
                alone.

        Returns:
            list of ticket ids that were marked failed.
        """
        cond = ["status = 'in_progress'",
                f"started_at < NOW() - interval '{int(max_age_minutes)} minutes'"]
        if only_zero_cost:
            cond.append("cost_usd = 0")
        where = " AND ".join(cond)
        rows = execute(
            f"""UPDATE control_plane.tickets
                   SET status = 'failed',
                       result_summary = 'janitor: stuck in_progress > '
                                        || %s || ' min with $0 cost — '
                                        || 'orchestrator hung past timeout '
                                        || '(see stuck-ticket janitor).',
                       completed_at = NOW(),
                       updated_at = NOW()
                 WHERE {where}
             RETURNING id, title""",
            (int(max_age_minutes),),
            fetch=True,
        ) or []
        failed_ids: list[str] = []
        for r in rows:
            tid = str(r.get("id"))
            failed_ids.append(tid)
            try:
                self.audit.log(
                    actor="system", action="ticket.janitor_failed",
                    resource_type="ticket", resource_id=tid,
                    detail={
                        "title": str(r.get("title", ""))[:100],
                        "reason": f"stuck in_progress > {max_age_minutes}min @ $0",
                    },
                )
            except Exception:
                pass
        if failed_ids:
            logger.warning(
                "tickets.janitor: marked %d stuck in_progress ticket(s) as failed: %s",
                len(failed_ids), failed_ids,
            )
        return failed_ids

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

_tickets: TicketManager | None = None
_lock = threading.Lock()

def get_tickets() -> TicketManager:
    global _tickets
    if _tickets is None:
        with _lock:
            if _tickets is None:
                _tickets = TicketManager()
    return _tickets
