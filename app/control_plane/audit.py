"""Immutable audit trail — INSERT-only PostgreSQL.

The DB user should have INSERT+SELECT only on control_plane.audit_log.
No UPDATE, no DELETE, no TRUNCATE. Agents cannot erase their tracks.

Satisfies the DGM safety invariant: audit infrastructure is at the
infrastructure level, not inside agent code.
"""
import json
import logging
import threading
from datetime import datetime, timezone
from typing import Optional

from app.control_plane.db import execute, execute_one

logger = logging.getLogger(__name__)


class AuditTrail:
    """Append-only audit log writer."""

    def log(
        self,
        *,
        actor: str,
        action: str,
        project_id: str = None,
        resource_type: str = None,
        resource_id: str = None,
        detail: dict = None,
        cost_usd: float = None,
        tokens: int = None,
    ) -> None:
        """Write an immutable audit entry. Fire-and-forget."""
        try:
            execute(
                """INSERT INTO control_plane.audit_log
                   (project_id, actor, action, resource_type, resource_id,
                    detail_json, cost_usd, tokens)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    project_id, actor, action, resource_type, resource_id,
                    json.dumps(detail or {}), cost_usd, tokens,
                ),
            )
        except Exception as e:
            # Never fail silently on audit — log to stderr as fallback
            logger.error(f"AUDIT WRITE FAILED: {actor}/{action}: {e}")

    def query(
        self,
        *,
        project_id: str = None,
        actor: str = None,
        action_prefix: str = None,
        resource_type: str = None,
        since: datetime = None,
        limit: int = 50,
    ) -> list[dict]:
        """Read-only query for dashboard and reporting."""
        conditions = []
        params = []

        if project_id:
            conditions.append("project_id = %s")
            params.append(project_id)
        if actor:
            conditions.append("actor = %s")
            params.append(actor)
        if action_prefix:
            conditions.append("action LIKE %s")
            params.append(f"{action_prefix}%")
        if resource_type:
            conditions.append("resource_type = %s")
            params.append(resource_type)
        if since:
            conditions.append("timestamp >= %s")
            params.append(since)

        where = " AND ".join(conditions) if conditions else "TRUE"
        params.append(limit)

        rows = execute(
            f"""SELECT id, timestamp, project_id, actor, action,
                       resource_type, resource_id, detail_json, cost_usd, tokens
                FROM control_plane.audit_log
                WHERE {where}
                ORDER BY timestamp DESC
                LIMIT %s""",
            tuple(params),
            fetch=True,
        )
        return rows or []

    def cost_summary(self, project_id: str = None, period: str = None) -> dict:
        """Aggregate cost by actor for a project/period."""
        conditions = ["cost_usd IS NOT NULL"]
        params = []
        if project_id:
            conditions.append("project_id = %s")
            params.append(project_id)
        if period:
            conditions.append("timestamp >= %s")
            params.append(f"{period}-01")

        where = " AND ".join(conditions)
        rows = execute(
            f"""SELECT actor, COUNT(*) as calls,
                       SUM(cost_usd) as total_cost,
                       SUM(tokens) as total_tokens
                FROM control_plane.audit_log
                WHERE {where}
                GROUP BY actor
                ORDER BY total_cost DESC""",
            tuple(params),
            fetch=True,
        )
        return {"by_actor": rows or [], "total_cost": sum(r.get("total_cost", 0) or 0 for r in (rows or []))}


# ── Singleton ────────────────────────────────────────────────────────────────

_audit: Optional[AuditTrail] = None
_lock = threading.Lock()


def get_audit() -> AuditTrail:
    global _audit
    if _audit is None:
        with _lock:
            if _audit is None:
                _audit = AuditTrail()
    return _audit
