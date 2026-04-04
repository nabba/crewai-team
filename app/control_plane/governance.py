"""Governance gates — approval required for sensitive operations.

The 'board' is the human user, reachable via Signal or dashboard.

Operations requiring approval:
  evolution_deploy, budget_override, code_change, agent_config

Operations NOT requiring approval (autonomous):
  evolution_experiment, skill_creation, learning, ticket execution
"""
import json
import logging
import threading
from datetime import datetime, timezone, timedelta
from typing import Optional

from app.control_plane.db import execute, execute_one, execute_scalar

logger = logging.getLogger(__name__)

REQUIRES_APPROVAL = {
    "evolution_deploy",
    "budget_override",
    "code_change",
    "agent_config",
}


class GovernanceGate:
    """Approval queue for sensitive operations."""

    def __init__(self):
        self._audit = None

    @property
    def audit(self):
        if self._audit is None:
            from app.control_plane.audit import get_audit
            self._audit = get_audit()
        return self._audit

    def needs_approval(self, request_type: str) -> bool:
        """Quick check: does this operation type require approval?"""
        return request_type in REQUIRES_APPROVAL

    def request_approval(
        self,
        project_id: str,
        request_type: str,
        requested_by: str,
        title: str,
        detail: dict = None,
        expires_hours: int = 24,
    ) -> dict:
        """Create an approval request. Returns the request record."""
        expires = datetime.now(timezone.utc) + timedelta(hours=expires_hours)
        row = execute_one(
            """INSERT INTO control_plane.governance_requests
               (project_id, request_type, requested_by, title, detail_json, expires_at)
               VALUES (%s, %s, %s, %s, %s, %s)
               RETURNING id, request_type, title, status, created_at""",
            (project_id, request_type, requested_by, title,
             json.dumps(detail or {}), expires),
        )
        if row:
            self.audit.log(
                actor=requested_by, action="governance.requested",
                project_id=project_id,
                resource_type="governance", resource_id=str(row["id"]),
                detail={"type": request_type, "title": title[:200]},
            )
        return row or {}

    def approve(self, request_id: str, reviewer: str = "user") -> bool:
        """Approve a pending request."""
        result = execute_one(
            """UPDATE control_plane.governance_requests
               SET status = 'approved', reviewed_by = %s, reviewed_at = NOW()
               WHERE id = %s AND status = 'pending'
               RETURNING id, request_type, title""",
            (reviewer, request_id),
        )
        if result:
            self.audit.log(
                actor=reviewer, action="governance.approved",
                resource_type="governance", resource_id=str(request_id),
                detail={"type": result.get("request_type"), "title": result.get("title", "")[:100]},
            )
            return True
        return False

    def reject(self, request_id: str, reviewer: str = "user",
               reason: str = None) -> bool:
        """Reject a pending request."""
        result = execute_one(
            """UPDATE control_plane.governance_requests
               SET status = 'rejected', reviewed_by = %s, reviewed_at = NOW()
               WHERE id = %s AND status = 'pending'
               RETURNING id, request_type, title""",
            (reviewer, request_id),
        )
        if result:
            self.audit.log(
                actor=reviewer, action="governance.rejected",
                resource_type="governance", resource_id=str(request_id),
                detail={"type": result.get("request_type"), "reason": reason or ""},
            )
            return True
        return False

    def expire_old(self) -> int:
        """Expire requests past their deadline."""
        rows = execute(
            """UPDATE control_plane.governance_requests
               SET status = 'expired'
               WHERE status = 'pending' AND expires_at < NOW()
               RETURNING id""",
            fetch=True,
        )
        return len(rows or [])

    def get_pending(self, project_id: str = None) -> list[dict]:
        """Get all pending requests."""
        if project_id:
            return execute(
                """SELECT * FROM control_plane.governance_requests
                   WHERE status = 'pending' AND project_id = %s
                   ORDER BY created_at""",
                (project_id,), fetch=True,
            ) or []
        return execute(
            """SELECT * FROM control_plane.governance_requests
               WHERE status = 'pending'
               ORDER BY created_at""",
            fetch=True,
        ) or []

    def pending_count(self, project_id: str = None) -> int:
        """Number of pending approvals."""
        if project_id:
            return execute_scalar(
                """SELECT COUNT(*) FROM control_plane.governance_requests
                   WHERE status = 'pending' AND project_id = %s""",
                (project_id,),
            ) or 0
        return execute_scalar(
            "SELECT COUNT(*) FROM control_plane.governance_requests WHERE status = 'pending'",
        ) or 0

    def format_pending(self, project_id: str = None) -> str:
        """Human-readable pending requests for Signal."""
        pending = self.get_pending(project_id)
        if not pending:
            return "No pending governance requests. ✅"
        lines = ["⚖️ Pending Approvals:"]
        for r in pending:
            rid = str(r["id"])[:8]
            lines.append(
                f"  #{rid} [{r.get('request_type')}] {r.get('title', '—')[:60]}\n"
                f"    by {r.get('requested_by')} · {r.get('created_at', '')[:10]}\n"
                f"    → approve {rid} / reject {rid}"
            )
        return "\n".join(lines)


# ── Singleton ────────────────────────────────────────────────────────────────

_gate: Optional[GovernanceGate] = None
_lock = threading.Lock()


def get_governance() -> GovernanceGate:
    global _gate
    if _gate is None:
        with _lock:
            if _gate is None:
                _gate = GovernanceGate()
    return _gate
