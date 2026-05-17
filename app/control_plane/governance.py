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

from app.control_plane.db import (
    execute, execute_one, execute_scalar, execute_one_required,
)

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
        """Create an approval request. Returns the request record.

        PR 3 (2026-05-16): switched the INSERT to ``execute_one_required``
        so a DB failure raises instead of silently returning ``{}``. The
        caller would otherwise proceed as if the approval was queued
        when nothing actually landed in the queue. Callers in
        ``llm_discovery`` already wrap this in ``try/except Exception``
        for fire-and-forget; the dashboard API call site doesn't go
        through here (operators approve/reject existing rows).
        """
        expires = datetime.now(timezone.utc) + timedelta(hours=expires_hours)
        row = execute_one_required(
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
        """Approve a pending request.  For request types that imply a
        concrete configuration change (currently ``model_id_remap`` and
        ``model_retired``), the corresponding persistent override is
        written so the change survives container restarts.

        PR 3 (2026-05-16): switched the UPDATE to ``execute_one_required``
        so a DB error raises instead of silently returning False.
        Returning False is now reserved for the legitimate
        "no pending request matched" case — distinct from
        "DB couldn't be reached". The dashboard handler wraps this
        and surfaces 500 on DB failure; without this fix a DB outage
        looked like a stale request to the operator.
        """
        result = execute_one_required(
            """UPDATE control_plane.governance_requests
               SET status = 'approved', reviewed_by = %s, reviewed_at = NOW()
               WHERE id = %s AND status = 'pending'
               RETURNING id, request_type, title, detail_json""",
            (reviewer, request_id),
        )
        if result:
            self.audit.log(
                actor=reviewer, action="governance.approved",
                resource_type="governance", resource_id=str(request_id),
                detail={"type": result.get("request_type"), "title": result.get("title", "")[:100]},
            )
            # ── Side-effects for request types that need persistent state ──
            # llm_discovery's retired-ID detector emits these requests with a
            # detail_json payload describing the remap/retirement.  Apply it
            # to the persistent overrides file so a restart doesn't re-detect
            # the same dead ID and create a duplicate request.
            try:
                req_type = (result.get("request_type") or "").strip()
                import json as _json
                raw_detail = result.get("detail_json") or "{}"
                detail = _json.loads(raw_detail) if isinstance(raw_detail, str) else (raw_detail or {})
                if req_type == "model_id_remap":
                    from app.llm_catalog import persist_model_id_remap
                    ck = detail.get("catalog_key") or ""
                    nm = detail.get("new_model_id") or ""
                    if ck and nm:
                        ok = persist_model_id_remap(ck, nm)
                        logger.info(
                            f"governance.approve: persisted model_id_remap "
                            f"{ck!r} -> {nm!r} (ok={ok})"
                        )
                elif req_type == "model_retired":
                    from app.llm_catalog import persist_model_retired
                    ck = detail.get("catalog_key") or ""
                    if ck:
                        ok = persist_model_retired(ck)
                        logger.info(
                            f"governance.approve: persisted model_retired {ck!r} (ok={ok})"
                        )
            except Exception:
                logger.debug(
                    "governance.approve: side-effect persistence failed "
                    "(approval itself stands)", exc_info=True,
                )
            return True
        return False

    def reject(self, request_id: str, reviewer: str = "user",
               reason: str = None) -> bool:
        """Reject a pending request.

        PR 3 (2026-05-16): mirror of ``approve`` — UPDATE uses
        ``execute_one_required`` so DB error raises instead of
        silently returning False.
        """
        result = execute_one_required(
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

_gate: GovernanceGate | None = None
_lock = threading.Lock()

def get_governance() -> GovernanceGate:
    global _gate
    if _gate is None:
        with _lock:
            if _gate is None:
                _gate = GovernanceGate()
    return _gate
