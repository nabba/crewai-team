"""Control plane — change-request endpoints at /api/cp/changes.

Phase 5.3a backend. Operators (via React or curl) can:

  GET    /api/cp/changes                    list (filtered by status)
  GET    /api/cp/changes/{id}                detail
  POST   /api/cp/changes/{id}/approve        approve + apply (gate 1)
  POST   /api/cp/changes/{id}/reject         reject (terminal)
  POST   /api/cp/changes/{id}/rollback       rollback (APPLIED only)
  POST   /api/cp/changes/{id}/retry-apply    retry after APPLY_FAILED

Auth: same `require_gateway_auth` dependency as the rest of /cp/.

These endpoints are the "React surface" of the change-request
system. The Signal surface (👍/👎 reactions) is handled by the
reaction-handler hook in main.py. Both surfaces dispatch through
the same lifecycle module — the audit log records which source
made each decision.

When operator React-side approves a change that already has a
Signal ASK out, the lifecycle's idempotent transitions handle
the race correctly: whoever wins first becomes the decided_by;
the loser sees "already approved" / "already rejected".
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.change_requests import (
    DecisionSource,
    Status,
    apply_change,
    approve,
    get,
    is_protected,
    list_all,
    reject,
    rollback_change,
)
from app.control_plane.auth_dep import require_gateway_auth

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/cp/changes",
    tags=["control-plane", "change-requests"],
    dependencies=[Depends(require_gateway_auth)],
)


# ── Request models for POST endpoints ───────────────────────────────


class _ApproveBody(BaseModel):
    operator: str = Field(
        default="react-operator",
        description="Identifier for the operator approving via React.",
    )
    reason: str | None = Field(
        default=None,
        description="Optional approval note (e.g. 'reviewed and looks correct').",
    )


class _RejectBody(BaseModel):
    operator: str = Field(default="react-operator")
    reason: str | None = Field(
        default=None,
        description="Required-for-good-hygiene rejection reason.",
    )


class _RollbackBody(BaseModel):
    operator: str = Field(
        default="react-operator",
        description="Identifier for the operator triggering the rollback.",
    )


# ── Helpers ─────────────────────────────────────────────────────────


def _serialize(cr) -> dict[str, Any]:
    """Tighten the dict shape for the React UI — adds derived fields
    that are awkward to compute client-side."""
    if cr is None:
        return {}
    d = cr.to_dict()
    d["is_terminal"] = cr.is_terminal
    d["is_rollbackable"] = cr.is_rollbackable
    d["is_protected"] = is_protected(cr.path)
    return d


# ── Routes ──────────────────────────────────────────────────────────


@router.get("")
def list_changes(
    status: str | None = Query(
        default=None,
        description=(
            "Filter by status (pending, approved, rejected, applied, "
            "apply_failed, rolled_back, tier_immutable_refused, timeout)."
        ),
    ),
    limit: int = Query(default=100, ge=1, le=500),
):
    """List change requests, newest first, optionally filtered by status."""
    status_enum: Status | None = None
    if status:
        try:
            status_enum = Status(status)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"invalid status {status!r}. Valid: "
                    f"{[s.value for s in Status]}"
                ),
            )
    items = list_all(status=status_enum, limit=limit)
    return {
        "count": len(items),
        "changes": [_serialize(c) for c in items],
    }


@router.get("/{request_id}")
def get_change(request_id: str):
    cr = get(request_id)
    if cr is None:
        raise HTTPException(status_code=404, detail=f"change request {request_id!r} not found")
    return _serialize(cr)


@router.post("/{request_id}/approve")
def approve_change(request_id: str, body: _ApproveBody):
    """Approve + apply. The lifecycle moves PENDING → APPROVED → APPLIED
    (or APPLY_FAILED). Idempotent: a second approve on an already-
    APPROVED request just retries the apply."""
    cr = get(request_id)
    if cr is None:
        raise HTTPException(status_code=404, detail="not found")
    if cr.status == Status.TIER_IMMUTABLE_REFUSED:
        raise HTTPException(
            status_code=403,
            detail=(
                "TIER_IMMUTABLE files cannot be approved through this "
                "path, even by operator React override. Operator must "
                "edit directly via PR."
            ),
        )
    if cr.status not in (Status.PENDING, Status.APPROVED, Status.APPLY_FAILED):
        raise HTTPException(
            status_code=409,
            detail=(
                f"cannot approve in status {cr.status.value}. "
                f"Already-applied / rolled-back / timed-out / rejected "
                f"requests cannot be re-approved."
            ),
        )
    # PENDING → APPROVED
    if cr.status == Status.PENDING:
        approve(
            request_id,
            source=DecisionSource.REACT_APPROVE,
            decision_reason=body.reason,
        )
    # APPROVED or APPLY_FAILED → trigger apply
    apply_result = apply_change(request_id)
    final = get(request_id)
    return {
        "ok": apply_result.ok,
        "change": _serialize(final),
        "apply_result": {
            "ok": apply_result.ok,
            "git_branch": apply_result.git_branch,
            "git_commit_sha": apply_result.git_commit_sha,
            "pr_url": apply_result.pr_url,
            "module_reload_ok": apply_result.module_reload_ok,
            "module_reload_note": apply_result.module_reload_note,
            "error": apply_result.error,
        },
    }


@router.post("/{request_id}/reject")
def reject_change(request_id: str, body: _RejectBody):
    cr = get(request_id)
    if cr is None:
        raise HTTPException(status_code=404, detail="not found")
    if cr.status != Status.PENDING:
        raise HTTPException(
            status_code=409,
            detail=f"cannot reject in status {cr.status.value}",
        )
    updated = reject(
        request_id,
        source=DecisionSource.REACT_REJECT,
        decision_reason=body.reason,
    )
    return {"ok": True, "change": _serialize(updated)}


@router.post("/{request_id}/rollback")
def rollback_route(request_id: str, body: _RollbackBody):
    """Roll back an APPLIED change. Reverts the commit + hot-reverts
    the file + opens a revert PR. Operator merges the revert PR to
    make the rollback durable in main."""
    cr = get(request_id)
    if cr is None:
        raise HTTPException(status_code=404, detail="not found")
    if not cr.is_rollbackable:
        raise HTTPException(
            status_code=409,
            detail=(
                f"cannot rollback in status {cr.status.value}. Only "
                f"APPLIED requests can be rolled back."
            ),
        )
    result = rollback_change(request_id, operator=body.operator)
    final = get(request_id)
    return {
        "ok": result.ok,
        "change": _serialize(final),
        "rollback_result": {
            "ok": result.ok,
            "revert_branch": result.git_branch,
            "revert_commit_sha": result.git_commit_sha,
            "revert_pr_url": result.pr_url,
            "module_reload_ok": result.module_reload_ok,
            "module_reload_note": result.module_reload_note,
            "error": result.error,
        },
    }


@router.post("/{request_id}/retry-apply")
def retry_apply(request_id: str):
    """Retry the apply step for an APPLY_FAILED request. Same code
    path as approve()'s apply call; useful when the original failure
    was transient (bridge briefly unreachable, etc)."""
    cr = get(request_id)
    if cr is None:
        raise HTTPException(status_code=404, detail="not found")
    if cr.status != Status.APPLY_FAILED:
        raise HTTPException(
            status_code=409,
            detail=(
                f"retry-apply only allowed for APPLY_FAILED; "
                f"current={cr.status.value}"
            ),
        )
    # Move back to APPROVED so apply_change accepts it; lifecycle
    # tolerates the re-approval as idempotent.
    approve(
        request_id,
        source=DecisionSource.REACT_APPROVE,
        decision_reason="retry after apply failure",
    )
    result = apply_change(request_id)
    final = get(request_id)
    return {
        "ok": result.ok,
        "change": _serialize(final),
        "apply_result": {
            "ok": result.ok,
            "git_branch": result.git_branch,
            "git_commit_sha": result.git_commit_sha,
            "pr_url": result.pr_url,
            "error": result.error,
        },
    }
