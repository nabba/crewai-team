"""Control plane — action-request endpoints at /api/cp/action-requests.

Operators (via React or curl) can:

  GET  /api/cp/action-requests                    list (filter by status / type)
  GET  /api/cp/action-requests/{id}               detail
  POST /api/cp/action-requests/{id}/approve       approve + apply (gate)
  POST /api/cp/action-requests/{id}/reject        reject (terminal)
  POST /api/cp/action-requests/{id}/retry-apply   retry after APPLY_FAILED
  GET  /api/cp/action-requests/types              registered action types

Auth: same ``require_gateway_auth`` as the rest of /cp/.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.action_requests import (
    ActionStatus,
    ActionType,
    DecisionSource,
    InvalidActionTransition,
    apply,
    approve,
    get,
    list_action_types,
    list_all,
    reject,
)
from app.control_plane.auth_dep import require_gateway_auth

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/cp/action-requests",
    tags=["control-plane", "action-requests"],
    dependencies=[Depends(require_gateway_auth)],
)


class _ApproveBody(BaseModel):
    operator: str = Field(default="react-operator")
    reason: str | None = Field(default=None)


class _RejectBody(BaseModel):
    operator: str = Field(default="react-operator")
    reason: str | None = Field(default=None)


def _serialize(req) -> dict[str, Any]:
    if req is None:
        return {}
    d = req.to_dict()
    d["is_terminal"] = req.is_terminal
    d["is_decided"] = req.is_decided
    return d


def _require(request_id: str):
    req = get(request_id)
    if req is None:
        raise HTTPException(
            status_code=404, detail=f"action-request {request_id!r} not found",
        )
    return req


@router.get("/types")
def get_supported_types():
    """Enumerate registered action types — what handlers are available."""
    return {
        "types": [t.value for t in list_action_types()],
    }


@router.get("")
def list_action_requests(
    status: str | None = Query(default=None),
    action_type: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
):
    status_enum: ActionStatus | None = None
    if status:
        try:
            status_enum = ActionStatus(status)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"invalid status {status!r}. Valid: "
                    f"{[s.value for s in ActionStatus]}"
                ),
            )
    items = list_all(status=status_enum, limit=limit)
    if action_type:
        try:
            t = ActionType(action_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"invalid action_type {action_type!r}. Valid: "
                    f"{[t.value for t in list_action_types()]}"
                ),
            )
        items = [r for r in items if r.action_type is t]
    return {
        "count": len(items),
        "action_requests": [_serialize(r) for r in items],
    }


@router.get("/{request_id}")
def get_action_request(request_id: str):
    return _serialize(_require(request_id))


@router.post("/{request_id}/approve")
def approve_action_request(request_id: str, body: _ApproveBody):
    """Approve + apply. Idempotent: a second approve on an
    already-APPROVED request just retries the apply."""
    req = _require(request_id)
    if req.status is ActionStatus.INVALID:
        raise HTTPException(
            status_code=403,
            detail=(
                "Cannot approve INVALID action-requests; the validator "
                "rejected the data before it reached this gate. Reason: "
                f"{req.invalid_reason}"
            ),
        )
    if req.status not in (
        ActionStatus.PENDING, ActionStatus.APPROVED, ActionStatus.APPLY_FAILED,
    ):
        raise HTTPException(
            status_code=409,
            detail=f"cannot approve in status {req.status.value}",
        )
    if req.status is ActionStatus.PENDING:
        approve(
            request_id,
            source=DecisionSource.REACT_APPROVE,
            decision_reason=body.reason,
        )
    elif req.status is ActionStatus.APPLY_FAILED:
        # Recover-after-failure: re-approve to allow retry.
        approve(
            request_id,
            source=DecisionSource.REACT_APPROVE,
            decision_reason=body.reason or "retry after apply failure",
        )
    try:
        applied = apply(request_id)
    except InvalidActionTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {
        "ok": applied.status is ActionStatus.APPLIED,
        "action_request": _serialize(applied),
    }


@router.post("/{request_id}/reject")
def reject_action_request(request_id: str, body: _RejectBody):
    req = _require(request_id)
    if req.status is not ActionStatus.PENDING:
        raise HTTPException(
            status_code=409,
            detail=f"cannot reject in status {req.status.value}",
        )
    updated = reject(
        request_id,
        source=DecisionSource.REACT_REJECT,
        decision_reason=body.reason,
    )
    return {"ok": True, "action_request": _serialize(updated)}


@router.post("/{request_id}/retry-apply")
def retry_apply(request_id: str):
    """Retry the apply step for an APPLY_FAILED request."""
    req = _require(request_id)
    if req.status is not ActionStatus.APPLY_FAILED:
        raise HTTPException(
            status_code=409,
            detail=(
                f"retry-apply only allowed for APPLY_FAILED; "
                f"current={req.status.value}"
            ),
        )
    approve(
        request_id,
        source=DecisionSource.REACT_APPROVE,
        decision_reason="retry after apply failure",
    )
    applied = apply(request_id)
    return {
        "ok": applied.status is ActionStatus.APPLIED,
        "action_request": _serialize(applied),
    }
