"""ActionRequest lifecycle — function-per-transition entry points.

Mirrors change_requests + architecture_requests. Every transition
saves the request; illegal transitions raise
:class:`InvalidActionTransition`.

State machine::

    PENDING ─┬─→ APPROVED ──→ APPLIED
             │             ╲
             ├─→ REJECTED    ╲─→ APPLY_FAILED
             ├─→ INVALID         (recoverable via retry_apply)
             └─→ TIMEOUT
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from app.action_requests import store
from app.action_requests.handlers import get_handler
from app.action_requests.models import (
    ActionRequest,
    ActionStatus,
    ActionType,
    DecisionSource,
)
from app.action_requests.validator import validate

logger = logging.getLogger(__name__)


class InvalidActionTransition(RuntimeError):
    """Raised when a state transition would violate the state machine."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_status(
    req: ActionRequest,
    expected: set[ActionStatus],
    transition: str,
) -> None:
    if req.status not in expected:
        raise InvalidActionTransition(
            f"cannot {transition}: action_request {req.id} is in status "
            f"{req.status.value}; expected one of "
            f"{sorted(s.value for s in expected)}"
        )


def _get_or_raise(request_id: str) -> ActionRequest:
    req = store.get(request_id)
    if req is None:
        raise KeyError(f"unknown action_request {request_id!r}")
    return req


# ── Public API ──────────────────────────────────────────────────────


def create_request(
    *,
    requestor: str,
    action_type: ActionType,
    summary: str,
    data: dict[str, Any],
    reason: str,
) -> ActionRequest:
    """Validate the action and persist as PENDING (or INVALID)."""
    req = ActionRequest(
        id=str(uuid.uuid4()),
        created_at=_now_iso(),
        requestor=requestor,
        action_type=action_type,
        summary=summary,
        data=dict(data),
        reason=reason,
        status=ActionStatus.PENDING,
    )
    result = validate(req)
    if not result.ok:
        req.status = ActionStatus.INVALID
        req.invalid_reason = result.reason
        req.decided_at = _now_iso()
        store.save(req)
        return req
    store.save(req)
    return req


def approve(
    request_id: str,
    *,
    source: DecisionSource,
    decision_reason: str | None = None,
) -> ActionRequest:
    req = _get_or_raise(request_id)
    if req.status is ActionStatus.APPROVED:
        return req
    if req.status is ActionStatus.APPLY_FAILED:
        # Retry-after-failure path: re-approve to allow another apply.
        req.decided_at = _now_iso()
        req.decided_by = source
        req.decision_reason = decision_reason
        req.status = ActionStatus.APPROVED
        store.save(req)
        return req
    _require_status(req, {ActionStatus.PENDING}, "approve")
    req.status = ActionStatus.APPROVED
    req.decided_at = _now_iso()
    req.decided_by = source
    req.decision_reason = decision_reason
    store.save(req)
    return req


def reject(
    request_id: str,
    *,
    source: DecisionSource,
    decision_reason: str | None = None,
) -> ActionRequest:
    req = _get_or_raise(request_id)
    _require_status(req, {ActionStatus.PENDING}, "reject")
    req.status = ActionStatus.REJECTED
    req.decided_at = _now_iso()
    req.decided_by = source
    req.decision_reason = decision_reason
    store.save(req)
    return req


def apply(request_id: str) -> ActionRequest:
    """Execute the approved action via its handler.

    Transitions APPROVED → APPLIED on success or APPROVED →
    APPLY_FAILED on handler failure. APPLY_FAILED is recoverable —
    the operator can re-approve to retry.
    """
    req = _get_or_raise(request_id)
    _require_status(
        req,
        {ActionStatus.APPROVED},
        "apply",
    )
    handler = get_handler(req.action_type)
    if handler is None:
        req.status = ActionStatus.APPLY_FAILED
        req.apply_error = (
            f"no handler registered for {req.action_type.value!r}"
        )
        req.applied_at = _now_iso()
        store.save(req)
        return req

    try:
        result = handler.apply(req.data)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "action_requests: handler.apply raised for %s: %s",
            request_id, exc, exc_info=True,
        )
        req.status = ActionStatus.APPLY_FAILED
        req.apply_error = f"handler raised: {exc}"
        req.applied_at = _now_iso()
        store.save(req)
        return req

    if result.ok:
        req.status = ActionStatus.APPLIED
        req.apply_artifact = dict(result.artifact)
        req.apply_error = None
    else:
        req.status = ActionStatus.APPLY_FAILED
        req.apply_error = result.error or "handler reported failure"
    req.applied_at = _now_iso()
    store.save(req)
    return req


def expire(request_id: str) -> ActionRequest:
    req = _get_or_raise(request_id)
    _require_status(req, {ActionStatus.PENDING}, "expire")
    req.status = ActionStatus.TIMEOUT
    req.decided_at = _now_iso()
    req.decided_by = DecisionSource.TIMEOUT
    store.save(req)
    return req


def attach_signal_ts(request_id: str, signal_ts: int) -> ActionRequest:
    """Record the Signal message timestamp for reaction correlation."""
    req = _get_or_raise(request_id)
    req.signal_message_ts = signal_ts
    store.save(req)
    return req
