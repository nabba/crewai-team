"""Lifecycle orchestration for architecture requests.

Public entry points (one per state transition):

  * :func:`create_request` — agent calls via the ``propose_architecture``
    tool. Validates; persists as PROPOSED (or TIER_IMMUTABLE_REFUSED).
    Returns the created request.
  * :func:`approve` — PROPOSED → APPROVED. Idempotent if already APPROVED.
  * :func:`reject` — PROPOSED → REJECTED.
  * :func:`scaffold` — APPROVED → SCAFFOLDED. Calls into
    :mod:`scaffolder` to write stub files into the staging directory.
  * :func:`record_child_change_request` — SCAFFOLDED → IMPLEMENTING
    on first child CR, then accumulates further child CR ids.
  * :func:`mark_complete` — IMPLEMENTING → COMPLETED when all child
    CRs are APPLIED. Caller is responsible for verifying that gate;
    we just record the transition.
  * :func:`abandon` — SCAFFOLDED|IMPLEMENTING → ABANDONED.
  * :func:`expire` — PROPOSED → TIMEOUT (cron-driven; rare path).

Every transition saves the request and appends an audit entry via
:mod:`store`. Illegal transitions raise :class:`InvalidTransition`.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from app.architecture_requests import store
from app.architecture_requests.models import (
    ArchitectureRequest,
    ArchStatus,
    DecisionSource,
    FileSpec,
    IntegrationPoint,
)
from app.architecture_requests.validator import validate

logger = logging.getLogger(__name__)


class InvalidTransition(RuntimeError):
    """Raised when a state transition would violate the state machine."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_status(
    req: ArchitectureRequest,
    expected: set[ArchStatus],
    transition: str,
) -> None:
    if req.status not in expected:
        raise InvalidTransition(
            f"cannot {transition}: request {req.id} is in status "
            f"{req.status.value}; expected one of "
            f"{sorted(s.value for s in expected)}"
        )


# ── Public API ──────────────────────────────────────────────────────


def create_request(
    *,
    requestor: str,
    intent: str,
    motivation: str,
    package_path: str,
    file_layout: list[FileSpec],
    integration_points: list[IntegrationPoint],
    env_switches: dict[str, str],
    test_plan: str,
) -> ArchitectureRequest:
    """Validate the proposal and persist as PROPOSED (or TIER_IMMUTABLE_REFUSED)."""
    req = ArchitectureRequest(
        id=str(uuid.uuid4()),
        created_at=_now_iso(),
        requestor=requestor,
        intent=intent,
        motivation=motivation,
        package_path=package_path,
        file_layout=list(file_layout),
        integration_points=list(integration_points),
        env_switches=dict(env_switches),
        test_plan=test_plan,
        status=ArchStatus.PROPOSED,
    )
    result = validate(req)
    if not result.ok:
        if result.is_tier_immutable:
            req.status = ArchStatus.TIER_IMMUTABLE_REFUSED
            req.decision_reason = "; ".join(result.errors)
            req.decided_at = _now_iso()
            store.save(req, audit_event="tier_immutable_refused")
        else:
            req.status = ArchStatus.REJECTED
            req.decision_reason = "; ".join(result.errors)
            req.decided_at = _now_iso()
            store.save(req, audit_event="validation_rejected")
        return req

    store.save(req, audit_event="created")
    return req


def approve(
    request_id: str,
    *,
    source: DecisionSource,
    decision_reason: str | None = None,
) -> ArchitectureRequest:
    req = _get_or_raise(request_id)
    if req.status is ArchStatus.APPROVED:
        return req
    _require_status(req, {ArchStatus.PROPOSED}, "approve")
    req.status = ArchStatus.APPROVED
    req.decided_at = _now_iso()
    req.decided_by = source
    req.decision_reason = decision_reason
    store.save(req, audit_event="approved")
    return req


def reject(
    request_id: str,
    *,
    source: DecisionSource,
    decision_reason: str | None = None,
) -> ArchitectureRequest:
    req = _get_or_raise(request_id)
    _require_status(req, {ArchStatus.PROPOSED}, "reject")
    req.status = ArchStatus.REJECTED
    req.decided_at = _now_iso()
    req.decided_by = source
    req.decision_reason = decision_reason
    store.save(req, audit_event="rejected")
    return req


def scaffold(request_id: str, scaffold_dir: str) -> ArchitectureRequest:
    """Mark the request SCAFFOLDED and record the staging directory.

    The actual file writing happens in :mod:`scaffolder` — this
    transition is the *record* of that having been done. Caller is
    responsible for invoking the scaffolder before this transition.
    """
    req = _get_or_raise(request_id)
    _require_status(req, {ArchStatus.APPROVED}, "scaffold")
    req.status = ArchStatus.SCAFFOLDED
    req.scaffolded_at = _now_iso()
    req.scaffold_dir = scaffold_dir
    store.save(req, audit_event="scaffolded")
    return req


def record_child_change_request(
    request_id: str, child_cr_id: str,
) -> ArchitectureRequest:
    req = _get_or_raise(request_id)
    _require_status(
        req,
        {ArchStatus.SCAFFOLDED, ArchStatus.IMPLEMENTING},
        "record child CR",
    )
    if child_cr_id not in req.child_change_request_ids:
        req.child_change_request_ids.append(child_cr_id)
    if req.status is ArchStatus.SCAFFOLDED:
        req.status = ArchStatus.IMPLEMENTING
        store.save(req, audit_event="implementing_started")
    else:
        store.save(req, audit_event="child_cr_added")
    return req


def mark_complete(request_id: str) -> ArchitectureRequest:
    req = _get_or_raise(request_id)
    _require_status(req, {ArchStatus.IMPLEMENTING}, "mark_complete")
    req.status = ArchStatus.COMPLETED
    req.completed_at = _now_iso()
    store.save(req, audit_event="completed")
    return req


def abandon(request_id: str, reason: str) -> ArchitectureRequest:
    req = _get_or_raise(request_id)
    _require_status(
        req,
        {ArchStatus.SCAFFOLDED, ArchStatus.IMPLEMENTING},
        "abandon",
    )
    req.status = ArchStatus.ABANDONED
    req.abandoned_at = _now_iso()
    req.abandon_reason = reason
    store.save(req, audit_event="abandoned")
    return req


def expire(request_id: str) -> ArchitectureRequest:
    req = _get_or_raise(request_id)
    _require_status(req, {ArchStatus.PROPOSED}, "expire")
    req.status = ArchStatus.TIMEOUT
    req.decided_at = _now_iso()
    req.decided_by = DecisionSource.TIMEOUT
    store.save(req, audit_event="timed_out")
    return req


def _get_or_raise(request_id: str) -> ArchitectureRequest:
    req = store.get(request_id)
    if req is None:
        raise KeyError(f"unknown architecture-request {request_id!r}")
    return req
