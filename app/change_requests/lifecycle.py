"""Lifecycle orchestration for change requests.

Public entry points (one per state transition):

  * ``create_request(...)`` — agent calls via the
    ``request_restricted_write`` tool. Validates; persists as
    PENDING (or TIER_IMMUTABLE_REFUSED); returns the request id.
  * ``send_signal_ask(request_id)`` — sends the ASK message to the
    Signal owner with diff + 👍/👎 prompt. Records the message
    timestamp on the request for reaction-correlation.
  * ``approve(request_id, *, source, decision_reason=None)`` —
    transition PENDING → APPROVED (idempotent if already APPROVED).
    Triggers ``apply``.
  * ``reject(request_id, *, source, decision_reason=None)`` —
    transition PENDING → REJECTED.
  * ``apply(request_id)`` — transition APPROVED → APPLIED (or
    APPLY_FAILED). Calls into ``app.change_requests.apply`` for
    the actual write + git operations.
  * ``rollback(request_id, *, operator)`` — transition APPLIED →
    ROLLED_BACK. Reverts via git + hot-revert via bridge.

Every transition saves the request + appends to the audit log.
The state machine is enforced — illegal transitions raise.
"""
from __future__ import annotations

import difflib
import logging
import uuid
from datetime import datetime, timezone

from app.change_requests import store
from app.change_requests.models import ChangeRequest, DecisionSource, Status
from app.change_requests.validator import ValidationResult, validate

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _compute_diff(path: str, old: str, new: str) -> str:
    """Unified diff for the operator review."""
    return "".join(difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        n=3,
    ))


# ── Public API ──────────────────────────────────────────────────────


def create_request(
    *,
    requestor: str,
    path: str,
    new_content: str,
    old_content: str,
    reason: str,
) -> ChangeRequest:
    """Validate the change and persist as PENDING (or
    TIER_IMMUTABLE_REFUSED if path is protected).

    Args:
        requestor: agent_id calling the tool (e.g. "coder").
        path: repo-relative target path.
        new_content: the proposed file contents.
        old_content: the current file contents (captured by the
            caller; used for diff + rollback). Empty string if the
            target file doesn't yet exist.
        reason: one-paragraph explanation for the operator.

    Returns:
        The persisted ChangeRequest. Status is either PENDING (call
        ``send_signal_ask`` next) or TIER_IMMUTABLE_REFUSED (terminal).
    """
    request_id = uuid.uuid4().hex[:12]
    diff = _compute_diff(path, old_content, new_content)

    # Validate — note this runs BEFORE persistence so an unsanctioned
    # path doesn't even leave a record (though TIER_IMMUTABLE refusal
    # IS persisted for audit).
    result: ValidationResult = validate(path=path, new_content=new_content)

    cr = ChangeRequest(
        id=request_id,
        created_at=_now_iso(),
        requestor=requestor,
        path=path,
        new_content=new_content,
        old_content=old_content,
        reason=reason,
        diff=diff,
    )

    if not result.ok:
        if result.is_tier_immutable:
            cr.status = Status.TIER_IMMUTABLE_REFUSED
            cr.decision_reason = result.reason
            store.save(cr, audit_event="tier_immutable_refused")
            logger.warning(
                "change_requests: TIER_IMMUTABLE refused %s by %s (path=%s)",
                request_id, requestor, path,
            )
        else:
            cr.status = Status.REJECTED
            cr.decision_reason = result.reason
            store.save(cr, audit_event="validation_failed")
            logger.info(
                "change_requests: validation rejected %s (reason=%s)",
                request_id, result.reason,
            )
        return cr

    # Validation passed. Phase F #5 (2026-05-09): consult the
    # rejected-hypothesis lessons KB before persisting so the
    # operator's review surface includes a "matches lesson X" banner
    # when the proposal looks like something previously rejected.
    # Best-effort and non-blocking — KB unavailable / empty is fine.
    try:
        from app.companion.lessons_learned import check_against
        proposal_text = f"{path}: {reason}"
        matches = check_against(proposal_text, top_k=1)
        if matches:
            top = matches[0]
            cr.reason = (
                f"{cr.reason}\n\n"
                f"⚠️ Matches rejected-pattern lesson `{top['id']}` "
                f"(similarity {top['similarity']:.2f}, seen {top['count']}× "
                f"before). Sample reason: {top['sample_reason'][:160]}"
            )
    except Exception:
        logger.debug("change_requests: lessons check failed", exc_info=True)

    store.save(cr, audit_event="created")
    logger.info(
        "change_requests: created %s by %s for path=%s",
        request_id, requestor, path,
    )
    return cr


def approve(
    request_id: str,
    *,
    source: DecisionSource,
    decision_reason: str | None = None,
) -> ChangeRequest:
    """Transition PENDING|APPLY_FAILED → APPROVED. Idempotent if
    already APPROVED.

    Does NOT apply — caller calls ``apply()`` next. This split lets
    the React UI override → approve → apply in three audit-trail
    steps so each is separately observable.

    Two valid entry statuses:
      * PENDING       — first-time approval (Signal 👍 or React click)
      * APPLY_FAILED  — retry path (the API endpoint
        ``/api/cp/changes/{id}/retry-apply`` and React UI's "Retry
        apply" button bring the request back to APPROVED so
        ``apply_change`` accepts it again).  2026-05-09 bug fix —
        previously this raised ValueError, breaking every retry.
    """
    cr = store.get(request_id)
    if cr is None:
        raise KeyError(f"change_request {request_id!r} not found")
    if cr.status == Status.APPROVED:
        return cr  # idempotent
    if cr.status not in (Status.PENDING, Status.APPLY_FAILED):
        raise ValueError(
            f"cannot approve {request_id!r} in status {cr.status.value}"
        )
    prior_status = cr.status
    cr.status = Status.APPROVED
    cr.decided_at = _now_iso()
    cr.decided_by = source
    if decision_reason:
        cr.decision_reason = decision_reason
    audit_event = (
        "re-approved-for-retry"
        if prior_status == Status.APPLY_FAILED
        else "approved"
    )
    store.save(cr, audit_event=audit_event)
    logger.info(
        "change_requests: %s %s by %s",
        audit_event, request_id, source.value,
    )
    return cr


def reject(
    request_id: str,
    *,
    source: DecisionSource,
    decision_reason: str | None = None,
) -> ChangeRequest:
    """Transition PENDING → REJECTED. Terminal."""
    cr = store.get(request_id)
    if cr is None:
        raise KeyError(f"change_request {request_id!r} not found")
    if cr.status == Status.REJECTED:
        return cr
    if cr.status != Status.PENDING:
        raise ValueError(
            f"cannot reject {request_id!r} in status {cr.status.value}"
        )
    cr.status = Status.REJECTED
    cr.decided_at = _now_iso()
    cr.decided_by = source
    if decision_reason:
        cr.decision_reason = decision_reason
    store.save(cr, audit_event="rejected")
    logger.info(
        "change_requests: rejected %s by %s (reason=%s)",
        request_id, source.value, decision_reason,
    )
    return cr


def mark_timeout(request_id: str) -> ChangeRequest:
    """Transition PENDING → TIMEOUT (no decision within window).
    Called by a background watchdog (not part of this PR — operator
    can manually do via React for now)."""
    cr = store.get(request_id)
    if cr is None:
        raise KeyError(f"change_request {request_id!r} not found")
    if cr.status != Status.PENDING:
        return cr
    cr.status = Status.TIMEOUT
    cr.decided_at = _now_iso()
    cr.decided_by = DecisionSource.TIMEOUT
    store.save(cr, audit_event="timeout")
    return cr


def mark_applied(
    request_id: str,
    *,
    git_branch: str,
    git_commit_sha: str,
    pr_url: str | None,
) -> ChangeRequest:
    """Transition APPROVED → APPLIED. Called by ``apply.apply_change``
    after successful hot-apply + git operations."""
    cr = store.get(request_id)
    if cr is None:
        raise KeyError(f"change_request {request_id!r} not found")
    if cr.status != Status.APPROVED:
        raise ValueError(
            f"cannot mark applied for {request_id!r} in status {cr.status.value}"
        )
    cr.status = Status.APPLIED
    cr.applied_at = _now_iso()
    cr.git_branch = git_branch
    cr.git_commit_sha = git_commit_sha
    cr.pr_url = pr_url
    store.save(cr, audit_event="applied")
    logger.info(
        "change_requests: applied %s — branch=%s sha=%s pr=%s",
        request_id, git_branch, git_commit_sha[:8] if git_commit_sha else "?",
        pr_url,
    )
    return cr


def mark_apply_failed(
    request_id: str,
    *,
    error: str,
) -> ChangeRequest:
    """Transition APPROVED → APPLY_FAILED. The change was approved but
    the actual apply (file write + git) failed. Operator can retry
    via React."""
    cr = store.get(request_id)
    if cr is None:
        raise KeyError(f"change_request {request_id!r} not found")
    if cr.status != Status.APPROVED:
        raise ValueError(
            f"cannot mark apply_failed for {request_id!r} in status {cr.status.value}"
        )
    cr.status = Status.APPLY_FAILED
    cr.apply_error = error[:1000]  # truncate
    store.save(cr, audit_event="apply_failed")
    logger.warning(
        "change_requests: apply_failed %s — %s",
        request_id, error[:200],
    )
    return cr


def mark_rolled_back(
    request_id: str,
    *,
    operator: str,
    rollback_commit_sha: str,
    rollback_pr_url: str | None = None,
) -> ChangeRequest:
    """Transition APPLIED → ROLLED_BACK."""
    cr = store.get(request_id)
    if cr is None:
        raise KeyError(f"change_request {request_id!r} not found")
    if not cr.is_rollbackable:
        raise ValueError(
            f"cannot roll back {request_id!r} in status {cr.status.value}"
        )
    cr.status = Status.ROLLED_BACK
    cr.rolled_back_at = _now_iso()
    cr.rolled_back_by = operator
    cr.rollback_commit_sha = rollback_commit_sha
    cr.rollback_pr_url = rollback_pr_url
    store.save(cr, audit_event="rolled_back")
    logger.info(
        "change_requests: rolled_back %s by %s — sha=%s",
        request_id, operator, rollback_commit_sha[:8],
    )
    return cr


def attach_signal_ts(request_id: str, signal_ts: int) -> ChangeRequest:
    """Persist the Signal message timestamp on the request. Called
    after ``send_signal_ask`` so the reaction handler can correlate."""
    cr = store.get(request_id)
    if cr is None:
        raise KeyError(f"change_request {request_id!r} not found")
    cr.signal_message_ts = signal_ts
    store.save(cr, audit_event="signal_ts_attached")
    return cr
