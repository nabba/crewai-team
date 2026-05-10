"""Lifecycle orchestration for change requests.

Public entry points (one per state transition):

  * ``create_request(...)`` — agent calls via the
    ``request_restricted_write`` tool. Validates; persists as
    PENDING (or TIER_IMMUTABLE_REFUSED); returns the request id.
    Accepts an optional ``risk_class=AUTO_APPLY`` to route through
    the auto-apply pathway (operator gate skipped).
  * ``send_signal_ask(request_id)`` — sends the ASK message to the
    Signal owner with diff + 👍/👎 prompt. Records the message
    timestamp on the request for reaction-correlation.
  * ``approve(request_id, *, source, decision_reason=None)`` —
    transition PENDING → APPROVED (idempotent if already APPROVED).
    Triggers ``apply``.
  * ``reject(request_id, *, source, decision_reason=None)`` —
    transition PENDING → REJECTED.
  * ``auto_approve(request_id)`` — AUTO_APPLY pathway:
    PENDING → APPROVED with ``decided_by=SELF_HEAL_AUTO_APPLY`` then
    immediately calls ``apply_change``. Loud Signal notification on
    apply; auto-revert watcher registers for the originating error
    pattern.
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
from app.change_requests.models import (
    ChangeRequest,
    DecisionSource,
    RiskClass,
    Status,
)
from app.change_requests.validator import (
    ValidationResult,
    validate,
    validate_auto_apply,
)

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
    risk_class: RiskClass = RiskClass.STANDARD,
    origin_pattern_signature: str | None = None,
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
        risk_class: ``STANDARD`` (default — operator-gated) or
            ``AUTO_APPLY`` (skips operator gate after passing the
            strict auto-apply validator). When ``AUTO_APPLY`` fails
            its validator, the CR is gracefully downgraded to
            ``STANDARD`` rather than rejected — the operator gate
            then handles it normally.
        origin_pattern_signature: error_monitor signature that
            triggered this CR. Required for auto-apply CRs so the
            auto-revert watcher knows which pattern to monitor for
            recurrence post-apply. Ignored for STANDARD CRs.

    Returns:
        The persisted ChangeRequest. Status is either PENDING (call
        ``send_signal_ask`` next) or TIER_IMMUTABLE_REFUSED (terminal).
        For ``AUTO_APPLY`` CRs that pass validation, the caller should
        call ``auto_approve(request_id)`` next.
    """
    request_id = uuid.uuid4().hex[:12]
    diff = _compute_diff(path, old_content, new_content)

    # Risk-class gate: AUTO_APPLY uses the strict validator. On
    # failure, DOWNGRADE to STANDARD rather than reject — the
    # operator gate is the right fallback when auto-apply criteria
    # aren't met.
    if risk_class == RiskClass.AUTO_APPLY:
        auto_result = validate_auto_apply(
            path=path, new_content=new_content,
            old_content=old_content, requestor=requestor,
        )
        if not auto_result.ok:
            logger.info(
                "change_requests: auto_apply downgraded to STANDARD for "
                "%s by %s — %s",
                path, requestor, auto_result.reason,
            )
            risk_class = RiskClass.STANDARD

    # Standard validation runs in BOTH risk-class paths. AUTO_APPLY
    # already ran it (validate_auto_apply calls validate first); for
    # STANDARD this is the canonical path.
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
        risk_class=risk_class,
        origin_pattern_signature=(
            origin_pattern_signature
            if risk_class == RiskClass.AUTO_APPLY
            else None
        ),
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


# ── Auto-apply pathway ──────────────────────────────────────────────


# Per-pattern rate limit for auto-applies. ``per_pattern_per_day``
# caps how many auto-applies can fire for the same originating
# pattern in a UTC day; ``global_per_day`` caps total across all
# patterns. Exceeding either limit downgrades the CR to STANDARD.
_AUTO_APPLY_RATE_LIMITS = {
    "global_per_day": 10,
    "per_pattern_per_day": 3,
}


def _auto_apply_rate_limit_ok(origin_pattern_signature: str | None) -> tuple[bool, str | None]:
    """Check whether a fresh auto-apply is allowed under the rate
    limit. Returns (ok, reason_if_blocked).

    Counts auto-applied CRs in the current UTC day from the audit
    log. Failure to read the audit log degrades to "ok" — we don't
    want a transient audit-read failure to silently block self-heal.
    """
    today = datetime.now(timezone.utc).date().isoformat()
    try:
        all_crs = store.list_all(limit=200)
    except Exception:
        return True, None

    global_count = 0
    pattern_count = 0
    for cr in all_crs:
        if cr.risk_class != RiskClass.AUTO_APPLY:
            continue
        if cr.decided_by != DecisionSource.SELF_HEAL_AUTO_APPLY:
            continue
        if not cr.decided_at:
            continue
        if not cr.decided_at.startswith(today):
            continue
        global_count += 1
        if (origin_pattern_signature
                and cr.origin_pattern_signature == origin_pattern_signature):
            pattern_count += 1

    if global_count >= _AUTO_APPLY_RATE_LIMITS["global_per_day"]:
        return False, (
            f"global rate limit reached "
            f"({global_count}/{_AUTO_APPLY_RATE_LIMITS['global_per_day']} "
            f"auto-applies today)"
        )
    if (origin_pattern_signature
            and pattern_count >= _AUTO_APPLY_RATE_LIMITS["per_pattern_per_day"]):
        return False, (
            f"per-pattern rate limit reached for "
            f"{origin_pattern_signature!r} "
            f"({pattern_count}/{_AUTO_APPLY_RATE_LIMITS['per_pattern_per_day']})"
        )
    return True, None


def _publish_auto_apply_event(cr: ChangeRequest, *, salience: float, summary: str) -> None:
    """Publish an auto-apply transition to the SubIA Global Workspace.

    Best-effort: failures degrade silently — see
    ``app.workspace_publish`` for the defensive pattern. Salience is
    caller-tuned by transition stage (apply: 0.6; auto-revert: 0.7).
    """
    try:
        from app.workspace_publish import publish_to_workspace
        publish_to_workspace(
            source="change-requests-auto-apply",
            content=summary,
            salience=salience,
            signal_type="disposition",
        )
    except Exception:
        logger.debug("auto_apply: GW publish failed", exc_info=True)


def auto_approve(request_id: str) -> ChangeRequest:
    """Bypass-the-operator-gate pathway for AUTO_APPLY CRs.

    State transition: PENDING → APPROVED with
    ``decided_by=SELF_HEAL_AUTO_APPLY``. The audit log records this
    as ``auto-approved`` (distinct from the ``approved`` event used
    for human approvals) so retrospective queries can cleanly
    separate auto from human decisions.

    Side effects:
      * Loud Signal alert (the operator should always know an
        auto-apply landed).
      * Calls ``apply_change`` synchronously — caller's thread does
        the host_bridge write + git ops.
      * Registers the CR with the auto-revert watcher so a
        recurrence of the originating error pattern triggers
        rollback.
      * Publishes to the SubIA Global Workspace.

    Refuses (without raising) when:
      * CR isn't risk_class=AUTO_APPLY (caller used the wrong path).
      * Per-day or per-pattern rate limit is exceeded — caller
        should fall back to filing a STANDARD CR or alerting only.

    Returns the CR after the transition. Check ``cr.status`` for
    APPLIED (success) or APPLY_FAILED.
    """
    cr = store.get(request_id)
    if cr is None:
        raise KeyError(f"change_request {request_id!r} not found")
    if cr.risk_class != RiskClass.AUTO_APPLY:
        raise ValueError(
            f"auto_approve called on non-AUTO_APPLY CR {request_id!r} "
            f"(risk_class={cr.risk_class.value})"
        )
    if cr.status != Status.PENDING:
        raise ValueError(
            f"cannot auto-approve {request_id!r} in status {cr.status.value}"
        )

    ok, why = _auto_apply_rate_limit_ok(cr.origin_pattern_signature)
    if not ok:
        cr.decision_reason = (
            f"rate-limited: {why}. CR remains PENDING for normal "
            f"operator review."
        )
        store.save(cr, audit_event="auto_apply_rate_limited")
        logger.warning(
            "change_requests: auto-apply rate-limited %s — %s",
            request_id, why,
        )
        return cr

    # Approve
    cr.status = Status.APPROVED
    cr.decided_at = _now_iso()
    cr.decided_by = DecisionSource.SELF_HEAL_AUTO_APPLY
    cr.decision_reason = (
        f"auto-approved by {cr.requestor} under risk_class=AUTO_APPLY"
    )
    store.save(cr, audit_event="auto-approved")
    logger.info(
        "change_requests: auto-approved %s by %s (pattern=%s)",
        request_id, cr.requestor, cr.origin_pattern_signature,
    )

    # Loud notification — operator should always see auto-applies.
    _send_auto_apply_alert(cr)

    # Apply synchronously. apply_change reads cr.status (APPROVED),
    # writes the file, runs git, calls mark_applied → APPLIED.
    try:
        from app.change_requests.apply import apply_change
        apply_result = apply_change(request_id)
    except Exception as exc:
        logger.warning(
            "change_requests: auto-apply apply_change raised for %s: %s",
            request_id, exc,
        )
        return store.get(request_id) or cr

    # Re-load post-apply (status may now be APPLIED or APPLY_FAILED).
    cr = store.get(request_id) or cr

    if cr.status == Status.APPLIED:
        # Register with the auto-revert watcher.
        try:
            from app.change_requests import auto_revert
            auto_revert.register(
                cr_id=cr.id,
                origin_pattern_signature=cr.origin_pattern_signature or "",
                applied_at_iso=cr.applied_at or _now_iso(),
            )
        except Exception:
            logger.debug(
                "auto_apply: auto_revert.register failed", exc_info=True,
            )
        _publish_auto_apply_event(
            cr,
            salience=0.6,
            summary=(
                f"auto-applied CR {cr.id} → {cr.path} "
                f"(pattern={cr.origin_pattern_signature or 'none'})"
            ),
        )
    else:
        _publish_auto_apply_event(
            cr,
            salience=0.45,
            summary=(
                f"auto-apply FAILED for CR {cr.id} → {cr.path}; "
                f"status={cr.status.value}"
            ),
        )

    return cr


def _send_auto_apply_alert(cr: ChangeRequest) -> None:
    """Loud Signal alert for an auto-applied CR — best-effort."""
    try:
        from app.signal_client import send_message
        from app.config import get_settings
        recipient = (get_settings().signal_owner_number or "").strip()
        if not recipient:
            return
        body = (
            f"⚡ Auto-apply CR `{cr.id}` was filed by `{cr.requestor}` "
            f"and is being applied without operator approval (risk_class="
            f"AUTO_APPLY).\n\n"
            f"**Path:** `{cr.path}`\n"
            f"**Origin pattern:** `{cr.origin_pattern_signature or 'none'}`\n"
            f"**Reason:** {cr.reason[:300]}\n\n"
            f"Auto-revert is armed for 30 min — if the originating error "
            f"pattern recurs, this change will roll back automatically.\n"
            f"Manual rollback: /cp/changes/{cr.id} → Rollback"
        )
        send_message(recipient, body)
    except Exception:
        logger.debug("auto_apply: alert send failed", exc_info=True)


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
    after successful hot-apply + git operations.

    Emits a ``soul_edit`` event into the identity continuity ledger
    when the applied path matches an identity-shaping artefact
    (``app/souls/*`` or ``wiki/governance/constitution.md``), so the
    annual reflection picks up cross-amendment soul drift.
    """
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
    if _is_soul_path(cr.path):
        try:
            from app.identity.continuity_ledger import record_event
            record_event(
                kind="soul_edit",
                actor=cr.requestor or "agent",
                summary=f"edited soul artefact {cr.path}",
                detail={
                    "request_id": request_id,
                    "path": cr.path,
                    "git_commit_sha": git_commit_sha,
                },
            )
        except Exception:
            logger.debug("identity ledger emission failed", exc_info=True)
    return cr


_SOUL_PATH_PREFIXES = ("app/souls/",)
_SOUL_PATH_EXACT = frozenset({"wiki/governance/constitution.md"})


def _is_soul_path(path: str) -> bool:
    """Match a change-request path against identity-shaping artefacts."""
    if not path:
        return False
    if path in _SOUL_PATH_EXACT:
        return True
    return any(path.startswith(p) for p in _SOUL_PATH_PREFIXES)


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
