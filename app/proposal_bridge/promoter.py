"""Promoter daemon — drives the proposal-bridge state machine.

Walks ``workspace/proposal_bridge/<source>/`` daily and:

  1. STAGED proposals past their cooldown → file a CR via the
     existing ``app.change_requests`` gate, transition to CR_FILED.
     Rate-limited (``_MAX_PROMOTIONS_PER_PASS``) so a backlog burns
     down over multiple days, not a single Signal flood.
  2. CR_FILED proposals → reconcile against the change-request
     store. APPLIED CR → APPLIED proposal. REJECTED / TIMEOUT /
     APPLY_FAILED / TIER_IMMUTABLE_REFUSED CR → REJECTED proposal.
  3. STAGED proposals beyond ``_MAX_AGE_DAYS`` (and never promoted —
     usually because the rate limit kept skipping them) → EXPIRED.
  4. APPLIED / REJECTED / EXPIRED proposals past
     ``_AUDIT_RETENTION_DAYS`` → workspace artefacts removed. The
     change-request audit chain is the durable record; the bridge's
     local files are housekeeping.

Daemon discipline mirrors ``app.healing.monitors`` and
``app.self_improvement.capability_gap_analyzer``:
  * eager start at import time, env-gated
  * idempotent restart, watchdog-friendly
  * each pass NEVER raises into the runtime
"""
from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from app.proposal_bridge.store import (
    ProposalState,
    ProposalStatus,
    cleanup_resolved,
    iter_proposals,
    read_body,
    update_proposal,
)

logger = logging.getLogger(__name__)


_DAEMON_THREAD_NAME = "proposal-bridge-promoter"
_WARMUP_S = 90
_POLL_INTERVAL_S = 24 * 3600  # daily

_MAX_PROMOTIONS_PER_PASS = 3
_MAX_AGE_DAYS = 30           # STAGED → EXPIRED beyond this
_AUDIT_RETENTION_DAYS = 14   # cleanup window for terminal proposals

_driver_lock = threading.Lock()
_driver_started = False
_stop_event = threading.Event()


def _enabled() -> bool:
    return os.getenv("PROPOSAL_BRIDGE_ENABLED", "true").lower() in (
        "true", "1", "yes", "on",
    )


def _is_running() -> bool:
    return any(
        t.name == _DAEMON_THREAD_NAME and t.is_alive()
        for t in threading.enumerate()
    )


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


# ── State transitions ───────────────────────────────────────────────────


def _stage_to_cr(state: ProposalState) -> Optional[str]:
    """Try to promote a STAGED proposal to CR_FILED. Returns the cr_id
    on success, None when the CR system is unavailable or rejects the
    proposal at validation time.

    Validation rejection at the CR gate is treated as REJECTED (a
    terminal "system said no" decision). This mirrors operator
    rejection: the producer's signature-based dedup will refuse to
    re-stage the same content, breaking what would otherwise be a
    producer/promoter loop on permanent failures (TIER_IMMUTABLE,
    blocked paths). If the underlying validator policy changes, the
    operator can manually clear the workspace artefact and the
    producer will re-stage fresh.

    EXPIRED is reserved for non-terminal misses: missing body,
    cooldown-elapsed-but-rate-limited (recoverable next pass), or
    long-staged-without-promotion.
    """
    body = read_body(state)
    if not body:
        # Body was deleted out from under us. Treat as recoverable —
        # the producer can re-stage on next pass.
        state.status = ProposalStatus.EXPIRED
        state.expired_at = _now().isoformat()
        state.notes["expire_reason"] = "body_missing_at_promotion"
        update_proposal(state)
        return None

    try:
        from app.change_requests import (
            Status,
            create_request,
            send_ask,
        )
    except Exception:
        logger.debug("proposal_bridge: change_requests import failed",
                     exc_info=True)
        return None

    # Q2 §39: append the coding-session spec to the CR body for
    # non-Tier-3 targets. Tier-3 paths route through
    # ``governance_amendment.protocol`` (different apply path)
    # and don't use coding sessions, so the spec is suppressed there.
    body_with_spec = _augment_body_with_spec(body, state)

    reason = (
        f"Proposal-bridge promotion of staged {state.source} draft "
        f"`{state.signature}` after {state.cooldown_days}-day cooldown. "
        f"Title: {state.title}. Approving lands the markdown at "
        f"`{state.target_path}` as a permanent paper trail; rejecting "
        f"discards the proposal."
    )
    try:
        cr = create_request(
            requestor=f"proposal_bridge:{state.source}",
            path=state.target_path,
            new_content=body_with_spec,
            old_content="",
            reason=reason,
        )
    except Exception:
        logger.warning(
            "proposal_bridge: create_request failed for %s/%s",
            state.source, state.signature, exc_info=True,
        )
        return None

    if cr.status != Status.PENDING:
        # Validator-side terminal decision (TIER_IMMUTABLE_REFUSED,
        # REJECTED, etc.). Mirror as a terminal REJECTED on our side
        # so the producer's signature-dedup prevents re-staging the
        # same content next pass.
        state.status = ProposalStatus.REJECTED
        state.resolved_at = _now().isoformat()
        state.notes["rejection_layer"] = "cr_validator"
        state.notes["cr_status"] = cr.status.value
        if cr.decision_reason:
            state.notes["cr_decision_reason"] = cr.decision_reason[:300]
        update_proposal(state)
        logger.info(
            "proposal_bridge: validator rejected %s/%s — status=%s",
            state.source, state.signature, cr.status.value,
        )
        return None

    state.status = ProposalStatus.CR_FILED
    state.cr_id = cr.id
    state.cr_filed_at = _now().isoformat()
    update_proposal(state)

    # Send Signal ASK best-effort. Failure is non-fatal — the CR is
    # already in /cp/changes and operator can approve from React.
    try:
        send_ask(cr.id)
    except Exception:
        logger.debug("proposal_bridge: send_ask failed", exc_info=True)

    return cr.id


def _reconcile_cr_filed(state: ProposalState) -> None:
    """Walk a CR_FILED proposal forward based on the CR's current status."""
    if not state.cr_id:
        return
    try:
        from app.change_requests import Status, get
    except Exception:
        return
    cr = get(state.cr_id)
    if cr is None:
        # CR vanished — treat as REJECTED so the proposal terminates.
        state.status = ProposalStatus.REJECTED
        state.resolved_at = _now().isoformat()
        state.notes["cr_status"] = "missing"
        update_proposal(state)
        return

    if cr.status == Status.APPLIED:
        state.status = ProposalStatus.APPLIED
        state.resolved_at = _now().isoformat()
        state.notes["cr_status"] = cr.status.value
        if cr.pr_url:
            state.notes["pr_url"] = cr.pr_url
        update_proposal(state)
    elif cr.status in (
        Status.REJECTED,
        Status.TIMEOUT,
        Status.APPLY_FAILED,
        Status.ROLLED_BACK,
        Status.TIER_IMMUTABLE_REFUSED,
    ):
        state.status = ProposalStatus.REJECTED
        state.resolved_at = _now().isoformat()
        state.notes["cr_status"] = cr.status.value
        if cr.decision_reason:
            state.notes["cr_decision_reason"] = cr.decision_reason[:300]
        update_proposal(state)
    # else: still pending / approved-not-yet-applied — leave as-is


def _maybe_cleanup_terminal(state: ProposalState, now: datetime) -> bool:
    """Remove workspace artefacts for terminal proposals past the
    audit retention window. Returns True on cleanup."""
    if state.status not in (
        ProposalStatus.APPLIED,
        ProposalStatus.REJECTED,
        ProposalStatus.EXPIRED,
    ):
        return False
    resolution_ts = (
        _parse_iso(state.resolved_at)
        or _parse_iso(state.expired_at)
    )
    if resolution_ts is None:
        return False
    if now - resolution_ts < timedelta(days=_AUDIT_RETENTION_DAYS):
        return False
    cleanup_resolved(state)
    return True


def _maybe_expire_stale(state: ProposalState, now: datetime) -> bool:
    """STAGED proposals beyond _MAX_AGE_DAYS expire."""
    if state.status != ProposalStatus.STAGED:
        return False
    staged_at = _parse_iso(state.staged_at)
    if staged_at is None:
        return False
    if now - staged_at < timedelta(days=_MAX_AGE_DAYS):
        return False
    state.status = ProposalStatus.EXPIRED
    state.expired_at = now.isoformat()
    state.notes.setdefault("expire_reason", "stale_in_staging")
    update_proposal(state)
    return True


def _is_promotable(state: ProposalState, now: datetime) -> bool:
    """Cooldown elapsed and not yet promoted."""
    if state.status != ProposalStatus.STAGED:
        return False
    staged_at = _parse_iso(state.staged_at)
    if staged_at is None:
        return False
    cooldown = timedelta(days=state.cooldown_days)
    return (now - staged_at) >= cooldown


# ── Pass orchestration ───────────────────────────────────────────────────


def run_one_pass() -> dict[str, Any]:
    """One promoter pass over the staging tree. Returns counters."""
    if not _enabled():
        return {"status": "disabled"}

    now = _now()
    counters: dict[str, int] = {
        "seen": 0,
        "promoted_to_cr": 0,
        "reconciled_applied": 0,
        "reconciled_rejected": 0,
        "validator_rejected": 0,
        "expired_stale": 0,
        "cleaned_up": 0,
    }
    promotions_this_pass = 0

    for state in iter_proposals():
        counters["seen"] += 1

        # 1. Cleanup terminal records past retention.
        if _maybe_cleanup_terminal(state, now):
            counters["cleaned_up"] += 1
            continue

        # 2. CR_FILED → reconcile against CR store.
        if state.status == ProposalStatus.CR_FILED:
            prior = state.status
            _reconcile_cr_filed(state)
            if state.status == ProposalStatus.APPLIED and prior != ProposalStatus.APPLIED:
                counters["reconciled_applied"] += 1
            elif state.status == ProposalStatus.REJECTED and prior != ProposalStatus.REJECTED:
                counters["reconciled_rejected"] += 1
            continue

        # 3. STAGED → expire if too old, else maybe promote (rate-limited).
        if state.status == ProposalStatus.STAGED:
            if _maybe_expire_stale(state, now):
                counters["expired_stale"] += 1
                continue
            if not _is_promotable(state, now):
                continue
            if promotions_this_pass >= _MAX_PROMOTIONS_PER_PASS:
                continue
            cr_id = _stage_to_cr(state)
            if cr_id is not None:
                counters["promoted_to_cr"] += 1
                promotions_this_pass += 1
            elif state.status == ProposalStatus.REJECTED:
                counters["validator_rejected"] += 1
            # else: transient failure (CR system unavailable) — try
            # again next pass; do NOT count toward the rate limit.

    counters["status"] = "ok"
    _publish_outcome(counters)
    return counters


# ── SubIA Global Workspace publish ───────────────────────────────────────


# Per-bucket salience weights. Higher = more consequential signal for
# the Global Workspace ignition gate.
#   * Operator decisions (applied/rejected) are the highest-signal
#     events the bridge can witness — somebody made a call.
#   * Promotions are the bridge's own active contribution.
#   * Validator rejections indicate the system rejected its own
#     proposal — useful for self-monitoring.
#   * Stale expiry + cleanup are housekeeping; low salience.
_SALIENCE_WEIGHTS: dict[str, float] = {
    "reconciled_applied": 0.35,
    "reconciled_rejected": 0.30,
    "promoted_to_cr": 0.25,
    "validator_rejected": 0.15,
    "expired_stale": 0.05,
    "cleaned_up": 0.02,
}
_SALIENCE_FLOOR = 0.20  # base salience when any significant event fires
_SALIENCE_CEILING = 0.70  # never crowd out higher-priority channels


def _publish_outcome(counters: dict[str, Any]) -> None:
    """Publish a one-line outcome summary to the SubIA Global Workspace.

    Skipped silently when nothing material happened (every count zero
    means a quiet pass — no signal worth competing for ignition).
    Salience aggregates per-bucket weights so reconciliation-only
    days (the bridge witnessed an operator decision but didn't
    promote anything itself) still publish. Failures are non-fatal —
    see ``app.workspace_publish`` for the defensive pattern.
    """
    weighted = sum(
        weight * int(counters.get(key, 0) or 0)
        for key, weight in _SALIENCE_WEIGHTS.items()
    )
    if weighted <= 0:
        return  # quiet pass — nothing competing for ignition

    salience = min(_SALIENCE_FLOOR + weighted, _SALIENCE_CEILING)
    promoted = int(counters.get("promoted_to_cr", 0) or 0)
    applied = int(counters.get("reconciled_applied", 0) or 0)
    rejected = int(counters.get("reconciled_rejected", 0) or 0)
    validator = int(counters.get("validator_rejected", 0) or 0)
    stale = int(counters.get("expired_stale", 0) or 0)
    cleaned = int(counters.get("cleaned_up", 0) or 0)
    summary = (
        f"proposal_bridge: {promoted} promoted, {applied} applied, "
        f"{rejected} rejected, {validator} validator-rejected, "
        f"{stale} expired-stale, {cleaned} cleaned"
    )
    logger.info(summary)
    try:
        from app.workspace_publish import publish_to_workspace
        publish_to_workspace(
            source="proposal-bridge",
            content=summary,
            salience=salience,
            signal_type="disposition",
        )
    except Exception:
        logger.debug("proposal_bridge: GW publish failed", exc_info=True)


# ── Daemon driver ────────────────────────────────────────────────────────


def _driver() -> None:
    if _stop_event.wait(_WARMUP_S):
        return
    while not _stop_event.is_set():
        try:
            run_one_pass()
        except Exception:
            logger.debug("proposal_bridge.promoter: pass raised", exc_info=True)
        if _stop_event.wait(_POLL_INTERVAL_S):
            return


def start() -> None:
    """Idempotent daemon launch."""
    global _driver_started
    if not _enabled():
        logger.info("proposal_bridge.promoter: disabled via PROPOSAL_BRIDGE_ENABLED")
        return
    with _driver_lock:
        if _is_running():
            return
        if _driver_started:
            logger.warning(
                "proposal_bridge.promoter: previous thread is dead, re-spawning",
            )
        _stop_event.clear()
        thread = threading.Thread(
            target=_driver, name=_DAEMON_THREAD_NAME, daemon=True,
        )
        thread.start()
        _driver_started = True
        logger.info(
            "proposal_bridge.promoter: daemon started "
            "(warm-up=%ds, poll=%dh, max-promotions/pass=%d)",
            _WARMUP_S, _POLL_INTERVAL_S // 3600, _MAX_PROMOTIONS_PER_PASS,
        )


def stop() -> None:
    _stop_event.set()


# Eager start at import — same pattern as healing/monitors and
# capability_gap_analyzer.
start()


# ── Coding-session spec rendering (Q2 §39) ───────────────────────────


def _augment_body_with_spec(body: str, state: ProposalState) -> str:
    """Append a "Coding-session spec" markdown section to the proposal
    body when applicable.

    Spec is added when ALL of:
      * ``state.coding_session_spec`` is a non-empty dict
      * ``state.target_path`` is NOT in TIER_IMMUTABLE (Tier-3 paths
        route through governance_amendment, never coding sessions)

    Spec is rendered as YAML in a fenced block — both human- and
    machine-readable. An agent reading the resulting CR (or an
    operator copy-pasting into a chat with the coder) can act on the
    scaffold directly.
    """
    spec = state.coding_session_spec
    if not isinstance(spec, dict) or not spec:
        return body
    if _path_is_tier_immutable(state.target_path):
        return body
    return body + "\n\n" + _render_spec_section(spec)


def _path_is_tier_immutable(path: str) -> bool:
    """Best-effort TIER_IMMUTABLE check. Failure → fail-safe (treat
    as immutable so we don't accidentally render a coding-session
    spec for a protected path)."""
    try:
        from app.auto_deployer import TIER_IMMUTABLE
        return path in TIER_IMMUTABLE
    except Exception:
        return True


def _render_spec_section(spec: dict[str, Any]) -> str:
    """Render the spec dict as a markdown section with a YAML fenced
    block. JSON inside a ```yaml fence is valid YAML (JSON is a YAML
    subset for our shapes), so we avoid a PyYAML dep."""
    intent = spec.get("intent") or "(no intent)"
    files = spec.get("files") or []
    acceptance = spec.get("acceptance") or []
    duration = spec.get("expected_duration_min")

    lines = [
        "---",
        "",
        "## Coding-session spec (non-Tier-3)",
        "",
        "An agent (or operator copy-pasting into a chat with the coder) ",
        "can use this scaffold to actually implement the proposal:",
        "",
        "```yaml",
        f"intent: {intent}",
    ]
    if files:
        lines.append("files:")
        for f in files:
            if not isinstance(f, dict):
                continue
            lines.append(f"  - path: {f.get('path', '?')}")
            if "action" in f:
                lines.append(f"    action: {f['action']}")
            if "size_estimate" in f:
                lines.append(f"    size_estimate: {f['size_estimate']}")
            if "purpose" in f:
                lines.append(f"    purpose: {f['purpose']}")
    if acceptance:
        lines.append("acceptance:")
        for a in acceptance:
            lines.append(f"  - {a}")
    if duration is not None:
        try:
            lines.append(f"expected_duration_min: {int(duration)}")
        except (TypeError, ValueError):
            pass
    lines.append("```")
    lines.append("")
    lines.append(
        "To execute: spawn a coding-session targeting the files "
        "above, run the acceptance commands until green, then "
        "`coding_session_submit` to fan out one CR per touched "
        "file through the standard /cp/changes operator gate."
    )
    return "\n".join(lines)
