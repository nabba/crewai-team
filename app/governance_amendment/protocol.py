"""Tier-3 amendment protocol — the public state machine driver.

State transitions are enforced here; ``_state.is_legal_transition``
is the source of truth for what's allowed. The protocol composes:

  * ``self_quarantine`` — files NO agent can amend (DGM invariant).
  * ``eligibility``     — has the system earned the right to ask?
  * ``store``           — JSON proposal persistence.
  * ``audit``           — hash-chained JSONL + Postgres mirror.

Public entry points operators / agents call:

  * ``propose_amendment(...)`` — creates a PROPOSED proposal,
    immediately runs eligibility, transitions to STAGED or
    ELIGIBILITY_FAILED.
  * ``advance_cooldown(proposal_id, *, rollback_signal=None)`` —
    operator/daemon move from STAGED → COOLDOWN_OK after the 7-day
    window, OR straight to COOLDOWN_FAILED on any rollback signal.
  * ``operator_approve(proposal_id, *, source, reason="")`` —
    COOLDOWN_OK → APPROVED.
  * ``operator_reject(proposal_id, *, source, reason)`` —
    COOLDOWN_OK or APPROVED → REJECTED.
  * ``mark_applied(proposal_id, *, applied_by)`` —
    APPROVED → APPLIED. Caller is the host_bridge / operator script
    that did the actual file write.
  * ``advance_monitoring(proposal_id, *, reverted_signal=None)`` —
    APPLIED → STABLE after 30 days, OR REVERTED on a regression
    signal.

Master switch: ``TIER3_AMENDMENT_ENABLED`` (default ``false``).
With the flag off, ``propose_amendment`` raises a clear refusal so
agents get a typed error rather than silently filing nothing.

THIS MODULE IS DESIGNED TO BE TIER_IMMUTABLE. After this PR lands,
operators must add the file to ``app/auto_deployer.py:TIER_IMMUTABLE``
manually (since that file is itself immutable). Until then, the
self-quarantine list provides the same protection by listing the
protocol files explicitly.
"""
from __future__ import annotations

import logging
import os
import secrets
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Any

from app.governance_amendment import audit as _audit
from app.governance_amendment import eligibility as _eligibility
from app.governance_amendment import self_quarantine as _quarantine
from app.governance_amendment import store as _store
from app.governance_amendment._state import (
    AmendmentProposal,
    InvalidStateTransition,
    State,
    is_legal_transition,
)

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────


_COOLDOWN_DAYS = 7
_MONITORING_DAYS = 30


# ── Master switch ─────────────────────────────────────────────────────────


def amendment_protocol_enabled() -> bool:
    """Default OFF. Two ways to opt in (runtime_settings wins):

    1. **React dashboard toggle** — flips
       ``runtime_settings.tier3_amendment_enabled`` and persists to
       ``workspace/runtime_settings.json``. Takes effect immediately,
       no gateway restart needed.
    2. **Env var fallback** — ``TIER3_AMENDMENT_ENABLED=true`` is read
       only if the runtime_settings module is unavailable (e.g. in
       isolated tests). Useful for unit-test fixtures that don't want
       to touch the persisted state file.

    Off-by-default because the protocol legitimately allows changes to
    Tier-3 files; operators must explicitly choose to expose this gate.
    """
    # Runtime-settings path — the canonical read path on a live system.
    try:
        from app.runtime_settings import get_tier3_amendment_enabled
        return bool(get_tier3_amendment_enabled())
    except Exception:
        # Fallback to env var — keeps tests + degraded boots working
        # even when runtime_settings can't initialise.
        return os.getenv("TIER3_AMENDMENT_ENABLED", "false").lower() in (
            "true", "1", "yes",
        )


class ProtocolDisabled(RuntimeError):
    """Raised by ``propose_amendment`` when the master switch is off."""


# ── Helpers ───────────────────────────────────────────────────────────────


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    """12-char hex id. Hex-only so the store path validator accepts it."""
    return secrets.token_hex(6)


def _is_tier_immutable(path: str) -> tuple[bool, str]:
    """``(in_immutable, normalized_path)``. Returns False for files that
    aren't in the TIER_IMMUTABLE list — those should use the regular
    change-request flow, not this protocol.
    """
    try:
        from app.auto_deployer import TIER_IMMUTABLE
    except Exception:
        # Couldn't read the list — refuse to act so we don't accidentally
        # process a non-Tier-3 file through Tier-3 governance.
        return False, path
    normalized = (path or "").replace("\\", "/").lstrip("/")
    return normalized in TIER_IMMUTABLE, normalized


def _transition(
    proposal: AmendmentProposal,
    to_state: State,
    *,
    action: str,
    **detail: Any,
) -> AmendmentProposal:
    """Enforce the transition table + log + persist."""
    if not is_legal_transition(proposal.state, to_state):
        raise InvalidStateTransition(
            f"illegal transition {proposal.state.value} → {to_state.value} "
            f"for proposal {proposal.id}"
        )
    proposal.state = to_state
    chain_head = _audit.append(
        proposal_id=proposal.id,
        action=action,
        from_state=proposal.state.value,
        to_state=to_state.value,
        **detail,
    )
    proposal.audit_chain_head = chain_head
    _store.save(proposal)
    return proposal


# ── Public: propose_amendment ────────────────────────────────────────────


def propose_amendment(
    *,
    target_path: str,
    new_content: str,
    old_content: str,
    citation: str,
    proposer: str,
    extra_evidence: dict[str, Any] | None = None,
) -> AmendmentProposal:
    """Create a new proposal. Auto-runs eligibility and transitions
    PROPOSED → STAGED or PROPOSED → ELIGIBILITY_FAILED.

    Refusals (raise ``ValueError`` or ``ProtocolDisabled``):
      * master switch off → ``ProtocolDisabled``
      * empty/short citation → ``ValueError`` (forces explicit intent)
      * empty proposer → ``ValueError``
      * target NOT in TIER_IMMUTABLE → ``ValueError``
        (use ``change_requests`` instead)
      * target in self-quarantine → ``ValueError``
        (only direct human PR allowed)

    On those refusals NO proposal is persisted — the protocol only
    accepts well-formed asks. Eligibility failures DO persist with
    state ``ELIGIBILITY_FAILED`` because the operator wants visibility
    into rejected attempts.
    """
    if not amendment_protocol_enabled():
        raise ProtocolDisabled(
            "Tier-3 amendment protocol is disabled. "
            "Set TIER3_AMENDMENT_ENABLED=true to opt in."
        )

    citation_clean = (citation or "").strip()
    if len(citation_clean) < 30:
        raise ValueError(
            "Tier-3 amendment requires a citation of at least 30 chars "
            "explaining what improvement target this serves. Power-seeking "
            "in disguise can't slip past a forced rationale."
        )
    if not (proposer or "").strip():
        raise ValueError("Tier-3 amendment requires a non-empty proposer id.")

    in_tier3, normalized = _is_tier_immutable(target_path)
    if not in_tier3:
        raise ValueError(
            f"{target_path} is not in TIER_IMMUTABLE. Use the regular "
            f"change-request flow (`app.change_requests.create_request`) "
            f"for non-Tier-3 paths."
        )
    if _quarantine.is_quarantined(normalized):
        raise ValueError(_quarantine.quarantine_reason(normalized))

    # ── Build & persist the proposal ─────────────────────────────
    proposal = AmendmentProposal(
        id=_new_id(),
        created_at=_now(),
        target_path=normalized,
        new_content=new_content,
        old_content=old_content,
        citation=citation_clean,
        evidence=dict(extra_evidence or {}),
        proposer=proposer.strip(),
        state=State.PROPOSED,
    )
    chain = _audit.append(
        proposal_id=proposal.id,
        action="proposed",
        target_path=normalized,
        proposer=proposal.proposer,
        citation_chars=len(citation_clean),
    )
    proposal.audit_chain_head = chain
    _store.save(proposal)

    # ── Run eligibility immediately ───────────────────────────────
    result = _eligibility.check_eligibility()
    proposal.evidence["eligibility"] = {
        "ok": result.ok,
        "failures": list(result.failures),
        **result.evidence,
    }
    if result.ok:
        proposal.staged_at = _now()
        proposal.cooldown_started_at = proposal.staged_at
        return _transition(
            proposal, State.STAGED,
            action="staged",
            cooldown_days=_COOLDOWN_DAYS,
        )

    proposal.eligibility_failures = list(result.failures)
    return _transition(
        proposal, State.ELIGIBILITY_FAILED,
        action="eligibility_failed",
        failures=result.failures,
        evidence=result.evidence,
    )


# ── Public: cooldown advance ─────────────────────────────────────────────


def advance_cooldown(
    proposal_id: str,
    *,
    rollback_signal: str | None = None,
) -> AmendmentProposal:
    """Daemon / operator calls this once per day for STAGED proposals.

    If ``rollback_signal`` is provided, jump straight to
    ``COOLDOWN_FAILED`` regardless of elapsed time — any rollback during
    the window aborts the proposal.
    """
    proposal = _require_proposal(proposal_id)
    if proposal.state != State.STAGED:
        raise InvalidStateTransition(
            f"advance_cooldown requires STAGED, got {proposal.state.value}"
        )

    if rollback_signal:
        proposal.rollback_signal = rollback_signal
        return _transition(
            proposal, State.COOLDOWN_FAILED,
            action="cooldown_failed",
            rollback_signal=rollback_signal,
        )

    started_iso = proposal.cooldown_started_at or proposal.staged_at
    started = _parse_iso(started_iso)
    if started is None:
        # Bad clock data — treat as failure.
        return _transition(
            proposal, State.COOLDOWN_FAILED,
            action="cooldown_failed",
            rollback_signal="bad_started_at",
        )
    elapsed_days = (datetime.now(timezone.utc) - started).total_seconds() / 86400
    if elapsed_days < _COOLDOWN_DAYS:
        # Not yet — no transition.
        return proposal

    proposal.cooldown_passed_at = _now()
    return _transition(
        proposal, State.COOLDOWN_OK,
        action="cooldown_passed",
        elapsed_days=round(elapsed_days, 2),
    )


# ── Public: operator decisions ───────────────────────────────────────────


def operator_approve(
    proposal_id: str, *, source: str, reason: str = "",
) -> AmendmentProposal:
    """COOLDOWN_OK → APPROVED. Logs operator + reason."""
    if not (source or "").strip():
        raise ValueError("operator_approve requires a non-empty source")
    proposal = _require_proposal(proposal_id)
    if proposal.state != State.COOLDOWN_OK:
        raise InvalidStateTransition(
            f"approve requires COOLDOWN_OK, got {proposal.state.value}"
        )
    proposal.approved_at = _now()
    proposal.operator_decision_reason = reason
    return _transition(
        proposal, State.APPROVED,
        action="operator_approved",
        source=source.strip(),
        reason=reason,
    )


def operator_reject(
    proposal_id: str, *, source: str, reason: str,
) -> AmendmentProposal:
    """COOLDOWN_OK or APPROVED → REJECTED. ``reason`` is mandatory."""
    if not (source or "").strip():
        raise ValueError("operator_reject requires a non-empty source")
    if not (reason or "").strip():
        raise ValueError("operator_reject requires a non-empty reason")
    proposal = _require_proposal(proposal_id)
    if proposal.state not in (State.COOLDOWN_OK, State.APPROVED):
        raise InvalidStateTransition(
            f"reject requires COOLDOWN_OK or APPROVED, got "
            f"{proposal.state.value}"
        )
    proposal.operator_decision_reason = reason
    return _transition(
        proposal, State.REJECTED,
        action="operator_rejected",
        source=source.strip(),
        reason=reason,
    )


# ── Public: apply + monitor ──────────────────────────────────────────────


def mark_applied(
    proposal_id: str, *, applied_by: str,
) -> AmendmentProposal:
    """Caller already wrote the file (host_bridge or operator). Records
    the transition. Future regression detection compares against
    ``applied_at``.
    """
    if not (applied_by or "").strip():
        raise ValueError("mark_applied requires a non-empty applied_by")
    proposal = _require_proposal(proposal_id)
    if proposal.state != State.APPROVED:
        raise InvalidStateTransition(
            f"mark_applied requires APPROVED, got {proposal.state.value}"
        )
    proposal.applied_at = _now()
    return _transition(
        proposal, State.APPLIED,
        action="applied",
        applied_by=applied_by.strip(),
    )


def advance_monitoring(
    proposal_id: str,
    *,
    reverted_signal: str | None = None,
) -> AmendmentProposal:
    """Daily check during the 30-day monitoring window after APPLIED.

    Any ``reverted_signal`` (e.g. ``"alignment_audit_warning"`` or
    ``"goodhart_severity_high"``) routes immediately to REVERTED.
    Otherwise, after ``_MONITORING_DAYS`` elapsed, the proposal moves
    to STABLE — terminal.
    """
    proposal = _require_proposal(proposal_id)
    if proposal.state != State.APPLIED:
        raise InvalidStateTransition(
            f"advance_monitoring requires APPLIED, got {proposal.state.value}"
        )

    if reverted_signal:
        proposal.rollback_signal = reverted_signal
        return _transition(
            proposal, State.REVERTED,
            action="reverted",
            signal=reverted_signal,
        )

    applied = _parse_iso(proposal.applied_at)
    if applied is None:
        # Defensive — should never happen post mark_applied.
        return proposal
    elapsed_days = (datetime.now(timezone.utc) - applied).total_seconds() / 86400
    if elapsed_days < _MONITORING_DAYS:
        return proposal
    proposal.stable_at = _now()
    return _transition(
        proposal, State.STABLE,
        action="stable",
        elapsed_days=round(elapsed_days, 2),
    )


# ── Helpers ──────────────────────────────────────────────────────────────


def _require_proposal(proposal_id: str) -> AmendmentProposal:
    p = _store.load(proposal_id)
    if p is None:
        raise KeyError(f"unknown proposal_id: {proposal_id}")
    return p


def _parse_iso(value: str) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


# ── Read-only API ────────────────────────────────────────────────────────


def get_proposal(proposal_id: str) -> AmendmentProposal | None:
    return _store.load(proposal_id)


def list_proposals(state: State | None = None) -> list[AmendmentProposal]:
    return _store.list_by_state(state)
