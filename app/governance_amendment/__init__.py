"""Tier-3 amendment protocol — public surface.

Lets the system *legitimately* propose modifications to TIER_IMMUTABLE
files (governance thresholds, evaluators, etc.) after demonstrating a
clean track record. Designed for years-of-uptime: without a path to
amend the core, the system either games metrics (Goodhart) or
plateaus.

Public API::

    from app.governance_amendment import (
        propose_amendment, advance_cooldown,
        operator_approve, operator_reject,
        mark_applied, advance_monitoring,
        get_proposal, list_proposals,
        amendment_protocol_enabled, ProtocolDisabled,
        AmendmentProposal, State,
        verify_audit_chain,
    )

Master switch: ``TIER3_AMENDMENT_ENABLED`` (default ``false``).

See ``docs/TIER3_AMENDMENT.md`` for the full protocol design + the
operator handover steps (adding the new files to TIER_IMMUTABLE in
``app/auto_deployer.py``).
"""
from app.governance_amendment._state import (
    AmendmentProposal,
    InvalidStateTransition,
    State,
    is_legal_transition,
)
from app.governance_amendment.audit import verify_chain as verify_audit_chain
from app.governance_amendment.eligibility import (
    EligibilityResult,
    check_eligibility,
)
from app.governance_amendment.protocol import (
    advance_cooldown,
    advance_monitoring,
    amendment_protocol_enabled,
    get_proposal,
    list_proposals,
    mark_applied,
    operator_approve,
    operator_reject,
    propose_amendment,
    ProtocolDisabled,
)
from app.governance_amendment.self_quarantine import (
    QUARANTINED_FILES,
    is_quarantined,
)

__all__ = [
    # state
    "AmendmentProposal", "State", "InvalidStateTransition",
    "is_legal_transition",
    # protocol
    "propose_amendment", "advance_cooldown", "operator_approve",
    "operator_reject", "mark_applied", "advance_monitoring",
    "get_proposal", "list_proposals",
    "amendment_protocol_enabled", "ProtocolDisabled",
    # eligibility
    "EligibilityResult", "check_eligibility",
    # quarantine
    "QUARANTINED_FILES", "is_quarantined",
    # audit
    "verify_audit_chain",
]
