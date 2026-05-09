"""Operator-controlled governance-threshold ratchet (Wave 3 #6).

Lets the operator raise (and, with typed-phrase confirmation, lower)
``SAFETY_MINIMUM`` and ``QUALITY_MINIMUM`` from
``app/governance.py`` over time as the system earns trust.

Public API::

    from app.governance_ratchet import (
        effective_value, get_state, list_thresholds,
        set_ratchet, relax_ratchet,
        ThresholdName, Direction,
        MonotonicViolation, FloorViolation, CeilingViolation,
        UnknownThresholdViolation,
        verify_audit_chain,
    )

The TIER_IMMUTABLE ``governance.py`` reads ``effective_value(name)``
to get the floor that ``evaluate_promotion`` enforces. The hardcoded
``*_FLOOR`` constants in ``governance.py`` are inviolable — even a
corrupted state file or a maliciously-edited JSON can't push the
effective threshold below the floor.

V1 is operator-only. Agents have NO mutating API path. Mutation
flows through the React ``/cp/settings`` page →
``/config/governance_ratchet/{set,relax}`` →
``set_ratchet`` / ``relax_ratchet``.
"""
from app.governance_ratchet._state import (
    CeilingViolation,
    Direction,
    FloorViolation,
    MonotonicViolation,
    RatchetEntry,
    RatchetViolation,
    ThresholdName,
    ThresholdState,
    UnknownThresholdViolation,
)
from app.governance_ratchet.audit import verify_chain as verify_audit_chain
from app.governance_ratchet.protocol import (
    effective_value,
    get_state,
    list_thresholds,
    relax_ratchet,
    set_ratchet,
)

__all__ = [
    # state types
    "ThresholdName", "ThresholdState", "RatchetEntry", "Direction",
    # exceptions
    "RatchetViolation", "MonotonicViolation", "FloorViolation",
    "CeilingViolation", "UnknownThresholdViolation",
    # protocol
    "effective_value", "get_state", "list_thresholds",
    "set_ratchet", "relax_ratchet",
    # audit
    "verify_audit_chain",
]
