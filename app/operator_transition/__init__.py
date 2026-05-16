"""operator_transition — operator-absence / succession protocol (Q17.4).

Five-phase observational state machine over operator presence:
ACTIVE / ABSENT_30D / ABSENT_90D / READ_MOSTLY / TRANSITIONED.

Reads audit.log request_received timestamps to classify. Emits
Signal alert + q17_landmark ledger event on transition. Designated-
successor declaration file is operator-authored, never system-acted.
"""
from __future__ import annotations

from app.operator_transition.state import (
    OperatorPhase,
    current_phase,
    operator_active_within_days,
    record_phase_transition,
)
from app.operator_transition.successor import (
    SuccessorDeclaration,
    load_successor,
    save_successor,
)

__all__ = [
    "OperatorPhase",
    "SuccessorDeclaration",
    "current_phase",
    "load_successor",
    "operator_active_within_days",
    "record_phase_transition",
    "save_successor",
]
