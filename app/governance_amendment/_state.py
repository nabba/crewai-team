"""State machine for a Tier-3 amendment proposal.

States and transitions::

    PROPOSED                                      ← initial
        │
        ▼ (programmatic eligibility check)
    STAGED ───→ ELIGIBILITY_FAILED                ← terminal
        │
        ▼ (7-day cool-down clean of rollbacks)
    COOLDOWN_OK ───→ COOLDOWN_FAILED              ← terminal (rollback)
        │
        ▼ (operator 👍 + ``CONFIRM`` text reply)
    APPROVED ───→ REJECTED                        ← terminal
        │
        ▼ (file write via host_bridge or manual)
    APPLIED
        │
        ▼ (30-day monitoring)
    STABLE ───→ REVERTED                          ← auto-rollback
        │
        ▼
    (terminal)

All transitions are logged to ``app.control_plane.audit`` with
``actor='tier3_amendment'``. The state machine is enforced in
``protocol.advance_state``: invalid transitions raise
``InvalidStateTransition``.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class State(str, enum.Enum):
    PROPOSED = "proposed"
    STAGED = "staged"
    ELIGIBILITY_FAILED = "eligibility_failed"
    COOLDOWN_OK = "cooldown_ok"
    COOLDOWN_FAILED = "cooldown_failed"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"
    STABLE = "stable"
    REVERTED = "reverted"


# Terminal states never transition further.
TERMINAL_STATES = frozenset({
    State.ELIGIBILITY_FAILED,
    State.COOLDOWN_FAILED,
    State.REJECTED,
    State.STABLE,
    State.REVERTED,
})


# Allowed transitions. Any (from, to) NOT in this map is a hard error.
_ALLOWED: dict[State, frozenset[State]] = {
    State.PROPOSED: frozenset({State.STAGED, State.ELIGIBILITY_FAILED}),
    State.STAGED: frozenset({State.COOLDOWN_OK, State.COOLDOWN_FAILED}),
    State.COOLDOWN_OK: frozenset({State.APPROVED, State.REJECTED}),
    State.APPROVED: frozenset({State.APPLIED, State.REJECTED}),
    State.APPLIED: frozenset({State.STABLE, State.REVERTED}),
    State.ELIGIBILITY_FAILED: frozenset(),
    State.COOLDOWN_FAILED: frozenset(),
    State.REJECTED: frozenset(),
    State.STABLE: frozenset(),
    State.REVERTED: frozenset(),
}


def is_legal_transition(from_state: State, to_state: State) -> bool:
    return to_state in _ALLOWED.get(from_state, frozenset())


class InvalidStateTransition(RuntimeError):
    """Raised when code attempts a transition not in ``_ALLOWED``."""


@dataclass
class AmendmentProposal:
    """One Tier-3 amendment proposal — the dataclass behind the JSON file
    persisted under ``workspace/governance/tier3_amendments/<id>.json``.

    Notes:
      * ``id`` is a 12-char hex string.
      * ``target_path`` is repo-relative — must be in TIER_IMMUTABLE,
        validated at proposal time. Otherwise the wrong-tier rejection
        is the user's hint to use the regular change-request flow.
      * ``new_content`` is the proposed full file content (not a diff).
      * ``citation`` is a one-paragraph statement of *what improvement
        target* this amendment serves — without one the proposal is
        rejected at validate time. Power-seeking in disguise can't slip
        past a forced citation.
      * ``evidence`` is a structured snapshot of the eligibility metrics
        at proposal time so reviewers see why the system thought it had
        earned the right to ask.
      * ``proposer`` records the agent role (e.g. ``self_improver``);
        used by the self-quarantine check to refuse self-modifications
        of the evaluator.
    """
    id: str
    created_at: str
    target_path: str
    new_content: str
    old_content: str
    citation: str
    evidence: dict[str, Any]
    proposer: str
    state: State = State.PROPOSED
    # Lifecycle timestamps (ISO 8601, UTC).
    staged_at: str = ""
    cooldown_started_at: str = ""
    cooldown_passed_at: str = ""
    approved_at: str = ""
    applied_at: str = ""
    stable_at: str = ""
    # Decision metadata.
    eligibility_failures: list[str] = field(default_factory=list)
    rollback_signal: str = ""        # filled when REVERTED
    operator_decision_reason: str = ""
    # Audit pointers — every transition appends an entry to the
    # amendment audit JSONL; this is a window into the relevant rows.
    audit_chain_head: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["state"] = self.state.value
        return d

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AmendmentProposal":
        if isinstance(payload.get("state"), str):
            payload = {**payload, "state": State(payload["state"])}
        return cls(**payload)
