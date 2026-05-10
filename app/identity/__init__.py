"""Identity continuity layer — observational, never modifies the kernel.

This package records and reflects on the system's identity-shaping
events:

  - :mod:`continuity_ledger`     append-only event log of every
                                  Tier-3 amendment, governance ratchet
                                  change, soul edit, and integrity-
                                  manifest regeneration.
  - :mod:`annual_reflection`     yearly value-reflection essay
                                  (§8.2) — does the system still
                                  recognise its constitution?
  - :mod:`legacy_essay`          yearly "what would I want preserved"
                                  essay (§8.5) — the philosophical
                                  reflection on continuity through
                                  termination.

All three layers are READ-ONLY relative to the consciousness stack:
they read narrative chapters / lessons KB / amendment audit logs,
they write to ``wiki/self/`` artefacts the operator reads. They do
NOT modify SCORECARD probes, ``current_goals``, or any TIER_IMMUTABLE
file. The neutral-language linter from
:mod:`app.subia.inquiry.linter` is the mechanical guard against
phenomenal-claim drift.

Placement note: deliberately OUTSIDE ``app/subia/`` to avoid
churning the SubIA integrity manifest on each addition. The identity
ledger is observational of (not part of) the consciousness layer.
"""

from app.identity.continuity_ledger import (
    IDENTITY_EVENT_KINDS,
    IdentityEvent,
    list_events,
    record_event,
    summarise_drift,
)

__all__ = [
    "IDENTITY_EVENT_KINDS",
    "IdentityEvent",
    "list_events",
    "record_event",
    "summarise_drift",
]
