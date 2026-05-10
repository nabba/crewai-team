"""Proposal bridge — unify observational draft → CR closure.

Three subsystems generate "proposal markdown" today and dead-end at
``docs/proposed_*/<sig>.md`` waiting for an operator to file CRs by
hand:

  * ``app.self_improvement.capability_gap_analyzer`` —
    learning-gap clusters
  * ``app.library_radar.proposer`` — tech-radar library adoptions
  * ``app.episteme.paper_pipeline`` — arXiv paper-to-experiment ideas

Each writes drafts the operator must read and manually translate into
change requests. The closure is unreliable: drafts pile up, the
operator never sees them, the loop never closes.

This package provides a single staging surface used by all three:

    proposal_bridge.stage(
        source="capability_gap" | "library_radar" | "paper_pipeline",
        signature=...,           # caller-supplied stable hash
        title=...,
        body_markdown=...,
        target_path="docs/proposed_<source>/<sig>.md",
        cooldown_days=7,
    )

Behaviour:

  1. Stage to ``workspace/proposal_bridge/<source>/<sig>.{md,json}``.
     Idempotent on signature; same body re-stages as a no-op,
     different body bumps the cooldown clock.
  2. After ``cooldown_days`` of stable existence, the promoter
     daemon files a CR with the markdown body landing at
     ``target_path``. Operator approval gate stays — the CR goes
     through Signal 👍 / ``/cp/changes`` like any other.
  3. Filed CRs are tracked through the change-request lifecycle
     (PENDING → APPROVED → APPLIED, or REJECTED / TIMEOUT). Once
     resolved, the workspace artefact is cleaned up after a 14-day
     audit window.

Promotion is rate-limited (default 3/pass) so a backlog of staged
drafts spreads over multiple days instead of flooding the operator
on a single Signal-ASK burst.

Master switch: ``PROPOSAL_BRIDGE_ENABLED`` (default ``true``).
"""
from app.proposal_bridge.store import (
    ProposalState,
    ProposalStatus,
    get_proposal,
    list_proposals,
    read_body,
    stage,
)
from app.proposal_bridge.promoter import (
    run_one_pass,
    start,
    stop,
)

__all__ = [
    "ProposalState",
    "ProposalStatus",
    "stage",
    "list_proposals",
    "get_proposal",
    "read_body",
    "run_one_pass",
    "start",
    "stop",
]
