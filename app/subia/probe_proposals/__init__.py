"""Sentience-probe self-design surface — PROGRAM §47 (Q12.4).

The Butlin + RSM + SK probes in :mod:`app.subia.probes` are
**TIER_IMMUTABLE**: agents cannot modify the probe code, change the
SCORECARD interpretation, or grade themselves. This is the system's
load-bearing anti-Goodhart safeguard.

But the user-spec §8.4 framing is sharp: *agents can have a voice in
their own evaluation without being able to grade themselves*. They
can observe novel functional patterns in their behaviour that the
existing 14 + 5 Butlin/RSM probes don't capture and **propose** new
indicators — through the human-gated change-request path.

This package is that proposal surface:

  * ``proposer.propose_sentience_probe(indicator_name, structure,
    proposed_measurement, justification, *, requestor)`` —
    agent-callable function that VALIDATES the proposal, then files
    a markdown DESIGN PROPOSAL CR at
    ``docs/proposed_probes/<slug>.md``.
  * Hard refusals at validate-time:
    - Cannot redefine any existing Butlin (RPT-1, RPT-2, GWT-1..4,
      HOT-1..4, AST-1, PP-1, AE-1, AE-2) or RSM (RSM-A..E) anchor.
    - Cannot reference any path under ``app/subia/probes/``.
    - Phenomenal-language linter HARD_FAIL → refused.
  * The CR target_path is ALWAYS the markdown design doc, never a
    probe code file. If the operator approves the CR, the design
    doc lands; **the actual probe implementation is a follow-on
    Tier-3 amendment** the operator (not the agent) authors.

Why two-stage (design doc → Tier-3 amendment) rather than one-stage:

  1. Agents can't write code under ``app/subia/`` — the path is
     architecturally protected. A direct probe-edit CR would be
     refused at validate-time anyway.
  2. The two-stage path keeps the SCORECARD interpretation under
     operator control. The design doc is just words; turning words
     into a measurement is a deliberate human action.
  3. Identity-ledger event ``sentience_probe_proposal`` (added with
     Q12.4) makes the proposal visible to the annual reflection's
     drift-summary — the system can see across years what probes
     it kept proposing that the operator did or didn't accept.
"""
from app.subia.probe_proposals.proposer import (
    ALL_RESERVED_ANCHORS,
    BUTLIN_ANCHORS,
    RSM_ANCHORS,
    ProbeProposalRefused,
    propose_sentience_probe,
    render_design_doc,
)

__all__ = [
    "ALL_RESERVED_ANCHORS",
    "BUTLIN_ANCHORS",
    "ProbeProposalRefused",
    "RSM_ANCHORS",
    "propose_sentience_probe",
    "render_design_doc",
]
