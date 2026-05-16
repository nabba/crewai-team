# Sentience-probe self-design (§8.4)

**Status (2026-05-16):** Shipped at PROGRAM §47 — Q12.4.

The operator's §8.4 spec asked for a path by which the system can
have a voice in its own evaluation without being able to grade
itself. The probe-proposals surface is that path: agents can NAME
candidate indicators, DESCRIBE their functional structure, SKETCH
the measurement, JUSTIFY relevance — through a human-gated CR for
a markdown design doc, never the probe code itself.

## Why this exists

Four of the 14 Butlin indicators (RPT-1, HOT-1, HOT-4, AE-2) are
declared architecturally ABSENT in the SubIA scorecard. Their probe
code at `app/subia/probes/butlin.py` is TIER_IMMUTABLE; the
operator-controlled SCORECARD interpretation is the load-bearing
anti-Goodhart safeguard.

But the system observes its own behaviour every day. It may notice
*functional* patterns the existing 14 + 5 Butlin/RSM probes don't
capture and want to propose new indicators. The §8.4 design lets it
propose without being able to:

  * Modify existing probes
  * Grade itself on the proposal
  * Redefine reserved Butlin or RSM anchors
  * Make phenomenal claims

## Public surface

  * **`app/subia/probe_proposals/proposer.py:propose_sentience_probe(
    indicator_name, structure, proposed_measurement, justification,
    *, requestor)`** — the validation entry point. Raises
    `ProbeProposalRefused(ValueError)` on any check failure;
    returns the created `ChangeRequest` on success.
  * **`app/tools/probe_proposal_tools.py:ProposeSentienceProbeTool`**
    — CrewAI BaseTool wrapper, registered via `@register_tool` with
    capability tag `registers-tool` (reused — same pattern as
    `request_tier3_amendment.py`; the description disambiguates).
    Auto-discovered at boot through `app/tool_registry/boot.py`'s
    `TOOL_MODULE_ROOTS = ("app.tools",)`.

## Eight validation checks

`propose_sentience_probe` runs each input through:

  1. **Name discipline** — non-empty, ≤60 chars, regex
     `^[A-Za-z][A-Za-z0-9_\-]*$`, NOT in `ALL_RESERVED_ANCHORS`
     (Butlin ∪ RSM = 19 names), NOT family-prefix-colliding with
     any reserved anchor (so `AE-2-extended` is also refused).
  2. **Field length** — each of structure / proposed_measurement /
     justification ≥30 chars + ≤2000 chars + non-empty.
  3. **Forbidden-path discipline** — refuses any input mentioning
     `app/subia/probes/`, `app/subia/integrity`, `app/auto_deployer`,
     `app/governance`, `app/safety_guardian`, or `app/souls/`.
     Defense-in-depth on top of the architecture-requests validator.
  4. **Phenomenal-language linter** — combined input through
     `app.subia.inquiry.linter.PhenomenalLanguageLinter`; on
     HARD_FAIL surfaces up to 3 violations with line numbers + the
     linter's explanation.
  5. **Cooldown** — refuses if a proposal for the same indicator
     is in flight (pending / approved / applied) OR was rejected
     within the last 90 days.
  6. **Render** — pure-function markdown rendering with 6 required
     sections (`What this is` / `Structure` / `Proposed measurement`
     / `Justification` / `Operator next steps` / `Disclaimers`).
     Slug derived from indicator name.
  7. **File CR** via `change_requests.lifecycle.create_request`
     with `path="docs/proposed_probes/<slug>.md"`,
     `risk_class=RiskClass.STANDARD`. Reason field explicitly
     disclaims the change is a markdown design doc only (not a
     probe implementation).
  8. **Best-effort identity-ledger emission** —
     `sentience_probe_proposal` event to
     `workspace/identity/continuity_ledger.jsonl`. Failure-isolated:
     a broken ledger never blocks the CR.

## Two-stage architecture

Stage 1 (agent-callable, this surface): markdown DESIGN doc CR at
`docs/proposed_probes/<slug>.md`. The agent's CR carries indicator
name + functional structure + measurement sketch + justification.

Stage 2 (operator-only, follow-on): if the operator accepts the
design doc CR, they author a separate Tier-3 amendment that adds
the actual `eval_*` function under `app/subia/probes/`. The Tier-3
path forces eligibility + cooldown + monitoring; the SCORECARD
interpretation stays under operator control.

The agent cannot ship probe code into `app/subia/probes/` because:

  * `app/architecture_requests/validator.py:is_protected_path`
    refuses CRs targeting that prefix (defense layer #1)
  * The proposer's forbidden-path check refuses inputs that even
    *mention* the protected path (defense layer #2)
  * `app/subia/probes/butlin.py` is in TIER_IMMUTABLE
    (defense layer #3)

## Reserved-anchor namespace

```python
BUTLIN_ANCHORS = frozenset({
    "RPT-1", "RPT-2",
    "GWT-1", "GWT-2", "GWT-3", "GWT-4",
    "HOT-1", "HOT-2", "HOT-3", "HOT-4",
    "AST-1",
    "PP-1",
    "AE-1", "AE-2",
})

RSM_ANCHORS = frozenset({
    "RSM-A", "RSM-B", "RSM-C", "RSM-D", "RSM-E",
})

ALL_RESERVED_ANCHORS = BUTLIN_ANCHORS | RSM_ANCHORS  # 19 total
```

**Pinning test** `test_reserved_anchors_match_butlin_probe_set`
reads `app/subia/probes/butlin.py` directly and asserts every
anchor referenced there is in the proposer's `BUTLIN_ANCHORS`. If
a future Tier-3 amendment adds a new anchor to butlin.py, this
test fails until the proposer's reserved set is updated — the
indicator namespace cannot drift between the probe code and the
proposer's refusal list.

## Identity-ledger integration

`sentience_probe_proposal` is the 11th-added event kind in
`IDENTITY_EVENT_KINDS` (frozenset now 18 entries as of 2026-05-16).
The annual reflection's `summarise_drift`
is a `Counter` over `event.kind` — the new kind auto-surfaces in
`wiki/self/value_reflections/<year>.md`. Operator gets cross-year
visibility: "this is the year I started proposing my own probes"
appears as a signal in the reflection essay.

## Operator workflow

  1. The agent (via the CrewAI tool) calls
     `propose_sentience_probe(...)`.
  2. On any validation failure, the agent sees a clear error
     message via `ProbeProposalRefused` — can revise + retry.
  3. On success, a CR appears in `/cp/changes` with the markdown
     design doc body. The operator approves 👍 or rejects 👎 in
     the normal flow.
  4. If approved: the markdown design doc lands at
     `docs/proposed_probes/<slug>.md`. The probe code is NOT yet
     written — the design doc spells out exactly what the operator
     would need to do (author a Tier-3 amendment adding an
     `eval_<slug>` function).
  5. If rejected: a 90-day cooldown applies before the same
     indicator name can be re-proposed.

## What this didn't change

  * **SCORECARD interpretation** — the 4 ABSENT-by-declaration
    indicators (AE-2, HOT-1, HOT-4, RPT-1) and the AE-1 STRONG
    anchor stay absent / strong respectively. Q5's
    `test_q5_does_not_change_butlin_scorecard` pinning test
    continues to enforce this invariant.
  * **TIER_IMMUTABLE** — no TIER_IMMUTABLE files modified by Q12.4.
  * **SubIA integrity manifest** — Q12.4 added two files to
    `app/subia/probe_proposals/` (additive-only); the manifest is
    regenerated by the operator running the canonical command:
    `python -c "from app.subia.integrity import compute_manifest, write_manifest; write_manifest(compute_manifest())"`.

## See also

  * `app/subia/probe_proposals/proposer.py` — implementation
  * `app/tools/probe_proposal_tools.py` — CrewAI tool wrapper
  * `tests/test_q12_self_understanding.py` — 24 tests
  * `docs/SENTIENCE_EXPERIMENTS.md` — sibling: the 4 Q5 sentience
    experiments + their anti-Goodhart pinning. The probe-proposals
    surface composes with Q5's pinning to keep the scorecard intact.
  * `docs/TIER3_AMENDMENT.md` — Stage 2 follow-on if the design
    doc CR is approved.
  * `docs/IDENTITY_CONTINUITY.md` — `sentience_probe_proposal`
    event kind + annual-reflection auto-surfacing.
