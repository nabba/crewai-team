# Targeted Sentience Experiments

**Status (2026-05-13):** Closed across six commits and four audit cycles
(PROGRAM §43.1 through §43.6). Observationally complete. Re-audit only
when a specific concrete concern arises.

## What this is

A package of four observational modules under `app/sentience_experiments/`
plus two infrastructure extensions (philosophy decision panel, ledger
governor). The four modules reify **functional approximations** of
cognitive capabilities the Butlin scorecard declares architecturally
ABSENT.

**Critical anti-Goodhart framing — read this first.**

The user requested modules labeled with four Butlin indicators
(AE-2, HOT-1, HOT-4, RPT-1). The descriptions the user attached to
each indicator are **functional capability names**, not the literal
Butlin definitions:

| Indicator | User's functional name | Literal Butlin definition |
|---|---|---|
| AE-2 | rare-event causal credit assignment | embodiment + environment coupling |
| HOT-1 | feelings-about-feelings reflection | generative top-down perception |
| HOT-4 | metacognitive monitor on reasoning chain | sparse and smooth coding |
| RPT-1 | forward-prediction self-scoring | algorithmic recurrence in inference |

The codebase's `app/subia/probes/butlin.py` evaluates each indicator
**strictly** and has declared all four ABSENT-by-declaration on an
LLM substrate (no body, no perception substrate, dense embeddings,
feed-forward inference).

The sentience experiment modules **do not change the scorecard.**
They live at `app/sentience_experiments/` precisely so the
canonical-path evaluators (which check `app/subia/*`) never see them.
The capabilities are worth building because the *theory says they
are necessary*, not because the *scorecard rewards them*.

This commitment is pinned by
`tests/test_q5_2_modules.py::test_q5_does_not_change_butlin_scorecard`
which invokes every Butlin evaluator and asserts the count remains
`{STRONG=7, PARTIAL=3, ABSENT=4, FAIL=0}` after Q5 ships. If this
test ever fails, the anti-Goodhart contract has been broken — that
is a P0 architectural regression.

## The four modules

### `ae2_causal_credit.py` — rare-event causal credit assignment

Reads `loadable_agent_usage.jsonl` (actions), `errors.jsonl` (failure
outcomes), `welfare_audit.jsonl` (affect outcomes), and `audit_log.jsonl`
(operator approvals/rejections; added in Q5.4). For each
`(action_signature, outcome_kind)` bucket in a rolling 7-day window,
computes co-occurrence within a 24h lookahead.

Flags pairs with:
- `outcome_rate ≤ 10%` (rare-event criterion)
- `outcome_density_ratio ≥ 3×` (signal threshold)
- `n_observations ≥ 5` (minimum support)

Persists to `workspace/sentience/ae2_associations.jsonl`.

**Honest naming note (Q5.4 fix):** the original ship used `lift =
P(outcome|action) / P(outcome)` field names, but the math doesn't
deliver true probabilities — numerator and denominator both mix
outcome events with action events. Q5.4 renamed to
`outcome_density_ratio` and `outcome_rate` to be honest about what
the values represent. The ratio rank-orders associations reasonably;
just don't treat the absolute value as a Bayesian likelihood.

### `hot1_meta_affect.py` — feelings-about-feelings reflection

Reads BOTH `welfare_audit.jsonl` (breach events — sparse) AND
`workspace/affect/trace.jsonl` (full V/A/C/attractor snapshot stream
— added in Q5.4). Detects five pattern kinds:

- `temporal_cluster` — ≥3 breaches in the same hour-bucket
- `recurring_trigger` — ≥2 breaches of same kind within 24h
- `sequence` — same kind appearing ≥3 times at >24h spacing
- `baseline_drift` (Q5.4) — mean V/A/C shift > 0.15 between window halves
- `attractor_lock` (Q5.4) — ≥70% concentration in one attractor

Persists to `workspace/sentience/hot1_meta_affect.jsonl`.

**SOUL.md guard:** any prose output passes through `decenter_text()`
which **hard-rejects first-person affect language** ("I feel", "I'm
anxious", "my emotion", etc.). The Q5.4 LLM enrichment (Claude
Haiku 4.5 via the canonical Anthropic SDK pattern) is gated behind
this filter — an LLM that produces forbidden prose falls back to the
deterministic template. Pinned by
`test_hot1_decenter_filter_rejects_first_person_affect` and
`test_hot1_llm_enrich_exercises_real_call_path`.

**Deliberate non-wire (Q5.6):** HOT-1 does NOT cross-reference the
existing `app/affect/decentered.py` (no-self pass over the affect
trace). The two systems represent complementary epistemic stances
on the same data; co-locating outputs would imply commensurability.
Re-open only with a specific concrete reason.

### `hot4_metacog_monitor.py` — metacognitive monitor on reasoning chain

Async-batch summariser over `loadable_agent_usage.jsonl`. Per-step
signals:

- `confidence_proxy` — output_tokens / input_tokens (inverse confidence)
- `cache_reliance` — cache_read / (cache_read + cache_create + input)
- `cascade_jump` — true if this step used a higher-tier model than
  the agent's rolling baseline
- `unusual_score` — combined z-score against per-agent baseline

Persists to `workspace/sentience/hot4_reasoning_signals.jsonl`.

**Write-only telemetry contract (pinned by
`test_hot4_signals_never_gate_dispatch_logic`):** the HOT-4 signals
are NEVER consumed by dispatch / routing / model-selection logic.
The grep test fails on any new importer outside an explicit
allow-list (sentience scheduler, dashboard read endpoint, briefing
read). Importing HOT-4 into LLM cascade or agent dispatch would
close the Goodhart loop this contract exists to prevent.

### `rpt1_self_calibration.py` — forward-prediction self-scoring

Three responsibilities:

1. **Registration**: `register_prediction(claim_kind, claim_text,
   predicted_p, resolution_at, scorer_ref, scorer_args)` persists a
   `Forecast` row.
2. **Reconciliation** (hourly idle): walks unresolved forecasts past
   their `resolution_at`, runs the registered scorer.
3. **Calibration aggregation** (daily idle): computes per-kind Brier
   score + ECE + 10-bucket calibration curve over a rolling 30-day
   window.

Built-in scorers: `tier3_approval`, `cr_apply`. Operators can
register more via `register_scorer(name, callable)`.

**Two pinning contracts:**

- **No feedback loop** (pinned by
  `test_rpt1_calibration_state_does_not_feedback_to_predictive_layer`):
  the calibration state must NOT be imported by `app/subia/prediction/*`.
  Closing this loop would create self-judging-its-own-predictions
  overfitting.
- **No LLM scorers** (pinned by
  `test_rpt1_scorer_registry_refuses_llm_modules`): `register_scorer`
  refuses callables defined in `app.llm.*`, `app.agents.*`,
  `app.crews.*`, `anthropic`, or `openai`. Deterministic outcome
  resolvers only.

**Wired producers (Q5.4):** Tier-3 amendment proposals register a
`tier3_approval` forecast at proposal time. CR creation registers a
`cr_apply` forecast. Without these, the calibration ledger sat
empty — the original Q5 ship missed the producers entirely.

**Stale-forecast timeout (Q5.5):** forecasts past `resolution_at` by
≥60 days with the scorer still returning `None` are terminated with
`score_error="stale_unresolved"` so they don't sit in eternal limbo.

## The two infrastructure extensions

### Philosophy decision panel (`philosophy/dialectics.py`)

`consult_panel(question, traditions, max_perspectives, use_cache)`
returns a `PanelResult` with structured `PerspectiveTension` records
(claim, counter-claim, optional synthesis, tradition, source,
confidence). 7-day result cache keyed on `(question, traditions)`.

**Wired into three high-stakes decision sites:**

1. **Tier-3 amendment proposals** (`tools/request_tier3_amendment.py`)
   — panel result lands in `proposal.evidence["philosophy_panel"]`;
   unresolved tensions bridge into Q4.1 tensions store.
2. **Identity-claim ratification** (`affect/narrative.py
   :_ratify_identity_claims`) — claims with ≥2 unresolved tensions
   are deferred to operator review (filed into the tensions store
   rather than auto-FIFO-ratified).
3. **Welfare-bound calibration** (`affect/calibration_proposals.py
   :evaluate_and_apply`) — panel consulted AFTER the apply branch
   (observational, never gates the 6-guardrail flow).

### Ledger governor (`identity/relevant_history.py`)

`relevant_history_by_kind(path, window_days=365)` extends the existing
per-path lookup with a per-kind track record. Classifies paths into
27 buckets (`soul_edit`, `welfare_envelope`, `kernel`,
`agent_definition`, etc.) and aggregates ledger + CR-audit events
across the window. Returns `counts_by_outcome` + `success_rate` +
`has_resolved_history` (the Q5.5 fix that distinguishes "no history"
from "proven 0% success").

Wired into `request_tier3_amendment.py` as a second history strand
alongside the per-file 90d lookup. Goodhart guard: informational only.

## Master switches

All seven default ON per operator approval during the Q5 plan:

| Setting | Purpose |
|---|---|
| `sentience_ae2_enabled` | AE-2 rare-event credit assignment |
| `sentience_hot1_enabled` | HOT-1 meta-affect reflection |
| `sentience_hot4_enabled` | HOT-4 reasoning-chain monitor |
| `sentience_rpt1_enabled` | RPT-1 self-calibration |
| `philosophy_panel_enabled` | Decision panel |
| `ledger_governor_enabled` | By-kind history aggregator |
| `sentience_llm_hypothesis_enabled` | HOT-1 LLM prose enrichment |

Each module fails open: master switch off → no-op return. Failures
inside a module never break the companion idle loop.

## Operator surfaces

Five REST endpoints under `/api/cp/sentience/*`:

| Endpoint | Returns |
|---|---|
| `GET /sentience/ae2/associations` | Recent causal associations |
| `GET /sentience/hot1/patterns` | Recent meta-affect patterns |
| `GET /sentience/hot4/flagged` | Flagged reasoning-chain signals |
| `GET /sentience/rpt1/calibration` | Per-kind Brier + ECE + curve |
| `GET /sentience/scorecard-pinning` | **Live anti-Goodhart audit** |

The last one is the operator-facing audit endpoint — invoke any
time to verify `anti_goodhart_intact: true` and that the targeted
four indicators are still ABSENT.

**Weekly briefing digest** (`life_companion/daily_briefing.py
:_gather_sentience_digest`): surfaces top-1 finding from each module
in the Sunday/weekly composer under a "🔬 Self-observation (week):"
section. Section disappears when nothing happened. Opaque counts
only — no action_signatures, no breach kinds, no identities.

## Continuity ledger emission

Each module emits a `sentience_observation` event to the identity
continuity ledger on LANDMARK events only (not every detection):

- **AE-2**: first emission for a `(action_sig, outcome_kind)` pair
  at `outcome_density_ratio ≥ 5.0`. Dedup state at
  `workspace/sentience/ae2_landmarks_emitted.json` capped at 10K
  entries with FIFO eviction (Q5.6).
- **HOT-1**: trace-level patterns (`baseline_drift` /
  `attractor_lock`) only — breach-clustering patterns are routine
  and don't reach the ledger.
- **HOT-4**: ≥5 flagged steps in one pass, with 7-day per-source
  cooldown consulting the ledger directly (Q5.6).
- **RPT-1**: a `claim_kind` crossing the min-resolutions threshold
  (10 resolutions) for the first time.

The annual reflection's `summarise_drift` Counter auto-surfaces the
new event kind, so the year-end self-reflection will naturally
include "the year the system started observing itself" without any
further wiring.

`ledger_bridge.emit_landmark()` enforces a per-process emission
ceiling of 50 per source_module (Q5.5) as a safety net against any
future logic bug looping landmark emission.

## Architecture audit trail

| Commit | Section | Scope |
|---|---|---|
| `d5a8eecb` | §43.1 | Foundation: philosophy panel + ledger governor + wires |
| `d5ee1f9d` | §43.2 | Four modules + anti-Goodhart pinning test |
| `0f9f86e9` | §43.3 | Operator surfaces (REST + scorecard-pinning endpoint) |
| `327844c0` | §43.4 | First post-ship audit — 6 P0/P1 + 6 wires |
| `45395b3a` | §43.5 | Second post-ship audit — 6 P0/P1 (incl LLM-API bug) |
| `36186f8b` | §43.6 | Third post-ship audit — 5 polish; **Q5 closed** |

Each audit cycle was productive. The trajectory flattened from
architectural findings (rounds 1-2) to polish (round 4). PROGRAM.md
§43.6.7 declares Q5 closed.

## Re-audit guidance

The four-audit pattern past round 4 risks finding-things-for-the-sake-
of-finding-things. Re-audit when:

- A new related subsystem ships that should integrate (e.g. a future
  AE-3 capability that should compose with AE-2)
- A specific concrete behavior surprises an operator
- The Butlin paper publishes a clarifying update that changes the
  functional interpretation of one of the four indicators
- The scorecard-pinning live endpoint returns `anti_goodhart_intact:
  false` — this is an instant P0

Do NOT re-audit on the cadence of "let's verify Q5 is still ok."
The CI pinning test plus the live endpoint cover that.

## Operator quick reference

```
# Check the anti-Goodhart contract live
curl /api/cp/sentience/scorecard-pinning | jq .anti_goodhart_intact
# → true (always, by contract)

# View recent causal associations
curl /api/cp/sentience/ae2/associations?limit=10

# View recent meta-affect patterns
curl /api/cp/sentience/hot1/patterns?limit=10

# View flagged reasoning-chain signals
curl /api/cp/sentience/hot4/flagged?limit=10

# View per-kind calibration state
curl /api/cp/sentience/rpt1/calibration

# Disable a specific module
curl -X POST /api/cp/settings -d '{"sentience_hot4_enabled": false}'
```

The four JSONL data files at `workspace/sentience/*.jsonl` are
operator-readable and survive across restarts. Deleting them resets
the module's state without affecting anything else.
