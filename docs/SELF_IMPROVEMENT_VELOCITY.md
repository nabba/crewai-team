# Self-Improvement Velocity

**Status (2026-05-16):** Q16 Theme 4 — decade-resilience hardening,
recursive self-improvement boundaries. Observational rollup of "is
the system actually getting better?"

## Why this exists

Q1–Q15 shipped a lot of self-improvement infrastructure: capability-
gap analyser, library radar, paper pipeline, proposal bridge,
architecture-request flow, meta-agent recipes, lessons-learned KB,
Forge tool registry, structured diagnosis, change-request lifecycle.

What was missing: a single surface that answers **"is this whole
machine producing value, and at what rate?"**

The Tier-3 amendment + self-quarantine + Goodhart hard gate are the
STRUCTURAL safety story. Velocity is the OBSERVATIONAL counterpart:

  * Are CRs being filed, applied, rejected?
  * Are architecture requests actually being used?
  * Are meta-agent recipes converging or collapsing?
  * Is `lessons_learned` growing AND being consulted?
  * Is the Forge pipeline alive?

Without these numbers, an observer cannot tell if the system is
improving or just churning.

## Design

`app/self_improvement/velocity.py` aggregates five sources into one
read-only summary:

  1. **Change requests** — by quarter × (requestor, status), with
     applied-rate-overall.
  2. **Architecture adoption** — distribution of adoption scores
     across APPLIED architecture requests (the same probe the
     `architecture_adoption` healing monitor uses).
  3. **Meta-agent recipes** — top-N most-used, total uses,
     success rate.
  4. **Lessons-learned** — total entries, recent additions, count
     consulted (when the KB schema records consultations).
  5. **Forge graduations** — SHADOW → CANARY → PROMOTED transitions
     from `workspace/forge/graduations.jsonl`.

Every section is **failure-isolated**. A broken upstream returns
`{"available": false}` but never breaks the rollup.

Pinning test asserts the module is **read-only**: no `store.save`,
`approve(`, `auto_approve`, or `create_request` calls in the source.

## REST endpoint

```
GET /api/cp/self-improvement/velocity?window_days=N
```

`window_days` ∈ [30, 3650]; default 365.

Returns the full rollup as JSON.

## Composing with other surfaces

This module is **observational**. It informs:

  * Future React dashboard at `/cp/self-improvement` (deferred —
    REST surface is enough for now).
  * Operator's quarterly retro reviews.
  * Annual reflection composer (could optionally aggregate the
    velocity table into the yearly value-reflection summary).

It composes with:

  * `architecture_adoption` healing monitor — the histogram surfaces
    the same scores the monitor uses to file rollback CRs.
  * `feedback_loop_drift` healing monitor — recipe Gini lives there;
    we surface uses-by-recipe; both look at the same ledger.
  * `notify_suppression_review` healing monitor — arbiter suppression
    rate complements the change-rate signal.

## Master switch

`self_improvement_velocity_enabled` (default ON). When OFF, the
endpoint returns `{"disabled": true}` and the function short-circuits.

## What this is NOT

  * NOT a gate. Velocity is observation only; never blocks anything.
  * NOT a Goodhart target. The system never trains against these
    numbers. If a recipe's apparent uses goes up while quality
    drops, the Goodhart guard catches it via `recipe_selection_
    divergence`, NOT via this rollup.
  * NOT real-time. Aggregates the JSON store / JSONL log on each
    call; for a high-traffic deployment, cache or move to a
    pre-computed table.

## Deliberately deferred

  * React dashboard at `/cp/self-improvement` — REST surface is
    sufficient for now.
  * Pre-aggregation into Postgres tables — JSON sources are fine
    at current volumes.
  * Annual rollup digest composer that summarises a whole year
    in one markdown report (parallels vacation digest).
