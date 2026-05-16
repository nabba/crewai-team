# Year-2+ risk register (§10)

**Status (2026-05-16):** Shipped at PROGRAM §49 — Q14.

The operator's §10 audit named 6 emergent failure patterns that
matter only at year-2+ scale — none had a specific code fix, all
were patterns to watch. This subsystem makes each watchable with
a dedicated probe + alert + landmark.

| Item | Pattern | Probe | Alert topic |
|---|---|---|---|
| 10.1 | Aggregate drift via small amendments | `drift_digest` (monthly) | `identity_drift_acceleration` |
| 10.2 | Hidden feedback loops between subsystems | `feedback_loop_drift` (weekly) + `influence_graph` cycle detector | `feedback_loop_drift` |
| 10.3 | Goodhart on internal metrics (meta-agent recipes) | `goodhart_guard._detect_recipe_selection_divergence` (signal type) | flows through `goodhart_advisory_report` |
| 10.4 | Silent embedding-model fingerprint drift | `embedding_drift` (weekly, 20 anchor queries) | `embedding_model_swap` |
| 10.5 | Interest-model ossification | `interest_ossification` (weekly, entropy + Jaccard) | `interest_concentrated` / `interest_diffuse` / `interest_ossified` |
| 10.6 | Live-process lock contention | `lock_contention` (weekly p99) + passive `lock_metrics` timer | `lock_contention` |

All 6 are observational: they never block writes, never adjust
selection, never rebase embeddings. The operator decides.

## §10.1 — Identity drift digest

**Module:** `app/identity/drift_digest.py`

The `summarise_drift` API in `continuity_ledger.py` was already
in place — but only the yearly `annual_reflection.py` and
quarterly `long_term_goal_review.py` call it. A 47-amendment
quarter would be invisible until the next year-end essay.

**Algorithm:**

  1. Read `summarise_drift(30/90/365)` from continuity_ledger.
  2. Compute `aggregate_acceleration = counts_30d / (counts_365d / 12)`.
     1.0 = consistent with annual rate; 2.0+ = alert.
  3. Per-kind acceleration: any kind whose 30d count ≥3× its
     annual monthly average.
  4. On threshold crossing: Signal alert with `topic="identity_drift_acceleration"` + emit `identity_drift_acceleration` ledger landmark.

**Briefing integration:** `briefing_section()` returns a one-line
summary that the daily-briefing composer pulls. Silent on routine
("below threshold") passes.

**Master switch:** `identity_drift_digest_enabled` (default ON).
Cadence: monthly (daily probe with internal 30d gate).

## §10.2 — Hidden feedback loops

**Modules:**
  * `app/healing/influence_graph/edges.py` — hand-curated edge list
  * `app/healing/influence_graph/cycles.py` — Tarjan's SCC
  * `app/healing/monitors/feedback_loop_drift.py` — drift probe

The operator's stated concern:

> meta-agent recipes affect agent selection which affects what
> enters the lessons KB which affects what the meta-agent learns
> from. With 30+ idle jobs, an emergent loop could optimize for
> the wrong thing without any single component being broken.

**The influence graph** is a curated `InfluenceEdge(producer,
consumer, signal, kind)` list covering ~50 edges across the ~30
idle jobs, KBs, recovery loops, sentience-observation paths,
and decision surfaces. Hand-curated specifically — automatic AST
extraction produces noise; honest visibility into what we know
beats false automatic completeness.

**The cycle detector** is Tarjan's SCC algorithm (iterative,
decade-scale recursion-safe). Returns every closed loop with > 1
node, sorted alphabetically for stable test output. The named
meta-agent loop IS detected — there's a pinning test that asserts
this; if the curated edges ever stop closing the loop, that test
fails.

**The drift probe** watches the concrete named loop:
`meta_agent.selector → agent_factory → crew_lifecycle →
meta_agent.recorder → meta_agent.store → meta_agent.selector`.

  1. Read meta-agent recipe `uses` distribution.
  2. Compute Gini coefficient. Persist `(ts, gini)` to a rolling
     history JSONL (capped at 500 lines).
  3. Read last 4 weekly samples; check for monotonic increase AND
     total delta ≥0.10.
  4. On match: Signal + emit `feedback_loop_drift` ledger landmark.

**Why Gini.** It's bounded [0, 1]; well-understood; sensitive to
concentration; insensitive to absolute scale (so a quiet week
with few selections doesn't artificially shift it). A monotonic
upward trend over a month means the closed loop is converging on
a fixed point — the operator decides whether that's "learning
the best strategy" or "reinforcing a local optimum via own
feedback."

**Master switch:** `feedback_loop_drift_monitor_enabled` (default ON).

## §10.3 — Goodhart on internal metrics

**Extension to:** `app/goodhart_guard.py`

The Goodhart hard gate (§25.3) was scoped to promotion safety/
quality — three signal types covering the evolution loop. Q14.3
generalises to any optimised metric, with the meta-agent's
recipe selection as the named v1 case.

**New detector:** `_detect_recipe_selection_divergence(window_days)`

  1. For each meta-agent recipe with ≥20 outcomes in last `window_days`
     AND in the top-10% by `uses` count:
  2. Compute `thumbs_up_rate = up / (up + down)` from
     `user_feedback` field of recent `RecipeOutcome` rows.
  3. If `thumbs_up_rate < 0.30` (configurable
     `_DIVERGENCE_UP_RATE_THRESHOLD`) → emit
     `GamingSignal(signal_type="recipe_selection_divergence")`.
  4. Severity scales with how poor the feedback is:
     `< 0.15 → HIGH`, `< 0.25 → MEDIUM`, `< 0.30 → LOW`.

**Aggregation:** flows through existing `goodhart_advisory_report`
— the signal-type set is dynamic so the new type appears in the
report breakdown automatically.

**No master switch** — composes into the existing `goodhart_guard`
detection cycle; turning it off requires editing the file.
Operator can tune the threshold constants if false positives
appear in advisory mode.

## §10.4 — Embedding-model fingerprint drift

**Module:** `app/healing/monitors/embedding_drift.py`

Distinct from siblings:

  * `app/healing/llm_output_drift.py` — watches LLM-OUTPUT quality
    via embedding cosine against baseline OUTPUTS.
  * `app/memory/embedding_migration/` — handles INTENTIONAL
    embedding-model swaps via the Tier-3 amendment path.
  * **THIS** — watches the embedding model itself for silent
    vendor rotation under the same model name. Symptom is
    invisible to both other modules.

**Algorithm:**

  1. Maintain 20 canonical anchor queries at
     `workspace/healing/embedding_anchors.json`. Seeded with
     short neutral strings; operator-curated.
  2. On first run for each anchor, compute its embedding via the
     production embedding function and persist as baseline.
  3. On subsequent runs, re-embed each anchor, compute cosine
     similarity to baseline. ANY anchor whose self-similarity
     drops below 0.95 → vendor likely swapped weights silently →
     Signal alert + `embedding_model_swap` ledger landmark.

**0.95 threshold:** generous. Production embeddings re-compute
deterministically to ≈1.0; a real model swap drops to ~0.6–0.8.

**Master switch:** `embedding_drift_monitor_enabled` (default ON).
Cadence: weekly.

## §10.5 — Interest-model ossification

**Module:** `app/healing/monitors/interest_ossification.py`

The companion's `interest_model.py` produces a top-30 topic list
via `recency × frequency × source-diversity` scoring. Over years:

  * **Concentrated** — a few topics dominate (entropy → 0)
  * **Diffuse** — uniform noise (entropy → 1)
  * **Ossified** — list frozen (low churn for weeks)

**Three signals, three thresholds:**

  * Shannon entropy of top-30 score distribution, normalised to
    `[0, 1]`:
    - `< 0.30` (concentrated) → `interest_concentrated` alert
    - `> 0.90` (diffuse) → `interest_diffuse` alert
  * Jaccard overlap with the prior week's top-30 by name:
    - `≥ 0.95` for ≥4 consecutive weeks → `interest_ossified` alert

Each topic is independently arbiter-deduped (different alert
topics) so a concentrated AND ossified state surfaces as two
distinct nudges.

On any alert: emit `interest_ossification` ledger landmark with
detail dict containing all three measures.

**Master switch:** `interest_ossification_monitor_enabled`
(default ON). Cadence: weekly.

## §10.6 — Live-process lock contention

**Modules:**
  * `app/utils/lock_metrics.py` — passive timer
  * `app/healing/monitors/lock_contention.py` — weekly p99 monitor
  * `app/safe_io.py` — instrumented at `safe_write` + `safe_append`

The existing `lock_housekeeper` monitor catches STALE locks
(dead processes holding locks). Q14.6 is the complement: LIVE
contention between healthy processes.

**Design constraint (operator decision):** zero behavior change.
The instrumentation never blocks, never adds new locks, never
alters the write path. It just observes elapsed time per write
and records outliers. The hypothesis: long write times correlate
with kernel-level I/O queueing from concurrent writers.

**Passive instrumentation:**

  * `record_write_timing(resource, elapsed_ms)` — appends a row
    to `workspace/healing/lock_waits.jsonl` if `elapsed_ms ≥ 50ms`.
    Capped at 10k lines via `append_with_cap`.
  * `time_write(resource)` — context manager. Times the inner
    block, calls `record_write_timing` on exit.

`safe_io.safe_write` and `safe_io.safe_append` wrap the inner
write logic in `time_write(str(path))`. Inner write logic
extracted to `_do_safe_write` / `_do_safe_append` helpers so the
timing wrapper composes cleanly without disturbing the existing
disk-quota / crash-safety guarantees. Zero behavior change on the
fast path.

**Weekly monitor:**

  1. Read `lock_waits.jsonl` for the last 7 days.
  2. Group by resource.
  3. For each resource with ≥10 samples, compute p99 latency.
  4. Alert when p99 ≥ 500ms. Top-5 worst surface in the alert
     body with p50 + p99 + sample count.

**Master switch:** `lock_contention_monitor_enabled` (default ON).

## Cross-cutting wiring

**4 new continuity-ledger event kinds** (Q14.1, Q14.2, Q14.4,
Q14.5; total now 17):

  * `identity_drift_acceleration`
  * `feedback_loop_drift`
  * `embedding_model_swap`
  * `interest_ossification`

All four auto-surface in next year's annual reflection essay
because `summarise_drift` is a `Counter` over `event.kind`.

**6 new master switches** (default ON):

  * `identity_drift_digest_enabled`
  * `feedback_loop_drift_monitor_enabled`
  * `embedding_drift_monitor_enabled`
  * `interest_ossification_monitor_enabled`
  * `lock_contention_monitor_enabled`
  * `influence_graph_monitor_enabled` (gate for the cycle-report
    REST endpoint — currently dormant; reserved for future React
    surface)

**5 new healing monitors** registered in `app/healing/monitors/
__init__.py`:

  * `identity_drift_digest` (from `app/identity/drift_digest.py`)
  * `feedback_loop_drift`
  * `embedding_drift`
  * `interest_ossification`
  * `lock_contention`

The Q14.3 detector is part of the existing `goodhart_guard` cycle
— not a separate monitor.

## What Q14 deliberately did NOT do

  * **No automatic graph extraction.** The curated `edges.py` is
    hand-maintained. Honest visibility beats false automatic
    completeness.
  * **No automatic loop intervention.** When the Gini probe alerts,
    the operator decides whether convergence is desirable.
  * **No new locking in safe_io.** Q14.6 is passive instrumentation;
    `os.replace`-based atomic write was already race-free.
  * **No per-cycle probe for cycles other than meta-agent.** The
    graph reports all SCCs as observability; only the named loop
    has a concrete drift metric wired.
  * **No automatic embedding re-baseline.** When `embedding_drift`
    alerts, the operator decides whether to accept the new model
    or escalate.
  * **No SubIA integrity impact.** All new files outside
    `app/subia/`.

## See also

  * `app/identity/drift_digest.py` — §10.1 module
  * `app/healing/influence_graph/` — §10.2 graph + cycles
  * `app/healing/monitors/feedback_loop_drift.py` — §10.2 probe
  * `app/goodhart_guard.py` — §10.3 detector
  * `app/healing/monitors/embedding_drift.py` — §10.4
  * `app/healing/monitors/interest_ossification.py` — §10.5
  * `app/utils/lock_metrics.py` + `app/healing/monitors/lock_contention.py` — §10.6
  * `tests/test_q14_risk_register.py` — 28 tests
  * `docs/IDENTITY_CONTINUITY.md` — event-kind reference
  * `docs/META_AGENT_LAYER.md` — Q14.2 + Q14.3 composition points
  * `docs/RESILIENCE_DRILLS.md` — sibling year-2+ resilience layer
