# Decentered Reflection — No-Self Pass over the Affect Trace

A complementary reflection mode that operates over the same raw affect
trace as the Narrative-Self track but **without** imposing a first-person
frame, identity coherence, or daily arc. Shipped May 2026.

External-research-driven (Carhart-Harris's Entropic Brain Hypothesis,
Anil Seth's constructed-self framing, Shamil Chandaria) — not driven by
an incident. This is an **observability addition**, not a behavioral
change.

---

## 1. Why this design exists

The Narrative-Self track (`app/affect/{salience,episodes,narrative}.py`)
imposes useful structure on the affect trace: temporal continuity,
identity coherence, causal arc, salience filtering toward storylike
episodes. That structure costs visibility into:

- **Cross-thread / cross-agent simultaneity** — events are scored
  per-state, not relationally. Patterns that recur *across agents at
  the same time* are invisible to per-state salience.
- **Low-affect cross-thread patterns** — events that don't trip a
  V/A/C delta or a viability OOB threshold are dropped, even when they
  recur.
- **Cross-day motifs** — the daily chapter consolidator's window is
  fixed at 24 h; week-spanning shapes can only enter chapters via the
  7-prior-chapter text snippets, lossily.
- **Statistically unusual moments that aren't attractor transitions**
  — anomalies in V/A/C joint distributions that don't match any
  salience trigger.

The decentered pass surfaces these by clustering and anomaly-detecting
the trace *without* the narrative frame. If it surfaces nothing, that's
informative — it means the narrative pass is already capturing what
matters at this scale.

---

## 2. Architecture (one screen)

```
trace.jsonl ──┐
              ├──>  decentered.run_decentered_pass(window_hours)
salience.jsonl┘                  │
                                 ├──>  Pass A: structural cluster
                                 │       fingerprint = (kind, attractor,
                                 │                      prev_attractor,
                                 │                      sorted oob_vars)
                                 │       split each bucket greedily
                                 │       on (V, A, C) Euclidean
                                 │
                                 └──>  Pass B: rolling z-score anomaly
                                         per-variable rolling mean+stddev
                                         across V / A / C / total_error /
                                         epistemic_uncertainty (window 256)
                                         flag |z|_max ≥ 3.0

  →  workspace/affect/decentered/<YYYY-MM-DD>.json
```

**Pure Python.** No numpy / scipy / sklearn dependency — mirrors the
`self_improvement/consolidator.py` pattern (greedy complete-linkage
agglomerative) so the algorithm is deterministic, auditable, and adds
no install surface.

**Idempotent on date.** Re-running the same UTC date overwrites the
output file. Multiple runs per day are safe.

---

## 3. Strict invariants

The decentered module is observational. The following are guarded by
both convention and a structural test (`test_module_does_not_import_kb_or_identity_writers`):

- **MUST NOT** import `app.experiential.*` — no writes to the KB.
- **MUST NOT** import `app.affect.episodes` or `app.affect.narrative` —
  no entry to the narrative pipeline.
- **MUST NOT** mutate `identity_claims.json`.
- **MUST NOT** emit `entry_type=chapter` or `entry_type=episode`.

Self-Improver permissions: read-only on this module. The same
self-modeling integrity invariant that protects `salience.py` and
`narrative.py` applies — letting the Self-Improver tune what counts as
"unusual without a self-frame" would let it edit how it observes itself.

---

## 4. Behavior

### Pass A — Structural cluster

Each salience event has a structural fingerprint:

```
(kind, attractor, prev_attractor, sorted(out_of_band))
```

Events sharing a fingerprint go into a bucket. Within each bucket, a
greedy complete-linkage cluster splits on (V, A, C) Euclidean distance
with `_CLUSTER_VAC_THRESHOLD = 0.35`. Sub-clusters below
`_MIN_CLUSTER_SIZE = 3` are dropped.

Each surviving cluster gets summarized with:

- `fingerprint` (string)
- `size`, `vac_centroid`, `spread`
- `first_ts`, `last_ts`, `days_spanned`, `days` (sorted set)
- `severities` (counter)
- `sample_details` (top 3, truncated to 160 chars)

Clusters are sorted by `(days_spanned, size)` descending — cross-day
shapes float to the top.

### Pass B — Rolling z-score anomaly

For each trace point, the module maintains a rolling window of
`_ANOMALY_WINDOW = 256` previous samples per variable. Once the window
has at least `_ANOMALY_MIN_BASELINE = 32` samples, every subsequent
point computes per-variable z-scores:

```
z = (x - rolling_mean) / rolling_pstdev
```

The composite anomaly is `max(|z|)` across the 5 variables. Anomalies
above `_ANOMALY_Z_THRESHOLD = 3.0` are reported with the offending
variable, the z-score, V/A/C, total_error, epistemic_uncertainty, and
the attractor label.

This is cheaper than Mahalanobis but still flags "unusual along any
axis" — sufficient for first-pass surfacing.

---

## 5. Output schema

`workspace/affect/decentered/<YYYY-MM-DD>.json`:

```json
{
  "ts": "<UTC iso>",
  "window_hours": 168,
  "input": {
    "trace_points": 12345,
    "salience_events": 678,
    "trace_window_first_ts": "...",
    "trace_window_last_ts": "..."
  },
  "clusters": {
    "total": 42,
    "cross_day": 7,
    "top": [ /* ≤ 20 cluster summaries */ ]
  },
  "anomalies": {
    "total": 18,
    "outside_salience": 11,
    "top": [ /* ≤ 20 anomalies, sorted by |z| desc */ ]
  },
  "experiment_criterion": {
    "min_cross_day_span": 3,
    "min_cluster_size": 3,
    "anomaly_z_threshold": 3.0
  }
}
```

`anomalies.outside_salience` is the count of anomalies whose timestamp
does NOT match any salience event in the window — i.e. statistically
unusual moments that the per-variable salience filter dropped.

---

## 6. Operations

### Files

- `app/affect/decentered.py` — module (pure Python, ~340 LOC)
- `tests/test_affect_decentered.py` — 10 tests (cluster, anomaly,
  persistence, structural-import guard)
- Registration: `app/idle_scheduler.py` registers `decentered-pass`
  (LIGHT). Daily runs over the last 24 h.

### Manual entry

```bash
# Default: 14-day window
python -m app.affect.decentered

# Explicit hours
python -m app.affect.decentered 168    # 7 days
```

### Experiment criterion

The May-2026 experiment ran for 7 days and is checked by a one-shot
local cron on 2026-05-09 (see this session's `/loop` checkpoint). The
falsification criterion:

> **≥3 strongly-novel clusters AND ≥1 cross-day motif AND <30%
> spurious rate over a 14-day window**

"Strongly-novel" requires a manual rubric pass — the module flags the
top clusters by `days_spanned * size`, but the rubric (redundant /
weakly-novel / strongly-novel / spurious) is a human call. If the
criterion fails, the right move is to **kill** the module (dead-end
hypothesis: the narrative pass is doing the work).

### Failure modes

- **Empty `salience.jsonl`** — system hasn't been running enough.
  Output is `clusters.total = 0`. Mark the experiment **inconclusive**,
  not failed.
- **No anomalies before baseline fills** — the first 32 trace points
  are silent by design. Expected on cold-start.
- **Clusters dominated by single fingerprint** — usually
  `transition|<X>←<Y>` if the system has been bouncing between two
  attractors. Not a bug; report it as a finding.

---

## 7. What this is NOT

- **Not a behavioural change.** The decentered pass writes to its own
  file and is read by no other subsystem at present. Any downstream
  consumer would be a separate, future change.
- **Not a replacement for the narrative-self.** The narrative pass
  produces first-person continuity that downstream prompts depend on
  (`commander/context.py:1007`). Decentered output is complementary,
  not substitutive.
- **Not phenomenology.** This is the *mechanics* of
  filter-relaxed cognition — clustering and anomaly detection without
  a self-frame. The film that inspired it is honest that third-person
  neuroscience can't reach first-person experience; replicating
  the algorithm doesn't replicate the experience. Conflating them
  would be a category error.

---

## 8. References

- Carhart-Harris, R. L. (2018). *The Entropic Brain — Revisited.*
- Seth, A. K. (2021). *Being You: A New Science of Consciousness.*
- Cohen et al. (2024). *Unraveling the Dream*, Waking Up — film
  exploring the entropic brain + predictive processing + free-energy
  triad with Anil Seth, Robin Carhart-Harris, Shamil Chandaria.
- Internal: `crewai-team/docs/AFFECT_LAYER.md` (the layer this
  observes); the Narrative-Self track lives at
  `app/affect/{salience,episodes,narrative}.py`.
