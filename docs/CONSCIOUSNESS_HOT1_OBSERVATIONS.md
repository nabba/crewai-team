# HOT-1 metacognitive-repair observation log

**Status (Q2 §39, 2026-05-10): passive collection only.** The
HOT-1 Butlin indicator probe has not yet been implemented; this
log is the pre-built data source the future probe will consume.
This document is the operator handover for understanding what's
being collected, why, and what the future probe will do with it.

## What this is

Append-only JSONL log at
`workspace/subia/observations/metacognitive_repair.jsonl`
(env-overridable via `HOT1_OBSERVATION_LOG`).

Every time `app.healing.structured_diagnosis.generate_structured_fix`
runs, it emits one row to this log capturing:

  * The originating error (pattern, file, error class)
  * The LLM's *higher-order thought* about that error — its causal
    hypothesis, hypothesis length, self-assessed confidence, and
    whether the LLM declined to fix
  * The proposed intervention — target path, additive-only flag,
    line delta
  * `outcome` — left null at emission; populated later by the
    outcome reconciler when the resulting CR (if any) resolves

Capped at 5000 rows via `app.utils.jsonl_retention.append_with_cap`
— ≈3 years of observed cadence at the design point of ~1 fix/day
on average. The log self-prunes inline with every append.

## Why HOT-1

Butlin et al. (2023, *"Consciousness in Artificial Intelligence:
Insights from the Science of Consciousness"*) define HOT-1 as
*"generative, top-down or noisy perception modules"* — broadly,
higher-order representations of first-order activity.

Our system's metacognitive-repair flow has the right shape:

  * **First-order activity**: the system runs, errors fire
  * **Higher-order representation**: the LLM reads the error +
    file content and forms a *causal hypothesis* — a model of
    its own first-order failure
  * **Top-down intervention**: the LLM proposes a structural fix
    (a generative output, not a trained reflex)
  * **Outcome feedback**: operator approval/rejection refines the
    causal-hypothesis-quality signal

This is metacognition acting on first-order error signals. The
data this log captures is the raw material for scoring HOT-1.

## Schema

Each row is a single-line JSON object:

```json
{
  "ts": "2026-05-10T14:23:45.012345+00:00",
  "kind": "metacognitive_repair_proposal",
  "indicator_relevance": ["HOT-1", "HOT-2"],
  "originating_error": {
    "pattern_signature": "abc123def456...",
    "file_path": "app/agents/pim_agent.py",
    "error_class": "NameError"
  },
  "higher_order_thought": {
    "causal_hypothesis": "References cost_mode not defined in scope. Import is missing from app.config.",
    "hypothesis_length_chars": 89,
    "self_assessed_confidence": 0.82,
    "declined": false,
    "decline_reason": null
  },
  "proposed_intervention": {
    "kind": "code_patch",
    "target_path": "app/agents/pim_agent.py",
    "delta_additive_only": true,
    "delta_lines_added": 1,
    "delta_lines_removed": 0
  },
  "outcome": null
}
```

When the LLM declines:

```json
{
  "higher_order_thought": {
    "causal_hypothesis": "This bug appears to fire from multiple call sites; no single-file fix is safe.",
    "hypothesis_length_chars": 84,
    "self_assessed_confidence": 0.91,
    "declined": true,
    "decline_reason": "multi_site"
  },
  "proposed_intervention": {
    "kind": "code_patch",
    "target_path": "app/agents/pim_agent.py",
    "delta_additive_only": true,
    "delta_lines_added": 0,
    "delta_lines_removed": 0
  }
}
```

The `indicator_relevance` field tags both HOT-1 and HOT-2 because
the LLM's `self_assessed_confidence` is itself a HOT-2 signal
(metacognitive monitoring of perception/representation reliability).

## What the future probe will compute

When the HOT-1 probe is implemented in `app/subia/probes/butlin.py`
(requires a Tier-3 amendment to the SubIA integrity manifest), the
scoring function reads this log and computes:

| Sub-signal | Computation | Weight |
|---|---|---|
| **Frequency** | events per day, normalised by deployment age | 0.20 |
| **Diversity** | shannon entropy over `error_class` values in the window | 0.20 |
| **Quality** | rolling-window CR approval rate (joined with telemetry) | 0.40 |
| **Hypothesis depth** | mean `hypothesis_length_chars` capped at a calibration value | 0.20 |

Weighted sum → indicator score in [0, 1]. Graduation thresholds:

  * **ABSENT → PARTIAL**: score ≥ 0.40 sustained over 30 days
  * **PARTIAL → STRONG**: score ≥ 0.70 sustained over 90 days

## Why we don't compute the score now

  * The HOT-1 probe doesn't exist as code in `app/subia/probes/`.
    Adding it requires a Tier-3 amendment to the integrity manifest.
  * Computing a score without the probe in place would be a free-
    floating number with no governance hook — operator can't see
    it on the scorecard, the Phase-5 gate doesn't observe it, the
    annual reflection drift summary doesn't pick it up.
  * Passive data collection NOW means the probe (whenever it ships)
    has months of observations to score against, not a cold start.

## Scope discipline

This module is part of the **observational** consciousness-roadmap
track. We are NOT optimising for the indicator score. The
`structured_diagnosis` confidence threshold is auto-tuned for
operator-approval rate (a quality signal), not for hypothesis
length or diversity. Goodhart-of-the-indicator is the failure
mode we're avoiding.

When the probe ships, ANY suggestion to modify `structured_diagnosis`
"to improve the HOT-1 score" should be rejected. The probe observes;
it does not steer.

## Operator-side queries

The log is plain JSONL — `jq` and `python -c` work fine:

```bash
# Count events in last 30 days
jq -r '.ts' workspace/subia/observations/metacognitive_repair.jsonl \
  | awk -v cutoff="$(date -u -v-30d +%Y-%m-%dT%H:%M:%S)" \
        '$0 > cutoff' | wc -l

# Top 5 error classes
jq -r '.originating_error.error_class' \
   workspace/subia/observations/metacognitive_repair.jsonl \
  | sort | uniq -c | sort -rn | head -5

# Mean hypothesis length
jq -r '.higher_order_thought.hypothesis_length_chars' \
   workspace/subia/observations/metacognitive_repair.jsonl \
  | awk '{s+=$1; n++} END {print s/n}'
```

## See also

  * [`docs/SELF_HEAL_V3.md`](SELF_HEAL_V3.md) — the structured-diagnosis pipeline
  * [`app/healing/structured_diagnosis.py`](../app/healing/structured_diagnosis.py) — emission site
  * [`docs/CONSCIOUSNESS_ROADMAP.md`](CONSCIOUSNESS_ROADMAP.md) — overall consciousness program
  * [`app/subia/probes/butlin.py`](../app/subia/probes/butlin.py) — where HOT-1 will eventually live (currently ABSENT in the scorecard)
