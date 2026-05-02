# Reducing-Valve Audit

Measurement-only instrumentation that detects filters that drop too
aggressively. Each instrumented filter writes a rejection event to a
shared JSONL sink; a daily replay job re-evaluates a sample two ways
(loose-replay + LLM second-opinion) and produces per-filter false
rejection rate (FRR) and disagreement rate (DR). Shipped May 2026.

External-research-driven (Huxley's "reducing valve" metaphor for
perception, mapped onto the system's filter / gate / rejection points).
**Measurement only — does NOT change filter behaviour.**

---

## 1. Why this design exists

The system has many filters: relevance, salience, refusal detection,
quality gates, novelty checks, surfacing thresholds, retrieval
min_score, panel critique cutoffs, epistemic pushback. Each was
calibrated independently. The hypothesis is empirical:

> At least one filter is probably too narrow in a way no individual
> filter knows.

There's no incident driving this — it's a deliberate test of whether
the cumulative effect of independent calibrations is dropping useful
signal somewhere. The audit produces **falsifiable** numbers: per-filter
FRR with magnitude (estimated useful items lost per day). Filters above
the threshold get a follow-on widening proposal through the existing
governance path; filters below are confirmed appropriately narrow.

The user's prior falsification of three "token economy wins" sets the
bar: this initiative MUST produce numbers, not vibes.

---

## 2. Filter inventory

The audit plan identified 18 candidate filters spanning recovery,
vetting, retrieval, companion, self-improvement, SubIA scene, affect
salience, and epistemic pushback. The May-2026 MVP instruments **3**
of them — covering output quality, refusal detection, and idea
surfacing — to validate the approach before instrumenting the rest.

**In scope (instrumented):**

| ID | File | Predicate (rejection cause) |
|---|---|---|
| F1 | `app/recovery/refusal_detector.py:259, 276` | borderline cases: refusal phrase matched but density or composite confidence below threshold (filter says "not really a refusal, skip recovery") |
| F4 | `app/agents/commander/execution.py:33-52` | quality gate: too short / refusal pattern / meta-commentary / coding-task without code block |
| F8 | `app/companion/surfacing.py:50-58` | surface gate: no_text / below_novelty / below_quality / below_panel / cooldown |

**Deferred for MVP (in inventory, not yet instrumented):**

F2/F3 (vetting — TIER_IMMUTABLE, must be instrumented at the caller),
F5 (retrieval min_score), F6 (cross-encoder reranker output cap), F7
(skill-chain relevance floor), F9 (companion 5-persona panel), F10
(companion cross-workspace transferability), F11 (companion
scheduler affect-block), F12 (Novelty Gate), F13 (librarian capability
inference), F14 (sandbox-execute pre-flight), F15/F16 (SubIA scene
buffer + GWT broadcast ignition), F17 (affect salience), F18 (epistemic
pushback detector).

**Out of scope (intentionally narrow — never instrumented):**

`app/safety_guardian.py`, `app/eval_sandbox.py`, `app/goodhart_guard.py`,
`app/alignment_audit.py`, `app/human_gate.py`, all of `app/forge/`,
`app/transfer_memory/sanitizer.py` Tier-1 hard-rejects,
`refusal_detector.py` policy category, `llm_registry_scanner.py`
rejection-learning. Refusals from these are intentional and not
"reducing valve" candidates.

---

## 3. Architecture (one screen)

```
filter rejection ──>  valve_audit.log_rejection(filter_id, reason, score, threshold, ...)
                           │
                           └──>  workspace/logs/valve_audit.jsonl  (append-only, fire-and-forget)

  daily idle job ──>  valve_audit_replay.run_daily_replay()
                           │
                           ├──>  load yesterday's rejections
                           ├──>  stratified sample (≤50/filter, deterministic seed)
                           ├──>  per rejection:
                           │       (A) loose-replay: re-check at relaxed threshold (no LLM)
                           │       (B) LLM second-opinion (gated VALVE_AUDIT_LLM_REPLAY=1):
                           │             cross-family judge "would this be useful?"
                           │             → "useful" | "not_useful" | "unclear"
                           ├──>  workspace/logs/valve_audit_verdicts.jsonl
                           └──>  workspace/logs/valve_audit_summary.jsonl
                                   per-filter: rejections_total, sampled, DR, FRR,
                                   estimated_useful_lost_per_day, needs_review
```

**Log writes** are fire-and-forget. The logger swallows all exceptions —
the audit MUST NOT raise into a filter's hot path.

**Cost ceiling.** When LLM replay is on, calls are capped at
`VALVE_AUDIT_LLM_BUDGET_USD` per day (default $1). When the cap is hit,
sampling continues with loose-replay only.

---

## 4. Configuration

Environment variables (read at process startup; restart required):

| Variable | Default | Effect |
|---|---|---|
| `VALVE_AUDIT_ENABLED` | `1` | Master kill-switch for the logger. Set `0` to suppress all rejection logging. |
| `VALVE_AUDIT_LLM_REPLAY` | `0` | When `1`, the daily replay also runs the cross-family LLM second-opinion. Required for FRR (the headline metric). |
| `VALVE_AUDIT_LLM_BUDGET_USD` | `1.0` | Daily LLM-call budget. Sampling shrinks when hit; loose-replay continues. |

The May-2026 deployment runs with `VALVE_AUDIT_ENABLED=1` and
`VALVE_AUDIT_LLM_REPLAY=1` from the start so the 7-day checkpoint has
both signals.

---

## 5. Output schemas

### `valve_audit.jsonl` (per rejection)

```json
{
  "ts": "<UTC iso>",
  "filter_id": "F4",
  "callsite": "app/agents/commander/execution.py:33",
  "input_text": "<truncated to 4kB>",
  "input_hash": "<sha1[:16] of full text>",
  "reason": "too_short",
  "score": 11.0,
  "threshold": 20.0,
  "model_used": null,
  "extra": {"crew_name": "writing"}
}
```

### `valve_audit_verdicts.jsonl` (per replayed sample)

```json
{
  "replay_date": "2026-05-08",
  "rejection_id": "<ts>|<filter_id>|<input_hash>",
  "filter_id": "F4",
  "reason": "too_short",
  "loose_would_pass": true,
  "llm_verdict": "useful",
  "llm_rationale": "<≤200 chars>",
  "cost_usd": 0.0007
}
```

### `valve_audit_summary.jsonl` (one line per day, all filters)

```json
{
  "date": "2026-05-08",
  "rejections_total": 142,
  "sampled_total": 100,
  "cost_usd": 0.0823,
  "filters": [
    {
      "filter_id": "F4",
      "rejections_total": 90,
      "sampled": 50,
      "disagreement_rate": 0.18,
      "false_rejection_rate": 0.06,
      "estimated_useful_lost_per_day": 5.4,
      "needs_review": false
    }
  ],
  "criterion": {"frr_threshold": 0.15, "min_samples_for_review": 50}
}
```

---

## 6. Decision criterion

**A filter "needs review" when** FRR ≥ **0.15** with ≥ 50 sampled
rejections in a day's summary. The 0.15 threshold is empirically
calibrated: ~10 % is the irreducible model-disagreement noise floor
(per the user's prior 3-of-3 token-economy falsifications), and 0.15
is one decisive band above that.

The headline number per filter is `rejections_total × FRR` —
**estimated useful material lost per day**. This is what gets reported
to the operator, not a percentage in isolation.

A filter that flunks the criterion does **not** automatically widen.
The follow-on widening proposal goes through the existing
`governance_requests` channel for human approval, where it must produce
a **shadow-mode** test: filter still rejects, but the
would-have-passed-at-relaxed-threshold path is logged for outcome
verification before the threshold actually moves. This means the audit's
own findings face the same falsification burden the user applied to the
prior token-economy claims.

---

## 7. Operations

### Files

- `app/observability/valve_audit.py` — shared logger (~150 LOC)
- `app/observability/valve_audit_replay.py` — daily replay job (~330
  LOC)
- `tests/test_valve_audit.py` — 14 tests (logger schema, kill switch,
  truncation, exception swallow, replay loading, sampling determinism,
  loose-replay logic, persistence, LLM gating, integration)
- Instrumentation in: `app/recovery/refusal_detector.py`,
  `app/agents/commander/execution.py`, `app/companion/surfacing.py`
- Registration: `app/idle_scheduler.py` registers `valve-audit-replay`
  (LIGHT). Defaults to running for yesterday's UTC date.

### Adding a new filter

1. Verify the filter is in scope (not in TIER_IMMUTABLE, not safety /
   governance / sanitiser).
2. Add `valve_audit.log_rejection(...)` call on each rejection path.
   Pattern:
   ```python
   from app.observability import valve_audit
   valve_audit.log_rejection(
       filter_id="F<n>", callsite="path/to/file.py:<line>",
       input_text=<the rejected material>,
       reason="<categorical, stable>",
       score=<the predicate's score>, threshold=<the threshold>,
       extra={...},  # filter-specific
   )
   ```
3. If the filter is in a TIER_IMMUTABLE file (e.g. `vetting.py`),
   instrument at the **caller**, not inside the file.
4. Add a relaxed threshold to `_RELAXED_THRESHOLDS` in
   `valve_audit_replay.py` if loose-replay applies (i.e. the rejection
   is score-vs-threshold). Categorical reasons go in
   `_NO_REPLAY_REASONS`.
5. Add a test in `tests/test_valve_audit.py`.

### Manual replay

```bash
# Replay a specific date
python -c "from app.observability.valve_audit_replay import run_daily_replay; run_daily_replay(target_date='2026-05-08')"

# Replay yesterday (default)
python -c "from app.observability.valve_audit_replay import run_daily_replay; run_daily_replay()"
```

### Failure modes

- **No `valve_audit.jsonl`** — instrumentation never fired (system
  wasn't running, or `VALVE_AUDIT_ENABLED=0`). Empty summary; record
  the day as inconclusive.
- **`disagreement_rate` set, `false_rejection_rate: null`** —
  `VALVE_AUDIT_LLM_REPLAY` is off. Only DR is informative; FRR is the
  headline metric and must be on for the experiment criterion.
- **All filters show low DR but high FRR** — the relaxed thresholds
  in `_RELAXED_THRESHOLDS` are too tight. Consider lowering the
  relaxation level (separately from the actual filter's threshold).

---

## 8. Constraints and invariants

- **Measurement only.** No filter behaviour changes. Widening is a
  separate, governance-gated follow-on.
- **TIER_IMMUTABLE files are not modified.** F2/F3 (vetting) is
  documented as an instrumentation point at the **caller**, not
  inside `vetting.py`.
- **No PII / secrets.** `input_text` is truncated to 4 kB and hashed;
  the audit instruments quality / surfacing / refusal filters, NOT the
  sanitiser, which is intentionally Tier-1 hard-reject.
- **Fire-and-forget.** Logger swallows all exceptions to protect filter
  hot paths.
- **Self-Improver permissions: read-only.** The audit shapes how the
  system measures its own narrowness; allowing the Self-Improver to
  tune it would be a self-modeling integrity violation.

---

## 9. References

- Huxley, A. (1954). *The Doors of Perception* — "reducing valve" of
  Mind at Large. The framing the audit literalizes.
- Internal: `crewai-team/docs/RECOVERY_LOOP.md` (filter F1 lives in
  the recovery loop), `crewai-team/docs/COMPANION_LAYER.md` (filters
  F8–F10 live in the companion).
- Audit plan + filter inventory was produced as a planning agent
  output during the May-2026 build session (Phase H of the hardening
  pass).
