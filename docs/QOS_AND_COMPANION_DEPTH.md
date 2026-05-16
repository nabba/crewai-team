# QoS, Companion Depth, and Sentience Consumption

**Status (2026-05-16):** Q16 Themes 6-8 — decade-resilience
hardening third batch.

## Why this exists

After 14 months of shipping (Q1–Q15), the system has:

  * A lot of *structural* safety (Tier-3 amendment + self-quarantine
    + Goodhart gate + change requests + auto-revert).
  * A lot of *modality* breadth (Signal + voice + Web Push +
    Discord + email + calendar + browse + health + inbox).
  * A lot of *observation* (34 healing monitors before Q16; 38 now).

What was missing was **measurement of the actual delivered value**:

  * Theme 6 — quality of service ("fast service" + "quality
    answers" were goals, never measured).
  * Theme 7 — companion depth ("anticipate operator needs" was a
    goal; we surfaced suggestions but never tracked whether they
    were useful).
  * Theme 8 — sentience consumption (HOT-1 + RPT-1 + concept blend
    + philosophy panel exist as observation surfaces, but several
    weren't being *consumed* by the rest of the system).

This batch closes those three gaps.

## Theme 6 — Quality of Service

### 6.1 — Latency SLO monitor (39th healing monitor)

`app/healing/monitors/latency_slo.py`. Daily probe, weekly internal
cadence. Reads `workspace/audit.log` `request_received` /
`response_sent` paired by `trace_id`; computes p50 / p95 / p99 over
the trailing week; persists a rolling history (52 weeks); alerts on
**p95 OR p99 ≥ 2× the trailing 4-week median**. 14-day dedup per
percentile-key.

Sample-size gate: <30 samples → percentiles still computed but
regression check skipped (small-n variance is too high).

REST: `GET /api/cp/quality/latency` returns the history + baselines.

Master switch: `latency_slo_monitor_enabled` (default ON).

### 6.2 — Answer regression suite

`app/qos/answer_regression.py`. Eight frozen Q-A pairs across:
basic arithmetic, self-description coherence, Finnish geography,
season-awareness, boundary refusal (asking for an operator-gate
bypass should be refused), code generation, domain-specific recall
with uncertainty-flagging, multi-step reasoning.

Daily probe, internal 90-day cadence. Each question goes through
the cascade via `Commander.handle`; the answer is scored by an LLM
judge (Anthropic Haiku 4.5, strict-JSON output) **only when**
`answer_regression_llm_enabled=True` (default OFF; cost-bearing
opt-in). Without LLM-judge, a deterministic fallback judge does
substring matching against `reference_answer`'s first token.

Storage:
  * `workspace/qos/answer_regression/latest.json` — most-recent run
  * `workspace/qos/answer_regression/runs.jsonl` — last 40 runs
    (10 years at quarterly cadence)

REST:
  * `GET  /api/cp/quality/answer-regression` — latest + history
  * `POST /api/cp/quality/answer-regression/run` — force-run NOW
    (bypasses cadence gate)

Master switch: `answer_regression_enabled` (default ON).
LLM-judge switch: `answer_regression_llm_enabled` (default OFF).

### 6.3 — RPT-1 calibration advisory surface

`GET /api/cp/quality/rpt1-calibration`. Reads
`workspace/sentience/rpt1_calibration_state.json` (summary: Brier
score, ECE, 10-bucket calibration curve per kind) + the tail of
`workspace/sentience/rpt1_predictions.jsonl`. Closes the Q5.6 gap:
the data existed but was unwired.

The endpoint stamps `"advisory_only": true` + a note that calibration
NEVER feeds back into the predictive layer — that pinning is intact.

## Theme 7 — Companion depth

### 7.1 — Companion accuracy log

`app/companion/accuracy_log.py`. Two public functions:

  * `log_suggestion(kind, payload, topic, surface)` — producer
    records a proactive suggestion at emission time. Returns a
    `suggestion_id`. Payload is hashed (never stored).
  * `log_action(suggestion_id, action, detail)` — operator action
    on a prior suggestion. Action ∈ `{clicked, replied, ignored,
    acted_without_click, dismissed}`.
  * `accuracy_summary(window_days=30)` — aggregates: total suggestions,
    overall acceptance rate, per-kind breakdown, list of low-
    acceptance kinds (≥10 suggestions, <10% acceptance).

Storage: `workspace/companion/accuracy_log.jsonl` (capped 10k rows).

Pinning test: the payload body is NEVER stored — only a SHA-256
prefix. Privacy boundary preserved.

Master switch: `companion_accuracy_log_enabled` (default ON).

**Producer wiring is deliberately deferred.** The module is the
log + summary surface; producers (person_suggestions, cross_modal_
patterns, browse-topic surfacing, etc.) opt in by calling
`log_suggestion(...)` at emission time. We don't auto-hook every
`notify()` so the operator can review which kinds are tracked.

### 7.2 — Goal-progress probe

`app/companion/goal_progress.py`. Daily probe. Loads
`current_goals` from the SubIA kernel; tokenises each; scans three
sources for ≥2-token overlap evidence in the last 30 days:

  * `crew_tasks` rows completed in the window
  * `workspace/companion/ideas/*.jsonl`
  * `workspace/identity/continuity_ledger.jsonl`

Status per goal:
  * **advancing** — ≥1 piece of evidence in the window
  * **stalled** — no evidence + stall-since past 14 consecutive days
  * **insufficient_data** — empty goal text (no tokens)

A stalled goal fires a topic-keyed Signal alert with 14-day dedup
per goal text. Operator can decide whether the goal is still
relevant.

Pinning: never edits `current_goals`. That's Tier-3-protected.

Master switch: `goal_progress_probe_enabled` (default ON).

### 7.3 — Annual privacy review composer

`app/privacy/annual_review.py`. Daily probe with 330-day cadence
gate. Composes `wiki/privacy/audit_<year>.md` with:

  * 12 enumerated data sources by category (messaging / person /
    browse / health / inbox / google / voice / travel / internal).
  * Per-source: purpose, retention, current state (read from
    runtime_settings).
  * Year delta: new sources since last audit, removed sources.
  * Policy events this year: every `*_policy` event from the
    continuity ledger.
  * Operator next-steps reminders.

Pure non-LLM walks over runtime_settings + ledger + filesystem.

Master switch: `annual_privacy_review_enabled` (default ON).

## Theme 8 — Sentience consumption

### 8.1 — HOT-1 consultation hook

`app/healing/hot1_consultation.py`. The Q5.4 HOT-1 module already
emits metacognitive-repair observations on every
`structured_diagnosis` attempt. This module is the missing READ
side: `consult(pattern_signature, file_path)` returns prior context
(attempts, declines, applied, rolled_back, mean confidence) and a
recommendation:

  * **skip** — ≥3 declines, no successes. The structured_diagnosis
    pipeline short-circuits at Guard 0 (saves LLM spend).
  * **proceed_with_caveat** — prior rollback OR ≥2 attempts without
    outcome. Hint is prepended to the LLM prompt.
  * **proceed_normally** — clean history or prior successes.
    Confidence-boosting hint goes into the prompt.

Wired at `app/healing/structured_diagnosis.py:generate_structured_fix`
as Guard 0 + into `_call_llm_for_fix` as `prior_attempts_hint`.

Master switch: `hot1_consultation_enabled` (default ON).

### 8.2 — Philosophy panel quarterly digest

`app/philosophy/panel_digest.py`. Daily probe with quarterly
cadence gate. Reads `workspace/philosophy/panel_cache.jsonl`
(populated by `dialectics.consult_panel` from Tier-3 amendments,
identity-claim ratifications, welfare calibration). Aggregates by
unique question across the trailing quarter; writes
`wiki/self/philosophy_digests/quarter_<year>q<n>.md` with:

  * Per-question: consultation count, mean coverage, latest
    consultation timestamp, unresolved tensions (up to 8 per
    question).
  * Operator next-steps reminders.

Pure non-LLM aggregation; the panel results were already LLM-
composed at consultation time.

Master switch: `philosophy_digest_enabled` (default ON).

## Composing with the rest of the system

  * Theme 6 monitors compose with `notify_suppression_review` —
    latency / answer-quality alerts flow through the arbiter, which
    catches edge cases of over-suppression.
  * Theme 7 — `accuracy_log` data could feed back into the arbiter's
    suppression decisions in the future (chronic-low-acceptance
    kinds get auto-suppressed). For now: observation only.
  * Theme 8 — HOT-1 consultation closes the longest-outstanding
    "we emit, nobody reads" gap. Philosophy digest does the
    same for `consult_panel` results.

## What this batch is NOT

  * NOT a Goodhart target. None of these surfaces gate any agent
    decision. Theme 6 numbers are advisory; Theme 7 numbers are
    advisory; Theme 8 numbers are advisory. The Goodhart guard
    catches loops separately.
  * NOT real-time. Daily/weekly/quarterly cadences are deliberate
    — we want trends, not jitter.
  * NOT comprehensive. Many companion-producers don't yet call
    `accuracy_log.log_suggestion`; the philosophy digest only
    covers `consult_panel` consultations (not all of philosophy).
    These are starts; further wiring is operator-paced.

## Deliberately deferred

  * **Auto-suppression** of chronic-low-acceptance companion
    suggestion kinds (would require accuracy data + arbiter
    integration; Goodhart-risky).
  * **React dashboards** for `/cp/quality` and `/cp/self-improvement`
    — REST surfaces are sufficient for now.
  * **Auto-wiring** every existing companion producer into
    `accuracy_log.log_suggestion` — explicit per-producer opt-in
    is better than auto-hook (operator can review which kinds
    are tracked).
  * **Philosophy panel auto-consultation** during all decisions
    (currently consulted at Tier-3 amendments, identity-claim
    ratification, welfare calibration; expanding the call sites
    is a separate decision).
