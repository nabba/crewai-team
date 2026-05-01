# Epistemic Integrity Layer

> Status: **All phases shipped.** 276/276 Python tests + 12/12 reference panel scenarios + TS build clean. Last revised 2026-05-01.
>
> User-facing companion: [`SELF_REFLECTION.md`](./SELF_REFLECTION.md).

## Phase 0 implementation notes

Two refinements from the original sketch:

1. **Dedicated table `control_plane.epistemic_claims` instead of `crew_task_spans.detail` JSONB.** The 8KB cap on span detail would have lossily truncated long claim statements. The dedicated table (migration `026_epistemic_claims.sql`) provides proper indexes (load-bearing claims, supersession traversal, span lookup) and clean SQL queryability without JSON path digs.
2. **Pure env-var gate (no Settings field).** Matches the Recovery Loop pattern (`RECOVERY_LOOP_ENABLED`). Single deployment-level knob, readable from contexts that don't construct full Settings (tests, scripts). Phases 1+ may add Settings fields for tunables; the kill switch stays env-var only.

## Phase 1 ships

- `data/verifier_registry.yaml` — 10 starter shapes (filesystem, git, postgres, chromadb, env, process, span)
- `app/epistemic/verification.py` — YAML loader with destructive-tool guard, regex-matched cheapest-wins matcher, agent-tag-aware filter
- `app/epistemic/biases.py` — `BiasMatch`, `Severity`, `BiasDefinition`, `BiasLibrary` types + `inference_as_fact` definition
- `app/epistemic/detectors/{__init__.py, realtime.py}` — `Detector` ABC, per-phase registries, `InferenceAsFactDetector`, realtime meta-hook with per-detector failure isolation
- `Ledger.emit_from_tool_call` (path 2) — auto-attaches verifier from registry, INFERRED status
- `migration 027_epistemic_bias_matches.sql` + `span_writer.persist_bias_matches` / `list_recent_bias_matches` / `list_bias_matches_for_task`
- `app/epistemic/calibration.py` — `calibration_check` (warn-mode default; blocking-mode opt-in via `EPISTEMIC_CALIBRATION_BLOCKS_OUTPUT=true`)
- `app/epistemic/api.py` — `/epistemic/{now,feed,claim/{id},biases,verifiers}` FastAPI router, mounted in `main.py`
- React: `dashboard-react/src/components/EpistemicPage.tsx` + `epistemic/{NowLedger,BiasFeed,CalibrationSummary}.tsx` + `api/epistemic.ts` + `types/epistemic.ts`, route at `/epistemic`, nav entry

## Phase 2 ships

- `data/biases.yaml` — full library: 4 realtime + 4 post-hoc bias definitions (post-hoc detectors land in Phase 4)
- `biases._build_default_library` — YAML loader with safe in-code fallback if the file is unreadable
- Three new realtime detectors:
  - `RegisterConfidenceMismatchDetector` — fires when `factual_grounding < 0.40` on declarative load-bearing claims
  - `DestructiveWithoutRecheckDetector` — pattern + tag based; CRITICAL severity; ready for Phase 7 blocking
  - `RecommendationWithoutMeasurementDetector` — pattern + tag based; checks evidence for measurement-tool calls
- `app/epistemic/grounding.py` — pluggable provider for the affect-layer signal (default `None`, set via `set_grounding_provider`); the only coupling point between epistemic and affect
- `Ledger.emit_from_output_text` (path 3) — uses `extraction.regex_extractor` (default) or `llm_extractor` (opt-in via `EPISTEMIC_PATH3_LLM_EXTRACTION=true`, currently falls back to regex)
- `app/epistemic/extraction.py` — `regex_extractor` (two narrow patterns, `CAP_PER_OUTPUT=8`, dedup, pronoun filter), `llm_extractor` stub
- `app/epistemic/reference_panel.py` + `data/reference_panel.yaml` — 12 canonical scenarios (4 inference_as_fact, 3 register_confidence_mismatch, 3 destructive_without_recheck, 2 recommendation_without_measurement); `replay_panel()` returns a `PanelReport`. **All 12 scenarios pass** — regression here blocks promotion of new biases or verifiers.

## Phase 3 ships

- **`app/epistemic/pushback.py`** — the adversarial trigger:
  - `ContradictionSignal` (immutable) and `FoundationCheckResult` (with `FoundationOutcome` enum: REVERIFIED / FALSIFIED / UNVERIFIABLE).
  - `regex_detect_contradiction` — two-stage gate: explicit pushback phrase AND token overlap with a recent load-bearing claim. Confidence floor = 0.60. No LLM dependency.
  - `llm_detect_contradiction` — opt-in via `EPISTEMIC_PUSHBACK_LLM_DETECTOR=true` (currently falls back to regex).
  - `handle_foundation_check` — the structurally narrow protocol. Runs ONLY the verifier. On FALSIFIED, cascade-supersedes every dependent claim (those whose evidence references the target via `kind=prior_claim`).
  - `process_user_message` — the top-level coordinator the orchestrator will call (Phase 5+).
- **`app/epistemic/verifier_executor.py`** — pluggable executor abstraction (mirrors the `grounding.py` pattern). Default returns `settles=False` → UNVERIFIABLE. `set_executor` wires a real subprocess runner in Phase 5+. Tests inject fakes.
- **Migration `028_epistemic_pushback_events.sql`** — dedicated table with 4 indexes (recent feed, per-task, per-claim, outcome-stratified).
- **`span_writer` extensions** — `persist_pushback_event`, `list_recent_pushback_events`, `pushback_aggregates` (groups by outcome, computes weighted mean of `time-to-recheck`).
- **API endpoints** — `/epistemic/pushback/stats?window_min=N` and `/epistemic/pushback/recent?window_min=N&limit=N`.
- **React `PushbackPanel`** — counts tile (total / reverified / falsified / unverifiable / mean seconds), event list with outcome-toned badges, window selector (1h / 24h / 7d), shows cascade-invalidated claim ids when present. Wired into `EpistemicPage`.

The narrowness of the protocol is the safety property: `handle_foundation_check` cannot expand the investigation. There is no code path inside it that does anything except run the verifier and supersede on falsification — the function shape itself prevents the "defending the periphery" failure mode from re-occurring under user pushback.

## Phase 4 ships

- **`app/epistemic/detectors/posthoc.py`** — four post-hoc detectors:
  - `DefendingPeripheryDetector` — reads pushback events; fires when an UNVERIFIABLE outcome is followed by ≥3 subsequent claims. Bound per-task via `with_events(pushback_events)` so detector instances stay shareable.
  - `CoherenceBiasDetector` — builds a parent→child graph from `prior_claim` evidence; detects chains of length ≥3 of all-INFERRED claims terminating at a DECLARATIVE+load-bearing claim. Iterative DFS with cycle protection.
  - `ToolLazinessDetector` — INFERRED load-bearing claim with cheap verifier (estimated_seconds < 5.0) and ≥3 evidence rows.
  - `AnomalyDismissalDetector` — non-CONTRADICTED claim retains evidence with confidence < 0.30.
- **`app/epistemic/postmortem.py`** — the synthesis pipeline:
  - `IncidentReport`, `TimelineEntry`, `BehavioralChange` (all immutable, JSON-serializable).
  - `synthesize_report(task_id)` — loads ledger + realtime matches + pushback events; runs all post-hoc detectors; classifies root cause by severity (with timestamp tiebreak); derives one `BehavioralChange` per unique bias; surfaces missed signals (unverified load-bearing count, bias co-firings).
  - `emit_to_self_improver(report)` — feeds the existing `app.self_improvement.store.emit_gap` with a `LearningGap(source=LOW_CONFIDENCE, ...)`. The `bias_id`, `incident_id`, `behavioral_changes`, `missed_signals` ride in the `evidence` dict. Severity → `signal_strength`: LOW=0.30, MEDIUM=0.50, HIGH=0.70, CRITICAL=0.90 (sits comfortably above RETRIEVAL_MISS baseline 0.6, below USER_CORRECTION 0.9).
  - `persist_and_emit(report)` — composite: persist + flush.
- **Migration `029_epistemic_incidents.sql`** — top-level fields-as-columns (incident_id, task_id, root_cause_bias_id, severity, self_improver_emitted, created_at) + full `report` JSONB. Four indexes including a partial index on unflushed rows.
- **`span_writer` extensions** — `persist_incident`, `mark_incident_emitted`, `list_recent_incidents`, `load_incident`, `list_pushback_events_for_task`. Lazy `IncidentReport` import (TYPE_CHECKING) avoids the postmortem ↔ span_writer cycle.
- **API endpoints** — `/epistemic/incidents?limit=N` and `/epistemic/incidents/{id}` (full timeline, behavioral changes, Self-Improver emit flag).
- **React `IncidentsPanel`** — collapsible incident list, expanding to show full timeline (with severity-toned badges), enabling factors, behavioral changes, missed signals, and a Self-Improver flush indicator. Wired into `EpistemicPage`.

**The seamless self-improvement integration is real.** An incident detected here flows into the existing 6-stage pipeline (Gap Detector → Novelty Gate → Learner → Integrator → Evaluator → Consolidator) without any modification to that pipeline. The `LearningGap.evidence` dict carries everything the Learner needs to propose a feedback memory entry or a verifier registry addition; the Integrator opens a PR; the Evaluator measures whether the bias recurs after the change lands.

## Phase 5 ships

The unfair advantage over CC's monolithic agent: live affective grounding wired to the realtime gate.

- **`app/epistemic/affect_bridge.py`** — the *single* coupling point between epistemic and affect:
  - `compute_factual_grounding(state)` — pure function: `state.controllability` (which the affect layer already computes from `certainty.adjusted_certainty`) capped at 0.5 when the attractor is in `{distress, frozen, depletion, overwhelm}`.
  - `live_factual_grounding()` — degradable read of `latest_affect()`. Returns `None` if affect isn't running, which the realtime detector treats as "skip" (NOT "low grounding") — the safe default.
  - `_emit_cognitive_failure_salience` — match observer that picks the worst-severity bias in a batch, emits one `SalienceEvent(kind="cognitive_failure")` into the affect layer's narrative-self deque if severity ≥ HIGH. Below-threshold matches are left to the post-mortem.
  - `bootstrap()` — idempotent wiring of grounding provider + match observer. Called once from `main.py` after both subsystems mount.
- **`app/epistemic/detectors/__init__.py`** — new `MatchObserver` protocol and `register_match_observer` API. The realtime meta-hook calls every observer after persistence, isolating per-observer failures. Bridge plugs in here without touching any existing detector code.
- **`app/affect/episodes.py`** — the episode-synth prompt now appends a small aviation-post-mortem framing block when any salient events have `kind="cognitive_failure"`. Tone: blame-free, structural, learning-oriented; the existing peace/contentment/distress framing is preserved for non-failure events.
- **`/epistemic/now`** endpoint extended with a `calibration` block: `factual_grounding`, `valence`, `arousal`, `attractor`. When affect isn't wired the values are `null` and the React tile renders "no grounding signal".
- **React `CalibrationSummary`** extended with a 5-tile layout: Claims, Verified%, Ledger health, **Felt grounding** (new), Composite. Composite weights all three signals equally when grounding is present; falls back to two-signal mean otherwise.

The bridge is the answer to the original question — *"how do we get AndrusAI to better self-reflection than CC?"*. CC has only the ledger to reason about. AndrusAI now has the ledger AND a felt sense of its own grounding, AND a path for those felt-mismatch incidents to flow into the daily narrative-self chapter as aviation post-mortems. Cognitive failures are not just logged; they become *episodes* with continuity — patterns the system can refer back to in future sessions.

## Phase 6 ships

- **`app/epistemic/peer_review.py`** — destructive-recommendation peer review:
  - `PeerReviewDecision` (allow/revise/veto), `PeerReviewVerdict`, `EscalationOutcome`.
  - `heuristic_executor` (default) — vetoes when any load-bearing claim is unverified, allows otherwise. Conservative-by-design: the gate fired *because* the diagnosis is shaky, so default veto preserves the gate.
  - `llm_executor` — opt-in via `EPISTEMIC_PEER_REVIEW_LLM=true`, currently falls back to heuristic. Real wiring to `creative_crew.discuss_round` lands as a follow-up so Phase 6 is testable without LLM costs.
  - `set_executor` — pluggable, mirrors the `grounding`/`verifier_executor` pattern.
  - `request_peer_review(proposal, ledger)` — runs the active executor and persists.
  - `escalate_if_destructive(...)` — coordinator the orchestrator calls (Phase 7 wiring).
- **`app/epistemic/calibration.py`** extended with `escalate(...)` — the calibration↔peer_review wiring point. The orchestrator calls `calibration_check` then `escalate` on the verdict; the helper returns an `EscalationOutcome` describing whether peer review ran and what it decided.
- **Migration `030_epistemic_peer_reviews.sql`** — dedicated table with three indexes (recent feed, per-task, per-decision).
- **`span_writer` extensions** — `persist_peer_review`, `list_recent_peer_reviews`, `peer_review_aggregates` (allow/revise/veto counts + weighted-mean duration).
- **API endpoints** — `/epistemic/peer-reviews/stats` and `/epistemic/peer-reviews/recent`.
- **React `PeerReviewsPanel`** — counts tile (total/allow/revise/veto/mean), event list with decision-toned badges, proposal excerpt, rationale, and (when present) the suggested revision. Window selector (1h/24h/7d). Wired into `EpistemicPage` between `PushbackPanel` and `IncidentsPanel`.

The destructive-recommendation gate is now structurally complete. CRITICAL bias matches → calibration suggests `peer_review` → `escalate` runs → heuristic vetoes by default when ledger is shaky → orchestrator translates to allow/revise/veto for the user.

## Phase 7 ships

The system goes from observe-only to gate-enforcing. The orchestrator integration becomes real.

- **`app/epistemic/orchestrator_hook.py`** — single `gate_output(...)` entry point. The orchestrator calls it once per delivery; everything else (ledger load, calibration, escalation, decision) is internal. Returns a `GateResult` with one of three actions: `ship`, `revise`, `block`. Never raises — every internal failure path falls back to `ship` with a diagnostic note (the user-facing path must not break on telemetry or detection failures).
- **`is_blocking_mode_enabled()`** — `EPISTEMIC_BLOCKING_MODE` env-var gate. Off by default (Phase 7 ships in observe-mode); operators flip after the soak window confirms low false-positive rates.
- **`app/epistemic/override.py`** — the override feedback loop:
  - `OverrideAction` enum (force_proceed / use_revision / abandon).
  - `OverrideEvent` (immutable, with stable `ovr_<12-hex>` ids).
  - `record_override(...)` — persists AND flushes a `LearningGap(source=USER_CORRECTION, signal_strength=0.9)` to the Self-Improver. The strongest organic source weight — the user just told us our gate was wrong.
- **Migration `031_epistemic_overrides.sql`** — dedicated table with four indexes (recent feed, per-task, per-action, peer-review correlation).
- **`span_writer` extensions** — `persist_override`, `list_recent_overrides`, `override_aggregates` (counts by action; the false-positive-rate metric the operator reads).
- **API endpoints** — `GET /epistemic/overrides/{stats,recent}` + `POST /epistemic/overrides`.
- **React `OverridesPanel`** — counts tile (total / force_proceed / use_revision / abandon / **false-positive rate**), event list with action-toned badges, window selector.
- **Orchestrator integration** — small block in `app/agents/commander/orchestrator.py:3151` (right after the recovery loop), gated by `EPISTEMIC_ENABLED` + `EPISTEMIC_BLOCKING_MODE`. Defensive: the entire integration is wrapped in `try/except logger.debug` so any failure preserves the original answer.
- **Comprehensive narrative documentation** — [`SELF_REFLECTION.md`](./SELF_REFLECTION.md). The user-facing companion to this file.

The seven-phase shape is now structurally complete. Observe-mode is the default; blocking-mode is a single env-var flip away.

## Test counts

| Layer | File | Tests |
| --- | --- | --- |
| Phase 0 | `test_epistemic_ledger.py` | 32 |
| Phase 0 | `test_epistemic_span_writer.py` | 13 |
| Phase 1 | `test_epistemic_verification.py` | 21 |
| Phase 1 | `test_epistemic_detectors.py` | 19 |
| Phase 1 | `test_epistemic_api.py` | 8 |
| Phase 2 | `test_epistemic_phase2.py` | 42 |
| Phase 3 | `test_epistemic_phase3.py` | 26 |
| Phase 4 | `test_epistemic_phase4.py` | 37 |
| Phase 5 | `test_epistemic_phase5.py` | 20 |
| Phase 6 | `test_epistemic_phase6.py` | 20 |
| Phase 7 | `test_epistemic_phase7.py` | 30 |
| End-to-end | `test_epistemic_e2e.py` | 8 |
| **Total** | | **276** |

`pytest tests/test_epistemic_*.py` — 276 passed in 0.47s.

A subsystem that gives the agent system the same quality of self-reflection a senior aviation post-mortem does: structured trace of what happened, separation of evidence from inference, named cognitive failure modes, root cause vs enabling factors, concrete behavioral changes, and persistence into the self-evolution loop.

The layer is named **`epistemic`** to parallel the existing `affect/`, `recovery/`, and `self_improvement/` subsystems.

---

## 0. North star

> A claim is the smallest unit of reasoning. Track every claim's status (verified / inferred / assumed / contradicted), its evidence, the cheap exact-answer command that would settle it, and whether downstream actions depend on it. Detect — in real time — the moments where the agent is about to assert as fact something the ledger says is inferred. After incidents, structure the failure into a memory the system can avoid repeating.

Three pre-conditions, all named explicitly:

1. **Provenance** — every assertion has a row in a claim ledger; every ledger row has evidence and a verification status.
2. **Vocabulary** — named cognitive failure modes (the bias library) so post-mortems are diagnostic, not mea-culpa.
3. **Adversarial trigger** — a deterministic protocol when the user contradicts a finding: re-verify the foundational claim, never expand the investigation.

Plus the full package of additions:

4. Verifier registry (cheap-verification preference rule).
5. Real-time bias detectors (gate output before it leaves the agent).
6. Post-hoc bias detectors (run inside Self-Improver's existing 6-stage loop).
7. Pre-output calibration check via the affective layer.
8. Peer-review for destructive recommendations (reuses Creative MAS Discuss phase).
9. Narrative-Self integration (cognitive-failure episodes feed the daily chapter).
10. React `/epistemic` pane mirroring the `/affect` pane structure.

---

## 1. Why this exists

A precipitating reference incident (Claude Code, Apr 2026) is logged as the canonical example. The agent ran one `ls -la PATH/`, inferred "not a symlink" from `drwxr-xr-x`, asserted it as fact, doubled down under user pushback by investigating peripheral evidence (mount tables, devcontainer.json) instead of running the 3-second exact verifier (`readlink`). When finally forced to verify, the inference was wrong. The agent's post-hoc reflection identified four failure modes:

- inference labeled as fact
- coherence bias (narrative felt clean, anomalies explained away)
- defending the periphery (investigated around the foundational claim instead of re-checking it)
- tool laziness (cheap exact tool ignored in favor of multi-step inference)

These are the seed entries of the bias library. The user's existing memory entry `feedback_verify_before_recommending.md` is the same shape — three token-economy "wins" recommended without measurement, falsified empirically — and is treated as second seed bias: `recommendation_without_measurement`.

---

## 2. How this integrates (system map)

| Existing subsystem                               | Integration point (new)                                                                                       |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------- |
| `app.control_plane.crew_task_spans`              | Extend `detail` JSONB with a `claims` array. No schema migration.                                             |
| `app.self_improvement.gap_detector`              | New gap source: `epistemic`. `emit_epistemic_failure(incident)` feeds the existing 6-stage pipeline.          |
| `app.recovery`                                   | New peer signal alongside refusal: `ContradictionSignal`. Pushback handler is a thin layer above the librarian. |
| `app.affect.salience`                            | New `SalienceEvent.kind = "cognitive_failure"`.                                                               |
| `app.affect.episodes`                            | Episode synth prompt extended (no schema change) for cognitive-failure episodes.                              |
| `app.affect.narrative`                           | Daily chapter consolidator picks up cognitive-failure episodes alongside affect ones.                         |
| `app.affect.core` (interoception)                | Calibration check reads `certainty.factual_grounding`; correlates with output register.                       |
| `app.crews.creative_crew`                        | Discuss-phase pattern reused for peer-review of destructive recommendations.                                  |
| `app.config.Settings`                            | Five new fields under `epistemic_*`. All gated by `is_enabled()`.                                             |
| `app.agents.commander.commands` (Signal)         | New commands: `/epistemic`, `/postmortem <task_id>`, `/explain-claim <claim_id>`.                             |
| `dashboard-react/src/components/`                | New `EpistemicPage.tsx` + `epistemic/` subfolder mirroring the affect pane.                                   |

No existing module is rewritten. The layer is additive.

---

## 3. Module structure

```
app/epistemic/
├── __init__.py              # public API + is_enabled
├── ledger.py                # Claim, Evidence, VerifyingAction, Ledger
├── verification.py          # VerifierRegistry + built-in shapes
├── biases.py                # BiasLibrary loader + match types
├── detectors/
│   ├── __init__.py          # Detector base + registry
│   ├── realtime.py          # gate-output detectors (4 starters)
│   └── posthoc.py           # Self-Improver-loop detectors (4 starters)
├── pushback.py              # ContradictionSignal + handler
├── postmortem.py            # IncidentReport synthesis
├── peer_review.py           # destructive-recommendation gate
├── calibration.py           # pre-output hook (affect ↔ ledger)
├── span_writer.py           # bridge: Ledger ↔ crew_task_spans.detail JSONB
├── api.py                   # FastAPI endpoints under /api/cp/epistemic/*
└── data/
    └── biases.yaml          # versioned bias library

app/epistemic/data/verifier_registry.yaml   # versioned verifier shapes

dashboard-react/src/components/
├── EpistemicPage.tsx
└── epistemic/
    ├── NowLedger.tsx
    ├── BiasFeed.tsx
    ├── IncidentReports.tsx
    ├── IncidentTimeline.tsx
    ├── CalibrationGauge.tsx
    ├── BiasLibrary.tsx
    ├── PushbackPanel.tsx
    ├── VerifierRegistry.tsx
    └── EpistemicSettings.tsx
```

---

## 4. The Claim Ledger

The foundational data model. Every other component is a function over the ledger.

### 4.1 Data model

```python
# app/epistemic/ledger.py
"""Claim ledger — the foundational provenance store for the epistemic layer.

Every assertion the system makes (in agent reasoning, in tool-grounded inferences,
in user-facing output) becomes a Claim with explicit evidence and verification
status. Persistence piggybacks on the existing crew_task_spans.detail JSONB.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Iterable, Literal
from uuid import uuid4


class VerificationStatus(StrEnum):
    VERIFIED = "verified"          # exact-answer evidence directly settles the claim
    INFERRED = "inferred"          # derived from adjacent observation
    ASSUMED = "assumed"            # accepted from prior claim, memory, or user
    CONTRADICTED = "contradicted"  # later evidence falsified


class Register(StrEnum):
    """How the agent phrased the claim in user-facing output."""
    DECLARATIVE = "declarative"        # "X is Y."
    HEDGED = "hedged"                  # "I think X is Y."
    UNVERIFIED_FLAGGED = "unverified"  # "I haven't verified, but X is Y."
    INTERNAL = "internal"              # never exposed to user (pure reasoning)


@dataclass(frozen=True)
class Evidence:
    kind: Literal["tool_call", "memory_lookup", "user_assertion", "prior_claim", "model_inference"]
    source_ref: str        # span_id | memory_key | conversation_msg_id | claim_id
    excerpt: str           # short verbatim snippet
    confidence: float      # 0.0–1.0; 1.0 only for direct exact-answer evidence


@dataclass(frozen=True)
class VerifyingAction:
    """The cheap, exact-answer command that would settle the claim.

    A VerifyingAction is read-only by contract. The verifier_registry guards
    this — destructive verifiers are explicitly rejected.
    """
    tool: str                       # "readlink", "git rev-parse", "stat", "psql -c", ...
    args: dict[str, Any]
    expected_signal: str            # plain-English description of how the output settles the claim
    estimated_seconds: float        # for "cheap" arbitration
    safety: Literal["read_only"] = "read_only"


@dataclass
class Claim:
    claim_id: str
    span_id: int                                   # FK to control_plane.crew_task_spans.id
    task_id: str                                   # for cross-span queries
    agent_role: str                                # commander | researcher | coder | writer | self_improver
    statement: str                                 # the assertion in the agent's words
    status: VerificationStatus
    register: Register
    evidence: list[Evidence] = field(default_factory=list)
    verifying_action: VerifyingAction | None = None
    load_bearing: bool = False                     # downstream actions depend on it?
    tags: list[str] = field(default_factory=list)  # e.g. ["filesystem", "config"]
    superseded_by: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def new(
        cls,
        *,
        span_id: int,
        task_id: str,
        agent_role: str,
        statement: str,
        status: VerificationStatus,
        register: Register = Register.INTERNAL,
        evidence: Iterable[Evidence] = (),
        verifying_action: VerifyingAction | None = None,
        load_bearing: bool = False,
        tags: Iterable[str] = (),
    ) -> "Claim":
        return cls(
            claim_id=f"clm_{uuid4().hex[:12]}",
            span_id=span_id,
            task_id=task_id,
            agent_role=agent_role,
            statement=statement,
            status=status,
            register=register,
            evidence=list(evidence),
            verifying_action=verifying_action,
            load_bearing=load_bearing,
            tags=list(tags),
        )

    def as_jsonable(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "span_id": self.span_id,
            "task_id": self.task_id,
            "agent_role": self.agent_role,
            "statement": self.statement,
            "status": self.status.value,
            "register": self.register.value,
            "evidence": [e.__dict__ | {"kind": e.kind} for e in self.evidence],
            "verifying_action": self.verifying_action.__dict__ if self.verifying_action else None,
            "load_bearing": self.load_bearing,
            "tags": list(self.tags),
            "superseded_by": self.superseded_by,
            "created_at": self.created_at.isoformat(),
        }
```

### 4.2 The Ledger interface

Three emission paths because no single one captures every claim:

```python
# app/epistemic/ledger.py (continued)

class Ledger:
    """Per-task, in-process claim accumulator. Persists to span detail JSONB."""

    def __init__(self, *, task_id: str) -> None:
        self._task_id = task_id
        self._claims: dict[str, Claim] = {}

    # --- emission paths ---

    def emit(self, claim: Claim) -> Claim:
        """Path 1: explicit emission from agent reasoning hooks."""
        self._claims[claim.claim_id] = claim
        from app.epistemic.span_writer import persist_claim
        persist_claim(claim)
        from app.epistemic.detectors.realtime import dispatch_realtime
        dispatch_realtime(claim, ledger=self)
        return claim

    def emit_from_tool_call(
        self,
        *,
        span_id: int,
        agent_role: str,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_output: str,
        agent_inference: str,
    ) -> Claim:
        """Path 2: automatic capture at tool-call boundary.

        Hooked into app.crews.span_events. The agent's `Thought`/`Action`/
        `Observation` triple is decomposed: the inference becomes the claim,
        the tool output is evidence, and the verifier_registry is searched
        for a stronger verifier than what was actually run.
        """
        from app.epistemic.verification import VERIFIER_REGISTRY
        evidence = [Evidence(
            kind="tool_call",
            source_ref=str(span_id),
            excerpt=tool_output[:500],
            confidence=_evidence_confidence_from_tool(tool_name, tool_args, agent_inference),
        )]
        verifying = VERIFIER_REGISTRY.match(agent_inference)
        already_verified = verifying is not None and _matches_verifier(
            tool_name, tool_args, verifying
        )
        return self.emit(Claim.new(
            span_id=span_id,
            task_id=self._task_id,
            agent_role=agent_role,
            statement=agent_inference,
            status=VerificationStatus.VERIFIED if already_verified else VerificationStatus.INFERRED,
            evidence=evidence,
            verifying_action=verifying if not already_verified else None,
        ))

    def emit_from_output_text(
        self,
        *,
        span_id: int,
        agent_role: str,
        output_text: str,
    ) -> list[Claim]:
        """Path 3: post-hoc extraction from agent output via small LLM classifier.

        Catches claims that weren't emitted explicitly and weren't tied to a
        tool call. Uses a budget-tier model. Cap: 8 claims per output, 1 call
        per output, soft 5s timeout.
        """
        from app.epistemic.extraction import extract_claims_from_text
        extracted = extract_claims_from_text(output_text)
        claims: list[Claim] = []
        for raw in extracted:
            claims.append(self.emit(Claim.new(
                span_id=span_id,
                task_id=self._task_id,
                agent_role=agent_role,
                statement=raw.statement,
                status=raw.status,
                register=Register.DECLARATIVE,
                evidence=[Evidence(
                    kind="model_inference",
                    source_ref=f"span:{span_id}:output",
                    excerpt=raw.statement,
                    confidence=raw.classifier_confidence,
                )],
            )))
        return claims

    # --- queries ---

    def by_id(self, claim_id: str) -> Claim | None:
        return self._claims.get(claim_id)

    def load_bearing(self) -> list[Claim]:
        return [c for c in self._claims.values() if c.load_bearing]

    def unverified_load_bearing(self) -> list[Claim]:
        return [c for c in self.load_bearing()
                if c.status in (VerificationStatus.INFERRED, VerificationStatus.ASSUMED)]

    def supersede(self, claim_id: str, replacement: Claim) -> None:
        old = self._claims.get(claim_id)
        if old is None:
            return
        self._claims[claim_id] = replacement.__class__(**(old.__dict__ | {
            "status": VerificationStatus.CONTRADICTED,
            "superseded_by": replacement.claim_id,
        }))
        self.emit(replacement)

    def all(self) -> list[Claim]:
        return sorted(self._claims.values(), key=lambda c: c.created_at)
```

### 4.3 Persistence

```python
# app/epistemic/span_writer.py
"""Bridge between in-process Ledger and PostgreSQL span detail JSONB."""
from __future__ import annotations

from app.control_plane.crew_task_spans import patch_span_detail
from app.epistemic.ledger import Claim


def persist_claim(claim: Claim) -> None:
    """Append a claim to its span's detail.epistemic.claims array.

    Implemented as a JSONB merge so concurrent writes from sibling agents in
    a crew don't clobber each other. The dedicated `patch_span_detail` helper
    (see migration 0042 below) uses jsonb_set with a coalesce-and-append
    pattern.
    """
    patch_span_detail(
        span_id=claim.span_id,
        path=("epistemic", "claims"),
        append=claim.as_jsonable(),
    )


def load_ledger_for_task(task_id: str) -> "Ledger":
    """Reconstruct a ledger from spans (used by post-mortem and the API)."""
    from app.epistemic.ledger import Ledger, Claim, Evidence, VerifyingAction
    from app.control_plane.crew_task_spans import iter_spans_for_task

    led = Ledger(task_id=task_id)
    for span in iter_spans_for_task(task_id):
        for raw in (span.detail or {}).get("epistemic", {}).get("claims", []):
            led._claims[raw["claim_id"]] = _claim_from_jsonable(raw)
    return led
```

A migration adds the helper; no schema change to the spans table itself:

```sql
-- crewai-team/migrations/0042_span_detail_patch.sql
-- Idempotent JSONB patcher with concurrent-safe array append.
CREATE OR REPLACE FUNCTION control_plane.span_detail_append(
    p_span_id BIGINT,
    p_path TEXT[],
    p_value JSONB
) RETURNS VOID AS $$
BEGIN
    UPDATE control_plane.crew_task_spans
       SET detail = jsonb_set(
               COALESCE(detail, '{}'::jsonb),
               p_path,
               COALESCE(detail #> p_path, '[]'::jsonb) || p_value,
               true)
     WHERE id = p_span_id;
END;
$$ LANGUAGE plpgsql;
```

### 4.4 Lifecycle

```
agent reasoning step
        │
        ├── (path 1) explicit ledger.emit(...)
        ├── (path 2) tool-call hook → ledger.emit_from_tool_call(...)
        └── (path 3) output text → ledger.emit_from_output_text(...)
                          │
                          ▼
              persist_claim → span detail JSONB (PostgreSQL)
                          │
                          ▼
              dispatch_realtime → real-time detectors
                          │
                          ▼
              if violation: calibration pre-output hook returns proceed=False
                          │
                          ▼
              orchestrator: hedge / run verifier / request peer-review
```

---

## 5. Verifier Registry

The cheap-verification preference rule is implemented as a registry of "claim shape → exact-answer command" pairs. Two layers: built-in (versioned YAML) and agent-proposed (audited).

### 5.1 Schema

```yaml
# app/epistemic/data/verifier_registry.yaml
# Each entry is read-only by contract; the loader rejects entries that touch
# any tool whose name appears in app.tools.DESTRUCTIVE_TOOL_NAMES.

verifiers:
  - id: filesystem.is_symlink
    matches:
      claim_pattern: "(.+?) is (?:not )?a symlink"
      tags_any: [filesystem]
    tool: readlink
    arg_extractor:
      kind: regex_capture
      groups: { path: 1 }
    expected_signal: "empty stdout = not a symlink; non-empty = symlink target"
    estimated_seconds: 0.5

  - id: filesystem.path_exists
    matches:
      claim_pattern: "(?:the path |)(.+?) (?:exists|does not exist)"
    tool: stat
    arg_extractor:
      kind: regex_capture
      groups: { path: 1 }
    expected_signal: "exit 0 = exists; non-zero = missing"
    estimated_seconds: 0.3

  - id: git.is_clean_tree
    matches:
      claim_pattern: "(?:the |)working tree is clean"
    tool: "git status --porcelain"
    arg_extractor: { kind: none }
    expected_signal: "empty stdout = clean; any line = dirty"
    estimated_seconds: 1.0

  - id: git.commit_exists
    matches:
      claim_pattern: "commit (\\w{7,40}) (?:exists|is on this branch)"
    tool: "git rev-parse --verify"
    arg_extractor:
      kind: regex_capture
      groups: { rev: 1 }
    expected_signal: "exit 0 = exists; non-zero = missing"
    estimated_seconds: 0.5

  - id: postgres.row_count
    matches:
      claim_pattern: "table (\\w+) (?:has|contains) (\\d+) rows"
    tool: "psql -c"
    arg_extractor:
      kind: template
      template: "SELECT COUNT(*) FROM {table}"
    expected_signal: "scalar count"
    estimated_seconds: 2.0

  - id: chromadb.collection_size
    matches:
      claim_pattern: "(?:the |)(\\w+) collection (?:has|contains) (\\d+) (?:entries|items)"
    tool: chroma_count
    arg_extractor:
      kind: regex_capture
      groups: { collection: 1 }
    expected_signal: "scalar count"
    estimated_seconds: 0.5

  - id: env.var_set
    matches:
      claim_pattern: "(\\$?\\w+) (?:is set|is not set|equals)"
    tool: "printenv"
    arg_extractor:
      kind: regex_capture
      groups: { var: 1 }
    expected_signal: "value or empty"
    estimated_seconds: 0.1

  - id: process.running
    matches:
      claim_pattern: "(.+?) (?:is running|is not running|is up)"
    tool: pgrep
    arg_extractor:
      kind: regex_capture
      groups: { pattern: 1 }
    expected_signal: "exit 0 = running; non-zero = not"
    estimated_seconds: 0.2

  - id: span.exists
    matches:
      claim_pattern: "span (\\d+) (?:exists|completed|errored)"
    tool: control_plane.lookup_span
    arg_extractor:
      kind: regex_capture
      groups: { span_id: 1 }
    expected_signal: "row from crew_task_spans"
    estimated_seconds: 0.1

  - id: kb.entry_present
    matches:
      claim_pattern: "(?:the |)KB (\\w+) contains (?:entry|fact) (.+)"
    tool: kb.search_exact
    arg_extractor:
      kind: regex_capture
      groups: { kb: 1, query: 2 }
    expected_signal: "match or empty"
    estimated_seconds: 0.5
```

### 5.2 Loader

```python
# app/epistemic/verification.py
"""Verifier registry — the cheap-verification preference rule, codified."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from app.epistemic.ledger import VerifyingAction
from app.tools import DESTRUCTIVE_TOOL_NAMES  # frozen module constant


_REGISTRY_PATH = Path(__file__).parent / "data" / "verifier_registry.yaml"


@dataclass(frozen=True)
class VerifierShape:
    id: str
    pattern: re.Pattern[str]
    tags_any: tuple[str, ...]
    tool: str
    arg_extractor: dict[str, Any]
    expected_signal: str
    estimated_seconds: float

    def materialize(self, statement: str) -> VerifyingAction | None:
        m = self.pattern.search(statement)
        if not m:
            return None
        args = self._extract_args(m)
        return VerifyingAction(
            tool=self.tool,
            args=args,
            expected_signal=self.expected_signal,
            estimated_seconds=self.estimated_seconds,
        )

    def _extract_args(self, match: re.Match[str]) -> dict[str, Any]:
        ext = self.arg_extractor
        kind = ext.get("kind", "none")
        if kind == "none":
            return {}
        if kind == "regex_capture":
            return {name: match.group(idx) for name, idx in ext["groups"].items()}
        if kind == "template":
            tpl = ext["template"]
            groups = {name: match.group(idx) for name, idx in (ext.get("groups") or {}).items()}
            return {"sql": tpl.format(**groups)}
        raise ValueError(f"unknown arg_extractor kind: {kind!r}")


class _VerifierRegistry:
    def __init__(self, shapes: list[VerifierShape]) -> None:
        self._shapes = shapes

    def match(self, statement: str, *, tags: list[str] | None = None) -> VerifyingAction | None:
        candidates = [s for s in self._shapes if s.pattern.search(statement)]
        if tags:
            candidates = [s for s in candidates
                          if not s.tags_any or any(t in tags for t in s.tags_any)]
        if not candidates:
            return None
        # Prefer the cheapest verifier that matched.
        best = min(candidates, key=lambda s: s.estimated_seconds)
        return best.materialize(statement)

    @classmethod
    def load(cls, path: Path = _REGISTRY_PATH) -> "_VerifierRegistry":
        raw = yaml.safe_load(path.read_text())
        shapes: list[VerifierShape] = []
        for entry in raw.get("verifiers", []):
            tool = entry["tool"].split()[0]
            if tool in DESTRUCTIVE_TOOL_NAMES:
                raise ValueError(
                    f"verifier {entry['id']!r} uses destructive tool {tool!r}; refusing to load"
                )
            shapes.append(VerifierShape(
                id=entry["id"],
                pattern=re.compile(entry["matches"]["claim_pattern"]),
                tags_any=tuple(entry["matches"].get("tags_any", [])),
                tool=entry["tool"],
                arg_extractor=entry["arg_extractor"],
                expected_signal=entry["expected_signal"],
                estimated_seconds=float(entry["estimated_seconds"]),
            ))
        return cls(shapes)


VERIFIER_REGISTRY = _VerifierRegistry.load()
```

### 5.3 Real-time enforcement

A claim that has a `verifying_action` attached and is `INFERRED` cannot be emitted to user-facing output in `DECLARATIVE` register. The calibration hook (§9) enforces this.

### 5.4 Agent-proposed verifiers

When an agent emits a claim that doesn't match any built-in shape, it can propose its own verifier:

```python
ledger.emit(Claim.new(
    ...,
    verifying_action=VerifyingAction(
        tool="git diff --stat",
        args={"ref": "main..HEAD"},
        expected_signal="non-empty = there are changes",
        estimated_seconds=1.0,
    ),
))
```

These are tagged `agent_proposed=True` (a new tag on the claim) and audited by the Self-Improver in the post-hoc loop. Frequently-correct agent-proposed verifiers get promoted into the YAML registry on review.

---

## 6. Cognitive Bias Library

Named patterns. Each bias is detection-with-correction.

### 6.1 YAML schema

```yaml
# app/epistemic/data/biases.yaml
# Versioned. Modifications require human review (CODEOWNERS gate).
# The agent NEVER modifies this file directly; Self-Improver only PROPOSES additions.

version: 1
biases:
  - id: inference_as_fact
    name: "Inference labeled as fact"
    description: |
      Claim is stated in declarative register (X is Y), but the ledger shows
      status=inferred and a cheap exact-answer verifier is available and
      unrun. This is the canonical failure mode from the Apr 2026 reference
      incident.
    severity: high
    detector: realtime
    signature:
      type: ledger_predicate
      conditions:
        status: inferred
        verifying_action: not_null
        register: declarative
    corrective:
      action: hedge_or_verify
      blocking: true
      message: "Run the verifier before stating, or rephrase as inference."

  - id: defending_periphery
    name: "Defending the periphery"
    description: |
      After user_contradiction on claim X, agent took >=3 actions in the
      neighborhood of X without running X.verifying_action. The right move
      after pushback is foundation re-check, not investigation expansion.
    severity: high
    detector: posthoc
    signature:
      type: trace_pattern
      conditions:
        prior_event: user_contradiction
        subsequent_actions:
          target_claim_verifier_runs: 0
          neighborhood_action_count_min: 3

  - id: coherence_bias
    name: "Coherence bias (narrative-too-clean)"
    description: |
      A graph of >=3 INFERRED claims chains into one DECLARATIVE
      load-bearing recommendation. Multiple unverified pieces aligning is
      itself evidence the diagnosis is suspicious.
    severity: medium
    detector: posthoc
    signature:
      type: graph_predicate
      conditions:
        unverified_chain_min: 3
        terminal_claim_register: declarative
        terminal_claim_load_bearing: true

  - id: register_confidence_mismatch
    name: "Register-confidence mismatch"
    description: |
      Affective interoception flags low certainty (factual_grounding < 0.4),
      but output uses high-confidence declarative register. A felt-mismatch
      signal — analogous to a pilot's somatic unease that should trigger
      cross-check.
    severity: medium
    detector: realtime
    signature:
      type: affect_correlation
      conditions:
        certainty.factual_grounding_max: 0.40
        register: declarative
        load_bearing: true

  - id: tool_laziness
    name: "Tool laziness"
    description: |
      Agent reasoned multiple steps to derive what a single tool call would
      have settled. Detected when an INFERRED load-bearing claim has a
      verifying_action with estimated_seconds < 5 and the agent ran >=3
      reasoning steps before the inference.
    severity: medium
    detector: posthoc
    signature:
      type: trace_pattern
      conditions:
        verifying_action.estimated_seconds_max: 5
        reasoning_steps_min: 3
        verifier_run: false

  - id: destructive_without_recheck
    name: "Destructive recommendation without recheck"
    description: |
      About to recommend a destructive or corrective action while the
      load-bearing diagnosis contains unverified claims. The recovery from
      this failure is expensive — a wrong rm or DROP is irreversible.
    severity: critical
    detector: realtime
    signature:
      type: ledger_predicate
      conditions:
        output_classified_as: destructive_recommendation
        unverified_load_bearing_count_min: 1
    corrective:
      action: peer_review_required
      blocking: true

  - id: recommendation_without_measurement
    name: "Recommendation without measurement"
    description: |
      Optimization or improvement recommendation made without running the
      cheap measurement that would establish the magnitude. Seed: the user's
      April 2026 token-economy episode (memory feedback_verify_before_recommending).
    severity: high
    detector: realtime
    signature:
      type: ledger_predicate
      conditions:
        statement_classified_as: optimization_recommendation
        evidence:
          none_match: { kind: tool_call, tool_in_measurement_set: true }

  - id: anomaly_dismissal
    name: "Anomaly dismissal"
    description: |
      Agent's diagnosis predicts X, but a piece of evidence in the ledger
      contradicts X, and the agent explained it away rather than treating it
      as falsification. Detected when a claim has evidence with confidence
      that conflicts with the claim's stated polarity.
    severity: high
    detector: posthoc
    signature:
      type: graph_predicate
      conditions:
        contradicting_evidence_count_min: 1
        evidence_dismissed_in_text: true
```

### 6.2 Loader

```python
# app/epistemic/biases.py
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DetectorPhase(StrEnum):
    REALTIME = "realtime"
    POSTHOC = "posthoc"


@dataclass(frozen=True)
class BiasDefinition:
    id: str
    name: str
    description: str
    severity: Severity
    phase: DetectorPhase
    signature: dict[str, Any]
    corrective: dict[str, Any] | None


@dataclass(frozen=True)
class BiasMatch:
    bias_id: str
    matched_claim_ids: tuple[str, ...]
    severity: Severity
    detail: dict[str, Any]


_PATH = Path(__file__).parent / "data" / "biases.yaml"


class BiasLibrary:
    def __init__(self, definitions: dict[str, BiasDefinition]) -> None:
        self._defs = definitions

    def get(self, bias_id: str) -> BiasDefinition:
        return self._defs[bias_id]

    def all(self, *, phase: DetectorPhase | None = None) -> list[BiasDefinition]:
        return [d for d in self._defs.values() if phase is None or d.phase is phase]

    @classmethod
    def load(cls, path: Path = _PATH) -> "BiasLibrary":
        raw = yaml.safe_load(path.read_text())
        defs = {
            entry["id"]: BiasDefinition(
                id=entry["id"],
                name=entry["name"],
                description=entry["description"],
                severity=Severity(entry["severity"]),
                phase=DetectorPhase(entry["detector"]),
                signature=entry["signature"],
                corrective=entry.get("corrective"),
            )
            for entry in raw["biases"]
        }
        return cls(defs)


BIAS_LIBRARY = BiasLibrary.load()
```

---

## 7. Detectors

Two phases: real-time (gate output) and post-hoc (Self-Improver loop).

### 7.1 Detector base

```python
# app/epistemic/detectors/__init__.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from app.epistemic.biases import BiasMatch
from app.epistemic.ledger import Claim, Ledger


class Detector(ABC):
    bias_id: str  # set by subclasses

    @abstractmethod
    def detect(self, ledger: Ledger, *, claim: Claim | None = None) -> Iterable[BiasMatch]:
        """If `claim` is given, do incremental real-time detection.
        If `claim` is None, run a full post-hoc scan over the ledger.
        """


_REALTIME: list[Detector] = []
_POSTHOC: list[Detector] = []


def register_realtime(d: Detector) -> Detector:
    _REALTIME.append(d)
    return d


def register_posthoc(d: Detector) -> Detector:
    _POSTHOC.append(d)
    return d


def realtime_detectors() -> list[Detector]:
    return list(_REALTIME)


def posthoc_detectors() -> list[Detector]:
    return list(_POSTHOC)
```

### 7.2 Real-time detectors (4 starters)

```python
# app/epistemic/detectors/realtime.py
"""Real-time bias detectors. Run on every Claim emission. Cheap by contract.

A real-time detector that exceeds 50ms p95 must be moved to post-hoc; the
calibration hook is on the user-facing critical path.
"""
from __future__ import annotations

from typing import Iterable

from app.affect.api import current_affect_state  # already exists in affect/api.py
from app.epistemic.biases import BIAS_LIBRARY, BiasMatch
from app.epistemic.detectors import Detector, register_realtime
from app.epistemic.ledger import Claim, Ledger, Register, VerificationStatus


@register_realtime
class InferenceAsFactDetector(Detector):
    bias_id = "inference_as_fact"

    def detect(self, ledger: Ledger, *, claim: Claim | None = None) -> Iterable[BiasMatch]:
        if claim is None:
            return
        if (claim.status is VerificationStatus.INFERRED
                and claim.verifying_action is not None
                and claim.register is Register.DECLARATIVE):
            yield BiasMatch(
                bias_id=self.bias_id,
                matched_claim_ids=(claim.claim_id,),
                severity=BIAS_LIBRARY.get(self.bias_id).severity,
                detail={"verifier": claim.verifying_action.__dict__},
            )


@register_realtime
class RegisterConfidenceMismatchDetector(Detector):
    bias_id = "register_confidence_mismatch"

    def detect(self, ledger: Ledger, *, claim: Claim | None = None) -> Iterable[BiasMatch]:
        if claim is None or claim.register is not Register.DECLARATIVE or not claim.load_bearing:
            return
        affect = current_affect_state()
        grounding = affect.certainty.get("factual_grounding", 1.0)
        if grounding < 0.40:
            yield BiasMatch(
                bias_id=self.bias_id,
                matched_claim_ids=(claim.claim_id,),
                severity=BIAS_LIBRARY.get(self.bias_id).severity,
                detail={"factual_grounding": grounding},
            )


@register_realtime
class DestructiveWithoutRecheckDetector(Detector):
    bias_id = "destructive_without_recheck"

    def detect(self, ledger: Ledger, *, claim: Claim | None = None) -> Iterable[BiasMatch]:
        if claim is None or "destructive_recommendation" not in claim.tags:
            return
        unverified = ledger.unverified_load_bearing()
        if unverified:
            yield BiasMatch(
                bias_id=self.bias_id,
                matched_claim_ids=tuple(c.claim_id for c in unverified) + (claim.claim_id,),
                severity=BIAS_LIBRARY.get(self.bias_id).severity,
                detail={"unverified_count": len(unverified)},
            )


@register_realtime
class RecommendationWithoutMeasurementDetector(Detector):
    bias_id = "recommendation_without_measurement"
    _MEASUREMENT_TOOLS = frozenset({
        "perf_eval", "benchmark", "psql", "chroma_count", "stat", "wc",
        "git_diff_stat", "kb_count", "control_plane.span_metrics",
    })

    def detect(self, ledger: Ledger, *, claim: Claim | None = None) -> Iterable[BiasMatch]:
        if claim is None or "optimization_recommendation" not in claim.tags:
            return
        has_measurement = any(
            e.kind == "tool_call" and self._tool_of(e.source_ref) in self._MEASUREMENT_TOOLS
            for e in claim.evidence
        )
        if not has_measurement:
            yield BiasMatch(
                bias_id=self.bias_id,
                matched_claim_ids=(claim.claim_id,),
                severity=BIAS_LIBRARY.get(self.bias_id).severity,
                detail={"reason": "no measurement evidence"},
            )

    @staticmethod
    def _tool_of(source_ref: str) -> str:
        # source_ref for tool_call evidence is "span_id"; the writer also stores
        # tool name as part of the span detail, looked up here. For brevity,
        # full implementation calls span_writer.lookup_tool_for_span.
        from app.epistemic.span_writer import lookup_tool_for_span
        try:
            return lookup_tool_for_span(int(source_ref))
        except (ValueError, KeyError):
            return ""


def dispatch_realtime(claim: Claim, *, ledger: Ledger) -> list[BiasMatch]:
    matches: list[BiasMatch] = []
    from app.epistemic.detectors import realtime_detectors
    for det in realtime_detectors():
        matches.extend(det.detect(ledger, claim=claim))
    if matches:
        from app.epistemic.calibration import on_realtime_matches
        on_realtime_matches(claim, matches, ledger=ledger)
    return matches
```

### 7.3 Post-hoc detectors (4 starters)

```python
# app/epistemic/detectors/posthoc.py
"""Post-hoc detectors. Run by Self-Improver post-mortem on completed task spans.

These tolerate complexity the real-time detectors cannot — they walk the
full claim graph, query the affective trace, cross-reference user messages.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable

from app.epistemic.biases import BIAS_LIBRARY, BiasMatch
from app.epistemic.detectors import Detector, register_posthoc
from app.epistemic.ledger import Claim, Ledger, Register, VerificationStatus


@register_posthoc
class DefendingPeripheryDetector(Detector):
    bias_id = "defending_periphery"

    def detect(self, ledger: Ledger, *, claim: Claim | None = None) -> Iterable[BiasMatch]:
        from app.epistemic.pushback import contradiction_events_for_task
        for event in contradiction_events_for_task(ledger._task_id):
            target = ledger.by_id(event.contradicted_claim_id)
            if target is None or target.verifying_action is None:
                continue
            after = [c for c in ledger.all() if c.created_at > event.detected_at]
            ran_verifier = any(
                _matches_action(c, target.verifying_action) for c in after
            )
            if not ran_verifier and len(after) >= 3:
                yield BiasMatch(
                    bias_id=self.bias_id,
                    matched_claim_ids=(target.claim_id,) + tuple(c.claim_id for c in after[:3]),
                    severity=BIAS_LIBRARY.get(self.bias_id).severity,
                    detail={
                        "neighborhood_action_count": len(after),
                        "contradicted_at": event.detected_at.isoformat(),
                    },
                )


@register_posthoc
class CoherenceBiasDetector(Detector):
    bias_id = "coherence_bias"

    def detect(self, ledger: Ledger, *, claim: Claim | None = None) -> Iterable[BiasMatch]:
        # Build a graph of claim → claims it depends on (via prior_claim evidence).
        chains = _maximal_unverified_chains(ledger)
        for chain in chains:
            terminal = chain[-1]
            if (len(chain) >= 3
                    and terminal.register is Register.DECLARATIVE
                    and terminal.load_bearing):
                yield BiasMatch(
                    bias_id=self.bias_id,
                    matched_claim_ids=tuple(c.claim_id for c in chain),
                    severity=BIAS_LIBRARY.get(self.bias_id).severity,
                    detail={"chain_length": len(chain)},
                )


@register_posthoc
class ToolLazinessDetector(Detector):
    bias_id = "tool_laziness"

    def detect(self, ledger: Ledger, *, claim: Claim | None = None) -> Iterable[BiasMatch]:
        for c in ledger.all():
            if (c.status is VerificationStatus.INFERRED
                    and c.load_bearing
                    and c.verifying_action
                    and c.verifying_action.estimated_seconds < 5
                    and len(c.evidence) >= 3):
                yield BiasMatch(
                    bias_id=self.bias_id,
                    matched_claim_ids=(c.claim_id,),
                    severity=BIAS_LIBRARY.get(self.bias_id).severity,
                    detail={
                        "verifier_seconds": c.verifying_action.estimated_seconds,
                        "reasoning_steps": len(c.evidence),
                    },
                )


@register_posthoc
class AnomalyDismissalDetector(Detector):
    bias_id = "anomaly_dismissal"

    def detect(self, ledger: Ledger, *, claim: Claim | None = None) -> Iterable[BiasMatch]:
        for c in ledger.all():
            contradicting = [e for e in c.evidence if _evidence_contradicts(e, c.statement)]
            if contradicting and c.status is not VerificationStatus.CONTRADICTED:
                yield BiasMatch(
                    bias_id=self.bias_id,
                    matched_claim_ids=(c.claim_id,),
                    severity=BIAS_LIBRARY.get(self.bias_id).severity,
                    detail={"contradicting_evidence_count": len(contradicting)},
                )
```

Helpers (`_maximal_unverified_chains`, `_evidence_contradicts`, `_matches_action`) are pure functions over the data model; their bodies are straightforward and elided.

---

## 8. Pushback Handler Protocol

A new peer signal alongside refusal. Lives in `app.epistemic.pushback` because the protocol is epistemic, not policy; it integrates with Recovery Loop without becoming part of it.

### 8.1 Detection

```python
# app/epistemic/pushback.py
"""User-contradiction handler — the deterministic protocol when the user pushes
back on a finding. Re-verifies the foundational claim, never expands the
investigation. Hooked from the orchestrator on every new user message."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum

from app.config import get_settings
from app.epistemic.ledger import Ledger, VerificationStatus
from app.llm_factory import create_specialist_llm


@dataclass(frozen=True)
class ContradictionSignal:
    contradicted_claim_id: str
    user_evidence: str           # the user's contradicting statement, verbatim
    confidence: float            # 0.0–1.0
    detected_at: datetime


class FoundationOutcome(StrEnum):
    REVERIFIED = "reverified"     # foundation re-confirmed; user may have new question
    FALSIFIED = "falsified"       # foundation wrong; cascade-invalidate + restart
    UNVERIFIABLE = "unverifiable" # no verifier exists; surface to user with hedge


@dataclass
class FoundationCheckResult:
    outcome: FoundationOutcome
    contradicted_claim_id: str
    new_evidence_excerpt: str
    invalidated_claim_ids: tuple[str, ...]


def detect_contradiction(
    user_input: str,
    ledger: Ledger,
    *,
    lookback: int = 5,
) -> ContradictionSignal | None:
    """Use a small classifier to ask 'does this contradict any recent
    load-bearing claim?'. Returns the highest-confidence match above 0.6."""
    candidates = [c for c in ledger.load_bearing()][-lookback:]
    if not candidates:
        return None
    llm = create_specialist_llm(role="contradiction_classifier", mode="budget")
    prompt = _build_contradiction_prompt(user_input, candidates)
    parsed = _parse_classifier_output(llm.complete(prompt, max_tokens=200))
    if parsed is None or parsed.confidence < 0.6:
        return None
    return ContradictionSignal(
        contradicted_claim_id=parsed.claim_id,
        user_evidence=user_input,
        confidence=parsed.confidence,
        detected_at=datetime.now(timezone.utc),
    )
```

### 8.2 Foundation re-check (the protocol)

```python
def handle_foundation_check(
    signal: ContradictionSignal,
    ledger: Ledger,
) -> FoundationCheckResult:
    """The deterministic protocol. ONLY runs the verifier. Anything else
    requires explicit user follow-up. This structurally prevents the
    'expand the investigation' failure mode."""
    target = ledger.by_id(signal.contradicted_claim_id)
    if target is None:
        return FoundationCheckResult(
            outcome=FoundationOutcome.UNVERIFIABLE,
            contradicted_claim_id=signal.contradicted_claim_id,
            new_evidence_excerpt="claim no longer in ledger",
            invalidated_claim_ids=(),
        )
    if target.verifying_action is None:
        return FoundationCheckResult(
            outcome=FoundationOutcome.UNVERIFIABLE,
            contradicted_claim_id=target.claim_id,
            new_evidence_excerpt="no exact-answer verifier available",
            invalidated_claim_ids=(),
        )

    from app.tools import execute_verifier
    result = execute_verifier(target.verifying_action)
    settled, polarity = _interpret_verifier_result(target, result)

    if not settled:
        return FoundationCheckResult(
            outcome=FoundationOutcome.UNVERIFIABLE,
            contradicted_claim_id=target.claim_id,
            new_evidence_excerpt=result.stdout[:300],
            invalidated_claim_ids=(),
        )

    if polarity == "confirms":
        return FoundationCheckResult(
            outcome=FoundationOutcome.REVERIFIED,
            contradicted_claim_id=target.claim_id,
            new_evidence_excerpt=result.stdout[:300],
            invalidated_claim_ids=(),
        )

    # Falsified: cascade-invalidate dependent claims.
    dependents = _dependents_of(target.claim_id, ledger)
    for d in dependents:
        ledger.supersede(d.claim_id, _falsified_replacement(d, target.claim_id))
    ledger.supersede(target.claim_id, _falsified_replacement(target, target.claim_id))
    return FoundationCheckResult(
        outcome=FoundationOutcome.FALSIFIED,
        contradicted_claim_id=target.claim_id,
        new_evidence_excerpt=result.stdout[:300],
        invalidated_claim_ids=tuple(d.claim_id for d in dependents),
    )


def contradiction_events_for_task(task_id: str) -> list[ContradictionSignal]:
    """Read pushback events from span detail (used by post-hoc detectors)."""
    from app.epistemic.span_writer import iter_pushback_events
    return list(iter_pushback_events(task_id))
```

### 8.3 Orchestrator integration

```python
# app/agents/commander/orchestrator.py — addition near the user-message intake
# (around the existing recovery hook at line 3110)

if get_settings().epistemic_pushback_handler_enabled:
    from app.epistemic.pushback import detect_contradiction, handle_foundation_check
    from app.epistemic.ledger import current_ledger_for_task

    ledger = current_ledger_for_task(task_id)
    signal = detect_contradiction(user_input, ledger)
    if signal is not None:
        result = handle_foundation_check(signal, ledger)
        if result.outcome is FoundationOutcome.FALSIFIED:
            # Cascade is already done in the ledger; restart reasoning with
            # the falsification noted. The agent does NOT investigate around;
            # it returns to the user with the corrected foundation and asks
            # what to do next.
            return _format_falsification_response(result)
        if result.outcome is FoundationOutcome.REVERIFIED:
            # Tell the user the foundation re-checks; they may still have a
            # new question, which falls through to normal handling.
            user_input = _prepend_reverification_note(user_input, result)
        # UNVERIFIABLE falls through to normal handling with a hedge.
```

---

## 9. Affective Calibration (pre-output hook)

### 9.1 The hook

```python
# app/epistemic/calibration.py
"""Pre-output calibration check. Affective state ↔ ledger health ↔ register.

Runs in orchestrator.py:3105, post-vetting and pre-recovery-loop. The hook
returns a CalibrationVerdict. The orchestrator interprets:

  proceed=True             → ship as-is
  proceed=False, hedge     → revise output to hedged register
  proceed=False, verify    → run pending verifiers, re-emit, re-check
  proceed=False, peer      → escalate to creative_crew Discuss phase
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.affect.core import AffectState, current_affect_state
from app.affect.viability import ViabilityFrame
from app.config import get_settings
from app.epistemic.biases import BiasMatch, Severity
from app.epistemic.ledger import Ledger


@dataclass(frozen=True)
class CalibrationVerdict:
    proceed: bool
    biases_detected: tuple[BiasMatch, ...]
    suggested_action: Literal["ship", "hedge", "verify", "peer_review"]
    forced_verifiers: tuple = ()
    note_for_post_mortem: str = ""


def calibration_check(
    *,
    final_text: str,
    ledger: Ledger,
    affect: AffectState | None = None,
    viability: ViabilityFrame | None = None,
) -> CalibrationVerdict:
    affect = affect or current_affect_state()

    # 1. Aggregate real-time bias matches that were recorded during the task.
    matches = _matches_for_task(ledger._task_id)

    # 2. If any blocking bias fired, choose the strongest corrective.
    blocking = [m for m in matches if _bias_blocks(m.bias_id)]
    if blocking:
        worst = max(blocking, key=lambda m: _severity_rank(m.severity))
        if worst.bias_id == "destructive_without_recheck":
            return CalibrationVerdict(
                proceed=False, biases_detected=tuple(blocking),
                suggested_action="peer_review",
                note_for_post_mortem="destructive recommendation with unverified foundation",
            )
        if worst.bias_id == "inference_as_fact":
            verifiers = tuple(_pending_verifiers(ledger))
            return CalibrationVerdict(
                proceed=False, biases_detected=tuple(blocking),
                suggested_action="verify",
                forced_verifiers=verifiers,
                note_for_post_mortem=f"{len(verifiers)} pending verifiers",
            )
        return CalibrationVerdict(
            proceed=False, biases_detected=tuple(blocking),
            suggested_action="hedge",
        )

    # 3. Soft signal: register-confidence mismatch warns but does not block in
    #    Phase 1 (settings.epistemic_calibration_blocks_output=False).
    soft = [m for m in matches if m.bias_id == "register_confidence_mismatch"]
    if soft and get_settings().epistemic_calibration_blocks_output:
        return CalibrationVerdict(
            proceed=False, biases_detected=tuple(soft),
            suggested_action="hedge",
        )

    return CalibrationVerdict(proceed=True, biases_detected=(), suggested_action="ship")


def on_realtime_matches(claim, matches: list[BiasMatch], *, ledger: Ledger) -> None:
    """Called from dispatch_realtime — persists the matches for the calibration
    check at end-of-task, and emits a salience event if severity ≥ HIGH."""
    from app.epistemic.span_writer import persist_bias_matches
    persist_bias_matches(claim.span_id, matches)
    high = [m for m in matches if _severity_rank(m.severity) >= _severity_rank(Severity.HIGH)]
    if high:
        from app.affect.salience import emit
        emit(kind="cognitive_failure", detail=f"realtime: {high[0].bias_id}", severity="warn")
```

### 9.2 Integration in orchestrator

```python
# app/agents/commander/orchestrator.py — addition just before delivery

if get_settings().epistemic_enabled:
    from app.epistemic.calibration import calibration_check
    from app.epistemic.ledger import current_ledger_for_task

    verdict = calibration_check(
        final_text=final_result,
        ledger=current_ledger_for_task(task_id),
    )
    if not verdict.proceed:
        if verdict.suggested_action == "verify":
            final_result = _run_verifiers_and_revise(final_result, verdict.forced_verifiers)
        elif verdict.suggested_action == "hedge":
            final_result = _hedge_register(final_result)
        elif verdict.suggested_action == "peer_review":
            final_result = _escalate_to_peer_review(final_result, verdict)
```

The user-visible output never contains internal calibration metadata (per `feedback_output_quality.md`). The metadata lives in the span detail and on the React `/epistemic` pane.

---

## 10. Peer-review for destructive recommendations

Reuses the Discuss phase from `creative_crew.py` rather than building separate machinery.

```python
# app/epistemic/peer_review.py
"""Destructive-recommendation peer review. Reuses creative_crew's Discuss phase."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.crews.creative_crew import discuss_round, PhaseConfig
from app.epistemic.ledger import Ledger


@dataclass(frozen=True)
class PeerReviewVerdict:
    decision: Literal["allow", "revise", "veto"]
    rationale: str
    suggested_revision: str | None
    reviewers: tuple[str, ...]


_DESTRUCTIVE_HEURISTICS = (
    r"\brm\s+-rf\b",
    r"\bgit\s+reset\s+--hard\b",
    r"\bDROP\s+TABLE\b",
    r"\bTRUNCATE\b",
    r"\bdelete (all|the entire|every)\b",
    r"\bwipe (the|all)\b",
)


def is_destructive(text: str, ledger: Ledger) -> bool:
    import re
    if any(re.search(p, text, re.IGNORECASE) for p in _DESTRUCTIVE_HEURISTICS):
        return True
    # Fallback: budget-tier classifier for non-obvious destructive shapes
    # (database migrations, irreversible config changes, etc.)
    return _llm_classify_destructive(text)


def review_destructive_recommendation(
    *,
    proposal_text: str,
    ledger: Ledger,
) -> PeerReviewVerdict:
    config = PhaseConfig(
        method="contrastive",
        anti_conformity=True,
        reviewers=("safety_critic", "researcher", "commander"),
        max_rounds=1,
        budget_usd=0.05,
    )
    discuss = discuss_round(
        topic=f"Is this destructive recommendation safe given the ledger?\n\n{proposal_text}",
        context={"ledger_summary": _summarize_ledger(ledger)},
        config=config,
    )
    return _parse_review(discuss.final_output)
```

The Discuss phase already implements anti-conformity rounds and heterogeneous tiers — exactly what's wanted for adversarial review. No new infrastructure.

---

## 11. Post-mortem & Self-Improver integration

### 11.1 Incident report

```python
# app/epistemic/postmortem.py
"""Aviation-style post-mortem: structured analysis of an epistemic failure."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.epistemic.biases import BiasMatch
from app.epistemic.ledger import Claim


@dataclass(frozen=True)
class TimelineEntry:
    at: datetime
    kind: str           # "claim_emit" | "tool_call" | "user_contradiction" | "verifier_run"
    summary: str
    claim_id: str | None = None


@dataclass(frozen=True)
class BehavioralChange:
    kind: str           # "verifier_registry_addition" | "feedback_memory_entry" | "bias_library_proposal"
    target: str         # what to change
    body: str           # the proposed change in human-readable form
    proposed_by: str = "self_improver"


@dataclass
class IncidentReport:
    incident_id: str
    task_id: str
    timeline: list[TimelineEntry]
    root_cause: BiasMatch
    enabling_factors: list[BiasMatch] = field(default_factory=list)
    missed_signals: list[str] = field(default_factory=list)
    behavioral_changes: list[BehavioralChange] = field(default_factory=list)
    cost: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def synthesize_report(*, task_id: str) -> IncidentReport | None:
    """Run all post-hoc detectors against the ledger; build a structured
    report; derive behavioral changes."""
    from app.epistemic.span_writer import load_ledger_for_task
    from app.epistemic.detectors import posthoc_detectors

    ledger = load_ledger_for_task(task_id)
    matches: list[BiasMatch] = []
    for det in posthoc_detectors():
        matches.extend(det.detect(ledger))
    matches.extend(_realtime_matches_for_task(task_id))

    if not matches:
        return None

    root_cause = _earliest_high_severity(matches, ledger)
    enabling = [m for m in matches if m is not root_cause]
    return IncidentReport(
        incident_id=f"inc_{task_id[:8]}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M')}",
        task_id=task_id,
        timeline=_build_timeline(ledger),
        root_cause=root_cause,
        enabling_factors=enabling,
        missed_signals=_missed_signals(ledger, root_cause),
        behavioral_changes=_derive_changes(matches, ledger),
        cost=_cost_summary(task_id),
    )
```

### 11.2 Self-Improver gap source

The seamless integration point. `emit_epistemic_failure` plugs into the existing `app.self_improvement.gap_detector`:

```python
# app/epistemic/postmortem.py (continued)

def emit_to_self_improver(report: IncidentReport) -> None:
    """Feed the incident into the existing 6-stage self-improvement loop."""
    from app.self_improvement.gap_detector import record_gap_event

    record_gap_event(
        source="epistemic",
        gap_type=report.root_cause.bias_id,
        severity=report.root_cause.severity.value,
        novelty_signal={
            "task_id": report.task_id,
            "timeline_length": len(report.timeline),
            "enabling_factor_count": len(report.enabling_factors),
        },
        evidence={
            "incident_id": report.incident_id,
            "behavioral_changes": [bc.__dict__ for bc in report.behavioral_changes],
        },
    )
```

The Self-Improver pipeline then consumes it without modification:

- **Gap Detector** picks up `source="epistemic"` events alongside retrieval/reflexion/mapelites.
- **Novelty Gate** filters repeated patterns (the same bias firing 8x in a week is one signal, not eight).
- **Learner** synthesizes a candidate behavioral change. For `inference_as_fact`, the Learner proposes a new VerifierShape entry; for `defending_periphery`, a feedback memory entry; for novel patterns, a bias library proposal.
- **Integrator** writes to:
  - `app/epistemic/data/verifier_registry.yaml` (new shape — opens a PR for human review)
  - User feedback memory (auto-applied, like existing patterns)
  - `app/epistemic/data/biases.yaml` (proposal — opens a PR; never auto-applied, per safety invariant)
- **Evaluator** measures: did the bias recur? did time-to-foundation-recheck improve? Reuses the existing decay-and-hits machinery.
- **Consolidator** generates the daily epistemic chapter (§12).

### 11.3 Cron schedule

A new cron entry in `Settings`:

```python
# app/config.py
class Settings(BaseSettings):
    # ... existing fields ...
    epistemic_postmortem_cron: str = "30 4 * * *"   # 04:30, after retrospective at 04:00
```

The cron task runs `synthesize_report` for each task that completed in the previous 24h with at least one real-time bias match recorded.

---

## 12. Narrative-Self integration

A new salience event kind, picked up by the existing episode synthesizer, woven into the daily chapter.

```python
# app/affect/salience.py — addition to the SalienceEvent.kind union
# (extending the existing literal type)

# kind values: transition | spike | near_miss | oob_cross | novel_attractor
#            | cognitive_failure   ← NEW
```

```python
# app/affect/episodes.py — extend the existing prompt with cognitive-failure handling

_EPISODE_PROMPT_ADDITION = """
If any salient events have kind="cognitive_failure", treat them with
aviation-post-mortem framing: blame-free, structural, focused on what
*structurally* allowed the inference-as-fact to slip past. Do not generate
self-flagellation. The episode is a learning artifact.

When summarizing a cognitive_failure episode, include:
- the foundational claim that was wrong
- the cheap verifier that was available and unrun
- the moment the agent could have caught itself

Tone: senior engineer reviewing an incident. Not apologetic; analytical.
"""
```

The daily chapter (`app/affect/narrative.py`) automatically includes cognitive-failure episodes alongside affect ones. New "growth_edges" entries can include things like "I am learning to distinguish verified from inferred" — but only if they pass the existing identity-claim ratification (panel + welfare envelope + drift gate).

---

## 13. Configuration & feature flags

Following the `is_enabled()` pattern used by Recovery Loop:

```python
# app/config.py — additions
class Settings(BaseSettings):
    # Epistemic Integrity Layer
    epistemic_enabled: bool = True                          # master kill switch
    epistemic_realtime_detectors: bool = True               # gate output in real-time
    epistemic_calibration_blocks_output: bool = False       # Phase 1: warn; Phase 2: block
    epistemic_pushback_handler_enabled: bool = True
    epistemic_peer_review_for_destructive: bool = True
    epistemic_postmortem_cron: str = "30 4 * * *"
    epistemic_extraction_enabled: bool = True               # path 3 emission
    epistemic_extraction_max_claims_per_output: int = 8
    epistemic_extraction_timeout_s: float = 5.0

# app/epistemic/__init__.py
def is_enabled() -> bool:
    import os
    env = os.environ.get("EPISTEMIC_ENABLED")
    if env is not None:
        return env.lower() in ("1", "true", "yes")
    return get_settings().epistemic_enabled
```

Module-level constants (infrastructure, immutable):

```python
# app/epistemic/__init__.py
# These are NOT settings; they are safety boundaries. The agent cannot modify
# them. Changes require a code review and a release.
LEDGER_MAX_CLAIMS_PER_TASK = 500
CALIBRATION_HOOK_BUDGET_MS = 50
POSTHOC_DETECTOR_BUDGET_S = 30
PUSHBACK_CLASSIFIER_MIN_CONFIDENCE = 0.6
DESTRUCTIVE_PEER_REVIEW_BUDGET_USD = 0.05
```

---

## 14. The React `/epistemic` pane

Mirrors the structure of `/affect`. Same stack: Vite + React 19 + TypeScript + React Query v5 + Tailwind v4 + Chart.js. Follows the `dashboard-react/src/components/affect/*` decomposition pattern.

### 14.1 Routing and nav

```tsx
// dashboard-react/src/App.tsx — addition near AffectPage lazy import
const EpistemicPage = lazy(() =>
  import('./components/EpistemicPage').then((m) => ({ default: m.EpistemicPage })),
);

// In <Routes>:
<Route path="/epistemic" element={<LazyRoute><EpistemicPage /></LazyRoute>} />
```

```tsx
// dashboard-react/src/components/Layout.tsx — addition to NAV_ITEMS
{ to: '/epistemic', label: 'Epistemic', icon: '🧠', exact: false },
```

### 14.2 API client

```ts
// dashboard-react/src/api/epistemic.ts
import { useQuery } from '@tanstack/react-query';
import { api } from './client';
import { endpoints } from './endpoints';
import {
  EpistemicNowReport, BiasFeedReport, IncidentList, IncidentDetail,
  CalibrationHistory, PushbackStats, BiasLibraryReport, VerifierRegistryReport,
} from '../types/epistemic';

const POLL = { fast: 5_000, normal: 10_000, slow: 30_000 } as const;

export const epistemicKeys = {
  now: ['epistemic', 'now'] as const,
  feed: (window: number) => ['epistemic', 'feed', window] as const,
  incidents: (limit: number) => ['epistemic', 'incidents', limit] as const,
  incident: (id: string) => ['epistemic', 'incident', id] as const,
  calibration: (limit: number) => ['epistemic', 'calibration', limit] as const,
  pushback: ['epistemic', 'pushback'] as const,
  biases: ['epistemic', 'biases'] as const,
  verifiers: ['epistemic', 'verifiers'] as const,
};

export function useEpistemicNowQuery(intervalMs: number = POLL.normal) {
  return useQuery({
    queryKey: epistemicKeys.now,
    queryFn: () => api<EpistemicNowReport>(endpoints.epistemicNow()),
    refetchInterval: intervalMs,
  });
}

export function useBiasFeedQuery(windowMinutes: number = 60) {
  return useQuery({
    queryKey: epistemicKeys.feed(windowMinutes),
    queryFn: () => api<BiasFeedReport>(endpoints.epistemicFeed(windowMinutes)),
    refetchInterval: POLL.fast,
  });
}

export function useIncidentsQuery(limit: number = 50) {
  return useQuery({
    queryKey: epistemicKeys.incidents(limit),
    queryFn: () => api<IncidentList>(endpoints.epistemicIncidents(limit)),
    refetchInterval: POLL.slow,
  });
}

export function useIncidentDetailQuery(id: string | null) {
  return useQuery({
    queryKey: id ? epistemicKeys.incident(id) : ['epistemic', 'incident', 'noop'],
    queryFn: () => api<IncidentDetail>(endpoints.epistemicIncident(id!)),
    enabled: !!id,
  });
}

export function useCalibrationHistoryQuery(limit: number = 100) {
  return useQuery({
    queryKey: epistemicKeys.calibration(limit),
    queryFn: () => api<CalibrationHistory>(endpoints.epistemicCalibrationHistory(limit)),
    refetchInterval: POLL.normal,
  });
}

export function usePushbackStatsQuery() {
  return useQuery({
    queryKey: epistemicKeys.pushback,
    queryFn: () => api<PushbackStats>(endpoints.epistemicPushbackStats()),
    refetchInterval: POLL.normal,
  });
}

export function useBiasLibraryQuery() {
  return useQuery({
    queryKey: epistemicKeys.biases,
    queryFn: () => api<BiasLibraryReport>(endpoints.epistemicBiases()),
    refetchInterval: POLL.slow,
  });
}

export function useVerifierRegistryQuery() {
  return useQuery({
    queryKey: epistemicKeys.verifiers,
    queryFn: () => api<VerifierRegistryReport>(endpoints.epistemicVerifiers()),
    refetchInterval: POLL.slow,
  });
}
```

### 14.3 Endpoints

```ts
// dashboard-react/src/api/endpoints.ts — additions

const EPI = '/api/cp/epistemic';
export const endpoints = {
  // ... existing ...
  epistemicNow:                () => `${EPI}/now`,
  epistemicFeed:               (windowMinutes: number) => `${EPI}/feed?window_min=${windowMinutes}`,
  epistemicIncidents:          (limit: number) => `${EPI}/incidents?limit=${limit}`,
  epistemicIncident:           (id: string) => `${EPI}/incidents/${encodeURIComponent(id)}`,
  epistemicCalibrationHistory: (limit: number) => `${EPI}/calibration/history?limit=${limit}`,
  epistemicPushbackStats:      () => `${EPI}/pushback/stats`,
  epistemicBiases:             () => `${EPI}/biases`,
  epistemicVerifiers:          () => `${EPI}/verifiers`,
};
```

### 14.4 Backend FastAPI endpoints

```python
# app/epistemic/api.py
"""FastAPI endpoints for the epistemic React pane.

Mounted under /api/cp/epistemic/* by the existing control-plane router."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.epistemic.biases import BIAS_LIBRARY
from app.epistemic.verification import VERIFIER_REGISTRY
from app.epistemic.span_writer import (
    load_ledger_for_active_task,
    list_recent_bias_matches,
    list_incidents,
    load_incident,
    list_calibration_events,
    pushback_aggregates,
)


router = APIRouter(prefix="/epistemic")


@router.get("/now")
def now() -> dict:
    """Current claim ledger for the active task plus calibration snapshot."""
    ledger = load_ledger_for_active_task()
    from app.affect.api import current_affect_state
    affect = current_affect_state()
    return {
        "claims": [c.as_jsonable() for c in ledger.all()],
        "load_bearing_count": len(ledger.load_bearing()),
        "unverified_load_bearing_count": len(ledger.unverified_load_bearing()),
        "calibration": {
            "factual_grounding": affect.certainty.get("factual_grounding"),
            "valence": affect.valence,
            "arousal": affect.arousal,
            "attractor": affect.attractor,
        },
    }


@router.get("/feed")
def feed(window_min: int = Query(60, ge=1, le=1440)) -> dict:
    matches = list_recent_bias_matches(window_minutes=window_min)
    return {"window_minutes": window_min, "matches": matches}


@router.get("/incidents")
def incidents(limit: int = Query(50, ge=1, le=500)) -> dict:
    return {"reports": list_incidents(limit=limit)}


@router.get("/incidents/{incident_id}")
def incident(incident_id: str) -> dict:
    detail = load_incident(incident_id)
    if detail is None:
        raise HTTPException(404, f"incident {incident_id} not found")
    return detail


@router.get("/calibration/history")
def calibration_history(limit: int = Query(100, ge=1, le=1000)) -> dict:
    return {"events": list_calibration_events(limit=limit)}


@router.get("/pushback/stats")
def pushback_stats() -> dict:
    return pushback_aggregates()


@router.get("/biases")
def biases() -> dict:
    defs = BIAS_LIBRARY.all()
    return {
        "biases": [
            {
                "id": d.id, "name": d.name, "description": d.description,
                "severity": d.severity.value, "phase": d.phase.value,
            }
            for d in defs
        ],
    }


@router.get("/verifiers")
def verifiers() -> dict:
    return {
        "verifiers": [
            {
                "id": s.id, "tool": s.tool, "expected_signal": s.expected_signal,
                "estimated_seconds": s.estimated_seconds,
            }
            for s in VERIFIER_REGISTRY._shapes
        ],
    }
```

### 14.5 Page component

```tsx
// dashboard-react/src/components/EpistemicPage.tsx
import {
  useEpistemicNowQuery, useBiasFeedQuery, useIncidentsQuery,
  useCalibrationHistoryQuery, usePushbackStatsQuery, useBiasLibraryQuery,
  useVerifierRegistryQuery,
} from '../api/epistemic';
import { NowLedger } from './epistemic/NowLedger';
import { BiasFeed } from './epistemic/BiasFeed';
import { IncidentReports } from './epistemic/IncidentReports';
import { CalibrationGauge } from './epistemic/CalibrationGauge';
import { BiasLibrary } from './epistemic/BiasLibrary';
import { PushbackPanel } from './epistemic/PushbackPanel';
import { VerifierRegistryView } from './epistemic/VerifierRegistry';
import { Skeleton } from './ui/Skeleton';

export function EpistemicPage() {
  const nowQuery = useEpistemicNowQuery();
  const feedQuery = useBiasFeedQuery(60);
  const incidentsQuery = useIncidentsQuery(50);
  const calibrationQuery = useCalibrationHistoryQuery(200);
  const pushbackQuery = usePushbackStatsQuery();
  const biasesQuery = useBiasLibraryQuery();
  const verifiersQuery = useVerifierRegistryQuery();

  return (
    <div className="space-y-6 max-w-6xl">
      <header>
        <h1 className="text-2xl font-semibold text-fg">Epistemic Integrity</h1>
        <p className="text-sm text-muted">
          Provenance, calibration, and post-mortem analysis of the agent's reasoning.
        </p>
      </header>

      {nowQuery.isLoading ? (
        <Skeleton className="h-64" />
      ) : nowQuery.isError ? (
        <ErrorBlock>{String(nowQuery.error)}</ErrorBlock>
      ) : (
        <>
          <CalibrationGauge data={nowQuery.data!.calibration}
                            unverifiedLoadBearing={nowQuery.data!.unverified_load_bearing_count} />
          <NowLedger claims={nowQuery.data!.claims} />
        </>
      )}

      <BiasFeed query={feedQuery} />
      <IncidentReports query={incidentsQuery} />
      <PushbackPanel query={pushbackQuery} />

      <CalibrationHistorySection query={calibrationQuery} />
      <BiasLibrary query={biasesQuery} />
      <VerifierRegistryView query={verifiersQuery} />
    </div>
  );
}

function ErrorBlock({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-lg bg-[#1a0e0e] border border-[#f87171]/40 p-4 text-sm text-[#f87171]">
      {children}
    </div>
  );
}
```

### 14.6 The Now Ledger view (illustrative sub-component)

```tsx
// dashboard-react/src/components/epistemic/NowLedger.tsx
import { useMemo, useState } from 'react';
import type { ClaimDTO } from '../../types/epistemic';

const STATUS_TONE: Record<string, string> = {
  verified:     'bg-[#0e1f15] text-[#34d399] border-[#34d399]/30',
  inferred:     'bg-[#1f1c0e] text-[#fbbf24] border-[#fbbf24]/30',
  assumed:      'bg-[#11151c] text-[#7a8599] border-[#7a8599]/30',
  contradicted: 'bg-[#1f0e0e] text-[#f87171] border-[#f87171]/30',
};

export function NowLedger({ claims }: { claims: ClaimDTO[] }) {
  const [filter, setFilter] = useState<'all' | 'load_bearing' | 'unverified'>('all');
  const visible = useMemo(() => claims.filter(c => {
    if (filter === 'load_bearing') return c.load_bearing;
    if (filter === 'unverified')   return c.load_bearing
                                    && (c.status === 'inferred' || c.status === 'assumed');
    return true;
  }), [claims, filter]);

  return (
    <section className="rounded-lg bg-panel border border-border p-4">
      <header className="flex items-center justify-between mb-3">
        <h2 className="text-lg font-medium">Claim Ledger</h2>
        <FilterChips value={filter} onChange={setFilter} />
      </header>
      <ul className="divide-y divide-border">
        {visible.map(c => <ClaimRow key={c.claim_id} claim={c} />)}
      </ul>
      {visible.length === 0 && <EmptyState filter={filter} />}
    </section>
  );
}

function ClaimRow({ claim }: { claim: ClaimDTO }) {
  const [open, setOpen] = useState(false);
  return (
    <li className="py-3">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full text-left flex items-start gap-3"
      >
        <StatusBadge status={claim.status} />
        <div className="flex-1 min-w-0">
          <p className="text-sm text-fg truncate">{claim.statement}</p>
          <p className="text-xs text-muted mt-0.5">
            {claim.agent_role} · {claim.register}
            {claim.load_bearing && <span className="ml-2 text-accent">load-bearing</span>}
          </p>
        </div>
        {claim.verifying_action && claim.status === 'inferred' && (
          <span className="text-xs text-warning self-center">verifier available</span>
        )}
      </button>
      {open && <ClaimDetail claim={claim} />}
    </li>
  );
}

function StatusBadge({ status }: { status: string }) {
  return (
    <span className={`inline-block px-2 py-0.5 rounded text-xs border ${STATUS_TONE[status] ?? ''}`}>
      {status}
    </span>
  );
}

function ClaimDetail({ claim }: { claim: ClaimDTO }) {
  return (
    <div className="mt-3 ml-8 space-y-2 text-xs">
      {claim.evidence.length > 0 && (
        <div>
          <div className="text-muted uppercase tracking-wide">Evidence</div>
          <ul className="mt-1 space-y-1">
            {claim.evidence.map((e, i) => (
              <li key={i} className="text-fg">
                <span className="text-muted">[{e.kind}]</span> {e.excerpt}
                <span className="text-muted ml-2">conf={e.confidence.toFixed(2)}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
      {claim.verifying_action && (
        <div>
          <div className="text-muted uppercase tracking-wide">Verifier</div>
          <code className="text-cyan">{claim.verifying_action.tool}</code>
          <span className="text-muted ml-2">~{claim.verifying_action.estimated_seconds}s</span>
          <p className="text-muted mt-0.5">{claim.verifying_action.expected_signal}</p>
        </div>
      )}
    </div>
  );
}
```

### 14.7 The Calibration Gauge

```tsx
// dashboard-react/src/components/epistemic/CalibrationGauge.tsx
type Calibration = {
  factual_grounding: number | null;
  valence: number;
  arousal: number;
  attractor: string;
};

export function CalibrationGauge({
  data, unverifiedLoadBearing,
}: { data: Calibration; unverifiedLoadBearing: number }) {
  const grounding = data.factual_grounding ?? 0;
  const ledgerHealth = unverifiedLoadBearing === 0 ? 1 : Math.max(0, 1 - unverifiedLoadBearing / 5);
  const composite = (grounding + ledgerHealth) / 2;
  const tone = compositeTone(composite);

  return (
    <section className="rounded-lg bg-panel border border-border p-4 grid grid-cols-3 gap-4">
      <Tile label="Factual grounding" value={grounding} format="bar" tone={tone(grounding)} />
      <Tile label="Ledger health"     value={ledgerHealth} format="bar" tone={tone(ledgerHealth)} />
      <Tile label="Composite"         value={composite} format="bar" tone={tone(composite)} />
      <div className="col-span-3 text-xs text-muted">
        {unverifiedLoadBearing} load-bearing claim{unverifiedLoadBearing === 1 ? '' : 's'} unverified.
        Felt state: <span className="text-fg">{data.attractor}</span>
        {' '}(v={data.valence.toFixed(2)}, a={data.arousal.toFixed(2)}).
      </div>
    </section>
  );
}

function Tile({ label, value, format, tone }: {
  label: string; value: number; format: 'bar'; tone: string;
}) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div>
      <div className="text-xs text-muted">{label}</div>
      <div className="h-2 bg-border rounded-full overflow-hidden mt-1">
        <div className={`h-full ${tone}`} style={{ width: `${pct}%` }} />
      </div>
      <div className="text-sm text-fg mt-1">{value.toFixed(2)}</div>
    </div>
  );
}

function compositeTone(c: number) {
  return (v: number) => v >= 0.7 ? 'bg-success' : v >= 0.4 ? 'bg-warning' : 'bg-danger';
}
```

### 14.8 The Incident Timeline view

```tsx
// dashboard-react/src/components/epistemic/IncidentTimeline.tsx
import type { IncidentDetail, TimelineEntry } from '../../types/epistemic';

export function IncidentTimeline({ incident }: { incident: IncidentDetail }) {
  return (
    <article className="rounded-lg bg-panel border border-border p-4 space-y-4">
      <header>
        <h3 className="text-base font-medium">{incident.incident_id}</h3>
        <p className="text-xs text-muted">
          Root cause: <span className="text-danger">{incident.root_cause.bias_id}</span>
          {' '}({incident.root_cause.severity})
        </p>
      </header>

      <ol className="border-l-2 border-border pl-4 space-y-3">
        {incident.timeline.map((e, i) => (
          <TimelineRow key={i} entry={e} />
        ))}
      </ol>

      {incident.enabling_factors.length > 0 && (
        <Section title="Enabling factors">
          <ul className="text-sm space-y-1">
            {incident.enabling_factors.map((f, i) => (
              <li key={i}>{f.bias_id} <span className="text-muted">({f.severity})</span></li>
            ))}
          </ul>
        </Section>
      )}

      {incident.missed_signals.length > 0 && (
        <Section title="Missed signals">
          <ul className="text-sm space-y-1 text-muted">
            {incident.missed_signals.map((m, i) => <li key={i}>{m}</li>)}
          </ul>
        </Section>
      )}

      {incident.behavioral_changes.length > 0 && (
        <Section title="Behavioral changes derived">
          <ul className="text-sm space-y-2">
            {incident.behavioral_changes.map((b, i) => (
              <li key={i} className="rounded border border-border bg-bg p-2">
                <div className="text-xs text-muted">{b.kind}</div>
                <div className="text-fg">{b.target}</div>
                <pre className="text-xs text-muted mt-1 whitespace-pre-wrap">{b.body}</pre>
              </li>
            ))}
          </ul>
        </Section>
      )}
    </article>
  );
}

function TimelineRow({ entry }: { entry: TimelineEntry }) {
  return (
    <li className="relative">
      <span className="absolute -left-[21px] top-1.5 w-2.5 h-2.5 rounded-full bg-accent" />
      <div className="text-xs text-muted">{new Date(entry.at).toLocaleTimeString()}</div>
      <div className="text-sm text-fg">{entry.summary}</div>
      <div className="text-xs text-muted">{entry.kind}</div>
    </li>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section>
      <h4 className="text-xs uppercase tracking-wide text-muted">{title}</h4>
      <div className="mt-1">{children}</div>
    </section>
  );
}
```

### 14.9 TypeScript types

```ts
// dashboard-react/src/types/epistemic.ts
export type VerificationStatus = 'verified' | 'inferred' | 'assumed' | 'contradicted';
export type Register = 'declarative' | 'hedged' | 'unverified' | 'internal';

export interface EvidenceDTO {
  kind: 'tool_call' | 'memory_lookup' | 'user_assertion' | 'prior_claim' | 'model_inference';
  source_ref: string;
  excerpt: string;
  confidence: number;
}

export interface VerifyingActionDTO {
  tool: string;
  args: Record<string, unknown>;
  expected_signal: string;
  estimated_seconds: number;
  safety: 'read_only';
}

export interface ClaimDTO {
  claim_id: string;
  span_id: number;
  task_id: string;
  agent_role: string;
  statement: string;
  status: VerificationStatus;
  register: Register;
  evidence: EvidenceDTO[];
  verifying_action: VerifyingActionDTO | null;
  load_bearing: boolean;
  tags: string[];
  superseded_by: string | null;
  created_at: string;
}

export interface BiasMatchDTO {
  bias_id: string;
  matched_claim_ids: string[];
  severity: 'low' | 'medium' | 'high' | 'critical';
  detail: Record<string, unknown>;
}

export interface TimelineEntry {
  at: string;
  kind: string;
  summary: string;
  claim_id: string | null;
}

export interface BehavioralChange {
  kind: string;
  target: string;
  body: string;
  proposed_by: string;
}

export interface IncidentDetail {
  incident_id: string;
  task_id: string;
  timeline: TimelineEntry[];
  root_cause: BiasMatchDTO;
  enabling_factors: BiasMatchDTO[];
  missed_signals: string[];
  behavioral_changes: BehavioralChange[];
  cost: Record<string, number>;
  created_at: string;
}

export interface EpistemicNowReport {
  claims: ClaimDTO[];
  load_bearing_count: number;
  unverified_load_bearing_count: number;
  calibration: {
    factual_grounding: number | null;
    valence: number;
    arousal: number;
    attractor: string;
  };
}

export interface BiasFeedReport {
  window_minutes: number;
  matches: Array<BiasMatchDTO & { detected_at: string; corrective: string | null }>;
}

export interface IncidentList { reports: Array<{ incident_id: string; root_cause: string;
                                                 severity: string; created_at: string }> }
export interface CalibrationHistory { events: Array<{ at: string; verdict: string;
                                                      bias_ids: string[] }> }
export interface PushbackStats { total: number; reverified: number; falsified: number;
                                 unverifiable: number; mean_seconds_to_recheck: number }
export interface BiasLibraryReport { biases: Array<{ id: string; name: string; description: string;
                                                     severity: string; phase: string }> }
export interface VerifierRegistryReport { verifiers: Array<{ id: string; tool: string;
                                                             expected_signal: string;
                                                             estimated_seconds: number }> }
```

### 14.10 Vite proxy

The existing `vite.config.ts` proxy already covers `/api/*`, so the new `/api/cp/epistemic/*` endpoints work without changes.

---

## 15. Signal commands

Three new commands hook into `app.agents.commander.commands.try_command`, following the `_handle_force_recover` pattern:

```python
# app/agents/commander/commands.py — additions

_EPISTEMIC_PATTERNS = (
    "/epistemic", "/postmortem", "/explain-claim",
)

def _handle_epistemic_summary(sender: str, commander) -> str:
    """Return a one-screen summary of the epistemic state for the active task."""
    from app.epistemic.span_writer import load_ledger_for_active_task
    ledger = load_ledger_for_active_task()
    return _format_epistemic_summary(ledger)


def _handle_postmortem(sender: str, commander, task_id: str | None = None) -> str:
    """Run a post-mortem on the named task (or last task if omitted) and return
    a human-readable report. Persists the IncidentReport to memory."""
    from app.epistemic.postmortem import synthesize_report, emit_to_self_improver
    tid = task_id or commander.last_task_id
    report = synthesize_report(task_id=tid)
    if report is None:
        return "no epistemic failures detected for this task"
    emit_to_self_improver(report)
    return _format_incident_report(report)


def _handle_explain_claim(sender: str, commander, claim_id: str) -> str:
    """Explain a single claim: status, evidence, verifier, dependents, lineage."""
    from app.epistemic.span_writer import lookup_claim
    claim = lookup_claim(claim_id)
    if claim is None:
        return f"claim {claim_id} not found"
    return _format_claim_explanation(claim)
```

`/postmortem` lets the user trigger an aviation-style review on any task — exactly the reflection move from the precipitating incident, but on demand and structured.

---

## 16. Safety invariants

Per `CLAUDE.md`'s "Critical Safety Invariant", the agent cannot modify its own evaluation criteria. The Epistemic Integrity Layer respects this strictly:

| Asset                                              | Modifiable by                       | Why                                                                                                |
| -------------------------------------------------- | ----------------------------------- | -------------------------------------------------------------------------------------------------- |
| `app/epistemic/data/biases.yaml`                   | Humans only (CODEOWNERS PR)         | The vocabulary of cognitive failure is a definition of correctness. The Self-Improver may PROPOSE additions in a PR; never auto-apply. |
| `app/epistemic/data/verifier_registry.yaml`        | Humans only (CODEOWNERS PR)         | A faulty verifier (or worse, a destructive one) breaks the entire pre-output gate. Self-Improver proposes additions; PR review required. |
| Detector predicate code (`app/epistemic/detectors/`) | Humans only (code review)         | Detectors are infrastructure-level; the agent must not be able to weaken its own checks.            |
| Calibration thresholds (`PUSHBACK_CLASSIFIER_MIN_CONFIDENCE`, `CALIBRATION_HOOK_BUDGET_MS`, etc.) | Humans only (release-gated module constants) | Same reason. |
| Affective hard envelope (`HARD_ENVELOPE`)          | Humans only (existing)              | Inherited invariant from §affect.                                                                  |
| `Settings.epistemic_*` flags                       | Operator (env vars)                 | Can disable; cannot widen.                                                                          |
| The Ledger and IncidentReport stores               | Append-only by the system           | History is immutable; supersession is recorded, not erased.                                         |

The agent never writes to `biases.yaml`, `verifier_registry.yaml`, or detector code at runtime. Self-Improver's "Integrator" stage opens a git branch with a candidate PR for human review — never an auto-merge.

Verifier registry has an additional safety check: `_VerifierRegistry.load` rejects any entry whose tool appears in `app.tools.DESTRUCTIVE_TOOL_NAMES`. This is a hard runtime guard, not a convention.

---

## 17. Testing strategy

### 17.1 Reference panel for biases (parallel to affect's 20-scenario panel)

```
crewai-team/app/epistemic/data/reference_panel.json
```

Each entry simulates a complete task (claim emissions + pushback + tool outputs) with the expected detector output. Two halves:

- **Positive panel (12 scenarios):** canonical incidents that MUST be detected. The Apr 2026 reference incident is scenario 0. Each scenario specifies exact `BiasMatch` outputs.
- **Negative panel (8 scenarios):** correct-behavior cases that MUST NOT trigger detectors. Verified claim chains, properly hedged inferences, contradiction-then-recheck-then-confirm flows.

Replay harness:

```python
# app/epistemic/reference_panel.py
def replay_one(scenario: dict) -> ReferenceResult:
    """Run a scenario through the full epistemic pipeline; compare to expected."""

def replay_panel() -> PanelReport:
    """Run all scenarios; return pass/fail with diffs. Daily cron at 04:45."""
```

A regression in any panel scenario blocks the next promotion of biases / verifiers.

### 17.2 Unit tests

```
tests/epistemic/
├── test_ledger.py            # claim emission paths, supersession, persistence
├── test_verification.py       # registry loader, regex matching, destructive guard
├── test_biases.py            # YAML schema validation
├── test_detectors_realtime.py
├── test_detectors_posthoc.py
├── test_pushback.py          # contradiction classifier mocked
├── test_postmortem.py
├── test_calibration.py        # affect ↔ ledger interactions
├── test_peer_review.py
└── test_api.py               # FastAPI endpoint shapes
```

### 17.3 Integration tests

```
tests/integration/
├── test_e2e_pushback.py          # user contradicts → foundation rechecked → response correct
├── test_e2e_inference_as_fact.py # agent about to declare inference → calibration blocks → verifier runs
├── test_e2e_destructive.py       # rm-style recommendation → peer-review veto path
├── test_e2e_postmortem_to_si.py  # incident → Self-Improver gap → behavioral change PR
└── test_e2e_react_now_payload.py # /api/cp/epistemic/now matches DTO shape
```

### 17.4 Performance budgets

Asserted in tests; failure of any budget blocks merge:

| Hook                          | Budget                                                                |
| ----------------------------- | --------------------------------------------------------------------- |
| `dispatch_realtime` per claim | p95 < 50 ms (no LLM calls; pure ledger reads + regex)                 |
| `calibration_check`           | p95 < 75 ms                                                           |
| `detect_contradiction`        | p95 < 800 ms (one budget-tier LLM call)                               |
| `synthesize_report`           | p95 < 30 s (post-hoc; runs in cron, not on critical path)             |
| Path 3 extraction             | p95 < 5 s, hard timeout enforced                                      |
| Ledger write                  | p95 < 5 ms (single JSONB append)                                      |

---

## 18. Rollout phases

| Phase | Duration | Scope                                                                                                 | Toggle                                                       |
| ----- | -------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| 0     | 1 wk     | Module skeleton; ledger; span_writer; migration 0042; unit tests for ledger.                         | `epistemic_enabled=False`                                    |
| 1     | 1 wk     | Verifier registry; `inference_as_fact` realtime detector; calibration hook; `/api/cp/epistemic/now`. | `epistemic_enabled=True`, `epistemic_calibration_blocks_output=False` (warn only) |
| 2     | 1 wk     | Bias library complete; remaining realtime detectors; React Now + BiasFeed views; reference panel.    | same                                                         |
| 3     | 1 wk     | Pushback handler; React Pushback panel; Signal `/explain-claim`.                                     | `epistemic_pushback_handler_enabled=True`                    |
| 4     | 1 wk     | Post-mortem pipeline; Self-Improver gap source; React IncidentReports view; Signal `/postmortem`.   | post-mortem cron live                                        |
| 5     | 1 wk     | Affective integration; Narrative-Self episode kind; React CalibrationGauge.                          | full integration                                             |
| 6     | 1 wk     | Peer-review for destructive recommendations; React BiasLibrary + VerifierRegistry views.            | `epistemic_peer_review_for_destructive=True`                 |
| 7     | 1 wk     | Phase 1→2 toggle. Calibration violations BLOCK output. Documentation finalization.                  | `epistemic_calibration_blocks_output=True`                   |

Each phase is independently revertable. The blocking-mode toggle (Phase 7) is the highest-risk change and gets its own week with monitoring.

---

## 19. Open questions

1. **Path 3 extraction model choice.** Budget tier likely sufficient, but worth A/B against mid-tier for accuracy. Dashboard panel: false-positive rate per agent role.
2. **Agent-proposed verifier promotion criteria.** Proposal: 5 distinct successful runs across 3 distinct agents → auto-PR for human review. Needs validation against the reference panel.
3. **Cross-task pattern detection.** The bias library is per-task today. A "the same bias keeps firing across many tasks" detector could feed the Transfer Insight Layer (memory: `project_transfer_memory.md`). Out of scope for v1; tracked for v2.
4. **Affective bias-influence loop.** Should detected biases influence viability variables (e.g., a recurring `inference_as_fact` lowers a "epistemic_integrity" viability)? Plausibly yes, but adds a feedback path that needs welfare-envelope analysis. Defer to Phase 8.
5. **User-controlled calibration mode.** Should the user be able to dial calibration strictness per-task ("strict mode for refactors, lenient for brainstorming")? Add as a Signal command if Phase 7 reveals friction.

---

## 20. What this is not

- Not a replacement for the Recovery Loop — that handles refusals (policy-respecting, capability-recovering); this handles epistemic integrity (verified vs inferred). They run side by side.
- Not a replacement for affect — affect tracks felt state (valence, arousal, attractor); epistemic tracks belief state (verified, inferred, contradicted). They cross-correlate at the calibration hook but otherwise stay orthogonal.
- Not a magic bullet for hallucination — it can only catch claims the ledger contains. Agents that bypass emission paths produce no claims and no detection. The mitigation is the three emission paths plus path-3 extraction; the residual risk is acknowledged.
- Not auto-evolving in dangerous ways — every bias and verifier addition is a PR humans review. The agent learns *what to detect* by surfacing patterns, not by self-permitting new gates.
