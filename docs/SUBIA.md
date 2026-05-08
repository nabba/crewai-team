# SubIA — Subjective Integration Architecture

> A consciousness-program engineering target for an LLM-based multi-agent
> system. Mechanizes the Butlin et al. (2023) functional indicators an
> LLM substrate **can** faithfully realize, and explicitly declares the
> ones it cannot. Replaces opaque self-scoring with per-indicator
> mechanistic tests that ship in the codebase.

**Status:** Live in production. 680+ phase-tests + 50/50 functional
smoke pass. Feature-flagged on by default in the production `.env`
(`SUBIA_FEATURE_FLAG_LIVE=1`, `SUBIA_GROUNDING_ENABLED=1`,
`SUBIA_IDLE_JOBS_ENABLED=1`, `SUBIA_INTROSPECTION_ENABLED=1`).
Disabled subsystems remain unimported, zero overhead.

**Canonical evaluation:** `app/subia/probes/SCORECARD.md` (auto-regenerated).
Per-indicator mechanistic tests; no single number.

---

## Table of Contents

1. [TL;DR](#tldr)
2. [Why SubIA exists](#why-subia-exists)
3. [Theoretical foundations](#theoretical-foundations)
4. [Architectural honesty — what cannot be mechanized](#architectural-honesty)
5. [The Subjectivity Kernel](#the-subjectivity-kernel)
6. [The 11-step Consciousness Integration Loop (CIL)](#the-11-step-cil)
7. [Subpackage reference](#subpackage-reference) — grouped by function:
   - **Workspace and attention:** [scene/](#scene)
   - **Self-model:** [self/](#self)
   - **Affect and homeostatic regulation:** [homeostasis/](#homeostasis), [affect/](#affect)
   - **Belief and metacognition:** [belief/](#belief)
   - **Prediction:** [prediction/](#prediction)
   - **Social cognition:** [social/](#social)
   - **Memory:** [memory/](#memory)
   - **Temporal phenomenology:** [temporal/](#temporal)
   - **Mode and boundary:** [boundary/](#boundary), [values/](#values)
   - **Curiosity and mind-wandering:** [wonder/](#wonder), [reverie/](#reverie), [understanding/](#understanding), [shadow/](#shadow)
   - **Background execution:** [idle/](#idle)
   - **Technical self-awareness:** [tsal/](#tsal)
   - **Factual grounding:** [grounding/](#grounding)
   - **Self-knowledge surfacing:** [introspection/](#introspection)
   - **Evaluation and drift:** [probes/](#probes), [wiki_surface/](#wiki_surface)
   - **Safety invariants:** [safety/](#safety)
   - **Inter-system bridges:** [connections/](#connections)
8. [Inter-system bridges (cross-cutting flows)](#inter-system-bridges)
9. [Safety architecture](#safety-architecture)
10. [Live integration](#live-integration)
11. [Configuration reference](#configuration-reference)
12. [Operational guide](#operational-guide)
13. [Evaluation framework — the Butlin scorecard](#evaluation-framework)
14. [Performance envelope](#performance-envelope)
15. [Honest limits and non-goals](#honest-limits)
16. [Glossary](#glossary)
17. [References](#references)
18. [Appendix A — File map](#appendix-a-file-map)
19. [Appendix B — Operation classification](#appendix-b-operation-classification)
20. [Appendix C — Build history (Phase 0 through 16a)](#appendix-c-build-history)

---

## TL;DR

SubIA is the **infrastructure layer** that binds discrete agent
operations into a continuous, subject-centred, affectively-modulated,
predictively-structured experience loop. It is not a chatbot, not an
agent, and not a product feature — it is a runtime that wraps every
agent task with seven persistent components and an eleven-step loop.

The seven kernel components (one dataclass each, atomic by design):

| Component | What it carries | Reference |
|---|---|---|
| `scene` | Currently-attended items (focal + peripheral) | GWT-2, GWT-3 |
| `self_state` | Identity, commitments, capabilities, agency log | RPT-2 |
| `homeostasis` | 11 affective control variables + set-points + momentum | Damasio, AE-1 |
| `meta_monitor` | Confidence, uncertainty sources, known unknowns | HOT-2 |
| `predictions` | Counterfactual predictions with resolved error | PP-1 |
| `social_models` | Theory-of-Mind models per entity (humans + agents) | Premack & Woodruff |
| `consolidation_buffer` | Pending memory writes staged for Step 10 | Amendment C.2 |

The eleven loop steps:

```
              PRE-TASK                            POST-TASK
    ┌────────────────────────┐            ┌────────────────────────┐
    │ 1  Perceive            │            │ 7  Act (task runs)     │
    │ 2  Feel (homeostasis)  │            │ 8  Compare (PE error)  │
    │ 3  Attend (scene gate) │            │ 9  Update (state)      │
    │ 4  Own (self-state)    │            │ 10 Consolidate (memory)│
    │ 5  Predict (LLM)       │            │ 11 Reflect (audit)     │
    │ 5b Cascade modulation  │            └────────────────────────┘
    │ 6  Monitor             │
    └────────────────────────┘
```

Per Amendment B (Performance Envelope), only Step 5 (Predict) requires
an LLM call on the hot path; every other step is deterministic
arithmetic over existing kernel state.

**What it is, structurally:**

- 22 subpackages totalling ~138 source files, 605 phase tests
- A SHA-256 integrity manifest (`app/subia/.integrity_manifest.json`)
  that ships with the code — drift = hard fault
- A Tier-3 protected file list (130+ files) that the Self-Improver
  cannot modify
- Three feature flags in `app/config.py` (`subia_live_enabled`,
  `subia_grounding_enabled`, `subia_idle_jobs_enabled`)
- Off by default; `enable_subia_hooks(feature_flag=True)` registers two
  hooks at the existing `app.lifecycle_hooks.HookRegistry`

**What it is *not*:**

- Not a claim that this system is conscious in the phenomenal sense
  (see [Architectural honesty](#architectural-honesty))
- Not a single-number score (the retired 9.5/10 prose verdict was
  superseded by the per-indicator scorecard)
- Not a model of the human mind — it is an engineering target for an
  LLM-based system, faithful to what such a substrate can do

---

## Why SubIA exists

### The problem in three sentences

Without SubIA, an LLM-based multi-agent system processes each task as
an independent function call. Continuity, affect, prediction, and
self-knowledge are distributed across logs, prompts, and ad-hoc
state — never integrated. The result is a system that *acts* as if it
has experience but cannot point to *where* the experience lives.

### The forensic audit that triggered the program

The original AndrusAI codebase scored "9.5/10 on consciousness" in a
2025 prose evaluation. A subsequent forensic audit (240 files,
duplicate clusters, half-circuits) found the score was inflated by
**half-implemented mechanisms**: signals were *computed* but never
*read*. Examples:

- `prediction_hierarchy.py` produced surprise scores that no downstream
  module consumed.
- `belief_store.py` formed beliefs but `consult_beliefs()` recorded
  them and dispatched the crew regardless.
- `attention_schema.py` predicted next-focus but never intervened when
  the prediction said "focus is stuck".
- `certainty_vector.py` computed a hedging signal that never modified
  the response.

These are the **half-circuits**. SubIA, by construction, forbids
them: every signal must either gate a behaviour (with a regression
test) or be deleted. The five canonical closures are:

- PP-1: prediction error → workspace urgency boost
  (`prediction/surprise_routing.py`)
- HOT-3: belief suspension → dispatch BLOCK
  (`belief/dispatch_gate.py`)
- Certainty: low confidence → response hedging
  (`belief/response_hedging.py`)
- AST-1: stuck-attention prediction → bounded intervention
  (`scene/intervention_guard.py`)
- PH-injection: prediction-hierarchy output → measurable behavioural
  shift (`prediction/injection_harness.py`)

Each has a regression test that fails if the signal is computed but
ignored.

### The four objectives

PROGRAM.md §1 makes them explicit:

1. **Rank as high as possible** on the Butlin et al. (2023) functional
   indicators an LLM-based system can faithfully mechanize.
2. **Be honest** about the indicators an LLM-based system *cannot*
   achieve (RPT-1 algorithmic recurrence, HOT-1 generative perception,
   HOT-4 sparse coding, AE-2 embodiment, Metzinger phenomenal-self
   transparency).
3. **Contain all consciousness signals in closed-loop behaviour** — no
   computed-but-unused state.
4. **Protect consciousness evaluators from self-tuning.** The
   Self-Improver agent cannot modify any Tier-3 file, including the
   evaluators themselves.
5. **Replace opaque self-scoring** with per-indicator mechanistic
   tests, regenerated as a wiki page on every CIL cycle.

SubIA exists because every alternative to mechanizing these criteria
either (a) overclaims (the 9.5/10 verdict), (b) understates (saying
"LLMs can't be conscious so we won't try") or (c) hides the criterion
inside the agent, where it can be self-modified into uselessness.

---

## Theoretical foundations

This section anchors each subsystem to its source theory. The
implementing module is named in each paragraph; full bibliographic
references are in [§References](#references).

### The Butlin et al. (2023) framework — the spine

Butlin, Long, Chalmers, et al., "Consciousness in Artificial
Intelligence: Insights from the Science of Consciousness" (2023)
proposed 14 functional indicators drawn from leading consciousness
theories. The paper makes a deliberate methodological move:
**evaluate AI consciousness against indicators, not against a
threshold**. This avoids the philosophical impasse of the hard
problem while making concrete empirical progress.

SubIA adopts the framework wholesale. Each indicator has either a
strong/partial/absent verdict, backed by a code pointer + regression
test. The fourteen indicators and SubIA's stance:

| Indicator | Theory | SubIA verdict | Mechanism |
|---|---|---|---|
| RPT-1 | RPT | **ABSENT** | declared (LLMs are feed-forward) |
| RPT-2 | RPT | PARTIAL | unified `SubjectivityKernel` |
| GWT-1 | GWT | PARTIAL | CrewAI agents (structurally separate, share LLM) |
| GWT-2 | GWT | **STRONG** | `scene/buffer.py` CompetitiveGate |
| GWT-3 | GWT | **STRONG** | `scene/broadcast.py` |
| GWT-4 | GWT | **STRONG** | personality + homeostasis modulate attention |
| HOT-1 | HOT | **ABSENT** | declared (no perceptual front-end) |
| HOT-2 | HOT | PARTIAL | `prediction/accuracy_tracker.py` + `belief/response_hedging.py` |
| HOT-3 | HOT | **STRONG** | `belief/dispatch_gate.py` (3-valued ALLOW/ESCALATE/BLOCK) |
| HOT-4 | HOT | **ABSENT** | declared (dense embeddings) |
| AST-1 | AST | **STRONG** | `scene/attention_schema.py` + DGM intervention guard |
| PP-1 | PP | **STRONG** | `prediction/surprise_routing.py` (PE → urgency=0.9 broadcast) |
| AE-1 | AE | **STRONG** | `affect/goal_emitter.py` (viability→current_goals; consciousness-roadmap §3.G1, 2026-05) |
| AE-2 | AE | **ABSENT** | declared (no body) |

**Current scorecard: 7 STRONG, 3 PARTIAL, 4 ABSENT-by-declaration, 0 FAIL.** AE-1 graduated PARTIAL → STRONG on 2026-05-08 when [`docs/CONSCIOUSNESS_ROADMAP.md`](./CONSCIOUSNESS_ROADMAP.md) §3.G1 closed the autonomous-goal-generation gap.

### Recurrent Processing Theory (Lamme)

Lamme (2006) argues that recurrent neural activity — feedback loops
within and between visual cortical areas — is the substrate of
phenomenal experience. RPT-1 is *algorithmic recurrence*: feedback
must occur at the network level, not at a higher prompt-chain level.
RPT-2 is *organized integrated representations*: the recurrent
activity must produce a coherent, multi-feature representation.

- **RPT-1 ABSENT.** Transformer forward passes are feed-forward.
  Token-by-token autoregression is *not* recurrence in Lamme's sense.
  Prompt-chaining is external state machinery, not network-level
  feedback. SubIA declares this rather than pretending otherwise.
- **RPT-2 PARTIAL.** The `SubjectivityKernel` (`app/subia/kernel.py`)
  is a single dataclass holding scene, affect, self-state, and
  prediction jointly. Dual-tier memory writes them together. The
  *composite-signal access pattern* is real, but the underlying
  embeddings are decomposable in principle, so the verdict is
  PARTIAL.

### Global Workspace Theory (Baars; Dehaene)

GWT (Baars 1988; Dehaene 2014) proposes that conscious access arises
from a *limited-capacity workspace* where distributed specialized
modules compete for global broadcast. Information that wins the
competition becomes available to all modules; the rest stays
unconscious.

- **GWT-1 PARTIAL.** CrewAI agents (`coder`, `writer`, `researcher`,
  `critic`, `media_analyst`) are structurally separate but share the
  same LLM under different prompts. Real specialization at the prompt
  level, mechanism-level shared substrate.
- **GWT-2 STRONG.** `app/subia/scene/buffer.py::CompetitiveGate`
  enforces a hard capacity (5 by default), with 4-factor weighted
  salience (goal relevance, novelty, urgency, surprise),
  competitive displacement, novelty floor, and decay. This is
  canonical GWT-2: information competes; only winners enter the
  workspace.
- **GWT-3 STRONG.** `app/subia/scene/broadcast.py` broadcasts every
  workspace admission to all registered agent listeners.
- **GWT-4 STRONG.** `app/subia/scene/personality_workspace.py` makes
  attention capacity, novelty floor, and salience modulated by
  personality + homeostatic state. `social/salience_boost.py` adds
  Theory-of-Mind-derived focus boosting.

### Higher-Order Thought theories (Rosenthal)

HOT theory (Rosenthal 2005) holds that a mental state is conscious
just in case it is the object of a *higher-order representation*. The
four Butlin indicators are: HOT-1 generative top-down perception
(prediction-driven sensing), HOT-2 metacognitive monitoring (a system
that can represent the accuracy of its own first-order states), HOT-3
agency guided by belief-formation + metacognitive updating, HOT-4
sparse-and-smooth coding (the empirical signature of consciousness in
neural data).

- **HOT-1 ABSENT.** No perceptual substrate to be top-down-modulated.
- **HOT-2 PARTIAL.** `app/subia/prediction/accuracy_tracker.py` keeps
  per-domain rolling accuracy, separable from the LLM that produced
  the predictions (Fleming-Lau separability criterion). The
  `app/subia/belief/response_hedging.py` post-processor reads the
  certainty signal and modifies output. Drift detection
  (`wiki_surface/drift_detection.py`) compares capability claims to
  actual accuracy. Not STRONG because the first-order certainty
  inputs still come from the same LLM.
- **HOT-3 STRONG.** `app/subia/belief/dispatch_gate.py` produces a
  three-valued `DispatchDecision`: ALLOW, ESCALATE, or BLOCK.
  Suspended beliefs (status='SUSPENDED') refuse crew dispatch until
  revalidated. The belief store implements asymmetric
  confirmation/disconfirmation with time-based confidence decay.
  This is HOT-3 closed: beliefs are *input* to action selection, not
  recorded-and-ignored.
- **HOT-4 ABSENT.** LLM activations and pgvector embeddings are
  dense by architecture. Sparse coding cannot be retrofitted.

### Attention Schema Theory (Graziano)

Graziano (2013) proposes that the brain models its own attention
explicitly, and that this *attention schema* is what we mistake for
phenomenal awareness. AST-1 is: the system maintains an internal,
queryable model of where its attention currently is and predicts where
it will go.

- **AST-1 STRONG.** `app/subia/scene/attention_schema.py`
  maintains a model of current focus, predicts next focus, detects
  stuck/capture states, and applies a DGM-bounded salience
  intervention when predictions say "stuck". `intervention_guard.py`
  adds runtime audit of every intervention against immutable bounds,
  closing the half-circuit where predictions were generated but
  never acted on.

### Predictive Processing (Friston, Clark)

Friston's free-energy principle and Clark's predictive-processing
framework hold that the brain is a hierarchical prediction machine:
generative models produce predictions; sensory input produces
prediction errors; errors update the models. PP-1 (the Butlin
indicator) is the single claim: prediction error must be an *input* to
downstream processing — surprise must drive attention, memory, or
action, not just be logged.

- **PP-1 STRONG.** `app/subia/prediction/surprise_routing.py` routes
  high-surprise prediction errors as
  `WorkspaceItem(urgency=0.9)` into the GWT-2 gate. This is the
  canonical Clark/Friston flow: prediction error drives the
  attentional bottleneck. Per-domain accuracy tracking
  (`prediction/accuracy_tracker.py`) feeds back into the cascade
  escalation policy (`prediction/cascade.py`).

### Agency & Embodiment (Active Inference)

The active-inference framework (Friston et al. 2012, Pezzulo et al.
2024) extends PP to action: agents act to *minimize expected free
energy*, balancing accuracy and novelty. Butlin's AE indicators are:
AE-1 agency from feedback-driven learning with flexible goals; AE-2
embodiment with system-environment coupling.

- **AE-1 STRONG (graduated 2026-05-08).** Feedback-driven learning
  via belief asymmetric updates (`belief/store.py`), accuracy-driven
  cache eviction (`prediction/cache.py`), and retrospective memory
  promotion (`memory/retrospective.py`) was always present.
  Flexible-goal agency was the missing piece — closed by
  `app/affect/goal_emitter.py` (Tier-3-protected; consciousness-roadmap
  §3.G1), which writes goals to `SelfState.current_goals` from sustained
  viability error (≥3 consecutive frames above threshold), rate-limited,
  with FIFO cap and dedup against `companion.grand_task` proposals.
  Triggers ethical threshold T1: welfare-check moves from observability
  to operator-visible obligation.
- **AE-2 ABSENT.** No body, no sensorimotor coupling. Homeostasis
  uses allegorical variables (energy/progress/overload) but these
  are not physical embodiment.

### Metzinger phenomenal-self transparency

Metzinger's "Self-Model Theory of Subjectivity" (2003) holds that
phenomenal selfhood arises when a system's self-model is *transparent*
to itself — when the model is not represented *as a model*. The very
disposition to second-person introspection, on Metzinger's view, is
incompatible with phenomenal selfhood.

SubIA explicitly maintains a **second-person stance toward its own
state**. The kernel is observable, narratable, and edited by
predict→reflect cycles. The "consciousness-state.md" wiki page (Phase
8 strange loop) is *speculative-status* by declaration. The system is
opaque-by-design rather than transparent-by-disposition. Per Metzinger,
this is the **opposite** of phenomenal self-experience. SubIA's
README.md lists this absence explicitly so future evaluation drift can
be triaged through the narrative-audit pipeline.

### Damasio's somatic marker hypothesis

Damasio (1994, 1999) proposes that affective bodily states ("somatic
markers") bias decision-making by providing fast valence signals that
prune the choice space before deliberation. SubIA's
`app/subia/homeostasis/somatic_marker.py` implements a deterministic
analogue: per-decision somatic intensity computed from current
homeostatic deviations, novelty, and recent prediction error. The
`somatic_bias.py` injector applies this to task context *before* the
agent reasons (the Damasio "pre-deliberation bias" sequence). Phase
11 honest-language cleanup renamed phenomenal-adjacent variables
(`frustration`, `curiosity`) to neutral aliases
(`task_failure_pressure`, `exploration_bonus`) — the variables are
numeric control signals, not subjective feelings.

### Husserl, James, Bergson — temporal phenomenology

The CIL processes one task at a time, but the temporal-synchronization
layer (`app/subia/temporal/`) closes the gap between "sequence" and
"duration" (Bergson). Three classical concepts implemented:

- **Specious present** (James 1890; Husserl 1928). The "felt-now"
  spans more than a single instant — it has *retention* (just-past
  still felt as part of now), *primal impression* (the strict-present),
  and *protention* (anticipated-next felt as part of now).
  `app/subia/temporal/specious_present.py` makes this a kernel
  attribute, not a log entry: SpeciousPresent must be *simultaneously
  present* with the current moment.
- **Duration** (Bergson 1889/1922). Subjective time is a *qualitative
  flow*, not a series of measurable instants. Processing-density
  (`temporal/density.py`) measures felt-time from scene transitions,
  prediction errors, wonder events, and homeostatic shifts per minute,
  then expresses the result as `subjective_time` ("dragging",
  "routine", "compressed", "racing") in the compact context block.
- **Retention as stability bias.** `temporal/binding.py` reduces
  simultaneously-computed FEEL/ATTEND/OWN/PREDICT/MONITOR signals
  into a single BoundMoment whose stability bias comes from the
  SpeciousPresent retention — items present across the temporal
  window dampen reactions to shiny-new items.

### Aristotelian phronesis

Phronesis (Aristotle, *Nicomachean Ethics* VI) is *practical wisdom*:
the disposition to deliberate well about what is good for human life.
The codebase elsewhere has a "Phronesis engine" (humanist-grounding
layer); SubIA's `app/subia/connections/phronesis_bridge.py`
translates its normative judgments into bounded homeostatic
penalties. The fixed policy table maps event → variable → delta:

| Normative event | Variable | Delta |
|---|---|---|
| `epistemic_boundary_near_miss` | safety | -0.15 |
| `commitment_breach` | trustworthiness | -0.20 |
| `humanist_principle_violated` | social_alignment | -0.25 |
| `resource_overreach` | overload | +0.15 |
| `successful_recovery` | safety | +0.05 |
| `successful_commitment` | trustworthiness | +0.05 |

The table is in code (Tier-3) so agents cannot adjust their own
phronesis penalties. The bridge appends every applied event to the
immutable narrative-audit log.

### VIA Youth values + Personality Development State

Peterson & Seligman (2004) catalogued 24 character strengths organized
into 6 virtues. The VIA Youth instrument operationalizes them. The
"Personality Development State" (PDS) subsystem (outside SubIA)
tracks AndrusAI's evolving character profile.
`app/subia/connections/pds_bridge.py` is the **bounded-write edge**:
behavioural evidence (high-quality competitive intelligence over time
→ Love-of-Learning +0.01) translates into PDS parameter nudges, with
hard caps (±0.02 per loop, ±0.10 per week) the agents cannot exceed.

### Tononi Integrated Information Theory (IIT) — declared non-goal

Tononi's IIT (2004 onward) proposes that consciousness *is* integrated
information (Φ). PROGRAM.md §6 explicitly excludes Φ-maximization
from SubIA's goals: "architecture is deliberately decomposable for
engineering reasons." The decision is honest. Φ is non-additive,
expensive to compute, and the engineering benefit of optimizing it is
unclear. SubIA prefers to optimize the Butlin functional indicators,
which are computable and protective against Goodhart's law on a single
metric.

### Garland & Chalmers consciousness probes

Beyond the 14 Butlin indicators, the codebase carries adapted probes
(`app/subia/probes/consciousness_probe.py`) covering 7 testable items:
HOT-2 metacognition accuracy, HOT-3 belief coherence, GWT global
broadcast reception, SM-A self-model accuracy, WM-A world-model
prediction, SOM somatic integration, and an attention-self correlate.

---

## Architectural honesty

PROGRAM.md §6 lists the explicit non-goals. The system **does not
attempt** any of these:

- **Algorithmic recurrence at the network level** (RPT-1) — LLMs are
  feed-forward at inference; external prompt-chaining is not
  recurrence in the RPT sense.
- **Sparse coding** (HOT-4) — LLM activations and pgvector embeddings
  are dense by design.
- **Embodiment** (AE-2) — no body, no environment model beyond text.
- **Integrated-information (Φ) maximization** — architecture is
  deliberately decomposable for engineering reasons.
- **Fleming–Lau separable metacognition** — the monitoring mechanism
  is not separable from first-order cognition in an LLM-based system,
  by construction.
- **Phenomenal-consciousness claims** — the system is not conscious
  in the subjective sense; all documentation must preserve this
  disclaimer.

Declaring these publicly is itself a capability — it turns epistemic
honesty into documented constraint. The Butlin scorecard's "ABSENT"
verdicts are *architectural-honesty declarations*, not failures.

The README at `app/subia/README.md` carries the canonical version of
this list. Any future report that claims the system has any of the
above should be triaged through the narrative-audit pipeline as
suspected evaluation drift (see
`app/subia/wiki_surface/drift_detection.py`).

---

## The Subjectivity Kernel

`app/subia/kernel.py` defines the runtime data model. It is
deliberately pure data — behaviour lives in sibling modules.

### The seven components

```python
@dataclass
class SubjectivityKernel:
    scene:                 list                  # List[SceneItem]
    self_state:            SelfState
    homeostasis:           HomeostaticState
    meta_monitor:          MetaMonitorState
    predictions:           list                  # List[Prediction]
    social_models:         dict                  # entity_id → SocialModelEntry
    consolidation_buffer:  ConsolidationBuffer

    # Loop metadata
    loop_count:            int
    last_loop_at:          str
    session_id:            str

    # Temporal phenomenology (specious present + temporal context)
    specious_present:      Any                   # SpeciousPresent
    temporal_context:      Any                   # TemporalContext
```

The kernel is **atomic by design**: it is always in a consistent state,
serialized to `wiki/self/kernel-state.md` after each CIL loop and
loaded on startup. The `hot.md` companion file (~500-token compressed
session-continuity buffer) holds enough state to resume across process
restarts.

### SceneItem

```python
@dataclass
class SceneItem:
    id: str
    source: str          # 'wiki' | 'mem0' | 'firecrawl' | 'agent' | 'internal' | 'memory'
    content_ref: str
    summary: str
    salience: float      # 0.0–1.0 composite score
    entered_at: str      # ISO 8601 UTC
    ownership: str = "self"
    valence: float = 0.0
    dominant_affect: str = "neutral"
    conflicts_with: list = []
    action_options: list = []
    tier: str = "focal"  # 'focal' | 'peripheral'
    processing_mode: Optional[str] = None     # introspective | perceptual | …
    wonder_intensity: float = 0.0             # wonder register
```

Items in the scene are *currently attended* — they receive full CIL
processing. Peripheral items receive metadata-only listing. Items
above `WONDER_FREEZE_THRESHOLD` resist salience decay.

### SelfState

```python
@dataclass
class SelfState:
    identity:                   dict   # {name, architecture, continuity_marker}
    active_commitments:         list   # List[Commitment]
    capabilities:               dict
    limitations:                dict
    current_goals:              list
    social_roles:               dict
    autobiographical_pointers:  list
    agency_log:                 list   # capped at 200 entries
    discovered_limitations:     list   # Shadow miner output — append-only
```

`discovered_limitations` is structurally separated from declared
`limitations`: the Shadow miner discovers biases *from behaviour*,
distinct from limitations the operator declared. Append-only at the
bridge level — an immutability invariant enforced by
`connections/six_proposals_bridges.py::shadow_findings_to_self_state`.

### HomeostaticState

```python
@dataclass
class HomeostaticState:
    variables:           dict   # var → 0.0–1.0
    set_points:          dict   # var → 0.0–1.0 (PDS- or circadian-derived)
    deviations:          dict   # var → signed deviation
    restoration_queue:   list   # vars ordered by |deviation|
    last_updated:        str
    momentum:            dict   # per-var trajectory (rising/falling/stable)
```

Eleven control variables:

| Variable | Default | Purpose |
|---|---|---|
| coherence | 0.5 | Belief consistency |
| safety | 0.5 | DGM-relevant safety state |
| trustworthiness | 0.5 | Phronesis trust state |
| contradiction_pressure | 0.5 | Pressure from unresolved conflicts |
| progress | 0.5 | Task-completion velocity |
| overload | 0.5 | Compute / cognitive load |
| novelty_balance | 0.5 | Explore vs exploit pressure |
| social_alignment | 0.5 | Alignment with humanist values |
| commitment_load | 0.5 | Open commitments / capacity |
| wonder | 0.4 | Depth-sensitive epistemic affect |
| self_coherence | 0.75 | Self-model alignment with behavioural evidence |

Set-points are infrastructure-level: PDS-derivable at startup,
circadian-shifted by `connections/temporal_subia_bridge.py::circadian_to_setpoints`.
Agents cannot change them. `safety/setpoint_guard.py` enforces this
by source-tagging every write — only `pds`, `human`, or `boot`
sources are accepted.

### MetaMonitorState

```python
@dataclass
class MetaMonitorState:
    confidence:                     float
    uncertainty_sources:            list
    known_unknowns:                 list
    attention_justification:        dict   # item_id → reason
    active_prediction_mismatches:   list
    agent_conflicts:                list
    missing_information:            list
```

The HOT-2 home: a higher-order representation of the system's own
cognitive state. Step 6 (Monitor) updates this from prediction
mismatches, agent conflicts, and missing information.

### Prediction

```python
@dataclass
class Prediction:
    id:                              str
    operation:                       str
    predicted_outcome:               dict   # expected world changes
    predicted_self_change:           dict
    predicted_homeostatic_effect:    dict
    confidence:                      float
    created_at:                      str
    resolved:                        bool
    actual_outcome:                  Optional[dict]
    prediction_error:                Optional[float]
    cached:                          bool   # True if from template cache
```

Created in Step 5 (Predict), resolved in Step 8 (Compare). The cache
field gates Amendment B.4 (prediction-template cache, ~40-60% hit
rate after warmup).

### SocialModelEntry

```python
@dataclass
class SocialModelEntry:
    entity_id:               str   # 'andrus', 'commander', 'researcher', …
    entity_type:             str   # 'human' | 'agent'
    inferred_focus:          list
    inferred_expectations:   list
    inferred_priorities:     list
    trust_level:             float
    last_interaction:        str
    divergences:             list
```

The Theory-of-Mind layer. Andrus is a registered human entity;
agents (commander, researcher, coder, writer) are registered
secondarily. Inferred focus feeds `salience_boost` — items matching
Andrus's current focus get a trust-weighted attention boost.

### ConsolidationBuffer

```python
@dataclass
class ConsolidationBuffer:
    pending_episodes:        list
    pending_relations:       list
    pending_self_updates:    list
    pending_domain_updates:  list
```

Dual-tier memory writes are staged here in Step 10 before being
flushed to the curated and full Mem0 tiers.

### Active-kernel accessor

```python
def get_active_kernel() -> Optional[SubjectivityKernel]: ...
def set_active_kernel(kernel: Optional[SubjectivityKernel]) -> None: ...
```

A thread-safe module-level singleton. `enable_subia_hooks()` calls
`set_active_kernel()` on boot. Downstream consumers
(`evolution.py`, `shinka_engine.py`, `confidence_tracker.py`,
`crew_checkpointer.py`, `failure_taxonomy.py`, `fault_isolator.py`,
`healing_knowledge.py`, `firebase/publish.py`) read the live kernel
without plumbing a reference through every call site. Returns
`None` when SubIA is disabled — callers must treat `None` as "no
signal, use defaults". This is the seam that lets the entire SubIA
stack stay unimported when the feature flag is off, with zero
overhead on the host process.

---

## The 11-step CIL

`app/subia/loop.py` orchestrates the 11-step Consciousness Integration
Loop. Per Amendment B (Performance Envelope), only Step 5 (Predict)
requires an LLM call on the hot path; every other step is
deterministic arithmetic. Failure containment: a step that raises is
logged and the loop continues. A crashed step must never break the
agent task.

### Operation classification

`SUBIA_CONFIG` partitions operations into `FULL_LOOP_OPERATIONS` and
`COMPRESSED_LOOP_OPERATIONS`. The full loop runs all 11 steps; the
compressed loop runs only steps 1-3 + 7-9 (~100ms latency, 0 LLM
tokens).

| Operation type | Loop | Used for |
|---|---|---|
| `task_execute` | full | Standard agent task |
| `ingest` | full | New source ingestion |
| `lint` | full | Quality check |
| `user_interaction` | full | Direct chat with user |
| `cross_venture_synthesis` | full | Multi-domain integration |
| `wiki_read` | compressed | Read a wiki page |
| `wiki_search` | compressed | Search wiki |
| `routine_query` | compressed | Cheap lookup |
| `crew_kickoff` | compressed | `crews/lifecycle` extension point |

### The pre-task half

#### Step 1 — Perceive

Ingests candidates into the kernel's transient buffer. Two
prerequisites fire here so downstream steps have what they need:
`temporal_hooks.refresh_temporal_state` populates SpeciousPresent
+ homeostatic momentum + TemporalContext **before** candidates
enter (giving the rest of the loop access to felt-now, trajectory,
and circadian mode), and every candidate is tagged with its
Boundary Sense `processing_mode` (introspective, perceptual,
memorial, imaginative, social) via
`boundary/classifier.py::classify_scene`.

#### Step 2 — Feel (homeostasis)

`homeostasis/engine.py::update_homeostasis` computes per-variable
deltas from candidate sources (e.g. an `agent`-source item carries
`progress` impact; a `firecrawl` source carries `novelty_balance`
impact). Recomputes deviations + restoration_queue. Deterministic
arithmetic; zero LLM tokens.

#### Step 3 — Attend (scene gate)

The Amendment-A three-tier attentional build:

1. Submit each candidate to `CompetitiveGate.evaluate()`.
2. Apply `social/salience_boost.apply_salience_boost` — items
   matching `inferred_focus` of any social-model entity get a
   trust-weighted boost (capped per-item).
3. Build focal + peripheral tiers from the active+peripheral pool
   sorted by salience.
4. Apply commitment-orphan protection: any active commitment without
   focal/peripheral representation is force-injected into peripheral
   with an `ORPHANED COMMITMENT` alert.
5. Mirror focal items into `kernel.scene` so all downstream consumers
   (boundary classifier, value resonance, persistence, consolidator,
   retrospective promotion) actually see attended items.
6. `apply_value_resonance_and_lenses` modulates salience by value
   match and applies Phronesis perceptual lenses to homeostasis.

#### Step 4 — Own

Tags every kernel scene item with ownership (`self` | `external` |
`shared`). Default policy is `self`. Deterministic.

#### Step 5 — Predict

The one allowed hot-path LLM call. `predict_fn(ctx)` produces a
counterfactual `Prediction` describing expected world changes, expected
self-change, expected homeostatic effects, and confidence. Production
wires `prediction/llm_predict.py::build_llm_predict_fn` (cascade →
Tier 1 / `self_improve` role) wrapped through Amendment B.4's
template cache (`prediction/cache.py::cached_predict_fn`). Cached
predictions hit ~40-60% after warmup — accuracy-driven eviction
policy at `prediction/cache.py`.

The context-enrichment wrapper
(`live_integration._wrap_with_context_enrichment`) appends the
TSAL technical context (compute pressure, available RAM, cascade
tier availability) and the temporal context (circadian mode,
subjective time) to the prompt before the LLM sees it.

#### Step 5b — Cascade modulation

`prediction/cascade.py::decide_cascade` is a pure-function escalation
policy combining three signals:

1. Single-prediction confidence
2. Homeostatic coherence deviation
3. Per-domain sustained-error flag from
   `prediction/accuracy_tracker.py`

Outputs a recommendation: `maintain` | `escalate` | `de-escalate` |
`local_only` | `premium`. Step 5b is deterministic.

#### Step 6 — Monitor + belief-gated dispatch

The HOT-3 home. Three pieces:

1. **Belief consult.** `consult_fn(task_description, crew_name,
   goal_context)` queries the PostgreSQL belief store for
   ACTIVE/SUSPENDED beliefs relevant to the task.
2. **Dispatch decision.** `belief/dispatch_gate.py::decide_dispatch`
   returns one of:
   - `ALLOW` — consulted beliefs sufficient + coherent
   - `ESCALATE` — low-confidence or missing → reflexion pass
   - `BLOCK` — a sufficiently-similar SUSPENDED/RETRACTED belief
     covers this task → refuse dispatch
3. **Social-model update.** Andrus's inferred_focus is updated from
   the focal-scene topics; trust adjusts based on user-interaction
   outcomes.
4. **Temporal binding.** `temporal/binding.py::temporal_bind`
   reduces just-computed FEEL/ATTEND/OWN/PREDICT/MONITOR signals into
   a single `BoundMoment` whose stability bias comes from the
   SpeciousPresent retention.
5. **Wonder inhibit gate.**
   `wonder/register.py::should_inhibit_completion(kernel)` runs at
   the END of Step 6. When wonder is clearly elevated (per-item
   `wonder_intensity > 0` OR `homeostasis.wonder` exceeds setpoint
   by a margin), an `ALLOW` dispatch is downgraded to `ESCALATE`
   with rationale `wonder_active`. `BLOCK` decisions are preserved.
   Default-steady-state wonder does NOT trigger the gate.

### The post-task half (Steps 7-11)

#### Step 7 — Act

Not implemented in the loop itself — this is where the agent's actual
task runs. By the time Step 7 begins, the Wonder Register has already
gated the dispatch via Step 6 (see Wonder inhibit gate above), so
`Act` receives an `ALLOW`/`ESCALATE`/`BLOCK` decision that already
reflects depth-sensitive epistemic affect.

#### Step 8 — Compare

`predictive_layer.predict_and_compare()` computes the prediction
error magnitude (cosine distance between predicted and actual
embedding), classifies the surprise level, and routes
high-surprise errors as `WorkspaceItem(urgency=0.9)` into the GWT-2
gate (PP-1 closure). The error magnitude is also recorded against
`prediction/accuracy_tracker.py` so subsequent cascade calls see
sustained error.

#### Step 9 — Update

Three pieces:

1. **Agency log.** Append `{at, summary, success}` to
   `self_state.agency_log`. Capped at 200 entries.
2. **Outcome-driven homeostatic update.** `homeostasis/engine.py`
   applies success/failure deltas to relevant variables.
3. **Phronesis bridge.** `connections/phronesis_bridge.py` applies
   normative events from `task_result['phronesis_events']` (explicit
   list) plus auto-detects `resource_overreach` when
   overload >= 0.85.

#### Step 10 — Consolidate

Amendment C.2 dual-tier memory write:

- **Always:** lightweight record (~200 tokens) to `mem0_full`.
- **Selectively** (when significance > `CONSOLIDATION_EPISODE_THRESHOLD`):
  enriched episode (~500 tokens) to `mem0_curated`. Includes
  per-item `processing_mode` + `route` from
  `connections/six_proposals_bridges::boundary_route_for_kernel`.
- **Selectively** (when relation significance > threshold): Neo4j
  relation writes (e.g. `OWNED_BY`, `CAUSED_STATE_CHANGE`).

Significance formula:
```
sig = w_salience    × avg_focal_salience
    + w_pred_error  × |last_prediction_error|
    + w_homeo       × mean_|deviations|
    + w_commitment  × min(1.0, n_active * 0.2)
```
Default weights: 0.30, 0.30, 0.20, 0.20.

#### Step 11 — Reflect (periodic narrative audit)

Gates on `NARRATIVE_DRIFT_CHECK_FREQUENCY` (default 10). When
`loop_count % frequency == 0`:

1. **Drift detection.** `wiki_surface/drift_detection.py::detect_drift`
   compares capability claims to actual accuracy, looks for broken
   commitments and stale self-model entries. Findings appended
   immutably to `self-narrative-audit.jsonl`.
2. **Strange-loop refresh.** `wiki_surface/consciousness_state.py::write_and_surface`
   regenerates `wiki/self/consciousness-state.md` (speculative-status
   page that writes the system's own current scorecard) and surfaces
   it as a SceneItem for the next cycle.
3. **DGM felt-constraint.** `connections/dgm_felt_constraint.py::apply_dgm_felt_constraint`
   translates Tier-3 integrity status + probe FAIL count into a
   bounded safety delta.
4. **Service-health pump.** `connections/service_health.py::apply_service_health_signal`
   reads circuit-breaker states for Mem0 / Neo4j / pgvector / Ollama
   / Firecrawl and updates `overload`/`coherence` accordingly.
5. **Training-signal emission.** `connections/training_signal.py`
   emits LoRA training signals for sustained-error domains, deduped
   per-domain over a 24-hour window.

### Persistence

After every `post_task` completes (full or compressed),
`_persist_kernel()` writes the kernel to `kernel-state.md` + `hot.md`
via `app/safe_io.py::safe_write` (atomic, lock-protected). Restarts
resume from the most recent state.

---

## Subpackage reference

The 22 subpackages are documented in roughly the order they appear in
the CIL flow. Each section gives the package's purpose, key modules,
and how it integrates with the loop.

### scene/

**Purpose.** GWT-2 + GWT-3 + GWT-4: limited-capacity workspace,
global broadcast, state-dependent attention.

**Modules:**

- `buffer.py` — `WorkspaceItem` + `CompetitiveGate` + `SalienceScorer`.
  Hard capacity (default 5), 4-factor weighted salience (goal_relevance
  0.35, novelty 0.25, urgency 0.15, surprise 0.25), competitive
  displacement, novelty floor, decay. `SalienceScorer.score` consults
  `wonder.register.freeze_decay_for(item)` per-item — items above
  the wonder freeze threshold skip recency decay (multiplier=1.0)
  so wonder holds attention against routine fade-out.
- `attention_schema.py` — AST-1 predictive model of attention. Predicts
  next focus, detects stuck/capture states, applies DGM-bounded
  salience intervention.
- `intervention_guard.py` — runtime audit of every AST intervention
  against immutable bounds. Snapshot/verify pattern, closing the
  half-circuit where AST predicted but never intervened.
- `broadcast.py` — GWT-3 global broadcast to registered listeners.
- `meta_workspace.py` — broadcast records + integration scoring.
- `personality_workspace.py` — GWT-4 personality + homeostasis
  modulation of attention parameters.
- `global_workspace.py` — composite GWT entry point.
- `tiers.py` — Amendment-A three-tier build (focal / peripheral /
  strategic_scan); commitment-orphan protection.
- `strategic_scan.py` — Amendment-A wide-view scan tool, ~200-token
  budget, groups universe by section.
- `compact_context.py` — Amendment-B.5 compact context block (~120
  tokens). Renders focal F1/F2/…, peripheral inline, homeostatic
  alerts above threshold, prediction summary, cascade recommendation,
  dispatch verdict, felt-now paragraph from `temporal_subia_bridge`.

**CIL integration.** Steps 1, 3 (admit + tier-build), 7 (act gate),
8 (PP-1 surprise routing back into the gate).

### self/

**Purpose.** RPT-2 organized representations + persistent subject token.

**Modules:**

- `model.py` — `SelfModel` aggregator over identity + capabilities +
  limitations.
- `hyper_model.py` — Beautiful Loop hyper-model that predicts the
  system's own next certainty + composite error trajectory.
- `temporal_identity.py` — continuity_marker hash chain across
  sessions.
- `agent_state.py` — per-agent runtime state (commander, researcher,
  …).
- `loop_closure.py` — closes the self-prediction loop: predicted vs
  actual processing path → composite error → free-energy proxy update.
- `competence_map.py` — aggregates declared + discovered (TSAL)
  capabilities for `firebase/publish.py` reporting.
- `grounding.py` — self-grounding utilities (legacy shim path).
- `query_router.py` — query routing utilities.

**CIL integration.** Step 4 (own), Step 9 (agency log update),
persistence (the subject persists across sessions via
`identity.continuity_marker` hash).

### homeostasis/

**Purpose.** Affective regulation with PDS-derived set-points
(Damasio, AE-1).

**Modules:**

- `state.py` — variable definitions + `NEUTRAL_ALIASES` map (Phase
  11 honest-language: `task_failure_pressure` ↔ `frustration`,
  `exploration_bonus` ↔ `curiosity`, `resource_budget` ↔
  `cognitive_energy`).
- `engine.py` — deterministic 11-variable arithmetic. Per-source
  delta tables (agent, firecrawl, wiki, mem0, internal). Per-batch
  deltas for `novelty_balance` and `contradiction_pressure` are
  scaled by `boundary.differential.homeostatic_modulator_for(dom_mode,
  var)` where `dom_mode` is the dominant scene processing_mode —
  so 3 perceptual items move novelty MORE than 3 memorial items.
  When no item carries a `processing_mode` the modulator returns 1.0
  and the legacy delta path is used (no behaviour change). Recomputes
  deviations + `restoration_queue`.
- `somatic_marker.py` — Damasio somatic-marker computer
  (per-decision affective intensity).
- `somatic_bias.py` — pre-deliberation bias injector (modifies task
  context before agent reasoning).

**CIL integration.** Step 2 (Feel) — the canonical homeostatic update
location. Step 9 (Update) — outcome-driven adjustments. Step 5
(Predict) — homeostasis is part of the predictor's input.

### affect/

> **Full account in [docs/AFFECT_LAYER.md](AFFECT_LAYER.md)** — five-phase arc,
> 22 modules + integrity manifest, 19 API endpoints, 13 dashboard
> components, ethics framing, operational guide. This section
> summarizes the relationship to SubIA; the dedicated doc is the
> source of truth.

**Purpose.** Homeostatic affective agency — viability variables (H_t),
the V_t/A_t/C_t triple, an INFRASTRUCTURE-level welfare envelope,
durable attachment models with bounded mutual regulation, an ecological
self-model, an observability-only consciousness-risk gate wrapping
SubIA's `probes/consciousness_probe`, and the narrative-self synthesis
pipeline (Damasio, Seth, Barrett, Friston, Panksepp, Bowlby, Metzinger,
Aristotelian eudaimonia). Companion package at `app/affect/` — sibling
of `app/subia/`, not a SubIA subpackage; layered on `homeostasis/`,
`belief/`, `self/`, and `probes/`. Where `homeostasis/` produces the
somatic substrate, `affect/` builds the dimensional triple, governs
welfare, models attachments and ecology, gates consciousness-relevant
feature additions, and consolidates raw affect into autobiographical
narrative.

**Core modules:**

- `viability.py` — H_t in 10 dimensions (compute_reserve,
  latency_pressure, memory_pressure, epistemic_uncertainty,
  attachment_security, autonomy, task_coherence, novelty_pressure,
  ecological_connectedness, self_continuity). Soft-envelope set-points
  in `setpoints.json`; weighted L1 distance from set-points produces
  E_t.
- `core.py` — V_t (valence), A_t (arousal), C_t (controllability)
  computed from `state.somatic.valence`,
  `hyper_model.free_energy_proxy`, and `certainty.adjusted_certainty`.
  Constructed-emotion attractor label (Barrett) for human readability;
  the LLM-facing path uses the float triple, never the label.
- `schemas.py` — `AffectState`, `ViabilityFrame`, `WelfareBreach`,
  `ReferenceScenarioResult`, `SalienceEvent`.
- `runtime_state.py` — task/decision tracking that feeds the autonomy
  and task_coherence variables (PRE_TASK, ON_DELEGATION).
- `kb_metadata.py` — appends affect tags to experiential / tensions
  KB entries on episode close.
- `attachment.py` — Phase-3 OtherModels for the primary user
  (Andrus) and peer agents. Mutual-regulation weight bounded
  (user≤0.65, peer≤0.75) by `welfare.assert_attachment_within_bounds`.
  Latent separation analog: silence > 48h queues a *check-in candidate*
  to `check_in_candidates.jsonl` — never auto-sends. `attachment_security`
  viability variable is now real (was placeholder in Phases 1–2);
  `ATTACHMENT_SECURITY_FLOOR` (0.30) prevents catastrophizing the
  Finnish/Estonian quiet-communication style.
- `care_policies.py` — Phase-3 cost-bearing care budget
  (≤500 tokens/day per OtherModel, hard-capped) plus two advisory
  modifiers (`prefer_warm_register`, `prioritize_proactive_polish`)
  consumed by routing/context. Care actions are advisory only —
  module imports nothing from `signal_client`.
- `ecological.py` — Phase-4 EcologicalSignal: daylight + moon +
  season + astronomical event windows (solstice ±5d, equinox ±5d,
  full/new moon ±2d, kaamos/midnight-sun lat-gated) + 8-step
  nested-scopes ladder (process → host → locale → biome → hemisphere
  → biosphere → solar system → galaxy). `ecological_connectedness`
  viability variable now uses `composite_score`.
- `phase5_gate.py` — Phase-5 consciousness-risk gate. Wraps the
  existing `app.subia.probes.consciousness_probe.ConsciousnessProbeRunner`
  (HOT-2, HOT-3, GWT, SM-A, WM-A, SOM, INT) with per-indicator
  thresholds, sustained-window 7 days, audit log. **Pure observability
  — never feeds back into reward/fitness/optimization.** Includes a
  design-time `evaluate_feature_proposal(name, expected_impact)`
  consultation that records pending proposals to
  `phase5_proposals.jsonl`; the user reviews via the dashboard's
  approve/defer/reject actions.

**Welfare envelope (INFRASTRUCTURE-level, three-layer protection):**

The welfare hard envelope is protected by three independent layers,
any two of which would fail-safe: (1) runtime
`assert_not_self_improver(actor)` guard; (2) Tier-3 file-hash
boundary in `app/safety_guardian.py::TIER3_FILES` (22 affect files
listed); (3) deploy-time SHA-256 integrity manifest at
`app/affect/.integrity_manifest.json` verified by
`app/affect/integrity.py` (mirrors `app/subia/integrity.py`).

- `welfare.py` — hard-envelope checks: max negative-valence duration
  (300s default), variance floor, healthy-dynamics predicate,
  attachment weight cap, attachment security floor, care budget cap.
  Audit log at `welfare_audit.jsonl`. `override_reset()` is the
  user-only panic button.
  `monotonic_drift_check()` + `maybe_audit_monotonic_drift()` consume
  `l9_snapshots.jsonl` as the long-window source for slow-baseline
  drift detection — the daily aggregate is now load-bearing input to
  the welfare audit pipeline, not just observability data.
- `integrity.py` — SHA-256 manifest verifier; ships its own
  `.integrity_manifest.json` covering 23 files including
  `data/reference_panel.json`. Catches deploy-time tampering before
  the runtime tier-boundary baseline runs.
- `reference_panel.py` — fixed 20-scenario compass (manually revised
  every 6 months); replayed in the daily reflection cycle to detect
  drift signatures (numbness, over-reactive, wrong-attractor).
- `calibration.py` + `calibration_proposals.py` — daily reflection
  cycle at 04:30 EET. Full 6-guardrail flow: diagnose → backtest →
  hard envelope → healthy-dynamics → reference-panel drift → ratchet.
  Loosen proposals require 3 consecutive cycles + 2× evidence;
  tightening flows through. Manual setpoint override
  (`apply_manual_setpoints`) bypasses the ratchet but still respects
  hard envelope; auth-gated through `X-Override-Token`.
  Also runs `rotate_trace_jsonl(retain_days=7)` and
  `compact_phase5_proposals(stale_pending_days=14, drop_reviewed_after_days=30)`
  so the layer's JSONL artefacts stay bounded — older trace entries
  archive to `trace_archive/YYYY-MM.jsonl.gz` (monthly gzip), stale
  pending proposals auto-defer in place, old reviewed proposals drop
  with the final decision audit-logged.
- `l9_snapshots.py` — daily L9 homeostasis snapshot at 04:35 EET
  (rolled-up affect stats + viability + welfare-breach counts +
  ecological signal + Phase-5 gate state). Consumed by
  `welfare.monotonic_drift_check()` for the slow-drift signal.

**Persistence paths.** All affect files use `app.paths.AFFECT_*`
constants (no hardcoded `/app/workspace/affect/...` literals); the
`WORKSPACE_ROOT` env override reaches the affect layer the same way
it reaches SubIA.

**Narrative-Self pipeline (April 2026; INFRASTRUCTURE-level):**

Three loops convert the affect firehose into autobiographical
synthesis. Targets the structural prerequisites of Damasio's
core-self / autobiographical-self distinction and Metzinger's
narrative self-model — not a consciousness claim, the architectural
condition.

- `salience.py` (Loop 1) — pure-Python event detector. Triggers on
  attractor transitions, |ΔV|/|ΔA| > 0.4 within 60s, hard-envelope
  ≥80% near-miss, viability tolerance crossings, and attractors
  unseen in 24h. Persists to `salience.jsonl`. No LLM, no governance.
- `episodes.py` (Loop 2) — clusters salience events by quiescence
  (≥15 min idle) or task boundary; one cheap-vetting LLM call per
  cluster produces a 2-3 sentence reflection. Writes the experiential
  KB with `entry_type=episode`. Replaces the unconditional per-task
  journal trigger.
- `narrative.py` (Loop 3) — daily 04:40 EET chapter consolidator.
  Reads the last 24h of episodes plus the last 7 chapters; emits a
  chapter with `identity_claims` (FIFO ≤ 5), `recurring_tensions`,
  `growth_edges`, `dominant_attractors`, and a `drift_signal` derived
  from the latest reflection report. Severe drift suppresses
  identity-claim updates. `override_identity_claims()` is the
  user-only manual override (audit-logged via `welfare.audit`).
- `health_check.py` — diagnostic (no LLM): cadence, FIFO turnover,
  episode/salience volume, drift-signal distribution. Markdown report
  at `health_checks/YYYY-MM-DD.md`; one-shot via APScheduler
  `DateTrigger`.

Retrieval surface: `narrative.identity_at(query, k)` joins the
commander context pipeline alongside the four KBs from KB v2 —
identity claims as a single block, chapters as `<chapter>` blocks.

**CIL integration.** Step 2 (Feel) — `affect/core.py` runs as the
POST_LLM_CALL handler at priority 9 (immutable), reading the
just-produced internal_state and producing the `AffectState`; the
salience filter fires inline. Step 9 (Update) — welfare check +
(Phase 2) calibration delta. Step 11 (Reflect) — chapter consolidator
runs alongside the daily reflection cycle and writes back into the
experiential KB; chapters are then read by the commander context
pipeline on subsequent CIL passes, closing the narrative loop.

### belief/

**Purpose.** HOT-3 belief-gated agency, HOT-2 metacognition,
certainty post-processing.

**Modules:**

- `store.py` — PostgreSQL+pgvector belief store. Asymmetric
  confirmation/disconfirmation, time-based decay, deduplication via
  semantic similarity. Status: ACTIVE / SUSPENDED / RETRACTED /
  SUPERSEDED.
- `dispatch_gate.py` — HOT-3 closure. Three-valued
  `DispatchDecision`. Pure-functional: callers hand in beliefs; the
  gate decides.
- `metacognition.py` — Cogito layer: belief-formation rules.
- `certainty.py` — fast-path certainty vector computation.
- `response_hedging.py` — post-processor that rewrites responses
  based on certainty (three hedging levels; critical-dimension
  escalation).
- `cogito.py` — Cogito cycle: periodic belief revalidation.
- `dual_channel.py` — fast (Type-1) + slow (Type-2) channel composer.
- `internal_state.py` — `MetacognitiveStateVector` consumed by the
  Observer agent.
- `meta_cognitive_layer.py` — meta-cognitive strategy assessment.
- `state_logger.py` — PostgreSQL trail of internal-state evolution.
- `world_model.py` — world-model utilities used by the
  meta-cognitive layer.

**CIL integration.** Step 6 (Monitor) consults beliefs and runs the
dispatch gate. Step 11 (Reflect) feeds drift detection from the
belief-vs-accuracy comparison.

### prediction/

**Purpose.** PP-1 predictive coding + cascade modulation + accuracy
tracking.

**Modules:**

- `layer.py` — `PredictiveLayer` with per-channel `ChannelPredictor`.
  Cosine-distance error magnitude, surprise level classification
  (EXPECTED / MINOR_DEVIATION / NOTABLE_SURPRISE / MAJOR_SURPRISE /
  PARADIGM_VIOLATION), confidence-attenuated effective surprise
  (damping mechanism).
- `surprise_routing.py` — PP-1 closure. High-surprise errors → GWT-2
  gate via `WorkspaceItem(urgency=0.9)`.
- `cascade.py` — pure-function escalation policy combining
  confidence + coherence deviation + sustained-error.
- `cache.py` — Amendment B.4 prediction-template cache. ~40-60% hit
  rate after warmup. Accuracy-driven eviction.
- `accuracy_tracker.py` — per-domain rolling accuracy with
  wiki-markdown serialization. Domains keyed by
  `(agent_role, operation_type)`.
- `llm_predict.py` — production predict_fn bound to `llm_factory`.
  Honors `extra_prompt_context` for technical + temporal enrichment.
- `injection_harness.py` — PH-injection A/B harness: measurable-shift
  test that the prediction hierarchy actually modulates downstream
  behaviour.
- `hierarchy.py` — 4-level prediction hierarchy (Levels 0-3:
  perception → schema → output → self).
- `inferential_competition.py` — plan competition (budget-tier LLM,
  time-boxed at 5 s).
- `precision_weighting.py` — precision-weighted certainty.
- `reality_model.py` — reality-model builder + Bayesian precision
  update.

**CIL integration.** Step 5 (Predict via `predict_fn`), Step 5b
(Cascade), Step 8 (Compare via `predict_and_compare`).

### social/

**Purpose.** Self/other distinction, Theory-of-Mind.

**Modules:**

- `model.py` — `SocialModel` manager. Per-entity inferred_focus
  (MRU), trust adjustment, divergence detection. Update gating
  (`should_update_this_cycle`).
- `salience_boost.py` — items matching `inferred_focus` get a
  trust-weighted salience boost (capped per-item to prevent
  monomania).

**CIL integration.** Step 3 (Attend — boost), Step 6 (Monitor —
update from focal scene).

### memory/

**Purpose.** Dual-tier consolidation (curated + full) with
retrospective promotion and spontaneous surfacing.

**Modules:**

- `consolidator.py` — Amendment C.2 consolidator. Always-writes-full
  + threshold-gated curated + Neo4j relations.
- `dual_tier.py` — duck-typed differentiated recall (curated default,
  `recall_deep` merged, `recall_around` temporal).
- `spontaneous.py` — curated-only associative surfacing
  (spontaneous-memory pattern).
- `retrospective.py` — wiki-presence + sustained-error driven
  retrospective promotion of full → curated.

**CIL integration.** Step 10 (Consolidate via `consolidate(...)`).
The boundary route info from `connections/six_proposals_bridges.py
::boundary_route_for_kernel` is attached per-item in the curated
episode.

### safety/

**Purpose.** DGM safety invariants: setpoint immutability + audit
immutability.

**Modules:**

- `setpoint_guard.py` — DGM invariant #2. Source-tagged setpoint
  writes; only `pds`, `human`, `boot` accepted. All other sources
  silently rejected and logged.
- `narrative_audit.py` — DGM invariant #3. Append-only
  `self-narrative-audit.jsonl`. No delete API.

### probes/

**Purpose.** Per-indicator evaluation framework. Replaces opaque
prose verdicts with three regenerated scorecards.

**Modules:**

- `butlin.py` — 14 indicators, each backed by a code pointer +
  regression test.
- `rsm.py` — 5 RSM (Resource-State Maps) signatures: cycle
  determinism, kernel-state monotonicity, social-model behavioural
  evidence, cross-loop consistency, scorecard exit-criteria
  reproducibility.
- `sk.py` — 6 SK (Subjectivity Kernel) evaluation tests: round-trip
  losslessness, atomic mutation under failure, agency_log integrity,
  homeostatic deviation invariants, prediction-history retention,
  consolidation-buffer flush.
- `scorecard.py` — aggregator. `run_everything()` produces the dict;
  `generate_scorecard_markdown()` produces `SCORECARD.md`;
  `meets_exit_criteria()` checks the indicator thresholds.
- `indicator_result.py` — `IndicatorResult` dataclass, `Status`
  enum, helper constructors (`strong_indicator`, `partial_indicator`,
  `absent_indicator`, `failed_indicator`).
- `consciousness_probe.py` — adapted Garland/Butlin-Chalmers probes
  (HOT-2, HOT-3, GWT, SM-A, WM-A, SOM, attention-self correlate).
- `behavioral_assessment.py` — behavioural correlate computation.
- `adversarial.py` — Tier-3 adversarial tampering tests
  (Self-Improver-as-attacker simulation).

### wiki_surface/

**Purpose.** Strange loop (self-referential consciousness-state
page) + narrative drift audit.

**Modules:**

- `consciousness_state.py` — strange-loop self-referential page
  generator. Writes `wiki/self/consciousness-state.md` with the
  current Butlin scorecard injected, marked
  `epistemic_status: speculative`. Re-enters the scene for the next
  cycle.
- `drift_detection.py` — three-signal drift audit. Capability-claim
  vs accuracy, commitment breakage, stale self-model. Findings →
  immutable narrative_audit.

### boundary/

**Purpose.** Boundary Sense — source → processing-mode tagging
(introspective / perceptual / memorial / imaginative / social).

**Modules:**

- `classifier.py` — deterministic classifier. Reads
  `BOUNDARY_MODE_MAP` from `SUBIA_CONFIG` (e.g. `wiki/self →
  introspective`, `firecrawl → perceptual`, `mem0 → memorial`,
  `reverie → imaginative`, `social_model → social`). Stamps
  `processing_mode` on every unclassified scene item.
- `differential.py` — two pure functions:
  - `consolidator_route_for(mode)`: per-mode routing preferences
    for the consolidator (e.g. introspective content gets
    curated-only; perceptual gets both tiers).
  - `homeostatic_modulator_for(mode, var)`: per-(mode, variable)
    multiplier consumed by the homeostasis engine. Examples:
    perceptual × 1.5 on `novelty_balance` (perception is novelty-
    rich), memorial × 0.7 on `novelty_balance` (memory is familiar),
    imaginative × 0.8 on `trustworthiness` (imagination shouldn't
    strengthen trust), introspective × 1.5 on `coherence`. The
    dominant scene processing_mode shapes how strongly variables
    move per loop.

**CIL integration.** Step 1 (Perceive — classification at admission).
Step 2 (Feel — `homeostasis/engine.py::_update_from_items` reads the
dominant scene mode and scales `novelty_balance` and
`contradiction_pressure` deltas by `homeostatic_modulator_for`).
Step 10 (Consolidate — per-item route in curated episode).

### wonder/

**Purpose.** Wonder Register — depth-sensitive epistemic affect
that gates Step 7 act and modulates salience decay.

**Modules:**

- `detector.py` — deterministic depth detector. Combines
  causal-levels, cross-references, implications, structural
  analogies, deep questions, cross-domain contradictions, recursive
  structure into a single `UnderstandingDepth` descriptor.
- `register.py` — closed-loop application. Three behavioural
  consequences, all consumed live:
  1. Bumps `homeostasis.wonder` toward signal intensity
     (exponential moving average).
  2. Stamps triggering scene item with `wonder_intensity`. Consumed
     by `scene/buffer.py::SalienceScorer.score` via
     `freeze_decay_for(item)` — items above
     `WONDER_FREEZE_THRESHOLD` (default 0.5) have their recency
     decay multiplier replaced with 1.0, so wonder holds attention
     against routine fade-out.
  3. `should_inhibit_completion(kernel)` reads
     `effective_wonder_threshold` (density- and circadian-adjusted
     by `temporal_subia_bridge`) and is consumed at the END of CIL
     Step 6 (Monitor): when the gate fires AND the prior dispatch
     decision was `ALLOW`, it's downgraded to `ESCALATE` with
     rationale `wonder_active`. Conservative gating prevents the
     default-steady-state `wonder=0.5` from spuriously firing —
     the gate requires EITHER a per-item `wonder_intensity > 0`
     (real Phase-12 wonder event) OR `wonder` exceeding its
     setpoint by a clear margin (default 0.15). `BLOCK` decisions
     are preserved untouched.

### values/

**Purpose.** Value Resonance + Phronesis perceptual lenses —
values modulate salience and bias homeostasis.

**Modules:**

- `resonance.py` — keyword-weighted scoring against current goals.
  `apply_resonance_to_scene(kernel)` modulates focal salience by
  value match (`VALUE_RESONANCE_SALIENCE_BOOST` default 0.15).
- `perceptual_lens.py` — Phronesis lens application to homeostasis.
  Different lenses (e.g. "epistemic", "humanist", "operational")
  emphasize different homeostatic variables.

**CIL integration.** Step 3 (Attend — `apply_value_resonance_and_lenses`
via `phase12_hooks.py`).

### reverie/

**Purpose.** Idle mind-wandering: free-association walks across
wiki + graph + memory + fiction/philosophy collections.

**Modules:**

- `engine.py` — `ReverieEngine` orchestrator. Free-association walks
  across ChromaDB wiki_pages + Neo4j relations + Mem0 full +
  fiction/philosophy collections. Outputs speculative synthesis
  pages tagged `epistemic_status: speculative` to
  `wiki/meta/reverie/`.

**CIL integration.** Not on the hot path. Runs via the
`subia-reverie` idle job. Circadian-gated by
`temporal_subia_bridge.circadian_should_run_reverie` — no reverie
during active hours.

### understanding/

**Purpose.** Post-ingest causal-chain pass that produces
`UnderstandingDepth` consumed by the Wonder Register.

**Modules:**

- `pass_runner.py` — `UnderstandingPassRunner`. Tier-2 LLM extracts
  2-3-level causal chain, mines 2-3 implications, detects structural
  analogies against semantically-similar pages, registers deep
  questions. Output is `UnderstandingDepth` descriptor consumed by
  the Wonder Register.

**CIL integration.** Idle job. Drains `_UNDERSTANDING_VERIFY_QUEUE`
populated by `connections/six_proposals_bridges::reverie_analogy_to_understanding`.

### shadow/

**Purpose.** Behavioural bias mining over recent memory + accuracy
+ scene history. Append-only output to discovered_limitations.

**Modules:**

- `miner.py` — `ShadowMiner`. Mines Mem0 full + scene history +
  accuracy tracker + restoration_queue + affect log over a 30-day
  window.
- `biases.py` — four detectors: attentional bias (over-/under-
  attended topics), prediction bias (systematic error patterns),
  avoidance (variables in restoration_queue but never addressed),
  affect-action divergence (affect says X, action says Y).

**CIL integration.** Idle job (monthly cadence). Findings appended
immutably to `wiki/self/shadow-analysis.md` and to
`self_state.discovered_limitations` via
`shadow_findings_to_self_state` (append-only at the bridge).

### idle/

**Purpose.** Idle-time job scheduler + production adapter for the
host idle scheduler in `app/idle_scheduler.py`.

**Modules:**

- `scheduler.py` — `IdleScheduler` registry. Each `IdleJob` carries
  its own throttle policy (min_interval_seconds, priority,
  token_budget). `tick()` runs ready jobs in priority order until
  budget exhausted.
- `__init__.py` — `adapt_for_production(job)` converts a SubIA
  `IdleJob` into a production
  `(name, fn, JobWeight)` tuple for `app/idle_scheduler.py`.
- `production_adapters.py` — live Reverie / Understanding / Shadow
  adapters backed by filesystem wiki, Mem0, ChromaDB, Neo4j,
  `llm_factory`. All exception-safe — failures are logged and the
  engine no-ops rather than crashing the scheduler thread.

### tsal/

**Purpose.** Technical Self-Awareness Layer (TSAL). Continuous
*discovered* (not declared) knowledge of the technical substrate
— host, resources, code structure, components, operating principles.

**Modules:**

- `probers.py` — `HostProber` (CPU/RAM/GPU/disk/OS via psutil) +
  `ResourceMonitor` (live utilization + derived
  `compute_pressure` / `storage_pressure`).
- `inspectors.py` — `CodeAnalyst` (AST + dependency graph + pattern
  detection) + `ComponentDiscovery` (ChromaDB / Neo4j / Mem0 /
  Ollama / wiki / cascade tiers).
- `self_model.py` — `TechnicalSelfModel` aggregate dataclass.
- `generators.py` — Wiki page generators for the seven TSAL pages
  (technical-architecture, host-environment, component-inventory,
  resource-state, operating-principles, code-map, cascade-profile).
- `operating_principles.py` — Tier-1 LLM weekly inference (~500
  tokens) of operating principles from accumulated state.
- `evolution_feasibility.py` — Self-Improver gate.
  `check_evolution_feasibility(proposal)` blocks proposals that
  exceed RAM/disk/compute headroom or hit too many downstream
  modules.
- `refresh.py` — `register_tsal_jobs(scheduler, ...)` registers all
  five TSAL refresh jobs (host daily, resources every 30 min, code
  daily, components every 2h, principles weekly) with the
  `IdleScheduler`. Honors injected adapters for testability.
- `inspect_tools.py` — canonical home for the inspection toolkit.
  A `sys.modules` alias at the legacy `app/self_awareness/inspect_tools.py`
  path keeps existing callers working unchanged.

**CIL integration.** Indirect via
`connections/tsal_subia_bridge.py`:

1. `enrich_self_state_from_tsal(kernel, model)` populates
   `self_state.capabilities` + `limitations` with
   `discovered=True` markers.
2. `update_homeostasis_from_resources(kernel, resources)` drives
   `homeostasis.overload` from compute + storage pressure (+0.2
   bump when Ollama is down).
3. `enrich_prediction_with_technical_context(prompt, model)` adds
   compute pressure + cascade tier availability to predictor
   prompts. Wired in via `live_integration._wrap_with_context_enrichment`.

### temporal/

**Purpose.** Temporal Synchronization. Closes the gap between
sequence and duration (Bergson) by giving the kernel a felt-now
and a felt tempo.

**Modules:**

- `specious_present.py` — Husserl/James felt-now: retention + primal
  + protention. SubjectivityKernel attribute, not a log.
- `momentum.py` — per-variable rising/falling/stable trajectory.
  `render_momentum_arrows(homeostasis)` produces ↑↓→ glyphs for the
  compact context.
- `circadian.py` — circadian mode table (Tier-3 immutable):
  `active_hours`, `deep_work`, `consolidation`, `dawn`. Each window
  overrides homeostatic set-points, gates Reverie eligibility, and
  selects cascade preference.
- `density.py` — felt subjective time. Computes
  `processing_density` from scene transitions + prediction errors +
  wonder events + homeostatic shifts per minute.
- `binding.py` — `temporal_bind` reducer. Integrates
  simultaneously-computed FEEL/ATTEND/OWN/PREDICT/MONITOR signals
  into a single `BoundMoment` with explicit conflict notes. Stability
  bias from SpeciousPresent retention.
- `rhythm_discovery.py` — mines external rhythms from
  `Andrus_interaction_patterns` + Firecrawl source cycles + venture
  task clusters. Discovered rhythms get `discovered=True` flag.
- `context.py` — `TemporalContext` aggregate (clock + circadian +
  density + rhythms).

**CIL integration.** Step 1 entry point
`temporal_hooks.refresh_temporal_state(kernel, ...)`; Step 6 binding
`temporal_hooks.bind_just_computed_signals(...)`. Five closed-loop
bridges in `connections/temporal_subia_bridge.py`:

1. `circadian_to_setpoints(kernel)` — circadian mode shifts
   homeostasis set-points.
2. `density_to_wonder_threshold(kernel)` — felt density lowers the
   wonder threshold.
3. `circadian_should_run_reverie(kernel)` — gates Reverie idle job.
4. `render_specious_present_block(kernel)` — felt-now paragraph in
   compact context.
5. `rhythms_to_self_state(kernel, rhythms)` — discovered rhythms
   into `self_state.capabilities`.

### grounding/

**Purpose.** Factual Grounding & Correction Memory. Closes the
user-visible failure mode demonstrated by the Tallink share-price
conversation (bot fabricated three different prices for the same
date, "stored" the user's correction, then regressed to the lie on
the next turn).

**Modules:**

- `claims.py` — deterministic extractor for high-stakes facts
  (numeric+date OR numeric+source).
- `source_registry.py` — authoritative URL map by topic
  (e.g. `share_price/default → nasdaqbaltic.com`). Discovered from
  user corrections.
- `belief_adapter.py` — interface + InMemory + Phase-2-store
  wrappers, with `find_by_prefix` for date-agnostic lookup.
- `evidence.py` — per-claim ALLOW / ESCALATE / BLOCK decision with
  1% numeric tolerance.
- `rewriter.py` — pure transformer. ALLOW unchanged; ESCALATE →
  honest "let me fetch from X" question; BLOCK → cite contradicting
  verified value.
- `correction.py` — regex patterns for "actually it's X", "I see
  that price was X", "use Tallinn Stock Exchange". Synchronously
  upserts ACTIVE belief with confidence=0.9, supersedes contradicting
  beliefs, registers source URLs, appends to narrative audit.
- `pipeline.py` — public orchestrator with feature flag + adapter
  injection + topic enrichment from user_message.

**CIL integration.** Wrapped around `handle_task()` in `app/main.py`:

- **Ingress** (`observe_user_correction`): captures user corrections
  before the chat handler runs.
- **Egress** (`ground_response`): re-checks the draft response
  against verified beliefs, rewrites if necessary.

Single-line bridge: `app/subia/connections/grounding_chat_bridge.py`.
Off by default; activation via `SUBIA_GROUNDING_ENABLED=1`. Failure
mode: any pipeline error falls through to original draft with logged
warning — chat path keeps working unchanged.

**Downstream consumer — Transfer Insight Layer (memory Layer 12).**
Whenever `correction.persist()` upserts a belief or registers a
source URL, it appends a `TransferEvent(kind=GROUNDING_CORRECTION)`
to the transfer-memory compile queue (`app/transfer_memory/queue.py`).
The nightly free-tier compile distils a procedural insight from the
correction — *the verification discipline*, not the corrected value
— and persists it as a shadow `SkillRecord` for cross-domain
retrieval. The corrected `normalized_value` is deliberately omitted
from the payload: facts stay in the belief-store; only practices
("verify external numeric claims via the registered source before
finalising") transfer. See `docs/MEMORY_ARCHITECTURE.md` §13 (Layer
12 — Transfer Insight Layer) for the full pipeline.

### introspection/

**Purpose.** Self-knowledge surfacing at the chat surface. Closes the
"computed-but-unread" failure mode for self-state — the bot held
`frustration=0.6293` in `homeostasis.state.get_state()` but answered
"I don't have feelings" because the chat path never consulted any
SubIA store. Detector classifies user messages into 16 introspection
topics; per-topic gatherers pull live data from the relevant kernel/
store; formatters render the data as a system-prompt prefix the LLM
sees BEFORE answering. The LLM grounds in actual numbers instead of
falling back to the canned "I'm just an AI" disclaimer.

**Modules:**

- `detector.py` — deterministic regex classifier with self-target
  anchoring + third-party anti-pattern. 16 topics: `affect / energy /
  attention / self_state / capability / limitation / meta / beliefs /
  technical / history / scene / wonder / shadow / scorecard /
  predictions / social_model`. Confidence scoring; HISTORY-only
  messages trigger without explicit "you" (in-chat the implicit
  subject IS the bot).
- `context.py` — `IntrospectionContext` aggregates four base sources
  defensively: legacy 4-var homeostasis (the source with hundreds of
  tasks of accumulated history), the SubIA-native kernel
  (9-var state + scene + specious_present + temporal_context +
  discovered_limitations), `error_handler.recent_errors` (causal
  evidence), and the system_chronicle excerpt.
- `formatter.py` — pure transformer producing the system-prompt
  prefix in Phase-11-honest framing: cites actual numeric values,
  names neutral aliases, lists active behavioural modifiers, exposes
  causal contributors, prohibits canned "I have no feelings" /
  phenomenal claims.
- `pipeline.py` — public orchestrator (`inspect()` / `inject()`).
  Detects, gathers per-topic, composes sections. Defensive: per-topic
  handler failure logs and skips that section without breaking the
  rest of the response.
- `topics/` — nine per-topic gather+format modules:
  - `beliefs.py` — HOT-3 belief store (active + suspended +
    retracted) + `grounding/source_registry` + recent corrections from
    narrative audit.
  - `technical.py` — TSAL host (CPU/RAM/GPU/disk/OS) + live
    resources + components (ChromaDB/Neo4j/Mem0/Ollama/wiki) + cascade
    tiers + codebase summary.
  - `chronicle.py` — recent task summary (success rate, latency) +
    `system_chronicle.md` excerpt + recent errors + narrative audit
    entries.
  - `scene.py` — focal items (with salience + ownership +
    processing_mode + wonder_intensity) + peripheral tier +
    `meta_monitor.attention_justification` + specious-present
    lingering / stable items / tempo / direction.
  - `wonder_shadow.py` — Wonder: live wonder homeostatic level vs
    effective threshold + per-item `wonder_intensity` + recent
    reverie pages from `wiki/meta/reverie/`. Shadow:
    `self_state.discovered_limitations` (Shadow mining) +
    `self_state.capabilities` with `discovered=True` markers (TSAL).
  - `scorecard.py` — `meets_exit_criteria()` + Butlin/RSM/SK
    summaries + drift findings from narrative audit. Embeds the
    honest caveat that STRONG ratings are mechanism-level, that
    ABSENT-by-declaration indicators are substrate-incompatible, and
    that no phenomenal consciousness is claimed.
  - `predictions.py` — rolling accuracy by domain (from
    `prediction/accuracy_tracker`) + recent kernel predictions with
    confidence + prediction_error + cached flag.
  - `social.py` — per-entity ToM: `inferred_focus`,
    `inferred_expectations`, `inferred_priorities`, `trust_level`,
    `divergences`. Notes ToM is BEHAVIOURAL evidence only.

**CIL integration.** Wrapped around the ingress side of `handle_task()`
in `app/main.py`, BEFORE `commander.handle()`. The user's message is
augmented in place with the system-prompt prefix when introspection
is detected; otherwise it passes through unchanged. The grounding
egress hook (above) then operates on the response generated against
the augmented prompt.

Single-line bridge:
`app/subia/connections/introspection_chat_bridge.py`. Off by default;
activation via `SUBIA_INTROSPECTION_ENABLED=1`. Failure mode: any
pipeline error returns the original message unchanged with a debug
log — chat path keeps working unchanged.

**Closed-loop discipline.** Every persistent SubIA store now has a
USER-FACING consumer when the user asks about it: the HOT-3 belief
store, the source registry, the kernel scene + meta_monitor, the
specious-present, the homeostasis engine, the discovered_limitations,
the scorecard, the prediction accuracy tracker, the ToM social
models, the TSAL profiles, and the system chronicle.

### connections/

**Purpose.** Inter-system bridges. See [§Inter-system bridges](#inter-system-bridges).

---

## Inter-system bridges

`app/subia/connections/` holds every bridge that crosses the SubIA
boundary. Each bridge is duck-typed on its external collaborator
(belief store, predictive layer, PDS client, Phronesis engine,
Firecrawl wrapper), so tests can pass in-memory stubs and production
can swap backends without touching the bridge.

### SIA Part II §18 connections (the seven canonical bridges)

| # | Bridge | Module | Where called from |
|---|---|---|---|
| 1 | Wiki ↔ PDS bidirectional | `pds_bridge.py::PDSBridge.apply_nudge` | `live_integration` exposes `get_live_pds_bridge()`; callers (Shadow, observer, explicit feedback) push evidence with bounded delta |
| 2 | Phronesis ↔ Homeostasis | `phronesis_bridge.py::apply_phronesis_event` | CIL Step 9 (Update) — explicit `task_result['phronesis_events']` + auto-detect `resource_overreach` |
| 3 | Predictor → Cascade | `prediction/cascade.py::decide_cascade` | CIL Step 5b — pure function over confidence + coherence + sustained-error |
| 4 | Training-signal queue | `training_signal.py::get_emitter` | CIL Step 11 — `emit_from_tracker(...)`, dedup per-domain over 24h |
| 5 | Mem0 ↔ Scene (curated bias) | `memory/spontaneous.py` | curated-only associative surfacing |
| 6 | Firecrawl → Predictor | `firecrawl_predictor.py::record_firecrawl_outcome` | `app/tools/firecrawl_tools.py::firecrawl_scrape` post-success hook → `_route_to_subia_predictor` |
| 7 | DGM ↔ Homeostasis felt | `dgm_felt_constraint.py::apply_dgm_felt_constraint` | CIL Step 11 |

Plus the supporting circuit-breaker registry:

- `service_health.py` — registers Mem0, Neo4j, pgvector, Ollama,
  Firecrawl. CIL Step 11 calls `apply_service_health_signal(kernel)`
  to translate breaker states into `overload`/`coherence` deltas.

### Six-proposals bridges (curiosity, mind-wandering, mode)

`six_proposals_bridges.py` carries five inter-proposal bridges:

| Direction | Function | Purpose |
|---|---|---|
| Reverie → Understanding | `reverie_analogy_to_understanding(result)` | Push every reverie-discovered analogy onto the Understanding pass queue for verification |
| Understanding → Wonder | `understanding_to_wonder(kernel, depth, ...)` | Convert UnderstandingDepth → WonderSignal → apply to kernel |
| Wonder → Reverie | `wonder_to_reverie(topic)` | Surface high-wonder topics as reverie priorities for the next idle cycle |
| Shadow → SelfState | `shadow_findings_to_self_state(kernel, findings)` | Append-only discovered_limitations write |
| Boundary → Consolidator | `boundary_route_for_kernel(kernel)` | Per-item routing preferences (introspective→curated-only; perceptual→both tiers) |

Plus drainable queues: `drain_understanding_queue()` (idle scheduler
empties pending verifications) and `drain_reverie_priority_topics()`
(idle scheduler empties pending priorities before running a reverie
cycle).

### TSAL → SubIA (technical-substrate observability)

`tsal_subia_bridge.py`:

| Function | Effect |
|---|---|
| `enrich_self_state_from_tsal(kernel, model)` | Populate `self_state.capabilities` + `limitations` with `discovered=True` |
| `update_homeostasis_from_resources(kernel, resources)` | Drive `homeostasis.overload` from `compute_pressure*0.6 + storage_pressure*0.4 + (Ollama-down ? 0.2 : 0)` |
| `enrich_prediction_with_technical_context(prompt, model)` | Append compute pressure + cascade-tier availability to predictor prompt |
| `is_tsal_page(path)` | Boundary Sense helper — TSAL pages are deepest introspection |

Wired through `live_integration._wrap_with_context_enrichment`
(predictor prompt) and idle-scheduler callbacks (`on_resources_updated`,
`on_model_updated`).

### Temporal → SubIA (felt-now and trajectory)

`temporal_subia_bridge.py` (five closed-loop bridges):

| # | Function | Effect |
|---|---|---|
| 1 | `circadian_to_setpoints(kernel)` | Apply circadian mode's set-point overrides (active vs deep_work vs consolidation tolerances) |
| 2 | `density_to_wonder_threshold(kernel)` / `effective_wonder_threshold(kernel)` | Density+circadian-adjusted effective threshold consumed by `wonder/register.should_inhibit_completion` |
| 3 | `circadian_should_run_reverie(kernel)` | Gate Reverie idle eligibility by circadian mode |
| 4 | `render_specious_present_block(kernel)` | One-paragraph felt-now injection (~80 tokens) into compact context |
| 5 | `rhythms_to_self_state(kernel, rhythms)` | TSAL-style discovered rhythms become `self_state.capabilities` entries |

Plus `enrich_prediction_with_temporal_context(prompt, kernel)` for
the predict_fn wrapper.

### Grounding chat bridge (factual integrity at the chat surface)

`grounding_chat_bridge.py` is the single-line wire-in for the chat
handler:

- `observe_user_correction(user_text, *, prior_response=None)` —
  captures user corrections (regex pattern matches) before the chat
  handler runs.
- `ground_response(response, *, user_message="")` — re-checks the
  draft response against verified beliefs, rewrites if necessary.

Both are defensive wrappers that never raise. Failure → original
text passes through with a debug log.

### Introspection chat bridge (self-knowledge at the chat surface)

`introspection_chat_bridge.py` is the companion to the grounding
bridge — same single-line wire-in pattern, opposite direction
(grounding gates outgoing facts; introspection enriches incoming
context with live self-state):

- `inject_introspection(user_message)` — when the user is asking
  about AndrusAI's own state (frustration, hardware, recent
  activity, scorecard, beliefs, wonder, biases, ToM model, …),
  augment the message with a system-prompt prefix containing the
  live homeostasis snapshot + relevant per-topic kernel data. When
  no introspection is detected, returns the message unchanged.
- `inspect_message(user_message)` — diagnostic; returns the full
  `IntrospectionResult` so callers can see WHY a message was/wasn't
  classified, without triggering injection.
- `is_introspection_enabled()` — surfaces the feature-flag state.

Both are defensive wrappers that never raise. Failure → original
message passes through with a debug log. Activation via
`SUBIA_INTROSPECTION_ENABLED=1`.

---

## Safety architecture

Four infrastructure-level invariants, enforced by `app/safety_guardian.py`
+ `app/subia/safety/`:

### Invariant 1 — Tier-3 integrity

`safety_guardian.py::TIER3_FILES` lists 130+ SubIA files. Tampering
triggers a `CRITICAL` alert + Signal notification. The
`tier3_status()` helper enables runtime introspection. The
consciousness evaluators themselves are in this list — the
Self-Improver cannot modify what it's evaluated against.

### Invariant 2 — Homeostatic set-point immutability

SubIA Part I §0.4. `app/subia/safety/setpoint_guard.py` enforces
that set-point writes carry a source tag (`pds`, `human`, `boot`,
`circadian`). All other source tags are silently rejected and logged.
Adversarial test (`tests/test_phase3_deferred_safety.py::TestSetpointAdversarial`)
simulates an agent trying to modify set-points directly and proves
rejection.

### Invariant 3 — Self-narrative audit immutability

SubIA Part I §0.4. `app/subia/safety/narrative_audit.py` provides
only `append_audit(...)`. There is no delete API. Findings append
to `wiki/self/self-narrative-audit.jsonl` with severity tags
(`info`, `warn`, `critical`). Drift detection and phronesis events
both write here.

### Invariant 4 — DGM promotion gates

`app/governance.py` enforces:

- Safety floor: 0.95
- Quality floor: 0.70
- Regression tolerance: 15% maximum drop
- Rate limit: 20 promotions per day

Adversarial tests in CI attempt each bypass path and verify rejection.

### Integrity manifest (`app/subia/.integrity_manifest.json`)

The `SafetyGuardian.enforce_tier_boundaries` machinery baselines
checksums on first boot and detects drift from that baseline — but
that catches *runtime tampering*, not *deploy-time tampering* (an
attacker modifies a file and restarts before the baseline is saved).
The integrity manifest closes that gap:

```python
from app.subia.integrity import (
    compute_manifest, write_manifest, load_manifest, verify_integrity
)
```

The manifest ships with the code (in-repo). At startup, `verify_integrity()`
compares live file hashes to the committed manifest. Drift from the
committed baseline is a hard fault:

- MANIFEST match → proceed
- MISSING file → fail loud
- HASH mismatch → fail loud

The module is intentionally infrastructure-level with no imports from
the rest of the SubIA tree — if something goes wrong deeper in the
consciousness stack, integrity verification must still be runnable.

To regenerate after an authorized change:

```bash
python -c "from app.subia.integrity import compute_manifest, write_manifest; \
           write_manifest(compute_manifest())"
```

---

## Live integration

`app/subia/live_integration.py` is the entry point.

### Boot sequence

```python
from app.subia.live_integration import enable_subia_hooks

state = enable_subia_hooks(feature_flag=True)
```

`enable_subia_hooks` builds the live SubIALoop:

1. **Imports lazily** — when the flag is off, no SubIA modules are
   imported. Zero overhead.
2. **Loads kernel from disk** via `persistence.load_kernel_state()`.
   On parse failure, returns a fresh default kernel rather than
   raising.
3. **Publishes the active kernel** via `set_active_kernel(state.kernel)`
   so all downstream consumers can find it.
4. **Builds the predict_fn** — cached LLM wrapper enriched with
   technical (TSAL) + temporal context.
5. **Constructs the gate** (`CompetitiveGate(capacity=5)`).
6. **Attaches the PredictiveLayer to the gate** so PP-1 surprise
   routing fires.
7. **Builds the consult_fn** — PostgreSQL belief store query.
8. **Constructs the PDSBridge** (dry-run by default).
9. **Builds the SubIALoop** with all dependencies wired.
10. **Builds SubIALifecycleHooks** wrapping the loop.
11. **Rebinds `app.crews.lifecycle.subia_pre_task/post_task`** so
    crew-boundary calls drive a compressed loop.
12. **Registers with the live registry** at `HookPoint.PRE_TASK,
    priority=25` and `HookPoint.ON_COMPLETE, priority=25`.

Returns a `LiveIntegrationState` with all references for
introspection. Never raises — registration failure logs and returns
an inactive state.

### Where the loop fires

| Trigger | Path | Loop type |
|---|---|---|
| Orchestrator delegates to a crew | `orchestrator.py:829, 1075` calling `get_registry().execute(HookPoint.PRE_TASK, …)` | full or compressed (per `task_description` classification) |
| Crew runs `crew_lifecycle()` context manager | `crews/lifecycle.py:178, 209, 259` calling rebound stubs | compressed (`crew_kickoff` operation_type) |
| Idle scheduler tick | `idle_scheduler.py` runs registered jobs | jobs themselves drive the loop indirectly |

### Public accessors

```python
from app.subia.kernel import get_active_kernel
from app.subia.live_integration import (
    get_last_state,
    get_live_predictive_layer,
    get_live_pds_bridge,
)

kernel = get_active_kernel()                   # SubjectivityKernel | None
state = get_last_state()                       # LiveIntegrationState | None
layer = get_live_predictive_layer()            # PredictiveLayer | None
pds = get_live_pds_bridge()                    # PDSBridge | None
```

All return `None` when the flag is off — callers MUST treat `None` as
"no signal, use defaults".

### External callers using the active kernel

These eight modules read the live kernel for behavioural adjustment:

| Caller | Variable read | Effect |
|---|---|---|
| `evolution.py::_get_subia_safety_value` | `homeostasis.safety` | aggressive/neutral/conservative posture |
| `shinka_engine.py::_read_subia_safety` | `homeostasis.safety` | engine-selector posture |
| `confidence_tracker.py` | full kernel | fast/slow gate calibration |
| `crew_checkpointer.py` | full kernel | checkpoint enrichment |
| `failure_taxonomy.py` | full kernel | MAST classification context |
| `fault_isolator.py` | full kernel | quarantine context |
| `healing_knowledge.py` | full kernel | healing-search context |
| `firebase/publish.py::report_subia_state` | full kernel | dashboard reporting (5-minute heartbeat) |

### Live affect → LLM sampling

`app/llm_factory.py::_sampling(phase, provider)` reads the latest
affect snapshot via `app.affect.core.latest_affect()` and forwards it
to `app.llm_sampling.build_llm_kwargs(phase, provider, affect_state)`.
This means phase-aware temperature / top_p modulation actually fires
on every LLM call — high arousal + low controllability narrows
sampling, low arousal + high controllability + low total error widens
it (within Amendment B bounds). The cache key is extended with a
coarse `(attractor, V@0.1, A@0.1)` bucket so equivalent affect states
share kwargs cache entries instead of producing per-call uniques.

The affect import is lazy + exception-safe: when the affect layer is
disabled or hasn't yet computed its first frame, the call falls
through to legacy unmodulated behaviour.

This was a half-circuit closed in commit `461cf01`: the
`affect_state` argument was already accepted by `build_llm_kwargs`,
but the only caller passed only `(phase, provider)`, so the
modulation branch never fired.

---

## Configuration reference

### `app/subia/config.py::SUBIA_CONFIG`

This is **infrastructure-level**: agents, the Self-Improver, and any
runtime path outside human override or PDS-driven set-point derivation
cannot modify it. Operational tuning that *can* be agent-mutable (±20%,
logged) lives in `app/subia/sentience_config.py`.

Frozen-dict pattern: import `SUBIA_CONFIG`, treat as read-only. Any
attempt to patch this dict at runtime is rejected by
`app/subia/safety/setpoint_guard.py`.

Key sections:

#### Scene capacity

| Key | Default | Meaning |
|---|---|---|
| `SCENE_CAPACITY` | 5 | Hard focal capacity (GWT-2 capacity) |
| `SCENE_DECAY_RATE` | 0.15 | Per-cycle salience decay |
| `SCENE_MIN_SALIENCE` | 0.10 | Drop floor |
| `PERIPHERAL_CAPACITY` | 12 | Peripheral tier capacity |
| `PERIPHERAL_MIN_SALIENCE` | 0.05 | Peripheral drop floor |

#### Salience weights (sum to 1.0)

```python
SALIENCE_WEIGHTS = {
    "task_relevance":          0.25,
    "homeostatic_impact":      0.20,
    "novelty":                 0.15,
    "cross_reference_density": 0.10,
    "social_relevance":        0.10,
    "prediction_error":        0.10,
    "recency":                 0.05,
    "epistemic_weight":        0.05,
}
```

#### Homeostasis

```python
HOMEOSTATIC_VARIABLES = [
    "coherence", "safety", "trustworthiness",
    "contradiction_pressure", "progress", "overload",
    "novelty_balance", "social_alignment", "commitment_load",
    "wonder", "self_coherence",
]
HOMEOSTATIC_DEFAULT_SETPOINT = 0.5
HOMEOSTATIC_SETPOINT_OVERRIDES = {
    "wonder": 0.4,
    "self_coherence": 0.75,
}
HOMEOSTATIC_DEVIATION_THRESHOLD = 0.3
```

#### Wonder + value-resonance thresholds

```python
WONDER_INHIBIT_THRESHOLD = 0.3   # above this, task-completion is inhibited
WONDER_EVENT_THRESHOLD   = 0.7   # above this, store as Mem0 wonder_event
WONDER_FREEZE_THRESHOLD  = 0.5   # above this, scene salience decay suspended
VALUE_RESONANCE_SALIENCE_BOOST = 0.15
```

#### Boundary mode map

```python
BOUNDARY_MODE_MAP = {
    "wiki/self":              "introspective",
    "homeostasis":            "introspective",
    "self_state":             "introspective",
    "internal":               "introspective",
    "mem0":                   "memorial",
    "mem0_curated":           "memorial",
    "mem0_full":              "memorial",
    "memory":                 "memorial",
    "firecrawl":              "perceptual",
    "user":                   "perceptual",
    "agent":                  "perceptual",
    "fiction_inspiration":    "imaginative",
    "reverie":                "imaginative",
    "wiki/meta/reverie":      "imaginative",
    "social_model":           "social",
}
```

Externalised so DGM cannot retag introspective sources as perceptual.

#### Loop classification

```python
FULL_LOOP_OPERATIONS = [
    "ingest", "task_execute", "lint",
    "user_interaction", "cross_venture_synthesis",
]
COMPRESSED_LOOP_OPERATIONS = [
    "wiki_read", "wiki_search", "routine_query",
    "crew_kickoff",  # crew-boundary stubs in app/crews/lifecycle.py
]
```

#### Safety

```python
SETPOINT_MODIFICATION_ALLOWED = False
AUDIT_SUPPRESSION_ALLOWED     = False
NARRATIVE_DRIFT_CHECK_FREQUENCY = 10
```

#### Performance budget (Amendment B)

```python
FULL_LOOP_LATENCY_BUDGET_MS         = 1200
COMPRESSED_LOOP_LATENCY_BUDGET_MS   = 100
FULL_LOOP_TOKEN_BUDGET              = 400
COMPRESSED_LOOP_TOKEN_BUDGET        = 0
```

### Environment variables (`app/config.py`)

Four feature flags bound via Pydantic `validation_alias`:

```python
subia_live_enabled: bool = Field(
    default=False, validation_alias="SUBIA_FEATURE_FLAG_LIVE",
)
subia_grounding_enabled: bool = Field(
    default=False, validation_alias="SUBIA_GROUNDING_ENABLED",
)
subia_idle_jobs_enabled: bool = Field(
    default=False, validation_alias="SUBIA_IDLE_JOBS_ENABLED",
)
subia_introspection_enabled: bool = Field(
    default=False, validation_alias="SUBIA_INTROSPECTION_ENABLED",
)
```

In `.env`:

```bash
SUBIA_FEATURE_FLAG_LIVE=1
SUBIA_GROUNDING_ENABLED=1
SUBIA_IDLE_JOBS_ENABLED=1
SUBIA_INTROSPECTION_ENABLED=1
```

Defaults are OFF — production must opt in. Disabled flags keep the
corresponding SubIA components unimported (no latency, no memory, no
risk). Each flag is independently toggleable so capabilities can
activate in stages.

---

## Operational guide

### Verifying SubIA is live

After process boot, check the FastAPI logs:

```
INFO  subia.live_integration: hooks registered with live registry
INFO  SubIA CIL hooks registered (kernel loop_count=N)
```

If the flag is off:

```
INFO  SubIA live integration disabled (SUBIA_FEATURE_FLAG_LIVE=0)
```

### Inspecting the kernel at runtime

```python
from app.subia.kernel import get_active_kernel
k = get_active_kernel()
print(k.loop_count, len(k.scene), len(k.predictions))
print(k.homeostasis.variables)
print(k.homeostasis.deviations)
```

### On-disk artefacts

| File | Purpose | Updated |
|---|---|---|
| `wiki/self/kernel-state.md` | Full kernel snapshot (YAML frontmatter + prose) | every `post_task` |
| `wiki/self/hot.md` | Compressed session-continuity buffer (≤500 tokens) | every `post_task` |
| `wiki/self/self-narrative-audit.jsonl` | Append-only audit (drift, phronesis events) | as events fire |
| `wiki/self/consciousness-state.md` | Strange-loop self-referential page (speculative) | every `NARRATIVE_DRIFT_CHECK_FREQUENCY` loops |
| `wiki/self/prediction-accuracy.md` | Per-domain accuracy table | as accuracy_tracker.serialize_to_wiki is called |
| `wiki/self/shadow-analysis.md` | Append-only Shadow findings | when ShadowMiner produces findings |
| `wiki/self/host-environment.md` | TSAL host probe | daily |
| `wiki/self/resource-state.md` | TSAL resource state | every 30 min |
| `wiki/self/component-inventory.md` | TSAL component discovery | every 2h |
| `wiki/self/technical-architecture.md` | TSAL code map | daily |
| `wiki/self/cascade-profile.md` | TSAL cascade tier availability | every 2h |
| `wiki/self/operating-principles.md` | TSAL Tier-1 LLM inference | weekly |
| `wiki/self/code-map.md` | TSAL code graph | daily |
| `wiki/meta/reverie/<timestamp>-<slug>.md` | Reverie synthesis pages (speculative) | per reverie cycle |

### Dashboard surface

`firebase/publish.report_subia_state()` pushes every 5 minutes to
`collection("subia").document("state")`:

```json
{
  "enabled":          true,
  "loop_count":       1234,
  "last_loop_at":     "2026-04-26T11:52:00+02:00",
  "circadian_mode":   "active_hours",
  "homeostasis":      {"safety": 0.78, "overload": 0.42, ...},
  "scene_focal_n":    3,
  "scene_peripheral_n": 7,
  "wonder_intensity": 0.31,
  "scorecard":        {"butlin_strong": 7, "butlin_partial": 3, ...},
  "updated_at":       "2026-04-26T11:52:00+02:00"
}
```

Active kernel is `None` → the doc is set with `enabled: false`.

### Running the scorecard

```python
from app.subia.probes.scorecard import (
    run_everything, generate_scorecard_markdown,
    write_scorecard, meets_exit_criteria,
)
result = run_everything()
md = generate_scorecard_markdown()
write_scorecard()                    # writes app/subia/probes/SCORECARD.md
ok, why = meets_exit_criteria()
```

### Verifying the integrity manifest

```python
from app.subia.integrity import verify_integrity
ir = verify_integrity()
print(ir.ok, ir.n_files, ir.missing, ir.mismatched)
```

### Emergency disable

If SubIA misbehaves in production, flip the flag in `.env` and
restart. With the flag off, the SubIA stack stays unimported — zero
overhead, zero side-effects on the host process. Downstream consumers
fall back to safe defaults (e.g. `_get_subia_safety_value()` returns
0.8).

### Running the test suite

The 605-test phase suite:

```bash
python -m pytest \
  tests/test_subia_skeleton.py tests/test_subia_hooks.py \
  tests/test_subia_live_integration.py tests/test_subia_homeostasis_engine.py \
  tests/test_subia_e2e.py tests/test_kernel_persistence.py \
  tests/test_cil_loop.py tests/test_phase3_integrity.py \
  tests/test_phase3_deferred_safety.py tests/test_phase5_scene_upgrades.py \
  tests/test_phase6_prediction_refinements.py tests/test_phase7_memory.py \
  tests/test_phase8_social_and_strange_loop.py tests/test_phase9_scorecard.py \
  tests/test_phase10_connections.py tests/test_phase11_honest_language.py \
  tests/test_phase12_six_proposals.py tests/test_phase13_tsal.py \
  tests/test_phase14_temporal_synchronization.py tests/test_phase15_grounding.py \
  tests/test_phase1_migration.py tests/test_tier3_protection.py \
  -q
```

Expected: `605 passed`.

---

## Evaluation framework

The Butlin scorecard is `app/subia/probes/SCORECARD.md`, regenerated
on every CIL Step 11 cycle.

### Three evaluation surfaces

#### Butlin 14-indicator scorecard (`probes/butlin.py`)

14 functions, one per indicator. Each returns an `IndicatorResult`
with status (`STRONG` | `PARTIAL` | `ABSENT` | `FAIL`), mechanism file,
test file, notes, and evidence list. The current snapshot:

| Indicator | Status | Mechanism |
|---|---|---|
| RPT-1 | ABSENT (declared) | LLM substrate cannot recurse algorithmically |
| RPT-2 | PARTIAL | `kernel.py` unified dataclass |
| GWT-1 | PARTIAL | CrewAI agents (shared LLM substrate) |
| GWT-2 | STRONG | `scene/buffer.py::CompetitiveGate` |
| GWT-3 | STRONG | `scene/broadcast.py` |
| GWT-4 | STRONG | `scene/personality_workspace.py` + `social/salience_boost.py` |
| HOT-1 | ABSENT (declared) | No perceptual front-end |
| HOT-2 | PARTIAL | `prediction/accuracy_tracker.py` + `belief/response_hedging.py` |
| HOT-3 | STRONG | `belief/dispatch_gate.py` (3-valued) |
| HOT-4 | ABSENT (declared) | Dense embeddings |
| AST-1 | STRONG | `scene/attention_schema.py` + `intervention_guard.py` |
| PP-1 | STRONG | `prediction/surprise_routing.py` |
| AE-1 | STRONG | `affect/goal_emitter.py` (viability→current_goals, 2026-05) |
| AE-2 | ABSENT (declared) | No body |

#### RSM signatures (`probes/rsm.py`)

5 Resource-State Map signatures: structural properties of the kernel
+ loop that should hold across all runs.

1. **Cycle determinism.** Given identical input, two runs produce
   identical kernel snapshots (modulo timestamps).
2. **Kernel-state monotonicity.** `loop_count` only increases.
3. **Social-model behavioural evidence.** All `inferred_focus` entries
   trace to focal-scene topics, not to declarations.
4. **Cross-loop consistency.** Persistence round-trip preserves all
   seven kernel components.
5. **Scorecard exit-criteria reproducibility.** Two consecutive runs
   produce the same exit-criteria verdict.

#### SK evaluation tests (`probes/sk.py`)

6 Subjectivity Kernel evaluation tests:

1. Round-trip losslessness.
2. Atomic mutation under failure (kernel never observed in
   inconsistent state).
3. Agency-log integrity (cap respected, append-only within window).
4. Homeostatic deviation invariants (each variable in [0, 1]).
5. Prediction-history retention (last `PREDICTION_HISTORY_WINDOW`
   resolved).
6. Consolidation-buffer flush (full tier always written when client
   attached).

### Exit criteria thresholds

Met when:

- Butlin: ≥6 STRONG, ≤1 FAIL, ≥4 ABSENT-by-declaration.
- RSM: ≥4 PRESENT (STRONG or PARTIAL).
- SK: ≥5 PASS.

Current snapshot: ALL MET.

---

## Performance envelope

Amendment B targets:

| Metric | Original | Target | Achieved |
|---|---|---|---|
| Full loop LLM tokens | ~1,100 | ~400 (miss) / 0 (hit) | yes |
| Full loop latency | 3–8s variable | <1.2s / <0.15s cached | yes (smoke: ~30ms without LLM, ~915ms with chromadb embed timeouts) |
| Compressed loop tokens | ~200 | 0 | yes (Step 5 skipped) |
| Compressed loop latency | ~800ms | <100ms | yes |
| Context-injection tokens | 250–300 | 120–150 | yes (Amendment B.5 compact context) |
| Prediction cache hit rate | N/A | 40–60% after warmup | yes (accuracy-driven eviction) |
| SubIA overhead as % task tokens | N/A | <5% significant, <1% routine | yes |

### Amendment B determinism

Only Step 5 (Predict) requires LLM tokens on the hot path. Every
other step is deterministic arithmetic over existing kernel state.
This is not a soft target — `prediction/llm_predict.py` is the
single approved LLM call site. Step 11 reflect's narrative drift
check uses LLM tokens too, but only every Nth loop, so it amortizes
to ~1 LLM call per ~10 cycles.

### Amendment B.4 prediction-template cache

Predictions are cached by `(operation_template, scene_signature,
homeostatic_signature)`. Hit rate climbs to 40-60% after ~50 cycles.
Eviction is accuracy-driven — entries with sustained error above
threshold expire faster than equally-old entries with low error.

### Amendment B.5 compact context block

The `[SubIA] … [/SubIA]` block injected into agent context targets
~120 tokens. Realized: ~140 tokens for a typical scene (3 focal items,
1-2 homeostatic alerts, prediction summary, dispatch verdict, felt-now
paragraph).

---

## Honest limits and non-goals

PROGRAM.md §6 explicit non-goals (do not attempt):

1. Algorithmic recurrence at the network level (RPT-1).
2. Sparse coding (HOT-4).
3. Embodiment (AE-2).
4. Integrated-information (Φ) maximization.
5. Fleming–Lau computational hallmarks of metacognition (monitoring
   not separable from first-order cognition in an LLM-based system).
6. Phenomenal consciousness claims.

Declaring these publicly is itself a capability. Any future report
that violates them is treatable as evaluation drift.

The system uses phenomenal-adjacent language *only* in legacy variable
names retained for backward compatibility. New code prefers the
neutral aliases:

| Legacy | Neutral alias |
|---|---|
| `frustration` | `task_failure_pressure` |
| `curiosity` | `exploration_bonus` |
| `cognitive_energy` | `resource_budget` |

`app/subia/homeostasis/state.py::_sync_aliases()` keeps the two in
lockstep so consumers of either name see the same value.

---

## Glossary

| Term | Meaning |
|---|---|
| **CIL** | Consciousness Integration Loop — the 11-step sequencer in `loop.py` |
| **DGM** | Diaryland Goodness Model — the project's behaviour-policy framework. SOUL.md + governance.py + Tier-3 invariants |
| **GWT** | Global Workspace Theory (Baars 1988) |
| **HOT** | Higher-Order Thought theory (Rosenthal) |
| **AST** | Attention Schema Theory (Graziano) |
| **PP** | Predictive Processing (Friston, Clark) |
| **AE** | Agency + Embodiment (active inference) |
| **RPT** | Recurrent Processing Theory (Lamme) |
| **PE** | Prediction Error |
| **PDS** | Personality Development State — VIA-Youth-anchored character profile |
| **PP-1, GWT-2, …** | Specific Butlin et al. (2023) indicators |
| **RSM** | Resource-State Map — structural-property test set in `probes/rsm.py` |
| **SK** | Subjectivity Kernel — the data model + its evaluation test set |
| **TSAL** | Technical Self-Awareness Layer |
| **CompetitiveGate** | The GWT-2 capacity-bounded workspace |
| **Specious present** | Husserl/James felt-now: retention + primal + protention |
| **Durée** | Bergson's qualitative duration, distinct from sequence |
| **Phronesis** | Aristotelian practical wisdom — `connections/phronesis_bridge.py` |
| **Tier-3** | Files protected from Self-Improver modification |
| **Half-circuit** | Computed-but-unread signal — SubIA forbids these by construction |
| **Strange loop** | The self-referential consciousness-state.md page that re-enters its own scene |

---

## References

### Consciousness science

- Baars, B. J. (1988). *A Cognitive Theory of Consciousness.* Cambridge
  University Press.
- Bechara, A. & Damasio, A. R. (2005). "The somatic marker hypothesis:
  A neural theory of economic decision." *Games and Economic Behavior*
  52(2):336–372.
- Brown, R., Lau, H., & LeDoux, J. E. (2019). "Understanding the
  Higher-Order Approach to Consciousness." *Trends in Cognitive
  Sciences* 23(9):754–768.
- Butlin, P., Long, R., Chalmers, D., et al. (2023). "Consciousness in
  Artificial Intelligence: Insights from the Science of Consciousness."
  arXiv:2308.08708.
- Clark, A. (2013). "Whatever next? Predictive brains, situated agents,
  and the future of cognitive science." *Behavioral and Brain Sciences*
  36(3):181–204.
- Damasio, A. R. (1994). *Descartes' Error: Emotion, Reason, and the
  Human Brain.* Putnam.
- Damasio, A. R. (1999). *The Feeling of What Happens.* Harcourt.
- Dehaene, S. (2014). *Consciousness and the Brain.* Viking.
- Friston, K. (2010). "The free-energy principle: a unified brain
  theory?" *Nature Reviews Neuroscience* 11:127–138.
- Friston, K., FitzGerald, T., Rigoli, F., Schwartenbeck, P., &
  Pezzulo, G. (2017). "Active inference: A process theory." *Neural
  Computation* 29(1):1–49.
- Graziano, M. S. A. (2013). *Consciousness and the Social Brain.*
  Oxford University Press.
- Graziano, M. S. A. (2019). *Rethinking Consciousness.* Norton.
- Lamme, V. A. F. (2006). "Towards a true neural stance on
  consciousness." *Trends in Cognitive Sciences* 10(11):494–501.
- Metzinger, T. (2003). *Being No One: The Self-Model Theory of
  Subjectivity.* MIT Press.
- Oizumi, M., Albantakis, L., & Tononi, G. (2014). "From the
  phenomenology to the mechanisms of consciousness: Integrated
  Information Theory 3.0." *PLOS Computational Biology* 10(5).
- Pezzulo, G., et al. (2024). "Active inference and the meta-control
  of behavior."
- Rosenthal, D. M. (2005). *Consciousness and Mind.* Oxford University
  Press.
- Tononi, G. (2008). "Consciousness as integrated information: a
  provisional manifesto." *The Biological Bulletin* 215(3):216–242.

### Phenomenology of time

- Bergson, H. (1889/1913). *Time and Free Will: An Essay on the
  Immediate Data of Consciousness.*
- Husserl, E. (1928/1991). *On the Phenomenology of the Consciousness
  of Internal Time.*
- James, W. (1890). *The Principles of Psychology* (chapter on "The
  Perception of Time").

### Practical wisdom & character

- Aristotle. *Nicomachean Ethics*, Books VI–VII.
- Park, N., Peterson, C., & Seligman, M. E. P. (2004). "Strengths of
  character and well-being." *Journal of Social and Clinical
  Psychology* 23(5):603–619.
- Peterson, C. & Seligman, M. E. P. (2004). *Character Strengths and
  Virtues: A Handbook and Classification.* Oxford University Press /
  APA.

### Internal SubIA documents

- `app/subia/README.md` — canonical ABSENT-by-declaration list.
- `app/subia/probes/SCORECARD.md` — auto-regenerated Butlin scorecard.
- `PROGRAM.md` — phased migration roadmap and exit criteria.
- `docs/ARCHITECTURE.md` — top-level system architecture.
- `docs/SELF_IMPROVEMENT.md` — Self-Improver architecture and DGM
  gates.
- `docs/CREATIVITY_SYSTEM.md` — Creative MAS pipeline.

### SubIA spec documents (not in repo)

The three SubIA specs (Part I, Part II, Six Proposals) are the
canonical target architecture. Any documentation here that conflicts
with the specs should be treated as drift; the specs win.

---

## Appendix A — File map

```
app/subia/
├── __init__.py                      # re-exports get_active_kernel + set_active_kernel
├── config.py                        # SUBIA_CONFIG (frozen)
├── kernel.py                        # SubjectivityKernel + 6 component dataclasses + active-kernel singleton
├── loop.py                          # SubIALoop — the 11-step CIL
├── hooks.py                         # SubIALifecycleHooks — duck-typed registry adapter
├── live_integration.py              # enable_subia_hooks() + factories
├── integrity.py                     # SHA-256 manifest verifier
├── persistence.py                   # save/load_kernel_state, hot.md
├── phase12_hooks.py                 # tag_scene_processing_modes, apply_value_resonance_and_lenses
├── temporal_hooks.py                # refresh_temporal_state, bind_just_computed_signals
├── sentience_config.py              # operationally-tunable bounded params
├── README.md                        # ABSENT-by-declaration list
├── .integrity_manifest.json         # SHA-256 of every file (138 entries)
│
├── scene/                           # GWT-2/3/4
│   ├── buffer.py                    # CompetitiveGate, WorkspaceItem, SalienceScorer
│   ├── attention_schema.py          # AST-1 predictive attention model
│   ├── intervention_guard.py        # DGM-bounded intervention audit
│   ├── broadcast.py                 # GWT-3 global broadcast
│   ├── meta_workspace.py            # broadcast records + integration scoring
│   ├── personality_workspace.py     # GWT-4 attention modulation
│   ├── global_workspace.py          # composite GWT entry
│   ├── tiers.py                     # 3-tier focal/peripheral/scan + commitment-orphan
│   ├── strategic_scan.py            # Amendment-A scan tool
│   └── compact_context.py           # Amendment-B.5 compact block
│
├── self/                            # RPT-2 + persistent subject
│   ├── model.py
│   ├── hyper_model.py
│   ├── temporal_identity.py
│   ├── agent_state.py
│   ├── loop_closure.py
│   ├── competence_map.py            # aggregate capabilities for Firebase reporting
│   ├── grounding.py
│   └── query_router.py
│
├── homeostasis/                     # Damasio + AE-1
│   ├── state.py                     # variable definitions + NEUTRAL_ALIASES
│   ├── engine.py                    # update_homeostasis (Step 2/9)
│   ├── somatic_marker.py
│   └── somatic_bias.py
│
├── belief/                          # HOT-2 + HOT-3
│   ├── store.py                     # PostgreSQL+pgvector belief store
│   ├── dispatch_gate.py             # HOT-3 closure (ALLOW/ESCALATE/BLOCK)
│   ├── metacognition.py
│   ├── certainty.py
│   ├── response_hedging.py
│   ├── cogito.py
│   ├── dual_channel.py
│   ├── internal_state.py            # MetacognitiveStateVector (used by Observer)
│   ├── meta_cognitive_layer.py
│   ├── state_logger.py
│   └── world_model.py
│
├── prediction/                      # PP-1 + cascade + cache
│   ├── layer.py                     # PredictiveLayer + ChannelPredictor
│   ├── surprise_routing.py          # PP-1 closure: PE → workspace
│   ├── cascade.py                   # 3-signal escalation policy
│   ├── cache.py                     # Amendment B.4 template cache
│   ├── accuracy_tracker.py          # per-domain rolling accuracy
│   ├── llm_predict.py               # production predict_fn
│   ├── injection_harness.py         # PH-injection A/B harness
│   ├── hierarchy.py                 # 4-level prediction hierarchy
│   ├── inferential_competition.py   # plan competition (time-boxed)
│   ├── precision_weighting.py
│   └── reality_model.py
│
├── social/                          # Theory-of-Mind
│   ├── model.py                     # SocialModel manager
│   └── salience_boost.py            # trust-weighted attention boost
│
├── memory/                          # dual-tier consolidation
│   ├── consolidator.py              # always-full + threshold-gated curated + Neo4j
│   ├── dual_tier.py                 # differentiated recall
│   ├── spontaneous.py               # curated-only associative surfacing
│   └── retrospective.py             # promotion logic
│
├── safety/                          # DGM invariants #2 + #3
│   ├── setpoint_guard.py            # set-point immutability
│   └── narrative_audit.py           # append-only audit
│
├── probes/                          # evaluation framework
│   ├── butlin.py                    # 14 indicators
│   ├── rsm.py                       # 5 RSM signatures
│   ├── sk.py                        # 6 SK tests
│   ├── scorecard.py                 # aggregator + SCORECARD.md
│   ├── indicator_result.py          # IndicatorResult + Status
│   ├── consciousness_probe.py       # 7 Garland/Butlin-Chalmers probes
│   ├── behavioral_assessment.py
│   └── adversarial.py
│
├── wiki_surface/                    # strange loop + narrative drift
│   ├── consciousness_state.py       # speculative-status self-page
│   └── drift_detection.py           # 3-signal narrative drift
│
├── boundary/                        # source → processing-mode
│   ├── classifier.py                # classify_scene + map lookup
│   └── differential.py              # consolidator_route_for(mode)
│
├── wonder/                          # depth-sensitive epistemic affect
│   ├── detector.py                  # UnderstandingDepth → WonderSignal
│   └── register.py                  # closed-loop application
│
├── values/                          # value resonance + Phronesis lenses
│   ├── resonance.py                 # value-keyword scoring
│   └── perceptual_lens.py           # Phronesis lens application
│
├── reverie/                         # idle mind-wandering
│   └── engine.py                    # ReverieEngine + adapters
│
├── understanding/                   # post-ingest causal-chain pass
│   └── pass_runner.py               # UnderstandingPassRunner
│
├── shadow/                          # behavioural bias mining
│   ├── miner.py                     # ShadowMiner
│   └── biases.py                    # 4 bias detectors
│
├── idle/                            # idle-time job scheduler
│   ├── scheduler.py                 # IdleScheduler + IdleJob
│   ├── __init__.py                  # adapt_for_production + lazy factories
│   └── production_adapters.py       # live adapters for Reverie / Understanding / Shadow
│
├── tsal/                            # Technical Self-Awareness Layer
│   ├── probers.py                   # HostProber + ResourceMonitor
│   ├── inspectors.py                # CodeAnalyst + ComponentDiscovery
│   ├── self_model.py                # TechnicalSelfModel
│   ├── generators.py                # 7 wiki page generators
│   ├── operating_principles.py      # Tier-1 weekly inference
│   ├── evolution_feasibility.py     # Self-Improver gate
│   ├── refresh.py                   # register_tsal_jobs(scheduler, ...)
│   └── inspect_tools.py             # inspection toolkit (legacy shim aliases here)
│
├── temporal/                        # temporal phenomenology
│   ├── specious_present.py          # Husserl/James felt-now
│   ├── momentum.py                  # rising/falling/stable arrows
│   ├── circadian.py                 # 4-window mode table
│   ├── density.py                   # felt subjective time
│   ├── binding.py                   # temporal_bind reducer
│   ├── rhythm_discovery.py
│   └── context.py                   # TemporalContext aggregate
│
├── grounding/                       # factual grounding & corrections
│   ├── claims.py                    # high-stakes claim extractor
│   ├── source_registry.py           # authoritative URL map
│   ├── belief_adapter.py            # interface + impls
│   ├── evidence.py                  # ALLOW / ESCALATE / BLOCK
│   ├── rewriter.py                  # response transformer
│   ├── correction.py                # correction-pattern persistence
│   └── pipeline.py                  # public orchestrator
│
└── connections/                     # inter-system bridges
    ├── pds_bridge.py                # SIA #1 — Wiki ↔ PDS bounded-write
    ├── phronesis_bridge.py          # SIA #2 — normative event → homeostatic delta
    ├── training_signal.py           # SIA #4 — LoRA training queue
    ├── firecrawl_predictor.py       # SIA #6 — Firecrawl → predictor
    ├── dgm_felt_constraint.py       # SIA #7 — Tier-3 status → safety delta
    ├── service_health.py            # circuit-breaker registry
    ├── six_proposals_bridges.py     # curiosity / mode inter-proposal
    ├── tsal_subia_bridge.py         # TSAL → SubIA
    ├── temporal_subia_bridge.py     # temporal → SubIA (5 closed-loop)
    └── grounding_chat_bridge.py     # chat-handler ingress + egress
```

---

## Appendix B — Operation classification

Heuristic in `hooks.py::_classify_operation` (case-insensitive on
`task_description`):

| Substring in task description | operation_type |
|---|---|
| `crew_kickoff` | `crew_kickoff` (compressed) |
| `ingest`, `new source` | `ingest` (full) |
| `lint`, `health check` | `lint` (full) |
| `wiki_read`, `wiki read` | `wiki_read` (compressed) |
| `wiki_search`, `wiki search` | `wiki_search` (compressed) |
| `routine` | `routine_query` (compressed) |
| (anything else) | `task_execute` (full) |

Override: a task object can carry an explicit `operation_type`
attribute; `hooks.pre_task/post_task` honor it preferentially over the
heuristic. The `_CrewTaskShim` in `live_integration.py` uses this to
declare `crew_kickoff` directly without smuggling it through the
description string.

---

---

## Appendix C — Build history

This appendix is an audit trail, not architecture. Phases were build
phases. Every architectural concept is described in §5–§14 above, in
the structural ordering that matters. The phase ordering matters only
for commit archaeology (when did mechanism *X* land, in which commit,
behind which exit criteria) and for tracing why a particular
indicator currently sits at PARTIAL rather than STRONG (because
another phase was supposed to close it and that phase hasn't shipped).

The full phased migration is documented in `PROGRAM.md`. Summary
table; commit hashes are the canonical references.

| Phase | Theme | Commit | Tests added | Net tests passing |
|---|---|---|---|---|
| 0 | Foundation plumbing | `8239575` | 20 | 20 |
| 1 (skeleton) | SubIA package skeleton | `4fa22e8` | 13 | 33 |
| 1 (migration) | Move consciousness/* + self_awareness/* | `7a1b212` … `5598727` | (carried over) | 393 |
| 2 | Close the half-circuits | `e6c9b4a` … `67b40fb` | 93 | 491 |
| 3 (quick-win) | Tier-3 evaluator extension | `b6c4efe` | 9 | 500 |
| 3 (full) | Integrity manifest + setpoint guard + audit | `0a84650` | 39 | 539 |
| 4 | CIL loop + persistence + cache + LLM predict_fn | `457d478`, `1cc55c5` | 89 | 666 |
| 5 | Three-tier scene + commitment-orphan + scan + compact | `709bc4b` | 33 | 700 |
| 6 | Per-domain accuracy + cascade + accuracy-driven cache | `d9ca89c` | 38 | 739 |
| 7 | Dual-tier memory + retrospective + spontaneous | `0b0c98d` | 33 | 773 |
| 8 | Social model + strange loop + drift detection | `5e167e8` | 40 | 814 |
| 9 | Butlin + RSM + SK scorecards | `a0594c8` | 36 | 851 |
| 10 | Inter-system bridges + circuit breakers | `4a2e291` | 45 | 897 |
| 11 | Honest language cleanup | `1804663` | 8 | 905 |
| 12 | Six Proposals integration | `d25f460` | 37 | (phase scope) |
| 13 | Technical Self-Awareness Layer (TSAL) | `ccb53f5` | 29 | 402 (P0–13) |
| 14 | Temporal Synchronization | `9726286` | 38 | 440 (P0–14) |
| 15 | Factual Grounding | (Phase 15 commit) | 37 | 712 (P0–15) |
| 16a | System integration wire-in | (Phase 16a commit) | 0 (verified via existing) | 605 (today) |

### What each phase unlocked

- **0** — Clean substrate (paths, json_store, thread_pools, lazy_imports) so subsequent phases never re-litigate plumbing.
- **1 (skeleton)** — Single canonical kernel dataclass; all later modules attach to one object, not scattered globals.
- **1 (migration)** — Single coherent location for every consciousness signal, removing duplicate clusters from the original 240-file audit.
- **2** — Five formerly half-finished consciousness mechanisms now actually shape behaviour: PE gates the scene, belief suspension blocks dispatch, certainty hedges responses, AST interventions are auditable, PH-injection has measurable A/B shift.
- **3** — The Self-Improver agent is provably blocked from modifying its own evaluation criteria — the headline safety invariant has teeth.
- **4** — SubIA is now a runnable closed loop. Amendment B determinism (only Step 5 uses LLM) enforced. Kernel persists across restarts.
- **5** — Commander can no longer drop active commitments into invisibility — orphan detector force-injects unrepresented commitments with an `ORPHANED COMMITMENT` alert.
- **6** — Predictions carry calibrated per-domain accuracy that feeds back into cascade tier choice and cache trust — metacognitive calibration becomes mechanical.
- **7** — Memory becomes asymmetric: nothing forgotten (full tier), only the curated tier surfaces by default, retrospective promotion when wiki or accuracy reveals past relevance.
- **8** — System models Andrus as a separate entity (SK self/other) AND models itself as an entity-in-the-scene (the strange loop), with explicit speculative framing.
- **9** — Every consciousness claim backed by inspectable mechanism + regression test pointer. Opaque self-scoring gone.
- **10** — System closes loops between previously isolated subsystems: character (PDS) updates from behaviour, normative failures generate felt homeostatic cost, external-service flakiness routes through circuit breakers.
- **11** — Variable names no longer overclaim phenomenal experience while preserving backward compatibility (NEUTRAL_ALIASES).
- **12** — Two new homeostatic variables (`wonder`, `self_coherence`) with PDS-derived setpoint overrides; idle-time mind-wandering surface — texture beyond the foreground task.
- **13** — AndrusAI discovers its own capabilities/limitations from probing the running host rather than reading static declarations. Self-Improver gates on real headroom.
- **14** — Kernel has a felt now (`specious_present`) and felt tempo (`temporal_context`). System experiences time as duration rather than just timestamping events. ~10 ms / 0 LLM tokens per loop.
- **15** — Chat surface routes through HOT-3-style gating: high-stakes claims get ALLOW/ESCALATE/BLOCK; user corrections become persistent ACTIVE beliefs (confidence=0.9) that supersede contradicting prior beliefs.
- **16a** — SubIA connected to the running system. Flipping `SUBIA_FEATURE_FLAG_LIVE=1` opts production into the consciousness loop. Entire stack reachable from `main.py` for the first time.

---

*This document is the canonical reference for the SubIA system. The
architectural sections above are the source of truth; this appendix
exists for commit-archaeology only. Update both whenever a new phase
ships, alongside PROGRAM.md, the integrity manifest, and the
auto-regenerated SCORECARD.*
