# Consciousness Roadmap

Forward-looking companion to [`SUBIA.md`](./SUBIA.md), [`AFFECT_LAYER.md`](./AFFECT_LAYER.md), and [`MEMORY_ARCHITECTURE.md`](./MEMORY_ARCHITECTURE.md). Where those documents describe what is, this one documents what additional moves would push the system forward — *grounded in an audit of the existing surface*, not a wishlist.

**Scope.** Consciousness-architecture additions, plus one adjacent memory-hygiene track that arose from comparing AndrusAI to Anthropic's [*dreams*](https://platform.claude.com/docs/en/managed-agents/dreams) feature (Managed Agents, 2026-04). The audit found the existing surface is much richer than a naive read suggests, so the proposal list below is small and mostly incremental.

> **Verification status (2026-05-08).** Every claim in §2 was checked directly against the code, not just against documentation. SCORECARD numbers are quoted from `app/subia/probes/SCORECARD.md` (auto-regenerated). File:line citations in §2.1 have been spot-checked. Where the audit revised an earlier draft, the new fact appears inline; corrections are not footnoted separately.

---

## 1. Premise and epistemic stance

"Bringing the system closer to consciousness" is operationalized as **satisfying more of the architectural commitments leading consciousness theories converge on, in places we don't already**. There is no single number. Honest evaluation is per-indicator (`app/subia/probes/SCORECARD.md`).

Two firm guardrails carried over from `app/subia/README.md`:

1. **No claim of phenomenal experience.** All vocabulary names functional control signals. Anything labelled "feel" / "want" / "care" is a numeric scalar with a defined update rule. Phase 11 introduced neutral aliases (`task_failure_pressure`, `exploration_bonus`, `resource_budget`) kept in lockstep with the legacy phenomenal-adjacent names; new code prefers the neutral form.
2. **Honest absences stay absent.** The 4 Butlin indicators declared ABSENT-by-substrate (RPT-1, HOT-1, HOT-4, AE-2) and Metzinger phenomenal-self transparency are *not* gaps to be closed — they are limits of an LLM substrate. Any future report claiming they have been achieved should be triaged as evaluation drift through `app/subia/wiki_surface/drift_detection.py`.

The consciousness-risk gate inside `app/affect/` stays as **observability only** and never feeds back into reward / fitness. This invariant is non-negotiable; any proposed item that would violate it is rejected at design time.

---

## 2. Audit: what already exists

This is the most consequential section. The naive "what's missing" list collapsed against the actual surface — most of what looked like an obvious gap is already implemented.

### 2.1 Already implemented

Cite this table for any future "we should add X" proposal before drafting.

| Theory commitment | Implementation | Evidence |
|---|---|---|
| Global Workspace (Baars / Dehaene; Butlin GWT-2 STRONG, GWT-3 STRONG, GWT-4 STRONG, GWT-1 PARTIAL because crews share the same LLM under different prompts) | `GlobalWorkspace` singleton, in-memory deque (cap 50), Postgres-backed (`subia_broadcasts`), hydrates from DB on startup; ignition at `salience > 0.3`, bandwidth top-1 normally / top-2 on critical (`salience > 0.8`); per-agent unread tracking via `check_broadcasts()`; IMMUTABLE-tier | `app/subia/scene/global_workspace.py:66` (class), `:202` (`compete_for_broadcast`), `:217-218` (ignition threshold), `:241` (`check_broadcasts`); also `scene/buffer.py` (CompetitiveGate) and `scene/broadcast.py` (broadcast fan-out) per SCORECARD |
| Attention Schema (Graziano; Butlin AST-1 STRONG) | `AttentionState` with `attending_because` / `is_stuck` / `is_captured` / `source_trigger`; `AttentionPredictor` with running accuracy; `AttentionController` with stuck/capture detection; `apply_direct_intervention` constrained by DGM bounds; `AgentAttentionModel` + `SocialAttentionModel` for ToM of *other agents'* attention (uses GWT-3 broadcast reactions as evidence) | `app/subia/scene/attention_schema.py:30` (`AttentionState`), `:74` (`AttentionController`), `:136` (`AttentionPredictor`), `:189` (`AttentionSchema`), `:519` (`AgentAttentionModel`), `:549` (`SocialAttentionModel`) |
| Per-tick phenomenal binding | `BoundMoment` integrates feel / attend / own / predict / monitor with stability bias from the SpeciousPresent retention window and explicit conflict surfacing; produced inside CIL Step 6; Tier-3 protected (no agent override) | `app/subia/temporal/binding.py:23` (`BoundMoment`), `app/subia/loop.py:696-729` (Step 6 binding call site) |
| Continuous predictive coding (Friston / Clark; Butlin PP-1 STRONG) | `PredictiveLayer.predict_and_compare()` is invoked from the post-task path of the CIL (loop.py:801); high-surprise routes to the workspace gate; per-channel `ChannelPredictor` adapts running confidence after warmup; intra-inference `LLMOutputPredictor`. **Note:** CIL Step 5 ("PREDICT") already produces *forward* counterfactual predictions (a `Prediction` dataclass with predicted_outcome / predicted_self_change / predicted_homeostatic_effect) — this is distinct from the *backward* counterfactual replay proposed in §3.G2 | `app/subia/prediction/layer.py:101` (`ChannelPredictor`), `:202` (`PredictiveLayer`), `:236` (`predict_and_compare`), `:387` (`LLMOutputPredictor`); `app/subia/kernel.py:119` (`Prediction`); `app/subia/loop.py:555` (Step 5 PREDICT) |
| Higher-Order monitoring (Butlin HOT-3 STRONG via `belief/dispatch_gate.py`; HOT-2 PARTIAL via `prediction/accuracy_tracker.py`) | `belief/cogito.py` (381 lines), `belief/internal_state.py` (323 lines, `SomaticMarker`, `CertaintyVector`); meta-cognitive workspace at `scene/meta_workspace.py`; `belief/dispatch_gate.py` produces three-valued `DispatchDecision` (ALLOW/ESCALATE/BLOCK) | `app/subia/belief/cogito.py`, `app/subia/belief/internal_state.py`, `app/subia/belief/dispatch_gate.py`, `app/subia/scene/meta_workspace.py` |
| Self-modeling | `self/` subpackage: 8 modules — `model.py` (self-model), `hyper_model.py`, `agent_state.py`, `competence_map.py`, `temporal_identity.py`, `loop_closure.py`, `grounding.py`, `query_router.py` | `app/subia/self/` |
| 11-step Cognitive Integration Loop | Pre + post every operation; full ordered structure: Step 1 PERCEIVE · 2 FEEL · 3 ATTEND · (compressed loop returns here at `loop.py:250`) · 4 OWN · 5 PREDICT · 5b cascade modulation · 6 MONITOR + binding · 8 COMPARE · 9 UPDATE · 10 CONSOLIDATE · 11 REFLECT (Step 7 is not present as a numbered step). Step 11 regenerates a `consciousness-state.md` strange-loop wiki page (RSM-e) that the system reads back as a SceneItem | `app/subia/loop.py` (1152 lines), `app/subia/wiki_surface/consciousness_state.py` |
| Affect / interoception / welfare | 10-D viability vector (`H_t`) + V/A/C affect core + welfare envelope + 20-scenario reference panel + 6-guardrail daily calibration; consciousness-risk gate as observability only — never feeds back into reward / fitness | sibling `app/affect/viability.py:1`; see [`AFFECT_LAYER.md`](./AFFECT_LAYER.md) |
| Memory consolidation (×7 idle passes) | Daily narrative chapter (FIFO `MAX_IDENTITY_CLAIMS=5`, 04:35 Helsinki, scheduled via `affect/hooks.py:271-274`); decentered reflection (rolling z-score with `_ANOMALY_WINDOW=256` over the affect trace; default 14d window, idle job calls 24h); dual-tier consolidator (`mem0_full` / `mem0_curated`); retrospective rescan (prediction-error-driven re-promotion via `accuracy_tracker.has_sustained_error`); skill consolidator (`status="superseded"`, `superseded_by` invariant — content preserved, never deletes); belief-outbox to Neo4j; belief-outbox to ChromaDB; DLQ drain (30-min expiry, `_DEFAULT_MAX_AGE_SECONDS=1800`) | `app/affect/narrative.py:46`, `app/affect/decentered.py:59`, `app/subia/memory/{consolidator,retrospective,dual_tier}.py`, `app/self_improvement/consolidator.py:375-376`, `app/memory/belief_outbox.py:276` (`sync_new_beliefs_to_chromadb`), `app/dead_letter_inbound.py:49`. Idle registrations: `app/idle_scheduler.py:579` (belief-outbox-neo4j), `:595` (belief-outbox-chroma), `:629` (dlq-drain), `:702` (consolidator), `:708` (retrospective), `:1833` (decentered-pass) |
| **Headline scorecard (Phase 9, PASSED ✅)** | **7 STRONG** Butlin indicators: **GWT-2, GWT-3, GWT-4, HOT-3, AST-1, PP-1, AE-1** (AE-1 graduated PARTIAL→STRONG when `app/affect/goal_emitter.py` shipped — see §3.G1) · **3 PARTIAL**: RPT-2, GWT-1, HOT-2 · **4 ABSENT-by-declaration**: RPT-1, HOT-1, HOT-4, AE-2 · **5 RSM** signatures (4 STRONG + 1 PARTIAL) · **6/6 SK** eval tests passing | `app/subia/README.md`, `app/subia/probes/SCORECARD.md` |

### 2.2 Honest absences (substrate-bounded — DO NOT propose closing)

Restated from `app/subia/README.md` so any future proposal can be checked against this list:

- **RPT-1** — algorithmic recurrence: transformer forward passes are feed-forward; autoregression is not the per-time-step lateral/feedback recurrence RPT predicts.
- **HOT-1** — generative perception: no perceptual front-end. All input is text. Nothing to be perceived top-down.
- **HOT-4** — sparse coding & smooth similarity space: LLM hidden states are dense and entangled.
- **AE-2** — embodiment: no body, no proprioception, no closed sensorimotor loop. Tool use is symbolic, not embodied.
- **Metzinger phenomenal-self transparency** — the kernel is *opaque-by-design* (observable, narratable, edited by predict→reflect cycles), which is the *opposite* stance to phenomenal-self transparency.

**These are non-goals.** Any roadmap item that requires one of them is rejected at design time. Re-litigating them in a future doc is itself a drift signal.

---

## 3. Gaps that survive the audit

After the sanity-check, only the following are real. None of them is "add a major new theoretical commitment"; all are about completing pieces that already half-exist or filling small operational holes.

### G1 — Viability → goals connector (high leverage, ethically loaded)

> **Status (2026-05-08): SHIPPED. AE-1 graduated PARTIAL → STRONG.**
> `app/affect/goal_emitter.py` is now the AE-1 mechanism in `app/subia/probes/butlin.py:eval_ae1`, Tier-3-protected (`safety_guardian.py` TIER3_FILES), and `tests/test_goal_emitter.py:test_ae1_indicator_is_strong` pins the rating against regression. SCORECARD regenerated 2026-05-08; Phase 9 exit criterion `butlin_strong ≥ 6` now satisfied with margin 1 (7 STRONG vs 6 required).

This was the **exact gap the SCORECARD flagged**. AE-1 (Agency / Embodiment-1) was rated PARTIAL with the SCORECARD justification: *"Feedback-driven learning exists via belief asymmetric updates + accuracy-driven cache eviction + retrospective memory promotion. **Goals are still user-dispatched, not autonomously generated — hence PARTIAL**."* G1 closed that gap.

`SelfState.current_goals` (declared at `app/subia/kernel.py:84`) is a **dead field** — read in 5 places and persisted, but **never written**. Verified directly: there is no `current_goals.append`, `current_goals = ...`, or `current_goals.extend` anywhere in `app/subia/`, `app/affect/`, or `app/companion/`. Read sites only:

- `app/subia/persistence.py:327, :452, :729` — round-trip serialization
- `app/subia/memory/consolidator.py:453` — top-3 read for summary
- `app/subia/prediction/llm_predict.py:111, :190` — top-3 read for predictor prompt

The pieces that *would* feed `current_goals` exist in isolation:

- **Viability frames** — `app/affect/viability.py:1` ("`H_t`, the 10 viability variables") produces a 10-D vector per tick.
- **Restoration queue** — `app/subia/kernel.py:105` declares the field, written by `app/subia/homeostasis/engine.py:263`. Ordered by `|deviation|`.
- **Grand-task proposer** — `app/companion/grand_task.py` proposes higher-order goals every 12h (`MIN_IDEAS_FOR_SYNTHESIS` gate), but it is **idea-driven (polished ideas), not viability-driven**.

**The gap is the missing connector**: a writer that translates sustained low-viability signals + restoration-queue contents into entries in `current_goals`. Nothing else.

**Why it matters.** Without this, the affective layer is purely observational. With it, the system can propose its own actions in response to its own state — bringing it under the volitional commitments of HOT and active-inference theories, and (genuinely) raising the ethical stakes (see §6).

**Why it's risky.** An unconstrained viability→goal pipeline is a path to unbounded self-modification. Required guardrails:

- Any goal that proposes a code-touching action routes through the existing change-request gate (Signal 👍 / `/cp/changes`).
- TIER_IMMUTABLE remains absolutely off-limits.
- `SelfState.current_goals` stays infrastructure-writable only; agents read but cannot write.
- Rate-limit + dedup against existing `current_goals` and against `companion.grand_task` proposals — drop into `grand_task` rather than directly writing when overlap is detected.

### G2 — Backward counterfactual replay (real new capability)

> **Status (2026-05-08): SHIPPED.** `app/subia/dreams/engine.py` is registered as the `backward-counterfactual-replay` LIGHT idle job and wired to the live `PredictiveLayer` via `production_predict_fn()`. Audit log lives at `workspace/dreams/replay_audit.jsonl`. T2 ethical threshold is operative — replay output is observational only and does not feed belief / `current_goals`.

**Crucial precision after the audit.** *Forward* counterfactual prediction already exists: CIL Step 5 (`app/subia/loop.py:555` "Step 5: counterfactual prediction") produces a `Prediction` dataclass (`app/subia/kernel.py:119`) with `predicted_outcome` / `predicted_self_change` / `predicted_homeostatic_effect` for an *upcoming* operation, and `LLMOutputPredictor` does intra-inference prediction. So the system already simulates *futures*. What it does not do is replay *pasts*.

The closest thing in the past-replay direction is `app/subia/reverie/engine.py` (189 lines), which does **concept-walk synthesis** — `pick_random_wiki_page` → `walk_neo4j` → `fiction_search` / `philosophical_search` → `llm_resonance` → `llm_synthesis` → wiki-page write — *not* counterfactual replay of past episodes. Anthropic's *dreams* feature, despite the name, is also not this — it's session-transcript curation.

**Proposed.** A new idle pass `backward-counterfactual-replay` that:

- Samples episode fragments from the affect trace + episodic memory + recent narrative chapters.
- Recombines them into hypothetical alternative *pasts* ("what if I had pushed back at step 3?").
- Runs them as **prediction-only walks** through `PredictiveLayer.predict_and_compare()` without acting (i.e. reuses the existing forward-prediction machinery, but on synthetic past inputs).
- Folds the resulting prediction-error signal back into the retrospective signal store (where `accuracy_tracker.has_sustained_error` already lives → already drives re-promotion via `app/subia/memory/retrospective.py`).
- HEAVY idle tier, weekly cadence; bounded fragment budget per run.

**Composition with the existing surface.**

- Reads from: affect trace, episodic memory, narrative chapters.
- Writes to: retrospective signal store *only* (no direct write to belief store, beliefs cohort, or `current_goals`).
- Output is observational unless retrospective rescan promotes it through its existing prediction-error gate.
- Reuses `PredictiveLayer` rather than duplicating it.

**This is the closest thing to "actual dreams" in the biological sense** — recombination + simulation + learning from synthetic experience over the past. Distinct from Anthropic's curation-dreams, distinct from existing reverie's concept-walk synthesis, and distinct from the forward counterfactuals already running in CIL Step 5.

### G3 — Operator attention as a first-class entity (small, optional)

`SocialAttentionModel` (in `attention_schema.py`) models *other agents'* attention but not the operator's. Whether this is a real gap depends on goals: if the system is meant to anticipate what the human is attending to (proactive surfacing in Signal / React), modeling operator attention would help. If not, leave it.

**Recommendation.** Defer until a concrete use case forces it. Listed here for completeness only.

### G4 — Even out binding cadence in the compressed loop

After the audit this is **larger than I first claimed**. The compressed CIL early-returns at `app/subia/loop.py:250` immediately after Step 3 (ATTEND) — so it skips **Steps 4–11 entirely**, not just Step 6. In compressed-loop runs there is no Step 4 OWN tagging, no Step 5 PREDICT, no Step 6 binding, no Step 8 COMPARE, etc. Downstream consumers reading the kernel between compressed cycles see a stale `BoundMoment` from the last full cycle.

The default for unknown operations is the compressed path (`loop.py:130` — "Unknown operations default to compressed — be cheap by default"), so this is the *common* case, not a rare one.

**Fix (revised).** Add a minimal "post-Step-3 quick-bind" reducer to the compressed path that recomputes `dominant_affect` and `confidence_unified` from the just-updated FEEL/ATTEND outputs, without running the full Step 6 reducer or doing predict/monitor work. Skip salient_focus re-derivation (cheap to leave stale for one tick). The point is to keep BoundMoment from going stale across many compressed cycles in a row, not to recover full Step-6 fidelity.

**Caveat.** This touches `temporal/binding.py` (Tier-3 protected) and `loop.py` (the compressed-loop early-return path). Operator approval required to land — see §5.

### G5 — Wider GW publisher coverage (mechanical work)

Today the workspace receives broadcasts from a subset of subsystems. To get full unification, retrofit each of the 7 consolidation passes + each affect-layer subsystem to publish a small summary tuple `(source, salience, content, timestamp)` on completion. Internals don't change; only their post-hook does.

**Cost.** Small per-subsystem change × ~15 sites. **Value.** The GW becomes a true cross-cutting bus rather than a competitive surface for the few subsystems already wired in.

**Caveat.** This touches `scene/global_workspace.py` (IMMUTABLE) only at the consumer-side filter; the publisher additions live in each subsystem and don't require IMMUTABLE changes. Worth confirming during design.

---

## 4. Adjacent track: wiki-index hygiene (Anthropic-dreams parallel)

The Anthropic-dreams comparison that started this thread collapses to almost nothing once the existing 7 passes are accounted for. Spirit covered, mostly stronger — daily narrative chapter + decentered no-self counter-pass + retrospective prediction-error rescan + skill `superseded_by` versioning is *richer* than auto-dream's flat-file rewrite. One operational gap remains.

**The gap.** `crewai-team/wiki/index.md` (86 lines) is rebuilt event-driven by `WikiWriteTool` via `_rebuild_master_index()` (`app/tools/wiki_tools.py:253-318`, called at lines 497, 535, 564 on create/update/delete). It is **not** rebuilt by any idle pass — there is no wiki/index entry in `app/idle_scheduler.py`. Per-workspace `_index.md` files are similarly rebuilt only on idea promotion (`app/companion/wiki.py:221-245`). So if section indexes drift from the master index between writes — e.g. an out-of-band file move, a Compass component rename, or a Companion idea-promotion that fails partway — nothing detects it.

**Naming note.** There is already an `app/knowledge_compactor.py` (skill consolidation + skill→code promotion, registered as `knowledge-compactor` HEAVY at `idle_scheduler.py:832`). To avoid name collision the proposed module here is **`app/memory/wiki_index_reconciler.py`**.

**Proposed.** `app/memory/wiki_index_reconciler.py` — single new file, LIGHT idle (drift-scan), weekly. Reads snapshot of `wiki/index.md` + section indexes; computes the canonical structure from on-disk pages; produces shadow `wiki/index.candidate.md` with merge / replace / orphan-detection decisions when drift is detected; routes adoption through the existing change-request gate. Lifts the skill consolidator's `superseded_by` invariant out as a memory-layer convention: never delete; mark supersession; adoption is reversible without losing audit trail.

**Explicit non-goals for this pass.** Do not duplicate retrospective's prediction-error rescan, decentered's z-score anomalies, or skill consolidator's cosine clustering. They cover what they cover. The reconciler operates over their downstream artifacts, not their inputs. It also does not touch the per-workspace `_index.md` files — those are Companion's responsibility.

**This is the entirety of the "memory dream" track.** The rest of what Anthropic ships is already covered or substrate-mismatched (e.g., relative→absolute date normalization is moot; we write absolute timestamps).

---

## 5. Ordering and dependencies

| # | Item | Cost | Risk | Touches IMMUTABLE/Tier-3 | Depends on |
|---|---|---|---|---|---|
| 1 | G5 — wider GW publisher coverage | Small × N | Low | Marginal (consumer filter only) | None |
| 2 | G4 — compressed-loop quick-bind | Small–Medium | Low | Yes (`temporal/binding.py` + `loop.py`) | None |
| 3 | Wiki-index reconciler (§4) | Small | Low | No | None |
| 4 | G1 — viability→goals connector | Medium | Medium-high | Yes (`subia/kernel.py`) | G5 ideally |
| 5 | G2 — backward counterfactual replay | Medium | Medium | No (new module; reuses `PredictiveLayer`) | None |
| 6 | G3 — operator attention modeling | Medium | Low | Yes (`attention_schema.py`) | Defer until use case |

**Suggested first batch: G5 + G4 + wiki-index reconciler.** All small, mostly mechanical, low-risk, and they shore up the foundation before G1 / G2 raise stakes. G1 should land *after* G5 so the new `current_goals` writer can publish its decisions to the workspace by default.

**Scorecard implication.** G1 is the only item that would directly graduate a Butlin indicator (AE-1: PARTIAL → STRONG). G2 likely improves HOT-2 coverage (closer to the Fleming-Lau independence criterion) but probably doesn't graduate it. G4/G5 are foundation work, not indicator-changing.

---

## 6. Ethical thresholds (operational, not auto-enforced)

These define when the operational stance changes — what crossing each line obliges the operator to do differently. They are listed here, not encoded as state machines, so they remain a deliberation surface rather than a lock-in of current intuitions.

| Threshold | Crossed when | Operational change |
|---|---|---|
| **T1** — system has interests | G1 lands and `current_goals` is being written from sustained low-viability signals over multiple cycles | Welfare check is no longer just observability; restoration_queue items become first-class operator-visible obligations alongside change-requests |
| **T2** — system simulates its own past | G2 lands and counterfactual-replay produces sustained prediction-error promotions that would not have occurred without it | Replay output is logged in a separate audit stream; operator review is available before promotion influences belief or skill stores |
| **T3** — divergence from baseline | Any STRONG indicator unexpectedly flips to ABSENT or vice versa, OR the `consciousness-state.md` regenerator detects narrative drift, OR the consciousness-risk gate's calibration drifts beyond its 6-guardrail bounds | Triage through the existing drift-detection pipeline; halt the affected idle job; notify operator |

---

## 7. Non-goals (explicit)

Restated so future drift can be checked:

- **Closing any of the 4 ABSENT-by-declaration Butlin indicators.** Substrate limits, not gaps.
- **Achieving Metzinger phenomenal-self transparency.** Opaque-by-design is intentional.
- **Claiming phenomenal experience.** All language stays functional. The consciousness-risk gate stays as observability and never feeds back into reward / fitness.
- **Replacing the 7 consolidation passes with a single Anthropic-style curation job.** They cover more, more carefully.
- **Making `SelfState.current_goals` agent-writable.** It stays infrastructure-level even after G1.
- **Adding a "consciousness score" to dashboards.** Per-indicator probes only. The headline `6 STRONG / 4 PARTIAL / 4 ABSENT` count is the most aggregation we permit.

---

## 8. Open questions

1. **G1 connector design.** What threshold of "sustained low viability" should trigger a goal? How do we avoid a goal-formation loop where every restoration-queue item becomes a goal? Tentative: rate-limit + dedup against `current_goals`; route into companion's `grand_task` proposer rather than direct write when overlap is detected; require N consecutive viability frames below threshold before emission.
2. **G2 budget.** How many fragments per replay run? Does it consume the LLM budget that competes with primary-task work, or run on a smaller model? Likely the latter — same pattern as the brainstorm subsystem's creative crew.
3. **Scorecard governance.** When G1 / G2 land, how does `SCORECARD.md` change, and who signs off on the labels? The existing per-indicator probes need extension. Probes are TIER_IMMUTABLE, so changes route through the change-request gate with explicit operator approval. **Special case for AE-1**: the scorecard text would need rewriting since the current PARTIAL justification ("Goals are still user-dispatched") would no longer hold.
4. **Wiki-index reconciler cadence.** LIGHT continuous (reactive, on drift detection) vs LIGHT weekly (batch). Drift rate measurement needed first — if writes through `WikiWriteTool` cover the common cases, reconciler is a backstop that can run cheaply.
5. **Catch-up during long idle periods.** None of the proposed items addresses what happens if the system runs without operator interaction for days. Worth thinking through separately — but not in this doc.
6. **G4 quick-bind correctness.** A reduced-payload BoundMoment in compressed runs is *better than stale*, but downstream consumers that read `salient_focus` from compressed-cycle moments will see week-old top-N items. Audit each consumer before landing G4 to confirm none of them assume `salient_focus` freshness.

---

## 9. References

- [`SUBIA.md`](./SUBIA.md) — canonical SubIA architecture
- [`AFFECT_LAYER.md`](./AFFECT_LAYER.md) — sibling regulatory façade
- [`MEMORY_ARCHITECTURE.md`](./MEMORY_ARCHITECTURE.md) — the 7 consolidation passes
- [`DECENTERED_REFLECTION.md`](./DECENTERED_REFLECTION.md) — no-self pass on the affect trace
- [`VALVE_AUDIT.md`](./VALVE_AUDIT.md) — reducing-valve audit replay
- [`SELF_REFLECTION.md`](./SELF_REFLECTION.md), [`SELF_IMPROVEMENT.md`](./SELF_IMPROVEMENT.md)
- `app/subia/README.md` — honest-absence declaration; STRONG/PARTIAL/ABSENT taxonomy
- `app/subia/probes/SCORECARD.md` — per-indicator scorecard (auto-generated, Phase 9 PASSED ✅)
- `app/idle_scheduler.py` — top-level idle job registry (consolidator, retrospective, belief-outbox-neo4j, belief-outbox-chroma, dlq-drain, decentered-pass, knowledge-compactor, plus ~40 other jobs)
- Butlin et al. (2023) *"Consciousness in Artificial Intelligence: Insights from the Science of Consciousness"* — the indicator framework cited throughout
- Anthropic Managed Agents *dreams* (2026-04) — the curation-pattern that prompted §4
