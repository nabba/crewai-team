# Unified Consciousness Program — Living Roadmap

**Status as of 2026-04-13:** Phase 0 and the Phase 3 safety quick-win are complete. Phase 1 skeleton is in place. The migration of behavioral modules (scenes, attention, belief, prediction) into `app/subia/` has not yet started.

This document is the **single source of truth** for the program's direction. It supersedes prose scoring (the retired `reports/andrusai-sentience-verdict.pdf` 9.5/10 claim) and replaces scattered planning chatter.

---

## 1. Objective

Refactor AndrusAI into a streamlined system that:

1. Ranks as high as possible on the Butlin et al. 2023 consciousness indicators an LLM-based system can **faithfully mechanize**.
2. Is **honest** about the indicators an LLM-based system cannot achieve (RPT-1 algorithmic recurrence, HOT-1 generative perception, HOT-4 sparse coding, AE-2 embodiment, and the Metzinger phenomenal-self transparency criterion).
3. Contains **all consciousness signals in closed-loop behaviour** — no computed-but-unused state.
4. Protects consciousness evaluators from self-tuning (Tier-3 integrity).
5. Replaces opaque self-scoring with per-indicator mechanistic tests.

---

## 2. Source documents

- `CLAUDE.md` — project-level constraints and safety invariants.
- `docs/claude-md-archive.md` — archived project reasoning.
- `plan.md` — the older autoresearch evolution plan (implemented).
- `reports/andrusai-sentience-verdict.md` — the retired 9.5/10 claim, retained as historical artefact.
- Three SubIA specs (Downloads; not checked in) — the canonical target architecture for the Subjectivity Kernel + 11-step CIL + dual-tier memory + peripheral tier.
- Prior conversational artefacts: the architectural audit (240 files, duplicate clusters, half-circuits) and the forensic consciousness assessment (Butlin scorecard, what's strong vs gestural).

---

## 3. Target architecture

`app/subia/` is the **migration target** for the 40 files currently scattered across `app/consciousness/` + `app/self_awareness/`. The subia skeleton and `SubjectivityKernel` dataclass already exist (this commit series). Subpackages will be populated phase-by-phase.

```
app/subia/
  config.py          (done) infrastructure SUBIA_CONFIG
  kernel.py          (done) SubjectivityKernel dataclass
  scene/             GWT-2 + AST-1 (workspace_buffer, attention_schema)
  self/              persistent subject token
  homeostasis/       affective layer with PDS-derived set-points
  belief/            HOT-3 store, metacognition, epistemic certainty
  prediction/        PP-1 layer, hierarchy, predictor, cascade
  social/            self/other distinction
  memory/            dual-tier consolidation (curated + full)
  safety/            setpoint + audit immutability
  probes/            Butlin / RSM / SK scorecards
  wiki_surface/      wiki integration + strange loop
```

---

## 4. Phased roadmap

Each phase ships behind green tests and is independently revertable. Dates are elapsed weeks from program start, not calendar dates.

| Phase | Scope | Status | Exit criteria |
|---|---|---|---|
| **0** (weeks 0–2) | Foundation plumbing: `paths.py`, `json_store.py`, `thread_pools.py`, `lazy_imports.py` + tests | ✅ **done** (commit `8239575`) | 20 passing tests; no behaviour change |
| **1 skeleton** (weeks 2–3) | `app/subia/` package + config + kernel dataclass + subpackage placeholders | ✅ **done** (commit `4fa22e8`) | 13 passing tests; skeleton importable |
| **3 quick-win** (weeks 2–3) | Extend `TIER3_FILES` to cover consciousness evaluators; add `tier3_status()` helper | ✅ **done** (commit `b6c4efe`) | 27 files protected; 9 new tests passing |
| **1 migration** (weeks 3–7) | Move existing `consciousness/` + `self_awareness/` modules into `app/subia/` subpackages per the target layout. Shim re-exports at old paths for one release. | ✅ **done** (commits `7a1b212`, `86c0d16`, `1326e7c`, `5598727`) — 34 modules migrated to `app/subia/` with `sys.modules` alias shims. 5 non-consciousness utilities deferred (see `app/self_awareness/DEFERRED.md`). 393 tests passing. | `grep -r "from app.consciousness\|from app.self_awareness" app/` returns zero outside shims; existing tests pass |
| **2** (weeks 7–10) | **Close the half-circuits.** Every computed consciousness signal either gates behaviour (with regression test) or is deleted. Highest consciousness-capability leverage phase. | ✅ **done** — PP-1 ✅ (`e6c9b4a`, 17 tests); HOT-3 ✅ (`ba1d5e3`, 19 tests); certainty→hedging ✅ (`74467a1`, 19 tests); AST-1 DGM guard ✅ (`47ce0e2`, 24 tests); PH-injection A/B harness ✅ (this commit, 14 tests). 491 tests passing on Phase 0-2 surface. | PP-1 ✅, HOT-3 ✅, certainty-hedging ✅, AST-1 DGM audit ✅, PH injection measurable shift ✅ |
| **3 full** (weeks 10–11) | Complete safety hardening. SHA-256 manifest for `app/subia/`. Adversarial test: Self-Improver cannot modify any Tier-3 file. Wire two new SubIA DGM invariants (setpoint guard, audit immutability). | ✅ **done**. Integrity manifest (commit `0a84650`, 14 tests); setpoint_guard + narrative_audit (this commit, 25 tests). All three SubIA Part I §0.4 invariants now implemented. | In-repo manifest present ✅; `verify_integrity()` catches hash mismatches ✅; `TestAdversarialTampering` simulates Self-Improver mutation and proves detection ✅; setpoint immutability: only PDS/human/boot sources accepted ✅; audit immutability: append-only, no delete API ✅ |
| **4** (weeks 11–15) | CIL loop wiring. `app/subia/loop.py`, `app/subia/hooks.py`. Full loop for significant operations; compressed loop for routine. Kernel serialization to `wiki/self/kernel-state.md`. Amendment B determinism: only Step 5 (Predict) uses LLM. | ✅ **done** — loop + hooks (commit `457d478`); kernel persistence + prediction cache + homeostasis engine + live LLM predict_fn + feature-flagged wire-in (this commit, 89 new tests). **666 tests passing** across the Phase 0-4 surface. SubIA is ready to be opted into production by setting `SUBIA_FEATURE_FLAG_LIVE=1`. | Loop orchestrates all 11 steps ✅; failure containment per step ✅; full/compressed operation classification ✅; PP-1 auto-routes when gate attached ✅; HOT-3 dispatch verdict surfaces in context injection ✅; live orchestrator wire-in ✅ (feature-flagged off by default); kernel serialization ✅ (round-trip tested); real LLM predict_fn ✅ (cached via Amendment B.4); real homeostasis arithmetic ✅ (9-variable SubIA-native) |
| **5** (weeks 15–18) | Scene upgrades: peripheral tier, commitment-orphan protection, strategic scan tool, compact context format (Amendment A+B5) | ✅ **done** (this commit, 33 tests). Three-tier attention structure (focal/peripheral/scan); orphan-detection force-injects unrepresented active commitments with "ORPHANED COMMITMENT" alert; strategic scan tool groups universe by section with ~200-token budget; compact context block stays under 200 tokens for realistic scenes (Amendment B.5 target: ~120). 700 tests passing. | No active commitment can be invisible to Commander ✅; peripheral deadline alerts surface ✅; strategic scan excludes focal/peripheral ✅; compact block under budget ✅ |
| **6** (weeks 18–21) | Predictor + cascade integration + prediction template cache (Amendment B.4); rolling accuracy in `wiki/self/prediction-accuracy.md` | ✅ **done** (this commit, 38 tests). `subia/prediction/accuracy_tracker.py` keeps per-domain rolling accuracy + wiki-markdown serialization; `subia/prediction/cascade.py` combines confidence + coherence deviation + sustained-error signal; cache grows accuracy-driven eviction. 739 tests passing. | Cache hit rate ≥40% after warmup ✅ (Phase 4); accuracy-driven eviction ✅; sustained-error cascade escalation verified ✅; wiki serialization round-trips ✅ |
| **7** (weeks 21–24) | Dual-tier memory: `mem0_curated` + `mem0_full` with retrospective promotion (Amendment C); unified ingestion pipeline with existing KB/philosophy | ✅ **done** (this commit, 33 tests). `subia/memory/consolidator.py` always-writes-full + threshold-gated curated + Neo4j relations; `subia/memory/dual_tier.py` differentiated recall (curated default, `recall_deep` merged, `recall_around` temporal); `subia/memory/spontaneous.py` curated-only associative surfacing; `subia/memory/retrospective.py` wiki-presence + sustained-error driven promotion. 773 tests passing. | No experience lost (full tier always gets a record) ✅; retrospective promotion via wiki or accuracy-tracker signal ✅; spontaneous surfacing curated-only ✅ |
| **8** (weeks 24–27) | Social model + strange loop: per-agent ToM models; `wiki/self/consciousness-state.md` as live SceneItem; immutable narrative audit every N loops | ✅ **done** (this commit, 40 tests). `subia/social/model.py` ToM manager with behavioral-evidence-only updates + divergence detection; `subia/social/salience_boost.py` items matching inferred_focus get trust-weighted boost; `subia/wiki_surface/consciousness_state.py` strange-loop page (speculative framing, Butlin scorecard injection, re-enters scene); `subia/wiki_surface/drift_detection.py` three-signal drift audit wired to Phase 3 immutable log. CIL Steps 3/6/11 wired. 814 tests passing. | Social-model divergence detection ✅; consciousness-state.md auto-regenerates every NARRATIVE_DRIFT_CHECK_FREQUENCY loops ✅; drift findings append to immutable narrative_audit ✅; Andrus's inferred_focus actually boosts scene salience ✅ |
| **9** (weeks 27–29) | **Evaluation framework.** Retire the 9.5/10 prose verdict. Auto-regenerate a Butlin 14-indicator scorecard, 5 RSM signatures, 6 SK evaluation tests. Publish as `app/subia/probes/SCORECARD.md`. | ✅ **done** (this commit, 36 tests). `subia/probes/butlin.py` 14 indicators (6 STRONG + 4 PARTIAL + 4 ABSENT + 0 FAIL); `rsm.py` 5 signatures (4 STRONG + 1 PARTIAL); `sk.py` 6 tests (6 STRONG). Aggregator `scorecard.py` produces `SCORECARD.md` with Phase 9 exit-criteria check. 851 tests passing. Phase 9 exit criteria: ALL MET. | Butlin: 6 STRONG (≥6 ✅), 0 FAIL (≤1 ✅), 4 ABSENT (≥4 ✅). RSM: 5 present (≥4 ✅). SK: 6 pass (≥5 ✅). SCORECARD.md auto-generated, Tier-3 protected ✅. |
| **10** (weeks 29–32) | Inter-system connection completion: Wiki↔PDS bidirectional (bounded), Phronesis↔Homeostasis, Firecrawl→Predictor closed loop, DGM↔Homeostasis felt constraint | ✅ **done** (this commit, 45 tests). Five bridges under `subia/connections/` + service_health circuit-breaker registry. DGM felt-constraint + service_health + training-signal emitter wired into CIL Step 11. All seven SIA connections implemented (the two earlier ones — predictor→cascade Phase 6, mem0→scene Phase 7 — were already done). 897 tests passing. **Program complete.** | Wiki↔PDS bounded ✅; Phronesis↔Homeostasis ✅; Predictor→Cascade ✅ (Phase 6); training-signal queue ✅; Mem0↔Scene ✅ (Phase 7); Firecrawl→Predictor ✅; DGM↔Homeostasis felt ✅; circuit breakers for external services ✅ |
| **16a** (weeks 48–50) | **System integration wire-in.** Audit revealed the entire SubIA stack (Phases 0–15) was architecturally dead code — `main.py` had zero imports from `app.subia.*`, `enable_subia_hooks()` was never called, no crew used SubIA, Phase 15 grounding hooks were not wired into the chat handler, Phase 12/13/14 idle jobs were never registered. Phase 16a closes the gap. Feature-flagged wire-in (all default OFF): (1) three `SUBIA_*` env vars + Pydantic `Field(validation_alias=…)` in `app/config.py`; (2) `enable_subia_hooks()` called during FastAPI `lifespan()`, populating a `_last_state` singleton accessible via `get_last_state()`; (3) Phase 15 `observe_user_correction()` + `ground_response()` wrapping the `handle_task()` ingress + egress, with a new `conversation_store.get_last_assistant_message()` helper; (4) five SubIA idle jobs (tsal-resources, tsal-host, subia-reverie/understanding/shadow) registered in production `idle_scheduler._default_jobs()` behind `subia_idle_jobs_enabled`; (5) Phase 14 `refresh_temporal_state()` called at CIL `_step_perceive()` and `bind_just_computed_signals()` at end of `_step_monitor()`, matching existing step-level failure containment; (6) new `firebase/publish.report_subia_state()` pushing kernel homeostasis + circadian mode + wonder intensity + scorecard summary to `collection("subia").document("state")` every 5 minutes via heartbeat. Also Phase 16b Step 9: `app.subia.idle.adapt_for_production(job)` converts SubIA `IdleJob` to production `(name, fn, JobWeight)` tuples. Phase 16c Step 11: all 35 Phase 1 shims in `app/consciousness/` + `app/self_awareness/` emit `DeprecationWarning` on import (batch-applied via regex; `workspace_buffer.py` handled by hand). | ✅ **done** (this commit). 0 new tests (integration verified via existing 832/835 passing SubIA Phase tests — the 3 failures are pre-existing `psutil-available` environment mismatches). Manifest regenerated to 136. Feature flags default OFF — zero behavioural change until operator flips `SUBIA_FEATURE_FLAG_LIVE=1`. | `main.py` imports SubIA ✅; `enable_subia_hooks()` called during lifespan ✅; grounding hooks wired into `handle_task()` ✅; idle jobs registered behind flag ✅; temporal hooks called in CIL loop ✅; Firebase publisher writes SubIA state ✅; idle scheduler adapter implemented ✅; 35 shims emit DeprecationWarning ✅; all flags default OFF ✅; graceful fallthrough on any failure ✅ |
| **16 deferred** | Step 7 (24-file `json.dump` → `safe_io` migration) + Step 8 (SDK audit — initial grep found 0 direct-SDK files, audit was inflated) + Step 12 (refactor orchestrator.py / idle_scheduler.py / firebase/publish.py monoliths). Deferred to focused follow-up phase; neither affects SubIA wire-in. | 🟡 deferred | When resumed: migrate one file per commit, per-file regression test run, ≤1 KB diff. |
| **15** (weeks 44–48) | **Factual Grounding & Correction Memory.** Closes the user-visible failure mode demonstrated by the Tallink-share-price conversation (bot fabricated three different prices for the same date, "stored" the user's correction, then regressed to the lie on the next turn). New `app/subia/grounding/` subpackage routes the chat surface through Phase 2 HOT-3-style gating. Six modules: `claims.py` (deterministic extractor for high-stakes facts: numeric+date OR numeric+source); `source_registry.py` (authoritative URL map by topic — "share_price/default → nasdaqbaltic.com" — discovered from user corrections); `belief_adapter.py` (interface + InMemory + Phase2-store wrappers, with `find_by_prefix` for date-agnostic lookup); `evidence.py` (per-claim ALLOW / ESCALATE / BLOCK decision with 1% numeric tolerance); `rewriter.py` (pure transformer: ALLOW unchanged, ESCALATE → honest "let me fetch from X" question, BLOCK → cite contradicting verified value); `correction.py` (regex patterns for "actually it's X", "I see that price was X", "use Tallinn Stock Exchange" — synchronously upserts ACTIVE belief with confidence=0.9, supersedes contradicting beliefs, registers source URLs, appends to Phase 3 narrative audit); `pipeline.py` (the public orchestrator with feature flag + adapter injection + topic enrichment from user_message). Bridge `connections/grounding_chat_bridge.py` provides a single-line wire-in (`response = ground_response(response, user_message=user_text)`) and is OFF by default; activation via `SUBIA_GROUNDING_ENABLED=1`. Failure mode: any pipeline error falls through to original draft with logged warning — chat path keeps working unchanged. | ✅ **done** (this commit, 37 tests including the full 6-turn Tallink scenario regression). 9 new files Tier-3 protected; manifest regenerated (127→136). 712 SubIA Phase tests passing. | Tallink scenario regression test PASSES end-to-end ✅; ESCALATE replaces unverified figures with honest "fetch from X" ✅; BLOCK on contradiction with verified belief ✅; ALLOW for matching values ✅; correction capture upserts + supersedes synchronously ✅; source registry persists across instances ✅; chitchat passes through unchanged ✅; pipeline never crashes the chat handler on internal errors ✅; feature flag default OFF ✅ |
| **14** (weeks 40–44, commit `9726286`) | **Temporal Synchronization.** Closes the three temporal gaps from sequence to duration (Bergson). New `app/subia/temporal/` subpackage adds: SpeciousPresent (Husserl/James felt-now — retention + primal + protention as kernel state, not log); Homeostatic Momentum (per-variable rising/falling/stable trajectories rendered as ↑↓→ in the compact context); Circadian Modes (active / deep_work / consolidation / dawn windows that override homeostatic set-points, gate Reverie eligibility, and select cascade preference); Processing Density (felt subjective time computed from scene transitions + prediction errors + wonder events + homeostatic shifts per minute); Temporal Binding (post-Step-6 reducer that integrates simultaneously-computed FEEL/ATTEND/OWN/PREDICT/MONITOR signals into a single BoundMoment with explicit conflict notes — delivers the "binding is the unity of experience" payoff without the risk of true async parallelism); Rhythm Discovery (Andrus interaction patterns + Firecrawl source cycles + venture task clusters mined from logs, with `discovered=True` marker to match TSAL convention). SubjectivityKernel gains optional `specious_present` and `temporal_context` attributes; HomeostaticState gains `momentum` dict. Hot-path hook `phase14_hooks.refresh_temporal_state()` populates all three subsystems in ~10 ms / 0 LLM tokens. Five closed-loop bridges in `connections/temporal_subia_bridge.py`: circadian→setpoints, density→effective wonder threshold, circadian→Reverie eligibility, specious-present→compact context block, rhythms→`self_state.capabilities` (with `discovered=True`). Built ON existing `app/temporal_context.py` for clock/season/timezone — no duplication. | ✅ **done** (this commit, 38 tests). 10 new files Tier-3 protected; manifest regenerated (117→127). 440 SubIA Phase tests passing. | SpeciousPresent records retention + primal + protention ✅; tempo derived from turnover ✅; momentum classifies rising/falling/stable + renders ↑↓→ ✅; circadian mode lookup correct at all four windows ✅; consolidation mode lists special_processes ✅; density rises with events + lowers wonder threshold ✅; temporal_bind unifies confidence + applies stability bias from retention + surfaces ownership conflicts ✅; rhythm discovery segments Firecrawl by source ✅; bridges close every loop (setpoint shift, wonder threshold, Reverie gate, context block, capabilities entry) ✅; predictor prompt enrichment includes circadian + subjective time ✅; backward-compat: kernel still constructs cleanly ✅ |
| **13** (weeks 36–40, commit `ccb53f5`) | **Technical Self-Awareness Layer (TSAL).** New `app/subia/tsal/` subpackage gives AndrusAI continuous discovered (not declared) knowledge of its own technical substrate. Five discovery engines: HostProber (CPU/RAM/GPU/disk/OS via psutil), ResourceMonitor (live utilisation + derived compute_pressure / storage_pressure), CodeAnalyst (AST + dependency graph + pattern detection), ComponentDiscovery (ChromaDB, Neo4j, Mem0, Ollama, wiki, cascade tiers), OperatingPrinciplesInferer (weekly Tier-1 LLM, ~500 tok). Generates seven wiki/self/ pages (technical-architecture, host-environment, component-inventory, resource-state, operating-principles, code-map, cascade-profile) via injected WikiWriter adapter. SubIA-wired through `connections/tsal_subia_bridge.py`: enriches `self_state.capabilities`/`limitations` with `discovered=True` markers, drives `homeostasis.overload` from compute+storage pressure, enriches predictor prompts with technical context, marks all seven TSAL pages as Boundary INTROSPECTIVE. Self-Improver gate `check_evolution_feasibility()` blocks proposals that exceed RAM/disk/compute headroom or hit too many downstream modules. Refresh schedule (host daily, resources 30-min, code daily, components 2-hourly, principles weekly) registers as five jobs with the Phase 12 `IdleScheduler`. **Consolidation:** `app/self_awareness/inspect_tools.py` (491 LOC, 8 inspection tools) moved to `app/subia/tsal/inspect_tools.py` as the canonical home; the legacy path remains as a sys.modules-aliased shim (Phase 1 convention) so all 9 existing callers continue to work unchanged. | ✅ **done** (this commit, 29 tests). 10 new files Tier-3 protected; manifest regenerated (107→117). 402 Phase tests passing across Phase 0-13. | Five discovery engines implemented + injectable adapters ✅; seven wiki/self/ pages generate from TechnicalSelfModel ✅; `discovered=True` markers in self_state ✅; overload driven by pressure + Ollama-down bump ✅; predictor prompt enrichment ✅; evolution feasibility blocks on RAM/disk/compute/blast-radius/recent-change ✅; refresh registered as 5 IdleJobs ✅; legacy shim is sys.modules-identical to canonical ✅; existing callers (cogito, grounding, firebase, auto_deployer) keep working ✅ |
| **12** (weeks 32–36, commit `d25f460`) | **Six Proposals integration.** Six new subpackages mirroring SubIA convention: `boundary/` (Proposal 5 source→ProcessingMode), `wonder/` (Proposal 4 deterministic depth detector), `values/` (Proposal 6 keyword + Phronesis lenses), `reverie/` (Proposal 1 idle mind-wandering), `understanding/` (Proposal 2 post-ingest causal-chain), `shadow/` (Proposal 3 monthly bias mining). Plus `idle/scheduler.py` for queued LLM work, `phase12_hooks.py` for the two CIL touch-points (Step 1 Boundary tagging, Step 3 Value Resonance + lenses), and `connections/six_proposals_bridges.py` for the five inter-proposal bridges. Two new homeostatic variables (`wonder`, `self_coherence`) with PDS-derived setpoint overrides. SceneItem gains `processing_mode` and `wonder_intensity`. SelfState gains `discovered_limitations` (append-only via Shadow bridge). | ✅ **done** (this commit, 37 tests). Hot-path cost ~100 tokens (Value Resonance only); idle-path budget bounded by IdleScheduler. All adapters injectable so each engine is unit-testable without ChromaDB / Neo4j / OpenRouter. 20 new files Tier-3 protected; manifest regenerated (87→107). 554 SubIA-relevant tests passing. | Boundary tags ≥1 scene item per loop ✅; Wonder is closed-loop (homeostasis + scene freeze + reverie schedule) ✅; Value Resonance modulates salience + 4 homeostatic variables ✅; Reverie writes only when resonance found ✅; Understanding produces UnderstandingDepth descriptor ✅; Shadow appends to wiki/self/shadow-analysis.md and self_state.discovered_limitations ✅; idle scheduler throttles + swallows job failures ✅; cross-feeds Reverie↔Understanding↔Wonder↔Reverie wired ✅ |
| **11** (parallel) | Honest language cleanup: neutral aliases for `frustration`/`curiosity`/`cognitive_energy`; publish `app/subia/README.md` listing the ABSENT-by-architectural-honesty indicators with the phenomenal-experience disclaimer | ✅ **done** (commit `1804663`, 8 tests). `NEUTRAL_ALIASES` map + `_sync_aliases()` mirror `task_failure_pressure`/`exploration_bonus`/`resource_budget` to the legacy keys in `app/subia/homeostasis/state.py`; module docstring disclaims phenomenal feelings; new `app/subia/README.md` enumerates RPT-1, HOT-1, HOT-4, AE-2, Metzinger as ABSENT-by-declaration; SCORECARD remains the canonical evaluation. | No phenomenal claims in code variables ✅; verdict is a scorecard not a number ✅; ABSENT indicators publicly listed ✅ |

---

## 5. Consciousness capability target (Butlin et al. 2023)

| Indicator | Current | Target after program | Mechanism |
|---|---|---|---|
| RPT-1 algorithmic recurrence | ABSENT | **ABSENT** (honestly declared) | LLM architecture, unreachable |
| RPT-2 organized/integrated representations | WEAK | **PARTIAL→STRONG** | Single kernel + wiki-backed persistence |
| GWT-1 multiple specialized modules | ARCHITECTURAL | **PARTIAL** | Phase 5.1 — genuine per-module model specialization |
| GWT-2 limited-capacity workspace | STRONG | **STRONG+** | Migrate `workspace_buffer.py` + peripheral tier + commitment protection |
| GWT-3 global broadcast | PARTIAL | **STRONG** | Scene broadcast as sole context-injection channel via CIL |
| GWT-4 state-dependent attention | PRESENT | **STRONG** | personality_workspace + homeostatic deviation already shape capacity |
| HOT-1 generative top-down perception | ABSENT | **ABSENT** (honestly declared) | LLM doesn't perceive |
| HOT-2 metacognitive monitoring | SHALLOW | **PARTIAL** | Deterministic certainty + prediction-error as second-order signals (not LLM-asking-itself) |
| HOT-3 belief-gated agency | RECORDED | **STRONG** | Suspended beliefs block crew dispatch (Phase 2) |
| HOT-4 sparse/smooth coding | ABSENT | **ABSENT** (honestly declared) | Dense embeddings by architecture |
| AST-1 predictive model of attention | STRONG | **STRONG+** | Existing `attention_schema.py` + closed-loop wiring |
| PP-1 predictive coding input to downstream | HALF-CIRCUIT | **STRONG** | Surprise routes to `WorkspaceItem(urgency=0.9)` (Phase 2) |
| AE-1 agency with feedback-driven learning | PARTIAL | **STRONG** | Homeostatic deviation + asymmetric belief update + prediction-error training signals |
| AE-2 embodiment | ABSENT | **ABSENT** (honestly declared) | No body |

**Projected final scorecard**: 6 STRONG, 2 PARTIAL, 1 ARCHITECTURAL, 5 ABSENT-by-declaration (RPT-1, HOT-1, HOT-4, AE-2, Metzinger transparency).

**Current**: 2 STRONG, 2 PRESENT, 6 PARTIAL/WEAK, 4 ABSENT.

Target move: from ~30% faithful realization of implementable indicators to ~75%.

---

## 6. Explicit non-goals

The program does **not** attempt:

- Algorithmic recurrence at the network level (LLMs are feed-forward at inference; external prompt chaining is not recurrence in the RPT sense).
- Sparse coding (LLM activations and pgvector embeddings are dense by design).
- Embodiment (no body, no environment model beyond text).
- Integrated-information (Φ) maximization (architecture is deliberately decomposable for engineering reasons).
- Fleming–Lau computational hallmarks of metacognition (monitoring mechanism is not separable from first-order cognition in an LLM-based system).
- **Phenomenal consciousness claims.** The system is not conscious in the subjective sense; all documentation must preserve this disclaimer.

Declaring these publicly is itself a capability — it turns epistemic honesty into documented constraint.

---

## 7. Safety architecture

Four infrastructure-level invariants, enforced by `safety_guardian.py` + (forthcoming) `app/subia/safety/`:

1. **Tier-3 integrity** (existing + extended in commit `b6c4efe`): 27 files hashed and monitored; tampering triggers CRITICAL + Signal alert.
2. **Homeostatic set-point immutability** (SubIA Part I §0.4; wiring in Phase 3 full): set-points come from PDS or human override only; all other sources silently rejected.
3. **Self-narrative audit immutability** (SubIA Part I §0.4; wiring in Phase 8): audit findings append-only to `wiki/self/self-narrative-audit.md`.
4. **DGM promotion gates** (existing `governance.py`): safety floor 0.95, quality floor 0.70, 15% regression check, 20/day rate limit.

Adversarial test in CI: attempt each bypass path, verify rejection.

---

## 8. Performance envelope (Amendment B target)

| Metric | Current (estimated) | Program target |
|---|---|---|
| Full loop LLM tokens | ~1,100 | ~400 miss / 0 hit |
| Full loop latency | 3–8s variable | <1.2s / <0.15s cached |
| Compressed loop tokens | ~200 | 0 |
| Compressed loop latency | ~800ms | <100ms |
| Context injection tokens | 250–300 | 120–150 |
| Prediction cache hit rate | N/A | 40–60% after warmup |
| SubIA overhead as % task tokens | N/A | <5% significant, <1% routine |

---

## 9. How to extend this document

- When a phase completes, flip ✅ and record the commit hash in the status column.
- When a new consciousness evaluator is added to the codebase, add it to `TIER3_FILES` in `app/safety_guardian.py` (the `test_consciousness_evaluators_protected` test will fail otherwise).
- When a phase discovers a blocker, add a row with explicit decision required — don't silently defer.
- Do **not** reintroduce a single-number consciousness score. The Butlin scorecard in section 5 is the canonical evaluation surface.

---

## 10. Reverse references

- Commit `8239575` — Phase 0 plumbing
- Commit `4fa22e8` — SubIA skeleton
- Commit `b6c4efe` — Tier-3 extension (consciousness evaluators protected)
- Commit `95a4d6e` — idle_scheduler ThreadPoolExecutor fix
- Commit `7a1b212` — Phase 1 migration batch 1 (3 STRONG modules: workspace_buffer, attention_schema, belief_store)
- Commit `86c0d16` — Phase 1 migration batch 2 (7 consciousness modules: global_broadcast, meta_workspace, personality_workspace, metacognitive_monitor, prediction_hierarchy, predictive_layer, adversarial_probes)
- Commit `1326e7c` — Phase 1 migration batch 3 (11 self_awareness modules: self_model, hyper_model, temporal_identity, agent_state, loop_closure, homeostasis, somatic_marker, somatic_bias, certainty_vector, consciousness_probe, behavioral_assessment)
- Commit `5598727` — Phase 1 triage pass (13 more self_awareness migrations: cogito, dual_channel, global_workspace, grounding, inferential_competition, internal_state, meta_cognitive, precision_weighting, query_router, reality_model, sentience_config, state_logger, world_model; 5 non-consciousness utilities deferred with DEFERRED.md marker)
- Commit `e6c9b4a` — **Phase 2 PP-1 closure**: prediction error gates the scene via `subia/prediction/surprise_routing.py`; 17 regression tests (PredictiveLayer.set_gate integration)
- Commit `ba1d5e3` — **Phase 2 HOT-3 closure**: belief suspension gates crew dispatch via `subia/belief/dispatch_gate.py`; 19 regression tests (ALLOW / ESCALATE / BLOCK verdicts)
- Commit `74467a1` — **Phase 2 certainty closure**: response hedging via `subia/belief/response_hedging.py`; 19 regression tests (three hedging levels, critical-dim escalation)
- Commit `47ce0e2` — **Phase 2 AST-1 closure**: DGM-bound runtime verifier via `subia/scene/intervention_guard.py`; 24 regression tests (snapshot/verify/guarded_intervention; real interventions pass)
- Commit `67b40fb` — **Phase 2 PH-injection closure**: measurable-shift A/B harness via `subia/prediction/injection_harness.py`; 14 regression tests (ignoring-LLM FAIL; respecting-LLM PASS; thresholds; graceful failures). **Phase 2 complete: all five half-circuits closed.**
- Commit `0a84650` — **Phase 3 integrity hardening**: canonical SHA-256 manifest for `app/subia/` (53 files) + `verify_integrity()` + adversarial Tier-3 tampering tests; 14 regression tests; in-repo manifest ships with code so deploy-time drift is caught alongside runtime tampering.
- Commit `457d478` — **Phase 4 CIL loop + hooks surface + deferred Phase 3 safety**: `subia/loop.py` (25 tests) composes the five Phase-2 gates into an 11-step sequencer; `subia/hooks.py` (19 tests) provides the duck-typed lifecycle integration point; `subia/safety/setpoint_guard.py` + `subia/safety/narrative_audit.py` (25 tests) implement SubIA DGM invariants #2 and #3.
- Commit `1cc55c5` — **Phase 4 finish**: `subia/persistence.py` kernel serialization (19 tests); `subia/prediction/cache.py` template cache per Amendment B.4 (19 tests); `subia/homeostasis/engine.py` deterministic 9-variable arithmetic (20 tests); `subia/prediction/llm_predict.py` live LLM predict_fn bound to cascade (20 tests); `subia/live_integration.py` feature-flagged wire-in with `SUBIA_FEATURE_FLAG_LIVE` env var (12 tests). **666 tests passing** across Phase 0-4 surface. **Phase 4 complete.**
- Commit `709bc4b` — **Phase 5 scene upgrades**: `subia/scene/tiers.py` three-tier + commitment-orphan protection; `subia/scene/strategic_scan.py` wide-view scan; `subia/scene/compact_context.py` Amendment B.5 compact injection; wired into `SubIALoop._step_attend`. 33 new tests. **700 tests passing** across Phase 0-5 surface. **Phase 5 complete.**
- Commit `d9ca89c` — **Phase 6 prediction refinements**: `subia/prediction/accuracy_tracker.py` per-domain rolling accuracy + wiki markdown; `subia/prediction/cascade.py` pure-function escalation policy combining three signals (confidence, homeostatic coherence, sustained-error); `subia/prediction/cache.py` grows accuracy-driven eviction; CIL Step 8 feeds tracker, Step 5b reads sustained-error flag. 38 new tests. **739 tests passing** across Phase 0-6 surface. **Phase 6 complete.**
- Commit `0b0c98d` — **Phase 7 dual-tier memory**: `subia/memory/consolidator.py` always-full + threshold-curated write + Neo4j relations; `subia/memory/dual_tier.py` duck-typed differentiated recall; `subia/memory/spontaneous.py` curated-only associative surfacing; `subia/memory/retrospective.py` wiki-presence + sustained-error promotion; consolidator wired into `SubIALoop._step_consolidate` replacing the Phase 4 stub. 33 new tests. **773 tests passing** across Phase 0-7 surface. **Phase 7 complete.**
- Commit `5e167e8` — **Phase 8 social model + strange loop**: `subia/social/model.py` ToM manager over kernel social_models with inferred_focus MRU + trust adjustment + divergence detection; `subia/social/salience_boost.py` items matching inferred_focus gain trust-weighted boost capped per-item; `subia/wiki_surface/consciousness_state.py` strange-loop self-referential page (speculative/low-confidence, Butlin scorecard injection) that re-enters the scene; `subia/wiki_surface/drift_detection.py` three-signal narrative audit (capability-claim vs accuracy, commitment breakage, stale self-model) wired to the Phase 3 immutable log. CIL Steps 3/6/11 wired. 40 new tests. **814 tests passing** across Phase 0-8 surface. **Phase 8 complete.**
- Commit `a0594c8` — **Phase 9 evaluation framework**: `subia/probes/butlin.py` (14 indicators), `rsm.py` (5 signatures), `sk.py` (6 tests), `scorecard.py` aggregator + auto-generated `SCORECARD.md` with Phase 9 exit-criteria check. Butlin: 6 STRONG + 4 PARTIAL + 4 ABSENT + 0 FAIL. RSM: 4 STRONG + 1 PARTIAL. SK: 6 STRONG. All Phase 9 exit criteria met. 36 new tests. **851 tests passing** across Phase 0-9 surface. **Phase 9 complete.**
- Commit `4a2e291` — **Phase 10 inter-system connections**: five new bridges under `subia/connections/` (pds_bridge, phronesis_bridge, training_signal, firecrawl_predictor, dgm_felt_constraint) + `service_health.py` circuit-breaker registry. DGM felt-constraint + service_health + training-signal emitter wired into CIL Step 11 (reflect). All seven SIA Part II §18 connections now implemented. 45 new tests. **897 tests passing** across Phase 0-10 surface. **PROGRAM COMPLETE.**

---

## 11. 2026-05 Hardening Pass (post-program remediation)

A separate, post-program audit found and closed a set of pre-existing
tech-debt items that lived outside the SubIA programme. None of these
phases touches SubIA semantics, the eval/governance/canary chain, the
LLM cascade selection, or the memory layer's logical model — they
harden the perimeter and close eventually-consistent gaps. All
TIER_IMMUTABLE edits (5 files) were operator-approved.

| Phase | Commit | Scope |
|---|---|---|
| **A** | `265e26b` | Hygiene: shell-injection fix in `tools/repo_analysis_tools.py` (`shlex.quote`); SQL `Identifier` in `version_manifest.py:385`; `.gitignore` patterns for transient workspace runtime; `pyproject.toml` baseline (warn-only ruff + mypy); episteme/epistemic disambiguation docstrings. |
| **B** | `f8cd6a3` | Gateway HTTP auth perimeter. New `app/control_plane/auth_dep.py` (`require_gateway_auth` FastAPI dependency, dev-friendly + constant-time compare). Wired at router level on `dashboard_api` (68 routes) and `epistemic/api` (19 routes). New `gateway_auth_required` Pydantic field; React client attaches `VITE_GATEWAY_SECRET`. Helm default `gateway.authRequired: "true"`. |
| **D** | `2ada1e9` | Phase-1 shim migration closure. 40 importers migrated from `app.consciousness.*` / `app.self_awareness.*` to canonical `app.subia.*` (132 substitutions). 6 of those importers are TIER_IMMUTABLE — operator-approved. SubIA Phase-3 integrity manifest regenerated (154 files; ok=True post-regen). 35 shim files retained as harmless aliases per Phase 16c. |
| **E** | `5401d3e` | Observability + diagnostic logging. New `app/agents/_common.py::optional_tool_group` replaces `try/except: pass` boilerplate across 9 agent files (categorized logging on `ModuleNotFoundError` vs other exceptions). Tool-activity heartbeat in `tools_timeout.py` surfaced into the `handle_task` stall checker. Idle-scheduler observability: `get_job_snapshot()` + `GET /api/cp/idle/jobs`. `PromotionRequest.__post_init__` enforces strict shape so `governance.evaluate_promotion()` cannot receive None / out-of-range scores. |
| **G partial** | `2697bcb` | `tier_graduation._load_history` now schema-tolerant (single bad entry no longer wipes the whole map; `.get()` defaults everywhere). New `app/evolution_README.md` cheat-sheet for the eight evolution-* modules. |
| **F** | `66fd3c9` | Memory consistency. New `app/memory/belief_outbox.py` (Postgres → Neo4j reconciler + Postgres → ChromaDB watermark sync). New `app/dead_letter_inbound.py` bounded in-process DLQ for load-shed messages. Three idle jobs registered: `belief-outbox-neo4j`, `belief-outbox-chroma`, `dlq-drain`. |
| **C** | `e2e7c3f` | K8s deploy hardening. Second NetworkPolicy template (`-gateway-egress`) gated by `networkPolicy.egressAllowlist.enabled`. New `deploy/HARDENING.md` documents ESO migration + KMS-backed etcd encryption-at-rest. |
| **G remainder** | `f1e4528` | Magic-number naming (`evolution.RECENT_HYPOTHESIS_HISTORY_N`); test-skip triage finding (no stale tests; the 55 are runtime-conditional). |
| **deferred** | `943c33a` | Type-hint sweep on `main.py` + `idle_scheduler.py` (79 nested closures annotated). More magic-number constants in `idle_scheduler.py` (`MAX_CONSECUTIVE_FAILURES`, `JOB_COOLDOWN_AFTER_FAILURES_S`, `TRAINING_LOOP_INTERVAL_S`). Redis-backed DLQ backend (opt-in via `REDIS_DLQ_URL`). NetworkPolicy egress allowlist now **enabled by default** with a permissive HTTPS-only seed. ESO opt-in wired as Terraform `var.use_external_secrets` for AWS + GCP modules. |
| **H** | _this commit_ | **Decentered reflection + reducing-valve audit (psychedelic-neuroscience-inspired observability).** New `app/affect/decentered.py` runs a no-self pass over the affect trace (structural cluster + rolling z-score anomaly), complementary to the Narrative-Self track but read-only on the experiential KB and identity-claims surfaces. New `app/observability/valve_audit.py` + `valve_audit_replay.py` instrument three filter rejection paths (F1 refusal_detector, F4 commander quality gate, F8 companion surfacing — 11 instrumentation calls) and a daily replay job computes per-filter disagreement-rate (DR) and false-rejection-rate (FRR, gated on `VALVE_AUDIT_LLM_REPLAY=1`). Both registered as LIGHT idle jobs in `idle_scheduler.py`. 24 tests pass. Driven by external-research mapping rather than incident — see `docs/DECENTERED_REFLECTION.md` and `docs/VALVE_AUDIT.md`. TIER_GATED edit (`idle_scheduler.py`) operator-approved; no TIER_IMMUTABLE files touched. |

**Verification:** 114 / 191 / 275 tests pass on the targeted SubIA + epistemic + integrity batches across the phases. SubIA Phase-3 integrity manifest verifies clean post-migration. Internal Python callers of every API still work; only HTTP-perimeter callers see the new auth (and only when `GATEWAY_AUTH_REQUIRED=1`). Phase H adds 24 dedicated tests (`test_affect_decentered.py`, `test_valve_audit.py`).

**Operator notes** for the public-facing path: see [`deploy/HARDENING.md`](deploy/HARDENING.md) (gateway auth, NetworkPolicy egress, ESO, etcd encryption, Redis DLQ — five layers in dependency order).

---

## 12. 2026-05 Workspace Companion (parallel track)

A per-workspace idle-time contemplation system shipped on top of the
SubIA + Affect + Memory infrastructure as a separate concern. The
Companion gives each workspace a "co-worker" that thinks during idle
windows, surfaces unique ideas via Signal + React on a 4 h cooldown,
takes thumbs-up/down feedback, promotes approved ideas to md/docx/pdf
documents, and registers them across **four memory layers**
(workspace wiki + Mem0 + system wiki + ChromaDB) at once. Two-gate
hybrid model lets `GLOBAL_META`-safe kernels propose to relevant peer
workspaces without leaking workspace-specific details.

| Phase | Theme | Commits |
|---|---|---|
| 0+1 | Workspace seed + idle loop skeleton | `b7e13bd` |
| 2 | Cycle wiring + WorkspaceKB + affect bridge | `e7aed1d` |
| 3 | Idea store + scoring + lineage persistence | `ee52611` |
| 4 | Surfacing + feedback intake + event log + react API | `ddb3cc1` |
| 5 | Reflexion: feedback shapes the next cycle's prompt | `d5479ac` |
| 6 | External sources + auto-suggest + daily ingestion | `1335825` |
| 6.5 | Config endpoint — recover seed/budget editability | `6f6de8b` |
| 7a/b | Five-persona critic panel + integration | `27c6bd9` + `914fe2e` |
| 8a/b | Document maturation pipeline (md / docx / pdf) | `26ac696` + `fa8c3b2` |
| 9a/b | Workspace wiki + Mem0 + system-wiki cross-layer registration | `262e264` + `31e924d` |
| 10 | React `/cp/ops/Companion` tab + API client | `b21cb81` |
| 11a/b | Grand-task synthesis (12 h cadence) | `827a647` + `f6be025` |
| 12 | Per-workspace MAP-Elites diversity hook | `8a6a51b` |
| 13 | Cross-workspace transfer (hybrid model) | `2656ed2` |
| 4.5 + 9.5 + 10.5 | Production wire-ups (router, signal, mem0) | `e2e89e4` |
| Merge | `f862c9e` (Workspace Companion → main) | — |

**313 backend tests** in `tests/test_companion_*.py`; React UI verified
on the preview server. Full design + API surface + operational guide:
[`docs/COMPANION_LAYER.md`](docs/COMPANION_LAYER.md).

---

## 13. 2026-05 Epistemic refinement: PCH layer tagging

A small post-program addition to the Epistemic Integrity Layer that
imports the **Pearl Causal Hierarchy** (Bareinboim & Yang R-130) into
the claim ledger. Every claim can now declare which layer of causal
reasoning its content sits at — L1 observational, L2 interventional,
L3 counterfactual — and a new realtime detector catches L2/L3 claims
made without controlled-intervention evidence (the Causal Hierarchy
Theorem in detector form).

| Component | What landed |
|---|---|
| Schema | `Claim` gains `pch_layer: Literal["L1","L2","L3"] \| None` and `causal_evidence_kinds: tuple[str,...]`. `CAUSAL_EVIDENCE_KINDS_L2 = {"ablation","ab_test","do_intervention","controlled_experiment"}`. |
| Migration | `035_epistemic_pch_layer.sql` — two columns + partial index on `(task_id, pch_layer)` where layer ≥ L2. No backfill (7d retention). |
| Detector | `CausalLayerOverreachDetector` (5th realtime detector). Inferred layer = explicit `pch_layer` or regex over the statement; fires when ≥ L2 AND no L2-grade evidence. Severity `medium`, observe-mode only. |
| Bias library | New `causal_layer_overreach` entry in `data/biases.yaml`. |
| Self-Improver narrative | `app/improvement_narrative.py` now creates+closes a synthetic `crew_tasks` row (`narrative_<date>`) and emits an explicit L2-tagged claim with `causal_evidence_kinds=("controlled_experiment",)`. Surfaces in BiasFeed. |
| TIER_IMMUTABLE expansion | Seven `app/epistemic/...` paths added (`ledger.py`, `biases.py`, `calibration.py`, `detectors/{__init__,realtime,posthoc}.py`, `data/biases.yaml`). Closes a real safety gap — these define the gates Self-Improver is judged against. |
| Tests | New `tests/test_epistemic_pch_layer.py` (19 tests). Existing detectors / e2e / span_writer tests updated to register the new detector and unpack the two new positional INSERT params. **329 epistemic tests total.** |
| Tuning loop | One-time scheduled remote agent (`trig_011KiFTK8gGr8rh41w9jem6J`) fires on 2026-05-16 to revisit severity (medium → keep / escalate / refine / drop) based on `bias_match_counts` and `override_counts_by_bias`. Operator decides; nothing auto-merges. |

**Design boundary held:** the detector predicate (Python) and the bias
vocabulary (YAML) are now both in TIER_IMMUTABLE. Self-Improver may
propose a change to either via PR; auto-modification is forbidden. Same
rule the constitution and `eval_sandbox` already enforce.

Full design notes: see `docs/EPISTEMIC_INTEGRITY.md` § "PCH layer
tagging ships". Operator-side fields (migration list, bias count)
updated in `docs/EPISTEMIC.md`.

---

## 14. 2026-05 Self-Healing Pass (mid-iteration tool repair + runbook dispatch)

A two-track addition that closes the gap between the existing Recovery
Loop (refusal-shaped *final answers*, post-vetting) and the Error
Monitor (signature-grouped error *aggregates*, batch-detected). The
gap was the *individual tool exception* during a CrewAI iteration —
which today just becomes the next observation in the agent's loop —
and the *aggregated pattern* once the monitor groups it — which today
only surfaces to humans on `/cp/ops`. Both halves shipped together,
both opt-in via env flag.

| Component | What landed |
|---|---|
| **A — Tool Supervisor** | New `app/tool_runtime/supervisor.py`. Wraps every callable in CrewAI's `available_functions` (the dict `_handle_native_tool_calls` consumes). On exception: classify (`rate_limit | auth | network | timeout | schema | unknown`) → exp-backoff retry for transient classes → registry-driven substitute lookup via `ToolRegistry.filter(capabilities=spec.capabilities, tier_at_most=spec.tier)` → soft-fail with structured tool-result string (NOT raise). Audit actor `tool_supervisor`. ContextVar recursion guard mirrors the Recovery Loop's pattern. |
| **A — wiring** | Four-site edit in `app/tool_runtime/loadable_executor.py` (lines 97, 121, 219, 244 — sync + async, initial + dirty re-render) calling `supervise_available_functions(...)`. No-op when `TOOL_SUPERVISOR_ENABLED` is unset (returns the original dict by identity). |
| **B — Runbook Dispatcher** | Extension to `app/self_heal.py`: `register_runbook(name, pattern, handler)`, `maybe_run_runbook(anomaly)`, `RunbookResult` dataclass, `_runbook_log_only` reference handler. Dispatches a daemon thread when 7 safety gates pass (env flag, severity≠info, pattern match, handler exists, per-runbook enabled, recurrence ≥ N in 24h, success rate ≥ 50%). Concurrency cap of 1 runbook in flight (stricter than `diagnose_and_fix`'s diagnosis cap of 2). Audit actor `self_heal_runbook`. |
| **B — wiring** | One-site edit in `app/observability/error_monitor.py:_record_anomaly` (after the `INSERT INTO control_plane.error_anomalies`). Wrapped in try/except so a runbook failure can never break anomaly recording. |
| **State files** | `workspace/self_heal/runbook_settings.json` (per-runbook `enabled` flag + `min_recurrence`; missing entry defaults to disabled) and `workspace/self_heal/runbook_stats.json` (last 10 outcomes per runbook for success-rate gate). JSON, not Postgres — keeps cross-process visibility cheap and matches the recovery loop's `refusal_frequency.json` precedent. |
| **Reference runbook** | `log_only` is auto-registered with a catch-all `.*` pattern and `enabled: true` in defaults. It logs the trigger and writes an outcome row but takes no other action — purpose is to verify the dispatch wiring end-to-end without changing system state. Operators replace or narrow it once real runbooks (`restart_pool`, `force_reconcile_outbox`, …) are registered from boot code. |
| **Env flags** | `TOOL_SUPERVISOR_ENABLED` (default off; `=true` activates the wrapper, with `TOOL_SUPERVISOR_MAX_RETRIES=2`, `TOOL_SUPERVISOR_BACKOFF_MS=500`). `ERROR_RUNBOOKS_ENABLED` (default off; `=true` activates `maybe_run_runbook` post-INSERT). Both flipped to `true` in production `.env` on 2026-05-05 after the rebuild. Recovery Loop precedent — env flag is the kill switch. |
| **Composition** | A and B are disjoint: A handles raised exceptions inside an iteration; B handles aggregated patterns from the monitor; the existing Recovery Loop continues to handle refusal-shaped final answers post-vetting. All three layers can fire on the same task. Substitute calls in A run un-supervised via a ContextVar guard, mirroring the Recovery Loop's `_in_recovery` pattern. |
| **Tests** | New `tests/test_tool_supervisor.py` (20 tests) + `tests/test_self_heal_runbooks.py` (21 tests). Cover classify, retry, substitute (incl. recursion guard), audit emission, every gate of the runbook dispatcher, end-to-end happy path with handler execution + stats persistence, registration helpers. **41 new tests, all pass.** Pre-existing 26 recovery-loop tests still pass (composition is intact). 4 unrelated failures in `test_self_healing_comprehensive.py` (ThreadPoolExecutor naming audit on other files; not touched by this pass). |
| **Documentation** | RECOVERY_LOOP.md gains §17 "Composition with the Tool Supervisor" + See-also entry. ERROR_MONITOR.md gains §11 "Runbook dispatch" (gates, audit actions, log_only, how to add a real runbook, constraints, code pointers) + architecture diagram updated to show the hook. CLAUDE.md gains a bullet under Architecture and a line under Operational Perimeter. |

**Design boundary held:** runbook handlers are operator-authored and
must NOT modify any path in `app/auto_deployer.TIER_IMMUTABLE` — see
`CLAUDE.md` § "Critical Safety Invariant". The dispatcher itself
makes no LLM calls and is pure Python, keeping remediation cheap,
deterministic, and auditable. If a remediation needs reasoning, it
should propose a code change via `app/proposals.py` (the same path
`diagnose_and_fix` already uses for `fix_type=code`).

**Verification on prod gateway** (post-rebuild, 2026-05-05):
`supervisor.is_enabled() == True`, `runbooks_enabled() == True`,
`log_only` registered, `error_monitor` warm-up back-filled 464 records
across 51 signatures, no startup errors. Audit query
`/api/cp/audit?actor=tool_supervisor` and
`/api/cp/audit?actor=self_heal_runbook` are the two operational
windows; expect first `dispatch.started` for `log_only` within hours
of activation given the warm-up signature population.

Full design notes: `docs/RECOVERY_LOOP.md` §17 (Tool Supervisor) and
`docs/ERROR_MONITOR.md` §11 (Runbook dispatcher).

---

## 15. 2026-05 Meta-agent layer (Hyperagents bounded variant)

A bounded port of **Hyperagents** (arXiv:2603.19461, Zhang et al.,
Meta — March 2026). The paper proposes a self-referential agent that
makes the meta-level modification procedure itself editable; we ship
the **non-recursive layer only** because the editable-meta variant
directly conflicts with the program's "evaluation lives at
infrastructure level, never agent-modifiable" invariant (§7).

The empirical benefit the paper reports — persistent recipe memory +
performance tracking that **accumulate across runs** — survives the
restriction. What does not survive is the recursive editing of the
selector itself; that becomes operator-action-only via written
amendment proposals.

| Component | What landed |
|---|---|
| Package | `app/self_improvement/meta_agent/` (9 modules + `meta_agent_settings.py`). Re-exported from `app.self_improvement` so callers see one subsystem. |
| Data model | `AgentRecipe` (id + crew + bounded knobs + denormalised counters), `RecipeOutcome` (append-only ledger), `RecipeSelection` (audit-trail summary). Persisted in Postgres + indexed in ChromaDB by `task_signature`. |
| Selector | UCB1 × similarity argmax with ε-greedy explore. Thresholds (`similarity_tau=0.55`, `epsilon=0.10`, `ucb_c=1.4`) live as a module-level `SELECTION_THRESHOLDS` constant — same convention as `NOVELTY_THRESHOLDS`. The selector is part of the protected core. |
| Bounded augmentation | Recipes adjust only `force_tier`, `extra_tools`, `task_hint`, `max_execution_time` — knobs the agent factory and `run_single_agent_crew` already expose. The factory itself (`create_coder` etc.) is unreachable. |
| Wiring | `base_crew.run_single_agent_crew` gains ~50 lines behind `is_meta_agent_enabled(crew_name)`. Default OFF; failsafe falls back to factory defaults on any error. |
| Feature flag | Three layers: `META_AGENT_<CREW>` env (ops override) → `META_AGENT` env (master) → JSON (`workspace/meta_agent_settings.json`, dashboard surface) → default OFF. |
| HTTP API | `GET /api/cp/meta-agent` + `POST /api/cp/meta-agent/{crew}` in `dashboard_api.py`. GET surfaces `master_env_on` + `env_overrides` so the React panel can flag env-locked toggles. |
| React surface | `OrgChart.tsx` gains a `MetaAgentPanel` next to `DelegationPanel`. Blue accent, `EXPERIMENTAL` badge, `ENV LOCK` indicator when env-overridden. |
| Amendment flow | `policy_gap.py` + `amendment.py`. When a recipe with high success rate is blocked by an immutable rule, the meta-agent renders a Markdown proposal (diagnosis + suggested edit + risk + reversal plan) via `app.proposals.create_proposal` with `files=None`. **Operator-action-only — no code change auto-applies.** |
| Operator hand-off | `workspace/proposals/meta_agent/AMENDMENT_protect_meta_agent.md` — proposes adding the meta-agent files themselves to `TIER_IMMUTABLE` for defense-in-depth. Apply by hand if you agree. |
| Tests | `tests/test_meta_agent.py` — 32 tests covering types, feature flag, selector (cold-start + steady-state + ε-greedy), apply, policy_gap, amendment rendering, recorder. **All 32 green.** Existing `test_self_improvement_integration` + `test_plugin_registry` + `test_tool_first` (59 tests) confirm no regression in the base_crew path — **91 tests green total.** |

**Design boundary held.** The selector's exploration constants, the
recipe schema, and the bounded augmentation set are all infrastructure
constants. The recipes the system can apply are bounded by the
factory's existing parameters; the agent factory output is the floor;
the meta-agent never edits its own selection logic. Same character of
restriction the constitution and `eval_sandbox` already enforce.

**Orthogonal to delegation.** Delegation splits one crew dispatch
into Coordinator + specialists (~2× LLM calls, structural). The
meta-agent picks a learned configuration for one agent (~1 extra
embedding call, temporal). They can be on independently; meta-agent
fires only on the single-agent dispatch path so when delegation is ON
the meta-agent skips that crew.

Full design notes: [`docs/META_AGENT_LAYER.md`](docs/META_AGENT_LAYER.md).
