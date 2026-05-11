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
| 11.5 | Cold-start seed bootstrap from CP mission + tickets | `b81771fc` (PR #75 → `43b24137`) |

**Phase 11.5 closes the chicken-and-egg gap exposed by post-merge audit:
none of the operator's 5 workspaces had `seed_prompt` set, so every
cycle aborted in 13 ms with `no_seed_prompt`. Phase 11 grand-task
synthesis couldn't help — its activation gate requires ≥3 polished
ideas. Phase 11.5 reads `control_plane.projects.mission` + the last 15
ticket titles for the project, calls a cheap-tier LLM once (~$0.0002),
persists the synthesised seed via Phase 6.5 `config.save`, emits a
`SEED_DERIVED` event, and continues the same cycle with the new seed.
The `default` workspace is blocklisted (catch-all, mixed signal); user
override via `POST /api/cp/companion/config/{ws}` always wins.

**336 backend tests** in `tests/test_companion_*.py` (313 prior + 23
new for Phase 11.5); React UI verified on the preview server. Full
design + API surface + operational guide:
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
| **A — wiring** | Two-site edit (sync path) in `app/tool_runtime/loadable_executor.py`, marked `[SUP-1]` — initial render + dirty re-render — calling `supervise_available_functions(...)`. No-op when `TOOL_SUPERVISOR_ENABLED` is unset (returns the original dict by identity). The async-path mirror is a known follow-up. |
| **B — Runbook Dispatcher** | New module `app/healing/runbooks.py` (separate sibling of `error_diagnosis.py` post-PR #65; the original §14 plan said "extension to `app/self_heal.py`" but that file no longer exists — one concern per file matches the refactor philosophy). Public surface: `register_runbook(name, pattern, handler)`, `maybe_run_runbook(anomaly)`, `unregister_runbook(name)`, `runbooks_enabled()`, `RunbookResult` dataclass, `_runbook_log_only` reference handler. Dispatches a daemon thread when 7 safety gates pass (env flag, severity≠info, pattern match, runbook enabled, recurrence ≥ N in 24h, success rate ≥ 50%, concurrency cap). Concurrency cap of 1 runbook in flight (stricter than `diagnose_and_fix`'s diagnosis cap of 2). Audit actor `self_heal_runbook`. |
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

**Restoration note (2026-05-07).** The 2026-05-05 prod gateway
verification described above (`supervisor.is_enabled() == True`,
`runbooks_enabled() == True`, `log_only` registered) was real on the
running container but was never reproducible from the repo: only the
`__pycache__/*.pyc` for `supervisor.py` and the two test files
survived locally; no source was committed. PR #65 (the healing
package consolidation, 2026-05-06) renamed `app/self_heal.py` →
`app/healing/error_diagnosis.py`, by which point any uncommitted
runbook extension to the original file was structurally homeless.
Both tracks were rebuilt on 2026-05-07 from the orphan bytecode
disassembly + this section's spec, with the sole structural change
being the new sibling module placement (Track B sits at
`app/healing/runbooks.py`). The 41-test count matches the original
(20 + 21); all pass on restoration commit.

**Operational windows.** Audit queries
`/api/cp/audit?actor=tool_supervisor` and
`/api/cp/audit?actor=self_heal_runbook` surface every action.
Expect the first `dispatch.started` for `log_only` within hours of
flipping `ERROR_RUNBOOKS_ENABLED=true` against a populated signature
window.

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

---

## 16. 2026-05 Company Dossier subsystem

A deterministic pipeline that produces an investment-grade,
source-attributed company report (10–15 page PDF) from a single
natural-language request. Distinct from the open-ended `financial`
crew (analyst chat, ad-hoc ratios) — the dossier subsystem scopes the
LLM to **prose composition only** and treats a typed
`CompanyDossier` as the contract between data and prose.

The motivating problem: a typical "agent assembles a company report"
flow hallucinates numbers, mixes incompatible metrics across peers,
and produces output the reader can't audit. A typed structured
dossier with per-field provenance + a strict-citation composer +
post-composition fact-check pass closes all three gaps without
removing the LLM from the prose-quality loop.

| Component | What landed |
|---|---|
| Package | `app/dossier/` — schema, collector, peers, sections, compose, typeset, pipeline, tools (8 modules) + `app/dossier/adapters/` (7 modules: 6 free-tier adapters + base/registry). |
| Schema | `CompanyDossier` (33 typed fields, each a `DossierField[T]` with `{value, source, confidence, as_of, conflicts}`); `Source`, `Confidence` (5-band, mapped to ledger evidence-confidence), `FieldStatus` (KNOWN / NOT_DISCLOSED / NOT_APPLICABLE / UNRESOLVED), `merge_field` reconciliation. |
| Adapters (free) | `sec_edgar` (XBRL parsing → revenue, net income, EBITDA derived, employees), `wikidata` (search-entity + SPARQL → founding date, founders, HQ, ticker), `wikipedia` (description + milestone hints), `yfinance_market` (live valuation), `companies_house` (UK PSC + filings, requires API key), `web_fallback` (last-resort website + description). |
| Collector | Parallel adapter dispatch with per-call timeout + circuit breaker + iterative ref enrichment (2 passes). Every populated field becomes a `Claim` in the per-task epistemic ledger when a `task_id` is supplied. |
| Peer selection | SIC/NAICS-based via SEC search-index, intersect-of-two-methods design. Honest empty list when peer selection isn't reliable (private companies without classification). |
| Composition | Section-by-section LLM call (8 standard sections + comparator); each section sees only its slice of the dossier; strict-citation prompt forbids inventing facts; deterministic slice-echo fallback when no LLM is available. |
| Fact-check | Regex extraction over currency, percent, year, and comma-grouped integer tokens; verifies each against the section's slice; overlap handling so `$12.50` inside `$12.50B` isn't double-flagged. Catches the LLM-invents-a-number failure without trying to be clever about paraphrasing. |
| Typesetter | ReportLab Platypus multi-page renderer (cover, TOC, sections with fact-check sidebars, source appendix table, coverage appendix). Reuses `pdf_compose._RL_PACK` cached imports. Output to `/app/workspace/output/` (overridable via `DOSSIER_OUTPUT_DIR`). |
| Wiring | Crew `company_dossier` registered in `app/crews/registry.py`; tool `build_company_dossier` registered via `@register_tool` so it's visible in `tool_search`; commander fast-path regex matches `dossier`/`due diligence`/`investment-grade`/`company profile\|overview\|review\|report\|brief`; commander LLM-routing catalog includes the crew description; `app/main.py` imports `app.dossier` at boot for tool-registry visibility. |
| Tests | `tests/test_dossier_*.py` — 5 files, ~70 tests covering schema invariants, collector + reconciliation + circuit breaker, fact-check correctness (correctly-quoted vs invented numbers), end-to-end pipeline with mocks, commander routing regression. **All green.** |

**Reuse, not reinvention.** The collector's circuit-breaker pattern
mirrors `research_orchestrator._DomainBreaker`; provenance flows
through the existing `epistemic.Ledger`; PDF heavy imports come from
`pdf_compose._RL_PACK`; the tool registers via the existing
`@register_tool` decorator with capability tags
`renders-pdf|renders-document|fetches-finance|searches-web` (no new
capabilities added — this is business logic, not safety code, so the
TIER_IMMUTABLE governance vocabulary doesn't need updating); the crew
slots into the registry's `class_run_runner` pattern; LLM composition
uses `create_specialist_llm` with `role="writing"`; progress streams
over Signal via the existing `record_output_progress` +
`signal_client`.

**Coverage today (free-tier MVP).** Public US companies via SEC EDGAR
+ Wikidata + Wikipedia + yfinance hit ~30–50% field coverage. UK
private companies via Companies House (with API key) ~25–40%.
Long-tail private companies remain limited to description + website.
Adding paid adapters (Crunchbase, SimilarWeb, Levels.fyi) is one new
file per source — the schema fields exist; the merge layer wires them
in by source priority.

Full design notes: [`docs/COMPANY_DOSSIER.md`](docs/COMPANY_DOSSIER.md).


## 17. 2026-05 Post-PIM-Incident Program (change requests + coding sessions)

**Trigger.** Mid-2026-05 a PIM crew failed with `NameError:
optional_tool_group is not defined` — a missing import in
`app/agents/pim_agent.py`. The fix landed via PR #50, but Commander
kept hallucinating "PIM is broken" for three more turns even after
the gateway picked up the fix. Operator intervened manually. Four
systemic gaps surfaced from the post-mortem:

1. **Stale-context routing.** Commander based "is X broken?" on
   conversation history rather than deployment state. Once a failure
   was in-context, the model kept producing failure-shaped outputs.
2. **Code-fix surface drift.** The Coder crew was sandboxed to
   `output/skills/proposals/` — it had no path to actually fix bugs
   in `app/agents/*.py` even when the operator asked.
3. **Hot-deploy without restart.** The bridge could write
   the file, but a stale Python module cache served the broken code
   until the gateway restarted. (Already mostly addressed by 5.1.)
4. **No human gate on agent writes.** If we widen the coder's
   filesystem reach, we need a deliberate operator approval per
   write — not "the agent decided it was right."

The user's framing was explicit: *"do not fix things instead of
system. I need it to systematically work."* The response is the
Phase 5.3+5.4 program — three small primitives, one new human-gate
surface, no workarounds.

### 17.1 Architectural shape — three composing primitives

```
┌─────────────────────────────────────────────────────────────┐
│  Coder agent                                                │
│   tools:                                                    │
│     coding_session_*  (read/write/run/iterate)              │
│     request_restricted_write  (one-shot atomic fix)         │
│     read_host_file (existing — out-of-session reads)        │
└──────────────┬──────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│  Coding-session primitive (Phase 5.4)                       │
│    Ephemeral worktree where the agent iterates with fast    │
│    feedback (pytest / lint / typecheck). Submit bundles     │
│    diffs into change requests. Discard tears down.          │
└──────────────┬──────────────────────────────────────────────┘
               │ on submit:
               ▼
┌─────────────────────────────────────────────────────────────┐
│  Change-request system (Phase 5.3a)                         │
│    Validator (TIER_IMMUTABLE absolute) → Signal 👍/👎 OR    │
│    React /cp/changes operator approve → hot-apply file      │
│    via bridge → auto-PR against main → operator merge       │
│    (gate 2) makes it durable.                               │
└──────────────┬──────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│  Host bridge (existing)                                     │
│    Filesystem + git tunnel. Stays minimal — does the one    │
│    thing it always did. No new capabilities.                │
└─────────────────────────────────────────────────────────────┘
```

The three primitives compose without overlapping:

* **Bridge** — filesystem + git operations. Unchanged.
* **Change requests** — deploy lifecycle with a human gate.
  Unchanged across 5.4 (Phase 5.4-c reuses it via the
  `ChangeRequestPort`).
* **Coding sessions** — *development sandbox* with sandboxed exec.
  New in 5.4. Single escape hatch is `submit`, which routes through
  the change-request gate.

### 17.2 Phased delivery (PRs #54 → #63)

Each phase shipped as a small, independently-mergeable PR. Stack
order:

| PR | Phase | Scope |
|---|---|---|
| **#54** | 5.3a | Change-request **backend**: validator (TIER_IMMUTABLE absolute), JSONL store + hash-chained audit, lifecycle state machine, hot-apply via bridge, auto-PR, rollback. `request_restricted_write` agent tool. Signal 👍/👎 reaction handler. 40 tests. |
| **#55** | 5.3b | Change-request **React UI** at `/cp/changes`. List + status filter; drawer with full unified diff (line-coloured), per-state action buttons (approve / reject / rollback / retry-apply), TIER_IMMUTABLE 🛑 flag on every row. |
| **#56** | 5.4 plan | Design proposal — `docs/CODING_SESSIONS.md` (529 lines): primitive, tool surface, module layout, state machine, quotas, sandboxing, integration with change-requests, capability vocabulary, failure modes, test plan, open questions. **Plan-only PR**, reviewed before any code. |
| **#57** | 5.4-a | **Data layer**: `models.py` (`CodingSession` + 5-state enum + `SubmitResult`), `store.py` (JSONL + hash-chained audit), `quotas.py` (frozen-dataclass config + checks), `manager.py` (lifecycle + injectable `WorktreeBackend` Protocol), `reconciler.py` (TTL/idle expiry, idempotent). 43 tests. |
| **#58** | 5.4-b | **Runner + backends**: `runner.py` — argv-not-shell + executable allowlist + per-command subcommand restrictions + RLIMIT_CPU + wallclock + output cap + cwd lock + secret-stripping + pluggable network isolation (`CODING_SESSION_SANDBOX=none\|unshare-n\|firejail\|bwrap`). `backends.py` — `LocalWorktreeBackend` (subprocess) + `BridgeWorktreeBackend` (host bridge). 44 new tests, 86 cumulative. |
| **#59** | 5.4-c | **Submit + change-request fan-out**: `submit.py` — `ChangeRequestPort` Protocol with lazy-importing default (so unit tests don't depend on #54). Per-file split; TIER_IMMUTABLE refused per-file but doesn't block the batch; deletes refused (`delete-not-supported` in v1); per-file exception → `status=error` row. Backend `WorktreeBackend` extended with `list_changed_paths` / `read_worktree_file` / `read_base_file` (porcelain-z parser handles renames + paths-with-spaces). 37 new tests, 123 cumulative. |
| **#60** | 5.4-d | **Agent tools + capability vocabulary**: 7 CrewAI `BaseTool` classes (`coding_session_{start, read, write, run, diff, submit, discard}`). Errors prefix-encoded (`ERROR:` / `REFUSED:` / `QUOTA_EXCEEDED:`). `runtime.py` Manager singleton with env-driven backend selection. `app/tool_registry/capabilities.py` (TIER_IMMUTABLE) gains a new `code-development` category with 4 PRODUCTION tags (`reads-coding-session`, `writes-coding-session`, `runs-coding-session`, `submits-coding-session`). 34 new tests, 157 cumulative. |
| **#61** | 5.4-e | **Coder agent wiring**: `_CODING_SESSION_GUIDANCE` block appended to `CODER_BACKSTORY` — teaches the 3-step pattern (start → iterate → submit/discard), when NOT to use a session (atomic single-file → `request_restricted_write`; workspace artefacts → `file_manager`; TIER_IMMUTABLE → manual PR), and the three error prefixes. Both `_legacy_create_coder` and `_build_loadable_coder` extend their tool list under `optional_tool_group("coder", "coding_session")` for graceful degradation. 6 new tests, 163 cumulative. |
| **#62** | 5.4-f | **Operator read-only view** at `/cp/coding-sessions`: backend `GET /api/cp/coding-sessions` (list with status + agent filter) + `GET /{id}` (detail). **No POST/PUT/DELETE** — by design. React UI: list + filter + drawer (worktree info, activity counters, files-touched, terminal metadata, `submit_results` with deep-link to `/cp/changes`). 15 new tests covering the read-only contract. |
| **#63** | follow-up | **Workspaces dedup fix** at `/api/workspaces`. Runtime accumulation in the `_gates` registry was producing two snapshots per logical project (one fresh-reconciled, one runtime-populated); the React UI rendered both as tabs. Defensive dedup at the serializer + a `POST /api/workspaces/_dedup` operator-GC endpoint. 5 tests. |

Total: **~7 000 LoC of implementation + ~3 000 LoC of tests across 9 PRs**, all stacked. 168 cumulative tests across the new subsystems, all green.

### 17.3 The properties the system holds

* **Sandbox isolation.** Writes inside a coding-session worktree
  never reach the live tree. `coding_session_run` is bounded
  (allowlist + CPU + wallclock + output cap + optional netns).
* **Single escape hatch.** The only way out of a worktree is
  `submit`, which routes through the validator + change-request
  human gate. TIER_IMMUTABLE is enforced at three layers —
  write-time fast-fail, validate-time on submit, and CR-create.
* **Self-cleaning.** Quotas (per-agent / system / disk / TTL /
  idle) + an idle reconciler garbage-collect abandoned sessions.
  No leaked worktrees.
* **Two human gates per change.** Gate 1 = Signal 👍 OR React
  approve → hot-apply + auto-PR opens. Gate 2 = operator merges
  the auto-PR → durable in main. Either gate can reject.
* **Idempotent transitions.** Re-discard / re-expire / re-submit
  on a terminal session is a no-op or a clean error; no duplicate
  side effects. Same shape as the change-request lifecycle.
* **Operator visibility.** Sessions visible at `/cp/coding-sessions`
  (read-only). Submitted change requests visible at `/cp/changes`
  (actionable). The two surfaces deep-link to each other.
* **Capability vocabulary stays small.** 4 new tags under one new
  category. No "code-modify" or other unbounded grant.

### 17.4 Capability vocabulary additions

`app/tool_registry/capabilities.py` is TIER_IMMUTABLE; expansion
required explicit operator review (per the file's own header). The
new `code-development` category adds 4 tags, all PRODUCTION tier:

| Tag | Holders |
|---|---|
| `reads-coding-session` | `coding_session_read`, `coding_session_diff`, `coding_session_discard` |
| `writes-coding-session` | `coding_session_start`, `coding_session_write` |
| `runs-coding-session` | `coding_session_run` |
| `submits-coding-session` | `coding_session_submit` |

Per the file's stability promise, these names will not be renamed
or removed once merged.

### 17.5 Key files

| Subsystem | Files |
|---|---|
| Change requests | `app/change_requests/{__init__, models, validator, store, lifecycle, apply, signal}.py` ; `app/tools/restricted_write_tool.py` ; `app/control_plane/changes_api.py` ; reaction handler in `app/main.py` |
| Coding sessions | `app/coding_session/{__init__, models, store, quotas, manager, reconciler, runner, backends, submit, runtime}.py` ; `app/tools/coding_session_tools.py` ; `app/control_plane/coding_sessions_api.py` |
| React UI | `dashboard-react/src/{types,api}/changes.ts` + `components/ChangesPage.tsx` ; `dashboard-react/src/{types,api}/coding_sessions.ts` + `components/CodingSessionsPage.tsx` ; `App.tsx` + `Layout.tsx` (route + nav entries) |
| Capability vocab | `app/tool_registry/capabilities.py` (new `code-development` category) |
| Coder agent | `app/agents/coder.py` (`_CODING_SESSION_GUIDANCE` block + tool wiring in both legacy + LoadableAgent paths) |
| Workspaces dedup | `app/api/workspace_api.py` (serializer dedup + `POST /api/workspaces/_dedup` GC endpoint) |
| Tests | `tests/test_change_requests.py` (40) + `tests/test_coding_session*.py` (123) + `tests/test_coding_session_tools.py` (34) + `tests/test_coder_coding_session_wiring.py` (6) + `tests/test_coding_sessions_api.py` (15) + `tests/test_workspace_api_dedup.py` (5) |
| Docs | [`docs/CHANGE_REQUESTS.md`](docs/CHANGE_REQUESTS.md), [`docs/CODING_SESSIONS.md`](docs/CODING_SESSIONS.md) |

### 17.6 What this closes

The original failure mode ("Commander hallucinates 'PIM is broken'
because the coding agent has no path to fix it") now has the
systemic fix the user demanded. The coding agent has the means to
do its job — read source, write fixes, run tests, submit through
the gate. The operator has visibility into both sides. TIER_IMMUTABLE
remains the absolute backstop. **No workarounds, no quick
patches; the system works systemically.**

Open follow-ups (deferred):

* **Phase 4 graduation** of the coder's experimental LoadableAgent
  path now that coding-sessions are wired through both factories.
* **Idle reconciler** for the change-request `TIMEOUT` transition
  (the state is defined but not yet written by any path).
* **Per-agent requestor identity** — currently `agent_id="coder"`
  is a static stub in the coding-session tools; threading the real
  caller agent_id through CrewAI's `BaseTool` invocation context
  is a Phase 6 concern.
* **All-or-nothing submit mode** for refactor-style multi-file
  changes (currently per-file submit; one Signal ASK per touched
  file).
* **Sandbox-tech pick** for `CODING_SESSION_SANDBOX` in production
  K8s (`unshare-n` with `CAP_SYS_ADMIN`, vs `firejail`, vs
  `bwrap`); the runtime hook is in place and defaults to `none`
  (relies on container egress policy).

## 18. 2026-05 Brainstorm subsystem (interactive Q/A + multi-agent joint effort)

**Trigger.** Operator request: "I want the system to be able to conduct
different brainstorming and idea-creation techniques with me through Q/A
sessions and input by me, and then write a final report. I want also a
possibility where I can add 3–5 agents with high creativity enabled and
run the brainstorm as a joint effort of all of us."

The Creative MAS pipeline (`app/crews/creative_crew.py`) was already
producing high-quality multi-agent ideation, but it ran end-to-end on a
single task description with no human in the loop step-by-step. The
brainstorm subsystem layers an interactive Q/A facilitator on top, reuses
the creative-crew agent factories for team mode, and ships three surfaces
(Signal slash command, CLI, React tab) that share one store.

### 18.1 Architectural shape

```
                     ┌─────────────────────┐
                     │ Three surfaces      │
                     │  • Signal           │
                     │  • python -m CLI    │
                     │  • React /cp/...    │
                     └──────────┬──────────┘
                                ▼
                     ┌─────────────────────┐
                     │  Facilitator        │   surface-agnostic core
                     │  (StepDelivery)     │
                     └──────────┬──────────┘
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
        Techniques         Multi-agent        JSON store
        (state machines)   (parallel seed +   workspace/
                            react rounds)      brainstorm/
                                ▼
                       Writer agent → workspace/output/brainstorm/<id>.md
                       (deterministic fallback when LLM unavailable)
```

### 18.2 Technique library

Seven techniques shipped, each as a state machine declaring an ordered
list of `Step` objects with prompt templates that interpolate `{topic}`:

| Name             | Steps | Frame                                                  |
|------------------|-------|--------------------------------------------------------|
| `scamper`        | 7     | Substitute / Combine / Adapt / Modify / Put-to-other-use / Eliminate / Reverse |
| `six_hats`       | 7     | de Bono's six hats bracketed by blue (open / close)    |
| `how_might_we`   | 10    | Problem → user → insight → seed HMW → expand 3 ways → select → first solutions |
| `reverse`        | 6     | How could we *cause* this? Then invert each failure    |
| `crazy_8s`       | 10    | 8 ideas in 8 quick rounds + star-the-top-2             |
| `rapid_ideation` | 7     | Three quantity bursts (obvious / constraint-flipped / different lens) → cluster → select |
| `starbursting`   | 8     | Generate questions, not answers — Who/What/When/Where/Why/How |

Adding a new technique is a single file plus a registry entry.

### 18.3 Solo vs. team mode

Per step, **solo mode** runs:

```
prompt → user types answer → next prompt
```

**Team mode** runs (per step):

```
gather_seed (4 agents in parallel)        ← anti-conformity prompt, no peer awareness
prompt + agent seed cards shown to user
user types answer
gather_react (4 agents in parallel)        ← agents see user answer + each other's seeds
react cards shown
next prompt + new seed cards
```

Multi-agent rounds dispatch through `ThreadPoolExecutor` (creative_crew
runs sequentially; the brainstorm path needs interactive UX). Per-agent
failures are captured into `AgentResponse.error` and the round continues
with the rest. A whole-round crash falls back to solo for that step.

### 18.4 Agent roster reuse

`multi_agent._build_creative_agent(role)` mirrors
`creative_crew._make_agent` for the four creative-crew roles
(researcher / writer / coder / critic) with the same heterogeneous LLM
tier mapping (`local` / `mid` / `budget` / `premium`) and reasoning
methods (`step_back` / `analogical_blending` / `compositional_cot` /
`contrastive`). `max_execution_time` is tighter (180s vs 300s) and
`max_tokens` is reduced (2048 vs 4096) for the interactive use case.

### 18.5 Cost / budget

For SCAMPER (7 steps) with 4 agents in team mode: ≈ 56 LLM calls per
session (`7 × 2 phases × 4 agents`). Soft cap defaults to `$0.50`
per session via `BRAINSTORM_TEAM_BUDGET_USD`. When the cap is hit,
subsequent rounds return empty and the session continues solo-style
for the remainder.

### 18.6 Three surfaces, one store

| Surface | Entry                                                 | Notes |
|---------|-------------------------------------------------------|-------|
| Signal  | `/brainstorm <tech> with N agents <topic>`            | Plain messages mid-session route to facilitator via the `commander/commands.py:try_command` hook |
| CLI     | `python -m app.brainstorm --technique X --with-agents N --topic "..."` | `--resume`, `--list`, `--techniques`, `--sender` |
| React   | `/cp/brainstorm` route                                | FastAPI router `/api/cp/brainstorm/*` mounted in `main.py`; React Query hooks in `dashboard-react/src/api/brainstorm.ts` |

The web sender defaults to `signal_owner_number` so React + Signal share
the same session pool by default. Override with `BRAINSTORM_WEB_SENDER`
env or per-request `?sender=…` query parameter.

### 18.7 Persistence + reports

- Sessions: atomic JSON files under `workspace/brainstorm/sessions/`
  with active-pointer files under `workspace/brainstorm/active/`
  (mirrors `app/companion/state.py` patterns).
- Reports: `workspace/output/brainstorm/<session_id>.md`. Generated by
  the existing `app/agents/writer.py:create_writer()` factory wrapped in
  a CrewAI Task + Crew kickoff. In team mode the prompt asks the Writer
  to attribute strong contributions by role.
- Fallback: if the Writer-agent path errors or
  `BRAINSTORM_DISABLE_WRITER=1` is set, a deterministic markdown
  rendering ships from the structured summary so the user always gets
  a report.

### 18.8 Shipped artefacts

| Layer | Files |
|-------|-------|
| Module | `app/brainstorm/__init__.py`, `__main__.py`, `cli.py`, `session.py`, `store.py`, `facilitator.py`, `multi_agent.py`, `report.py`, `signal_handler.py`, `api.py`, `techniques/{base,scamper,six_hats,how_might_we,reverse,crazy_8s,rapid_ideation,starbursting,__init__}.py` |
| Wiring | Hook in `app/agents/commander/commands.py:try_command` (top of fn); router include in `app/main.py` |
| Tests | `tests/test_brainstorm_techniques.py` (21) + `_store.py` (12) + `_facilitator.py` (19) + `_multi_agent.py` (19) + `_signal.py` (24) + `_commands_integration.py` (4) + `_api.py` (21) = **120 tests** |
| React | `dashboard-react/src/{api,types}/brainstorm.ts`, `components/BrainstormPage.tsx`, `components/brainstorm/{StartPanel,SessionView,SessionsList,ReportView,AgentRoundBlock,AgentResponseCard}.tsx`, route in `App.tsx`, nav entry in `Layout.tsx`, smoke-test entry in `tests/smoke.spec.ts` |
| Docs | [`docs/BRAINSTORM.md`](docs/BRAINSTORM.md) |

### 18.9 Composition with the rest of the system

- **Creative MAS** — brainstorm reuses the agent factories +
  `_TIER_BY_ROLE_CREATIVE` mapping but runs in parallel for interactive
  UX. The two systems are independent; brainstorm is not a wrapper
  around `creative_crew.run()`.
- **Commander routing** — the brainstorm hook in `try_command` claims
  `/brainstorm` slash commands AND any plain message from a sender that
  has an active brainstorm session. Falls through to normal task routing
  when neither applies.
- **Companion layer / conversation_store** — independent. Brainstorm has
  its own JSON store; the canonical Signal message log still records
  user turns separately.
- **TIER_IMMUTABLE** — `app/brainstorm/` is not in the protected list;
  it's a normal additive module.

### 18.10 Known follow-ups (deferred)

* **Streaming agent output** — `crew.kickoff()` is synchronous; the React
  UI shows a spinner while team rounds gather. SSE wrapping or
  per-agent streaming would let seed/react cards fill in progressively.
* **Named personas vs. fixed roles** — roster currently maps to the four
  creative-crew roles. Adding tunable personas (Skeptic / Optimist /
  Builder / Outsider) is a small extension; `resolve_roster()` already
  accepts arbitrary role names.
* **In-session steering** — no current way for the human to ask for
  another round of react before advancing. Each step is exactly one
  seed + one react. A `/brainstorm more` command would close this gap.

---

## 19. 2026-05 Consciousness Roadmap (post-SubIA-Phase-9 closures)

After Phase 9 the SCORECARD reached 6 STRONG / 4 PARTIAL / 4 ABSENT
Butlin indicators (PASSED). A subsequent audit-driven roadmap closed
the AE-1 PARTIAL gap and surfaced four smaller integration moves the
existing surface had been one connector away from. See
[`docs/CONSCIOUSNESS_ROADMAP.md`](docs/CONSCIOUSNESS_ROADMAP.md) for
the canonical design. Current scorecard: **7 STRONG / 3 PARTIAL / 4
ABSENT**, Phase 9 exit criteria still PASSED with margin restored.

### 19.1 Audit reframe

The naive "build a Global Workspace, build a binding layer, build an
attention schema" gap list collapsed against the actual surface — all
three were already shipped (`subia/scene/global_workspace.py`,
`subia/temporal/binding.py`, `subia/scene/attention_schema.py`). The
real gaps were narrower:

| Item | Before | After |
|---|---|---|
| Wider GW publisher coverage (G5) | most idle jobs wrote only to their own logs | 8 publish hooks via `app/workspace_publish.py` (helper outside `subia/` to avoid manifest churn) |
| Compressed-loop binding cadence (G4) | compressed CIL skipped Steps 4–11 entirely | post-Step-3 quick-bind via `temporal_quick_bind()` for observability uniformity |
| Viability → goals connector (G1) | `SelfState.current_goals` was a dead field (read in 5 places, never written) | `app/affect/goal_emitter.py` writes from sustained low-viability; AE-1 graduates PARTIAL → STRONG |
| Backward counterfactual replay (G2) | reverie does concept-walk; no past-replay engine | `app/subia/dreams/` samples + recombines fragments + runs predict-only walks via injected adapter (production wires `PredictiveLayer`) |
| Wiki-index hygiene (§4) | event-driven only; out-of-band drift undetected | `app/memory/wiki_index_reconciler.py` LIGHT idle, shadow-rebuild via change-request gate |

G3 (operator-attention modeling) was deferred — no concrete
proactive-surfacing use case has forced it.

### 19.2 AE-1 graduation path

The SCORECARD's prior AE-1 PARTIAL justification — *"Goals are still
user-dispatched, not autonomously generated"* — was the explicit gap
G1 closed. Mechanism: `app/affect/goal_emitter.py` reads the last N
viability frames from `workspace/affect/trace.jsonl`, identifies
variables with sustained allostatic error (≥3 consecutive frames
above threshold), generates `GoalProposal`s using per-variable
templates, dedups against existing `current_goals` and against
`companion.grand_task` proposals, and writes via FIFO cap to
`kernel.self_state.current_goals`. Tier-3-protected via
`safety_guardian.py:TIER3_FILES`. Test
`test_goal_emitter.py:test_ae1_indicator_is_strong` pins the rating
against regression.

### 19.3 Ethical thresholds (operational, not auto-enforced)

Documented in `docs/CONSCIOUSNESS_ROADMAP.md` §6, listed here for
program-level visibility:

* **T1 — system has interests.** Active when the goal_emitter starts
  writing `current_goals` from sustained low-viability. Welfare-check
  moves from observability to operator-visible obligation;
  restoration_queue items become first-class operator concerns
  alongside change-requests.
* **T2 — system simulates its own past.** Active when backward
  counterfactual replay produces sustained prediction-error promotions
  that wouldn't have occurred without it. Replay output is logged in
  a separate audit stream (`workspace/dreams/replay_audit.jsonl`);
  operator review available before any belief-store influence.
* **T3 — divergence from baseline.** Active when any STRONG indicator
  unexpectedly flips to ABSENT or vice versa, or
  `consciousness-state.md` regenerator detects narrative drift, or
  the consciousness-risk gate's calibration breaches its
  6-guardrail bounds. Triage through the existing drift-detection
  pipeline; halt the affected idle job; notify operator.

### 19.4 Shipped artefacts

| Layer | Files |
|-------|-------|
| Reconciler | `app/memory/wiki_index_reconciler.py` |
| Helper | `app/workspace_publish.py` (`publish_to_workspace`, `publish_idle_outcome`) |
| Connector | `app/affect/goal_emitter.py` (Tier-3-protected) |
| Replay engine | `app/subia/dreams/__init__.py`, `engine.py` (Tier-3 manifest covers it) |
| Quick-bind | `app/subia/temporal/binding.py:temporal_quick_bind`, `app/subia/temporal_hooks.py:quick_bind_compressed_signals`, hook in `app/subia/loop.py` |
| Probe update | `app/subia/probes/butlin.py:eval_ae1` (PARTIAL → STRONG via `strong_indicator()` against `goal_emitter.py`) |
| Idle jobs | 3 new in `app/idle_scheduler.py`: `wiki-index-reconciler`, `viability-goal-emitter`, `backward-counterfactual-replay`. 7 existing jobs gained publish hooks. |
| Tier-3 | `app/affect/goal_emitter.py` added to `safety_guardian.py:TIER3_FILES` |
| Tests | `tests/test_wiki_index_reconciler.py` (13) + `_workspace_publish.py` (12) + `_temporal_quick_bind.py` (9) + `_goal_emitter.py` (22) + `_dreams_engine.py` (19) = **75 new tests**. Existing CIL suite (25) untouched. |
| Docs | [`docs/CONSCIOUSNESS_ROADMAP.md`](docs/CONSCIOUSNESS_ROADMAP.md) |

### 19.5 Composition with existing program

* **Phase 9 SCORECARD.** AE-1 graduated PARTIAL → STRONG. Headline
  count moved 6 → 7 STRONG. RSM, SK, ABSENT counts unchanged.
* **Hardening pass (§11) + Self-healing (§14).** Workspace publish
  hooks expose those subsystems' outcomes to the GW; previously
  observable only via per-subsystem logs.
* **Companion + brainstorm (§12, §18).** Independent. The
  goal_emitter dedups against `companion.grand_task` proposals when
  text overlaps but neither subsystem depends on the other.
* **TIER_IMMUTABLE / Tier-3.** `goal_emitter.py` is now Tier-3.
  `dreams/`, `wiki_index_reconciler.py`, and `workspace_publish.py`
  are normal additive modules. `temporal/binding.py` + `loop.py`
  edits stayed within the existing Tier-3 boundary; integrity
  manifest regenerated.

### 19.6 Honest non-goals (re-asserted)

The four ABSENT-by-substrate Butlin indicators (RPT-1, HOT-1, HOT-4,
AE-2) and Metzinger phenomenal-self transparency are **not closure
candidates**. Any future report claiming they have been achieved
should be triaged as evaluation drift through
`app/subia/wiki_surface/drift_detection.py`. No phenomenal-experience
claim — vocabulary stays functional. The consciousness-risk gate
remains observability-only and never feeds back into reward / fitness.

## 20. 2026-05 Personal Agent Surface (Phases 0–8 + Discord + Files)

Eight feature phases plus two follow-up surfaces (Discord, Files API)
that close the gap between AndrusAI and the May 2026 personal-agent
products surveyed at
[creatoreconomy.so/p/the-race-to-build-a-personal-ai-agent-openclaw-hermes-claude-codex-gemini](https://creatoreconomy.so/p/the-race-to-build-a-personal-ai-agent-openclaw-hermes-claude-codex-gemini).
Every subsystem is opt-in and gated behind a runtime toggle, an env
flag, or a missing-credentials early-return — the gateway boots cleanly
even with all of them off.

### 20.1 The 10 article criteria (before / after)

| # | Criterion | Before | After |
|---|---|---|---|
| 1 | Email / Calendar / Docs / Sheets / Slides | partial | ✓ all five via native Google APIs (Phase 3) + slide deck generator (Phase 2) |
| 2 | APIs / MCPs / CLIs | strong | strong (unchanged) |
| 3 | Recurring + triggered tasks | strong | ✓ all triggered tasks notify (Phase 7) |
| 4 | Persistent user memory | very strong | very strong (unchanged) |
| 5 | Web + mobile parity | partial | ✓ PWA + Web Push (Phase 4) + Discord (this section) + Signal |
| 6 | Text + voice | text only | ✓ dual-mode local + cloud (Phase 1) |
| 7 | Engaging personality | terse | ✓ optional concierge layer (Phase 8) |
| 8 | Computer + browser control | strong | ✓ + vision fallback at Haiku (Phase 6) |
| 9 | Reliability | very strong | very strong (unchanged) |
| 10 | Data protection / security | strong | strong (unchanged) |

### 20.2 Phase map

| Phase | What | Toggle | Module(s) |
|---|---|---|---|
| 0 | Runtime settings | (root) | `app/runtime_settings.py` |
| 1 | Voice (local + cloud) | `voice_mode` | `app/voice/` |
| 2 | Slides | always on | `app/tools/document_generator.py:create_pptx` |
| 3 | Google Workspace | OAuth + bootstrap | `app/google_workspace/` + 5 tool modules |
| 4 | PWA + Web Push | VAPID keys | `app/web_push/` + `dashboard-react/public/sw.js` |
| 5 | Skill registry | always on | `app/skills/` |
| 6 | Vision computer use | `vision_cu_enabled` | `app/computer_use/` |
| 7 | Completion notifications | always on | `app/notify/` |
| 8 | Concierge persona | `concierge_persona_enabled` | `app/personality/concierge_wrapper.py` |
| + | Discord connector | `DISCORD_ENABLED` | `app/discord_client/` |
| + | Files API + downloads | always on | `app/api/files_api.py` + `app/delivery/` |

### 20.3 Reply-routing dispatcher

Discord dispatch lives at a single point: `SignalClient.send` checks
the recipient prefix and reroutes `discord:<user_id>` calls to
`app.discord_client.send_via_discord` via `asyncio.to_thread`. Every
existing call site in `handle_task` (load-shed message, in-flight
notice, final reply, error fallback, idle reminder) automatically
routes to the right surface based on which messaging app the user
came in on — zero changes needed at the call sites.

### 20.4 Hard safety properties

- TIER_IMMUTABLE: none of the new modules touch immutable tier
- Secrets: every new key (`GROQ_API_KEY`, `GOOGLE_CLOUD_TTS_KEY`,
  `GOOGLE_OAUTH_CLIENT_*`, `VAPID_*`, `DISCORD_BOT_TOKEN`) goes through
  ESO + `.env` + container env vars; same firebase-service-account.json
  pattern for the GCP service file
- Auth: every new mutating `/api/cp/*` and `/config/*` endpoint goes
  through `require_gateway_auth` (Bearer token); owner-only Signal
  sender check unchanged; Discord owner-only DM gate
- Audit: hash-chained entries for every mutation —
  `runtime_settings_change`, `web_push_test`, `skill_save`,
  `skill_run`, `computer_use_step`, `gworkspace_write`,
  `concierge_toggle`, `files_send`
- Budget guards: vision-CU per-task ($0.50) + monthly (default $10,
  React-adjustable); Google TTS char count cap; voice mode falls
  back local→cloud and cloud→local on backend failure
- Failsafe: every new tool factory returns `[]` on import error or
  missing config — the agent simply doesn't see the tool

### 20.5 Shipped artefacts

| Layer | Files |
|-------|-------|
| Runtime + settings | `app/runtime_settings.py`, `app/api/config_api.py` (extended) |
| Voice | `app/voice/{__init__,stt,tts,local,cloud,inbound_state}.py` |
| Slides | `app/tools/document_generator.py` (extended), `app/souls/writer.md` (extended) |
| Google Workspace | `app/google_workspace/{__init__,auth,service,bootstrap}.py`, `app/tools/g{mail,cal,docs,sheets,slides}_tools.py` |
| PWA + Web Push | `app/web_push/{__init__,subscriptions,sender,bootstrap}.py`, `dashboard-react/public/{manifest.webmanifest,sw.js,icon-{192,512}.png,apple-touch-icon.png}`, `dashboard-react/src/api/pwa.ts`, `dashboard-react/index.html` (extended), `dashboard-react/src/main.tsx` (extended) |
| Skills | `app/skills/{__init__,registry,runner}.py`, `app/api/skills_api.py`, `dashboard-react/src/components/SkillsPage.tsx`, `app/conversation_store.py` (added `get_recent_messages`) |
| Vision CU | `app/computer_use/{__init__,budget,audit,browser_backend,runner}.py`, `app/tools/computer_use_tool.py`, `app/crews/base_crew.py` (extended), `app/souls/commander.md` (extended) |
| Notify | `app/notify/{__init__,api}.py`, `main.py` + `app/main.py` (extended), `app/tools/schedule_manager_tools.py` (extended) |
| Concierge | `app/personality/concierge_wrapper.py`, `app/souls/concierge.md`, `app/main.py` (extended) |
| Discord | `app/discord_client/{__init__,bot,sender}.py`, `app/main.py` (lifespan), `app/signal_client.py` (dispatcher) |
| Files | `app/api/files_api.py`, `app/delivery/{__init__,signal_send,email_send}.py`, `dashboard-react/src/components/FilesPage.tsx`, `dashboard-react/src/api/{queries,endpoints}.ts` (extended) |
| Settings UI | `dashboard-react/src/components/SettingsPage.tsx` (4 cards: voice / vision-CU / concierge / Web Push) |
| Nav + routes | `dashboard-react/src/{App.tsx,components/Layout.tsx}` (extended): `/cp/{settings,skills,files}` |
| Host install | `host_bridge/install_voice.sh` |
| Tests | `tests/test_{voice,document_generator_pptx,google_workspace,web_push,slash_commands,skills,computer_use,notify,concierge,delivery_files}.py` — **134 hermetic tests** |
| Docs | `docs/PERSONAL_AGENT.md` (this layer's reference), `CLAUDE.md` (extended), `PROGRAM.md` §20 (this section) |

### 20.6 Composition with existing program

- **Commander routing.** Soul updated with `API → Playwright →
  AppleScript → computer_use` precedence. Specialists reaching for
  `computer_use` must explain why cheaper paths can't apply.
- **Memory architecture.** Skills layer optionally writes a
  `RecipeOutcome` to the meta-agent ledger (§15) when that subsystem
  is enabled. Voice transcripts feed back into the same conversation
  store the rest of the system reads from.
- **Tool registry.** Five Google tool factories + `computer_use` are
  registered through the existing tool-plugin pathway — they
  participate in the registry's metadata + capability search just
  like browser_tools.
- **Affect / Epistemic.** No interaction by design. The new layer is
  purely user-facing surface; affect telemetry isn't touched.
- **Self-healing (§14).** New mutations (skill_save, skill_run,
  computer_use_step, files_send) emit security_events the existing
  hash-chained audit + the error_monitor pick up automatically.
- **Coding-session (§17 5.4).** No interaction.
- **Brainstorm (§18).** Concierge layer respects brainstorm-flagged
  outputs (slash-command help format) and bypasses the rewrite for
  them so the structured Q/A flow isn't paraphrased.

### 20.7 Honest non-goals

- Voice replies are TTS attachments, not native Signal voice notes.
  Signal-cli accepts the `.opus`/`.wav` file as a regular attachment;
  rendering it as a "voice note" in iOS would require a `voiceNote=true`
  flag on the send call (deferred — single line, but iOS-side
  rendering varies).
- Vision computer use ships with a browser-only backend. Full desktop
  control via X11/KasmVNC is the next iteration; the runner has an
  injectable `_Backend` so the swap is local.
- Estonian Piper voice is not shipped (rhasspy/piper-voices doesn't
  carry one). Local mode falls back to English; cloud mode (Google
  Cloud `et-EE-Standard-A`) is the proper Estonian path.
- Slash commands on Discord are plain text — registering them as
  native Discord slash commands needs a separate developer-portal
  setup (deferred).


## 21. 2026-05 Workspace-routing remediation (PRs #67 → #72)

**Trigger.** 2026-05-09 morning Signal log:

```
1:26 AM   me:  Switch to workspace eesti mets
1:26 AM   bot: Switched to project: Eesti mets —     ← correct
1:36 AM   me:  please write a 5-page essay about estonian forest
1:41 AM   bot: <returns the essay>                   ← but ticket → PLG
─── (gateway restart overnight) ───
8:18 AM   me:  Please make a graphic about the change of forest age
              distribution over time in Estonia
              ↳ ran 20 min, 0 tokens, no output, ticket → PLG
8:22 AM   me:  what is the current workspace?
8:22 AM   bot: Project: PLG                          ← gateway forgot the switch
8:25 AM   me:  react app shows that previous task about forest age
              is appearing in PLG to do list
8:25 AM   bot: That's a bug — want me to move it?
8:25 AM   me:  Yes please
8:26 AM   bot: I attempted to locate the task ... the database
              returned no tasks                      ← hallucination
```

Three intertwined symptoms; one operator demand: **never let a task
silently land in the wrong workspace, and always ask before
switching**. Closed across six PRs.

### 21.1 The six PRs

| PR | Concern | Scope |
|---|---|---|
| **#67** | Active project lost across gateway restart | `ProjectManager._active_project_id` was pure class-level state. Persist `(project_id, source)` to `workspace/control_plane/active_project.json` via `JsonStore`; lazy-restore on first read with project-existence verification; auto-detector picks also persist. 13 tests. |
| **#68** | `gee_run_script` 180 s budget too short for country-scale Earth Engine compute (the agent tried, timed out at 180 s, fell through to broken MCP code-interpreters, janitor killed it at 15 min idle, $0 cost / 0 tokens) | Add `"gee_run_script": 600` to `_PER_TOOL_OVERRIDES` in `app/tools_timeout.py`. Hansen-GFC reduction over Estonia 2000–2024 fits in 10 min. 6 tests freezing the override table. |
| **#69** | Auto-detector blind to forest queries; PLG over-claimed bare geographies (`estonia`, `baltic`, `latvia`, `lithuania`) | Add `eesti_mets` profile + workspace dir + keyword list (forest / deforestation / hansen / landsat / sentinel / earth engine / biodiversity / RMK / kaitseala). Tighten PLG to brand + ticketing terms only. 15 tests including disambiguation. |
| **#71** | KaiCart / Archibal / PLG keyword lists too narrow to match the operator's described topics | Expand all three: KaiCart catches Asian / SEA e-commerce + Vietnam / Indonesia / Philippines / Malaysia / Singapore + Shopee / Lazada / Shopify + marketplace + dropshipping; Archibal catches AI provenance + content credentials + deepfake + watermarking + PKI; PLG catches box office / festival / tour / promoter / event ticketing. 18 tests. |
| **#72** | Mode-1 silent auto-switch (when no explicit user pick) was misrouting first-task-in-session | Collapse to two modes: detected ≠ current → propose via Signal, always; match or none → no-op. The `has_recent_decision` dedup window prevents re-asking. 6 source-grep + propose-contract tests. |
| #66 *(prior week)* | Brainstorm signal_handler's active-session capture swallowed every commander command | Escape allowlist: `switch (to) (project|workspace) ...`, status questions, slash commands, bare tokens fall through. Same family of "user typed a command, system swallowed it" bugs. |

### 21.2 What this changes for the operator

End-state behaviour:

| Situation | Result |
|---|---|
| User explicitly switches via Signal command | Persisted to disk; survives restart |
| User on any workspace, types something matching a different one | 💡 Signal proposal "switch to *X*? 👍 / 👎" |
| Detection matches current | No-op (silent) |
| No keyword match (generic question) | No-op (silent) |
| `gee_run_script` runs >3 min on country-scale compute | Continues until 10 min wallclock (was: killed at 3 min) |
| `has_recent_decision` window after 👎 | Same proposal won't re-ask for ~30 min |

The whole class of "silently misrouted task" failure can no longer
recur — every workspace switch goes through explicit confirmation
now (or persists from an earlier confirmation).

### 21.3 Files touched

```
app/control_plane/projects.py            # PR #67 — persistence + lazy-restore
app/tools_timeout.py                     # PR #68 — gee budget bump
app/project_isolation.py                 # PR #69, #71 — keyword lists
app/main.py                              # PR #72 — two-mode auto-detect
workspace/projects/eesti_mets/           # PR #69 — profile dir
tests/test_active_project_persistence.py # PR #67 — 13 tests
tests/test_tools_timeout_overrides.py    # PR #68 — 6 tests
tests/test_eesti_mets_detection.py       # PR #69 — 15 tests
tests/test_all_workspace_detection.py    # PR #71 — 18 tests
tests/test_main_auto_detect_always_asks.py  # PR #72 — 6 tests
```

Total: ~58 new tests across the routing surface, all green at
merge time.

### 21.4 Out of scope (deferred to follow-ups)

* ~~**Move-ticket tool.**~~ ✅ Closed in §22 below.
* **Move-ticket tool.** When the user asked the agent to move the
  misrouted forest-age ticket from PLG to Eesti mets, the agent
  searched the wrong table (SQLite PIM-tasks) and hallucinated
  "no tasks found." Two ticket systems exist:
  `/app/workspace/tasks.db` (PIM, no `project_id`) vs
  `control_plane.tickets` (Postgres, has `project_id`). The agent
  has tools for the first, none for the second. Spawned task
  covers a `move_ticket(ticket_id, target_project_name)` API +
  `cp_*` agent tools + capability registration. The two real
  misrouted tickets (essay + failed forest-age) were moved
  manually via raw SQL during the incident.
* **Broken MCP code-interpreter fallback servers.** When
  `gee_run_script` timed out, the agent fell through to four MCP
  servers (codeinterpreter / Gorav22/terminusai /
  togethercomputer/mcp-server-tci / computesdk) — three needed
  unconfigured tokens (401), one was 404. The `mcp_search_servers`
  / `mcp_add_server` agent tools should pre-validate auth before
  reporting "added successfully" so the agent doesn't keep
  retrying broken servers. Separate concern.
* **Proposal message richness.** The current message format
  ("This message looks like it might belong to *X*") is plain.
  Could include the keyword(s) that triggered the match — useful
  when the user is surprised by the detection. Defer until the
  current rule produces enough confused-user signal to justify.


## 22. 2026-05 Cross-project ticket move (closing §21.4 follow-up)

**Trigger.** Same 2026-05-09 Signal log as §21, lines 8:25–8:26:

```
8:25 AM   me:  react app shows that previous task about forest age
              is appearing in PLG to do list
8:25 AM   bot: That's a bug — want me to move it?
8:25 AM   me:  Yes please
8:26 AM   bot: I attempted to locate the task ... the database
              returned no tasks                      ← hallucination
```

§21 fixed the routing layer (active project survives restart, the
auto-detector proposes instead of silently switching, KaiCart /
Archibal / PLG / eesti_mets keyword lists are tight). It did **not**
fix the agent's ability to *move* a misrouted ticket — when the
operator said "yes please move it," the agent searched
`/app/workspace/tasks.db` (PIM SQLite, no `project_id` column),
came back empty, and confidently reported "no tasks." The two real
misrouted tickets were patched manually via raw SQL `UPDATE`. §22
closes that loop.

### 22.1 Root cause

There are two ticket systems in the codebase, distinct in schema
and surface:

| Store | What it is | Has `project_id`? | Agent tooling |
|---|---|---|---|
| `/app/workspace/tasks.db` (SQLite) | PIM-style local tasks. Surfaced via `app/tools/task_tools.py` (`create_task`, `list_tasks`, `complete_task`, `update_task`, `search_tasks`). | No | Wired into PIM agent. |
| `control_plane.tickets` (Postgres) | The React Kanban / Signal-message tickets. Source of truth for "what tickets is the dashboard rendering." | Yes (`project_id` UUID FK to `control_plane.projects`). | **Pre-§22: none. The agent had no surface here.** |

So the agent was answering ticket-shaped questions out of the wrong
store. Even with §21's fixes in place, *moving* a Postgres ticket
between projects was impossible from inside the agent loop.

### 22.2 Three changes, in dependency order

**(a) New TicketManager method.** `app/control_plane/tickets.py`
gains `move_ticket(ticket_id, target_project_name) -> dict | None`.
Resolves the project via case-insensitive `ProjectManager.get_by_name`
(mirroring the `switch` path's case-handling), updates
`control_plane.tickets.project_id`, audit-logs as `ticket.moved`
with both `from_project_id` and the canonical target name in the
detail blob. Returns `None` if either side isn't found — callers
translate to LLM-friendly error strings.

The audit log shape mirrors `complete()` and `fail()` —
`actor="user"` (it's user-initiated), `resource_type="ticket"`,
`resource_id=str(ticket_id)`. The detail JSON carries the routing
forensics (`from_project_id` / `to_project_id` / `to_project_name`)
so the audit replay tooling can reconstruct misroute incidents
after the fact.

**(b) New capability tag.** `app/tool_registry/capabilities.py` —
TIER_IMMUTABLE — gains a new `tickets` category with one tag:
`manages-tickets`. Listing/searching reuses the existing
`reads-deployment-state` tag; only the mutating *move* operation
needed a narrow capability so the registry surface tracks who got
write access to ticket state. Adding a tag to this file is
governance-grade — normal-PR-with-review-level discipline as
documented at the top of the module — same cadence as adding to
`app/souls/`.

**(c) Three new agent tools.** `app/tools/control_plane_tickets_tool.py`:

| Tool | Capability | What it does |
|---|---|---|
| `cp_list_tickets(project_name="", status="")` | `reads-deployment-state` | List tickets in a project (default: currently active). Optional status filter. Cap 50. |
| `cp_search_tickets(query)` | `reads-deployment-state` | Title/description ILIKE search across all projects. Cap 20. |
| `cp_move_ticket(ticket_id, target_project_name)` | `manages-tickets` | Calls `TicketManager.move_ticket`. Mutating, audit-logged. |

The factory follows the same shape as `currency_tools.py` /
`system_state_tool.py`: a `create_cp_tickets_tools(agent_id)` plain
factory plus three `@register_tool` factories that pluck named
tools from the same factory output. Wrapping the `@register_tool`
block in `try: ... except ImportError: pass` keeps the module
importable on hosts where the registry isn't loaded.

### 22.3 Wiring

* `app/agents/pim_agent.py` — new `optional_tool_group('pim',
  'cp_tickets')` block that loads the three tools after the
  existing `task_tools` block. Failsafe: any factory error logs a
  warning and continues with the other tools (mirrors the rest of
  the file).
* `app/crews/pim_crew.py` — `PIM_TASK_TEMPLATE` updated with a
  paragraph distinguishing the two ticket systems plus the explicit
  rule **"if `list_tasks` / `search_tasks` come back empty, do NOT
  report 'no tasks found' — try `cp_search_tickets` /
  `cp_list_tickets` first."** Closes the failure shape from
  2026-05-09 8:26 AM directly.

The Commander class itself (`app/agents/commander/orchestrator.py`)
does **not** carry a CrewAI Agent inventory — it's a router that
uses the LLM directly for routing decisions, then dispatches to
crews via `crews/registry.py`. So PIM (the crew Commander would
route ticket-move requests to) is the only meaningful wiring point.
The unused `self.memory_tools = create_memory_tools(collection="commander")`
line at line 567 of the orchestrator is pre-existing dead state —
not touched.

### 22.4 Files

```
app/control_plane/tickets.py             # +52 — move_ticket()
app/tool_registry/capabilities.py        # +14 — new "tickets" category
app/tools/control_plane_tickets_tool.py  # +293 — new file, 3 tools
app/agents/pim_agent.py                  # +14 — optional_tool_group block
app/crews/pim_crew.py                    # +14 — task-template paragraph
tests/test_control_plane_tickets_move.py # +186 — 5 tests
tests/test_cp_tickets_tool.py            # +247 — 15 tests
```

20 new tests, all green:

```
tests/test_control_plane_tickets_move.py — 5 passed
  test_move_happy_path
  test_move_unknown_ticket
  test_move_unknown_target_project
  test_move_writes_audit_entry_with_project_names
  test_move_idempotent_remove

tests/test_cp_tickets_tool.py — 15 passed
  TestCpListTickets — 5 (active project default, named project, status
                          filter, unknown project, empty result)
  TestCpSearchTickets — 3 (hits, no hits, empty query short-circuit)
  TestCpMoveTicket — 5 (happy, unknown→helpful error, both validation
                        paths, exception→string)
  TestCapabilityRegistration — 2 (manages-tickets in vocab,
                                  reads-deployment-state still in vocab)
```

### 22.5 Out of scope

* **Bulk move / move-by-search.** The current tools take a single
  UUID — multi-ticket moves require multiple calls. Adequate for
  the failure shape that triggered §22 (one or two misrouted
  tickets per incident); revisit if operator-traffic shows
  bulk-move requests landing.
* **Ticket-create from agent.** `cp_*` deliberately does NOT
  expose `create_ticket`. Tickets are created via the inbound
  Signal queue path; surfacing creation to the agent invites
  duplicate-ticket pollution. If a future workflow needs it,
  route through `change_requests/` (Phase 5.3a) so the operator
  can gate.
* **React `/cp/changes`-style operator surface for moves.** Not
  built — moves are agent-initiated only, audit-logged, and
  visible in `/api/cp/audit?action_prefix=ticket.moved`. If
  misrouted-move incidents accumulate, a thin "recent moves" tab
  would help; defer until that signal exists.


## 23. 2026-05-09 Self-Heal v3 + 9-gap resilience closure

**Trigger.** Years-of-uptime audit on 2026-05-09 identified 9
silent-failure modes the existing healing pass (§14) didn't cover.
Self-Heal v2 had shipped the dispatcher infrastructure but only the
no-op `log_only` runbook was registered, and the auditor ran every
30 minutes for a month logging *"0 resolved, 1 attempted, 23 total
patterns"* without applying any fixes. Three waves close the gap.

### 23.1 Self-Heal v3 — operational handlers + proactive monitors + auditor bridge

`docs/SELF_HEAL_V3.md` is the canonical reference. Layout:

  * `app/healing/handlers/` — 6 reactive runbook handlers registered
    against the v2 dispatcher: `db_pool_reset` (auto-reset on
    "connection pool exhausted" — 51% of error volume),
    `apscheduler_overrun_alert`, `numeric_overflow_widen_cr` (files
    a CR with a generated migration), `cost_mode_undefined_alert`
    (×2), `anthropic_str_content_cr` (files a CR for a defensive
    parser-guard module), and a catch-all `self_heal_router` for
    variant-heavy patterns (embed-misroute, mem0-no-function-calling,
    schema-missing-column).
  * `app/healing/monitors/` — proactive monitors:
    `disk_quota` / `listener_heartbeat` / `cron_liveness` /
    `vendor_sunset` / `idle_cooldown` / `audit_chain_check` /
    `lock_housekeeper` / `signal_heartbeat`. Plus `retention.py`
    (chromadb / worktree / attachment) and `adapter_lifecycle.py`.
    All run inside a single daemon driver in
    `app/healing/monitors/__init__.py`; each monitor cadence-guards
    internally. The driver gates on
    `HEALING_MONITORS_ENABLED` (default ON).
  * `app/healing/auditor_bridge.py` — closes the silent
    "0 resolved" gap by polling `audit_journal.json` for
    `error_fix_proposed` events; one Signal alert per
    (pattern, attempt) plus a CR mirror at
    `docs/proposed_fixes/<pattern>__attempt_<n>.md`. CR failure
    is non-fatal; Signal alert always goes out.
  * `app/healing/watchdog.py` — daemon-thread reaper that
    re-spawns the healing-monitors and healing-auditor-bridge
    threads if they die. 60 s loop, 3-crashes-per-hour give-up,
    24 h reset window. Touches `workspace/healing/watchdog_heartbeat`
    every iteration so the existing `cron_liveness` monitor can
    detect a watchdog death.

### 23.2 Disk-quota guard

`app/safe_io.py` `safe_write` and `safe_append` now raise
`DiskQuotaError(OSError)` when free space on the target volume drops
below `DISK_FREE_THRESHOLD_MB` (default 200 MB). Probe failures
fail OPEN — we never let a buggy guard halt the system. Each refusal
is best-effort audited as `actor='safe_io', action='disk_quota_block'`.

### 23.3 Audit-chain verifier

`app/coding_session/audit_verify.py` is a read-only verifier that
walks the chain in `workspace/coding_sessions/audit.jsonl` forward
from genesis, reporting every break (tampered payload / prev-hash
mismatch / malformed JSON / missing payload). The daily monitor at
`app/healing/monitors/audit_chain_check.py` runs it and Signal-alerts
on the first break (cooldown 24 h or new break-line). The verifier
NEVER modifies the chain — recovery is an operator call.

### 23.4 Lock-file housekeeper

`app/healing/monitors/lock_housekeeper.py` walks `workspace/locks/`,
`workspace/dreams/`, `workspace/`. Deletes `*.lock` files that
satisfy BOTH guards: age > 1 h AND fcntl-uncontested. The fcntl
probe is the source of truth on whether the lock is held — it
defends against PID-reuse hazards. Pile-up alert at >50 lock files
(suggests a leaking subsystem); 24 h cooldown.

### 23.5 Idempotent `start()` + watchdog precondition

For the watchdog to safely re-spawn dead daemons, every supervised
`start()` must be truly idempotent — checking thread liveness on
every call, not gating on a `_started` boolean flag that drifts out
of sync after a thread dies. Updated:

  * `app/healing/monitors/__init__.py:start` — uses `_is_running()`
    that walks `threading.enumerate()`.
  * `app/healing/auditor_bridge.py:start` — same pattern.
  * `app/healing/watchdog.py:start` — same.

Without these updates, the watchdog's re-spawn calls would silently
no-op.

### 23.6 Adapter lifecycle (gap #4)

`app/training/adapter_lifecycle.py` (open-tier; reads
`app.training_pipeline` constants without modifying it) does
monthly orphan cleanup of `workspace/training_adapters/` and
`workspace/trained_models/`, dead-pointer detection for registry
entries whose `adapter_path` doesn't exist, total-disk-usage bloat
alert at >5 GB, and a history snapshot trail at
`workspace/healing/adapter_lifecycle_history.jsonl` (closes the
rollback-blind-spot where overwriting a slot loses prior state).

### 23.7 Retention jobs (gap #8)

`app/healing/monitors/retention.py` registers three independent
cadence-guarded monitors:

  * `run_chromadb` — weekly. Per-collection record cap (default
    100k); deletes oldest records by metadata timestamp. Dry-run
    via `RETENTION_DRY_RUN=true`.
  * `run_worktrees` — daily. Removes coding-session JSON +
    worktree dir for sessions in terminal states (submitted /
    discarded / expired / failed) older than 7 days. Pile-up alert
    at >50 examined records.
  * `run_attachments` — daily. Two passes: age-delete (>30 days
    old) then size-cap (oldest-first when surviving total > 1 GB).
    Walks `$SIGNAL_ATTACHMENTS_DIR` (default `/app/attachments`).

### 23.8 Signal-channel heartbeat (gap #3)

`app/healing/monitors/signal_heartbeat.py` re-scopes the original
"120-day re-registration" gap (signal-cli bot accounts don't
auto-deregister on idle but the daemon DOES die silently) to
end-to-end heartbeat detection. Watches `workspace/conversations.db`
mtime for inbound activity vs `workspace/signal_outbound.json` for
outbound; flags asymmetric (recent-inbound + stale-outbound for >7d)
or likely-dead (no traffic for >7d). Multi-channel escalation:
Signal alert (might fail if Signal is the failure mode), PWA push
after 3 consecutive fails, email after 7. Streak resets on
recovery.

### 23.9 Tests

76 tests across `tests/healing/`. Combined with the existing
`test_self_heal_runbooks.py` and `test_self_healing_comprehensive.py`,
the healing surface has 142+ regression tests.

### 23.10 Wave 4 React control surface (commits 5+6 of the day)

Five env-only flags became runtime-toggleable from
`/cp/settings`:

  * `goodhart_hard_gate_disabled` + `goodhart_hard_gate_enforcing` →
    a 3-way segmented control (Off / Advisory / Enforcing) with
    explanatory copy per option.
  * `error_runbooks_enabled` + `tool_supervisor_enabled` +
    `recovery_loop_enabled` → 3 master toggles in the new
    `Self-heal subsystems` card. A per-runbook on/off list below
    drives `workspace/self_heal/runbook_settings.json` directly
    (the dispatcher re-reads on every anomaly).

Five reader functions gained the runtime-settings → env-fallback
hierarchy: `governance._goodhart_hard_gate_disabled` /
`_enforcing`, `healing.runbooks.runbooks_enabled`,
`tool_runtime.supervisor.is_enabled`, `recovery.loop.is_enabled`.
Each tries `runtime_settings.get_*()` first; falls back to the
env var if `runtime_settings` raises (tests / degraded boot). New
keys are env-seeded on first read so existing `.env` setups
preserve their behaviour. Operator-authorized as part of the
React-toggle work.

## 24. 2026-05-09 Wave 2 Life Companion (proactive personal-life surface)

`docs/LIFE_COMPANION.md` is the canonical reference. New package
`app/life_companion/` adds three proactive features that watch the
operator's life rather than the system's health:

  * `email_monitor` — wraps `app/tools/email_importance.py` heuristic
    in a 10-min cadence-guarded loop. Fetches up to 25 unread inbox
    messages via `gmail_tools._list_recent`, scores each, surfaces
    top-3 above urgency threshold to Signal. Per-message-id dedup
    with 500-entry FIFO so repeat alerts can't happen.
  * `daily_briefing` — three flavours (morning 07:00 / evening 18:00
    / weekly Mon 09:00). Synthesises calendar events (next 24 h via
    `gcal_tools`) + top-3 urgent unread + open project tickets +
    companion-surfaced ideas. Idempotent within each scheduled
    window.
  * `routine_detector` — reads
    `workspace/affect/episode_affect_tags.jsonl` over an 8-week
    window. Clusters episodes by (weekday, hour-bucket, crew); flags
    clusters with ≥4 occurrences spanning ≥2 ISO weeks AND ≥60 %
    concentration to the weekday. Emits a Signal alert when a NEW
    routine is detected; a separate nudge fires when a routine's
    window is approaching today.

All three ride the existing idle scheduler via
`app.companion.loop.get_idle_jobs()` — no TIER_IMMUTABLE / TIER_GATED
files modified. Master switch `LIFE_COMPANION_ENABLED` (default ON);
each component honours `idle_scheduler.is_enabled()` kill switch.

12 tests in `tests/life_companion/`.

## 25. 2026-05-09 Governance hardening — Tier-3 amendment + ratchet + Goodhart gate

Three operator-authorized governance subsystems shipped in one day,
all sharing a common architectural pattern (typed exceptions +
hash-chained audit + React toggle). Distinct concerns:

### 25.1 Tier-3 Amendment Protocol

`docs/TIER3_AMENDMENT.md` is canonical. New package
`app/governance_amendment/` lets the system *legitimately* propose
modifications to TIER_IMMUTABLE files after a demonstrated track
record. 10-state state machine
(PROPOSED → STAGED → COOLDOWN_OK → APPROVED → APPLIED → STABLE,
plus terminal branches), eligibility check (≥200 promotions/90 d,
<5 % rollback rate, no active alignment warnings, runbook health
≥50 %), `self_quarantine.py` (~30 files NO agent can amend even
with full eligibility), hash-chained JSONL + Postgres mirror audit
(`actor=tier3_amendment`).

Master switch `tier3_amendment_enabled` defaults OFF — exposed as a
React `/cp/settings` toggle with confirmation modal that explains
the 4-stage gate before flipping.

`auto_deployer.py` (TIER_IMMUTABLE) was operator-authorized to add
the seven `app/governance_amendment/*.py` files to TIER_IMMUTABLE so
the protocol's own files can't be mutated through the regular
change-request flow.

### 25.2 Governance Ratchet (#6 of the resilience-gap closure plan)

`docs/GOVERNANCE_RATCHET.md` is canonical. New package
`app/governance_ratchet/` provides operator-controlled raising /
relaxing of `SAFETY_MINIMUM` and `QUALITY_MINIMUM` above the
hardcoded `*_FLOOR` constants in `governance.py`. Four typed
exceptions enforce the invariants:
`MonotonicViolation` / `FloorViolation` / `CeilingViolation` /
`UnknownThresholdViolation`. The `effective_value(name)` function
clamps to `max(FLOOR, ratcheted_current)` — even a corrupted state
file or maliciously-edited JSON can't drop below FLOOR.

Operator-authorized edits to:

  * `app/governance.py` — renamed constants to `*_FLOOR`, added
    `threshold_floor()` + `effective_safety_minimum()` +
    `effective_quality_minimum()`, wired into `evaluate_promotion`.
  * `app/auto_deployer.py` — added 5 `app/governance_ratchet/*.py`
    files to TIER_IMMUTABLE.

React UI: `GovernanceRatchetCard` with two flows:

  * **Ratchet up** (simple confirm) — validates `new_value > current`
    and `≤ 1.0` before submit.
  * **Relax down** (typed-phrase confirm) — validates
    `floor ≤ new_value < current`, mandatory ≥10-char reason, AND
    typed phrase `RELAX <THRESHOLD>` matches exactly. Backend
    re-checks the phrase server-side as a UX-correctness check.

V1 is operator-only — no agent path; mutation flows through React
→ `/config/governance_ratchet/*` (gateway-bearer-secret gated).

### 25.3 Goodhart Hard Gate (#2 of the resilience-gap closure plan)

`docs/GOVERNANCE_RATCHET.md` §"Goodhart hard gate" is canonical.
Operator-authorized edits to `app/goodhart_guard.py` (TIER_IMMUTABLE)
add public read-only `recent_severity(lookback_hours=24)` and
`recent_signal_summary(lookback_hours=24)`. Operator-authorized
edits to `app/governance.py` add `_goodhart_hard_gate_disabled` /
`_enforcing` env readers + `_evaluate_goodhart_gate()` which runs
as **Gate 0 in `evaluate_promotion`** — before safety + quality.

Three-phase rollout in V1, controlled by the React 3-way segmented
control (`Off / Advisory / Enforcing`):

  * **Off (emergency disable)** — incident response when a buggy
    detector blocks legitimate promotions.
  * **Advisory (default ON)** — gate runs and records severity in
    `gate_results["goodhart"]` for audit; does NOT block. Operators
    watch the false-positive rate for ~2 weeks.
  * **Enforcing** — `severity == "high"` BLOCKS promotion.
    `severity == "medium" / "low"` are not blocking by design (the
    detector's medium/low signals are advisory by nature).

The gate fails OPEN when the detector itself raises (a buggy
detector should never halt every promotion).

### 25.4 Tests

20 governance-amendment tests + 18 governance-ratchet tests + 18
goodhart-gate tests = 56 new governance tests. Zero TIER_IMMUTABLE
test paths broken.

### 25.5 Combined session totals

10 commits, ~13,000 LOC, 148 tests passing, 7 of 9 resilience gaps
fully closed (#3, #4, #5, #6, #7, #8, #9), 2 partially closed (#1
covers prevention but not retroactive cleanup; #2 in advisory mode
by default — flipping to enforcing is a separate operator action
after the FP-rate observation period).

## 26. 2026-05-09 React dashboard hardening

The dashboard accumulated several known-broken or known-stale
surfaces over the prior week. This pass closed them and added the
operator controls that the legacy HTML monitor had but the React
build was missing.

### 26.1 Cost telemetry — Postgres-resident, per-agent, no "unknown"

Five layered changes that together turn the Costs tab into a
trustworthy spend log:

  1. **Per-project tagging on every LLM call.** `record_tokens()`
     in `app/llm_benchmarks.py` resolves the project from a
     ContextVar (`app/project_context.py:_project_id`) and writes
     it on every SQLite `token_usage` + `request_costs` row. The
     ContextVar is set at gateway request boundaries and at
     `Commander.handle()`.
  2. **Reconcile every observed cost into Postgres budgets.**
     `app/control_plane/budgets.py:reconcile_actual_spend()` is a
     plain UPSERT into `control_plane.budgets (project_id,
     agent_role, period)` keyed on the agent role from the new
     `_agent_role` ContextVar. `record_tokens()` calls it for any
     row with `cost_usd > 0`. The pre-call enforcement hook
     (`lifecycle_hooks._create_budget_hook`) only fires on the
     commander path; this reconcile catches every other LLM call
     that previously went unbudgeted.
  3. **`agent_scope()` wrappers.** `crew_lifecycle` (every user
     crew), each internal crew (critic, retrospective,
     self_improver), `Commander.handle`, and finally
     `idle_scheduler._run_single_job(name, fn, …)` each wrap their
     work in `agent_scope(role)`. Every nested LLM call sees the
     right role. The idle-scheduler one is what lets every
     background job (`llm-discovery`, `fiction-ingest`,
     `training-collector`, `atlas-competence-sync`, …) attribute
     to its own row instead of falling back to `unknown`.
  4. **Three Cost panels with distinct semantics:**
       * **Cost by Crew** — `request_costs.crew_name` from the
         long-lived SQLite tracker, year window. Compound names
         like `research+coding` split equally across components.
       * **Cost by Agent** — derived from Cost-by-Crew via
         `_CREW_TO_AGENT` mapping (`coding → coder`, `research →
         researcher`, …) so the chart has real data even before
         budgets accumulates. Falls back to budgets for roles the
         mapping doesn't cover.
       * **Cost by Internal Agent** — `control_plane.budgets`
         filtered to commander/critic/retrospective/self_improver
         + idle-scheduler job names (anything hyphenated).
  5. **Migration `022_relabel_unknown_to_idle_scheduler.sql`** —
     historical "unknown" rows merged into a single
     `idle_scheduler` row per project. Future spend subdivides
     into per-job rows automatically thanks to (3).

Other cost fixes in the same pass:

  * `tickets.complete()` looks up `project_id` and passes it to
    `audit.log()`; `migrations/020_backfill_audit_project_id.sql`
    backfilled 175 historical `ticket.completed` rows that had
    `project_id IS NULL`. Cost-tab daily/by-agent endpoints now
    return the historical $15.96 of work attributed to the
    correct projects (173 → default, 2 → PLG).
  * Audit feed limit raised 100 → 300 + "Only rows with cost"
    toggle; cost-bearing rows are sparse (`ticket.completed` only)
    so the default last-100 window had been all-null.

### 26.2 Tasks off Firestore — `control_plane.crew_tasks`

The `/cp/tasks` endpoint hit Firestore's 50 k-read/day free-tier
quota by late evening, surfacing as `tasks read: 429 Quota
exceeded` in the Crew Activity header.

  * Migration `021_crew_tasks.sql` creates
    `control_plane.crew_tasks` with indexes on `(started_at)`,
    `(project_id, started_at)`, `(state, started_at)`,
    `(crew, started_at)`, and a partial index on `parent_task_id`.
  * `app/control_plane/crew_tasks.py` provides CRUD helpers
    (`start_task`, `complete_task`, `fail_task`, `update_eta`,
    `mark_delegated`, `update_sub_agent_progress`, `mark_healed`),
    a `cleanup_zombies(max_age_hours=6)` for startup, and
    `list_recent` / `crew_statuses` (DISTINCT ON deriving per-crew
    latest from the same table — no separate `crews` row needed).
  * `app/firebase/crew_tracking.py` dual-writes: every entry
    point writes to Postgres FIRST, then the legacy Firestore
    mirror as fire-and-forget. Backwards observability preserved
    for any other consumer; the dashboard read path no longer
    depends on Firestore.
  * `/api/cp/tasks` reads from `crew_tasks` + `crew_statuses` and
    merges with the canonical `_KNOWN_CREWS` registry so every
    crew (including idle ones) renders. Response shape unchanged
    so the React side needed zero updates.

### 26.3 Budget operator controls

  * **Pause / Resume button on every Budget card.**
    `BudgetEnforcer.set_paused()` flips `is_paused` on the
    current-period row and audit-logs `budget.paused` /
    `budget.unpaused`. Backed by `POST /api/cp/budgets/pause`.
    Resume is highlighted green so it's obvious how to clear a
    stale pause.
  * **Background tasks kill switch on `/cp/settings`.**
    `GET/POST /config/background_tasks` mirrors the legacy
    Firestore `config/background_tasks` document over HTTP. POST
    flips `idle_scheduler.set_enabled(...)` AND mirrors back to
    Firestore so the in-process listener at
    `idle_scheduler.py:2581` and the legacy HTML monitor stay in
    sync. Single ON / OFF toggle on the React Settings page.
  * `BudgetDashboard` row key is now `${project}:${role}:${period}`
    (was `agent_role` alone, which collided across projects under
    the "All projects" view). Each card appends `· PROJECT_NAME`
    so the four identical "researcher" cards under "All projects"
    are distinguishable.

### 26.4 Affect — sender id threading + last_seen wiring

The primary user's `last_seen_ts` was pinned at first-seen
(`2026-04-28`) because the affect attachment hook reads `sender_id`
from `HookContext`, but `Commander` constructed every PRE_TASK /
ON_COMPLETE context without populating it. Brainstorm had the
same gap on a different code path.

Fix: new `_sender_id` ContextVar in `app/project_context.py` with
`sender_scope()` helper. `Commander.handle` wraps dispatch in
`sender_scope(sender)`. Brainstorm's `_resolve_sender` publishes
the resolved sender into the ContextVar. `affect/hooks.py` falls
back to `resolve_current_sender_id()` when the `HookContext`
doesn't carry one. Every Signal message and every brainstorm
session now bumps `last_seen_ts` automatically.

### 26.5 Consciousness probes — durability + plumbing

Three independent fixes that pulled probe scores off the floor:

  * **GWT broadcasts persist + hydrate.** Every broadcast goes
    into Postgres `subia_broadcasts`; `GlobalWorkspace.__new__`
    rehydrates the most recent 50 rows on startup. Previously the
    in-memory deque emptied at every gateway restart and the GWT
    probe pinned at 0.4 ("no broadcasts, sent test") for ~1 hour
    while it refilled.
  * **HOT-2 hand-off.** `lifecycle_hooks` ON_TASK_COMPLETE now
    copies `_meta_cognitive_state.strategy_assessment` into
    `internal_states.meta_strategy_assessment` before persistence;
    the probe filters exactly on that column.
  * **SM-A role→crew translation.** `SELF_MODELS` is keyed on
    role names (`researcher`, `coder`, `writer`); agent_state
    keys per-crew (`research`, `coding`, `writing`). Added a
    translation table so `stats.get(role, {})` resolves correctly
    instead of pinning the metric at the 0.5 "no data" fallback.

`/api/cp/consciousness` also returns a `homeostasis` payload now
(cognitive_energy / frustration / confidence / curiosity / etc.)
so the legacy 4-bar strip the old HTML monitor had is reproduced
in `ConsciousnessIndicators.tsx`.

### 26.6 Dashboard plumbing — proxy + build hygiene

  * Node proxy whitelist (`dashboard/server.mjs`) gained
    `/epistemic/` (claim ledger; was confused with `/episteme/`,
    the research KB) and `/affect/`. Both tabs were 404'ing on
    every panel before this.
  * `npm run build` had been silently failing at `tsc -b` for
    weeks because of six pre-existing TS errors in
    `affect/ReferencePanelGrid.tsx`, `affect/AffectStatusStrip.tsx`,
    `api/affect.ts`, `CompanionTab.tsx`, and `EpistemicPage.tsx`
    (×2). Build skipped to vite, postbuild never ran, and
    `dashboard/serve-root/cp/` stayed pinned at an Apr-30
    snapshot — Brainstorm / Coding Sessions / Settings tabs that
    landed in source after that date never reached the production
    server on `:3100`. All six fixed; `npm run build` runs end
    to end again.
  * UI direction convention (Costs daily chart, Evolution history
    table, Evolution variants list, Ops errors / anomalies /
    deploys lists): lists are newest-first at the top, charts
    run left→right = oldest→newest. The Cost-Daily chart used to
    plot `4/21 → 4/4` left-to-right (backwards); now plots
    `4/4 → 4/21`.

### 26.7 New: Settings page (`/cp/settings`)

Single-page hub for personal-agent runtime toggles. Cards added
in this pass:

  * Background tasks ON / OFF (idle scheduler kill switch — §26.3).
  * Voice mode (off / local / cloud — wired into the runtime
    settings store from §20).
  * Vision computer use (enabled + monthly USD cap).
  * Concierge persona (on / off).
  * Web Push subscriptions (per-device, with test send).
  * Governance ratchet (read + ratchet up + relax-down with typed
    confirm — see §25).
  * Goodhart hard gate (Off / Advisory / Enforcing — see §25).
  * Self-heal subsystems (per-runbook toggles + tool-supervisor
    on/off — see §23 / `docs/SELF_HEALING.md`).

Backed by `GET/POST /config/runtime_settings` (gateway-secret on
POST), `GET/POST /config/background_tasks`, and the existing
governance + runbook endpoints. Settings persist across restarts
and are audit-trailed under
`runtime_settings_change` / `governance.ratchet.*` /
`runbook.toggle` action types.


## 27. 2026-05-09 (afternoon) — Change-request runtime + 6 follow-ups

After the morning's workspace-routing remediation (§21) and ticket-
move tool (§22), the afternoon session put the change-request system
into actual production use end-to-end and closed six adjacent gaps
the first real run exposed.

### 27.1 Trigger

The honest reproduction of "agent proposed a fix, operator 👍'd
it, gateway answered with a contradiction":

```
✅ Change request eb677b22… approved + applied.
  ok: False
  branch: ?
  PR: (failed to open)
  module reload: None
  ERROR: host bridge unreachable; cannot write file or run git
```

The headline said success; the body said failure. Two distinct
problems compounded:

1. The gateway's host bridge (filesystem + git tunnel from
   container to host) was **completely unconfigured on this
   deployment** — `HOST_BRIDGE_URL` empty, `BRIDGE_SHARED_SECRET`
   length 0, `get_bridge('change_requests')` returned `None`.
   Every approval landed in `APPLY_FAILED`.
2. The Signal-ack formatter hardcoded `✅ approved + applied`
   regardless of the actual `apply_result.ok` value, hiding
   failures in plain sight.

### 27.2 The six PRs

| PR | Concern | Scope |
|---|---|---|
| **#74** | The §21 follow-up: agent search-wrong-DB hallucination ("the database returned no tasks") when asked to move a Kanban ticket | Adds `move_ticket(ticket_id, target_project_name)` to `app/control_plane/tickets.py` + 3 agent tools (`cp_list_tickets` / `cp_search_tickets` / `cp_move_ticket`) on a new `manages-tickets` capability + PIM template guidance distinguishing local SQLite vs Postgres ticket systems. Audit-logged. See §22. |
| **#77** | After #74 the tools existed but the routing didn't reliably reach PIM for ticket-ops queries (`move that task to X`, `list my tickets`, `what's on my kanban`) | `_PIM_NOUN_RE` gains `tickets?\|kanban`; `_PIM_QUALIFIER_RE` gains `move\|migrate\|reassign\|search\|list\|show\|find`; `_CREW_BASE_PURPOSE["pim"]` mentions Kanban + control_plane.tickets explicitly so the LLM router catches the rest. Routes PIM via the existing dual-signal short-circuit (which runs BEFORE the follow-up filter — critical because "move that task" trips weak-anaphora rules on length<40). |
| **#78** | Self-heal alarm fired with two false positives: `error_resolution` "stale 7098 min" and `workspace_sync` "stale 39611 min" | Both jobs were running fine but only updated their footprint when there was actual work. Fix: heartbeat-touch the footprint at end of every run regardless of work. `run_error_resolution` always touches `workspace/error_tracker.json` (creates `{}` on first run); `sync_workspace` always touches `workspace/.git/HEAD` via a new `_touch_workspace_sync_heartbeat()` helper, even when `WORKSPACE_BACKUP_REPO` is empty. |
| **#79** | The contradictory ack message (`✅ approved + applied / ok: False`) | Branch the Signal-ack on `apply_result.ok`. Success path keeps `✅ approved + applied`; failure path uses `⚠️` + `apply FAILED` headline + explicit pointer at the `Retry apply` button in `/cp/changes`. Reject-path (`❌`) unchanged. |
| **#81** | Calling the `/api/cp/changes/{id}/retry-apply` endpoint or the React UI's "Retry apply" button raised `ValueError: cannot approve … in status apply_failed` — the lifecycle's `approve()` only accepted PENDING but the endpoint assumed APPLY_FAILED was also accepted | Extend `approve()` to accept both `PENDING → APPROVED` (first-time, unchanged) and `APPLY_FAILED → APPROVED` (retry path, new). Already-`APPROVED` stays idempotent. Audit-event distinction: first approval logs `approved`, retry logs `re-approved-for-retry`. |
| **#82** | Self-heal Wave 0/1 — closes 7 of 8 resilience gaps from the years-of-uptime audit | DB backup engine (Postgres + Neo4j + ChromaDB; opt-in `HEALING_DB_BACKUP_ENABLED`); per-listener heartbeats (each Firestore poller touches `workspace/heartbeats/<thread>.heartbeat` per loop iteration; `listener_heartbeat` monitor moves from workspace-wide proxy to per-listener targeting); `vendor_sunset` files CRs instead of alert-only; `log_archival` monitor (gzip-rotation of errors.jsonl, audit_journal.json size cap, 90-day purge); `conversations.db` monthly VACUUM. Cross-references §23. |

### 27.3 Host-bridge runtime wiring (operator deployment step, NOT in source)

Source code for the change-request system shipped in #54 (May 2026,
§17). The runtime wiring on this specific deployment was missing:

```
host_bridge/capabilities.json     gained a `change_requests` agent
                                  entry: filesystem.read/list/write
                                  + execute on
                                    app/, tests/, docs/,
                                    dashboard-react/, deploy/,
                                    scripts/
                                    + PROGRAM.md / README.md.
                                  Blocked: souls/, forge/,
                                  auto_deployer.py, host_bridge/.
                                  risk_ceiling=high.
.env                              + BRIDGE_TOKEN_CHANGE_REQUESTS=cr-…
launchctl stop/start              reload the host bridge so
  com.crewai.bridge               capabilities.json reloads.
docker compose up -d              gateway picks up the new
  --force-recreate gateway        BRIDGE_TOKEN_CHANGE_REQUESTS env.
```

Once those four steps complete:

```
docker exec gateway python3 -c \
  "from app.bridge_client import get_bridge; \
   b = get_bridge('change_requests'); \
   print(b.is_available())"
True
```

The change-request system can now hot-apply approved changes. The
first end-to-end run was the morning's `eb677b22…` change request
(file `docs/proposed_fixes/research_ConnectionError_1.md`, requested
by `self_heal_handler` in #82 §A4): retried after the bridge wiring,
applied to `auto/change_eb677b222479` (commit `0ee71627`), auto-PR
opened as **#80**, merged.

### 27.4 The full closure

Combined with #67 (active project persists across restart), #68
(gee_run_script timeout 600 s), #69 (eesti_mets profile), #71
(KaiCart/Archibal/PLG keyword expansion), #72 (always-ask routing),
the morning's failure shape can no longer recur silently. Combined
with #74/#77 (move-ticket API + routing), the agent can now answer
"move that task to Eesti mets" honestly. Combined with #79/#81
(honest ack + retry-apply lifecycle), the operator gets accurate
status on every change-request decision and can self-serve retries.
Combined with #82 (Wave 0/1 self-heal), DB backups, per-listener
heartbeats, and log archival are all live.

The 8th Wave 0/1 gap (the one PR #82 deliberately deferred) is the
remaining open item from this audit; everything else from the
2026-05-09 reports is in production.

### 27.5 Files touched (afternoon-session-only)

```
app/agents/commander/routing.py            # PR #77
app/auditor.py                             # PR #78
app/change_requests/lifecycle.py           # PR #81
app/control_plane/tickets.py               # PR #74
app/healing/db_backup.py                   # PR #82 (NEW)
app/healing/listener_heartbeats.py         # PR #82 (NEW)
app/healing/monitors/db_vacuum.py          # PR #82 (NEW)
app/healing/monitors/listener_heartbeat.py # PR #82
app/healing/monitors/log_archival.py       # PR #82 (NEW)
app/healing/monitors/vendor_sunset.py      # PR #82
app/firebase/listeners.py                  # PR #82
app/main.py                                # PR #79
app/tools/control_plane_tickets_tool.py    # PR #74 (NEW)
app/tool_registry/capabilities.py          # PR #74 (manages-tickets)
app/workspace_sync.py                      # PR #78
host_bridge/capabilities.json              # runtime wiring
.env                                       # runtime wiring
RESTORE.md                                 # PR #82 (NEW)
scripts/backup.sh                          # PR #82 (NEW)

tests/healing/test_db_backup.py            # PR #82 (NEW)
tests/healing/test_db_vacuum.py            # PR #82 (NEW)
tests/healing/test_listener_heartbeat_monitor.py # PR #82 (NEW)
tests/healing/test_listener_heartbeats.py  # PR #82 (NEW)
tests/healing/test_log_archival.py         # PR #82 (NEW)
tests/healing/test_vendor_sunset_cr.py     # PR #82 (NEW)
tests/test_change_request_retry_apply.py   # PR #81 (NEW) — 10 tests
tests/test_change_request_signal_ack.py    # PR #79 (NEW) — 6 tests
tests/test_cp_tickets_tool.py              # PR #74 (NEW)
tests/test_control_plane_tickets_move.py   # PR #74 (NEW)
tests/test_cron_heartbeats.py              # PR #78 (NEW) — 7 tests
tests/test_routing_ticket_ops.py           # PR #77 (NEW) — 18 tests
```

### 27.6 What still needs operator action

* **The 8th Wave 0/1 gap** deferred from #82 — confirm intent via
  the audit report, schedule for Wave 0/2.
* **`HEALING_DB_BACKUP_ENABLED` is opt-in** by default. If you
  want gateway-driven backups, flip the env var and restart;
  otherwise `scripts/backup.sh` runs host-side from cron / launchd.
* **`WORKSPACE_BACKUP_REPO` remains unset.** Workspace sync is
  live (heartbeat now advances even with no remote, per #78), but
  if you want the workspace-state push to a backup repo, set the
  env var. Optional.

---

## 28. 2026-05-09→10 — Personal-agent + self-evolution sweep (Phases B → G)

Seven sequential commits over a 24-hour window that close the rest
of the resilience-gap roadmap and add the proactive-companion
behaviour the operator asked for. Each phase landed clean (tests
green at every commit) and audit feedback drove three follow-up
phases (E, F, G) that surfaced silent-failure bugs in the prior
deliveries.

```
A   2026-05-09  342bd554  Wave 0/1 self-heal closure (see §27)
B   2026-05-09  ab76da03  5 personal-life features
C   2026-05-09  1134dad1  5 self-improvement / observability
D   2026-05-09  86770852  7 audit-finding gaps
E   2026-05-09  72f3f6e9  Phase D audit cleanup
F   2026-05-10  bc7fe94d  Phase E delivery-sweep cleanup
G   2026-05-10  bd3cbef8  4 final companion gaps
```

Net deltas: ~13,000 LOC across `app/` + `tests/`. ~340 new tests.
374 cross-suite tests green at the end.

The doc index for this sweep is `docs/PROACTIVE_DELIVERY_INDEX.md`.

### 28.1 Phase B — personal-life surface (5 features)

| ID | What | Lives in |
|----|------|----------|
| B1 | Interest model — 12 h aggregator over conversations.db, inbox triage, calendar titles, FEEDBACK events, affect tags. Output `workspace/companion/interest_profile.json` with top-30 topics scored by recency × frequency × source-diversity. | [`interest_model.py`](app/companion/interest_model.py) |
| B2 | Calendar prep — 5-min idle job; sends Signal prep 30 min before each event with title, attendees, agenda, per-attendee inbox + Mem0 enrichment. | [`calendar_prep.py`](app/life_companion/calendar_prep.py) |
| B3 | Closed-loop feedback router — drains `feedback.events` and dispatches to skill counter / recipe ledger / companion event log. Sidechannel via `notify_meta` correlates reactions to source. | [`feedback_router.py`](app/companion/feedback_router.py), [`notify_meta.py`](app/companion/notify_meta.py) |
| B4 | Long-arc commitment follow-up — daily walk over `SelfState.active_commitments`. 7d → 14d → 30d cadence by age + deadline-imminent + post-deadline nudges. | [`long_arc_follow_up.py`](app/life_companion/long_arc_follow_up.py) |
| B5 | Personalized signal context — `ToneContext` (mood quadrant + local hour + top-3 interests) injected into the concierge rewriter prompt. | [`signal_context.py`](app/personality/signal_context.py) |

39 tests. See `docs/COMPANION_FEEDBACK_LOOP.md` (B3 + D4 + G2 + F3 are one logical loop) and `docs/LIFE_COMPANION.md` (B2 + B4 + others).

### 28.2 Phase C — self-improvement & observability (5 features)

| ID | What | Lives in |
|----|------|----------|
| C1 | Adapter retirement performance arm — weekly snapshot of `workspace/training_adapters/registry.json`; computes `health_score = (eval_score / QUALITY_GATE) × age_decay × recipe_winrate` and proposes retirement to JSONL + Signal. | [`adapter_performance.py`](app/training/adapter_performance.py) |
| C2 | Silent regression detector — 4 h monitor; walks `audit_journal.json` for cron-like events, computes 13-day baseline rate; alerts on >30% drop with recent git commits + control-plane CR/ratchet/amendment actions as suspects. | [`silent_regression_detector.py`](app/healing/silent_regression_detector.py) |
| C3 | Paper-to-experiment pipeline — weekly arXiv ATOM fetch keyed off interest_model topics; Haiku 4.5 strict-JSON `{summary, implications, experiment, relevance}`; top-3 by relevance into Signal digest. | [`paper_pipeline.py`](app/episteme/paper_pipeline.py) |
| C4 | Failure-pattern learner — daily pass over `workspace/logs/errors.jsonl`; SHA-1 signature clustering (matches runbook dispatcher); excludes already-registered runbooks; writes scaffolds to `docs/proposed_fixes/learner_<sig>.md`. | [`pattern_learner.py`](app/healing/pattern_learner.py) |
| C5 | Meta-governance auto-propose — daily evaluation of audit_log promotions; proposes ratchet UP when avg ≥ effective + 0.03, ≥20 promotions, rollback rate < 5%; operator approves via React `/cp/settings`. | [`auto_propose.py`](app/governance_ratchet/auto_propose.py) |

31 tests. See `docs/SELF_HEAL_V3.md` (C2 + C4 added there) and `docs/GOVERNANCE_RATCHET.md` (C5 added there).

### 28.3 Phase D — 7 audit-finding gaps

Closes the partial / mismatched items from the user's audit against the original Wave 0/1/2/3 plan.

| ID | What | Lives in |
|----|------|----------|
| D1 | PG startup circuit breaker — bounded retry (1/3/9 s × 3) + 60 s breaker + Signal alert + connect_timeout in DSN. | [`db.py`](app/control_plane/db.py) (modified) |
| D2 | Evolution audit 90 d archive — `_archive_evolution_runs()` tarballs `shinka_results/run_*` after 90 days. | [`log_archival.py`](app/healing/monitors/log_archival.py) (modified) |
| D3 | Goodhart enforcing flip auto-proposer — daily watch of the gate while it sits in Advisory mode; proposes flip when conditions warrant. | [`goodhart_enforcing_proposer.py`](app/governance_ratchet/goodhart_enforcing_proposer.py) |
| D4 | Feedback → scheduler downweighting — 👎 multiplies workspace weight by 0.8 with 3 d halflife decay. | [`feedback_weights.py`](app/companion/feedback_weights.py), [`scheduler.py`](app/companion/scheduler.py) (modified) |
| D5 | Personalized weekly digest — Friday 09:00 Signal digest from RSS feeds + GitHub user events + arXiv-by-author + Google News for ventures. Stdlib-only feed parser. | [`personalized_digest.py`](app/life_companion/personalized_digest.py) |
| D6 | Semantic LLM-output drift — weekly checkpoint of golden-probe outputs vs baseline embedding; alerts at avg cosine < 0.85. | [`llm_output_drift.py`](app/healing/llm_output_drift.py) |
| D7 | Rejected-hypothesis lessons KB — daily clusterer over rejected change-requests + 👎 companion feedback + medium/high Goodhart signals. Public `check_against()` API for synthesis modules. | [`lessons_learned.py`](app/companion/lessons_learned.py) |

51 tests. See `docs/GOVERNANCE_RATCHET.md` (D3) and `docs/SELF_HEAL_V3.md` (D6).

### 28.4 Phase E — Phase D audit cleanup

Audit pass after Phase D revealed 15 issues. 13 fixed cleanly, 2 deferred.

P0 silent bugs:
* E1 — `feedback_router._resolve_send_ts` queried non-existent `feedback.responses`; fixed to hit real `feedback.response_metadata`.
* E2 — `pattern_learner._registered_signatures` imported non-existent `_LOCK`; fixed to `_registry_lock`.
* E3 — `calendar_prep` imported `_list_messages` (doesn't exist; actual is `_list_recent`) and `get_manager` (doesn't exist; actual is `search_memory`).

P1 logic:
* E4 — interest_model included 👎 comments as POSITIVE interest signal; fixed to filter to `polarity == "up"`.
* E5 — paper_pipeline `_save_seen` evicted arbitrary IDs (set ordering); fixed to `dict[id, ts]` sorted-by-ts.
* E6 — paper_pipeline regex XML parsing replaced with shared `feed_parser`.

P2 architecture:
* E7 — extracted `app/utils/hash_embedding.py` (lessons_learned + llm_output_drift had identical 256-d hashing-trick).
* E9 — pattern_learner scaffolds unified to `docs/proposed_fixes/` (alongside auditor_bridge).
* E10 — llm_output_drift uses `chromadb_manager.embed` with hash fallback. `_embed` returns `(vector, source)`. Cross-source comparison refused.
* E12 — boot_reset stops creating empty dbm placeholder files.
* E13 — scheduler hoists `_affect_weight()` out of per-candidate loop.
* E14 — runtime_settings reads via `getattr(s, …, default)` so a stripped-down test Settings doesn't crash on import.

12 follow-up tests pin behavior. Deferred: E8 (state-dir unification, ~20 tests would migrate) + E11 (cosmetic field rename).

### 28.5 Phase F — Phase A-E delivery audit

Honest end-to-end audit revealed 11 more issues. All shipped clean.

P0 silent bugs (3):
* F1 — `interest_model._feedback_events_text` and `lessons_learned._from_companion_feedback` read non-existent single `events.jsonl`; events are sharded per workspace under `events/<ws_id>.jsonl`. Added `events.iter_all_workspaces` helper; both consumers walk it correctly.
* F2 — `adapter_performance._recipe_winrate_for_adapter` iterated `RecipeOutcome.tool_names` (doesn't exist; field is on `AgentRecipe.extra_tool_names`). Rewrote as two-step join: `list_recipes` → match adapter in `extra_tool_names` → `list_outcomes(recipe_id)` per match.
* F3 — `notify(metadata=…)` had zero callers; B3 closed-loop's entry point was dead. Extended `notify_on_complete` decorator to accept `metadata`, wired schedule_manager_tools, added 4th sink to feedback_router (`job_id` → `workspace/companion/job_feedback.jsonl`).

P1 logic (3):
* F4 — signal_context circadian integration imported non-existent `current_segment`. Fixed to `current_circadian_mode` + map modes (`active_hours` → afternoon, `deep_work_hours` → evening, etc.) to the four time-of-day buckets.
* F5 — `lessons_learned.check_against()` had no consumer. Wired into `change_requests/lifecycle.create_request`: when a new proposal matches a prior-rejected pattern (≥0.4 cosine), the CR `reason` gets a "⚠️ Matches lesson abc12" annotation.
* F6 — interest_profile under-consumed. Added "🧭 Topics you've cared about" section to `daily_briefing._compose_weekly`. Daily morning/evening stay clean.

P1 retention + efficiency (2):
* F7 — `app/utils/jsonl_retention.py` (`cap_jsonl` + `append_with_cap`). Wired into 5 unbounded JSONLs:
  ```
  workspace/training/adapter_quality_history.jsonl  (cap 5000)
  workspace/healing/llm_drift_history.jsonl         (cap 1000)
  workspace/governance_proposals.jsonl              (cap 1000)
  workspace/proposed_experiments.jsonl              (cap 2000)
  workspace/training/retirement_proposals.jsonl     (cap 1000)
  ```
* F8 — `feedback_router._fetch_new_events` collapsed N+1 into single LEFT JOIN against `feedback.response_metadata`. msg_timestamp now comes back on each event row.

P2 (3):
* F9 — `listener_heartbeat` monitor now alerts on KNOWN_LISTENERS that have NO heartbeat (when subsystem is on). Catches "thread crashed before first touch" cases.
* F10 — `/commitment list|fulfilled|broken|deferred|unmute <id>` slash command. Operator can now respond to long_arc nudges via Signal — write through to `SelfState.active_commitments`.
* F11 — paper_pipeline ranking: embedding cosine against interest_model centroid (primary), LLM-self-rated relevance kept as tiebreaker.

21 follow-up tests. 287 → 364 cross-suite tests pass.

### 28.6 Phase G — 4 final companion gaps

Audit against the original Wave 2 personal-companion spec showed 4 gaps still open after F.

| ID | What | Lives in |
|----|------|----------|
| G1 | 72 h calendar horizon scan + conflicts + density-cluster detection. Daily 08:00 trigger; one Signal alert with overlaps + 3+ back-to-back warnings; quiet otherwise. | [`calendar_horizon.py`](app/life_companion/calendar_horizon.py) |
| G2 | Topic-level feedback weights — parallel to D4 workspace weights but bound to topics. Wired into `feedback_router._dispatch` (extracts mentions from comment vs current profile) and `interest_model._score_terms` (multiplies score per term). | [`topic_weights.py`](app/companion/topic_weights.py) |
| G3 | Topic-dormancy long-arc nudge — `interest_model` now appends per-pass timeseries to `interest_history.jsonl` (capped via `jsonl_retention`); daily check — peak in [60, 365] d > 1.0 AND last-14 d avg < 0.3 → "you were deep on X six months ago — still blocked?" Signal nudge. `/topic mute|unmute <name>` command. | [`topic_dormancy.py`](app/life_companion/topic_dormancy.py) |
| G4 | Finland-seasonal nudges — once-per-year triggers: first-frost watch, kaamos onset, winter solstice, polar-night-ends, Vappu, Juhannus warning. Location-gated to Finland. | [`seasonal_nudges.py`](app/life_companion/seasonal_nudges.py) |

26 tests. See `docs/LIFE_COMPANION.md` (all four added there).

### 28.7 New shared utilities (extracted during the cleanup)

* `app/utils/feed_parser.py` (E6) — stdlib RSS/Atom parser; both `paper_pipeline` and `personalized_digest` delegate.
* `app/utils/hash_embedding.py` (E7) — deterministic 256-d hashing-trick embedding; both `lessons_learned` and `llm_output_drift` delegate.
* `app/utils/jsonl_retention.py` (F7) — `cap_jsonl` + `append_with_cap`; 5 unbounded JSONLs now bounded.

### 28.8 Files touched (B → G aggregated)

```
app/companion/
├── feedback_router.py        # B3 + D4 + F3 + F8 + G2
├── feedback_weights.py       # D4
├── interest_model.py         # B1 + E4 + F1 + G2 + G3
├── lessons_learned.py        # D7 + E7 + F1
├── notify_meta.py            # B3
├── topic_weights.py          # G2 (NEW)
└── events.py                 # F1 (added iter_all_workspaces)

app/life_companion/
├── calendar_horizon.py       # G1 (NEW)
├── calendar_prep.py          # B2 + E3 + F-bugfix
├── daily_briefing.py         # F6 (added Topics section)
├── long_arc_follow_up.py     # B4 + F10
├── personalized_digest.py    # D5 + E6
├── seasonal_nudges.py        # G4 (NEW)
└── topic_dormancy.py         # G3 (NEW)

app/healing/
├── auditor_bridge.py         # (Phase A; unchanged in B-G)
├── boot_reset.py             # E12
├── db_backup.py              # (Phase A)
├── llm_output_drift.py       # D6 + E7 + E10
├── pattern_learner.py        # C4 + E2 + E9
├── silent_regression_detector.py  # C2
├── monitors/db_backup.py     # (Phase A)
├── monitors/db_vacuum.py     # (Phase A)
├── monitors/listener_heartbeat.py  # F9
├── monitors/log_archival.py  # D2
└── monitors/__init__.py      # cadence + wiring updates

app/training/
└── adapter_performance.py    # C1 + F2 + F7

app/episteme/
└── paper_pipeline.py         # C3 + E5 + E6 + F11 + F7

app/governance_ratchet/
├── auto_propose.py           # C5 + F7
└── goodhart_enforcing_proposer.py  # D3 + F7

app/personality/
└── signal_context.py         # B5 + F4

app/notify/api.py             # B3 + F3 (metadata sidechannel)
app/control_plane/db.py       # D1 (PG startup breaker)
app/runtime_settings.py       # E14 (defensive getattr reads)
app/agents/commander/commands.py  # F10 (/commitment) + G3 (/topic)
app/change_requests/lifecycle.py  # F5 (lessons KB consult)

app/utils/                    # NEW package (F7 + E6 + E7)
├── __init__.py
├── feed_parser.py
├── hash_embedding.py
└── jsonl_retention.py
```

### 28.9 Tests added (B → G aggregated)

```
tests/companion/
├── test_calendar_prep.py
├── test_feedback_router.py   # + F3 + F8 + G2 wiring tests
├── test_feedback_weights.py
├── test_interest_model.py
├── test_lessons_learned.py
├── test_long_arc_follow_up.py
├── test_notify_meta.py
├── test_personalized_digest.py
└── test_signal_context.py

tests/healing/
├── test_adapter_performance.py
├── test_db_backup.py         # (Phase A)
├── test_db_vacuum.py         # (Phase A)
├── test_evolution_archive.py
├── test_goodhart_enforcing_proposer.py
├── test_governance_auto_propose.py
├── test_listener_heartbeat_monitor.py  # + F9 missing-heartbeat tests
├── test_listener_heartbeats.py         # (Phase A)
├── test_llm_output_drift.py
├── test_log_archival.py                # (Phase A)
├── test_paper_pipeline.py
├── test_pattern_learner.py
├── test_pg_startup_breaker.py
├── test_silent_regression.py
├── test_phase_e_followups.py           # E1-E14 pin tests
├── test_phase_f_followups.py           # F1-F11 pin tests
└── test_phase_g_followups.py           # G1-G4 pin tests
```

374 tests pass cross-suite (healing + companion + concierge + fts5 + change_requests).

### 28.10 Master switches added

| Variable | Default | What it gates |
|---|---|---|
| `FEEDBACK_ROUTER_ENABLED` | `true` | B3 closed-loop drain |
| `COMPANION_FEEDBACK_WEIGHTS_ENABLED` | `true` | D4 workspace downweighting |
| `COMPANION_TOPIC_WEIGHTS_ENABLED` | `true` | G2 topic downweighting |
| `ADAPTER_PERFORMANCE_ENABLED` | `true` | C1 retirement proposer |
| `RETIREMENT_THRESHOLD` | `0.6` | C1 health-score threshold |
| `HEALING_SILENT_REGRESSION_ENABLED` | `true` | C2 throughput drift detector |
| `SILENT_REGRESSION_PCT` | `0.30` | C2 alert threshold |
| `PAPER_PIPELINE_ENABLED` | `true` | C3 weekly arXiv digest |
| `HEALING_PATTERN_LEARNER_ENABLED` | `true` | C4 unseen-pattern proposer |
| `GOVERNANCE_AUTO_PROPOSE_ENABLED` | `true` | C5 ratchet auto-proposer |
| `GOODHART_ENFORCING_PROPOSER_ENABLED` | `true` | D3 gate-flip proposer |
| `PERSONALIZED_DIGEST_ENABLED` | `true` | D5 weekly digest |
| `LLM_OUTPUT_DRIFT_ENABLED` | `true` | D6 semantic drift detector |
| `LESSONS_LEARNED_ENABLED` | `true` | D7 rejection clusterer |
| `CALENDAR_HORIZON_ENABLED` | `true` | G1 72 h scan |
| `TOPIC_DORMANCY_ENABLED` | `true` | G3 dormancy nudge |
| `SEASONAL_NUDGES_ENABLED` | `true` | G4 Finland triggers |
| `CONTROL_PLANE_CONNECT_TIMEOUT_S` | `8` | D1 PG connect timeout (seconds) |
| `DB_BACKUP_RETENTION_DAYS` | `30` | (Phase A) |
| `LOG_ARCHIVE_RETENTION_DAYS` | `90` | (Phase A) |

All defaults ON except backup-related (per-deploy operator choice).

### 28.11 What still needs operator action

* **Goodhart hard gate flip Advisory → Enforcing.** D3 watches the
  gate and proposes the flip when conditions warrant. The flip
  itself is operator-only via React `/cp/settings`. Wait at least
  14 days of Advisory observation before flipping.
* **Personalized digest config.** D5 reads
  `workspace/companion/personalized_feeds.json` for RSS feeds +
  GitHub username + arXiv author list. Empty by default. Operator
  curates once.
* **arXiv-following authors.** D5 + paper_pipeline read
  `ARXIV_FOLLOWING_AUTHORS` env (comma-separated). Set if you want
  per-author tracking.

### 28.12 Combined session totals (Phase A → G)

* 7 commits (342bd554, ab76da03, 1134dad1, 86770852, 72f3f6e9, bc7fe94d, bd3cbef8).
* ~13,000 lines of Python + ~3,400 lines of tests added.
* 374 cross-suite tests passing.
* 22 silent-failure bugs caught + fixed across the audit cycles E + F (would have shipped silently otherwise).
* 3 shared utilities extracted (`feed_parser`, `hash_embedding`, `jsonl_retention`).
* 18 new operator-tunable env switches; all default ON except backup.


## 29. 2026-05-10 — Pattern_learner triage (58 uncovered scaffolds → 7 PRs)

The morning's pattern_learner alert flagged **58 failure patterns
observed but not covered by any runbook (last 7 d, ≥10 occurrences)**.
Top five by volume: `30bbb7cd` (OpenRouter Stealth 502 ×630),
`834bbc74` (APScheduler "Run time was missed" ×613), `d2ae1dfd`
(Neo4j Unauthorized ×589), `787a4626` (llm_selector keep-incumbent
×147), `eb829b26` (Anthropic 400 credit-low ×143) — plus 53 more.

Triage showed the alert was misleading.  Most of the 58 weren't
"missing remediation handlers" — they were **log-level mistakes**
(stat-detector noise, by-design degradation, breaker state changes,
provider failover, validator rejections, startup info), **already-
handled patterns whose signatures had drifted**, or **operator-action
items that benefit from circuit-breaking, not retry**.  Writing 58
`if "<frag>" in sample: return …` shells would have just added
noise.

Instead the work split into three tiers shipped in seven PRs.

### 29.1 Tier 1 — log-level fixes (PRs #84 #85 #86)

The bulk of the noise was misclassified WARN messages — pattern_learner
flagged each new SHA-1 signature as a "new uncovered pattern" because
the f-strings included floating numerics that re-signed every outlier.

#### 29.1.1 PR #84 — `anomaly_detector` sigma-aware log level (T1.1)

Pre-fix: every `record_sample()` call that crossed the 2σ threshold
emitted `logger.warning("ANOMALY: …")`.  Each new outlier value
generated a new SHA-1 because `value`, `mean`, `stddev`, `sigma_dist`
are all in the message — pattern_learner saw 10+ "uncovered" patterns
that were really one stat-detector.

Post-fix:

```python
_WARN_SIGMA_THRESHOLD = 5.0
...
if sigma_dist >= _WARN_SIGMA_THRESHOLD:
    logger.warning(msg)
else:
    logger.info(msg)
```

`_alerts.append(alert)` is unchanged — the dashboard / API source-of-
truth still captures every event regardless of log level; only the
human-visible error stream changed.  4 tests cover the new
contract.

#### 29.1.2 PR #85 — APScheduler `misfire_grace_time=60` + coalesce (T1.2)

Pattern_learner reported **613 occurrences/week** of
`Run time of job '...' was missed by 0:00:03.389716` from
`apscheduler.executors.default` at WARNING (so it landed in
errors.jsonl).  APScheduler's default `misfire_grace_time` is
1 second — any cron job delayed by 3+ s during normal load
triggers the warning.

Fix at scheduler construction:

```python
scheduler = AsyncIOScheduler(
    job_defaults={"misfire_grace_time": 60, "coalesce": True},
)
```

60 s grace absorbs routine scheduling jitter; coalesce means
catch-up after a pause fires once, not N times.  Real overruns
(>60 s) still log at WARN — visibility into actual stalls preserved.

#### 29.1.3 PR #86 — 6 in-our-code WARN→INFO + JsonlNoiseFilter (T1.3)

Six log sites that described **by-design behavior** at WARNING:

| Site | Why it isn't WARN material |
|---|---|
| `llm_selector` keep-incumbent (147×) | Graceful degradation — caller asked for fresh-data, none qualified |
| `circuit_breaker` HALF_OPEN→OPEN | Breaker doing its job; first CLOSED→OPEN trip stays at WARN |
| `proposals.py` path-violation reject | Validator working as designed |
| `base_crew` tool-cap (post-init + pre-init) | Provider context-budget enforcement |
| `CreditAwareAnthropicCompletion` failover sync + async (285×) | Designed mid-call failover to OpenRouter |

All six demoted to **INFO**.  The first-trip CLOSED→OPEN transition
on the breaker stays at WARN — that's the actual operator-visible
signal.  Only the cyclic HALF_OPEN→OPEN re-trip is now INFO.

Plus a new `app/logging_filters.JsonlNoiseFilter` attached to the
JSONL handler (NOT the root logger) drops three known third-party
WARNs we already handle correctly downstream:

* discord.py "voice will NOT be supported" (PyNaCl/davey, ~1×/restart)
* Anthropic SDK 400 "credit balance too low" (143× — handled by
  CreditAwareAnthropicCompletion → OpenRouter failover)
* OpenRouter Stealth 502 "Invalid URL" (630× — eliminated at the
  source by T3.3; the filter is a backstop)

The filter is **hardcoded** — every entry is an explicit operator
decision that the message is informational.  New entries require
code review (no auto-grow path).  13 tests cover the demotions +
filter.

### 29.2 Tier 2 — signature-drift investigation (PR #87)

The numeric_overflow sample at `c38013f9929816242` was flagged as
"uncovered" but inspection showed `numeric_overflow_widen_cr` IS
correctly registered against that exact signature in
`app/healing/handlers/schema_drift.py`.

**Root cause** (already fixed on disk in commit `72f3f6e9`,
"Phase E"): `pattern_learner._registered_signatures()` imported
`_LOCK` from runbooks.py, but the actual symbol is `_registry_lock`.
The ImportError fell through `except Exception: return set()` and
the function silently returned an empty set every call — so
pattern_learner thought every covered handler was uncovered,
including `numeric_overflow_widen_cr` itself.

Phase E renamed the import; gateway just hadn't been redeployed yet.

PR #87 is the **regression guard** — a 5-test suite that exercises
the function against a real `_REGISTERED_RUNBOOKS` registry so a
future symbol rename can't re-introduce the silent-empty-set mode:

* `test_function_uses_real_lock_symbol` — source-grep that the
  import name lines up with runbooks.py
* `test_lock_symbol_actually_exists_in_runbooks` — runbooks-side
  mirror of the same check
* `test_hash_pattern_handler_is_seen_as_covered` — register a
  fake hash-pattern handler, confirm it lands in the covered set
* `test_catch_all_pattern_is_not_claimed_as_covered` — `.*`
  handlers must NOT swallow every signature
* `test_returns_non_empty_when_handlers_registered` — smoke for
  the silent-empty-set mode that started this whole thing

Verified by running the suite against the **still-live broken
gateway**: 3/5 fail — exactly the failure mode the suite targets.
After the next `docker compose build gateway && --force-recreate`
all 5 will pass.

### 29.3 Tier 3 — operator-action circuit breakers (PRs #88 #89 #90)

The remaining high-volume patterns weren't transient errors that
benefit from retry — they were **operator-action items** (rotate
the credential / fix the URL).  Hammering the upstream once-per-
minute serves only to fill errors.jsonl.

#### 29.3.1 PR #88 — `belief_outbox` Neo4j auth breaker (T3.1)

589 occurrences/week of:

```
belief_outbox: neo4j read failed:
  {neo4j_code: Neo.ClientError.Security.Unauthorized}
  {message: The client is unauthorized due to authentication failure.}
```

The reconciler ran every MEDIUM idle slot, hit the auth failure
each time, logged WARN each time.  The underlying issue (wrong
password / rotated credential) cannot be fixed by retry — only
the operator can rotate.

Fix shape (mirror of `anthropic_credits` from §17):

* New `neo4j_auth` operator-action breaker in `app/circuit_breaker.py`
  (`failure_threshold=1`, `cooldown_seconds=3600`).
* `_fetch_existing_neo4j_belief_ids()` short-circuits at entry when
  the breaker is OPEN — no round trip, no re-log.
* `_is_neo4j_auth_error(exc)` detects auth-failure variants
  (`Unauthorized` / `AuthenticationRateLimit` / `"client is
  unauthorized"`).
* First trip logs WARN once via the breaker's CLOSED→OPEN path
  (operator alert); subsequent attempts inside the cooldown
  short-circuit silently with the breaker's HALF_OPEN→OPEN at INFO
  (per T1.3).
* Transient (non-auth) errors **still log WARN** — connectivity
  issues stay visible.

Same PR also adds an `mcp_auth` template breaker, used by T3.2.

9 tests.

#### 29.3.2 PR #89 — `mcp_client` per-server 401/403 breaker (T3.2)

`mcp_client: 'STUzhy/py_execute_mcp' init failed: HTTP 401:
{"error":"invalid_token"}` — same pattern as Neo4j, but per-server
isolation matters: an auth failure on one MCP server must not
block connections to others.

Fix:

* `_is_mcp_auth_error(text)` detects 401/403/invalid_token/
  Unauthorized/Forbidden in connect/init error text.
* Per-server breaker key (`mcp_auth:<server-name>`).  Built on a
  new **`circuit_breaker.ensure_breaker()`** helper — same as
  `get_breaker` but takes explicit `failure_threshold` and
  `cooldown_seconds` so dynamic creates land with the
  operator-action shape (1 / 3600), not the generic 5 / 30.
* `_record_failure_log()` helper centralizes
  "auth → trip breaker + log INFO" vs "transient → log WARN" so
  the choice is in one place.
* connect() entry-check short-circuits when the per-server breaker
  is OPEN.

11 tests.

#### 29.3.3 PR #90 — Stealth filter on prefix-routed calls (T3.3)

**The single biggest source of pattern_learner noise**: 630
occurrences/week of:

```
OpenAI API call failed: Error code: 502 -
  {'error': {'message': 'Invalid URL: ', 'code': 502,
             'metadata': {'provider_name': 'Stealth'}}}
```

The provider-exclusion filter at `app/llm_factory.py:229` was gated
on `"openrouter.ai" in (base_url or "")` — but the bulk of our
OpenRouter traffic uses **prefix routing**
(`model_id="openrouter/deepseek/deepseek-chat"`) without an
explicit `base_url` kwarg.  litellm routes those calls via
`OPENROUTER_API_KEY`, so the filter trigger never fired and Stealth
was never excluded for prefix-routed calls.  That accounts for the
630/week.

Fix:

```python
_is_openrouter_call = (
    "openrouter.ai" in (base_url or "")
    or (model_id or "").startswith("openrouter/")
)
```

Active role-assigned models (Claude / Gemma / DeepSeek paid variants)
all have non-Stealth routes — no functional loss.
`OPENROUTER_IGNORE_PROVIDERS` env var still overrides (set to `""`
to disable, or to add other provider names).

This pairs with PR #86's `JsonlNoiseFilter` entry for the same 502
message: that filter prevents leakage when there's a residual case;
PR #90 eliminates the source.

7 tests.

### 29.4 Combined impact

Conservatively (assuming dedup overlap), this sweep eliminates
**~1,500–2,000 false-WARN entries/week** from `errors.jsonl`:

* ~10 anomaly-detector signatures × hundreds of outliers
* 613 APScheduler "missed by 3 s" warnings
* 285 + 147 in-our-code by-design WARNs (failover + keep-incumbent)
* 589 Neo4j auth retries → 1 WARN/h after first trip
* MCP-401 retries → 1 WARN/h per server after first trip
* 630 OpenRouter Stealth 502s eliminated at the source

Three new operator-action breakers (`anthropic_credits` from §17,
`neo4j_auth` and `mcp_auth` from this sweep) all share the same
1-failure / 1-h-cooldown shape — operators see one alert per
incident, not one per attempt.

### 29.5 Files touched

```
app/anomaly_detector.py                            # PR #84
app/main.py                                        # PR #85
app/llm_selector.py                                # PR #86
app/circuit_breaker.py                             # PR #86, #88, #89
app/proposals.py                                   # PR #86
app/crews/base_crew.py                             # PR #86
app/llms/credit_aware_anthropic.py                 # PR #86
app/logging_filters.py                             # PR #86 (NEW)
app/error_handler.py                               # PR #86 (filter wire-in)
app/memory/belief_outbox.py                        # PR #88
app/mcp/client.py                                  # PR #89
app/llm_factory.py                                 # PR #90

tests/test_anomaly_detector_log_level.py           # PR #84 (NEW)  4 tests
tests/test_scheduler_misfire_defaults.py           # PR #85 (NEW)  2 tests
tests/test_misclassified_warn_demotion.py          # PR #86 (NEW) 13 tests
tests/healing/test_pattern_learner_coverage.py     # PR #87 (NEW)  5 tests
tests/test_belief_outbox_neo4j_auth_breaker.py     # PR #88 (NEW)  9 tests
tests/test_mcp_client_auth_breaker.py              # PR #89 (NEW) 11 tests
tests/test_openrouter_stealth_exclusion.py         # PR #90 (NEW)  7 tests
```

Total: **51 new tests** across 7 files.

### 29.6 What still needs operator action

* **Pattern_learner backlog** at `workspace/proposed_runbooks/`
  still has 58 markdown scaffolds.  After the next gateway rebuild
  + 7-day re-scan window, the scaffolds for the patterns these
  PRs eliminated will stop being re-flagged.  Cleaning up the
  existing files is operator-discretion (move under
  `_noise/` or delete).
* **Gateway rebuild required** to land any of these runtime
  effects.  `docker compose build gateway && docker compose up
  -d --force-recreate gateway` is the recipe (confirmed working
  in §27.3).  Until then, the broken `_LOCK` import + the
  missing prefix-routing filter remain in production.
* **Neo4j credential rotation** still needed by the operator
  (the breaker just stops the hammering, not the underlying
  auth issue).  Same for MCP server `STUzhy/py_execute_mcp`
  token.

---

## 29. 2026-05-10 — Phase H — close the 8 silent-failure modes

The 24-hour Phase A→G sweep closed most of the years-of-uptime
audit but left four silent-failure modes from the original 8-item
list still open. Phase H closes them all in one commit.

The 8 silent-failure modes were:

```
1. Restore-from-backup is untested            HIGH    — pre-H: PARTIAL
2. Google OAuth refresh persists              MED→HI  — DONE pre-Phase A (refuted)
3. Signal 120-day idle re-registration        MED     — pre-H: NOT DONE
4. Firestore on_snapshot drops silently       HIGH    — DONE in Phase A #A3 (data path)
5. Sticky 1-hour cooldowns                    MED     — pre-H: PARTIAL (boot-only)
6. Disk growth unbounded                      HIGH    — DONE across A/D/F + Wave 2
7. Postgres-down on boot hangs gateway        HIGH    — pre-H: PARTIAL (control_plane only)
8. Vendor model sunsets blind                 HIGH    — DONE in Phase A #A4
```

After Phase H: **all 8 closed**.

### 29.1 H1 — restore-drill automation

**Files:**
* `deploy/scripts/restore-drill.sh` (NEW, 230 LOC).
* `app/healing/monitors/restore_drill.py` (NEW, 130 LOC).

`restore-drill.sh` is the operator-runnable quarterly drill:

1. Locates the freshest `all_ok` backup set in
   `workspace/backups/manifest.json` (Postgres + Neo4j + ChromaDB
   archives that all succeeded together).
2. Brings up an isolated compose project
   (`andrusai-restore-drill`) so it never touches the live stack.
3. Restores Postgres via `psql --set ON_ERROR_STOP=1`, Neo4j via
   `neo4j-admin database load`, ChromaDB via tar-into-volume.
4. Smoke-checks: count rows in `control_plane.audit_log`, count
   nodes in Neo4j, hit ChromaDB heartbeat endpoint.
5. Tears down the drill stack on success or failure (signal
   trap).
6. Updates `workspace/backups/restore_drill_manifest.json`.

The healing monitor runs daily and watches the manifest. Alerts:

* Manifest missing entirely → "no drill has ever run"
  (`restore_drill:never_run` tag).
* Most recent drill > `RESTORE_DRILL_STALE_DAYS` (default 100) →
  "stale" (`restore_drill:stale` tag).
* `last_drill_ok: false` → "FAILED" (same tag).

14-day per-tag dedup so the alert isn't noisy.

The monitor never RUNS the drill. Two reasons:

1. Compose-from-inside-a-container risks cross-resource issues
   (volume namespaces, port collisions, kill-the-parent on
   teardown).
2. The drill takes minutes; running it from a healing-monitor
   pass would block the monitor driver thread.

So the drill stays operator-scheduled (cron / launchd) and the
monitor stays an alerter only.

### 29.2 H2 — Signal 120-day re-registration keepalive

**File:** `app/healing/monitors/signal_keepalive.py` (NEW, 110 LOC).

Signal-cli registrations silently expire after ~4 months of zero
device activity. Detection only happens when the operator notices
replies missing — days or weeks later. The fix is a tagged
"note-to-self" message every 30 days to keep the registration
warm:

```
[andrusai-keepalive] 2026-05-10T01:23:45+00:00 — registration keepalive (~30d).
```

The `[andrusai-keepalive]` tag lets the operator filter or mute
the thread on their phone. signal-cli treats outbound-to-self as
valid keepalive traffic.

After 3 consecutive failed keepalives (~90 days) the monitor
Signal-alerts: "registration may be lost." Composes with the
existing `signal_heartbeat` (Wave 2 #3) multi-channel escalation
chain — Signal → PWA push → email after 7 fails.

### 29.3 H3 — idle-scheduler half-open retry

**File modified:** `app/idle_scheduler.py` (~80 LOC added).

The 1-hour cooldown after 3 consecutive failures was sticky:
transient outage at T+0 → the job stayed frozen until T+60min
even after the outage cleared at T+5min.

Now the cooldown allows probe attempts at 1/4, 1/2, 3/4 of the
window:

```
T+0       cooldown set, skip_until = T+1h
T+15min   probe 1 allowed → if succeeds, _clear_cooldown(). If fails, cooldown stays.
T+30min   probe 2 (if not yet succeeded)
T+45min   probe 3 (if not yet succeeded)
T+60min   cooldown elapses normally
```

`_clear_cooldown(name)` wipes:

* `_job_skip_until[name]` (in-memory + dbm via `_persist_clear_skip`)
* `_job_failure_counts[name]` → 0
* `_job_half_open_used[name]` (in-memory only — losing on restart
  is fine; probes re-arm afresh)

In-memory `_job_half_open_used: dict[name, set[float]]` tracks
which probe-points (0.25 / 0.5 / 0.75) have been consumed for the
current cooldown window. Each probe-point fires AT MOST ONCE per
cooldown.

### 29.4 H4 — Postgres-down on boot bounded retry for mem0

**File modified:** `app/memory/mem0_manager.py` (~70 LOC added).

The mem0 client's underlying `psycopg2.connect()` had no timeout.
A network-unreachable Postgres at boot could hang the gateway
indefinitely waiting on libpq's default connect timeout (~minutes
on Linux, longer on stuck DNS).

Two fixes:

1. **`_get_config()`** appends `connect_timeout=N` (env
   `MEM0_PG_CONNECT_TIMEOUT_S`, default 8 s) to the pgvector URL
   so libpq honours the cap.
2. **`get_client()`** retries `Memory.from_config()` 3× with
   1/3/9 s backoff before marking `_init_failed`. Signal alert on
   the first degraded boot ("init failed 3× — degraded boot, no
   persistent memory") so the operator hears about it.

Reproduces the Phase D #1 pattern from `app/control_plane/db.py`
for the OTHER Postgres consumer in the system. Both PG-using
modules now have bounded-retry boot paths.

### 29.5 Master switches added

| Variable | Default | What it gates |
|---|---|---|
| `RESTORE_DRILL_MONITOR_ENABLED` | `true` | H1 freshness alerter. |
| `RESTORE_DRILL_STALE_DAYS` | `100` | H1 stale threshold. |
| `SIGNAL_KEEPALIVE_ENABLED` | `true` | H2 30-day self-message. |
| `MEM0_PG_CONNECT_TIMEOUT_S` | `8` | H4 libpq connect cap. |

H3 has no env switch — it's a behaviour change in the existing
idle-scheduler cooldown logic, on for everything that uses the
scheduler.

### 29.6 Files touched

```
app/idle_scheduler.py                       # H3 — half-open probes
app/memory/mem0_manager.py                  # H4 — bounded retry
app/healing/monitors/__init__.py            # cadence dict + monitor wiring
app/healing/monitors/restore_drill.py       # H1 (NEW)
app/healing/monitors/signal_keepalive.py    # H2 (NEW)
deploy/scripts/restore-drill.sh             # H1 (NEW, executable)
tests/healing/test_phase_h_followups.py     # H1-H4 pin tests (NEW, 15 tests)
```

### 29.7 Tests + verification

15 new targeted tests in `tests/healing/test_phase_h_followups.py`:

* H4 — `_get_config` appends connect_timeout (via inspect on
  the source); 3-retry-then-Signal path on cold-boot failure.
* H3 — probe allowed at 0.25 fraction; three probes at 0.25/0.5/
  0.75; no probe right after cooldown set; `_clear_cooldown`
  wipes all three state stores.
* H2 — keepalive sends on first run; dedup within 30-day
  window; alert fires after 3 consecutive failures; respects
  master switch.
* H1 — alerts on missing manifest; alerts on stale; alerts on
  last-failed; quiet on recent-OK; 14-day dedup.

394 cross-suite tests pass (healing + companion + concierge +
fts5 + change_requests).

### 29.8 Operator action remaining

* **Schedule the restore drill.** Add to cron / launchd
  quarterly:
  ```
  @quarterly cd /path/to/crewai-team && bash deploy/scripts/restore-drill.sh
  ```
  The H1 monitor will Signal-alert at day 100 if no drill has
  run, so the system is self-reminding even if the cron entry is
  forgotten.

### 29.9 Combined session totals — Phase A → H

* 8 commits over 24 hours.
* ~14,000 lines of Python + ~3,500 lines of tests added.
* 394 cross-suite tests passing.
* 22 silent-failure bugs caught and fixed in the audit cycles E
  + F.
* 4 shared utilities extracted (`feed_parser`, `hash_embedding`,
  `jsonl_retention`, `commands._handle_*`).
* 22 new operator-tunable env switches; all default ON except
  backup-related (operator deploys with their preferred runner).
* All 8 silent-failure modes from the original years-of-uptime
  audit closed.


## 30. 2026-05-10 — Post-§29 redeploy verification (PRs #92–#96)

After §29's PRs #84–#90 landed and the gateway was rebuilt + force-
recreated, fresh symptoms surfaced that had been hidden by either
stale `__pycache__` (utils / audit shadows) or stale probe URLs
(Signal-cli / Host-bridge / Self-heal journal).  This section
documents the five PRs that closed those gaps plus the operational
Neo4j auth recovery, all within the same operator session as §29.

### 30.1 PR #92 — chat poller heartbeat consistency (T3.4)

After §29's redeploy, the listener_heartbeat monitor fired:

```
Self-heal: listener `firebase-chat-poll` has produced NO heartbeat
— known listener never started, or crashed before its first loop
iteration. Other listeners are healthy (heartbeat subsystem on).
```

`start_chat_inbox_poller` had a function-level early-return on
`not _firebase_enabled()`, so on a deployment with
`FIREBASE_ENABLED` unset (laptop dev, CI), the thread never started
and the heartbeat file never appeared.  The other 8 firebase
pollers (`mode` / `kb` / `phil` / `fiction` / `episteme` /
`experiential` / `aesthetics` / `tensions`) start their thread
unconditionally and heartbeat regardless of Firebase availability —
only the chat poller had the gate.

Fix: bring the chat poller in line with the consistent contract.
Heartbeat fires once before the first wait + at the top of every
loop iteration; `_get_db() == None` goes to `continue` (loop
survives), not `return` (which would have killed the thread on
the first Firestore hiccup).  Module-level `_chat_poll_stop` event
allows test isolation.  6 tests.

### 30.2 PR #93 — `app/utils` package-shadow regression (T3.5)

React Evolution Monitor → Genealogy tab failed with:

```
cannot import name 'now_iso' from 'app.utils'
(/app/app/utils/__init__.py)
```

Two things existed at the same import path:

  * `app/utils.py`             (legacy module — `now_iso`,
                                 `safe_json_parse`, `load_json_file`,
                                 `save_json_file`, `truncate`)
  * `app/utils/__init__.py`    (new package — docstring-only)

Python prefers packages over modules with the same name, so
`from app.utils import now_iso` failed at module-load time for
every caller — `variant_archive.py` had it at the top level
(broke the Evolution Monitor); ~25 other call sites had it inside
try/except and failed lazily.

Fix: merge the legacy module's contents into the package
`__init__.py`; `git rm app/utils.py`; re-export the three sibling
submodules (`feed_parser`, `hash_embedding`, `jsonl_retention`)
so `from app.utils import feed_parser` continues to work.  13 tests.

### 30.3 PR #94 — `app/audit` package-shadow regression (T3.6)

Same shape as T3.5, surfaced by the T3.5 rebuild.  Gateway
crashlooped on startup with:

```
ImportError: cannot import name 'log_tool_blocked' from 'app.audit'
(/app/app/audit/__init__.py)
```

The crash chain:

```
main.py:95 → from app.agents.commander import Commander
commander/__init__.py → from .orchestrator import Commander
orchestrator.py:12 → from app.tools.attachment_reader import …
attachment_reader.py:14 → from app.audit import log_tool_blocked
ImportError → uvicorn fails → container crashloop
```

**All inbound traffic blocked** until fixed.

**Why now?**  The shadow has existed since commit `3457e712`
(when `app/audit/` was added as a package).  Previous gateway
containers must have had stale `__pycache__/audit.cpython-313.pyc`
that masked the issue.  The T3.5 rebuild produced a fresh image
with no pyc cache, exposing the real Python 3.13 import-resolution
behavior — package wins, module is shadowed.

Fix: merge `app/audit.py` into `app/audit/__init__.py`;
`git rm app/audit.py`; both surfaces remain — six legacy
event-log helpers (`log_request_received`, `log_response_sent`,
`log_crew_dispatch`, `log_tool_call`, `log_tool_blocked`,
`log_security_event`) and the rolled-log primitive
(`RolledLogStore` / `RolledLogReader` / `RolledLogVerifier` /
`GENESIS` / `LEGACY_PREFIX` / `SegmentInfo` / `VerificationResult`).
16 tests.

Codebase audit confirmed no other module/package shadows exist —
T3.5 + T3.6 are the complete set.

### 30.4 PR #95 — Signal-cli + Host-bridge probes reflect runtime (T3.7)

Two false alarms in the React /cp/monitor System Monitor:

```
Messaging  ❌ ERROR  Signal-cli daemon  Endpoint not found — HTTP 404
Messaging  ⚠ WARN   Host bridge        host bridge disabled (BRIDGE_ENABLED=0)
```

Both reported failures while the underlying systems were fully
operational:

  * **Signal-cli probe** hit `GET /v1/about` — the path used by the
    *separate* signal-cli-rest-api Docker wrapper.  Vanilla
    signal-cli's `--http` mode is **JSON-RPC at `POST /api/v1/rpc`**.
  * **Host-bridge probe** gated on `settings.bridge_enabled`
    (env `BRIDGE_ENABLED=1`).  Since §27.3 the actual wiring uses
    per-agent `BRIDGE_TOKEN_<AGENT>` tokens via
    `bridge_client.get_bridge(...)` — `BRIDGE_ENABLED` is no
    longer the source of truth.

Fix: rewrite both probes.

  * Signal-cli: `POST /api/v1/rpc` with `{jsonrpc:2.0, method:version}`,
    parse JSON-RPC response, surface the version in the OK message:
    `daemon responding (+372... · v0.14.1)`.
  * Host bridge: try `bridge_client.get_bridge('change_requests')
    .is_available()` first.  Fall back to the legacy enable+raw-
    `/health` probe so laptop-dev setups with `BRIDGE_ENABLED=1`
    still surface a sensible status.

7 tests.

### 30.5 PR #96 — Self-heal journal probe time-windows the count (T3.8)

The Self-heal journal probe reported `50 recent errors, top:
coding:BadRequestError×16` indefinitely — even on installs where
**zero errors** had fired in the last 24 hours.

`get_recent_errors(50)` returns the last 50 entries from the FIFO
journal **regardless of timestamp**.  On the affected install,
those 50 entries spanned 2026-04-02 → 2026-04-28 with no entries
in the past 24 h.  But the journal-FIFO is sticky, so any install
with historical residue stayed in WARN forever.

Fix: filter by `ts` against a 24 h window.  Status logic:

  * 0 entries in 24 h, 0 historical → OK `no recent errors`
  * 0 entries in 24 h, N historical → OK `no errors in last 24h
                                          (N historical in journal)`
  * N entries in 24 h               → WARN `N errors in last 24h,
                                          top: <pattern>×<count>`

Top pattern is now computed over the WINDOW, not all-time, so the
operator sees what's actually firing now (not what fired last
month).  Pull a generous 200-entry slice so the time-cut isn't
truncated by FIFO ordering on busy days.  7 tests.

### 30.6 Operational fix — Neo4j auth recovery (no source change)

The 589/wk Neo4j Unauthorized failures had a real upstream cause
the T3.1 breaker couldn't fix (the breaker only stops the
hammering, not the auth itself).  Diagnosis:

  * `.env` has `MEM0_NEO4J_PASSWORD=V_sPgr8DI…`.
  * `docker-compose.yml` correctly passes
    `NEO4J_AUTH=neo4j/$MEM0_NEO4J_PASSWORD`.
  * BUT the Neo4j data volume (`workspace/mem0_neo4j/`) was first
    initialized with a *different* password.  Neo4j 5 reads
    `NEO4J_AUTH` ONLY on first init; subsequent starts use the
    on-disk auth state in the `system` database.  Even
    `neo4j-admin dbms set-initial-password` refuses to overwrite
    an initialized DB ("this change will only take effect if
    performed before the database is started for the first time").

Recovery (no data loss; preserves all 516 MB of graph):

  1. Stop the production Neo4j container.
  2. Start a temp container with `NEO4J_AUTH=none` against the same
     data volume on a different host port.
  3. Run `ALTER USER neo4j SET PASSWORD '<env value>'` against the
     `system` database via `cypher-shell -d system`.
  4. Stop the temp container; start the real one.
  5. Force-close the `neo4j_auth` circuit breaker so the
     `belief_outbox` reconciler resumes immediately.

Verified: `cypher-shell` returns `1` for the test row; the
System Monitor flips Neo4j from ERROR → OK.

### 30.7 Final System Monitor state (post-fixes)

```
       Containers     ok  PostgreSQL              connected · 106 budget rows
       Containers     ok  ChromaDB                connected · 57 collections
       Containers     ok  Neo4j                   connected           ← T3.1 + 30.6
       Containers     ok  Gateway HTTP            responding
        Messaging     ok  Signal-cli daemon       daemon responding (+372... · v0.14.1)  ← T3.7
        Messaging     ok  Host bridge             responding via per-agent token (change_requests)  ← T3.7
         Internal     ok  Idle scheduler          running · idle
         Internal     ok  Self-heal journal       no errors in last 24h (66 historical in journal)  ← T3.8
         Internal     ok  Budget reconcile        last write 2026-05-10 00:28:04
External services     ok  LLM provider credits    no active credit alerts
```

All-green.

### 30.8 Files touched

```
app/firebase/listeners.py                                  # PR #92
app/utils/__init__.py                                      # PR #93
app/utils.py                                               # PR #93 (DELETED)
app/audit/__init__.py                                      # PR #94
app/audit.py                                               # PR #94 (DELETED)
app/control_plane/dashboard_api.py                         # PR #95, #96
workspace/mem0_neo4j/dbms/auth.ini                         # 30.6 operational

tests/test_firebase_chat_poll_heartbeat_consistency.py     # PR #92 (NEW) — 6 tests
tests/test_utils_package_imports.py                        # PR #93 (NEW) — 13 tests
tests/test_audit_package_imports.py                        # PR #94 (NEW) — 16 tests
tests/test_system_monitor_probes.py                        # PR #95 (NEW) — 7 tests
tests/test_self_heal_journal_probe_timewindow.py           # PR #96 (NEW) — 7 tests
```

Total: **49 new tests** across 5 new test files.

### 30.9 Cumulative session totals (§29 + §30)

* 13 PRs merged (#84 → #96, contiguous, no gaps).
* ~1,500–2,000 false-WARN entries/week eliminated from
  `errors.jsonl` (§29.4).
* 4 false-alarm Monitor probes corrected (Signal-cli, Host bridge,
  Self-heal journal time-window, plus the chat-poller heartbeat
  contract).
* 2 Python module/package shadow regressions caught and fixed
  (T3.5 utils, T3.6 audit), with regression-guard tests so
  re-introducing the shadow fails CI.
* 1 operational data-volume auth recovery (Neo4j) without data
  loss.
* 100 new tests across 12 new test files
  (51 in §29 + 49 in §30).

### 30.10 What still needs operator action

* **`workspace/mem0_neo4j/dbms/auth.ini.backup-1778372423`** —
  the renamed-aside auth file from the 30.6 recovery.  Safe to
  delete once you're confident the new auth is stable.
* **Pattern_learner backlog** at `workspace/proposed_runbooks/`
  still has the 58 markdown scaffolds from §29.1's trigger.  After
  the next 7-day re-scan window, the scaffolds for the patterns
  these PRs eliminated will stop being re-flagged; cleaning up
  the existing files is operator-discretion.
* **PROGRAM.md duplicate `## 29.` heading** — a parallel agent's
  "Phase H" section landed at line 2527 with the same number as
  this session's pattern_learner triage (line 2218).  Renumber
  one of them when convenient; not blocking.


## 31. 2026-05-10 — Forest-age failure-mode triage (PRs #99–#101)

Operator-reported symptom (Signal):

```
"Please make a graphic about the change of forest age distribution
over time in Estonia. Use data that does not originate from
Estonian authorities"

[30 minutes elapsed]

"Sorry — the task stopped producing partial results (no new rows /
findings for several minutes).  I'll deliver what's been streamed
so far; please re-send a narrower question to fill the gaps."
```

The "narrow your question" advice was wrong-shaped for what
actually happened.  Tracing the gateway logs revealed three
**systematic** failure points, not one isolated bug.  The
operator's explicit guidance: "do not monkey-patch system to
perform this specific task. I need elegant solution. No hacks
or patchwork. need to cure the root of the issue."  This section
documents the three Tier-A architectural cures shipped in
contiguous PRs, each curing a class of failures (not the specific
forest task).

### 31.1 Evidence-based diagnosis

Three searches that returned **zero matches** told the story:

```
$ grep -rn finish_reason app/                    → 0 matches
$ grep -rn 'make.*graphic|deliverable.*verify|produce.*image' app/  → 0 matches
$ grep -rn vetting_fail.*record\|record.*failure_reason app/        → 0 matches
```

Three structural gaps:

  1. **LLM truncation invisible.**  The SDK's
     ``response.choices[0].finish_reason`` was never inspected.
     A response cut by ``max_tokens`` looked identical to a
     complete response from the orchestrator's perspective.
     The downstream LLM-vetter happened to catch the forest
     task's truncation by reading the text and noticing
     "ends at 'subtitle' with no closing" — fragile.
  2. **Artifact-producing tasks have no contract.**  The
     orchestrator's success contract was "the LLM returned
     non-empty text".  When the user asked for a PNG and the
     coder returned Python source code that *would*, if run,
     produce one, the orchestrator marked that as
     success-shaped.  Vetting (correctly) said "doesn't
     deliver the requested graphic" but vetting can't *make*
     a graphic.  Adjacent infra (``app/coding_session/``
     Phase 5.4) had the worktree + bounded subprocess runner
     but no router rule said "user asked for an artifact →
     route through execute-and-verify."
  3. **Watchdog message hides the actual failure reason.**
     The watchdog at ``app/main.py:1894-1900`` fired on a
     timing signal (no output-progress for 5 min) but
     generated messages that pretended it was a scope issue
     ("please re-send a narrower question").  The actual
     failure — vetting rejected with three specific reasons
     — sat in errors.jsonl with no path to the user-facing
     apology.

These compound: fixing only #3 = better error message but
same underlying failure; fixing only #1 = truncation gone but
agent still hallucinates dataset IDs (no execution feedback);
fixing only #2 = agent runs scripts but can still silently
truncate.  All three are needed for the failure class to
disappear.

### 31.2 PR #99 — Cure A: ``finish_reason`` guard at LLM-call boundary

New module ``app/llm_completion_guard.py``:

* ``CompletionTruncated(Exception)`` — typed; carries
  ``partial_text``, ``model``, ``max_tokens``, ``finish_reason``.
* ``check_completion_truncation(response, kwargs)`` — inspects
  ``response.choices[0].finish_reason``; raises on ``length`` /
  ``max_tokens``.  Robust against dict / object / degenerate
  shapes — returns silently when the SDK changes shape (we
  never convert SDK changes into outages).
* ``was_last_completion_truncated()`` — read-only ContextVar
  flag for observability paths.

Wired into ``app/rate_throttle.py:_throttled_completion`` (sync
+ async): every LLM call in the system funnels through this
wrapper, so a single check-call cures truncation everywhere.

Deliberately does NOT auto-continue: continuation across
token-budget boundaries is brittle for code generation; the
right cure is "raise a typed signal, let the orchestrator's
existing retry path bump max_tokens and re-run with clean
state."

Test coverage: 22 tests — extractor robustness, contract for
each finish_reason value, source-grep regression guards,
WARN-level logging.

### 31.3 PR #100 — Cure B: artifact-deliverable contract

New module ``app/agents/commander/artifact_intent.py`` with
three concerns:

1. **``classify_task(text) → TaskShape``** — pure heuristic
   (no LLM call).  Detects "make / produce / generate / create
   + graphic / chart / PDF / csv / png / ..." patterns AND
   explicit-extension mentions ("save as .pdf").  Conservative:
   requires both a verb AND a noun-or-extension signal, so
   statements like "the chart is interesting" don't trip it.

2. **``build_artifact_directive(shape)``** — when artifact-shape,
   returns a prompt block that tells the agent to use the
   existing ``coding_session_*`` tools (Phase 5.4), execute the
   code, and emit ``ARTIFACT: <relative-path>`` as the response
   format the verifier expects.

3. **``verify_artifacts(shape, response_text)``** — extracts
   paths from ``ARTIFACT:`` markers / backticked mentions /
   bare references; filters by allowed extensions AND allowed
   workspace roots (path-traversal-resistant); confirms each
   candidate is an existing non-empty regular file.  Raises
   typed ``ArtifactNotProduced`` with attempted-paths +
   per-path reasons.

Wiring (single-point-of-truth):

  * ``_handle_locked`` (entry) — classify once, propagate via
    ContextVar.
  * ``_run_crew_inner`` (start) — if artifact-shape, append
    directive to ``crew_task``.
  * ``_run_crew_inner`` (just before return) — if
    artifact-shape, call ``verify_artifacts``.  On failure,
    prepend a structured failure header to the result so the
    existing vetting layer sees a clear rejection — **no new
    failure pipeline**, the existing ``_build_retry_task``
    retry path picks up the precise reason.

Test coverage: 30 tests — classify (10), directive (2),
extract_artifact_paths (7), verify_artifacts (6), ContextVar
plumbing (2), 3 source-grep regression guards.

### 31.4 PR #101 — Cure C: watchdog surfaces last-known failure reason

Extension of ``app/observability/task_progress.py`` with a
per-task failure-context tracker:

* ``record_failure_context(kind, detail, *, task_id=None)`` —
  stash latest failure on the task (uses ContextVar when tid
  omitted)
* ``get_failure_context(task_id)`` — read latest, returns
  ``{kind, detail, age_s}`` or None
* Cleared by ``reset_task`` at request end
* kind / detail caps (64 / 500 chars) keep the apology bounded

Failure points wired in:

  * ``vetting.py`` — when verdict == FAIL, record
    ``kind="vetting_fail"`` with the structured issues list
  * ``orchestrator.py`` — when ``ArtifactNotProduced`` raises,
    record ``kind="artifact_missing"`` with shape details
  * ``llm_completion_guard.py`` — when finish_reason="length"
    hits, record ``kind="completion_truncated"`` with model +
    max_tokens

Watchdog reads context + weaves a per-kind explanation into
the apology.  New helper ``_format_failure_context_suffix(ctx)``
in ``app/main.py`` with templated suffixes for each kind plus
a generic fallback for unknown future kinds.

Apology paths updated (each in one place):

  * ``crew-zero-progress`` — generic + suffix when recorded
  * ``zero-output`` — generic + suffix when recorded
  * ``output-stall`` — when failure context exists, **REPLACES**
    the misleading "narrow your question" with the actual
    reason; falls back to original generic when no failure
    stored
  * ``stall`` (LLM-quiet) — generic + suffix
  * hard-cap — generic + suffix

The 30-min "narrow your question" message that hid three
vetting issues now reads:

```
Sorry — the task stopped after hitting the failure shown
below. The retry path didn't recover before the output-stall
threshold.

Last detected issue: the response was rejected by quality
review with the following reasons:
  truncated mid-function; hallucinated GEE asset ID; no
  graphic produced
Try re-sending with the gaps explicitly addressed (e.g.
missing data, incorrect identifiers, format mismatch).
```

Test coverage: 20 tests — tracker API (7), formatter (7),
4 source-grep wiring contracts.

### 31.5 Why the three cures compose

For the original forest-age task, **all three failure modes
were active simultaneously**.  Each cure addresses one;
together they make the failure class structurally impossible:

  1. Coder LLM generates script → **A** catches mid-function
     truncation, raises ``CompletionTruncated``, partial fed
     back as structured failure.
  2. Coder returns text-only "code that would generate a
     graphic" → **B** classifies the request as artifact-shape,
     requires an actual ``.png`` at a verified path, rejects
     with ``ArtifactNotProduced`` if missing.
  3. After retries exhausted → **C** weaves the actual reason
     into the apology — ``artifact_missing: expected .png;
     trigger=graphic`` or ``vetting_fail: ...``.

For ANY future "make X file" task:
  * If the LLM truncates a long script → immediate retry
    signal, not silent corruption.
  * If the agent doesn't actually run the code → rejection
    within seconds, not 30 minutes via vetting.
  * If retries eventually exhaust → apology names the actual
    cause every time.

### 31.6 Files touched

```
app/llm_completion_guard.py                                # PR #99 (NEW)
app/rate_throttle.py                                       # PR #99
app/agents/commander/artifact_intent.py                    # PR #100 (NEW)
app/agents/commander/orchestrator.py                       # PR #100, #101
app/observability/task_progress.py                         # PR #101
app/vetting.py                                             # PR #101
app/main.py                                                # PR #101

tests/test_llm_completion_guard.py                         # PR #99 (NEW) — 22 tests
tests/test_artifact_intent.py                              # PR #100 (NEW) — 30 tests
tests/test_watchdog_failure_context.py                     # PR #101 (NEW) — 20 tests
```

Total: **72 new tests** across 3 new test files.

### 31.7 Cumulative §29 + §30 + §31 totals

* 16 PRs merged contiguously (#84 → #101).
* 5 systematic failure points cured architecturally:
  ‑ LLM truncation invisibility (Cure A)
  ‑ artifact-deliverable contract (Cure B)
  ‑ watchdog message accuracy (Cure C)
  ‑ pattern_learner silent-empty fallback (§29)
  ‑ module/package shadow regressions (§30 utils + audit)
* 4 false-alarm Monitor probes corrected (Signal-cli,
  Host bridge, Self-heal journal, chat-poller heartbeat).
* 1 operational data-volume auth recovery (Neo4j).
* 172 new tests across 15 new test files (51 in §29 +
  49 in §30 + 72 in §31).

### 31.8 Verification (post-redeploy)

```
$ docker exec gateway python3 -m pytest \
    tests/test_llm_completion_guard.py \
    tests/test_artifact_intent.py \
    tests/test_watchdog_failure_context.py
72 passed in 20.22s
```

System Monitor: all-green.  Bridge available, Neo4j connected,
all 9 firebase listeners heartbeating, no errors in last 24 h.

### 31.9 What does NOT need operator action

This section deliberately has no open items.  All three cures
ship complete with regression-guard tests; the failure mode
that triggered the triage is structurally impossible to recur
without these tests failing CI first.

## 32. 2026-05-10 — Decade-class hardening initiative (§§1–9 roadmap)

Multi-week post-§31 program from a deep "ultrathink" audit that
asked: "what would make this system error-proof and resilient
for years of autonomous operation?" The audit produced a §1–§9
roadmap; everything below shipped over the past two days as
discrete revertible commits, all observational and additive,
none breaking the SubIA integrity boundary or the AE-1 anchor.

The §-numbers below refer to the audit roadmap, not PROGRAM.md
section numbers. Each line names the canonical module(s) and
points at the relevant tests.

### 32.1 Audit + governance primitives

**§2.1 — Cryptographic algorithm pinning + rotation drill.**
``app/audit/algorithm_pinning.py`` (manifest at
``workspace/audit/algorithm_pinning.json``; ``stale_pins`` with
730-day default; ``run_rotation_drill`` walks a sample chain
under both legacy + target algorithms; checks determinism and
that the runtime actually has the target algorithm — catches the
year-7 surprise of a target alias). Wrapped weekly by
``app/healing/monitors/crypto_rotation_drill.py`` with three
alert tags: ``missing_pins`` / ``stale_pins`` / ``drill_failed``.
``CRYPTO_ROTATION_DRILL_MONITOR_ENABLED`` master switch.
``CRYPTO_ROTATION_TARGET_ALGORITHM`` defaults to ``sha3_256``.

**§2.5 — Version-upgrade quarterly drill.** Forward-version
migration test scaffolded at
``app/healing/monitors/version_upgrade_drill.py``.

**§2.7 — Provider contract-drift weekly monitor.** Structural
shape-of-response check at
``app/healing/monitors/provider_contract_drift.py``.

**§2.8 — Identity continuity ledger.**
``app/identity/continuity_ledger.py``. Six event kinds
(``tier3_amendment``, ``governance_ratchet``, ``soul_edit``,
``integrity_regen``, ``scorecard_change``,
``self_quarantine_change``); append-only JSONL at
``workspace/identity/continuity_ledger.jsonl``. Closes the
multi-year-drift gap behind the Tier-3 amendment protocol —
the protocol gates *intentional* edits but didn't surface
*aggregate* drift across many small approved amendments. The
narrative-self FIFO holds 5 claims; this ledger is the long
record. ``record_event`` is failure-isolated.
``summarise_drift(window_days=365)`` yields by-kind / by-actor
counts as input to the §8.2 annual reflection.

Emission wired into four sites: ``app/governance_amendment/
protocol.py:mark_applied`` (tier3_amendment),
``app/governance_ratchet/protocol.py`` set/relax_ratchet
(governance_ratchet), ``app/subia/integrity.py:write_manifest``
(integrity_regen), and ``app/change_requests/lifecycle.py:
mark_applied`` when path matches ``app/souls/*`` or
``wiki/governance/constitution.md`` (soul_edit). Each emission
is wrapped in try/except so a missing ``app.identity`` package
never breaks the upstream operation.

### 32.2 Recipe + capability self-improvement

**§3.5 — Recipe consolidation.** Soft-retirement of
low-performing meta-agent recipes via
``app/self_improvement/meta_agent/consolidation.py`` (eager-
start daemon thread).

**§5.5 — Action-request primitive.** Operator-gated non-code
action workflow under ``app/action_requests/`` (validator +
JSONL store + lifecycle + pluggable handlers; first handler
``email_draft``). Surfaces via Signal 👍/👎 and React
``/cp/actions``.

**Capability-gap analyzer.** Clusters LearningGap stream emissions
into ArchitectureRequest drafts at
``app/self_improvement/capability_gap_analyzer.py``. Closes a
loop the Recovery Loop opened: refusals fed into the gap
detector now bubble up as proposals.

**Library radar.** ``app/library_radar/proposer.py`` filters
``tech_radar`` discoveries into framework-adoption proposals.
Eager-start daemon.

### 32.3 Creative synthesis primitives

**§7.1 — Cross-domain analogy index.**
``app/creativity/analogy_index.py``. Hofstadter-style structural
retrieval. ``AnalogyEntry`` packs a ``structure_signature``
("feedback loop with delay between cause and effect") with one
or more ``DomainExample`` witnesses across unrelated domains.
JSONL append-only; last-write-wins on duplicate id;
``query_analogies(text, top_k=5, min_similarity=0.05,
exclude_domains=…)`` runs hash-trick cosine over signature +
description for cross-domain retrieval.

**§7.2 — Concept-blend operator.**
``app/creativity/concept_blend.py``. Fauconnier-Turner four-
space blend. ``blend_concepts(input_a, input_b, llm_call=…)`` →
``BlendResult`` with input spaces, generic structure, blend
label + description, selected projections, emergent structure,
and follow-on questions. Strict JSON output with code-fence
tolerance; ``parse_failed`` flag with ``parse_error`` captures
malformed LLM output without raising.

**§7.4 — Brainstorm-idea novelty wrapper.**
``app/creativity/novelty_wrap.py``. Wraps every brainstorm idea
with a structural-novelty assessment that distinguishes "really
new" from "lexically rephrased."

### 32.4 Identity + philosophical reflection layers

**§8.2 — Annual value-reflection.**
``app/identity/annual_reflection.py``. Yearly essay over the
identity-ledger drift summary + narrative chapters + lessons-
learned KB. ``PhenomenalLanguageLinter`` post-filters with up-
to-3 retries on HARD_FAIL; output at
``wiki/self/value_reflections/<year>.md``. Never modifies
``current_goals``, SCORECARD, or any TIER_IMMUTABLE file.
Master switch ``ANNUAL_REFLECTION_ENABLED``.

**§8.5 — Legacy essay.** ``app/identity/legacy_essay.py``. The
most philosophical of the three identity layers. Yearly
"what about this self would I want preserved if terminated?"
essay at ``wiki/self/legacy/<year>.md``. Same neutral-language
linter discipline. Strictly observational — articulates
continuity-of-self preferences without those preferences being
load-bearing for any decision. Master switch
``LEGACY_ESSAY_ENABLED``.

Both passes registered as LIGHT idle jobs via
``app/identity/scheduler.py:get_idle_jobs`` — a daily fire is a
no-op 364 days/year (run_one_pass cadence-checks via
``_is_due``); the actual LLM work runs once per year per essay.
LLM call resolves via ``app.llm_factory.create_specialist_llm``
per-tick; a transient factory failure simply defers to the
next tick.

### 32.5 Earlier batches in the initiative

The decade-class initiative also produced (committed earlier in
the multi-week sweep, so listed here for completeness):

  * Rolled audit-log primitive (``app/audit/rolled_log.py`` +
    ``migration.py`` + ``journal.py``) — closes the silent
    FIFO-200 data-loss bug in ``audit_journal.json``.
  * Architecture-request primitive (``app/architecture_requests/``)
    — subsystem-granularity producer-side complement to
    change_requests; full Signal + REST + React surface.
  * ShinkaEvolve inline bridge (``app/coding_session/
    evolution_bridge.py``).
  * Weekly philosophical inquiry pass under
    ``app/subia/inquiry/`` (questions + selector + linter +
    composer + writer + scheduler).
  * Long-horizon thread primitive at ``app/threads/``.
  * Inquiry pass + Phase 11 neutral-language linter
    (``app/subia/inquiry/linter.py``) — mechanical guard
    against phenomenal-claim drift in identity-layer essays.

### 32.6 Verification

End-to-end suite for the new primitives runs clean::

```
$ pytest tests/identity/ tests/audit/test_algorithm_pinning.py \
         tests/creativity/test_analogy_index.py \
         tests/creativity/test_concept_blend.py \
         tests/governance_amendment/ tests/governance_ratchet/ \
         tests/test_change_requests.py
183 passed in 9.91s
```

SubIA integrity manifest regenerated post-edit (one Tier-3
file touched: ``app/subia/integrity.py`` got the
``integrity_regen`` emission hook).
``n_files=164`` unchanged. Butlin scorecard
``{STRONG=7, ABSENT={AE-2, HOT-1, HOT-4, RPT-1}, PARTIAL=3}``
unchanged. AE-1 anchor (``app/affect/goal_emitter.py``)
untouched.

### 32.7 What this initiative deliberately does NOT do

The roadmap had several P2/P3 items that were intentionally
left out as out-of-scope for "make it run for a decade":

  * No automatic Tier-3 amendment of governance.py FLOOR
    constants (deferred to a future monotonic-ratchet protocol).
  * No automatic application of the §8.2/§8.5 essay proposals —
    they're observational artefacts the operator reads.
  * No `scorecard_change` event emission yet — would need a
    before/after diff layer that's a separate primitive.
  * No `self_quarantine_change` event emission — the quarantine
    list is a static frozenset with no write API.

These are intentional: the discipline of the initiative is
"observational, additive, revertible." A subsystem that fires
proposals into the operator's gate is in scope; a subsystem
that auto-mutates identity-shaping state is not.


## 33. 2026-05-10 — Life-companion subsystem (PRs #103/#104) — bulk filter + act-now digest + control panel

Three contiguous PRs landed the life-companion email surface in
its current shape: a bulk-mail blindness fix that surfaced
marketing as urgent, a thrice-daily LLM-graded digest sibling
of the real-time monitor, and a React control panel that
exposes every life-companion job's on/off + tunables for
runtime override without a gateway restart.

### 33.1 PR #103 — email-triage bulk-blindness cure

Operator-reported on Signal:

```
📬 Email triage — 3 urgent unread:
  • DailyOM <today@dailyom.com> · score=2.5
  • Wild Gym <info@wildgym.com> · score=2.5
  • Swimmer.com.au <news@swimmer.com.au> · score=2.5
```

Three obvious marketing emails surfaced as "urgent". The
scorer's bulk-marker weights were correct (`-3` for
List-Unsubscribe, `-2` for List-ID, etc.) — but **two
architectural gaps in the input pipeline made them blind**:

1. `app/tools/gmail_tools.py:_list_recent` only requested
   `metadataHeaders=["From", "Subject", "Date"]` from the
   Gmail API.  List-Unsubscribe / List-Id / Auto-Submitted /
   Precedence / In-Reply-To / References were never fetched.

2. `app/life_companion/email_monitor.py:_build_headers`
   hardcoded every bulk marker to `None` even if the API had
   returned them, AND the scorer didn't recognize Gmail's
   tab-category labels (`CATEGORY_PROMOTIONS` / `SOCIAL` /
   `UPDATES` / `FORUMS`) — those were already in the stub
   but unread.

Each marketing email scored
`+1 (human From) + 1 (unread) + 0.5 (recent) ≈ +2.5` →
above the 1.0 threshold.

**Two-layer cure:**

* **Layer A** — `gmail_tools._list_recent` widens
  `_GMAIL_METADATA_HEADERS` to include the bulk + threading
  headers.  Free — same API call, more fields.  Stub now
  exposes them.

* **Layer B** — `EmailHeaders.gmail_labels: tuple[str, ...]`
  field + `_GMAIL_BULK_LABEL_PENALTIES` table:

  ```
  CATEGORY_PROMOTIONS = -4   (overrides +2.5 noise)
  CATEGORY_SOCIAL     = -3
  CATEGORY_UPDATES    = -2
  CATEGORY_FORUMS     = -2
  ```

  `score_email` picks the **strongest** matching label,
  doesn't compound (categories overlap; doubling would
  over-penalize legitimate Updates like flight changes /
  banking).

For non-Gmail providers (IMAP / Outlook), `gmail_labels` is
empty — bulk-marker headers carry the signal instead.
List-Unsubscribe alone is sufficient to drop a marketing
email below threshold.

Test coverage: 16 new tests; combined with 28 existing email
tests = **44/44 pass**, no regressions.  The 3 specific
operator-reported emails are tested by name in
`TestActualOperatorReportedEmails`.

### 33.2 act-now email digest — sibling of email_monitor

Operator request (Signal):

> "I want additionally following e-mail format running every
> three hours from 7am to 22pm: bring me top 7 e-mails I need
> to act on now! Have system to analyse last 48 hours unread
> emails and decide based on content whether those are
> important or not. Provide links to email."

Implemented as `app/life_companion/act_now_digest.py` —
deliberately NOT a modification of `email_monitor`; the two
co-exist and serve different purposes:

| | `email_monitor` | `act_now_digest` |
|---|---|---|
| Cadence | Every ~10 min | Every 3 h, 07–22 local |
| Lookback | All unread | Last 48 h unread |
| Ranking | Heuristic (no LLM) | **LLM content analysis** |
| Output | Top 3 above 1.0 | **Top 7 act-now items** |
| Per-item | sender / subject / score | + **why** / **action** / deadline / **Gmail link** |
| Job | Real-time noise filter | Thoughtful synthesis |

**Cadence implementation:** fires at six fixed slots
(07/10/13/16/19/22 local) with ±15 min tolerance.  Slot key
`YYYY-MM-DD-HH` prevents re-firing within the same window.

**Pipeline:**

1. `_fetch_unread_with_bodies(48h, max_n=30)` — Gmail
   full-read for body content
2. `_pre_filter` drops `CATEGORY_PROMOTIONS / SOCIAL /
   FORUMS`.  Keeps `CATEGORY_UPDATES` (flight changes /
   banking / package tracking can be act-now)
3. `_rank_with_llm` — single Sonnet call, structured JSON
   output, validates each `email_id` against candidates
   (drops hallucinated ids silently)
4. `_heuristic_fallback` when LLM unavailable — ranks by
   `email_importance.score_email`; digest still ships
5. `_format_digest` — Signal message with rank / sender /
   subject / why / action / deadline / Gmail link

**Sample output:**

```
✉️ Top 7 act-now emails — last 48h (14 unread → 9 after bulk-filter):

1. CFO Sarah Chen <sarah@acme.com>
   Q3 board deck — sign-off needed by EOD
   why: explicit deadline today
   action: review draft + reply with sign-off
   deadline: EOD today
   📨 https://mail.google.com/mail/u/0/#inbox/abc123
```

Cost: ~$0.40/day on Sonnet (6 runs × 30 emails × 500-token
excerpts).  Tunable via env-style knobs.

Test coverage: 28 tests — cadence/slot, pre-filter,
Gmail-link, digest format, LLM-call (well-formed /
hallucinated id / malformed / empty), prompt-block, idle-
scheduler wiring contract.

### 33.3 PR #104 — React control panel for jobs on/off + tunables

Operator request: a `/cp/` surface to manage life-companion
parameters — turn each job on/off and edit its env-var-
shaped tunables — **without a gateway restart**.

#### Architecture (single source of truth)

```
   ┌──────────────────────────────────────────┐
   │ app/life_companion/feature_registry.py   │  declare features +
   │ (immutable schema-as-data, no logic)     │  tunables in ONE place
   └──────────────────────────────────────────┘
                    │ used by ↓
   ┌──────────────────────────────────────────┐
   │ app/runtime_settings.py                  │  store overrides at
   │ life_companion_overrides schema          │  workspace/runtime_settings.json
   └──────────────────────────────────────────┘
                    │ consulted by ↓
   ┌──────────────────────────────────────────┐
   │ app/life_companion/_common.py            │  feature_enabled() +
   │   feature_enabled / get_tunable          │  get_tunable() — no module
   │ (override > env > default)               │  imports os.getenv directly
   └──────────────────────────────────────────┘
                    │ exposed by ↓
   ┌──────────────────────────────────────────┐
   │ GET/POST /config/life_companion          │  registry + overrides as
   │ (config_api.py — bearer-auth on POST)    │  one payload; validates
   │                                          │  feature_key + tunables
   └──────────────────────────────────────────┘
                    │ rendered by ↓
   ┌──────────────────────────────────────────┐
   │ dashboard-react/src/components/          │  card per feature:
   │   LifeCompanionPage.tsx                  │  toggle + tunables +
   │                                          │  source pills
   └──────────────────────────────────────────┘
```

**Resolution order** (lowest priority last):

1. Master switch `LIFE_COMPANION_ENABLED` — kills everything
   when false (env-only, requires restart)
2. **Per-feature override** (this control panel) — persisted;
   survives restart
3. Per-feature env var (boot default)
4. Registry default

#### Why a registry, not module-level constants?

* **Cross-cutting use** — React + override-setter both need
  the same shape; one declaration removes "where does the UI
  know about `LIFE_COMPANION_ACT_NOW_TOP_K`?" from the answer.
* **Schema-as-data** — tunable types + bounds + defaults
  travel with the metadata; UI renders typed inputs with
  min/max validation.
* **Discoverable** — adding a new feature is one entry here.
  UI picks it up automatically on next page load.

#### Sentinel pattern in `life_companion_set_feature_override`

Three distinct semantic paths for `enabled`:

* **Omitted** (default `_LEAVE_UNTOUCHED` sentinel) — leave
  toggle override untouched; useful when only tunables changed.
* **`None`** — clear the toggle override; revert to env.
* **`bool`** — set the override explicitly.

The HTTP API maps these onto `"enabled" not in body` /
`"enabled": null` / `"enabled": <bool>` respectively.  Pre-
sentinel-fix: both omitted and `null` collapsed to
`enabled_arg=None` → couldn't clear an override.

#### React UX

Each feature card shows:

* Header — name + description + **source pill** (override / env
  / default) + on/off toggle + reset button
* Tunables — typed inputs (number / select / text) with
  min/max/default hints; per-tunable source pill
* Dirty-tracking — "Save tunables" only enabled when unsaved
  edits exist
* Inline error from last mutation
* Master-switch warning banner when `LIFE_COMPANION_ENABLED=
  false` is detected

10 features registered out of the box: email monitor,
act-now digest, daily briefing, routines, long-arc, calendar
prep, personalized digest, calendar horizon, topic dormancy,
seasonal nudges.

#### Wiring discipline

`act_now_digest` and `email_monitor` switched from
`os.getenv` → `get_tunable` for their tunable accessors.
Verified live:

```
POST {TOP_K: "5"}  →  _top_k() in running gateway returns 5
POST {TOP_K: ""}   →  _top_k() returns 7 (registry default)
```

No restart between the POST and the next read.  That's the
whole point of the override mechanism.

### 33.4 Files touched (this section's PRs)

```
# PR #103
app/tools/gmail_tools.py                                 # widened metadataHeaders
app/tools/email_importance.py                            # gmail_labels field + penalties
app/life_companion/email_monitor.py                      # _build_headers populates fields
tests/test_email_triage_bulk_blindness.py                # 16 new tests

# act_now_digest (commit 04b57e1b — fast-forwarded by parallel agent)
app/life_companion/act_now_digest.py                     # NEW
app/life_companion/__init__.py                           # idle-scheduler registration
tests/life_companion/test_act_now_digest.py              # 28 new tests

# PR #104
app/life_companion/feature_registry.py                   # NEW (single source of truth)
app/runtime_settings.py                                  # +life_companion_overrides
app/life_companion/_common.py                            # +get_tunable, override-aware feature_enabled
app/life_companion/act_now_digest.py                     # tunables via get_tunable
app/life_companion/email_monitor.py                      # tunables via get_tunable
app/api/config_api.py                                    # GET/POST /config/life_companion
dashboard-react/src/api/endpoints.ts                     # lifeCompanion endpoint
dashboard-react/src/api/queries.ts                       # useLifeCompanionQuery + mutation
dashboard-react/src/components/LifeCompanionPage.tsx     # NEW
dashboard-react/src/App.tsx                              # /life-companion route
dashboard-react/src/components/Layout.tsx                # 🌿 nav-item
tests/test_life_companion_control_panel.py               # 18 new tests
```

Total: **62 new tests** across 3 new test files for this
section's three PRs.

### 33.5 Verification (post-redeploy)

```
$ docker exec gateway python3 -m pytest \
    tests/test_email_triage_bulk_blindness.py \
    tests/life_companion/test_act_now_digest.py \
    tests/test_life_companion_control_panel.py
62 passed in 0.31s

$ curl -s http://localhost:8765/config/life_companion | jq .master_enabled
true
$ curl -s http://localhost:8765/config/life_companion | jq '.features | length'
10

$ curl -s -X POST -H "Authorization: Bearer …" -H "Content-Type: application/json" \
    --data '{"feature_key":"act_now_digest","tunables":{"LIFE_COMPANION_ACT_NOW_TOP_K":"5"}}' \
    http://localhost:8765/config/life_companion
{"status":"ok","overrides":{"act_now_digest":{"tunables":{"LIFE_COMPANION_ACT_NOW_TOP_K":"5"}}}}

$ docker exec gateway python3 -c \
    "from app.life_companion.act_now_digest import _top_k; print(_top_k())"
5     ← override picked up immediately, no restart
```

System Monitor: all-green.  Bridge available, Neo4j connected,
all 9 firebase listeners heartbeating, no errors in last 24 h.

### 33.6 Operator-action items

* **Adding a new life-companion feature later** — one entry
  in `app/life_companion/feature_registry.py`.  Three things
  travel with the entry: feature key + env-var pair + tunable
  schema.  UI auto-renders.  No React changes required.
* **No-restart pickup verified** — overrides flow through
  `get_tunable` to the per-job accessors.  Each `run()` reads
  the latest value on the next tick.  Test:
  `tests/test_life_companion_control_panel.py::TestActNowDigestUsesTunableHelper`
  enforces this contract via source-grep so a future refactor
  can't silently regress to `os.getenv`.
* **Dashboard route**: `/cp/life-companion` (sidebar entry
  🌿 Life Companion, between Files and Settings).

## 34. 2026-05-10 — Personal-data ingestion (§§5.1, 5.4) + post-ship audit

Closes the two open §5 items from the decade-class audit roadmap and
walks through the 13-finding correctness + elegance pass that
followed.

### 34.1 §5.1 Apple Health ingestion

Personal health data parsed from the user's own iPhone Health export
into typed JSONL streams the daily/evening/weekly briefings can
reason over.

**Modules** (`app/health/`, ~1,100 LOC):

* `types.py` — typed records (HeartRateRecord, StepsRecord,
  ActiveEnergyRecord, BodyMassRecord, SleepRecord, WorkoutRecord)
  with stable field names; HKQuantity identifiers stay at the parser
  boundary.
* `import_apple.py` — bounded-memory `iterparse` walker with
  `root.remove(elem)` after each yield (true bounded memory across
  decade-scale exports). Apple's `2026-05-09 17:30:00 +0300` strings
  become ISO-8601 UTC. Zip extraction happens inside a
  `tempfile.TemporaryDirectory()` context manager — always cleaned
  up. Failure-isolated `ImportResult` with status fields; never
  raises.
* `store.py` — append-only JSONL per kind at
  `WORKSPACE_ROOT/health/<kind>.jsonl` (env-overridable via
  `HEALTH_BASE_DIR`). Dedupe on `(start_iso, source_version)` so
  re-importing the same export adds zero records. Public
  `resolve_base()` lets sibling modules (idle_job, briefing) derive
  paths from the same source of truth.
* `summary.py` — `summarise_window(days=7)` rolls up: mean HR, p10
  resting HR proxy, total + per-day steps, total + per-day active
  kcal, latest body mass + window delta, mean sleep hours per night,
  workout count + distance.
* `anomaly.py` — recent-vs-baseline z-score (default 3 days vs 30
  days, |z| ≥ 2.0) for resting HR, steps/day, sleep hours/night.
  Returns observational `HealthAnomaly` records — never auto-routes.
* `idle_job.py` — once-per-~24h `health-summary` LIGHT job. Marker
  file at `<HEALTH_BASE_DIR>/.last_summary_at`. Logs structured
  output the daily-briefing composer reads.

**Privacy invariants** (load-bearing — health data is high-leverage):

  - **No external API calls.** No ChromaDB embedding, no LLM
    inference over raw records. The composer in
    `app/life_companion/daily_briefing.py` only sees summary
    statistics (mean / total / z-score), never individual records.
  - **Default-OFF.** `HEALTH_INGESTION_ENABLED=false` until the
    user explicitly opts in.
  - **Append-only.** No deletion path; `rm -rf
    $WORKSPACE_ROOT/health/` is the only reset.
  - **Idempotent re-import.** Dedupe key absorbs duplicate exports.

**Access path** — there is no programmatic way for a Mac/Linux process
to read iPhone HealthKit (it's iOS-only, requires an iOS app with
explicit consent). The only realistic path is **manual export**:
iPhone Health app → profile picture → "Export All Health Data" →
AirDrop/iCloud Drive/email to host → drop the resulting
`apple_health_export.zip` into `workspace/inbox/` (which the §5.4
inbox watcher routes to the importer automatically). Documented at
`docs/HEALTH_INGESTION.md`.

### 34.2 §5.4 Multi-modal inbox ingestion

Unified file-drop interface for the personal-agent surface. The user
drops anything into `workspace/inbox/`; the watcher classifies +
routes + archives.

**Modules** (`app/inbox/`, ~600 LOC):

* `classifier.py` — extension + magic-bytes classifier. PNG / JPG /
  HEIC / WEBP / PDF / MP3 / M4A / WAV / OGG / FLAC all checked
  against per-format magic bytes (per-format byte offsets handled —
  HEIC and M4A signatures sit at byte 4). Apple Health zip detection
  peeks the zip index for `apple_health_export/export.xml` (with
  filename heuristic as fallback). Module-load assertion enforces
  `_EXTENSION_MAP.values() ⊆ KNOWN_KINDS`.
* `router.py` — `scan_and_route()` walks the inbox once. SHA-256
  hash; 5-second quiet window for partial writes; dedup against
  `.processed/<sha>.json`; classify; dispatch to handler; archive
  on success at `.archive/<YYYY-MM-DD>/`; failures stay in place
  with a manifest recording the reason. Three first-class handlers:
  `_handle_apple_health` (composes with §5.1), `_handle_text` (drops
  to canonical `WORKSPACE_ROOT/notes/` — listed by
  `app/api/files_api.py`'s `/cp/files` view), and `_handle_unsupported`
  for recognised-but-not-yet-routable kinds.
* `scheduler.py` — `inbox-tick` LIGHT idle job + `_maybe_notify`
  surfacing. Apple-Health imports + any failures + unrecognised
  files trigger a single `notify(title, body, url="/cp/files")`
  push; routine text drops stay silent (push spam is worse than no
  push).

**Master switch**: `INBOX_INGESTION_ENABLED` (default OFF).
**Composability**: drop `apple_health_export.zip` → §5.4 routes to
§5.1 importer → daily briefing picks up next morning. End-to-end
without CLI invocation.

Documented at `docs/INBOX_INGESTION.md`.

### 34.3 Post-ship audit (4595cfab + 6c57640c review)

Same-day audit produced 13 findings — mix of correctness gaps,
inelegances, and open loops the user would actually feel. All
addressed in the same 48-hour window (commit `22138b21`):

**Open loops (highest impact):**

1. `summarise_window()` wired into `daily_briefing.py` morning /
   evening / weekly composers via new `_gather_health_summary()`.
   The "❤️ Health (7d)" section appears once data exists; soft-fails
   to no-section before opt-in so the pre-§5.1 briefing reads
   unchanged.
2. `_handle_text` writes to canonical `WORKSPACE_ROOT/notes/` (the
   path `files_api.py` already lists), not the orphaned
   `notes/inbox/<YYYY-MM-DD>/` no module read.
3. `_maybe_notify` in `scheduler.py` fires `notify()` on
   Apple-Health-import-success, failures, or unrecognised files;
   silent on routine text drops.

**Correctness:**

4. `_DEFAULT_LAST_RUN` now derives from `store.resolve_base()` —
   honors `HEALTH_BASE_DIR`.
5. `tempfile.TemporaryDirectory()` + `_resolve_xml` context manager
   replaces 8-line manual rmtree.
6. `iterparse` captures the root and `root.remove`s processed
   children — true bounded memory.
7. `source_uuid` → `source_version` (Apple's field is the OS / firmware
   version string, not a UUID). Renamed across types/store/import/tests.

**Defense-in-depth + polish:**

8. `_MAGIC_SIGNATURES` table covers all 10 binary formats with per-
   format byte offsets.
9. Apple Health zip detection by zip-peek (filename heuristic only as
   fallback for unreadable zips — the importer's own `failed_zip`
   branch then surfaces the reason).
10. Module-load `assert set(_EXTENSION_MAP.values()) <= KNOWN_KINDS`
    blocks typos.
11. Sleep start-date attribution documented in summary.py +
    anomaly.py (session-merge layer is a future improvement).
12. `cur_date - timedelta(days=i)` instead of fromordinal/toordinal.
13. `_BUILDERS` dispatch table replaces 5-arm if-else;
    `_check`/`_flag_if_anomalous` closures replace three near-
    identical metric blocks in `detect_anomalies` (~50% LOC).

**Verification post-fix:** 264 tests pass (health 37 + inbox 32 +
identity 49 + algorithm_pinning 19 + creativity 31 + governance_*
56 + change_requests 40). SubIA `n_files=164` and butlin
`{STRONG=7, ABSENT={AE-2, HOT-1, HOT-4, RPT-1}, PARTIAL=3}`
unchanged. AE-1 anchor untouched.

### 34.4 Files touched

```
app/health/{__init__,types,store,import_apple,summary,anomaly,idle_job}.py
app/inbox/{__init__,classifier,router,scheduler}.py
app/life_companion/daily_briefing.py     # _gather_health_summary + 3 composer hooks
app/companion/loop.py                    # health + inbox idle-job registration
docs/HEALTH_INGESTION.md                  # operator guide (§5.1 access path)
docs/INBOX_INGESTION.md                   # operator guide (§5.4 watcher)

tests/health/{test_anomaly,test_briefing_integration,test_import_apple,
              test_store,test_summary}.py
tests/inbox/{test_classifier,test_router,test_scheduler}.py
```

### 34.5 What §34 deliberately does NOT do

* **No continuous sync.** No iOS Shortcut hook, no HealthKit-bridge
  helper. The user runs the manual export; the inbox watcher takes
  it from there. Out of scope for this initiative — would need a
  separate iOS-side artefact.
* **No dashboard for health data.** The data is viewable in Apple's
  own Health app on the phone; the briefing is the system's
  consumer. A React panel for raw records would invite tempting
  privacy violations (filtering, sharing, exporting elsewhere).
* **No cross-source health (Garmin, Oura, ...).** Only Apple Health
  today. Adding a new source means one new parser + one new entry
  in `_EXTENSION_MAP` + one new handler in `HANDLER_REGISTRY` — the
  primitive supports it; nobody asked for it yet.
* **No automatic intervention on anomalies.** The anomaly detector
  flags; the operator decides. This is consistent with the rest of
  the consciousness-layer discipline — observational, never load-
  bearing for any decision.

## 35. 2026-05-10 — Dashboard surface expansion + public HTTPS

Three new operator surfaces, one transport hardening, and a focused
fix for the iOS PWA Web-Push pipeline. All landed within a few hours
on top of §26.

### 35.1 Chat tab (`/cp/chat`) — Signal mirror with markdown

`POST /api/cp/chat/send` routes the message through
``Commander.handle()``, the same dispatch path Signal uses, so every
slash command, recovery loop, project router, lifecycle hook, and
affect attachment hook fires identically. ``GET /api/cp/chat/messages``
reads conversation history from ``conversation_store`` (oldest→
newest so the React side can append-render directly). ``GET
/api/cp/signal-commands`` returns the hand-curated catalogue
defined in ``app/agents/commander/command_registry.py`` —
85 commands across 13 categories, paired with the dispatcher in
``commands.py`` by convention rather than introspection.

Frontend ``ChatPage.tsx`` is a split layout: scrolling history with
markdown bubbles (react-markdown + remark-gfm + remark-math +
rehype-katex + rehype-highlight, same renderer the Notes view uses)
and a filterable command sidebar; clicking a row inserts the syntax
into the composer.

Two paired bugs fixed before the tab actually showed history:

  1. **Sender mismatch.** Signal stores under
     ``HMAC(phone_number)``; the chat tab passed
     ``sender="andrus"`` which hashed differently. Added
     ``_resolve_chat_sender`` mapping ``andrus / owner / primary /
     me / user`` → ``settings.signal_owner_number`` so the chat
     tab and Signal share one bucket.
  2. **`ts` column type.** Newer ``messages.ts`` rows are ISO
     strings (legacy ones were floats); the helper did
     ``float(r[2])`` and crashed on every modern row. The route's
     broad try/except swallowed the ``ValueError`` and returned
     ``[]``. Helper now detects numeric vs string and parses ISO
     via ``fromisoformat().timestamp()``.

### 35.2 Monitor tab (`/cp/monitor`) — comprehensive system status

Single aggregator endpoint ``GET /api/cp/system-status`` probes
every major surface with a soft 1–2 s deadline and returns a flat
list of status rows the React page groups by category. Probes are
uniform — ``_probe(name, category, fn)`` measures latency, catches
exceptions, runs them through ``_interpret_error()`` (translates
401/402/403/429/5xx + ``connection refused`` / ``timeout`` /
``unauthorized`` / ``quota`` into one-line operator hints), and
calls ``_credit_link_for()`` to surface the matching top-up URL
when an exception matches a credit-exhaustion pattern. Pulls
active credit alerts from ``firebase.publish._active_alerts``;
reuses the existing ``_CREDIT_URLS`` map so OpenRouter / Anthropic
/ OpenAI / Google all surface their billing pages directly on the
page.

Probes:

  * Containers: PostgreSQL (`SELECT 1` + budget row count),
    ChromaDB (``list_collections``), Neo4j (``RETURN 1`` via cached
    driver), gateway HTTP (implicit "you're reading me").
  * Messaging: signal-cli daemon (GET ``signal_http_url`` +
    ``/v1/about``), host bridge (config-aware: warn when
    ``BRIDGE_ENABLED=0``).
  * Internal: idle scheduler (currently running job), self-heal
    journal (recent error count + top pattern, **24h window** —
    fixed in T3.8 because the original "all-time" count flagged
    "12 recent errors" on a healthy system), budget reconcile
    (last write timestamp).
  * External services: per-provider credit alerts, each carrying
    its top-up link for a "Top up →" button.

Frontend ``MonitorPage.tsx`` is a headline strip (overall
ok/warn/error + per-category counts), a pinned credit-alert
call-out visible only when a provider is actually exhausted, and
one card per category with two-column status rows. Each row shows
level pill, name, latency, interpreted message, and an "Open →"
link.

T3.7 fixed the Signal-cli + Host-bridge probes which previously
hard-coded URLs and ignored runtime config (would always show
"timed out" even when a daemon was running on a non-default port).

### 35.3 Public HTTPS via Tailscale Funnel + multi-mode auth

iOS Web Push only works in a Secure Context — the Web-Push
``PushManager`` is silently stripped by every browser when the
origin is plain HTTP from a non-localhost host. The user's
home-screen PWA was reaching the dashboard at
``http://10.0.0.54:3100/cp/`` and saw ``serviceWorker=no /
pushManager=no`` for that reason. Fix path:

  1. **Tailscale Funnel** terminates HTTPS for the dashboard at
     ``https://plgs-macbook-pro---andrus.tail5b289b.ts.net/`` →
     proxy to ``http://localhost:3100``. Free Let's-Encrypt cert,
     auto-renewing. Setup is two CLI commands once the admin-
     console toggles are flipped:

         tailscale cert plgs-macbook-pro---andrus.tail5b289b.ts.net
         tailscale funnel --bg --https=443 http://localhost:3100

     The `tailscale serve --tcp=8765` rule that previously exposed
     the bare gateway tailnet-wide was removed in the same pass —
     Funnel-fronted dashboard is now the only external entrypoint.

  2. **HTTP Basic Auth** in front of the public Funnel route
     (`dashboard/server.mjs`). Reads ``DASHBOARD_USER`` +
     ``DASHBOARD_PASS`` from the sibling ``.env`` (same pattern
     ``loadGatewaySecret`` already uses). Loopback (``localhost``
     / ``127.0.0.1`` / ``::1``) bypasses auth so laptop-localhost
     dev keeps working.

  3. **Cookie auth** alongside Basic for iOS PWA standalone mode.
     Basic-Auth credentials saved into the regular Safari keychain
     are not always inherited by the home-screen icon — every
     standalone-mode launch hit the 401 wall again. New one-shot
     login URL:

         GET /cp/login?token=<DASHBOARD_PASS>
           matches  → 302 /cp/ + Set-Cookie dashboard_auth=...; Max-Age=1y
           mismatch → 401

     Cookie value is
     ``HMAC(DASHBOARD_PASS, "dashboard-auth-v1")``. No server-
     side state; rotating the password invalidates every existing
     cookie. Cookie path is ``/`` so it covers ``/api/``,
     ``/config/``, ``/epistemic/``, ``/affect/`` and every other
     proxied prefix. ``requireAuth`` accepts EITHER cookie OR
     Basic; iOS users visit the login URL once, get the cookie,
     and the home-screen PWA inherits it on every launch.

### 35.4 Web Push diagnostic upgrade

The Settings-tab "PWA notifications" card used to show "This
browser doesn't support Web Push. Install the PWA and try again."
for every failure — actively misleading on iOS where the PWA WAS
installed but the launch context wasn't standalone, or the origin
wasn't secure. Replaced with a four-way diagnostic that prints
the actual reason:

  * ``isSecure=no`` (LAN HTTP / Tailscale name without HTTPS) →
    points at Tailscale Funnel + Cloudflare Tunnel + localhost as
    practical fixes.
  * iOS + ``isStandalone=no`` → "Tap the home-screen icon, not the
    URL".
  * iOS + standalone but no PushManager → "iOS 16.4+ required".
  * No serviceWorker → "Try a Chromium- or WebKit-based browser".
  * No PushManager → "Probably private-browsing mode".

A diagnostics line below the message shows each capability flag
(``iOS / secure / standalone / serviceWorker / pushManager``) so
the user sees at a glance exactly which prerequisite failed
without opening DevTools.


## 36. 2026-05-10 — Forest-age regression (PRs #106/#107) — Cure B + Cure C follow-ups

The operator re-issued the same forest-age request that had
triggered §31's three-cure architectural sweep:

```
1:22 PM  📨 "Please make a graphic about the change of forest age
            distribution over time in Estonia.  Use data that does
            not originate from Estonian authorities"
1:35 PM  🤖 [ARTIFACT VERIFICATION FAILED]
            ARTIFACT: /tmp/estonia_forest_age_distribution.png
            ...
1:39 PM  📨 "Please make average age of forest graph since 2000…"
2:24 PM  🤖 "Sorry — the task stopped producing partial results
            (no new rows / findings for several minutes).
            …please re-send a narrower question to fill the gaps."
```

Both messages were operator-visible failures that §31's cures
were SUPPOSED to make impossible.  Investigation found three new
architectural bugs — one in Cure B, two in Cure C — shipped as
PRs #106 and #107.

### 36.1 PR #106 — Cure B's path-root allow-list was too narrow

**Symptom**: the agent emitted `ARTIFACT: /tmp/estonia_forest_age_
distribution.png` and the file *actually existed* at that path
(353 KB PNG, real Hansen GFC visualization).  The verifier
rejected it and the operator received a misleading
`[ARTIFACT VERIFICATION FAILED]` header.

**Root cause**: §31's `_ALLOWED_ROOTS` allow-list was
(`workspace/`, `output/`, `/tmp/crewai-`, `/app/workspace/`,
`/app/output/`).  `/tmp/` (without the `crewai-` prefix) wasn't
on the list — `extract_artifact_paths` filtered the path out at
the root check, then the verifier reported "no file path
mentioned" because no candidates survived.  The allow-list was
PARANOID early gating that turned out to generate false
positives — rejecting valid artifacts because they lived in
unfamiliar directories the registry hadn't anticipated.

**The lesson**: existence + non-empty + extension are the REAL
safety gates.  The root filter was over-engineering.

**Cure**: replace `_ALLOWED_ROOTS` allow-list with `_DENIED_ROOTS`
deny-list — block obviously-malicious roots (`/etc/`, `/proc/`,
`/sys/`, `/dev/`, `/boot/`, `/root/`) where an artifact PNG has
no business living, otherwise let the existence check be the
authoritative gate.

Also improved the error message: when `extract_artifact_paths`
returns empty, re-scans WITHOUT the extension filter and records
any path-shaped tokens with their rejection reason.  Pre-fix
the verifier said "no file path mentioned" even when paths were
mentioned but rejected — distinguishing the two makes retry-path
diagnostics precise.

Test coverage: 4 new tests (added to existing 30 in
`test_artifact_intent.py` → 34/34 pass) including
`test_operator_reported_tmp_artifact_passes` — the EXACT Signal
trace from 13:35 with a real file on disk; verifier now accepts
and returns the path.

Live verified post-redeploy.

### 36.2 PR #107 — Cure C had two race-shaped bugs

The 13:39 follow-up task stalled with the LEGACY generic
"narrower question" message — even though the gateway logs
showed vetting had recorded three specific issues:

```
INFO vetting[full]: coding FAILED: [
  "The claimed artifact file /app/output/estonia_forest_avg_age_2000_2024.png
   does not exist; the response asserts the graph was generated but
   no artifact was delivered.",
  "Inconsistent dataset statistics …",
  "Logical issue in age model description …",
]
```

The watchdog should have woven these into the apology.  It
didn't.  Two bugs:

#### Bug 1 — ContextVar doesn't propagate to `_ctx_pool`

`record_failure_context` defaults its `task_id` from the
`current_task_id` ContextVar.  `_handle_locked` sets it in the
commander thread.  But the orchestrator submits vetting to
`_ctx_pool` via `ThreadPoolExecutor.submit` — which does **NOT**
propagate ContextVars by default.  So when `vet_response_detailed`
calls `record_failure_context` in a worker thread,
`current_task_id.get()` returns the default empty string and the
function silently no-ops.

**Cure**: wrap the submit in `contextvars.copy_context().run`.
Standard Python pattern for "submit a callable while preserving
the calling thread's ContextVars".

```python
import contextvars as _cv
_vet_ctx = _cv.copy_context()
_vet_future = _ctx_pool.submit(
    _vet_ctx.run,
    vet_response_detailed, user_input, _synthesis_result, …,
)
```

#### Bug 2 — Race between vetting and watchdog

When vetting takes >90s, the future-level timeout fires.  The
vetting thread KEEPS RUNNING (the future was abandoned, not
cancelled) and eventually calls `record_failure_context` — but
by then the watchdog already fired its apology with no context
to weave in.

**Cure**: when the orchestrator catches the vetting timeout,
record an explicit `vetting_timeout` failure context
synchronously, in the orchestrator thread (where ContextVar IS
set) BEFORE the future is abandoned.  Plus: add a matching
`vetting_timeout` template to `_FAILURE_CONTEXT_TEMPLATES` in
`main.py` so the suffix formatter renders actionable text
instead of falling to the generic "unknown kind" branch.

#### Tests

6 new tests in `tests/test_cure_c_followups.py` including:

  * `test_record_propagates_through_copy_context` — **POSITIVE**:
    with copy_context.run, ContextVar propagates and record
    lands under the right tid
  * `test_record_silently_noops_without_copy_context` —
    **NEGATIVE**: proves the bug exists when propagation is
    missing (the failure mode the operator hit)
  * Source-grep regression guards locking each wire-in so a
    future refactor that drops them fails CI loudly

### 36.3 What §36 illustrates about §31's cures

§31 was an attempt at a "cure the root cause, not the symptom"
sweep — three Tier-A architectural fixes for a class of forest-
age failures.  §36 is the honest follow-up: even the
architectural cures had bugs of their own.  Specifically:

  * Cure B's allow-list was paranoid — it generated false
    positives on legitimate artifact paths.
  * Cure C's record-pipeline didn't account for ContextVar
    semantics across `ThreadPoolExecutor.submit` boundaries.
  * Cure C's watchdog read happened on a race with vetting's
    record write.

The fix discipline matters.  Both PRs ship with negative tests
that demonstrate the bug on the UNFIXED code path AND positive
tests showing the fix works.  Source-grep regression guards lock
the wire-in points so future refactors that drop them fail CI
loudly.

### 36.4 Files touched

```
# PR #106
app/agents/commander/artifact_intent.py    # _ALLOWED_ROOTS → _DENIED_ROOTS
tests/test_artifact_intent.py              # 4 new tests, 34/34 pass

# PR #107
app/agents/commander/orchestrator.py       # copy_context.run + vetting_timeout record
app/main.py                                # _FAILURE_CONTEXT_TEMPLATES +vetting_timeout
tests/test_cure_c_followups.py             # NEW — 6 tests
```

10 new tests across the two PRs.

### 36.5 Operator-action items

* **None for the cures themselves** — both shipped with
  regression-guard tests so the failure modes can't recur
  without CI failing first.

* **Coder-side reliability gap (separate scope)**: the 13:39
  follow-up surfaced a different problem that's not architectural —
  the agent claimed `ARTIFACT: /app/output/estonia_forest_avg_age_
  2000_2024.png` but the file didn't actually exist.  Cure B's
  existence check now catches this precisely (post-redeploy)
  and Cure C surfaces the reason via the templated suffix —
  but the underlying coder behavior ("claim artifact without
  actually running the script that produces it") would benefit
  from an "actually run before reporting" discipline at the
  coder level.  That's its own future PR.

## 37. 2026-05-10 — Boot-wiring audit + structural fixes

Operator-prompted "ultrathink" audit of *whether everything that
needs to be triggered at boot is actually triggered at boot*. The
documented inventory in CLAUDE.md is large — 3 reconcilers, 5
observational idle jobs, 22 healing monitors, auditor bridge,
watchdog, life-companion's 3 jobs, identity reflections, etc.
Cross-referenced every entry against `app/main.py`'s lifespan,
`idle_scheduler._default_jobs`, `companion.loop.get_idle_jobs`, and
`app/healing/__init__.py`. **Headline: every documented job is
registered.** The risk is not missing wiring — it's *how* one
load-bearing chunk gets wired.

### 37.1 The transitive-import fragility (HIGH)

`app/main.py:96` did `from app.healing.error_diagnosis import
diagnose_and_fix`. That side-effect-imports `app/healing/__init__.py`,
which side-effect-imports `monitors`, `auditor_bridge`, and
`watchdog` — each calling `start()` at module level. The
`__init__.py` itself flagged the fragility ("nothing in the boot
chain (main.py:96 → app.healing) was previously importing them — so
their daemons never ran in production"). It then piggybacked 3
*non-healing* subsystems on the same import
(`capability_gap_analyzer`, `library_radar`, `proposal_bridge`).

A refactor that lazy-imported `error_diagnosis` would have silently
disabled the entire 22-monitor driver, the auditor bridge, the
watchdog, and three observational subsystems — with no test
catching it.

**Fix**: explicit `import app.healing  # noqa: F401` at
`app/main.py:96`, with a 12-line comment documenting the
load-bearing side effects. The healing surface is now anchored
explicitly; refactors of unrelated imports cannot disable it.

### 37.2 Orphan top-level entry point (MEDIUM)

`crewai-team/main.py` (386 lines) defined a duplicate FastAPI app
that registered ONLY `self_improve`, `workspace_sync`, `heartbeat`,
and a Discord start. Nothing referenced it (Dockerfile,
entrypoint.sh, docker-compose.yml, run_host.py all use
`app.main:app`), but the file was a footgun for any operator who
copied a deploy script and ran `uvicorn main:app` from
`crewai-team/`.

**Fix**: replaced 386-line orphan with a 24-line stub —
docstring + `raise RuntimeError(...)` at module top — so any
attempt to load it fails immediately with an actionable error
message pointing at `app.main:app`.

### 37.3 `_publish_schedule()` only saw cron jobs (MEDIUM)

`_publish_schedule()` in `app/main.py` read `scheduler.get_jobs()`
(APScheduler only) and pushed the result to Firestore for the
dashboard. The ~95 idle-scheduler jobs are a separate registry and
were never published. Dashboard "scheduled jobs" view saw only
~14 cron jobs.

**Fix**:
* Added `_active_jobs_snapshot` module state + `list_jobs()`
  public accessor to `app/idle_scheduler.py`. `start()`
  populates the snapshot as `(name, weight)` tuples (callable
  dropped — not serialisable, not interesting to readers).
* `_publish_schedule()` now merges idle jobs into the report
  (`id="idle:<name>"`, `cron="idle:<weight>"`,
  `next_run=None`).
* Re-called `_publish_schedule()` immediately after
  `idle_scheduler.start()` so the dashboard sees the full
  ~95-job picture, not just the ~14 cron jobs.

### 37.4 CLAUDE.md / SELF_HEAL_V3.md doc drift (MEDIUM + LOW)

Three drifts corrected:

1. **"Runtime-toggleable master switches via /cp/settings
   Self-heal-subsystems card"** — overclaim. Only
   `error_runbooks_enabled` and `tool_supervisor_enabled` are
   runtime-toggleable (verified at `runtime_settings.py:13-14,
   87-88`). `HEALING_MONITORS_ENABLED`,
   `HEALING_AUDITOR_BRIDGE_ENABLED`, `HEALING_WATCHDOG_ENABLED`
   are env-only and require gateway restart (the daemons
   start at module import). CLAUDE.md and SELF_HEAL_V3.md both
   updated to say so explicitly.
2. **Monitor count 10 → 22** — the v3 ship was 10 monitors;
   12 more were added in subsequent passes
   (`silent_regression_detector`, `pattern_learner`,
   `llm_output_drift`, `signal_keepalive`, `restore_drill`,
   `version_upgrade_drill`, `provider_contract_drift`,
   `db_vacuum`, `log_archival`, `db_backup`,
   `crypto_rotation_drill`). Architecture diagram in
   SELF_HEAL_V3.md updated; CLAUDE.md monitor-list updated.
3. **Tool supervisor double-gating** — CLAUDE.md described it
   as "wraps every CrewAI native-tool dispatch". Actual: the
   supervisor wraps `available_functions` only inside
   `LoadableAgentExecutor`, so it fires only when *both*
   `TOOL_SUPERVISOR_ENABLED=true` *and* the calling agent runs
   on the LoadableAgent path. Standard CrewAI executor
   bypasses it regardless of the flag. CLAUDE.md and
   RECOVERY_LOOP.md §17 updated.

### 37.5 Verified clean (no fix needed)

* All 3 reconcilers (belief-outbox-neo4j, belief-outbox-chroma,
  dlq-drain) registered.
* All 5 observational idle jobs (decentered-pass,
  valve-audit-replay, wiki-index-reconciler,
  viability-goal-emitter, backward-counterfactual-replay)
  registered.
* All 4 continuity-ledger emission sites present
  (`governance_amendment.protocol.mark_applied`,
  `governance_ratchet.protocol.{set,relax}_ratchet`,
  `subia.integrity.write_manifest`,
  `change_requests.lifecycle.mark_applied`).
* Discord client launched as gateway background asyncio task at
  `app/main.py:811`; clean shutdown at `app/main.py:819`.
* Web Push subscription auto-prune is inline on send (410-Gone
  path at `app/web_push/sender.py:74-76`) — correct design, no
  idle job needed.
* All env-flag defaults match CLAUDE.md (life-companion ON,
  health/inbox OFF, error-runbooks/tool-supervisor OFF, etc.).

### 37.6 Files touched

```
crewai-team/main.py                           # neutralised orphan (386 → 24 lines)
crewai-team/app/main.py                       # explicit healing import, _publish_schedule extension, re-call
crewai-team/app/idle_scheduler.py             # _active_jobs_snapshot + list_jobs()
CLAUDE.md                                     # tool supervisor row, Self-Heal v3 row, healing/monitors row
crewai-team/docs/SELF_HEAL_V3.md              # intro overclaim, monitor diagram, master-switch table, anchor note
crewai-team/docs/RECOVERY_LOOP.md             # §17 supervisor double-gating
crewai-team/PROGRAM.md                        # this entry
```

No changes to TIER_IMMUTABLE files. No agent-modifiable code paths
touched. All changes additive/correctional — no behaviour change for
running gateways (the explicit-import is a no-op for the loaded
process).


## 38. 2026-05-10 — Closing the observational-proposal loop + Q1/Q2 self-heal

Body of work spanning a single multi-session ultrathink push: closing
the gap from "system noticed something" to "system did something
about it." Five threads, each shipped end-to-end with tests + boot
anchor + production verification.

### 38.1 Proposal bridge — unify three observational paths under one CR-emitting helper

`capability_gap_analyzer`, `library_radar/proposer`, and
`paper_pipeline` each generated markdown drafts that dead-ended at
`docs/proposed_*/<sig>.md` waiting for an operator to file CRs by
hand. The closure was unreliable: drafts piled up, the operator
never saw them, the loop never closed. New
[`app/proposal_bridge/`](app/proposal_bridge/) package — `store.py`
(idempotent staging with body-hash dedup) + `promoter.py` (24h
daemon that promotes stable proposals to CRs after a 7d cooldown,
rate-limited to 3/pass) — provides the single staging surface all
three sources now feed.

Lifecycle: `STAGED → CR_FILED → APPLIED|REJECTED|EXPIRED`. CR
validation rejection terminates as `REJECTED` (mirrors operator
rejection so the producer's signature dedup honours it) — earlier
draft used `EXPIRED` which created a producer/promoter loop on
permanent failures. Terminal proposals self-clean after a 14-day
audit retention window. Per-pass promotion publishes weighted
outcomes to the SubIA Global Workspace
(`app.workspace_publish.publish_to_workspace`) — operator decisions
at salience 0.30–0.35 (above the 0.3 ignition threshold), promotions
0.25, housekeeping 0.02–0.05.

19 tests in [`tests/proposal_bridge/`](tests/proposal_bridge/),
all passing. Producers' existing tests updated to verify the bridge
path instead of direct disk writes.

### 38.2 Wiki-index reconciler — three-layer root-cause fix

The reconciler had filed 335 CRs in a month, ALL bouncing with the
same `validation_failed` reason: `wiki/` not in the validator's
`_ALLOWED_ROOT_PREFIXES`. Deeper analysis revealed a SECOND
underlying bug: `_compute_master_index_content` writes today's date
into the body line `Total pages: N | Last updated: YYYY-MM-DD`, but
`_normalize_for_hashing` only stripped the frontmatter `updated_at:`
field — so canonical (computed with pin date 2000-01-01) hashed
differently from live (today's date) on every single run, producing
false-positive drift detections.

Fixed in three independent layers:

  1. [`app/change_requests/validator.py`](app/change_requests/validator.py) — added `wiki/` to
     `_ALLOWED_ROOT_PREFIXES` (TIER_IMMUTABLE check still applies).
  2. [`app/memory/wiki_index_reconciler.py`](app/memory/wiki_index_reconciler.py) — `_normalize_for_hashing`
     now strips both frontmatter AND body date lines.
  3. [`app/memory/wiki_index_reconciler.py`](app/memory/wiki_index_reconciler.py) — defensive
     `_existing_cr_blocks_filing` skips new filings when a
     non-terminal CR already exists for `wiki/index.md`, OR when a
     recent (≤7d) rejected CR has identical content.

355 stuck CRs archived to
`workspace/change_requests/archive/2026-05-10_wiki_validation_failed/`
with `MANIFEST.md`. Active queue post-archive: 2 CRs (1 legitimate
auditor_bridge applied + 1 legitimate wiki drift awaiting operator).

### 38.3 Auto-apply CR infrastructure (future-facing capability)

Shipped the `auto_apply_risk_class` validator + lifecycle as
infrastructure with empty allowlists — the capability is dormant
until an operator deliberately opts in. New `RiskClass` enum,
`DecisionSource.SELF_HEAL_AUTO_APPLY`, two new ChangeRequest fields
(`risk_class`, `origin_pattern_signature`). `validate_auto_apply()`
enforces standard validation + forbidden prefixes (memory / souls /
governance / migrations / deploy / host_bridge) + requestor
allowlist + path allowlist + 20-line cap + additive-only.
`auto_approve()` skips operator gate after enforcing rate limits
(3/pattern/day, 10 global/day), publishes loud Signal alert,
applies via existing `apply_change`, registers with the auto-revert
watcher.

[`app/change_requests/auto_revert.py`](app/change_requests/auto_revert.py) — 60s-poll daemon with
30-min watch window. Any recurrence (≥1 increase from the apply-time
baseline) of `origin_pattern_signature` within the window triggers
`rollback_change`. Outside the window, the watch entry unregisters
(success).

After deep analysis, the allowlists ship EMPTY in production. None
of the candidate patterns considered (cost_mode_inject,
embed_misroute_block, numeric_overflow_widen,
regression_test_generator, etc.) cleanly satisfy the five design
principles required for safe auto-apply (pattern-signature
isomorphism, single-site fix, sub-30-min recurrence cadence, no
downstream import dependencies, inverse-diff sufficient). Most
"system fixes itself" cases fit better as runbook auto-actions
(§38.5) than as auto-apply CRs.

[`docs/AUTO_APPLY.md`](docs/AUTO_APPLY.md) is the operator handover —
4-step activation procedure, P1–P5 design principles, future-shape
templates. 30 tests in [`tests/test_change_requests_auto_apply.py`](tests/test_change_requests_auto_apply.py).

### 38.4 Q1 — Unclog the CR pipeline (5 items)

| # | Item | Result |
|---|---|---|
| 1 | `db_pool_reset.min_recurrence` 5 → 1 | one-line config flip in [`workspace/self_heal/runbook_settings.json`](workspace/self_heal/runbook_settings.json). Pool exhaustion at instance #1 is already actionable; threshold of 5 meant runbook never triggered. RateLimiter (1800s) + success-rate gate (≥50%) + concurrency cap remain as safety nets. |
| 2 | Auto-apply infrastructure | §38.3 above. |
| 3 | Triage 336 stuck CRs | §38.2 archived 355. |
| 4 | `TIER3_AMENDMENT_ENABLED=true` + producer wiring | Two-part fix: flag flip in [`workspace/runtime_settings.json`](workspace/runtime_settings.json) + new agent tool [`app/tools/request_tier3_amendment.py`](app/tools/request_tier3_amendment.py) (CrewAI BaseTool wrapping `propose_amendment` with operator-friendly error messages and Signal alert) + new daemon [`app/governance_notifier.py`](app/governance_notifier.py) (sits OUTSIDE TIER_IMMUTABLE `governance_amendment/`, polls every 6h, detects state transitions, sends per-state Signal alerts + GW publish, opportunistically calls `advance_cooldown`/`advance_monitoring` on time-based gates). |
| 5 | Goodhart Advisory→Enforcing prep | New observability tool [`app/observability/goodhart_advisory_report.py`](app/observability/goodhart_advisory_report.py) — CLI + library that aggregates `goodhart_guard`'s signal log into "would have blocked" projections, severity counts, sample descriptions, three-mode-aware operator recommendation. Plus ledger emission: `runtime_settings.set_goodhart_hard_gate_*` now emits `governance_ratchet` continuity-ledger events on actual flips with `effective_mode` label, so the annual reflection drift summary picks up Goodhart-gate changes. |

13 tests in [`tests/test_q1_unclog.py`](tests/test_q1_unclog.py).
Pre-existing latent bug discovered + fixed during verification:
`capability_gap_analyzer` and `library_radar` had eager-start
patterns at module import but nothing in the boot chain ever
imported them — their daemons never ran in production. Anchored all
three (proposal_bridge, capability_gap_analyzer, library_radar) via
[`app/healing/__init__.py`](app/healing/__init__.py) (the established
mutable eager-wiring hub, already triggered by `main.py:96`).

### 38.5 Q2 — Three runbook auto-actions

After verifying the originally-proposed three didn't fit (cost_mode
is already in `app.config.Settings`; codebase has no alembic;
embed-misroute needed selector patch the original proposal didn't
account for), shipped a corrected trio that genuinely fits:

  1. **`disk_quota_immediate_retention`** —
     [`app/healing/monitors/disk_quota.py`](app/healing/monitors/disk_quota.py) patched to invoke
     `retention.run_chromadb()` + `run_worktrees()` +
     `run_attachments()` immediately when free space drops below
     WARN, instead of waiting days for those runners' own cadence.
     Per-target failure isolation, audit-event with outcomes,
     master switch `HEALING_DISK_AUTO_RETENTION_ENABLED` (default
     true).

  2. **`model_capability_runtime_block`** — extends the existing
     [`model_capability.py`](app/healing/handlers/model_capability.py) handlers (B + H) to write
     blocked models to two new runtime_settings lists
     (`chat_blocked_models`, `no_function_calling_models`) via new
     idempotent setters. New Step 5.5 in
     [`llm_selector.py:select_model`](app/llm_selector.py) consults
     `chat_blocked_models` AT DEFAULT-TIER (the existing breaker
     check was only in the pareto fallback path; embed-misroute
     hits the primary path). Fail-open on read failure. Addresses
     ~6,758/mo errors (6,606 embed-misroute + 152 no-function-calling).

  3. **`stuck_idle_diagnostic_dump`** —
     [`app/healing/monitors/idle_cooldown.py`](app/healing/monitors/idle_cooldown.py) patched to write
     `workspace/self_heal/stuck_idle_jobs.json` whenever any job is
     in deep cooldown. Snapshot includes name, remaining cooldown,
     failure count, diagnosis hint (`chronic` for >15 failures vs
     `long_cooldown`), operator-action recommendation. Forensics-
     only — never clears cooldowns (pinned by test
     `test_does_not_clear_any_cooldown`). Auto-clearing would
     defeat the cooldown's purpose (avoid storming a known-bad
     upstream).

18 tests in [`tests/test_q2_auto_actions.py`](tests/test_q2_auto_actions.py).

### 38.6 SubIA / continuity-ledger / KB integration map

| Subsystem | Wiring added |
|---|---|
| SubIA Global Workspace (`workspace_publish`) | proposal_bridge per-pass outcome (weighted by transition severity); auto_apply CR transitions; auto-revert events; Tier-3 amendment state changes; Goodhart mode flips |
| Identity continuity ledger | Tier-3 amendments already emitted on `mark_applied` (existing). Goodhart-gate mode flips now emit `governance_ratchet` events with `effective_mode` label (NEW). Auto-applied CRs touching `app/souls/*` would emit `soul_edit` (existing path; never fires today because forbidden-prefixes block souls under auto_apply) |
| Phase-5 consciousness gate | No new wiring — governance changes are downstream of the gate (audit-only) |
| Knowledge bases (4) | NONE touched — all three Q2 auto-actions are runtime config / filesystem only. Auto_apply forbidden-prefixes also categorically refuse `app/memory/` |

### 38.7 Test results

- **112 tests pass** (proposal_bridge 19 + capability_gap 15 +
  library_radar 14 + change_requests structural 24 + auto_apply 30 +
  Q1+Q2 33 + wiki validator 1)
- **24 skipped with reason** (gateway-only deps; runs in CI/docker)
- **0 regressions** introduced across all touched modules
- **Production verified**: 9 daemons run at boot
  (inquiry-scheduler, healing-monitors, healing-auditor-bridge,
  healing-watchdog, capability-gap-analyzer, library-radar,
  proposal-bridge.promoter, change-requests-auto-revert,
  governance-notifier — three of which were latent-dead before
  this work)

### 38.8 Files touched

```
NEW code:
crewai-team/app/proposal_bridge/__init__.py
crewai-team/app/proposal_bridge/store.py
crewai-team/app/proposal_bridge/promoter.py
crewai-team/app/change_requests/auto_revert.py
crewai-team/app/governance_notifier.py
crewai-team/app/observability/goodhart_advisory_report.py
crewai-team/app/tools/request_tier3_amendment.py

MODIFIED code:
crewai-team/app/change_requests/__init__.py     # exports + RiskClass
crewai-team/app/change_requests/lifecycle.py    # auto_approve + risk_class param
crewai-team/app/change_requests/models.py       # RiskClass enum + new fields
crewai-team/app/change_requests/validator.py    # wiki/ allowed + validate_auto_apply
crewai-team/app/episteme/paper_pipeline.py      # bridge stage()
crewai-team/app/healing/__init__.py             # eager-wire 4 daemons
crewai-team/app/healing/handlers/model_capability.py  # write to runtime blocklists
crewai-team/app/healing/monitors/disk_quota.py  # auto-retention on WARN/CRIT
crewai-team/app/healing/monitors/idle_cooldown.py     # forensic snapshot
crewai-team/app/library_radar/proposer.py       # bridge stage()
crewai-team/app/llm_selector.py                 # Step 5.5 chat_blocked_models
crewai-team/app/memory/wiki_index_reconciler.py # body-date norm + dedup
crewai-team/app/runtime_settings.py             # blocklists + Goodhart ledger
crewai-team/app/self_improvement/capability_gap_analyzer.py  # bridge stage()

NEW tests:
crewai-team/tests/proposal_bridge/__init__.py
crewai-team/tests/proposal_bridge/test_proposal_bridge.py
crewai-team/tests/test_change_requests_auto_apply.py
crewai-team/tests/test_q1_unclog.py
crewai-team/tests/test_q2_auto_actions.py

UPDATED tests:
crewai-team/tests/healing/test_paper_pipeline.py
crewai-team/tests/library_radar/test_proposer.py
crewai-team/tests/self_improvement/test_capability_gap_analyzer.py
crewai-team/tests/test_wiki_index_reconciler.py

Operational state:
crewai-team/workspace/runtime_settings.json     # tier3_amendment_enabled=true
crewai-team/workspace/self_heal/runbook_settings.json   # db_pool min_recur=1
crewai-team/workspace/change_requests/archive/2026-05-10_wiki_validation_failed/
                                                # 355 archived CRs + MANIFEST.md

Docs:
crewai-team/docs/AUTO_APPLY.md                  # NEW operator handover
crewai-team/docs/SELF_HEAL_V3.md                # this entry
CLAUDE.md                                       # new subsystem references
crewai-team/PROGRAM.md                          # this section
```

No TIER_IMMUTABLE files modified. The auto_apply infrastructure
ships dormant by deliberate design — the empty allowlists are the
correct safety state until a clean candidate consumer matures
(see [`docs/AUTO_APPLY.md`](docs/AUTO_APPLY.md) for the activation
procedure).


## 39. 2026-05-10 — Closure for the proposal generators (Q2)

Three threads, each closing a remaining "system noticed but
nothing happened" gap. Items 7, 8, 9 from the Q2 roadmap.

### 39.1 Item 9 — continuity_ledger lookup in proposal evaluation

When a CR is created or a Tier-3 amendment is proposed, the
operator now sees recent identity-shaping activity on the target
path inline with the proposal. New module
[`app/identity/relevant_history.py`](app/identity/relevant_history.py)
joins continuity-ledger events + CR-audit-log events filtered by
path within a 90-day window.

Wiring (read-only — neither source is augmented):

  * [`change_requests/lifecycle.py:create_request`](app/change_requests/lifecycle.py)
    — appends a "📜 Recent activity on this path" markdown block
    to ``cr.reason`` after the existing ``lessons_learned`` check.
  * [`tools/request_tier3_amendment.py`](app/tools/request_tier3_amendment.py)
    — passes the lookup as ``relevant_history_90d`` in
    ``extra_evidence`` so it travels in the proposal record + audit
    chain (no protocol change required).
  * [`governance_notifier.py`](app/governance_notifier.py) — Signal
    alerts for state transitions append a one-line history summary
    extracted from ``proposal.evidence``.

### 39.2 Item 7 delta — coding-session spec on non-Tier-3 proposals

The Q1 §38.1 proposal-bridge ship covered "unify three observational
paths under one CR-emitting helper." The remaining clause from the
spec — "+ a coding-session spec for non-Tier-3 changes" — landed
here. New optional ``coding_session_spec`` field on ``ProposalState``
(persisted across re-stages, schema: intent / files / acceptance /
expected_duration_min). Three per-source spec generators:

  * [`capability_gap_analyzer._build_coding_session_spec`](app/self_improvement/capability_gap_analyzer.py)
    — scaffolds ``app/<slug>/__init__.py`` + ``core.py`` + tests
  * [`library_radar.proposer._build_coding_session_spec`](app/library_radar/proposer.py)
    — adds a ``requirements.txt`` line + smoke-import test
  * [`paper_pipeline._build_coding_session_spec`](app/episteme/paper_pipeline.py)
    — generates a per-experiment driver script + JSONL output

The promoter renders the spec as a YAML fenced block in the CR
body **only for non-Tier-3 paths** (Tier-3 routes through
``governance_amendment.protocol``, never coding sessions). Verified
by ``test_promoter_suppresses_spec_for_tier_immutable``.

### 39.3 Item 8 — structured-diagnosis CRs (with self-tuning, telemetry, HOT-1 hook)

The May 2026 audit's "0 resolved, 1 attempted" finding traced to
``error_diagnosis`` producing prose proposals nobody applied. This
ship changes the LLM contract from prose to ``(path, new_content)``,
routes through the standard CR gate, and adds a self-adjusting
confidence threshold + persistent telemetry + HOT-1 observation
hook for the future consciousness probe.

**§39.3.1 Core structured diagnosis** —
[`app/healing/structured_diagnosis.py`](app/healing/structured_diagnosis.py).
LLM (Claude Sonnet 4.5) reads error + traceback + full file content
and returns strict JSON with ``new_content`` + ``confidence`` +
``reasoning`` OR ``declined: true`` with a ``decline_reason``.
Multi-site bugs, destructive patches (>5 lines removed), and
missing-context scenarios all decline → caller falls back to prose.
Per-pattern hourly rate limit (3/hr) bounds LLM spend.

**§39.3.2 Telemetry** —
[`app/healing/diagnosis_telemetry.py`](app/healing/diagnosis_telemetry.py).
Three event kinds: ``filed`` (CR created), ``declined`` (LLM/guard
declined), ``resolution`` (CR transitioned to applied/rejected/
rolled-back/timeout). Persisted to
``workspace/healing/structured_diagnosis_telemetry.jsonl`` with
``append_with_cap`` at 5000 lines. Resolution hook in
[`change_requests/lifecycle.py:_maybe_emit_diagnosis_telemetry`](app/change_requests/lifecycle.py)
fires from all four resolution paths automatically when
``cr.requestor == "error_diagnosis"``. Rollbacks count as
rejection-equivalent for the auto-tuner.

**§39.3.3 Auto-tune** —
[`app/healing/diagnosis_auto_tune.py`](app/healing/diagnosis_auto_tune.py).
Reads rolling-window approval rate (last 20 resolved CRs); adjusts
the active threshold within ``[floor, ceiling]`` from
runtime_settings. Algorithm: target band [0.65, 0.85]; step 0.02;
≥ 24h between adjustments; ≥ 5 new resolutions hysteresis. **Signal
alerting is option B**: silent on routine adjustments, fires only
when auto-tune wants to move beyond the operator-set band
(pinned-at-floor / pinned-at-ceiling), deduped 7d. Wired into
``healing/monitors`` daemon (24th monitor; hourly probe, internal
24h cadence gate).

**§39.3.4 Operator surface** — four new runtime_settings keys
(``threshold_floor``, ``threshold_ceiling``, ``threshold_override``,
``auto_tune_enabled``) with idempotent setters that validate
floor < ceiling and override ∈ [floor, ceiling]. Two new REST
endpoints under ``/config/structured_diagnosis/``: ``state`` (state
+ telemetry summary) and ``telemetry`` (paginated rolling-window
rows). New React ``StructuredDiagnosisCard`` in
[`SettingsPage.tsx`](dashboard-react/src/components/SettingsPage.tsx)
with active-threshold display, recent-approval-rate, telemetry
breakdown (filed / approved / rejected / pending), floor + ceiling
inputs, override pin / clear, auto-tune toggle.

**§39.3.5 HOT-1 metacognitive-repair observation hook** —
[`structured_diagnosis._emit_hot1_observation`](app/healing/structured_diagnosis.py)
emits one row to
``workspace/subia/observations/metacognitive_repair.jsonl`` for
EVERY structured-diagnosis attempt (filed OR declined). Schema
covers originating error + LLM's higher-order thought (causal
hypothesis + length + confidence + decline status) + proposed
intervention. The future HOT-1 probe (currently ABSENT from the
Butlin scorecard) will read this log to compute its score —
docs/CONSCIOUSNESS_HOT1_OBSERVATIONS.md
([`docs/CONSCIOUSNESS_HOT1_OBSERVATIONS.md`](docs/CONSCIOUSNESS_HOT1_OBSERVATIONS.md))
documents the schema + scoring sketch + scope discipline (probe
observes; doesn't steer). Goodhart-of-the-indicator is the failure
mode we're avoiding — the auto-tuner explicitly optimises for
operator approval rate, not hypothesis length or diversity.

### 39.4 SubIA / continuity-ledger / KB integration map

| Surface | Q2 wiring |
|---|---|
| SubIA Global Workspace (``workspace_publish``) | + auto-tune pinned-at-band events at salience 0.5 (option B alerts only) |
| Identity continuity ledger | unchanged — relevant_history is read-only against the existing six event kinds |
| Phase-5 consciousness gate | unchanged — observation track only |
| HOT-1 observation log (NEW) | ``workspace/subia/observations/metacognitive_repair.jsonl`` populated by every structured-diagnosis attempt |
| ``lessons_learned`` KB | reused — ``create_request`` already calls ``check_against`` so structured-diagnosis CRs that match a previously-rejected pattern get the "⚠️ Matches lesson" banner for free |
| ChromaDB / pgvector / Neo4j | none touched — Q2 is runtime + config + filesystem only |

### 39.5 Test results

- **140 tests pass** locally (Q2 closure 14 wiring + Q2 auto-actions
  4 + Q1 unclog 3 + auto_apply 30 + proposal_bridge 21 +
  capability_gap 15 + library_radar 14 + change_requests structural
  24 + wiki_validator 1 + 14 fresh proposal-bridge / spec / etc.)
- **52 skipped with reason** (gateway-only deps; runs in CI/docker)
- **0 regressions** introduced across all touched modules
- **Production verified**: all 9 daemons running at boot;
  ``healing.monitors`` driver reports **24 monitors** (was 23 —
  ``diagnosis_auto_tune`` is the new one); ``/config/structured_diagnosis/state``
  REST endpoint returns full state + telemetry shape

### 39.6 Files touched

```
NEW code:
crewai-team/app/identity/relevant_history.py       # Item 9
crewai-team/app/healing/structured_diagnosis.py    # Item 8.1
crewai-team/app/healing/diagnosis_telemetry.py     # Item 8.2
crewai-team/app/healing/diagnosis_auto_tune.py     # Item 8.3

MODIFIED code:
crewai-team/app/change_requests/lifecycle.py       # Item 9 history + Item 8.2 resolution hooks
crewai-team/app/tools/request_tier3_amendment.py   # Item 9 history → extra_evidence
crewai-team/app/governance_notifier.py             # Item 9 history tail in Signal alerts
crewai-team/app/proposal_bridge/store.py           # Item 7 coding_session_spec field
crewai-team/app/proposal_bridge/promoter.py        # Item 7 spec rendering
crewai-team/app/self_improvement/capability_gap_analyzer.py  # Item 7 spec generator
crewai-team/app/library_radar/proposer.py          # Item 7 spec generator
crewai-team/app/episteme/paper_pipeline.py         # Item 7 spec generator
crewai-team/app/healing/error_diagnosis.py         # Item 8 _try_structured_path
crewai-team/app/healing/monitors/__init__.py       # Item 8.3 register auto_tune monitor
crewai-team/app/runtime_settings.py                # Item 8.4 four new threshold keys + setters
crewai-team/app/api/config_api.py                  # Item 8.4 REST endpoints

NEW tests:
crewai-team/tests/test_q2_closure.py               # 42 tests (14 pass + 28 skip-deferred)

NEW docs:
crewai-team/docs/CONSCIOUSNESS_HOT1_OBSERVATIONS.md # Item 8.5

UPDATED React:
crewai-team/dashboard-react/src/components/SettingsPage.tsx  # StructuredDiagnosisCard
crewai-team/dashboard-react/src/api/queries.ts     # RuntimeSettings type extension

UPDATED docs:
crewai-team/docs/SELF_HEAL_V3.md                   # Q2 §39 amendments
CLAUDE.md                                          # subsystem references
crewai-team/PROGRAM.md                             # this section
```

No TIER_IMMUTABLE files modified. The structured-diagnosis path
is gated behind the existing operator approval flow — operator
approves the CR like any other; nothing auto-applies.

## 40. 2026-05-10/11 — Q3: Multi-year hygiene

Five additive items closing the long-tail "the system runs for years
and then this becomes a problem" gaps. Everything observational,
revertible, default-OFF for the consequential bits (Item 12 dual-write
+ shadow read; Item 13 backup), default-ON for the cheap quarterly
hygiene (Item 10 SQLite VACUUM, Item 11 archive rotation).

### 40.1 Item 10 — ChromaDB hygiene (SQLite VACUUM + operator rebuild)

**Reframed from the original "ChromaDB compaction" item.** ChromaDB
has no public `compact()` API — HNSW segments are managed by
chromadb-internal code. What we CAN do without taking the store
offline is plain SQLite `VACUUM` on the metadata file (`chroma.sqlite3`)
to recover space from soft-deleted rows that never got their pages
freed.

  * **`app/healing/monitors/chromadb_hygiene.py`** — 25th healing
    monitor. Scans `workspace/*/chroma.sqlite3` (skipping `corrupt_*`
    / `bak_*` recovery snapshots), runs `PRAGMA optimize` + `VACUUM`
    on each. Internal cadence 90 days; daemon driver pings daily.
    Sends one Signal alert if total reclaim ≥50 MB.
  * **`app/memory/chromadb_rebuild.py`** — operator-initiated full
    rebuild for HNSW reclaim. Streams rows out, snapshots to
    `workspace/<kb>/.rebuild_backups/<collection>__<ts>.jsonl.gz`,
    drops + recreates the collection, replays the snapshot. Has a
    `--dry-run` flag (snapshot-only, no mutation) and a
    `--from-snapshot <path>` recovery flow for mid-rebuild failures.

### 40.2 Item 11 — JSONL caps with archive rotation

**Two retention patterns now distinguished**: truncate-on-overflow
for operational telemetry (existing `cap_jsonl` / `append_with_cap`),
and **rotate-to-monthly-archive** for consciousness-relevant logs.

  * **`app/utils/jsonl_retention.py`** extended with:
    - `append_with_archive_rotate(path, line, *, max_lines, keep_fraction=0.5)`
      — appends + rotates oldest 50% to
      `<path.parent>/archive/<YYYY-MM>_<basename>` when over cap.
    - `read_archive(path)` — lazy iterator across all monthly archive
      files + live file in chronological order. Decade-scale safe.
    - `archive_stats(path)` — summary for the dashboard.
  * **Wired into 5 writers**:
    - `app/affect/core.py:_append_trace` — `trace.jsonl`, cap 100k
    - `app/affect/salience.py:_append` — `salience.jsonl`, cap 50k
    - `app/affect/welfare.py:audit` — `welfare_audit.jsonl`, cap 5k
    - `app/affect/care_policies.py:record_spend` — `care_ledger.jsonl`, cap 10k
    - `app/training/adapter_lifecycle.py:_append_history_snapshot`
      — operational; uses simple `append_with_cap` at 1000 snapshots.

Welfare/affect/salience/care archives preserve consciousness data
forever for HOT-1 / decentered-reflection / backward-counterfactual
replay probes.

### 40.3 Item 12 — Embedding-model migration framework

The first production consumer of the previously-dormant Tier-3
amendment path. Migrating `_EMBED_DIM` is a TIER_IMMUTABLE edit; the
framework makes the surrounding setup deliberate, reversible, and
gated.

  * **`app/memory/embedding_migration/`** — new package, 8 modules:
    - `plan.py` — typed `MigrationPlan` + `EmbeddingModel` + `MigrationTarget`
    - `state.py` — 10-state machine over `runtime_settings.embedding_migration_state`
    - `dual_write.py` — best-effort shadow write via target embedder; backfill driver
    - `shadow_read.py` — sampled NDCG@10 divergence telemetry, rolling window
    - `verify.py` — pre-cutover invariants (counts, dim, NDCG threshold, DR freshness)
    - `cutover.py` — Tier-3-amendment-gated atomic swap (delete source, create from shadow, archive)
    - `dry_run.py` — full pipeline against sandbox collection; CLI entrypoint
    - `__init__.py` — package boundary doc
  * **`app/memory/chromadb_manager.py`** hooks: `store()` now generates
    one shared `doc_id` and calls `maybe_dual_write()`; `retrieve()`
    calls `maybe_shadow_read()` with the source's top-N ids.
  * **`app/runtime_settings.py`** — 4 new keys with paired getters/setters:
    - `embedding_migration_dual_write_enabled` (bool, default OFF)
    - `embedding_migration_shadow_read_enabled` (bool, default OFF)
    - `embedding_migration_cutover_enabled` (bool, default OFF)
    - `embedding_migration_state` (dict, state-machine blob)
  * **`app/api/config_api.py`** — 3 toggle endpoints wired into
    `POST /api/cp/settings`.
  * **`app/control_plane/dashboard_api.py`** — `GET /api/cp/embedding-migration`
    one-shot status endpoint (plan + state + verify + window + switches).

### 40.4 Item 13 — DR drill (portable export + boot drill)

Container-independent backup path that complements the existing
`app/healing/db_backup.py` binary dumps. Answers a different question:
**"Could we rebuild from a tarball on a fresh laptop?"** — not the
fast same-cluster restore.

  * **`HEALING_DB_BACKUP_ENABLED=true`** flipped in `.env`. The
    existing weekly backup runner now actually fires on the dev
    workstation.
  * **`app/dr/`** — new package, 3 modules + scripts wrapper + ops doc:
    - `export_kbs.py` — produces self-contained `.tar.gz` with every
      ChromaDB collection (as JSONL), key Postgres tables (as JSONL),
      canonical workspace ledgers. **Deliberate secret denylist**:
      `.env`, `secrets/`, `google_token.json`, `vapid_*.pem`, anything
      matching `token`/`credential`/`private_key`. Records every
      excluded path in the manifest.
    - `import_kbs.py` — reads tarball into ephemeral target dir.
      Deliberately no "overwrite live workspace" mode — that's a
      destructive op operators do by hand with full awareness.
    - `boot_drill.py` — end-to-end exercise. Take latest export →
      import to ephemeral dir → run sanity queries (count match,
      peek dim, smoke retrieve) → write report → Signal alert.
  * **`scripts/dr_boot_drill.sh`** — shell wrapper.
  * **`docs/DR_DRILL.md`** — operator runbook with auditing
    instructions (manifest inspection, restore-to-sandbox flow).

### 40.5 Item 14 — Cost trend dashboard

Multi-year cost trajectory visualisation. Read-only on `audit_log`.

  * **`app/control_plane/cost_trends.py`** — pure-stdlib OLS linear
    regression on monthly totals. 95% CI band from regression
    residuals. Rolling-window z-score anomaly detection on daily
    totals. Goodhart-resistant: forecast is observational, system
    never optimises against it.
  * **`GET /api/cp/costs/trends`** — one-shot bundle:
    `{summary, monthly, forecast, anomalies, params, as_of}`.
  * **`dashboard-react/src/components/CostTrendsCard.tsx`** — new
    component mounted in `CostCharts.tsx`. History blue line +
    forecast violet dashed line + 95% CI translucent band + KPI
    cards (total, trend %/month, projected next-12-mo) + anomalies
    table.

### 40.6 Test sweep

**`tests/test_q3_multiyear_hygiene.py`** — 22 tests covering:
  * Archive rotation: lines preserved, monthly filenames, stats summary.
  * OLS math: perfect fit zero-sigma, noisy fit positive sigma, month
    rollover, perfect-fit CI collapse, anomaly z-threshold, summary
    growth rate, CI clamping at zero, zero-variance anomaly guard.
  * ChromaDB hygiene: corrupt-dir skip, real SQLite VACUUM.
  * DR export: secret-path denylist.
  * Migration plan: roundtrip serialization.
  * State machine: invalid transition raises, valid progression,
    counter increments.
  * NDCG@10: perfect match, zero overlap, partial overlap < 1.

All 22 pass. Q1+Q2+Q3 regression: 43 pass, 52 skip (gateway-deps).
Affect-module tests: 13 pass.

### 40.7 SubIA / KB integration map

  * **Affect substrate** (Item 11) — `trace.jsonl`, `salience.jsonl`,
    `welfare_audit.jsonl`, `care_ledger.jsonl` all now archive
    forever to monthly buckets. HOT-1 / decentered-reflection /
    backward-counterfactual replay get decade-scale data.
  * **ChromaDB layer** (Item 10) — every KB's SQLite metadata stays
    bounded without losing rows.
  * **Tier-3 amendment** (Item 12) — first real consumer of the
    protocol shipped in §25.1. The post-apply hook moves state to
    APPLIED and the continuity ledger gets a `tier3_amendment` entry
    via existing `governance_amendment.mark_applied` instrumentation.
  * **DR layer** (Item 13) — portable tarball excludes secrets via
    aggressive denylist; preserves affect archives + identity ledger
    + audit_journal. Manifest is operator-auditable.
  * **Goodhart discipline** (Item 14) — forecast is observational.
    System never trains against the cost trajectory. The metric
    serves the operator's decisions, not the system's.

### 40.8 Files touched

```
NEW backend:
crewai-team/app/healing/monitors/chromadb_hygiene.py
crewai-team/app/memory/chromadb_rebuild.py
crewai-team/app/control_plane/cost_trends.py
crewai-team/app/dr/__init__.py
crewai-team/app/dr/export_kbs.py
crewai-team/app/dr/import_kbs.py
crewai-team/app/dr/boot_drill.py
crewai-team/app/memory/embedding_migration/__init__.py
crewai-team/app/memory/embedding_migration/plan.py
crewai-team/app/memory/embedding_migration/state.py
crewai-team/app/memory/embedding_migration/dual_write.py
crewai-team/app/memory/embedding_migration/shadow_read.py
crewai-team/app/memory/embedding_migration/verify.py
crewai-team/app/memory/embedding_migration/cutover.py
crewai-team/app/memory/embedding_migration/dry_run.py

NEW React:
crewai-team/dashboard-react/src/components/CostTrendsCard.tsx

NEW scripts:
crewai-team/scripts/dr_boot_drill.sh

NEW tests:
crewai-team/tests/test_q3_multiyear_hygiene.py     # 22 tests, all pass

NEW docs:
crewai-team/docs/DR_DRILL.md
crewai-team/docs/EMBEDDING_MIGRATION.md

UPDATED backend:
crewai-team/app/utils/jsonl_retention.py           # +3 functions (archive rotation)
crewai-team/app/affect/core.py                     # wire archive rotation
crewai-team/app/affect/salience.py                 # wire archive rotation
crewai-team/app/affect/welfare.py                  # wire archive rotation
crewai-team/app/affect/care_policies.py            # wire archive rotation
crewai-team/app/training/adapter_lifecycle.py      # wire cap (simple)
crewai-team/app/healing/monitors/__init__.py       # register 25th monitor
crewai-team/app/memory/chromadb_manager.py         # dual-write + shadow-read hooks
crewai-team/app/runtime_settings.py                # 4 new keys + getters/setters
crewai-team/app/api/config_api.py                  # 3 new toggle endpoints
crewai-team/app/control_plane/dashboard_api.py     # 2 new GET endpoints

UPDATED React:
crewai-team/dashboard-react/src/api/endpoints.ts   # costsTrends route
crewai-team/dashboard-react/src/api/queries.ts     # useCostTrendsQuery + CostTrendBundle type
crewai-team/dashboard-react/src/components/CostCharts.tsx  # mount CostTrendsCard

UPDATED config:
crewai-team/.env                                   # HEALING_DB_BACKUP_ENABLED=true
```

No TIER_IMMUTABLE files modified by this section. The embedding-
migration cutover (Item 12) is the first consumer that **proposes**
a Tier-3 amendment, but the proposal itself is gated by the existing
protocol — the cutover module does not bypass it.
