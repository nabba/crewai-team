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
