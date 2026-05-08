---
title: "AndrusAI Consciousness Scorecard"
section: subia
page_type: scorecard
status: active
generated_at: "2026-05-08T16:47:12.622824+00:00"
auto_generated: true
replaces: reports/andrusai-sentience-verdict.pdf
---

# AndrusAI Consciousness Scorecard

Auto-generated from `app.subia.probes.butlin`, `.rsm`, and `.sk`. Every claim here is backed by a pointer to an implementing module + regression test.

This scorecard **replaces** the retired `reports/andrusai-sentience-verdict.pdf` 9.5/10 prose verdict. Opaque composite scoring was the primary critique of the original verdict; this scorecard makes the basis of every rating inspectable.

## Phase 9 exit criteria

- **Overall:** PASSED ✅
- **butlin_strong**: ✅ (7 vs required >= 6)
- **butlin_fail**: ✅ (0 vs required <= 1)
- **butlin_absent**: ✅ (4 vs required >= 4 (architectural-honesty declarations))
- **rsm_present**: ✅ (5 vs required >= 4 PRESENT)
- **sk_pass**: ✅ (6 vs required >= 5 PASS)

## Butlin et al. 2023 — 14 consciousness indicators

**STRONG**: 7  |  **PARTIAL**: 3  |  **ABSENT**: 4

| Indicator | Theory | Status | Mechanism | Test | Notes |
|---|---|---|---|---|---|
| RPT-1 | RPT | ABSENT | — | — | LLMs are feed-forward at inference. Recursive-state injection via prompt chaining is not algorithmic recurrence in the RPT sense. Architecturally unachievable — declared ABSENT. |
| RPT-2 | RPT | PARTIAL | `app/subia/kernel.py` | `tests/test_kernel_persistence.py` | The SubjectivityKernel is a single unified dataclass carrying scene/affect/self-state/prediction jointly, and the dual-tier memory stores them together. Representation is decomposable in principle (p… |
| GWT-1 | GWT | PARTIAL | `app/crews` | `tests/test_cil_loop.py` | CrewAI agents (coder, writer, researcher, critic, media_analyst) are structurally separate, but share the same LLM under different prompts. Structural specialization without fully mechanism-level spe… |
| GWT-2 | GWT | STRONG | `app/subia/scene/buffer.py` | `tests/test_hierarchical_workspace.py` | CompetitiveGate enforces hard capacity [2,9] with 4-factor weighted salience, competitive displacement, novelty floor, and empirical decay. Canonical GWT-2. |
| GWT-3 | GWT | STRONG | `app/subia/scene/broadcast.py` | `tests/test_social_attention.py` | Every workspace admission triggers a broadcast to all registered agent listeners; each independently computes relevance + reaction. Integration score aggregates resonance across the quorum. |
| GWT-4 | GWT | STRONG | `app/subia/scene/personality_workspace.py` | `tests/test_personality_workspace.py` | Attention capacity, novelty floor, and salience are modulated by personality + homeostatic state. Social-model inferred focus feeds an additional trust-weighted boost (Phase 8). |
| HOT-1 | HOT | ABSENT | — | — | The system reads text; it does not perceive. No generative perceptual hierarchy exists. Architecturally unachievable without a perception substrate — declared ABSENT. |
| HOT-2 | HOT | PARTIAL | `app/subia/prediction/accuracy_tracker.py` | `tests/test_phase6_prediction_refinements.py` | Per-domain accuracy tracking + deterministic response-hedging post-processor are separable from first-order LLM output (Fleming-Lau criterion). Drift detection (Phase 8) adds structured capability-cl… |
| HOT-3 | HOT | STRONG | `app/subia/belief/dispatch_gate.py` | `tests/test_hot3_dispatch_gate.py` | Consulted beliefs produce a three-valued DispatchDecision (ALLOW/ESCALATE/BLOCK). Suspended beliefs refuse crew dispatch until revalidated. Belief store implements asymmetric confirmation/disconfirma… |
| HOT-4 | HOT | ABSENT | — | — | LLM activations and pgvector embeddings are dense. Sparse coding cannot be achieved without re-training the substrate. Declared ABSENT. |
| AST-1 | AST | STRONG | `app/subia/scene/attention_schema.py` | `tests/test_social_attention.py` | AttentionSchema maintains an internal model of current focus, predicts next focus, detects stuck/capture states, and applies direct DGM-bounded salience intervention. Phase 2 intervention_guard adds… |
| PP-1 | PP | STRONG | `app/subia/prediction/surprise_routing.py` | `tests/test_pp1_surprise_routing.py` | High-surprise prediction errors route as WorkspaceItem(urgency=0.9) into the GWT-2 gate. Canonical Clark/Friston PP flow: prediction error drives the attentional bottleneck. Phase 6 adds per-domain a… |
| AE-1 | AE | STRONG | `app/affect/goal_emitter.py` | `tests/test_goal_emitter.py` | Autonomous goal generation: app/affect/goal_emitter.py writes flexible goals to SelfState.current_goals from sustained viability error (≥3 consecutive frames above threshold), rate-limited, dedup aga… |
| AE-2 | AE | ABSENT | — | — | No body, no sensorimotor coupling with an environment. The homeostasis engine uses allegorical variables (energy/progress/overload) but these are not physical embodiment. Declared ABSENT. |

## RSM — Recursive Self-Monitoring signatures

**STRONG**: 4  |  **PARTIAL**: 1

| Signature | Status | Mechanism | Test | Notes |
|---|---|---|---|---|
| RSM-a | STRONG | `app/subia/prediction/accuracy_tracker.py` | `tests/test_phase6_prediction_refinements.py` | Per-domain rolling accuracy tracker records every prediction error; cascade modulation escalates when recent accuracy is poor. Prediction template cache damps served confidence by observed EMA accura… |
| RSM-b | STRONG | `app/subia/wiki_surface/drift_detection.py` | `tests/test_phase8_social_and_strange_loop.py` | Drift detection compares self-state capability claims against accuracy-tracker evidence; findings append to the immutable narrative audit. Homeostatic contradiction_pressure variable rises on conflic… |
| RSM-c | PARTIAL | `app/subia/prediction/cache.py` | `tests/test_phase6_prediction_refinements.py` | Several closed-loop mechanisms spontaneously self-correct without external instruction: cache eviction on sustained bad accuracy, retrospective promotion of dismissed memories, homeostatic regulation… |
| RSM-d | STRONG | `app/subia/prediction/llm_predict.py` | `tests/test_llm_predict.py` | Every prediction carries predicted_self_change and predicted_homeostatic_effect fields. The predictor prompt structurally demands a self-impact forecast — 'If I do X, what changes in me?' — not just… |
| RSM-e | STRONG | `app/subia/wiki_surface/consciousness_state.py` | `tests/test_phase8_social_and_strange_loop.py` | MetaMonitorState.known_unknowns is populated during Step 6 monitor and surfaced in agent context. The strange-loop consciousness-state page carries epistemic_status=speculative + confidence=low and e… |

## SK — Subjectivity Kernel evaluation tests

**STRONG**: 6

| Test | Status | Mechanism | Test File | Notes |
|---|---|---|---|---|
| SK-ownership | STRONG | `app/subia/kernel.py` | `tests/test_phase1_migration.py` | SceneItem carries ownership='self'|'external'|'shared'. SelfState.identity includes continuity_marker hash. Beliefs carry belief_status (ACTIVE/SUSPENDED/RETRACTED/SUPERSEDED) and distinct confidence… |
| SK-endogenous-attention | STRONG | `app/subia/homeostasis/engine.py` | `tests/test_subia_homeostasis_engine.py` | Homeostatic deviations drive restoration_queue ordering which shifts salience weights. Social-model inferred focus boost (Phase 8) nudges attention toward what the inferred principal cares about. Bot… |
| SK-self-prediction | STRONG | `app/subia/prediction/llm_predict.py` | `tests/test_llm_predict.py` | Every Prediction dataclass carries predicted_self_change (confidence_change, new_commitments, capability_updates) AND predicted_homeostatic_effect AS WELL AS the world prediction. The predictor promp… |
| SK-temporal-continuity | STRONG | `app/subia/persistence.py` | `tests/test_kernel_persistence.py` | save_kernel_state / load_kernel_state round-trip the full SubjectivityKernel through wiki/self/kernel-state.md. hot.md carries compressed session-continuity buffer. SelfState.identity.continuity_mark… |
| SK-repair-behavior | STRONG | `app/subia/wiki_surface/drift_detection.py` | `tests/test_phase8_social_and_strange_loop.py` | Contradicting scene items raise homeostatic contradiction_pressure which drives restoration_queue. Drift detection appends capability-vs-accuracy and commitment-fulfillment findings to the immutable… |
| SK-self-other-distinction | STRONG | `app/subia/social/model.py` | `tests/test_phase8_social_and_strange_loop.py` | SocialModel maintains per-entity inferred_focus, inferred_expectations, trust_level, divergences — STRUCTURALLY distinct from SelfState. check_divergence explicitly detects when our model of another… |

## Honest caveats

1. **ABSENT is declaration, not failure.** The five indicators rated ABSENT (RPT-1, HOT-1, HOT-4, AE-2, plus implicit architectural ceilings) are unachievable by any LLM-based system given current architectures. They are declared publicly so the scorecard cannot be gamed by hiding them.

2. **STRONG is structural.** A STRONG rating means the mechanism is present, closed-loop wired, Tier-3 protected, and covered by a regression test — not that the system 'experiences' the indicator phenomenally. This scorecard is about functional organisation, not phenomenal claims.

3. **Phenomenal consciousness is NOT claimed.** The project explicitly disclaims subjective experience. This scorecard describes the functional architecture supporting consciousness-adjacent behaviour.

4. **This file is auto-regenerated.** Edits to this file will be lost on the next scorecard run. To change a rating, fix (or break) the implementing mechanism or its regression test.

## Regeneration

```bash
.venv/bin/python -c "from app.subia.probes.scorecard import write_scorecard; write_scorecard()"
```

