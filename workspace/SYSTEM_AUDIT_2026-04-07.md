# AndrusAI System Audit Report
**Date:** 2026-04-07
**Auditor:** Claude Opus 4.6 (automated deep analysis)
**Scope:** Full system — 167 Python files, 43K+ LoC, 10 PostgreSQL schemas, 3 databases

---

## Executive Summary

The system has strong operational foundations — Signal integration, crew dispatch, LLM cascade, and knowledge management all work. However, **the self-improvement subsystems are largely non-functional**: 4 of 5 evolution mechanisms never produce results, the training pipeline is orphaned, the personality system has zero assessments, and self-healing never deploys fixes. 18 Python files are imported by nothing, and 3 tool modules reference packages not in requirements.txt.

**Overall health: 65/100** — Solid request path, broken adaptation path.

---

## 1. ORPHANED CODE (18 files)

Files that exist but are imported by NOTHING in the codebase:

| File | Size | Issue |
|------|------|-------|
| `app/agents/critic.py` | 59 lines | Critic agent defined but no crew instantiates it |
| `app/agents/self_improver.py` | 37 lines | Dead — SelfImprovementCrew uses inline agent |
| `app/atlas/audit_log.py` | ~60 lines | ATLAS audit — written but never wired |
| `app/atlas/learning_planner.py` | ~80 lines | ATLAS planner — written but never wired |
| `app/contracts/events.py` | 100 lines | Event schemas defined, never consumed by any handler |
| `app/contracts/firestore_schema.py` | ~50 lines | Schema docs, never consumed |
| `app/control_plane/heartbeats.py` | 150 lines | New — not yet integrated into idle_scheduler |
| `app/evolution_db/eval_sets.py` | ~60 lines | Evaluation sets — written but never imported |
| `app/firebase_infra.py` | ~80 lines | Superseded by app/firebase/infra.py |
| `app/ollama_fleet.py` | ~120 lines | Superseded by ollama_native.py (Metal GPU) |
| `app/personality/probes.py` | ~80 lines | PDS probes — assessment.py uses its own |
| `app/proactive/proactive_behaviors.py` | ~100 lines | Behaviors defined, trigger_scanner never calls them |
| `app/self_awareness/grounding.py` | ~60 lines | Reality grounding — written but never called |
| `app/tool_executor.py` | ~150 lines | Generic executor — superseded by direct tool calls |
| `app/tools/bridge_tools.py` | ~80 lines | Bridge tools — bridge_enabled=False, 0 registrations |
| `app/tools/browser_tool.py` | 240 lines | Playwright not in requirements.txt, never imported |
| `app/tools/composio_tool.py` | 127 lines | composio not in requirements.txt, never imported |
| `app/training_pipeline.py` | 494 lines | **CRITICAL** — entire MLX training pipeline, never imported |

**Recommendation:** Delete truly dead files (ollama_fleet, firebase_infra, tool_executor). Wire or delete the rest. The training_pipeline needs to be imported by idle_scheduler to function.

---

## 2. NOT-WIRED COMPONENTS

### 2a. Missing from lifespan initialization

| Component | Status |
|-----------|--------|
| `adaptive_ensemble.py` | Never initialized — phase-dependent LLM selection is dead code |
| `cascade_evaluator.py` | Only invoked by island/parallel evolution (which never run) |
| `meta_learning.py` | UCB1 strategy selection — never initialized |
| `implicit_feedback.py` | Implicit signal detection — never initialized |
| `control_plane/heartbeats.py` | New module, not yet integrated |

### 2b. Tools not registered with any agent

| Tool | Exists | Registered? |
|------|--------|-------------|
| `browser_tool.py` | ✅ | ❌ Zero agent registrations |
| `composio_tool.py` | ✅ | ❌ Zero agent registrations |
| `document_generator.py` | ✅ | ❌ Only in commands.py, not in agent tools list |
| `ocr_tool.py` | ✅ | ❌ Only used by attachment_reader, not directly by agents |

### 2c. Missing lifecycle hooks

| Hook Point | Registered | Missing |
|------------|-----------|---------|
| `PRE_LLM_CALL` | humanist_safety, budget_enforcement | — |
| `POST_LLM_CALL` | self_correct, training_data | — |
| `PRE_TOOL_USE` | block_dangerous | — |
| `POST_TOOL_USE` | memorize_tools | — |
| `ON_COMPLETE` | health_metrics | — |
| `PRE_TASK` | **NONE** | Task interception not used |
| `ON_DELEGATION` | **NONE** | Delegation events not tracked |
| `ON_ERROR` | **NONE** | Errors bypass hooks entirely |

History compression hook is defined (`create_history_compression_hook`) but never registered in `_register_defaults()`.

---

## 3. NON-OPERATIONAL COMPONENTS

### 3a. Permanently non-functional (missing dependencies)

| Module | Missing Package | In requirements.txt? |
|--------|----------------|---------------------|
| `tools/composio_tool.py` | composio-core | ❌ |
| `tools/browser_tool.py` | playwright | ❌ |
| `training_pipeline.py` | mlx, mlx-lm | ❌ (host-only, correct) |

All use graceful try/except — they don't crash the system but are permanently non-functional.

### 3b. Training pipeline end-to-end

```
Signal messages → training_collector.py → raw/ JSONL (9 interactions)
                                        → quality scoring: 0 scored
                                        → curated/: EMPTY
                                        → training_pipeline.py: ORPHANED (never imported)
                                        → MLX training: NEVER RUNS
                                        → Adapter deployment: NEVER HAPPENS
```

**The entire self-training capability is a dead end.** Data collection works minimally but curation never quality-scores, and the training pipeline module is imported by nothing.

---

## 4. DATA FLOW ANALYSIS

### 4a. Request path: Signal → Response ✅ OPERATIONAL

- 153 tasks completed, 91.5% success rate
- Average response time: **171.4 seconds** (heavy tasks)
- Max: 3,228 seconds (53.8 minutes — likely a multi-page research task)
- `tasks.crew` column always empty — crew name never recorded

### 4b. Feedback → Modification ⚠️ WIRED BUT DATA-STARVED

- Pipeline exists and is wired into idle_scheduler
- Cannot verify actual pattern counts (PostgreSQL data, Docker offline)
- Modification engine job runs but likely never triggers (insufficient feedback patterns)

### 4c. Evolution → Deploy ⚠️ RUNS BUT NEVER SUCCEEDS

- 722 experiments run
- 46 variants archived — **ALL 46 DISCARDED** (0 kept)
- Fitness improvements exist in results.tsv (0.76–0.84) but never cross the promotion threshold
- Code mutations consistently fail — only skill file additions work

### 4d. Error → Self-heal → Deploy ⚠️ DIAGNOSES BUT NEVER HEALS

- 32 errors in journal
- Self-heal diagnoses errors (BadRequestError, ImportError most common)
- **0 actual code remediations deployed**
- Auto-deployer has no deploy_log.json — suggesting it has never successfully deployed

### 4e. Knowledge base → RAG retrieval ✅ OPERATIONAL

- ChromaDB: 20MB data, actively used
- Philosophy KB: 27 texts, 3,026 chunks
- Knowledge base: 24MB of ingested documents
- RAG injection confirmed working in crew context building

---

## 5. SELF-LEARNING EFFICIENCY

| Metric | Value | Assessment |
|--------|-------|------------|
| Skill files created | **222** | High volume |
| Topic diversity | 33/222 are "ecological" variants (15%) | **Topic rut detected** |
| Last skill created | 2026-04-06 17:51 | Recent |
| Training interactions collected | 9 | Far below 100 minimum |
| Quality scored | 0 | **Curation never runs** |
| Eligible for training | 0 | Pipeline broken |
| PDS assessments completed | **0 for all 4 roles** | **PDS non-functional** |
| Cogito reflections | 68 files | Active |
| ATLAS competence tracked | Yes (file exists) | Partially working |

**Key issue:** The learning loop creates skills prolifically but gets stuck in topic ruts. The Personality Development System has never completed a single assessment despite being wired. Quality scoring of training data never happens.

---

## 6. SELF-EVOLVING EFFICIENCY

| Mechanism | Idle Job? | Results | Status |
|-----------|-----------|---------|--------|
| `evolution.py` (basic loop) | ✅ | 722 experiments, 0 kept | **Runs but never promotes** |
| `island_evolution.py` | ✅ | Generation 0, no results | ❌ Never produces output |
| `parallel_evolution.py` | ✅ | Archive = 1 baseline entry | ❌ Never produces output |
| `map_elites.py` | ✅ | Generation 0, 0 grid cells | ❌ Never populates grid |
| `adaptive_ensemble.py` | ❌ | Never initialized | ❌ Dead code |

**Root cause:** The basic evolution loop's fitness threshold is too strict — variants score 0.76-0.84 but apparently never exceed the promotion gate. Island/parallel/MAP-Elites all depend on cascade_evaluator which itself depends on reference_tasks.py test suite — this chain likely fails silently.

---

## 7. SELF-HEALING EFFICIENCY

| Metric | Value |
|--------|-------|
| Errors detected | 32 |
| Most common: BadRequestError (codestral tools) | 12 (now fixed) |
| Most common: ImportError (commander split) | 7 (now fixed) |
| Auto-remediations deployed | **0** |
| Code audits run | 81 |
| Code audit findings | 0 (always "0 issues") |
| Homeostatic state | energy=0.97, confidence=0.95 |

**The self-healing system is a diagnostic tool, not a healing tool.** It logs errors and diagnoses root causes but never successfully generates and deploys code fixes.

---

## 8. DATABASE STRUCTURE

### SQLite (conversations.db)

| Table | Rows | Issue |
|-------|------|-------|
| messages | 287 | ✅ Normal |
| tasks | 153 | ⚠️ `crew` column always empty |

### SQLite (llm_benchmarks.db)

| Table | Rows | Issue |
|-------|------|-------|
| benchmarks | **0** | ❌ **Never written to** — cron runs but write path broken |
| token_usage | 23,919 | ✅ Active |
| request_costs | 45 | ✅ Active |

### PostgreSQL schemas (10 total, cannot verify row counts — Docker offline)

| Schema | Tables | Expected State |
|--------|--------|---------------|
| mem0 | memories, entities | Active (Mem0 persistence) |
| evolution | experiments, variants | 722+ rows |
| feedback | events, patterns, reactions | Unknown |
| modification | attempts, versions | Unknown |
| meta_learning | strategy_scores | Unknown (likely empty) |
| atlas | skills, competence, apis | Unknown |
| map_elites | archive, grid | Likely empty (gen 0) |
| training | runs, tiers, evaluations | Likely empty (pipeline broken) |
| personality | traits, assessments | Empty (0 assessments) |
| governance | promotions | 4+ rows |
| control_plane | 8 tables | New — tickets, budgets, audit |

### Potential orphaned data

- `workspace/applied_code/tools/` — contains deployed code but auto_deployer has no log
- `workspace/evolution_archive/` — multiple files but archive shows only 1 entry
- `workspace/island_evolution/` — state files at gen 0 for 4 roles, never advancing

---

## 9. BOTTLENECKS

### Response time distribution
- Simple commands (ping, status): **~1 second** (after latency optimization)
- Research tasks: **30-180 seconds**
- Complex multi-crew tasks: **3-10 minutes**
- Worst case: **53 minutes** (outlier)

### LLM provider concentration
- DeepSeek: **95.5%** of all calls (22,948 of 23,919)
- Local Ollama: **0.9%** (157 calls)
- Claude: **0.7%** (168 calls)
- MiniMax: **0.9%** (213 calls)
- Gemini: **0.4%** (95 calls)

**The system is dangerously dependent on DeepSeek.** If DeepSeek goes down or raises prices, the entire system stops. Local Ollama is barely used despite having multiple models.

### Cost efficiency
- Total spend: **$53.06** across ~24K LLM calls
- Average cost per call: $0.0022
- DeepSeek V3 dominates cost at $20.55 (38.8% of spend)
- Gemini 3.1 Pro: $2.03 for only 95 calls (expensive per-call)

### Idle scheduler throughput
- 23 jobs registered
- Evolution (even at 2 iterations) takes 5-10 minutes per run
- Jobs 10-23 likely rarely execute in a single idle window
- The `should_yield()` mechanism means user messages interrupt background work

---

## 10. RECOMMENDED FIXES (Priority Order)

### P0 — Critical (broken subsystems)

1. **Wire training_pipeline.py into idle_scheduler** — add `training-pipeline` job that calls `run_training_cycle()`. Currently the module is orphaned.

2. **Fix benchmarks write path** — `llm_benchmarks.db` benchmarks table has 0 rows. The `get_benchmark_summary()` cron job runs but never writes. Trace the write function.

3. **Fix tasks.crew column** — `start_task()` should record which crew handles each task. Currently always empty.

4. **Wire the training_collector quality scoring** — 9 interactions collected but 0 quality-scored. The curation pipeline's LLM judge call appears to never execute.

### P1 — High (non-functional subsystems)

5. **Investigate why all 46 evolution variants are discarded** — the promotion threshold may be miscalibrated, or the comparison logic may have a bug.

6. **Fix island_evolution / parallel_evolution / MAP-Elites** — all stuck at generation 0. The cascade_evaluator dependency chain likely fails silently.

7. **Fix PDS (Personality Development System)** — 0 assessments completed despite idle_scheduler job. The assessment battery may fail on the first LLM call.

8. **Register history compression hook** — defined but never added to `_register_defaults()`.

### P2 — Medium (orphaned/dead code cleanup)

9. **Delete truly dead files:** `ollama_fleet.py`, `firebase_infra.py`, `tool_executor.py`

10. **Wire or delete:** `proactive_behaviors.py`, `grounding.py`, `eval_sets.py`, `learning_planner.py`

11. **Add to requirements.txt or remove:** `composio_tool.py` (composio), `browser_tool.py` (playwright)

12. **Fix topic diversity** — the learning loop needs a deduplication/diversity check before adding skill topics to prevent ecological topic ruts.

### P3 — Low (optimizations)

13. **Increase local Ollama usage** — with gemma4:26b arriving, configure the selector to prefer local models more aggressively in hybrid mode.

14. **Add DeepSeek circuit breaker** — 95.5% provider concentration is a single point of failure.

15. **Add ON_ERROR lifecycle hook** — errors currently bypass the hook system entirely.

16. **Wire adaptive_ensemble.py** — phase-dependent LLM selection could improve cost efficiency.

---

## Appendix: File Counts

| Category | Count |
|----------|-------|
| Total Python files | 167 |
| Orphaned (0 imports) | 18 (10.8%) |
| Tools registered with agents | 12 of 16 (75%) |
| Lifecycle hooks registered | 7 of 8 hook points |
| Idle scheduler jobs | 22 (was 23, evolution-2 removed) |
| PostgreSQL schemas | 10 + control_plane |
| SQLite databases | 2 (conversations + benchmarks) |
| Skill files | 222 |
| Evolution experiments | 722 |
| Training interactions | 9 |
| LLM calls recorded | 23,919 |
| Total LLM spend | $53.06 |
