# AndrusAI System Audit v3 — Final Comprehensive Report
**Date:** 2026-04-07
**Audit rounds:** v1 (10 criticals) → v2 (6 fixes) → v3 (this report)
**System:** 166 Python files, 10 PostgreSQL schemas, 2 SQLite DBs, 8 Ollama models

---

## Executive Summary

**Overall health: 76/100** (up from 65 in v1, 72 in v2)

| Category | v1 | v2 | v3 | Trend |
|----------|----|----|-------|-------|
| Request path | 90 | 90 | 92 | ✅ Stable |
| Knowledge bases | — | — | 75 | ⚠️ Fiction KB empty |
| Memory layers | — | — | 80 | ⚠️ Neo4j driver issue |
| Self-learning | 40 | 50 | 55 | ↑ Pipeline wired |
| Self-evolving | 35 | 55 | 60 | ↑ 73% keep rate |
| Self-healing | 30 | 35 | 40 | ↑ ON_ERROR hook |
| Self-awareness | — | — | 85 | ✅ Active |
| Infrastructure | 85 | 75 | 70 | ↓ DB rebuild lost data |
| Code hygiene | 50 | 65 | 70 | ↑ 3 files deleted |
| Control plane | — | — | 80 | ✅ New, working |

---

## 1. ORPHANED CODE — 15 files

Down from 18 (v1) → 16 (v2) → **15 files** (3 deleted across rounds).

**Active orphans:**
| File | Size | Category | Action Needed |
|------|------|----------|---------------|
| `agents/critic.py` | 2.5 KB | Agent | Wire into vetting crew or delete |
| `agents/self_improver.py` | 1.6 KB | Agent | Delete (SelfImprovementCrew uses inline) |
| `atlas/audit_log.py` | 3.6 KB | ATLAS | Wire or delete |
| `atlas/learning_planner.py` | 17.6 KB | ATLAS | Wire or delete (has broken brave_search import) |
| `contracts/events.py` | 3.2 KB | Docs | Keep (documentation-as-code) |
| `contracts/state.py` | 1.9 KB | Docs | Keep (documentation-as-code) |
| `contracts/firestore_schema.py` | 7.8 KB | Docs | Keep (documentation-as-code) |
| `control_plane/heartbeats.py` | 6.5 KB | Control | Wire into idle_scheduler (planned) |
| `evolution_db/eval_sets.py` | 10.5 KB | Evolution | Wire or delete |
| `personality/probes.py` | 7.5 KB | PDS | assessment.py uses own — delete |
| `proactive/proactive_behaviors.py` | 3.2 KB | Proactive | Wire into trigger_scanner or delete |
| `self_awareness/grounding.py` | 9.2 KB | Awareness | Wire into cogito or delete |
| `tools/bridge_tools.py` | 6.3 KB | Tools | Bridge disabled — delete |
| `tools/browser_tool.py` | 8.2 KB | Tools | Playwright not in Docker — delete |
| `tools/composio_tool.py` | 3.9 KB | Tools | Composio not installed — delete |

**2 broken imports:** `app/atlas/learning_planner.py` and `app/atlas/api_scout.py` both import `app.tools.brave_search` which doesn't exist.

---

## 2. KNOWLEDGE BASES

| KB | Status | Chunks | Size | Wired |
|----|--------|--------|------|-------|
| **General (enterprise)** | ✅ Operational | 1,641 | 18.4 MB | Yes — KnowledgeSearchTool in agents |
| **Philosophy (humanist)** | ✅ Operational | 3,026 | 88.8 MB | Yes — PhilosophyRAGTool, read-only |
| **Fiction/Literature** | ❌ **EMPTY** | 0 | 0 | Code exists, NO DATA ingested |
| **Web knowledge** | ✅ Operational | 1 | — | Yes — Firecrawl ingest |

**Fiction KB issue:** `app/fiction_inspiration.py` expects data at `/app/workspace/fiction_library/texts/` but the directory doesn't exist. The idle_scheduler `fiction-ingest` job runs but has nothing to ingest. Upload fiction texts via the dashboard Fiction Library section or create the directory with .md files.

---

## 3. MEMORY LAYERS

| Layer | Status | Size/Count | Notes |
|-------|--------|-----------|-------|
| **ChromaDB operational** | ✅ | 29 collections, 1,524 total embeddings | 12 active, 17 empty |
| **Conversation store** | ✅ | 289 messages, 154 tasks | 91.6% success rate |
| **Scoped memory** | ✅ | 573 policies, 535 tech radar, 136 shared | Active accumulation |
| **Mem0 (PostgreSQL)** | ⚠️ | 46 response_metadata rows | Working but Neo4j driver issue |
| **Mem0 (Neo4j)** | ❌ | 516 MB data | `No module named 'neo4j'` in cogito |
| **Result cache** | ✅ | 5 cached results | Working |
| **Belief state** | ✅ | 4 beliefs | Working |

**Neo4j issue:** The latest cogito reflection reports `No module named 'neo4j'`. The Neo4j Python driver may not be in requirements.txt, or the container can't reach Neo4j. The entity relationship graph (516MB of data) is inaccessible.

**ChromaDB `KeyError('_type')` issue:** This occurs when the chromadb Python client (v1.x) tries to talk to the ChromaDB server (v0.5.23) via HTTP. The system uses PersistentClient (direct disk) which avoids this, but some code paths may still try HttpClient.

---

## 4. SELF-AWARENESS

| Component | Status | Data |
|-----------|--------|------|
| System chronicle | ✅ | 6,610 chars, auto-generated |
| Cogito reflections | ✅ | 70 reflection files |
| Activity journal | ✅ | 59 journal entries |
| Homeostasis | ✅ | energy=0.97, frustration=0.005, confidence=0.95 |
| Agent state tracking | ⚠️ | 3 agents, ALL confidence=0.00 (broken) |
| Self-model | ✅ | Auto-generated from chronicle |
| World model | ✅ | 2 predictions tracked |

**Agent confidence issue:** `workspace/agent_state.json` shows all agents at confidence=0.00 while homeostasis reports confidence=0.95. These are two different systems with contradictory data. Agent state confidence updates may be broken.

---

## 5. TRAINING PIPELINE

```
Signal messages → POST_LLM_CALL hook → training_collector.py → raw/ JSONL
  9 interactions collected ✅
  0 quality-scored ❌ (curation not running effectively)
  0 curated ❌

idle_scheduler "training-curate" → run_curation()
  Quality scoring requires LLM judge call → likely fails silently

idle_scheduler "training-pipeline" → run_training_cycle()
  Checks MIN_TRAINING_EXAMPLES (100) → returns "insufficient_data"
  MLX training never executes (correct — need 100+ scored examples first)
```

**Root cause:** The training collector captures interactions but the curation pipeline's quality scoring LLM call likely fails or produces no output. The 9 interactions all have `quality_score: None`.

---

## 6. DATABASE HEALTH

### SQLite — conversations.db ✅
| Table | Rows | Status |
|-------|------|--------|
| messages | 289 | ✅ Active |
| tasks | 154 | ⚠️ crew column empty |
| INTEGRITY | OK | ✅ |

### SQLite — llm_benchmarks.db ❌ EMPTY
| Table | Rows | Status |
|-------|------|--------|
| benchmarks | 0 | ❌ Empty since rebuild |
| token_usage | 0 | ❌ **No tracking since rebuild** |
| request_costs | 0 | ❌ Empty |
| INTEGRITY | OK | ✅ |

**CRITICAL:** llm_benchmarks.db was rebuilt but the token tracking patch in rate_throttle.py may not be activating in the new container. The `_patched_track` function hooks into CrewAI's `BaseLLM._track_token_usage_internal` at import time — if rate_throttle.py is imported after CrewAI initializes, the patch may not apply.

### PostgreSQL — 32 tables across 10 schemas
Active tables: control_plane.tickets (6), control_plane.audit_log (14), control_plane.projects (4), feedback.response_metadata (46)
Empty tables: 22+ (feedback.events, personality.*, modification.*, training.runs, etc.)

---

## 7. EVOLUTION STATUS

| Metric | Value |
|--------|-------|
| Total experiments | 722+ |
| Variants archived | 48 (35 kept, 13 discarded) |
| Keep rate | 73% |
| Current baseline fitness | 0.91 (up from 0.50 initial) |
| Island evolution | Gen 0 (stuck) |
| MAP-Elites | 0 cells (stuck) |
| Adaptive ensemble | Never initialized |

**Topic fixation:** Recent experiments heavily focused on API credit error handling. The fuzzy dedup in `_auto_discover_topics()` should help diversify future topics.

---

## 8. SELF-HEALING STATUS

| Metric | Value |
|--------|-------|
| Errors in journal | 37 |
| Most common: BadRequestError | 15 |
| Auto-remediations deployed | 0 |
| Code audits run | 85 |
| Code audit findings | 0 |
| Homeostasis health | Stable (energy=0.97) |

Self-healing detects errors and diagnoses them but has never deployed an automated code fix.

---

## 9. LLM PROVIDER DISTRIBUTION

**UNKNOWN** — llm_benchmarks.db empty since rebuild. No visibility into:
- Which models handle requests
- Cost per provider
- Whether gemma4:26b is being used for background tasks
- Total LLM spend

---

## 10. CONTROL PLANE

| Table | Last Known Rows | Status |
|-------|----------------|--------|
| projects | 4 | ✅ Seeded |
| org_chart | 8 | ✅ Seeded |
| tickets | 6+ | ✅ Active |
| audit_log | 14+ | ✅ Active |
| budgets | 0 | Not configured |
| governance_requests | 0 | Not triggered |
| heartbeats | 0 | Scheduler not started |
| ticket_comments | 0 | No comments |

---

## 11. LIFECYCLE HOOKS

Expected registrations (from code analysis):
1. Priority 0: humanist_safety (PRE_LLM_CALL, immutable)
2. Priority 1: block_dangerous (PRE_TOOL_USE, immutable)
3. Priority 2: budget_enforcement (PRE_LLM_CALL)
4. Priority 10: self_correct (POST_LLM_CALL)
5. Priority 20: history_compress (PRE_LLM_CALL) — **may fail due to NameError (fixed in v2)**
6. Priority 50: memorize_tools (POST_TOOL_USE)
7. Priority 55: training_data (POST_LLM_CALL)
8. Priority 60: health_metrics (ON_COMPLETE)
9. Priority 65: error_audit (ON_ERROR)

---

## 12. PERSONALITY DEVELOPMENT

All 4 agents at initial state: assessments=0, stage=system_trust, coherence=0.50.
PDS was moved to idle job #5 in v2 fixes but has not yet had time to execute.

---

## RECOMMENDATIONS

### P0 — Critical (do now)
1. **Fix llm_benchmarks.db not recording** — the rate_throttle patch may not be hooking into CrewAI. Verify the patch applies at startup.

### P1 — High (this session)
2. **Fix Neo4j driver** — verify `neo4j` is in requirements.txt and importable in container
3. **Populate Fiction KB** — create `workspace/fiction_library/texts/` with at least one .md file to bootstrap
4. **Fix agent_state confidence** — all showing 0.00, likely never updated after task completion

### P2 — Medium (next session)
5. Fix broken brave_search import in atlas modules
6. Delete 7 clearly dead orphans (agents/self_improver.py, personality/probes.py, tools/bridge_tools.py, tools/browser_tool.py, tools/composio_tool.py, proactive/proactive_behaviors.py)
7. Wire heartbeats.py into idle_scheduler
8. Clean up 17 empty ChromaDB collections

### P3 — Low (backlog)
9. Diversify evolution topics beyond API credit errors
10. Investigate why training quality scoring produces no output
11. Reconcile homeostasis vs agent_state confidence values
12. Wire grounding.py into cogito cycle
