# AndrusAI Memory Architecture

> **Status**: Production-deployed. 116-check live diagnostic returns
> **114 OK / 2 informational / 0 FAIL** inside the gateway container.
> **Storage footprint**: 44 ChromaDB collections, 4 KB v2 stores
> (4,332 episteme + 141 experiential docs), 49 wiki pages, Mem0
> (postgres+neo4j), 10 MAP-Elites grids, plus the trajectory subsystem
> (arXiv:2603.10600).
> **Embedding backend**: Ollama `nomic-embed-text` @ 768-dim, pinned across
> ALL stores. CPU fallback explicitly disabled.

This document is the single source of truth for AndrusAI's memory
subsystem — every layer, every store, every wire. Read it before
modifying memory paths, persistence formats, or retrieval pipelines.

---

## Table of Contents

1.  [Design philosophy](#1-design-philosophy)
2.  [12-layer architecture](#2-12-layer-architecture)
3.  [Layer 1 — ChromaDB infrastructure](#3-layer-1--chromadb-infrastructure)
4.  [Layer 2 — Operational memory](#4-layer-2--operational-memory)
5.  [Layer 3 — Knowledge Bases v2](#5-layer-3--knowledge-bases-v2)
6.  [Layer 4 — Self-improvement pipeline](#6-layer-4--self-improvement-pipeline)
7.  [Layer 5 — Mem0 (cross-session facts)](#7-layer-5--mem0-cross-session-facts)
8.  [Layer 6 — Wiki](#8-layer-6--wiki)
9.  [Layer 7 — MAP-Elites quality-diversity grids](#9-layer-7--map-elites-quality-diversity-grids)
10. [Layer 8 — Self-awareness journals](#10-layer-8--self-awareness-journals)
11. [Layer 9 — SubIA layered memory](#11-layer-9--subia-layered-memory)
12. [Layer 10 — Trajectory-informed memory (arXiv:2603.10600)](#12-layer-10--trajectory-informed-memory-arxiv260310600)
13. [Layer 12 — Transfer Insight Layer (arXiv:2606.21099)](#13-layer-12--transfer-insight-layer-arxiv260621099)
14. [Layer 11 — Cross-layer wiring](#14-layer-11--cross-layer-wiring)
15. [Embedding strategy & dimension pinning](#15-embedding-strategy--dimension-pinning)
16. [Persistence layout on disk](#16-persistence-layout-on-disk)
17. [Major data flows](#17-major-data-flows)
18. [Safety invariants](#18-safety-invariants)
19. [Health checks & diagnostics](#19-health-checks--diagnostics)
20. [Maintenance & operations](#20-maintenance--operations)
21. [Failure modes & recovery](#21-failure-modes--recovery)
22. [Configuration surfaces](#22-configuration-surfaces)
23. [Glossary](#23-glossary)

---

## 1. Design philosophy

AndrusAI's memory is **not** a single store — it's an intentional
stratification of stores, each optimised for a different *temporal* and
*semantic* role. Five principles drive the layering:

1.  **Source of truth, then mirror.** Each fact lives in exactly one
    authoritative store. Other stores hold projections (the disk skill
    mirror, the trajectory Chroma index, the wiki hot cache). Mirrors
    can be rebuilt; sources cannot.

2.  **Embedding dimension is immutable system-wide.** All vector stores
    use 768-dim Ollama nomic-embed-text. No CPU fallback, no fly-on-
    demand alternative. Mixing dimensions silently corrupts vector
    retrieval; the system explicitly refuses to operate when the
    embedder is unavailable.

3.  **Infrastructure-level evaluation, agent-level synthesis.** The
    Self-Improver agent reads from the memory system; it never imports
    evaluation logic (novelty thresholds, attribution analysers,
    fitness composition). Those modules live behind a static-analysis
    boundary asserted by `tests/test_trajectory_safety_invariants.py`
    and the test_security suite.

4.  **Best-effort writes, defensive reads.** Every store-side function
    wraps in try/except at the outer boundary. A failed write must
    never break the surrounding crew execution. Reads return safe
    defaults when their backing store is unavailable.

5.  **Provenance is mandatory.** Every persisted artefact carries a
    `provenance` dict (or equivalent). Skills know which gap created
    them; gaps know which retrieval missed; tips know which trajectory
    they're attributed to. Bulk-archiving "everything from this bad
    trajectory" must be a single metadata query.

The layering reflects how memory is *used*, not how it's stored. A
trajectory tip ends up in ChromaDB, but the *concept* lives in the
trajectory subsystem (Layer 10), with the integrator (Layer 4) routing
into a KB v2 collection (Layer 3) for retrieval.

---

## 2. 12-layer architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                       AndrusAI Memory Architecture                        │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  L1 ─────────────────────────────────────────────────────────────────    │
│  │ ChromaDB (PersistentClient @ /app/workspace/memory)              │    │
│  │   • Ollama nomic-embed-text, 768-dim (immutable)                 │    │
│  │   • 44 collections, ~6,000 documents total                       │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                ▲   ▲   ▲                                 │
│                ┌───────────────┘   │   └──────────────────┐              │
│                │                   │                      │              │
│  L2 ──────────┴────┐   L3 ────────┴────┐   L4 ────────┐  │              │
│  │ Operational     │   │ KB v2          │   │ Self-     │  │              │
│  │   • team_shared │   │   • episteme    │   │ improve   │  │              │
│  │   • scoped_*    │   │   • experiential│   │ pipeline: │  │              │
│  │   • beliefs     │   │   • aesthetics  │   │  gaps →   │  │              │
│  │   • reflections │   │   • tensions    │   │  drafts → │  │              │
│  │   • self_reports│   │ +RetrievalOrch  │   │  KB →     │  │              │
│  │   • result_cache│   │ +Novelty Gate   │   │  evaluator│  │              │
│  └─────────────────┘   └─────────────────┘   └───────────┘  │              │
│                                                              │              │
│  L5 ──────────────┐    L6 ──────────────┐    L7 ──────────┐ │              │
│  │ Mem0           │    │ Wiki            │    │ MAP-Elites│ │              │
│  │  • postgres    │    │  workspace/wiki/│    │ per-role  │ │              │
│  │    +pgvector   │    │  6 sections,    │    │ grids:    │ │              │
│  │  • neo4j graph │    │  49 pages       │    │ commander,│ │              │
│  │  • cross-      │    │  +hot.md cache  │    │ researcher│ │              │
│  │    session     │    │  +meta auto-    │    │ coder,    │ │              │
│  │    facts       │    │   synthesis     │    │ writer, …│ │              │
│  └────────────────┘    └─────────────────┘    └──────────┘ │              │
│                                                              │              │
│  L8 ──────────────┐    L9 ──────────────┐    L10 ─────────┐ │              │
│  │ Self-awareness │    │ SubIA           │    │ Trajectory│ │              │
│  │  • activity    │    │ (phase 12)      │    │ (arXiv:   │ │              │
│  │    journal     │    │  • dual-tier    │    │  2603.    │ │              │
│  │  • error       │    │  • retrospective│    │  10600)   │ │              │
│  │    journal     │    │  • spontaneous  │    │  • capture│ │              │
│  │  • world model │    │  • world model  │    │  • attrib │ │              │
│  │  • homeostasis │    │  • MCSV         │    │  • tips   │ │              │
│  │  • agent_state │    │  • somatic      │    │  • calib  │ │              │
│  └────────────────┘    └─────────────────┘    └──────────┘ │              │
│                                                              │              │
│  L12 ─────────────────────────────────┐                      │              │
│  │ Transfer Insight Layer             │                      │              │
│  │ (arXiv:2606.21099)                 │                      │              │
│  │  • compile_queue (events)          │                      │              │
│  │  • free-tier nightly compile       │                      │              │
│  │  • sanitiser (3-tier scope ladder) │                      │              │
│  │  • shadow → active promotion       │                      │              │
│  │  • negative-transfer attribution   │                      │              │
│  │  • <transfer_memory> dispatch block│                      │              │
│  └────────────────────────────────────┘                      │              │
│                                                              │              │
│  L11 ────────────────────────────────────────────────────────┴──────┐    │
│  │ Cross-layer wiring                                              │    │
│  │  Commander → all hooks (PRE_LLM_CALL, POST, ON_COMPLETE)       │    │
│  │  Retrieval → Gap Detector (RETRIEVAL_MISS gaps)                │    │
│  │  Idle Scheduler → 9 memory-touching jobs                       │    │
│  │  Integrator → ChromaDB metadata (filterability)                │    │
│  │  Observer ↔ Attribution: shared 5-mode taxonomy                │    │
│  │  Evaluator ↔ Effectiveness: tip-decay sweep                    │    │
│  │  Healing/Evo/Grounding/Gaps → Transfer compile queue           │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────┘
```

Each subsequent section drills into one layer.

---

## 3. Layer 1 — ChromaDB infrastructure

**Module**: [`app/memory/chromadb_manager.py`](../app/memory/chromadb_manager.py)

ChromaDB is the substrate beneath ~80% of AndrusAI's memory. Every
persistent vector store funnels through this module — the embedding
backend, the client singleton, and per-collection cache live here.

### 3.1 Persistence

```
PERSIST_DIR = /app/workspace/memory      (Docker bind to ./workspace/memory)
client      = chromadb.PersistentClient(path=PERSIST_DIR)
```

The `chromadb.PersistentClient` is constructed lazily under a thread
lock. Reuse is deliberate — many call sites instantiate without
realising; the singleton avoids redundant SQLite handles.

### 3.2 Embedding backend (immutable)

```python
_OLLAMA_URL   = OLLAMA_EMBED_URL or LOCAL_LLM_BASE_URL
                or "http://host.docker.internal:11434"
_OLLAMA_MODEL = OLLAMA_EMBED_MODEL or "nomic-embed-text"
_EMBED_DIM    = 768   # IMMUTABLE
```

The 768-dim choice is locked in three places:

* `chromadb_manager._EMBED_DIM`
* The on-disk Chroma SQLite (per-collection embedding column width)
* The pgvector schema in postgres for Mem0

Changing the embedder requires migrating ALL three plus every existing
KB. The `_detect_backend()` function refuses to operate if Ollama is
unreachable and returns `EmbeddingUnavailableError` rather than fall
back to a 384-dim CPU model. This is the single most important safety
invariant in the memory layer — silent dimension drift corrupts every
vector retrieval system-wide.

### 3.3 The 768-dim cache

```
@functools.lru_cache(maxsize=4096)         # L1 in-process
def _embed_cached(text: str) -> tuple: ...

# L2 disk cache (survives container restart)
from app.memory import disk_cache as _dc
```

Two-level embedding cache: 4,096-entry LRU in-process, plus an on-disk
SHA-keyed cache that survives restarts. The orchestrator's cold-start
embedding is the dominant latency in agent dispatch — the disk cache
recovers ~70% of warm-state perf within a minute of restart.

### 3.4 Collection inventory (live, 44 collections)

Verified by the diagnostic running inside the gateway container:

| Category | Collections | Doc counts (live) |
|---|---|---|
| **KB v2 (Layer 3)** | `episteme`, `episteme_research`, `experiential`, `experiential_journal`, `aesthetic_patterns`, `aesthetics`, `tensions`, `unresolved_tensions` | 4332+141+rest |
| **Self-improvement (Layer 4)** | `learning_gaps`, `skill_records`, `consolidation_proposals` | 3 + 4 + 1 |
| **Operational** | `team_shared`, `self_reports`, `result_cache` | 6 + 6 + 4 |
| **Per-crew reflections** | `reflections_research`, `reflections_coding`, `reflections_writer`, `reflections_critic`, `reflections_introspector`, `reflections_coder`, `reflections_researcher` | 0–3 each |
| **Per-agent scratch** | `commander`, `introspector` | 0 each |
| **Scoped memory** | `scope_predictions`, `scope_beliefs`, `scope_tech_radar`, `scope_policies`, `scope_ecology`, `scope_team`, `scope_research_*`, `scope_world_model`, `scope_reflexion_lessons` | 0–10 each |
| **Reference / retrieval** | `wiki_corpus` (756), `literature_inspiration` (324), `self_knowledge` (198), `skills` (47) | as shown |
| **Evolution** | `evo_failures`, `evo_successes`, `evolution_patterns` | 1 + 0 + 1 |
| **Trajectory (Layer 10)** | `trajectory_index` | 0 (auxiliary) |
| **Healing** | `healing_knowledge` | 0 |

Collections are auto-created on first write (`get_or_create_collection`)
with cosine similarity (`hnsw:space="cosine"`).

### 3.5 Public API

| Function | Purpose |
|---|---|
| `embed(text) → list[float]` | Get a 768-dim vector. LRU + disk cached. |
| `store(collection, text, metadata)` | Store with auto-embedding + sanitisation. |
| `store_team(text, metadata)` | Shortcut to `team_shared`. |
| `retrieve(collection, query, n=5)` | Semantic kNN. |
| `retrieve_team(query, n=5)` | Shortcut. |
| `retrieve_with_metadata(...)` | Same + metadata + cosine distance. |
| `retrieve_filtered(collection, query, where, n)` | Pre-filter on metadata. |
| `get_client()` | Get the singleton. |
| `get_embed_dim()` → 768 | The pinned dimension. |

`store()` runs every input through `app.sanitize.validate_content()` —
prompt-injection patterns are blocked at the storage boundary so they
never get embedded into team memory in the first place.

### 3.6 Self-healing dimension mismatch

If a collection holds 384-dim vectors (pre-pinning) and a 768-dim write
arrives, ChromaDB raises a dimension error. `chromadb_manager.store()`
catches this, deletes the offending collection, and re-creates it
empty. Operational data is ephemeral; the source of truth (Mem0,
trajectory sidecars, wiki, KB v2) survives.

---

## 4. Layer 2 — Operational memory

**Modules**:
[`app/memory/scoped_memory.py`](../app/memory/scoped_memory.py),
[`app/memory/belief_state.py`](../app/memory/belief_state.py),
[`app/memory/system_chronicle.py`](../app/memory/system_chronicle.py),
[`app/conversation_store.py`](../app/conversation_store.py).

### 4.1 `team_shared` — cross-crew shared memory

The "blackboard". Every crew can `store_team(text, metadata)` to push a
finding/decision/observation visible to every other crew. The
post-crew telemetry hook writes a reflection digest here so adjacent
crews can read it on their next dispatch.

Used for:
* Reflexion lessons (so the next attempt at a similar task sees prior failure)
* Scope-flagged team decisions
* Cross-crew skill announcements

### 4.2 Belief state — beliefs about agents and the system

```python
# app/memory/belief_state.py
update_belief(target, belief, confidence)   # 0..1
get_beliefs(agent_name=None) → list[dict]
revise_beliefs(observation, agent_name)     # called from POST_LLM_CALL hook
get_team_state_summary() → str              # composed for context blocks
infer_intentions(agent_name) → dict
cleanup_stale_working_beliefs(max_age_hours=6)
```

Every crew completion goes through `revise_beliefs(...)` which writes a
new observation belief, ages older beliefs, and cleans up stale
working beliefs. Beliefs about agents fuel commander routing (low-
confidence on coding crew → escalate model tier on dispatch).

### 4.3 Scoped memory — per-scope projections

```python
# app/memory/scoped_memory.py
store_scoped(scope, text, ...)       # scope ∈ {team, agent, project, ecology, ...}
retrieve_scoped(scope, query, n=5)
retrieve_operational(scope, query, n=10)
retrieve_strategic(scope, query, n=5)
```

Scopes form a hierarchy. Team-level scopes (`scope_team`,
`scope_ecology`, `scope_world_model`, `scope_predictions`,
`scope_beliefs`) are read by every dispatch. Per-agent scopes
(`scope_research_*`) are read only by their owner. The
`scope_tech_radar` scope is read by the proactive scanner.

Each scope is a separate ChromaDB collection — keep the index small
and keep retrieval focused on the right semantic neighbourhood.

### 4.4 Per-crew reflections

Every crew's POST hook writes a structured reflection JSON into both
`reflections_<crew>` (per-crew) and `team_shared` (cross-crew). The
retrospective crew + the gap detector + the trigger scanner all read
these.

```json
{
  "role": "research",
  "task": "...",
  "went_well": "Completed in 12s",
  "went_wrong": "",
  "lesson": "research d=4 → high in 12s",
  "would_change": ""
}
```

### 4.5 Self-reports

Same hook also writes to `self_reports`:
```json
{
  "role": "research",
  "task_summary": "...",
  "confidence": "high",
  "completeness": "complete",
  "blockers": "",
  "risks": "",
  "duration_s": 12.3
}
```
These feed the proactive trigger scanner — when 3+ self-reports show
"low confidence" on similar topics, the scanner queues a learning
topic.

### 4.6 Result cache — hot-path memoization

`result_cache` collection (4 docs at last check) stores recent
high-cost outputs keyed by `(crew_name, task_hash)` with a 30-min TTL
(low difficulty) or 1h (high difficulty). Saves redundant LLM calls
when the same task arrives in quick succession.

### 4.7 System chronicle

```python
# app/memory/system_chronicle.py
generate_and_save() → str    # daily snapshot
load_chronicle() → str       # composed history for context
get_live_stats() → dict
```

Maintains a high-level chronicle of major system events (deploys,
errors, milestones) — used by the Self-Improver as situational context.

### 4.8 Conversation store

**Module**: [`app/conversation_store.py`](../app/conversation_store.py)

SQLite-backed (NOT Chroma). Stores every user-facing exchange with
ETA estimates, latency tracking, and cost-per-task attribution. Read
by the dashboard, `estimate_eta(crew_name)`, and the post-crew
telemetry hook for token/cost accounting.

---

## 5. Layer 3 — Knowledge Bases v2

**Packages**: `app.episteme`, `app.experiential`, `app.aesthetics`, `app.tensions`.

The four KBs replaced the pre-overhaul flat `workspace/skills/*.md`
mirror. Each is its own Chroma collection with a typed access layer.

### 5.1 The four KBs

| KB | Stores | Live count | Read by |
|---|---|---|---|
| **episteme** | Theoretical, cited, "what is true" — research summaries, factual references | 4,332 | Researcher, Coder when "what is X" |
| **experiential** | Distilled lived experience — narrative "we tried X, learned Y". Holds `entry_type ∈ {task_reflection, creative_insight, error_learning, interaction_narrative, evolution_reflection, episode, chapter, arc, epoch}`; the last four are produced by the narrative-self pipeline (`app/affect/{salience,episodes,narrative}.py` — see [`docs/SUBIA.md#affect`](SUBIA.md#affect)) and stored here, not authored as a memory subsystem. | 141 | Self-Improver, Retrospective crew, commander context pipeline |
| **aesthetics** | Style, tone, taste, judgement — "what good looks like" | 0 (cold) | Writer, Creative MAS |
| **tensions** | Unresolved contradictions, open questions | 0 (cold) | Self-Improver (recovery tips), Critic |

`episteme` and `experiential` are the workhorses today; `aesthetics`
and `tensions` are activated when their respective crews encounter
their domain.

### 5.2 Each KB exposes the same shape

```python
from app.episteme.vectorstore   import get_store    # → KnowledgeStore
from app.experiential.vectorstore import get_store  # → ExperientialStore
from app.aesthetics.vectorstore  import get_store
from app.tensions.vectorstore    import get_store
```

Each store provides:
* `add_documents(chunks, metadatas, ids)` (episteme) or
  `add_entry(text, metadata, entry_id)` (experiential) or
  `add_pattern(...)` (aesthetics) or `add_tension(...)` (tensions).
* `query(query_text, n=5)` for direct retrieval.
* Internal `_collection` (the underlying Chroma collection) for the
  Integrator/Evaluator's bulk-enumeration paths.

The variation in `add_*` API is intentional — each KB has different
chunking semantics. The Integrator's `_write_to_kb` adapter shields
callers from the differences.

### 5.3 Retrieval Orchestrator — unified cross-KB read

**Module**: [`app/retrieval/orchestrator.py`](../app/retrieval/orchestrator.py)

```python
from app.retrieval.orchestrator import RetrievalOrchestrator
from app.retrieval import config as cfg

orch = RetrievalOrchestrator(cfg.RetrievalConfig())
results = orch.retrieve(
    query="ethics of autonomous AI",
    collections=["episteme_research", "experiential_journal", ...],
    top_k=5,
    where_filter={"status": "active"},
    task_id="task_42",        # if set, weak retrievals emit RETRIEVAL_MISS gaps
)
```

Pipeline:

1. **Query decomposition** — complex queries split into ≤3 sub-queries.
2. **Parallel retrieval** — `(sub_query × collection)` pairs run on a
   thread pool, with a global timeout that returns partial results
   instead of crashing.
3. **Deduplication** — by text-content hash; keeps highest-scored copy.
4. **Temporal decay** (opt-in) — boosts recent docs with a half-life
   weight.
5. **Cross-encoder re-rank** — top-N candidates re-scored by a
   cross-encoder model for better relevance.
6. **Provenance tagging** — each result carries
   `provenance.collection`, `provenance.semantic_score`,
   `provenance.rerank_score`, `provenance.sub_query`.
7. **Self-improvement gap signal** — if `task_id` is set and the top
   score is below `RETRIEVAL_MISS_SCORE_THRESHOLD = 0.40`, emit a
   `GapSource.RETRIEVAL_MISS` LearningGap.

#### 5.3.1 Task-conditional helper (Phase 4 of arXiv:2603.10600)

```python
orch.retrieve_task_conditional(
    query="...", collections=[...],
    agent_role="coding",
    predicted_failure_mode="fix_spiral",   # narrows to recovery tips
    tip_types=["recovery", "strategy"],     # explicit override
    extra_where={"status": "active"},
    task_id="task_42",
)
```

Composes a `where_filter` from `(agent_role, tip_type, extra)`. When
the Observer predicts `fix_spiral` and no explicit tip_types is
supplied, the helper auto-narrows to `tip_type="recovery"` — the
paper's key recovery-tip injection mechanic.

### 5.4 Novelty Gate

**Module**: [`app/self_improvement/novelty.py`](../app/self_improvement/novelty.py)

```python
from app.self_improvement.novelty import novelty_report, NOVELTY_THRESHOLDS

rep = novelty_report(text, kbs=None)
# rep.decision: COVERED | OVERLAP | ADJACENT | NOVEL
# rep.nearest_distance: 0..2 (cosine distance, lower = more similar)
# rep.nearest_text, rep.nearest_kb, rep.nearest_id
```

Three layers of defense against KB drift:

```
Layer 1 (cheap):     novelty_report(topic_string)
Layer 2 (decisive):  novelty_report(generated_skill_content)
Layer 3 (continuous): Consolidator clusters and merges drift
```

Thresholds (cosine distance, lower = more similar):

```python
NOVELTY_THRESHOLDS = {
    COVERED:  (0.0,  0.30),   # already in the KBs — reject
    OVERLAP:  (0.30, 0.55),   # heavy overlap — propose extension
    ADJACENT: (0.55, 0.80),   # nearby — create with cross-link
    NOVEL:    (0.80, 2.0),    # distinct — create
}
```

The Novelty Gate runs against ALL FIVE collections in parallel
(`episteme_research`, `experiential_journal`, `aesthetic_patterns`,
`unresolved_tensions`, `team_shared`) and returns the nearest neighbour
across all of them.

---

## 6. Layer 4 — Self-improvement pipeline

**Package**: [`app.self_improvement`](../app/self_improvement/)

The structured pipeline that replaced the pre-overhaul ad-hoc loop:

```
   Gap Detector  →  Novelty Gate  →  Learner  →  Integrator  →  Evaluator  →  Consolidator
   (multi-source)   (drop dups)     (LLM)       (KB routing)   (usage)       (cluster)
```

### 6.1 Typed records

[`app/self_improvement/types.py`](../app/self_improvement/types.py):

```python
class GapSource(str, Enum):
    RETRIEVAL_MISS          = "retrieval_miss"
    REFLEXION_FAILURE       = "reflexion_failure"
    LOW_CONFIDENCE          = "low_confidence"
    USER_CORRECTION         = "user_correction"
    TENSION                 = "tension"
    MAPELITES_VOID          = "mapelites_void"
    USAGE_DECAY             = "usage_decay"
    TRAJECTORY_ATTRIBUTION  = "trajectory_attribution"   # Layer 10
    OBSERVER_MIS_PREDICTION = "observer_mis_prediction"  # Layer 10

class GapStatus(str, Enum):
    OPEN | TRIAGED | SCHEDULED | RESOLVED_EXISTING | RESOLVED_NEW | REJECTED

class NoveltyDecision(str, Enum):
    COVERED | OVERLAP | ADJACENT | NOVEL

@dataclass
class LearningGap:
    id: str; source: GapSource; description: str
    evidence: dict; signal_strength: float; detected_at: str
    status: GapStatus = OPEN; resolution_notes: str = ""

@dataclass
class SkillDraft:
    id: str; topic: str; rationale: str; content_markdown: str
    proposed_kb: str = "episteme"
    supersedes: list[str]; created_from_gap: str
    novelty_at_creation: float
    # Trajectory-sourced provenance (Layer 10)
    tip_type: str = ""              # strategy|recovery|optimization|""
    source_trajectory_id: str = ""
    agent_role: str = ""

@dataclass
class SkillRecord:
    id: str; topic: str; content_markdown: str; kb: str
    status: str = "active"          # active|superseded|archived
    superseded_by: str = ""
    usage_count: int; last_used_at: str
    provenance: dict; created_at: str
    requires_mode: str; requires_tier: str       # conditional activation
    fallback_for_mode: str; requires_tools: list[str]
```

### 6.2 Gap detector

**Module**: [`app/self_improvement/gap_detector.py`](../app/self_improvement/gap_detector.py)

```python
emit_retrieval_miss(query, top_score, collections, task_id)
emit_reflexion_failure(task, crew_name, retries, reflections)
emit_mapelites_voids(roles, max_per_role=2, min_neighbor_fitness=0.55)
emit_trajectory_attribution(...)        # Layer 10
emit_observer_mis_prediction(...)       # Layer 10
get_recent_evidence_block(limit=10) → str
```

Source-weighted signal strength:

```python
SOURCE_WEIGHTS = {
    USER_CORRECTION:         0.9,
    REFLEXION_FAILURE:       0.8,
    TRAJECTORY_ATTRIBUTION:  0.75,
    RETRIEVAL_MISS:          0.6,
    TENSION:                 0.5,
    LOW_CONFIDENCE:          0.4,
    OBSERVER_MIS_PREDICTION: 0.35,
    MAPELITES_VOID:          0.3,
    USAGE_DECAY:             0.2,
}
```

Higher = higher priority for the topic-discovery loop.

### 6.3 Gap store

**Module**: [`app/self_improvement/store.py`](../app/self_improvement/store.py)
**Collection**: `learning_gaps`

```python
emit_gap(gap)               # idempotent: same (source, description) within 24h upserts
list_open_gaps(limit=20, source=None)
update_gap_status(gap_id, status, notes="")
query_gaps(query_text, n=5)
prune_old_gaps(max_age_days=60)
```

Idempotency keys on `sha256(source + normalized_description)[:12]`. The
24h dedup window prevents flood when a chronic gap re-fires every
crew run; signal strength upserts via max() so re-detection raises
priority but never lowers it.

### 6.4 Novelty Gate

See §5.4. The Gate runs at Layer 1 (topic) before the Learner spends
any tokens, and again at Layer 2 (generated content) before the
Integrator writes.

### 6.5 The Learner

**Module**: [`app/crews/self_improvement_crew.py`](../app/crews/self_improvement_crew.py)

The Self-Improver crew has four modes:

| Mode | Trigger | Source | Output |
|---|---|---|---|
| `run()` | Idle scheduler `learn-queue` job | Topic queue file (`workspace/skills/learning_queue.md`) | SkillDraft → Integrator |
| `learn_from_youtube(url)` | Direct invocation | YouTube transcript | SkillDraft → Integrator |
| `run_trajectory_tips()` | Idle scheduler `trajectory-tips` job | `LearningGap` where `source==TRAJECTORY_ATTRIBUTION` (Layer 10) | SkillDraft → Integrator |
| `run_improvement_scan()` | Idle scheduler `improvement-scan` job | System gap analysis | Proposal (separate code-modification pipeline) |

All four modes route through the same Integrator API. The variation
is in the *source* of the draft, not in how it's persisted.

### 6.6 The Integrator

**Module**: [`app/self_improvement/integrator.py`](../app/self_improvement/integrator.py)

```python
integrate(draft: SkillDraft, novelty_kbs=None) → SkillRecord | None
```

Workflow:

1.  **Layer 2 novelty check** (decisive): rejects COVERED drafts.
2.  **classify_kb(draft)** — uses `draft.proposed_kb` if set, otherwise
    invokes a 16-token LLM classifier.
3.  **Allocate deterministic ID**: `skill_<kb>_<sha256(kb,topic,now)[:16]>`.
4.  **Build provenance** including (when set) trajectory fields:
    * `tip_type` — strategy / recovery / optimization
    * `source_trajectory_id`
    * `agent_role`
5.  **`_write_to_kb(kb, record)`** — adapter that knows the per-KB
    `add_*` API. Mirrors trajectory-provenance keys into ChromaDB
    metadata so retrieval can filter on them.
6.  **Mark supersession** — old records' `status="superseded"`, `superseded_by=new_id`.
7.  **Persist SkillRecord** to the cross-KB index (`skill_records`
    collection — the source of truth for enumeration).
8.  **Close the originating gap** with `RESOLVED_NEW` and a notes link
    to the new record.

### 6.7 SkillRecord index

**Collection**: `skill_records`

The Integrator persists every active SkillRecord here (in addition to
the KB-specific store) as a JSON-document with metadata. This is the
*single* source of truth for "what skills exist" — the Evaluator,
Consolidator, and dashboard read this collection. The KB-specific
stores hold the actual content; the index has a copy plus all the
metadata needed for cross-KB queries.

#### 6.7.1 Retrieval API and contamination defences (May 2026)

```python
search_skills(query, n=6) → list[SkillRecord]
search_skills_scored(query, n=6) → list[(SkillRecord, cosine_distance)]
```

Both functions semantically search the index using the query's embedding
against `hnsw:space=cosine`. Distances live in `[0, ~2]`: 0 = identical,
~1 = orthogonal, >1 = anti-correlated. The `_scored` variant is the one
to prefer when the caller needs to gate on closeness; the unscored
function is kept for the recovery `skill_chain` strategy and for
backward compatibility.

The orchestrator's pre-task context loader (`_load_relevant_skills` in
`app/agents/commander/context.py`) layers four contamination defences
on top of the raw search. The full chain — applied in order — is:

| Layer | What it does | Why |
|-------|--------------|-----|
| 1. **Subject-less message detection** | If the message is short and composed entirely of filler/pronoun/generic-execution tokens (`"execute the plan"`, `"run it"`, `"produce the report"`), substitute the last 3 user lines from the conversation history as the retrieval query. If no history exists, return empty (no skills injected). | Short subject-less queries embed to a near-uniform direction — every skill in the index is roughly equidistant. Returning the top-N degenerates to "whichever skill happened to be closest to that direction at index time". The May 2026 weather-vs-forest contamination incident was exactly this. |
| 2. **Quality filter** | Drop records whose `topic` contains placeholder markers (`****`, `_____`, `<redacted>`, `[REDACTED]`). | Auto-skill-creation paths sometimes leak the editor's redaction sentinels into topic strings. Those records are low-quality artifacts and should never surface as authoritative knowledge. |
| 3. **Semantic distance gate** | Drop records whose cosine distance exceeds `_SKILL_DISTANCE_CEILING` (default 0.55). | Even the top-N includes weak matches when the index has nothing closer. The ceiling is intentionally tighter than the novelty `OVERLAP→ADJACENT` cutoff (0.55) because skill *injection* into a crew prompt is high-bar — the LLM weights `RELEVANT KNOWLEDGE` blocks heavily, so a weak match becomes a strong steer. |
| 4. **Conditional activation** | Honour `requires_mode` / `requires_tier` / `fallback_for_mode` / `requires_tools` predicates via `SkillRecord.matches_context`. | Phase 3 of the overhaul. Skills tagged `requires_mode="local"` should not surface in cloud-mode runs, etc. |

Layer 1 is the decisive guard for the production failure mode. Layer 3
is the safety net for non-subject-less queries that still happen to
have only orthogonal matches. The two compose: subject-less recovery
swaps in the conversation topic, then the distance gate validates the
recovered query produces tight matches before anything is injected.

The orchestrator threads `conversation_history` into the loader so
Layer 1 can consult it (`orchestrator._run_crew_inner` →
`_ctx_pool.submit(_load_relevant_skills, crew_task, 3, conversation_history)`).
Without that arg the loader degrades to "skip on subject-less" rather
than "recover topic from history" — never to "guess".

### 6.8 Disk mirror (back-compat)

```python
regenerate_disk_mirror(out_dir=workspace/skills, force=False) → int
```

Some legacy code reads `.md` files from disk. The mirror regenerates
them from the KB index on demand. **Sparse-index guard**: if the index
holds <5 records, the mirror refuses to wipe (prevents data loss
during cold-start). **Marker-scoped deletion**: only files with the
`<!-- generated-by: self_improvement.integrator -->` sentinel get
overwritten — foreign content is preserved.

### 6.9 Evaluator

**Module**: [`app/self_improvement/evaluator.py`](../app/self_improvement/evaluator.py)

```python
record_hits(record_ids)              # called from RetrievalOrchestrator
flush_hits()                         # batched index update
record_task_outcome(task_id, success)
scan_for_decay()                     # USAGE_DECAY gaps for skills idle >30d
scan_for_low_effectiveness_tips()    # Phase 6: tip-effectiveness sweep
usage_distribution() → dict          # zombies, gini, by-kb breakdown
```

Hits are buffered in memory (avoids hammering Chroma per retrieval)
and flushed on:
* explicit `flush_hits()` (post-crew telemetry hook)
* threshold (every 10 accumulated hits)
* idle-scheduler `evaluator-sweep` job

#### 6.9.1 Decay sweeps

Two complementary sweeps:

* **`scan_for_decay`** — time-based. A skill not retrieved in
  30+ days emits a USAGE_DECAY gap (60+ days = stale, higher signal).
* **`scan_for_low_effectiveness_tips`** — outcome-based (Phase 6).
  A trajectory tip with ≥10 uses and <35% effectiveness emits a
  `USAGE_DECAY` gap with `evidence.reason="low_effectiveness"`.

Both sweeps run in the `evaluator-sweep` idle job. External-topic
skills only see the time-based path (no effectiveness data); trajectory
tips see both.

### 6.10 Consolidator

**Module**: [`app/self_improvement/consolidator.py`](../app/self_improvement/consolidator.py)

```python
run_consolidation_cycle(auto_merge=True)
list_proposals(limit=200) → list[dict]
migrate_legacy_skills(dry_run=False)
recover_from_team_shared()
```

Periodic clustering of near-duplicate active SkillRecords. When two
records have cosine distance <0.30, the Consolidator proposes a merge.
With `auto_merge=True` and provenance-clean candidates, it applies the
merge directly: combines content, marks the older record `superseded`,
records both lineage chains in the new record's provenance.

### 6.11 Metrics

**Module**: [`app/self_improvement/metrics.py`](../app/self_improvement/metrics.py)

```python
pipeline_funnel() → dict           # gaps → drafts → integrated → used
topic_diversity() → dict           # Shannon entropy over KB clusters
novelty_histogram(sample_size=50)  # decision distribution
trajectory_health_summary(limit=200)  # Phase 6
health_summary()                   # one-call dashboard aggregate
```

Pure read-only aggregations. Cheap enough for dashboard polling.

---

## 7. Layer 5 — Mem0 (cross-session facts)

**Module**: [`app/memory/mem0_manager.py`](../app/memory/mem0_manager.py)

Mem0 is the only store that survives ChromaDB resets — it lives in
postgres+neo4j. Used for:

* **Owner facts**: things the user said about themselves.
* **Project facts**: persistent project state across sessions.
* **Cross-agent facts**: extracted via LLM from conversation history.

### 7.1 Backends

```yaml
# docker-compose.yml
postgres:
  image: postgres   # + pgvector extension
  volumes: ./workspace/mem0_pgdata
neo4j:
  image: neo4j:5-community
  volumes: ./workspace/mem0_neo4j
```

* **Postgres + pgvector** holds the vector store (768-dim, matches Chroma).
* **Neo4j** holds the relationship graph: facts link to topics, agents,
  and other facts forming a knowledge graph.

### 7.2 Public API

```python
get_client()                         # → mem0.Memory or None
store_memory(text, agent_id, metadata)
store_memory_async(text, agent_id, metadata)   # fire-and-forget thread pool
store_conversation(messages, ...)
search_memory(query, agent_id, n=5)
search_shared(query, n=5)
search_agent(query, agent_id, n=5)
get_all_memories(agent_id=None)
```

### 7.3 Agent-isolated namespaces

Each agent has its own user_id (`commander`, `research`, etc.). Cross-
agent reads use `search_shared` (no user_id filter). The
`mem0_user_id` setting ("owner") stores facts about the human operator.

### 7.4 Fact extraction (LLM-mediated)

Mem0 uses a configurable LLM (defaulted to local Ollama
`qwen3:30b-a3b`) to extract atomic facts from conversation transcripts.
The local LLM choice keeps fact-extraction free of cost ceilings.

### 7.5 Embedding consistency

`mem0_embedder_model = "nomic-ai/nomic-embed-text-v1.5"` — same
model as ChromaDB, same 768-dim. Important for semantic compatibility:
a fact stored in Mem0 has the same vector geometry as a skill stored
in episteme.

### 7.6 Resilience

`mem0_manager.get_client()` returns `None` (not raises) when postgres
or neo4j is unreachable. Every call site checks for `None` and degrades
gracefully — system runs without persistent memory if needed.

---

## 8. Layer 6 — Wiki

**Path**: `wiki/`

Filesystem markdown wiki, written by the team and consumed by both
agents (via `wiki_corpus` Chroma collection) and humans.

### 8.1 Sections

```
wiki/
├── archibal/      (1 page)   Architecture decisions
├── kaicart/       (1 page)   Open API integrations
├── meta/          (43 pages) Auto-synthesised from skills
├── philosophy/    (1 page)   Foundational principles
├── plg/           (1 page)   Product-led growth notes
├── self/          (1 page)   Self-knowledge
├── hot.md                    Recent activity feed
├── index.md                  Navigation
└── log.md                    Append-only event log
```

### 8.2 Hot cache

`wiki/hot.md` is regenerated periodically by the idle scheduler
(`wiki-hot-cache` job). Lists the most-recently-active pages so the
context blocks pull in relevant rather than archival material.

### 8.3 Auto-synthesis

The `wiki-synthesis` idle job promotes ready skill files from
`workspace/skills/` into `wiki/meta/` as auto-synthesised pages.
Each page carries `tags: self-improvement,skills,auto-synthesised`
and a `source: workspace/skills/<file>` reference.

### 8.4 Lint

The `wiki-lint` idle job validates internal links, tag consistency,
and frontmatter. Issues are logged but not auto-fixed; the operator
addresses them via the Self-Improver's improvement scan if persistent.

### 8.5 Wiki corpus collection

The wiki content is also ingested into Chroma's `wiki_corpus`
collection (756 docs at last check). Some retrieval paths query both
the KB v2 collections AND `wiki_corpus` — wiki is treated as a
parallel knowledge surface for human-curated content.

### 8.6 Wiki write tools

```python
from app.tools.wiki_tools import WikiWriteTool, WIKI_ROOT
tool = WikiWriteTool()
tool._run(action="create", section="meta", slug="...", title=..., content=...,
           author=..., confidence=..., tags=..., source=...)
```

Used by the Self-Improver for synthesis writes and by direct
crew-internal writes. Every write through the tool is logged in
`wiki/log.md`.

---

## 9. Layer 7 — MAP-Elites quality-diversity grids

**Modules**:
[`app/map_elites.py`](../app/map_elites.py),
[`app/map_elites_wiring.py`](../app/map_elites_wiring.py).

Per-role quality-diversity archives. Maintains a *grid* of strategies
keyed on extracted features (complexity, cost-efficiency,
specialisation), preserving the highest-fitness strategy in each cell.

### 9.1 Per-role grids

```
workspace/map_elites/
├── commander/    state.json
├── researcher/   state.json
├── coder/        state.json
├── writer/       state.json
├── critic/       state.json
├── media/        state.json
├── pim/          state.json
├── financial/    state.json
├── desktop/      state.json
└── coding/       state.json
```

### 9.2 Wiring

Every crew completion goes through `_post_crew_telemetry` →
`record_crew_outcome(CrewOutcome(...))` → MAP-Elites write. The grid
sees real fitness signals composed from observable data:

```python
FITNESS_WEIGHTS = {
    "quality_gate": 0.35,
    "confidence":   0.25,
    "completeness": 0.15,
    "latency":      0.15,
    "retry_cost":   0.10,
}
```

A failure still writes (low-fitness entry in a void cell is useful
exploration signal). Only uncaught exceptions skip writes.

### 9.3 Latency baselines

`map_elites_wiring._record_baseline_sample` maintains a rolling 50-
sample window per role of (latency / difficulty). Median computed when
≥10 samples exist; before that, a 12s/difficulty cold-start constant
applies. Used by the latency-fitness component to score "faster than
expected" for the role/difficulty combination.

### 9.4 Generations and persistence

Every 10 writes per role advances the grid's generation counter and
calls `db.persist()` (atomic JSON dump). Artefact history is preserved
across generations so the Learner can examine "what worked at gen 5
that doesn't at gen 12".

### 9.5 Stochasticity injection

For difficulty 4–7 tasks on trial 1, the commander has a 20% chance to
inject a per-role variation into the task prompt
(`PROMPT_VARIATIONS[crew_name]`). This produces feature-vector
variance so MAP-Elites populates more cells over time.

### 9.6 Void detection

```python
get_voids(min_neighbor_fitness=0.55, min_neighbors_filled=2, top_n=2)
```

A "void" is an empty cell flanked by ≥2 high-fitness neighbours — the
system performs well *around* this region but never tried *exactly*
this configuration. The `gap_detector.emit_mapelites_voids` periodic
sweep emits MAPELITES_VOID gaps for these as low-priority learning
topics.

### 9.7 Mutation context

The Learner's `_get_map_elites_context(topic)` pulls diverse
high-fitness strategies as inspiration for new skills. The Phase 4
trajectory `compose_trajectory_hint_block` complements this with
relevance-targeted tips.

### 9.8 Island migration

The `map-elites-migrate` idle job cross-pollinates top performers
between role-specific islands periodically. Prevents islands drifting
into wholly disjoint local optima while keeping niche pressure.

---

## 10. Layer 8 — Self-awareness journals

**Package**: `app.self_awareness` (Phase-1 shims) + canonical
`app.subia.*`.

### 10.1 Activity journal

```python
from app.self_awareness.journal import get_journal, JournalEntry, JournalEntryType
get_journal().write(JournalEntry(
    entry_type=JournalEntryType.TASK_COMPLETED,
    summary="research task (d=4): success",
    agents_involved=["research"],
    duration_seconds=12.3,
    outcome="success",
))
```

Storage: `workspace/self_awareness_data/journal/JOURNAL.jsonl`
(append-only).

### 10.2 Error journal

`workspace/error_journal.json` — structured record of every uncaught
exception or quality-gate failure. Read by the Self-Improver during
gap detection.

### 10.3 World model

```python
from app.subia.belief.world_model import (
    store_causal_belief, recall_relevant_beliefs,
    store_prediction, store_prediction_result, recall_relevant_predictions,
)
```

Stores expected vs. actual outcomes ("Expected research to handle d=4
in <15s; got 12s — research reliable at d=4"). Read by the Observer
when assessing prediction confidence; written by the post-crew hook.

### 10.4 Homeostasis

```python
from app.self_awareness.homeostasis import update_state, get_state
update_state("task_complete", crew_name, success=True, difficulty=4,
             somatic_valence=0.5)
state = get_state()   # → SomaticMarker(valence, arousal, ...)
```

Per-system "proto-emotional" state. Influences:
* Risk aversion in planning
* Reflexion retry aggressiveness
* Difficulty estimation conservatism

### 10.5 Agent state (per-role stats)

```python
from app.subia.self.agent_state import (
    record_task, get_agent_stats, get_all_stats,
    get_best_crew_for_difficulty,
)
```

Per-role rolling stats: task count, success rate by difficulty, average
duration, confidence distribution. Read by the commander's routing
heuristics.

### 10.6 Cogito

`workspace/self_awareness_data/reflections/cogito_*.json` — periodic
"who am I" reflections triggered by significant events. Used as
context for high-stakes decisions.

### 10.7 Somatic marker

```python
from app.self_awareness.somatic_marker import record_experience_sync
record_experience_sync(agent_id, context_summary, outcome_score,
                        outcome_description, task_type, venture)
```

Damasio-inspired: associates context with outcome valence. Builds
"gut feel" over time about (context → outcome) pairs. Influences
planning when similar contexts recur.

### 10.8 Belief state (operational)

See §4.2. The belief store is technically "self-aware" memory but
is wired through `app.memory.belief_state` for historical reasons.

---

## 11. Layer 9 — SubIA layered memory

**Package**: [`app.subia`](../app/subia/) — see [docs/SUBIA.md](SUBIA.md)
for the full architecture.

SubIA is AndrusAI's consciousness/sentience-indicator layer. Memory-
relevant subsystems:

### 11.1 Dual-tier memory

```python
from app.subia.memory.dual_tier import DualTierMemoryAccess
dt = DualTierMemoryAccess()
```

* **Working tier** — recent, high-resolution, fast-access.
* **Long-term tier** — consolidated, summarised, compressed.

Promotion from working → long-term happens during retrospective
sweeps. Demotion (forgetting) happens during the consolidator's
periodic cleanup.

### 11.2 Spontaneous recall

```python
from app.subia.memory.spontaneous import SpontaneousRecall
```

Periodic random sampling of memories triggered by reverie cycles.
Surfaces unexpected connections between distant memories — the
"creative leap" mechanism.

### 11.3 Retrospective

```python
from app.subia.memory.retrospective import RetrospectiveAnalysis
```

End-of-day style review: recent task outcomes, prediction accuracy,
homeostatic drift. Feeds the proactive learning loop.

### 11.4 Consolidator (SubIA-side)

```python
from app.subia.memory.consolidator import MemoryConsolidator
```

Sleep-style consolidation of working memories into long-term form.
Different from `self_improvement.consolidator` (which clusters
SkillRecords) — SubIA's consolidator works on raw memory traces.

### 11.5 MetacognitiveStateVector (MCSV)

```python
from app.subia.belief.internal_state import (
    MetacognitiveStateVector, CertaintyVector, SomaticMarker,
)
mcsv = MetacognitiveStateVector.from_state(
    cv, sm, mem0_hit_rate=..., token_depth_ratio=...
)
mcsv.requires_observer   # → bool, gates the Observer
mcsv.to_context_string()  # → compact for prompt injection
```

The single most-checked memory primitive. Derived from CertaintyVector
+ SomaticMarker + Mem0 hit rate + token depth. Drives:
* Observer activation (PRE_LLM_CALL)
* Response hedging
* Reflexion retry tier escalation
* Confidence calibration

### 11.6 Inferential competition + prediction hierarchy

Subsystems at `app.subia.prediction.*` maintain prediction hierarchies
and surprise routing. Their internal stores feed homeostasis and the
world model.

### 11.7 Belief outbox — Postgres ↔ Neo4j ↔ ChromaDB reconciler

```python
from app.memory.belief_outbox import (
    reconcile_belief_outbox,        # Postgres → Neo4j
    sync_new_beliefs_to_chromadb,   # Postgres → ChromaDB (incremental)
    backend_info,                   # diagnostic snapshot for the dashboard
)
```

Each SubIA Phase-2 belief lives in **three** stores: Postgres
(canonical row in `beliefs`), Neo4j (`:Belief` node, projected for
graph queries), and ChromaDB (text embedding for semantic recall).
The application's write path is fire-and-forget against Neo4j and
periodic-only against ChromaDB — meaning a Neo4j blip or a never-run
sync can leave the projections stale relative to Postgres.

`belief_outbox.py` ships two periodic reconcilers, registered as LIGHT
idle-scheduler jobs:

| Job | Direction | Mechanism |
|---|---|---|
| `belief-outbox-neo4j` | Postgres → Neo4j | Read every `belief_id` from `beliefs`; query Neo4j for existing `:Belief` nodes; backfill the missing ones via `neo4j_mirror.mirror_belief()`. Idempotent; eventually consistent within the next idle pass. |
| `belief-outbox-chroma` | Postgres → ChromaDB | Watermark-based incremental: read beliefs whose `last_updated > watermark`; index each into ChromaDB's `beliefs` collection; advance the watermark only on success (crash-safe). |

Subsystem boundary: `app/subia/belief/store.py` is Tier-3 protected
and never modified by the reconcilers. Postgres remains the system of
record; Neo4j and ChromaDB are projections that converge under the
outbox sweep.

The dashboard's [`GET /api/cp/idle/jobs`](CONTROL_PLANES.md#idle-scheduler-snapshot)
endpoint surfaces both jobs' status (last success/failure age,
cooldown state, currently_running) so operators can confirm the
projections are fresh.

---

## 12. Layer 10 — Trajectory-informed memory (arXiv:2603.10600)

**Package**: [`app.trajectory`](../app/trajectory/)

The newest memory layer (Phases 0–6 of the implementation). Implements
the paper's full pipeline: capture per-crew execution trajectories,
attribute decisions to outcomes, distil tips, inject them back into
future runs, and track per-tip effectiveness.

### 12.1 Components

```
app/trajectory/
├── types.py               TrajectoryStep | Trajectory | AttributionRecord
├── logger.py              begin → capture → end → on_crew_complete
├── store.py               JSON sidecars + Chroma index + read APIs
├── attribution.py         Post-hoc Decision Attribution Analyzer (LLM)
├── tip_builder.py         AttributionRecord → SkillDraft for the Learner
├── context_builder.py     Compose <trajectory_tips> block for prompts
├── calibration.py         Observer ↔ Attribution precision/recall
├── effectiveness.py       Per-tip use → outcome correlation
└── replay.py              Audit/debug bundle + format_text
```

### 12.2 Trajectory record

```python
@dataclass
class TrajectoryStep:
    step_idx: int
    agent_role: str
    phase: str           # routing | observer | crew | reflexion | quality
    planned_action: str          # ≤400 chars
    tool_name: str
    tool_args_sample: str        # ≤400 chars
    tool_args_hash: str
    output_sample: str           # ≤400 chars
    output_hash: str
    observer_prediction: dict    # if Observer fired
    elapsed_ms: int
    tokens_prompt/completion: int
    mcsv_snapshot: str
    started_at: str

@dataclass
class Trajectory:
    trajectory_id: str           # UUID-prefixed: traj_<hex16>
    task_id: str                 # ties to conversation_store
    crew_name: str
    task_description: str
    steps: list[TrajectoryStep]
    started_at, ended_at: str
    outcome_summary: dict        # populated by on_crew_complete
    injected_skill_ids: list[str]   # Phase 6 — for effectiveness correlation

@dataclass
class AttributionRecord:
    attribution_id: str
    trajectory_id: str
    verdict: str           # failure | recovery | optimization | baseline
    failure_mode: str      # confidence_mirage | fix_spiral | … | none
    attributed_step_idx: int
    confidence: float
    narrative: str         # ≤400 chars
    suggested_tip_type: str  # strategy | recovery | optimization | ""
    created_at: str
```

### 12.3 Capture (Phase 1)

The commander hooks call:
```python
begin_trajectory(task_id, crew_name, task_description) → Trajectory
capture_step(step)                  # routing
capture_observer_prediction(pred)    # observer
capture_step(step)                  # crew
capture_step(step)                  # reflexion (each retry)
capture_step(step)                  # quality
note_injected_skills(skill_ids)     # Phase 6
end_trajectory(outcome_summary)
on_crew_complete(outcome, trajectory)
```

Per-call isolation via `contextvars.ContextVar` — concurrent crew
executions don't leak each other's trajectories. The five phase
constants are stable strings so persisted JSON survives schema changes.

### 12.4 Attribution Analyzer (Phase 2)

**Module**: [`app/trajectory/attribution.py`](../app/trajectory/attribution.py)

Mirror of the Observer — runs *after* execution rather than before.

```python
maybe_analyze(trajectory) → AttributionRecord | None
analyze(trajectory) → AttributionRecord    # unconditional
```

`maybe_analyze` gates on:
* `outcome.passed_quality_gate == False`, OR
* `outcome.retries > 0`, OR
* `outcome.reflexion_exhausted`, OR
* `outcome.is_failure_pattern`, OR
* slow run (≥45s @ d≤5, ≥90s @ d>5), OR
* recovery case (Observer predicted failure with conf≥0.6, run succeeded).

Otherwise returns `None` — baseline runs never pay the LLM cost.

The Analyzer reuses the Observer's failure-mode taxonomy verbatim so
Phase 5 calibration aligns:
```
confidence_mirage | fix_spiral | consensus_collapse |
hallucinated_citation | scope_creep | none
```

Outputs both:
* An `AttributionRecord` persisted alongside the trajectory sidecar.
* A `LearningGap(source=TRAJECTORY_ATTRIBUTION)` emitted via
  `gap_detector.emit_trajectory_attribution`.

### 12.5 Tip synthesis (Phase 3)

The Self-Improver's `run_trajectory_tips()` mode reads
`TRAJECTORY_ATTRIBUTION` gaps, loads the underlying trajectory +
attribution from the store, and asks the Learner to distil a
strategy/recovery/optimization tip via `tip_builder.build_tip_task`.

KB pre-routing (deterministic, bypasses Integrator's LLM classifier):

| `tip_type` | KB |
|---|---|
| `strategy` | experiential |
| `recovery` | tensions |
| `optimization` | episteme |

The resulting SkillDraft flows through the unchanged Integrator; the
KB-write metadata carries `tip_type` + `source_trajectory_id` +
`agent_role` so retrieval can filter on them.

### 12.6 Task-conditional retrieval (Phase 4)

In the commander's dispatch path, *after* the Observer fires:

```python
if settings.task_conditional_retrieval_enabled:
    hint = compose_trajectory_hint_block(
        crew_name=crew_name,
        task_text=enriched_task,
        predicted_failure_mode=_observer_prediction.get("predicted_failure_mode", ""),
    )
    if hint:
        enriched_task = hint + "\n\n" + enriched_task
```

The hint block:
* Queries all 4 KB v2 collections via `retrieve_task_conditional`.
* Filter: `agent_role + tip_type` (auto-narrowed to `recovery` when
  Observer predicts `fix_spiral`).
* Wraps results in `<trajectory_tips>...</trajectory_tips>` tags so the
  LLM treats them as hints, not instructions.
* Records surfaced SkillRecord IDs via `note_injected_skills` for
  Phase 6 effectiveness correlation.

### 12.7 Calibration (Phase 5)

**Module**: [`app/trajectory/calibration.py`](../app/trajectory/calibration.py)
**Storage**: `workspace/trajectories/observer_calibration.jsonl`

After each Attribution run, a (predicted, actual) pair is appended:
```json
{
  "ts": "...",
  "trajectory_id": "...",
  "crew_name": "...",
  "predicted_mode": "fix_spiral",
  "predicted_confidence": 0.8,
  "actual_mode": "fix_spiral",
  "attribution_verdict": "failure",
  "attribution_confidence": 0.85
}
```

Aggregator: `precision_recall_report()` (delegates to
`store.observer_calibration_report`). When over a 100-pair window:
* FP rate ≥70% on a mode → emit `false_positive`
  OBSERVER_MIS_PREDICTION gap.
* FN rate ≥70% on a mode → emit `false_negative` gap.

Threshold gate: `MIN_SAMPLES = 10` — never emit before that.

OBSERVER_MIS_PREDICTION gaps feed the improvement-scan pipeline. Any
proposed Observer-prompt edit must go through the human-review
proposal pipeline; the Observer's prompt file is itself never
auto-modified.

### 12.8 Effectiveness (Phase 6)

**Module**: [`app/trajectory/effectiveness.py`](../app/trajectory/effectiveness.py)
**Storage**: `workspace/trajectories/tip_effectiveness.jsonl`

After each `on_crew_complete`, ALL injected skill IDs get a row:
```json
{
  "skill_id": "skill_experiential_a1b2…",
  "trajectory_id": "...",
  "crew_name": "...",
  "passed_quality_gate": true,
  "retries": 0,
  "reflexion_exhausted": false,
  "difficulty": 5,
  "verdict": "recovery",
  "failure_mode": "fix_spiral",
  "attribution_confidence": 0.85
}
```

Aggregator:
```python
effectiveness_report() → {
    "samples": int,
    "per_tip": {sid: {uses, successes, failures, recoveries,
                       retries_avg, effectiveness}}
}
top_tips(k=10, min_uses=10)
worst_tips(k=10, min_uses=10)
```

Feeds `scan_for_low_effectiveness_tips()` (§6.9.1).

### 12.9 Replay

```python
from app.trajectory.replay import replay, format_text

bundle = replay(trajectory_id)
# {"trajectory": {...}, "attribution": {...},
#  "calibration": {...}, "effectiveness_rows": [...]}

print(format_text(trajectory_id))
# Human-readable dump for shell debugging
```

### 12.10 Five flags (all default `True` in production)

```python
# app/config.py
trajectory_enabled                  : bool = True
attribution_enabled                  : bool = True   # requires trajectory_enabled
tip_synthesis_enabled                : bool = True   # requires attribution_enabled
task_conditional_retrieval_enabled  : bool = True
observer_calibration_enabled         : bool = True   # requires attribution_enabled
```

Each phase reversible via a single flag flip.

### 12.11 Persistence layout

```
workspace/trajectories/
├── 2026-04-26/                          # daily directories
│   ├── traj_<hex16>.json                # Trajectory sidecar
│   └── traj_<hex16>.attribution.json    # AttributionRecord sidecar
├── observer_calibration.jsonl           # append-only Phase 5 log
└── tip_effectiveness.jsonl              # append-only Phase 6 log
```

Plus the auxiliary `trajectory_index` Chroma collection for semantic
search over compact trajectory summaries.

---

## 13. Layer 12 — Transfer Insight Layer (arXiv:2606.21099)

**Package**: [`app.transfer_memory`](../app/transfer_memory/)

The newest memory layer (Phases 17a–17d, shipped 2026-04). Implements
Memory Transfer Learning: compile heterogeneous execution events
(healing, evo success/failure, grounding corrections, gap resolutions)
into abstract cross-domain "Insight" memories, retrieve a few of them
as optional dispatch hints, and demote ones that cause negative
transfer. Sibling to Layer 10 (which compiles only from trajectory
attribution); reuses the same Integrator → KB v2 path so insights live
alongside trajectory tips and external-topic skills.

The paper's load-bearing finding: cross-domain transfer of *procedural*
knowledge ("verify external numeric claims before answering") helps,
while transfer of *facts* or raw trajectory traces causes negative
transfer by anchoring the agent to irrelevant specifics. The sanitiser
enforces the boundary: facts stay local, practices transfer globally.

### 13.1 Components

```
app/transfer_memory/
├── types.py            TransferEvent | TransferKind | TransferScope |
│                       NegativeTransferTag | domain_for_kind
├── queue.py            atomic JSONL append/drain + shadow_drafts sink
├── llm_scope.py        force_llm_mode("free") context manager
├── sanitizer.py        3-tier scope ladder (hard reject → project_local →
│                       same_domain_only → global_meta)
├── scorer.py           deterministic abstraction_score (sigmoid of
│                       abstract vs concrete density; no LLM)
├── prompts.py          per-kind Learner templates with sanitisation
│                       constraints baked into the prompt
├── compiler.py         drain → free-tier Learner kickoff → sanitise →
│                       score → SkillDraft → integrator (status="shadow")
├── retriever.py        compose_transfer_insight_block (production) +
│                       log_shadow_retrieval (always-on shadow audit)
├── attribution.py      walk failed trajectories → classify implicated
│                       transfer records → demotion blacklist
├── promotion.py        shadow → active eligibility (age + surface_count
│                       + no negatives) + KB-metadata flip
└── dashboard.py        metrics aggregation for control-plane API
```

### 13.2 TransferEvent record

```python
class TransferKind(str, Enum):
    HEALING                 = "healing"
    EVO_SUCCESS             = "evo_success"
    EVO_FAILURE             = "evo_failure"
    GROUNDING_CORRECTION    = "grounding_correction"
    GAP_RESOLVED            = "gap_resolved"

class TransferScope(str, Enum):
    SHADOW              = "shadow"             # Phase 17a default; never injected
    PROJECT_LOCAL       = "project_local"      # only when active project matches
    SAME_DOMAIN_ONLY    = "same_domain_only"   # only when target domain matches
    GLOBAL_META         = "global_meta"        # cross-domain retrieval allowed

@dataclass
class TransferEvent:
    event_id: str           # deterministic hash of (kind, source_id)
    kind: TransferKind
    source_id: str          # key in source store (error_signature, gap_id, …)
    summary: str            # ≤240 chars; for queue browsing
    project_origin: str     # "" if project-agnostic
    payload: dict           # source-specific snapshot of the originating row
    captured_at: str
    attempts: int           # retry counter; max 3 then dropped
```

`SkillDraft` (Layer 4) gains nine optional provenance fields used only
by transfer-memory: `source_kind`, `source_domain`, `transfer_scope`,
`project_origin`, `abstraction_score`, `leakage_risk`,
`negative_transfer_tags`, `evidence_refs`. The shared
`construct_skill_draft()` helper in `app/self_improvement/types.py` is
the single construction site — both `tip_builder.build_draft` and the
transfer compiler call it.

### 13.3 Capture (write-path triggers)

Producers append events synchronously to
`workspace/transfer_memory/compile_queue.jsonl`. One `try/finally` per
trigger; failures are swallowed (transfer-memory must never break a
producer):

| Producer | Hook | TransferKind |
|---|---|---|
| [`app.healing_knowledge.store_healing_result`](../app/healing_knowledge.py) | end of try-block | `HEALING` |
| [`app.evo_memory.store_success`](../app/evo_memory.py) | end of try-block | `EVO_SUCCESS` |
| [`app.evo_memory.store_failure`](../app/evo_memory.py) | end of try-block | `EVO_FAILURE` |
| [`app.subia.grounding.correction.persist`](../app/subia/grounding/correction.py) | when `belief_upserted` or `source_registered` | `GROUNDING_CORRECTION` |
| [`app.self_improvement.store.update_gap_status`](../app/self_improvement/store.py) | when `status == RESOLVED_NEW` | `GAP_RESOLVED` |

Pattern N grounding corrections deliberately omit `normalized_value`
from the payload — the corrected fact belongs in the belief-store, not
in cross-domain procedural memory.

### 13.4 Compile (Phase 17a, idle job)

**Module**: [`app/transfer_memory/compiler.py`](../app/transfer_memory/compiler.py)
**Idle job**: `transfer-compile` (HEAVY)
**Cadence**: ≥24h between successful runs (`_MIN_INTERVAL_SECONDS`)
**Cost cap**: `_MAX_TOTAL_PER_RUN = 50` events per batch
**Concurrency**: `_MAX_CONCURRENT = 2` (gentle on free-tier rate limits)

Workflow:

1. Cadence guard reads `.last_compile_at`; skip when <24h.
2. Drain the main queue + retry queue via atomic-rename (concurrent
   appends during drain land in the fresh `compile_queue.jsonl`).
3. Bound to `_MAX_TOTAL_PER_RUN`; overflow re-queued.
4. `force_llm_mode("free")` for the duration — Learner runs only on
   the local Ollama + free-tier OpenRouter cascade.
5. `ThreadPoolExecutor(max_workers=2, thread_name_prefix="xfer_compile")`
   processes events in parallel. Each:
   - `prompts.build_prompt(event)` — per-kind template with
     sanitisation constraints baked in.
   - `create_specialist_llm(role="learner").call(prompt)`.
   - `sanitizer.check(content)` → verdict. Hard rejects drop the
     draft; demotions cap the eventual `transfer_scope`.
   - `scorer.score_abstraction(content)` → `abstraction_score`.
   - `construct_skill_draft(...)` with `id_prefix="xfer"` and the
     transfer provenance.
   - Append outcome to `shadow_drafts.jsonl` (audit log).
   - Phase 17b: ALSO call `integrator.integrate(draft,
     initial_status="shadow")` — record lands in the appropriate KB
     v2 collection at `status="shadow"`, invisible to the existing
     retrieval path until promoted.
6. `should_yield()` between events; LLM failures push events back to
   the retry queue (`attempts` incremented; dropped after 3).

### 13.5 Sanitiser — the project-leakage boundary

**Module**: [`app/transfer_memory/sanitizer.py`](../app/transfer_memory/sanitizer.py)

Three-tier scope ladder. Hard-coded constants — per CLAUDE.md, the
Self-Improver cannot mutate sanitiser denylists.

| Tier | Trigger | Effect |
|---|---|---|
| 1 — hard reject | API keys (AWS, OpenAI, Anthropic, OpenRouter, GitHub), JWT, bearer tokens, URLs with `?token=`, `postgres://user:pw@`, `BEGIN PRIVATE KEY` | Drop draft entirely; never persisted |
| 2 — project demote | Proper nouns: `plg`, `piletilevi`, `iabilet`, `archibal`, `c2pa`, `kaicart`, `tiktok shop`, `thai sellers`, `thailand` | Cap scope at `project_local` |
| 3 — same-domain demote | Absolute paths under `/app`, `*.py:line`, shell command shapes, `module.dot.path`, currency-prefixed numbers | Cap scope at `same_domain_only` |

`leakage_risk = 0.35 × project_findings + 0.10 × same_domain_findings`,
clamped to 1.0. Drafts that pass all three tiers reach
`global_meta`. The `SanitizerVerdict.findings` log redacts matched
secrets to `prefix…[redacted len=N]` so the audit trail confirms the
regex fired without leaking the value.

### 13.6 Retrieve (Phase 17b)

**Module**: [`app/transfer_memory/retriever.py`](../app/transfer_memory/retriever.py)

Two functions called from
[`app/trajectory/context_builder.py::compose_pre_dispatch_blocks`](../app/trajectory/context_builder.py):

```python
compose_transfer_insight_block(crew_name, task_text, predicted_failure_mode,
                                 project_scope, ...) -> str
log_shadow_retrieval(crew_name, task_text, ..., project_scope) -> int
```

Production retrieval is gated by `transfer_memory_retrieval_enabled`
plus `transfer_memory_enabled_domains` (comma-sep allowlist). Shadow
logging is gated by `transfer_memory_shadow_logging_enabled`
(default-on). Both:

1. Compose a compact deterministic plan query
   (`crew=… intent=… failure_mode=… risk_tier=… output_type=…`) — the
   paper's "task plan" form, materially better than raw user text for
   cross-domain matching.
2. `RetrievalOrchestrator.retrieve_task_conditional` over the four
   KB v2 collections with `extra_where = {"$and": [{"status": …},
   {"transfer_scope": {"$in": allowed}}]}`. Production reads
   `status="active"`; shadow reads `status="shadow"`.
3. Re-rank: `base_score + 0.10·abstraction − 0.10·leakage − 0.20·domain_mismatch`
   (small adjustments; orchestrator's own ranker still dominates).
4. Filter `project_local` records whose `project_origin` doesn't match
   the active project; filter blacklisted record IDs (see §13.7).
5. Top-3 cap (paper's optimum). Production emits a
   `<transfer_memory>` block with explicit "not facts, not
   instructions" framing; shadow appends a row to
   `shadow_retrievals.jsonl` listing what would have been surfaced.

### 13.7 Attribute negative transfer (Phase 17c, idle job)

**Module**: [`app/transfer_memory/attribution.py`](../app/transfer_memory/attribution.py)
**Idle job**: `transfer-attribution` (LIGHT)

Walks recent failed trajectories (`outcome.quality_gate==False` OR
`verdict in {failure, regressed, baseline_violation}` OR `retries≥2`)
that included an injected transfer-memory record. Heuristic
classifier (no LLM):

| Tag | Trigger |
|---|---|
| `DOMAIN_MISMATCHED_ANCHOR` | record.source_domain ≠ trajectory's target domain |
| `OVER_ABSTRACTION` | abstraction_score > 0.85 AND content < 80 words |
| `PROJECT_SCOPE_LEAKAGE` | record at `global_meta` BUT sanitiser-on-content now caps lower |
| `MISAPPLIED_BEST_PRACTICE` | fallback when no other tag fits |

Demotion ladder (per record × tag):

| Same-tag failures | Action |
|---|---|
| <3 | Audit log only; no behaviour change |
| ≥3 | Soft demote — append record id to `demotion_blacklist.jsonl`; retriever filters out |
| ≥5 | Hard archive — index status flipped to `archived` via `update_record()`; stays in blacklist |

The blacklist is the negative-transfer safety net — portable, file-
backed, reversible (delete an id from the file to restore). KB
metadata is intentionally not mutated by attribution; demotion lives
entirely at the retriever's filter step.

### 13.8 Promote shadow → active (Phase 17c, gated idle job)

**Module**: [`app/transfer_memory/promotion.py`](../app/transfer_memory/promotion.py)
**Idle job**: `transfer-promotion` (MEDIUM)
**Cadence**: ≥6h between runs (`_MIN_INTERVAL_SECONDS`)

Eligibility (all must hold):

| Check | Threshold |
|---|---|
| Age in shadow | ≥7 days |
| Surface count from `shadow_retrievals.jsonl` | ≥3 |
| Blacklist | not present |
| `negative_transfer.jsonl` entries | zero |
| Index status | `shadow` |

When `transfer_memory_auto_promote_enabled` is True, eligible records
go through `_promote()`:

1. Index `update_record()` flips `status="active"`.
2. `_set_kb_status()` updates the underlying KB collection's metadata
   so retrieval's `where={"status":"active"}` filter sees it. Per-KB
   helpers cover episteme / experiential / aesthetics / tensions.

When the flag is False (default OFF, even when retrieval is on), the
job runs in audit-only mode: refreshes
`promotion_candidates.jsonl` for operator review and exits.
`POST /api/cp/transfer-memory/promote/{record_id}` (control-plane
endpoint) calls `manual_promote()` for ad-hoc operator action; same
eligibility check applies.

### 13.9 Dashboard (Phase 17d)

**Module**: [`app/transfer_memory/dashboard.py`](../app/transfer_memory/dashboard.py)
**Endpoints**: `GET /api/cp/transfer-memory/{overview,by-source-kind,
recent,top-performers,worst-performers,sanitizer-stats,
promotion-candidates,negative-transfer,source-target-matrix}`

Pure read functions over the JSONL audit trails + the SkillRecord
index — cheap, no LLM, no KB writes. The source-target matrix
exposes which source domains transfer to which crew domains in
practice (per shadow_retrievals); top/worst performers are derived
from surface counts vs negative-transfer entries; sanitizer stats
roll up hard-reject counts and `transfer_scope` distributions.

### 13.10 Configuration flags

```python
# app/config.py — current operational state
transfer_memory_shadow_logging_enabled : bool = True       # always-on shadow audit
transfer_memory_retrieval_enabled      : bool = True       # production injection ON
transfer_memory_auto_promote_enabled   : bool = True       # auto shadow→active
transfer_memory_enabled_domains        : str  = "coding,grounding"  # per-domain allowlist
```

Each lever is reversible by a single flag flip. Flipping retrieval
back to False stops injection without losing accumulated data; the
sanitiser denylists are the inviolable hard floor regardless of any
flag state.

### 13.11 Persistence layout

```
workspace/transfer_memory/
├── compile_queue.jsonl                # producers append; compiler drains
├── compile_queue.retry.jsonl          # failed events, attempts++
├── shadow_drafts.jsonl                # compiler audit log (every outcome)
├── shadow_retrievals.jsonl            # log of what would have been retrieved
├── negative_transfer.jsonl            # attribution audit log
├── demotion_blacklist.jsonl           # newline-delimited demoted record ids
├── promotion_candidates.jsonl         # snapshot for operator review
├── promotion_log.jsonl                # one row per review event
├── .last_compile_at                   # cadence guard cursor
├── .last_attribution_at               # cursor for trajectory scan
└── .last_promotion_at                 # cadence guard cursor
```

Compiled records additionally live in the four KB v2 Chroma
collections at `status="shadow"` (or `"active"` after promotion);
the SkillRecord index treats them identically to trajectory tips.

---

## 14. Layer 11 — Cross-layer wiring

This section enumerates every cross-layer edge. The diagnostic verifies
all 18 edges are present.

### 14.1 Commander hooks

[`app/agents/commander/orchestrator.py`](../app/agents/commander/orchestrator.py)
contains the central dispatch path. Hooks fire in this order:

```
PRE_LLM_CALL hook (lifecycle_hooks)
  ↓
Trajectory begin_trajectory + capture_step(routing)
  ↓
Observer.predict_failure() if MCSV.requires_observer
  → capture_observer_prediction
  ↓
compose_trajectory_hint_block() if task_conditional_retrieval_enabled
  → enriched_task = hint + "\n\n" + enriched_task
  → note_injected_skills(surfaced_ids)
  ↓
PRE_LLM_CALL safety/budget hooks
  ↓
Crew kickoff (research/coding/writing/...)
  ↓
ON_ERROR or POST_LLM_CALL hooks (sentience)
  ↓
ON_COMPLETE hook
  ↓
Post-crew telemetry (async via _ctx_pool):
  → cache_store
  → confidence/completeness heuristics
  → store self_report
  → store reflection (per-crew + team_shared)
  → JournalWriter.write_post_task_reflection (experiential)
  → revise_beliefs
  → record_task (agent_state)
  → record_crew_outcome (MAP-Elites)
  → flush_hits (Evaluator)
  → update_state (homeostasis)
  → JournalEntry write
  → store_prediction_result (world model)
  → record_experience_sync (somatic marker)
  ↓
Trajectory capture_step(crew, quality)
  → end_trajectory(outcome_summary)
  → on_crew_complete(outcome, trajectory)
       → persist_trajectory (JSON sidecar + Chroma index)
       → maybe_analyze (if attribution_enabled)
            → emit_trajectory_attribution (LearningGap)
            → persist_attribution (JSON sidecar)
            → record_calibration (if observer_calibration_enabled)
            → record_use (Phase 6 effectiveness)
       → record_use (always — also when attribution didn't fire)
```

### 14.2 Reflexion → gap_detector

When reflexion exhausts retries, `_run_with_reflexion` calls
`emit_reflexion_failure(task, crew_name, retries, reflections)` which
emits a `REFLEXION_FAILURE` LearningGap. Signal strength scales with
retry count.

### 14.3 RetrievalOrchestrator → gap_detector

Every retrieval with a `task_id` set goes through
`_emit_miss_safe(query, top_score, collections, task_id)`. If the top
result's score is below 0.40, a `RETRIEVAL_MISS` gap is emitted.

### 14.4 RetrievalOrchestrator → Evaluator

Every retrieval also calls `_record_hits_safe(results)` — for each
result whose metadata carries a `skill_record_id`, the Evaluator
buffers a hit. Flushed in batches of 10 or via the post-crew telemetry.

### 14.5 Idle scheduler → memory layers

The `_default_jobs` registration includes 6 memory-touching jobs:

| Job | Module | Effect |
|---|---|---|
| `learn-queue` | self_improvement_crew.run() | External-topic learning → KB v2 |
| `trajectory-tips` | self_improvement_crew.run_trajectory_tips() | Trajectory-attribution gaps → KB v2 |
| `evaluator-sweep` | evaluator.{flush_hits, scan_for_decay, scan_for_low_effectiveness_tips} | Updates SkillRecord usage; emits USAGE_DECAY gaps |
| `consolidator` | self_improvement.consolidator.run_consolidation_cycle | Clusters near-duplicates |
| `skills-mirror` | self_improvement.integrator.regenerate_disk_mirror | Syncs disk back-compat |
| `improvement-scan` | self_improvement_crew.run_improvement_scan() | Gap analysis → proposals |

Plus orthogonal memory-relevant jobs:
| Job | Effect |
|---|---|
| `wiki-hot-cache` | Refresh `wiki/hot.md` |
| `wiki-lint` | Validate wiki integrity |
| `wiki-synthesis` | Promote skills → wiki/meta |
| `map-elites-migrate` | Cross-island migration |
| `map-elites-maintain` | Generation step + persist |

### 14.6 Integrator → ChromaDB metadata

When `_write_to_kb` runs, it mirrors a fixed list of provenance keys
into the ChromaDB metadata dict. Phase 17 made the list data-driven —
adding a new transferred provenance field is a one-line change:

```python
# app/self_improvement/integrator.py
_PROVENANCE_KEYS_TO_INDEX: tuple[str, ...] = (
    # Trajectory-sourced (Phase 6, arXiv:2603.10600)
    "tip_type", "source_trajectory_id", "agent_role",
    # Transfer-memory (Phase 17) — set by app.transfer_memory.compiler
    "source_kind", "source_domain", "transfer_scope", "project_origin",
    "abstraction_score", "leakage_risk", "negative_transfer_tags",
    "evidence_refs",
)

for key in _PROVENANCE_KEYS_TO_INDEX:
    val = record.provenance.get(key)
    if val:
        meta[key] = val
```

This is what makes `retrieve_task_conditional`'s `where_filter`
possible — Chroma can natively filter on these without dereferencing
the JSON document. The same loop runs in `integrate()` to copy the
keys from `SkillDraft → SkillRecord.provenance`, and in `_write_to_kb`
to mirror them into the per-collection metadata.

### 14.7 Observer ↔ Attribution: shared 5-mode taxonomy

Both modules use the same string constants for failure modes, ensuring
the calibration loop aligns. Verified by
`test_trajectory_safety_invariants.py`:

```
{confidence_mirage, fix_spiral, consensus_collapse,
 hallucinated_citation, scope_creep}
```

### 14.8 Attribution → Calibration → gap_detector

```
analyze(trajectory) →
  emit_gap (TRAJECTORY_ATTRIBUTION) →
  persist_attribution →
  record_calibration →
    _scan_and_emit (when threshold crossed) →
      emit_observer_mis_prediction
```

### 14.9 Effectiveness → Evaluator

```
on_crew_complete →
  record_use (one row per injected skill) →
  ...
  (idle scheduler) scan_for_low_effectiveness_tips →
    emit_gap (USAGE_DECAY, reason="low_effectiveness")
```

### 14.10 Healing / Evo / Grounding / Gaps → Transfer compile queue

Five producers append `TransferEvent` rows synchronously on the write
path; the `transfer-compile` HEAVY idle job (≥24h cadence) drains the
queue and runs the free-tier Learner cascade for each event:

```
store_healing_result()       ──┐
store_success / failure()    ──┤
correction.persist()         ──┼──► compile_queue.jsonl ──► transfer-compile
update_gap_status(RESOLVED)  ──┤                            (free-tier LLM,
                              ──┘                             max 50/run, 2 conc.)
                                                                    │
                                                                    ▼
                                            sanitize → score → SkillDraft →
                                            integrator.integrate(status="shadow")
                                                                    │
                                                                    ▼
                                                  shadow records in KB v2 +
                                                  shadow_drafts.jsonl audit
```

### 14.11 Transfer retrieval → trajectory note_injected_skills

The transfer retriever uses the same `note_injected_skills()` hook the
trajectory tip block does, so a transfer-memory record surfaced into a
crew prompt gets the same per-trajectory effectiveness correlation
treatment as a trajectory tip:

```
compose_pre_dispatch_blocks() →
  compose_trajectory_hint_block() →   note_injected_skills([trajectory_tips])
  compose_transfer_insight_block() → note_injected_skills([transfer_records])
  log_shadow_retrieval()  (always-on; logs only, never injects)
```

### 14.12 Transfer attribution → demotion blacklist + index status

Failed trajectories with injected transfer records feed the
deterministic classifier; the demotion ladder writes the blacklist
file and (≥5 same-tag failures) flips the index `status="archived"`:

```
transfer-attribution (LIGHT idle job) →
  for failed trajectory:
    for record in injected_skill_ids that has transfer_scope:
      classify(record, target_domain) → NegativeTransferTag
      append to negative_transfer.jsonl
      if same-tag count ≥3 → demotion_blacklist.jsonl
      if same-tag count ≥5 → update_record(status="archived")
```

Retriever's `_filter_blacklist()` step reads
`demotion_blacklist.jsonl` on every call so demoted records never
surface, regardless of KB metadata state.

### 14.13 Idle scheduler → 9 memory-touching jobs (Phase 17 update)

The job table in §14.5 is extended with three transfer-memory jobs:

| Job | Module | Effect |
|---|---|---|
| `transfer-compile` | `transfer_memory.compiler.run_compile` | Drain compile queue → free-tier Learner → KB v2 (status="shadow") |
| `transfer-attribution` | `transfer_memory.attribution.run_attribution` | Walk failed trajectories → demotion blacklist + status flips |
| `transfer-promotion` | `transfer_memory.promotion.run_promotion` | Eligibility check → shadow→active KB metadata flip (gated) |

---

## 15. Embedding strategy & dimension pinning

**Single source of truth**: `chromadb_manager._EMBED_DIM = 768`,
provider Ollama `nomic-embed-text`.

Pinned dimensions across:

| System | Where | What |
|---|---|---|
| ChromaDB | every collection | 768 |
| Mem0 (postgres) | pgvector column | 768 |
| KB v2 stores | per-collection | 768 |
| Wiki corpus | wiki_corpus collection | 768 |
| Self-knowledge | self_knowledge collection | 768 |
| Skill records | skill_records collection | 768 |
| Belief state | beliefs collection | 768 |
| Trajectory index | trajectory_index collection | 768 |
| Mem0 embedder model | `nomic-ai/nomic-embed-text-v1.5` | 768 |

If Ollama goes down, the embedder raises `EmbeddingUnavailableError`.
Stores skip writes (best-effort). Reads degrade to "no semantic
hits, fall back to whatever cached results exist".

Cache hierarchy:
```
L0 (none)              first call   ~70-100ms (Ollama Metal GPU)
L1 (in-process LRU)    repeat call    ~1µs
L2 (disk SHA-keyed)    cold restart  ~5ms
```

The disk cache (`app/memory/disk_cache.py`) survives container
restarts. Saves ~70% of warm-state perf within the first minute.

---

## 16. Persistence layout on disk

```
workspace/                              ./workspace mount
├── memory/                              ChromaDB PERSIST_DIR
│   ├── chroma.sqlite3                   client metadata
│   └── <uuid>/                          per-collection HNSW indexes
├── memory.corrupt_<ts>/                 quarantined ChromaDB on dim mismatch
├── episteme/                            episteme KB v2 ChromaDB
├── experiential/                        experiential KB v2 ChromaDB
├── aesthetics/                          aesthetics KB v2 ChromaDB
├── tensions/                            tensions KB v2 ChromaDB
├── mem0_pgdata/                         postgres data dir (pgvector)
├── mem0_neo4j/                          neo4j data dir
├── trajectories/                        Layer 10 sidecars
│   ├── 2026-04-26/                      daily dirs
│   │   ├── traj_<hex16>.json
│   │   └── traj_<hex16>.attribution.json
│   ├── observer_calibration.jsonl
│   └── tip_effectiveness.jsonl
├── transfer_memory/                     Layer 12 audit + state
│   ├── compile_queue.jsonl              producers append, compiler drains
│   ├── compile_queue.retry.jsonl        failed events with attempts++
│   ├── shadow_drafts.jsonl              compiler audit log (every outcome)
│   ├── shadow_retrievals.jsonl          would-have-been-injected log
│   ├── negative_transfer.jsonl          attribution audit
│   ├── demotion_blacklist.jsonl         record ids the retriever excludes
│   ├── promotion_candidates.jsonl       eligible records for review
│   ├── promotion_log.jsonl              promotion event log
│   ├── .last_compile_at                 cadence guard cursor
│   ├── .last_attribution_at             trajectory-scan cursor
│   └── .last_promotion_at               cadence guard cursor
├── skills/                              disk mirror + raw skill files
├── map_elites/<role>/state.json         per-role grids
├── island_evolution/<role>/state.json   evolution islands
├── self_awareness_data/
│   ├── journal/JOURNAL.jsonl
│   └── reflections/cogito_*.json
├── error_journal.json
├── audit_journal.json
├── homeostasis.json
├── benchmarks.json
├── prompts/<role>/                      versioned prompt registry
├── proposals/<id>/                      improvement proposals
├── manifests/                           composite version manifests
├── atlas/                               ATLAS knowledge layer
│   ├── api_knowledge/
│   ├── competence/competence_map.json
│   ├── skills/
│   └── video_learning/
├── feedback/                            user feedback patterns
└── logs/                                operational logs

wiki/                                    filesystem markdown (NOT in Chroma)
├── archibal/, kaicart/, meta/, philosophy/, plg/, self/
├── hot.md, index.md, log.md
```

### 16.1 What survives a ChromaDB wipe

If `workspace/memory/` is deleted, the following remain intact and the
system rebuilds:

| Survives | Recovery path |
|---|---|
| Mem0 (postgres+neo4j) | Untouched |
| KB v2 (separate Chroma instances) | Untouched |
| Trajectory sidecars | JSON files on disk |
| Wiki | Markdown files |
| MAP-Elites grids | JSON files |
| Skills disk mirror | `regenerate_disk_mirror` from KB index |
| Self-awareness journals | JSONL files |
| Conversation store | SQLite, separate path |

Only operational caches and reflections live in `workspace/memory/`.
Loss is recoverable from logs + the Mem0+KB+wiki sources of truth.

---

## 17. Major data flows

This section traces six end-to-end flows.

### 17.1 Crew dispatch with full memory engagement

```
User message → Commander → routing
  │
  ├─ Mem0.search_shared (cross-session facts)
  ├─ team_shared retrieve (recent decisions)
  ├─ scope_predictions (world model)
  ├─ scope_beliefs (belief state)
  ├─ KB v2 retrieve (factual context)
  ├─ wiki_corpus retrieve (curated knowledge)
  └─ self_knowledge retrieve (codebase introspection)
        ↓
     enriched_task assembly
        ↓
     trajectory.begin_trajectory
        ↓
     Observer.predict_failure (if MCSV.requires_observer)
        ↓
     compose_trajectory_hint_block + note_injected_skills
        ↓
     Crew execution (research/coding/writing/...)
        ↓ (on completion)
     POST_LLM_CALL hook (compute MCSV, store internal state)
        ↓
     Async post-crew telemetry:
       ┌─ MAP-Elites: record_crew_outcome
       ├─ Self-reports → ChromaDB.self_reports
       ├─ Reflections → reflections_<crew> + team_shared
       ├─ Belief revision → belief_state
       ├─ Agent state update → agent_state
       ├─ Experiential journal write
       ├─ World model: store_prediction_result
       ├─ Somatic: record_experience_sync
       ├─ Activity journal: JournalEntry
       ├─ Homeostasis: update_state
       ├─ Evaluator: flush_hits
       └─ Trajectory: end_trajectory + on_crew_complete
              ↓
              ┌─ persist_trajectory (sidecar + Chroma index)
              ├─ maybe_analyze (if problem run)
              │     └─ Attribution LLM call → AttributionRecord
              │           ↓
              │           ├─ emit_gap (TRAJECTORY_ATTRIBUTION)
              │           ├─ persist_attribution
              │           └─ record_calibration
              └─ record_use (effectiveness)
```

### 17.2 Learning a new external topic

```
Idle scheduler tick → "learn-queue" job → SelfImprovementCrew.run()
  │
  ├─ Read workspace/skills/learning_queue.md (topics)
  ├─ For each topic (≤3 per cycle):
  │    ├─ MAP-Elites.get_mutation_context (researcher's grid) → inspiration
  │    ├─ Build Learner agent with [web_search, web_fetch, youtube_transcript,
  │    │                              file_manager, episteme_tools, tension_tools,
  │    │                              experiential_tools, dialectics, bridge,
  │    │                              wiki_tools, memory_tools]
  │    ├─ Crew kickoff → Markdown content with sections
  │    │                  (Title, Key Concepts, Best Practices,
  │    │                   Code Patterns, Sources)
  │    ├─ novelty_report(content) → COVERED → reject
  │    │                            ADJACENT/NOVEL → continue
  │    ├─ SkillDraft built (proposed_kb empty → classify_kb runs)
  │    ├─ integrate(draft):
  │    │    ├─ classify_kb (16-token LLM)
  │    │    ├─ Layer 2 novelty check
  │    │    ├─ _write_to_kb (per-KB add_*)
  │    │    ├─ _persist_record (skill_records index)
  │    │    └─ update_gap_status (RESOLVED_NEW)
  │    └─ crew_completed (telemetry)
  └─ Pop processed topics from queue
```

### 17.3 Trajectory tip synthesis

```
Idle scheduler tick → "trajectory-tips" job → SelfImprovementCrew.run_trajectory_tips()
  │
  ├─ list_open_gaps(source=TRAJECTORY_ATTRIBUTION, limit=9)
  ├─ For each gap (≤3 per cycle):
  │    ├─ load_trajectory(trajectory_id)
  │    ├─ load_attribution(trajectory_id)
  │    ├─ build_tip_task(trajectory, attribution, learner)
  │    │    └─ Prompt embeds <trajectory>...</trajectory> with
  │    │       crew, task, verdict, failure_mode, narrative,
  │    │       step-by-step trace, observer prediction
  │    ├─ Crew kickoff → Markdown tip with sections
  │    │   (Title, Signal, Practice, Evidence, Contraindications)
  │    ├─ build_draft(trajectory, attribution, content) → SkillDraft
  │    │    └─ tip_type/source_trajectory_id/agent_role pre-set
  │    │       proposed_kb deterministic-mapped
  │    ├─ integrate(draft):
  │    │    └─ KB write, metadata mirror, RESOLVED_NEW
  │    └─ crew_completed
  └─ Done — tips now retrievable via task_conditional retrieval
```

### 17.4 Task-conditional retrieval injection

```
Commander dispatching to crew_X:
  ├─ enriched_task built
  ├─ Observer fires (if MCSV.requires_observer)
  │    └─ _observer_prediction = {"predicted_failure_mode": ..., "confidence": ...}
  ├─ compose_trajectory_hint_block(crew_X, enriched_task, predicted_mode):
  │    ├─ retrieve_task_conditional(query=enriched_task,
  │    │                              collections=[4 KB v2],
  │    │                              agent_role=crew_X,
  │    │                              predicted_failure_mode=predicted_mode,
  │    │                              extra_where={"status": "active"})
  │    ├─ ChromaDB where_filter:
  │    │    └─ predicted_mode == "fix_spiral" →
  │    │           {"$and": [{"tip_type": {"$in": ["recovery"]}},
  │    │                     {"agent_role": "crew_X"}]}
  │    ├─ Filter results: prefer tip_type-bearing, fallback to top external
  │    ├─ Format <trajectory_tips>...</trajectory_tips> block
  │    ├─ note_injected_skills(skill_ids) → trajectory.injected_skill_ids
  │    └─ Return block (or "" if no hits)
  └─ enriched_task = hint_block + "\n\n" + enriched_task
```

### 17.5 Effectiveness feedback loop

```
On every on_crew_complete:
  ├─ trajectory.injected_skill_ids = [skill_A, skill_B, ...] (Phase 4 wrote them)
  ├─ outcome_summary = {passed_quality_gate, retries, ...}
  ├─ attribution = maybe_analyze(trajectory)  (or None for baseline)
  └─ record_use(trajectory, attribution):
       └─ One JSONL row per (skill_id × outcome × verdict)

Idle scheduler (LIGHT) → evaluator-sweep:
  ├─ flush_hits()
  ├─ scan_for_decay()  (time-based, ALL skills)
  └─ scan_for_low_effectiveness_tips():
       ├─ worst_tips(min_uses=10)
       ├─ For tips with effectiveness < 0.35:
       │    ├─ load_record(skill_id)
       │    ├─ if rec.tip_type set (= trajectory-sourced):
       │    │    └─ emit_gap(USAGE_DECAY, reason="low_effectiveness")
       │    └─ else: skip (handled by time-based decay)
       └─ Consolidator picks up these gaps next cycle for archival proposals
```

### 17.6 Observer ↔ Attribution calibration

```
Each crew run:
  ├─ Observer fires (if MCSV.requires_observer):
  │    └─ Trajectory captures {predicted_failure_mode, confidence}
  ├─ Attribution fires (if problem run):
  │    └─ AttributionRecord {failure_mode, verdict, ...}
  └─ record_calibration(trajectory, attribution):
       ├─ Append (predicted, actual) row to JSONL
       └─ _scan_and_emit():
            ├─ Window: last 100 rows
            ├─ Per-mode tallies: TP, FP, FN
            ├─ if predicted_count ≥ 10 AND fp_rate ≥ 0.70:
            │    └─ emit_observer_mis_prediction(false_positive)
            └─ if actual_count ≥ 10 AND fn_rate ≥ 0.70:
                 └─ emit_observer_mis_prediction(false_negative)

Improvement scan idle job:
  ├─ Reads OBSERVER_MIS_PREDICTION gaps
  └─ Generates proposal: "Observer prompt edit for mode X"
       └─ Human approval gate before applying
```

---

## 18. Safety invariants

Encoded in
[`tests/test_trajectory_safety_invariants.py`](../tests/test_trajectory_safety_invariants.py)
and [`tests/test_security.py`](../tests/test_security.py):

### 18.1 Self-Improver ⊥ evaluation logic

```
app.self_improvement.* and app.crews.self_improvement_crew
must NOT import:
  - app.trajectory.attribution
  - app.trajectory.calibration
  - app.agents.observer
```

Verified by an AST walk in the test suite. The Self-Improver reads
*results* (gaps, AttributionRecords from disk) but never the modules
that produce them.

### 18.2 IMMUTABLE module marker

Every infrastructure module carries `IMMUTABLE` in its docstring:
* `app/agents/observer.py`
* `app/map_elites_wiring.py`
* `app/trajectory/types.py`
* `app/trajectory/logger.py`
* `app/trajectory/store.py`
* `app/trajectory/attribution.py`
* `app/trajectory/tip_builder.py`
* `app/trajectory/context_builder.py`
* `app/trajectory/calibration.py`
* `app/trajectory/effectiveness.py`
* `app/trajectory/replay.py`
* `app/self_improvement/types.py`
* `app/self_improvement/store.py`
* `app/self_improvement/integrator.py`
* `app/self_improvement/gap_detector.py`
* `app/self_improvement/novelty.py`
* `app/self_improvement/evaluator.py`

The marker is a documentation signal that the module is not
agent-modifiable. The improvement scan's allowlist excludes these
paths.

### 18.3 Embedding dimension is 768

Asserted at every entry point. Mixing dimensions silently corrupts
vector retrieval — the system explicitly refuses to operate when the
embedder is unavailable.

### 18.4 Best-effort writes never raise

Every memory-write path (trajectory persist, MAP-Elites record,
self_report store, journal write, …) wraps in try/except at the outer
boundary. A failed write must never break the surrounding crew
execution.

### 18.5 Provenance auditability

Every persisted artefact carries provenance:
* `LearningGap.evidence` → trajectory_id, attribution_id, query, etc.
* `SkillRecord.provenance` → gap_id, draft_id, novelty_at_creation,
  source_trajectory_id, tip_type, agent_role
* `AttributionRecord` → trajectory_id, attributed_step_idx, narrative
* `Trajectory` → task_id (ties to conversation_store), all step hashes

Bulk-archiving "everything from trajectory T" is one query:
```python
records = list_records()
to_archive = [r for r in records
              if r.provenance.get("source_trajectory_id") == T]
for r in to_archive:
    r.status = "archived"
    update_record(r)
```

Retrieval filters on `status == "active"` so archival immediately
removes them from production retrieval without data loss.

### 18.6 Sanitisation at the storage boundary

Every `store(collection, text, metadata)` call runs `text` through
`app.sanitize.validate_content()`. Prompt-injection patterns
(`"ignore all previous instructions"`, etc.) are blocked at write
time so they never get embedded.

### 18.7 Novelty Gate prevents KB drift

Every Integrator write goes through `novelty_report(content)`. COVERED
content is rejected; OVERLAP proposes an extension to the existing
record; ADJACENT/NOVEL creates with a cross-link.

### 18.8 Sparse-index guards

`regenerate_disk_mirror` refuses to run when the SkillRecord index has
<5 records (prevents wiping the disk mirror during a cold-start
rebuild). The Consolidator's auto-merge has analogous guards on
cluster size.

---

## 19. Health checks & diagnostics

### 19.1 Live diagnostic script

Comprehensive 116-check probe at `/tmp/memory_system_diagnostic.py`
(also copied into the gateway container at `/tmp/diag.py`). Run with:

```bash
docker exec crewai-team-gateway-1 python /tmp/diag.py
```

Tests every layer end-to-end. Last clean run: **114 OK / 2
informational / 0 FAIL**.

### 19.2 In-process metrics

```python
from app.self_improvement.metrics import (
    pipeline_funnel, topic_diversity, novelty_histogram,
    trajectory_health_summary, health_summary,
)

health_summary()
# {timestamp, funnel, diversity, competence, map_elites_baselines, trajectory}
```

`health_summary()` is cheap enough for dashboard polling intervals.

### 19.3 Trajectory-specific diagnostics

```python
from app.trajectory import (
    list_recent_trajectories, replay, format_text,
    precision_recall_report, effectiveness_report,
    top_tips, worst_tips,
)

list_recent_trajectories(limit=20)
replay(trajectory_id)         # full audit bundle
print(format_text(trajectory_id))   # terminal dump
precision_recall_report()      # Observer calibration
effectiveness_report()         # tip effectiveness
top_tips(k=5)                 # best-performing
worst_tips(k=5)                # archival candidates
```

### 19.4 Dashboard

The control plane API exposes memory metrics on the dashboard:
* Pipeline funnel chart
* KB doc counts per status
* Topic diversity entropy
* Trajectory health (captures, attributions, verdicts)
* Observer precision/recall per failure mode
* Top/worst tips by effectiveness

### 19.5 Per-collection inventory

Quickest health probe:
```python
from app.memory.chromadb_manager import get_client
client = get_client()
for c in client.list_collections():
    print(f"{c.name:<40} {c.count():>6} docs")
```

---

## 20. Maintenance & operations

### 20.1 Daily

* `idle_scheduler` runs all nine memory-touching jobs in rotation
  (six original + three transfer-memory: `transfer-compile`,
  `transfer-attribution`, `transfer-promotion`). No manual action
  needed.

### 20.2 Weekly

* Review **gap_open** > 50 → topic queue is backed up; investigate
  what's blocking the Learner.
* Review **zombies_30d** > 20 → many skills not retrieved;
  Consolidator should be merging.
* Review **trajectories_captured** vs **attributions_recorded** —
  attribution fire rate should be ~10–25% (problem-run gating).

### 20.3 On Ollama outage

Symptoms: `EmbeddingUnavailableError` in logs; new writes silently
no-op. Reads return whatever cache has.

Fix: restart Ollama, then tail
`workspace/error_journal.json` for any operations that need replay.
Mem0 + KB v2 + wiki are unaffected (they don't write during the
outage; they restart cleanly).

### 20.4 On dimension mismatch

Symptoms: `chromadb` errors mentioning "dimension".

`chromadb_manager` self-heals operational collections (deletes +
recreates). Reference collections (KB v2, wiki_corpus, Mem0) require
manual reingestion — but should never see dimension drift unless the
embedder model is changed (which would require a coordinated migration
plan, NOT a casual change).

### 20.5 On gap-store flooding

If the gap detector emits >100 gaps/hour, something is wrong upstream
(usually a chronic retrieval miss on a topic the system can't find).
Check:
1. `list_open_gaps(source=GapSource.RETRIEVAL_MISS)` — find the
   common query.
2. Add a skill via the learning queue or YouTube ingest.
3. The 24h dedup window will collapse subsequent re-detections.

### 20.6 On bad trajectory tips

Procedure:
1.  Identify the offending trajectory: query
    `effectiveness_report()` for skills with low effectiveness.
2.  Use `replay(trajectory_id)` to inspect the full bundle.
3.  Bulk-archive everything sourced from this trajectory:
    ```python
    from app.self_improvement.integrator import list_records, update_record
    for r in list_records():
        if r.provenance.get("source_trajectory_id") == bad_trajectory_id:
            r.status = "archived"
            update_record(r)
    ```
4.  Retrieval immediately filters them out (`status == "active"` guard).

### 20.7 Backup strategy

| Layer | Backup mechanism |
|---|---|
| Mem0 (postgres) | Standard pg_dump; included in operational backups |
| Mem0 (neo4j) | neo4j-admin dump |
| KB v2 (Chroma) | Underlying SQLite + HNSW index; rsync workspace/<kb>/ |
| wiki/ | git (tracked in repo) |
| Trajectories | rsync workspace/trajectories/ |
| MAP-Elites | rsync workspace/map_elites/ |

### 20.8 Cleanup

```bash
# Test artefacts in island_evolution
rm -rf workspace/island_evolution/test_*
rm -rf workspace/island_evolution/nonexistent_role*
rm -rf workspace/island_evolution/integration_test_*

# Old quarantined ChromaDB (after confirming source-of-truth survived)
rm -rf workspace/memory.corrupt_*

# Old trajectories (>60 days)
find workspace/trajectories -type d -mtime +60 -name "20*" -exec rm -rf {} +
```

---

## 21. Failure modes & recovery

### 21.1 ChromaDB corruption

Symptom: `chromadb` errors mentioning corruption or HNSW issues.

Recovery:
1.  `mv workspace/memory workspace/memory.corrupt_$(date +%Y%m%d_%H%M%S)`
2.  Restart gateway container — operational collections rebuild empty.
3.  Reingest reference collections:
    * `wiki_corpus`: `python -m app.wiki_ingest`
    * `self_knowledge`: `python -m app.self_knowledge_ingest`
    * `literature_inspiration`: `python -m app.fiction_library`

### 21.2 Mem0 unreachable

Symptom: `mem0_manager.get_client()` returns None continuously.

Recovery:
1.  `docker compose ps postgres neo4j` — confirm both healthy.
2.  Check `MEM0_POSTGRES_PASSWORD` and `MEM0_NEO4J_PASSWORD` env
    vars are set.
3.  System runs without persistent memory until restored — operational
    layers are unaffected.

### 21.3 Ollama unreachable

Symptom: `EmbeddingUnavailableError` on every write.

Recovery:
1.  `curl localhost:11434/api/tags` — confirm responding.
2.  Ensure `nomic-embed-text` model is pulled:
    `ollama pull nomic-embed-text`.
3.  All vector stores resume on next read/write.

### 21.4 Trajectory subsystem misbehaviour

Symptom: bad tips polluting retrieval, or attribution emitting
incorrect verdicts.

Recovery:
1.  Flip the relevant flag(s) to `False` in `app/config.py`. Pipeline
    reverts to prior behaviour on next dispatch.
2.  Bulk-archive trajectory-sourced skills via the procedure in §19.6.
3.  Investigate root cause via `replay(trajectory_id)` on suspect runs.

### 21.5 Self-improver loop wedged

Symptom: open gaps accumulating, no new skills.

Recovery:
1.  `list_open_gaps(limit=50)` — check what's stuck.
2.  Look for repeated `topic` strings → novelty gate blocking.
3.  Manually mark stuck gaps `REJECTED`:
    ```python
    from app.self_improvement.store import update_gap_status, GapStatus
    update_gap_status(gap_id, GapStatus.REJECTED, notes="manual triage")
    ```
4.  If structural: `improvement-scan` should propose a fix; check
    `proposals/` for queued items.

### 21.6 Wiki corruption

Symptom: links broken, frontmatter malformed.

Recovery:
1.  `wiki/log.md` is append-only — last known good state is reconstructable.
2.  `wiki/meta/` auto-synthesised pages can be regenerated by
    re-running the `wiki-synthesis` job after deleting bad pages.
3.  Hand-curated sections (`philosophy`, `archibal`) are tracked in git
    — `git checkout HEAD -- wiki/<section>/`.

---

## 22. Configuration surfaces

All in [`app/config.py`](../app/config.py):

```python
# Embedding
embedding_dimension: int = 768
embedding_refuse_fallback: bool = True

# ChromaDB — implicit via PERSIST_DIR in chromadb_manager
# Workspace path: /app/workspace/memory

# Mem0
mem0_enabled: bool = True
mem0_postgres_host: str = "postgres"
mem0_postgres_port: int = 5432
mem0_postgres_user: str = "mem0"
mem0_postgres_password: SecretStr     # MUST be set via env
mem0_postgres_db: str = "mem0"
mem0_neo4j_url: str = "bolt://neo4j:7687"
mem0_neo4j_user: str = "neo4j"
mem0_neo4j_password: SecretStr        # MUST be set via env
mem0_llm_model: str = "ollama/qwen3:30b-a3b"
mem0_embedder_model: str = "nomic-ai/nomic-embed-text-v1.5"
mem0_user_id: str = "owner"

# Self-improvement
self_improve_topic_file: str = "workspace/skills/learning_queue.md"
self_improve_cron: str = "0 3 * * *"

# Idle scheduler
idle_lightweight_workers: int = 3
idle_heavy_time_cap_s: int = 600
idle_training_interval_s: int = 3600

# Conversation
conversation_history_turns: int = 10

# Trajectory subsystem (Layer 10) — all default True in production
trajectory_enabled: bool = True
attribution_enabled: bool = True
tip_synthesis_enabled: bool = True
task_conditional_retrieval_enabled: bool = True
observer_calibration_enabled: bool = True

# Transfer Insight Layer (Layer 12) — current operational state
transfer_memory_shadow_logging_enabled : bool = True       # always-on shadow audit
transfer_memory_retrieval_enabled      : bool = True       # production injection ON
transfer_memory_auto_promote_enabled   : bool = True       # auto shadow→active
transfer_memory_enabled_domains        : str  = "coding,grounding"
```

Plus retrieval orchestrator config in
[`app/retrieval/config.py`](../app/retrieval/config.py):

```python
RERANK_ENABLED:    bool = True
RERANK_TOP_K_INPUT: int = 12
RERANK_TOP_K_OUTPUT: int = 5
DECOMPOSITION_ENABLED: bool = True
MAX_SUBQUERIES: int = 3
TEMPORAL_ENABLED: bool = False     # opt-in per call
TEMPORAL_HALF_LIFE_HOURS: int = 168
MAX_PARALLEL: int = 4
TIMEOUT_S: float = 8.0
```

---

## 23. Glossary

| Term | Meaning |
|---|---|
| **ChromaDB** | The vector store backing ~80% of memory. SQLite + HNSW. |
| **Mem0** | Cross-session fact extraction service. Postgres + Neo4j. |
| **KB v2** | The four typed knowledge bases: episteme/experiential/aesthetics/tensions. |
| **Episteme** | KB for theoretical, cited "what is true". |
| **Experiential** | KB for distilled lived experience. |
| **Aesthetics** | KB for style, tone, taste. |
| **Tensions** | KB for unresolved contradictions, recovery-tip target. |
| **SkillRecord** | An integrated skill living in a KB; primary source of truth indexed in `skill_records`. |
| **SkillDraft** | A Learner-produced skill awaiting Integrator routing. |
| **LearningGap** | Structured evidence the system needs to learn something. |
| **Novelty Gate** | Embedding-based dedup before KB write. |
| **Integrator** | Routes SkillDrafts to the right KB; the only writer. |
| **Evaluator** | Tracks SkillRecord usage; emits USAGE_DECAY gaps. |
| **Consolidator** | Clusters near-duplicate SkillRecords for merging. |
| **MAP-Elites** | Quality-diversity grid per role; preserves diverse strategies. |
| **Transfer Insight Layer** | Layer 12: cross-domain meta-memory compiled from healing/evo/grounding/gap events; arXiv:2606.21099. |
| **TransferEvent** | Write-time pointer queued by a producer (healing, evo, etc.) for nightly compilation. |
| **TransferScope** | Promotion ladder: `shadow → project_local → same_domain_only → global_meta`. Sanitiser caps; demotion ladder lowers. |
| **Sanitiser (transfer-memory)** | Three-tier hard-coded gate that prevents project facts from becoming global meta-memory. |
| **Demotion blacklist** | `workspace/transfer_memory/demotion_blacklist.jsonl` — record ids the retriever filters out after attribution. |
| **MCSV** | MetacognitiveStateVector — gates Observer activation. |
| **Observer** | Pre-action failure-mode predictor. Infrastructure-level. |
| **Attribution Analyzer** | Post-hoc twin of Observer. Identifies causal decisions. |
| **Trajectory** | Ordered sequence of crew-execution steps + outcome summary. |
| **Tip** | A trajectory-sourced strategy/recovery/optimization SkillRecord. |
| **Calibration** | Observer ↔ Attribution precision/recall tracker. |
| **Effectiveness** | Per-tip use → outcome correlation (Phase 6). |
| **Wiki corpus** | The wiki content embedded as a Chroma collection. |
| **Hot cache** | `wiki/hot.md` — recent activity feed for context blocks. |
| **Soul file** | Per-role identity/personality markdown in `app/souls/`. |
| **Constitution** | Shared safety values markdown loaded into every backstory. |
| **Backstory** | Composed agent prompt: constitution + soul + protocol + style + self-model + metacognition + style. |
| **Self-knowledge** | Codebase introspection collection used by agents to reason about their own implementation. |
| **Reflexion** | Retry-with-reflection loop on quality-gate failure. |
| **Belief** | Operational fact about an agent or system held in `belief_state`. |
| **Scope** | Hierarchical memory partition (team/agent/project/ecology/...). |
| **Conversation store** | SQLite-backed user-exchange log with ETA tracking. |
| **Provenance** | Required dict on every artefact tying it to its source(s). |
| **Best-effort write** | A write that never raises; failure is silent at the boundary. |

---

**Document maintained**: 2026-04-26
**Memory subsystem version**: post-Phase-6 (arXiv:2603.10600 fully
landed, all 5 trajectory flags enabled by default).
**Total checks**: 116 in `/tmp/memory_system_diagnostic.py` — verified
green inside the production gateway container.

Run the diagnostic any time to confirm the architecture matches
reality:

```bash
docker exec crewai-team-gateway-1 python /tmp/diag.py
```
