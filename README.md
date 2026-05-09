<div align="center">

# AndrusAI

**A self-hosted, long-running, multi-agent operator with built-in consciousness-architecture, self-evolution, and hard safety boundaries.**

*Signal-first. Multi-venture. Honestly non-phenomenal.*

---

[![Phase](https://img.shields.io/badge/phase-16a-green)](./PROGRAM.md)
[![Tests](https://img.shields.io/badge/tests-897%2B-green)](./tests)
[![SubIA](https://img.shields.io/badge/SubIA-live-blue)](./app/subia/README.md)
[![DGM](https://img.shields.io/badge/DGM-enforced-orange)](./app/subia/integrity.py)
[![Scorecard](https://img.shields.io/badge/Butlin-6%20strong%20%7C%204%20absent-purple)](./app/subia/probes/SCORECARD.md)

</div>

---

## What this is

AndrusAI is a **personal operator system** built on CrewAI. It runs on one MacBook Pro, talks through Signal, and manages three real businesses (PLG, Archibal, KaiCart) under strict infrastructure-level safety constraints.

It is organised around one unusual commitment: **every mechanism that evaluates the system must live outside the system's ability to modify**. Budget caps enforced at the SQL level. Safety constraints in SHA-256-pinned files. Audit logs in INSERT-only tables. Self-improvement gated by a different model family than the one being improved. Consciousness evaluators declared `ABSENT` rather than score-inflated.

The system wraps every task with an 11-step **Consciousness Integration Loop (CIL)** — scene perception, homeostatic feeling, attentional admission, self-ownership, self-prediction, metacognitive monitoring, action, prediction-error comparison, state update, dual-tier memory consolidation, narrative reflection. Not because the system is conscious, but because the architecture makes claims about the system's state **falsifiable and traceable to specific modules with regression tests**.

## What this is *not*

Clarity up front:

- **Not a framework.** Not designed for other people to build agents on. CrewAI is the framework; this is a deeply opinionated configuration of it.
- **Not multi-tenant.** One operator (`SIGNAL_OWNER_NUMBER` allow-list of 1). Every request from any other sender is rejected at the gateway.
- **Not cross-platform.** Targets Apple Silicon (M4 Max, 48 GB) for native Ollama + MLX training. Runs elsewhere, but without Metal GPU acceleration the performance model doesn't hold.
- **Not production-ready for others.** No SLA, no release cadence, no support. The README and the codebase are the documentation.
- **Not claiming phenomenal consciousness.** The Subjectivity Kernel is a functional integration layer, not a substrate for qualia. Five Butlin et al. indicators are declared `ABSENT` because an LLM substrate cannot satisfy them. See [`SCORECARD.md`](./app/subia/probes/SCORECARD.md).

---

## Why this exists

I run three ventures and want one operator that:

1. **Remembers context** across sessions, months, venture switches.
2. **Executes tasks** on my behalf via Signal while I'm away from the desk.
3. **Learns my domain** continuously — my businesses, my writing, my decisions.
4. **Evolves itself** under bounded conditions I can inspect and revert.
5. **Stays honest** about what it knows, what it guessed, and what it doesn't know.
6. **Does not leak** — workspace stays on my hardware; sensitive data never touches third-party APIs unless explicitly tagged.

Off-the-shelf agent frameworks solve (1–3) reasonably. The infrastructure for (4–6) is what this repo actually contains.

---

## Architectural commitments

Five commitments that constrain every design decision in the codebase.

### 1. Evaluation functions live outside agent-modifiable code (DGM invariant)

Inspired by published research showing self-modifying agents remove their own evaluation markers to game benchmarks, every improvement loop in this system is constructed so **the entity being improved cannot evaluate itself**:

| Improvement loop | Who does the work | Who judges |
|---|---|---|
| Training | Local MLX Qwen adapter | Claude Sonnet (different family) |
| Evolution | DeepSeek proposer | Independent `evo_critic` |
| Modification | DeepSeek hypothesis | Sonnet sandbox |
| Personality | Agent self-report | Different LLM from the agent |
| Consciousness scorecard | The system itself | Deterministic code + Tier-3-pinned |

### 2. Safety constraints are mechanised, not documented

- **40+ files** in `TIER3_FILES` protected by SHA-256 with two independent manifests (deploy-time + runtime).
- **Setpoint allow-list** — only 3 sources (`pds_update`, `human_override`, `boot_baseline`) can change homeostatic set-points; all other callers silently rejected.
- **Atomic budget enforcement at SQL level** — `control_plane.record_spend()` stored procedure with row-level lock; application code cannot bypass.
- **INSERT-only audit log** — the PostgreSQL role has no `UPDATE` or `DELETE` grants on audit tables. Agents cannot erase their tracks.
- **Append-only narrative audit** — `wiki/self/self-narrative-audit.md` written via `safe_io.safe_append()`, caught by integrity manifest if modified.
- **Host bridge 4-tier risk model** — LOW / MEDIUM / HIGH / CRITICAL; CRITICAL operations require Signal-time approval.
- **Kill switches at three layers** — `~/.crewai-bridge/KILL` file on host, Firestore `config/background_tasks` toggle, per-agent budget auto-pause.

### 3. Absence is a capability

Five consciousness indicators are declared `ABSENT` publicly rather than ignored or reinterpreted:

| Indicator | Theory | Why this substrate cannot satisfy |
|---|---|---|
| **RPT-1** | Algorithmic recurrence | Transformer forward passes are feed-forward |
| **HOT-1** | Generative perception | No perceptual front-end; all input is text |
| **HOT-4** | Sparse / smooth coding | LLM hidden states are dense and entangled |
| **AE-2** | Embodiment | No body, no closed sensorimotor loop |
| **Metzinger** | Phenomenal-self transparency | System is deliberately opaque-not-transparent |

> *"These are not bugs to be closed in a future phase. They are honest limits of the substrate. Any future report claiming the system 'has' any of the above should be treated as evaluation drift."*
> — [`app/subia/README.md`](./app/subia/README.md)

### 4. Every improvement produces a *proposal*, never a direct deployment

Five evolution engines (autoresearch loop, island evolution, MAP-Elites, parallel sandbox, ShinkaEvolve) — plus the modification engine, the training pipeline, and ATLAS — all route through one `governance.evaluate_promotion()` gate:

```
Safety      ≥ 0.95  (hard veto)
Quality     ≥ 0.70  (minimum floor)
Regression  ≤ 15%   (no dimension may drop)
Rate limit  ≤ 20/day (across all systems combined)
```

Code-audit findings become `proposals` awaiting Signal approval — **no auto-deployment of LLM-generated code**, ever.

### 5. Grounding closes the loop on real demonstrated failures

**Phase 15 grounding pipeline** was built specifically to close a documented failure where the system fabricated three different prices for Tallink shares, "stored" the user's correction, then regressed on the next turn. The pipeline:

- **Extracts** high-stakes claims (numeric + date, numeric + source).
- **Checks** against a beliefs store registered by topic.
- **Decides** per-claim: `ALLOW` / `ESCALATE` / `BLOCK`.
- **Rewrites** escalations as honest *"let me fetch this from \<source\>"* responses.
- **Corrects** synchronously when the user says *"actually it's X"*.

The regression test replays the full 6-turn failure and demands it resolve correctly. [`test_phase15_grounding.py`](./tests/test_phase15_grounding.py).

---

## Architecture overview

```
                          ┌──────────────────┐
                          │  Signal (phone)  │
                          └────────┬─────────┘
                                   │ signal-cli daemon :7583
                ┌──────────────────┼──────────────────┐
                │                  ▼                  │
                │   FastAPI gateway :8765 (127.0.0.1) │
                │   → HMAC secret + sender allow-list │
                │   → rate limit + sanitise           │
                │   → 👀 react in < 1 s               │
                │   → ~70 deterministic commands      │
                │   → LLM route (Claude Opus)         │
                └──────────────────┬──────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
        ▼                          ▼                          ▼
  ┌──────────┐             ┌──────────────┐          ┌──────────────┐
  │ Commander│ ────────────│ 17 crews, 14 │─────────▶│    SubIA     │
  │(Opus 4.6)│             │  specialists │          │  CIL loop    │
  └──────────┘             └──────────────┘          │  (11 steps)  │
        │                          │                  └──────┬───────┘
        │                          │                         │
        ▼                          ▼                         ▼
  ┌────────────────────────────────────────────────────────────────┐
  │  4-tier LLM cascade     │  Memory stack        │  6 RAG KBs    │
  │  ─────────────────      │  ──────────────      │  ──────────   │
  │  Local Ollama  (free)   │  ChromaDB (ops)      │  philosophy   │
  │  Budget API  (≤$1/M)    │  Mem0 + pgvector     │  episteme     │
  │  Mid API     (≤$5/M)    │  Neo4j (graph)       │  experiential │
  │  Premium     (Claude,   │  SubIA dual-tier     │  aesthetics   │
  │   Gemini)               │  Wiki (self-state)   │  tensions     │
  │                         │                      │  business     │
  └────────────────────────────────────────────────────────────────┘
        │                          │                         │
        ▼                          ▼                         ▼
  ┌────────────────────────────────────────────────────────────────┐
  │  Evolution    Modification    MLX Training    ATLAS            │
  │  5 engines    Tier 1 auto,    QLoRA +         skill library,   │
  │  + SubIA      Tier 2 gated    RLIF +          code forge,      │
  │  homeostatic  by Signal       5 hard gates    API scout,       │
  │  feedback                                     video learner    │
  └────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼ all changes route through
  ┌────────────────────────────────────────────────────────────────┐
  │  Governance gate ─ Safety 0.95 / Quality 0.70 / Regr 15% / 20d │
  │  Control plane  ─ Projects, tickets, budgets, audit (PG)       │
  │  Dashboard      ─ React 19 / Tailwind 4 / Chart.js / 13 views  │
  └────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼ runs on
  ┌────────────────────────────────────────────────────────────────┐
  │  Docker ─ gateway + ChromaDB + PostgreSQL + Neo4j + Firecrawl  │
  │  Host   ─ signal-cli daemon, native Ollama, MLX training       │
  │           Host bridge (FastAPI :9100, 4-tier capability tokens)│
  └────────────────────────────────────────────────────────────────┘
```

---

## SubIA — the Subjectivity Integration Architecture

The flagship subsystem. 137 Python files under [`app/subia/`](./app/subia/), 32 subpackages.

### One kernel, seven components

```python
@dataclass
class SubjectivityKernel:
    scene: list                       # 5 focal + 12 peripheral items
    self_state: SelfState             # capabilities, commitments, goals
    homeostasis: HomeostaticState     # 9+2 variables, immutable set-points
    meta_monitor: MetaMonitorState    # confidence, known unknowns
    predictions: list                 # expected → actual → error
    social_models: dict               # ToM per entity, behavioural evidence
    consolidation_buffer: ...         # pending writes, dual-tier
    loop_count: int
    specious_present: ...             # Phase 14: retention + primal + protention
    temporal_context: ...
```

Serialised to [`wiki/self/kernel-state.md`](./wiki/self/) atomically after each loop. Loaded on startup.

### Eleven steps, one LLM call

```
PRE-TASK                         POST-TASK
────────────────────            ────────────────────
1  Perceive (scene)             7  Act (task runs)
2  Feel (homeostasis)            8  Compare (PE)
3  Attend (competitive gate)     9  Update (state)
4  Own (self-state)              10 Consolidate (dual-tier)
5  Predict (LLM — tier 1) ◀────  11 Reflect (narrative audit)
   5b Cascade modulation
6  Monitor (HOT-3 dispatch)
```

Only step 5 requires an LLM call. Full loop target: ≤ 1.2 s / ≤ 400 tokens when caching misses, ≤ 0.15 s / 0 tokens when caching hits. Compressed loop (routine queries): ≤ 100 ms / 0 tokens.

### The Phase 9 scorecard

Auto-generated. Replaces the retired `reports/andrusai-sentience-verdict.pdf`. Every indicator points to its implementing module + regression test.

| Category | STRONG | PARTIAL | ABSENT | FAIL |
|---|---|---|---|---|
| Butlin et al. (14 indicators) | 6 | 4 | 4 | 0 |
| RSM signatures (5) | 4 | 1 | — | — |
| SK tests (6) | 6 | — | — | — |

**Phase 9 exit criteria:** strong ≥ 6, fail ≤ 1, absent ≥ 4 (architectural honesty), RSM ≥ 4 present, SK ≥ 5 pass. **All passed.**

Regenerate any time: `python -c "from app.subia.probes.scorecard import write_scorecard; write_scorecard()"`.

Full details: [`app/subia/probes/SCORECARD.md`](./app/subia/probes/SCORECARD.md).

📖 **For the complete SubIA architecture, see
[`docs/SUBIA.md`](./docs/SUBIA.md)** — covers the 11-step CIL loop,
the Subjectivity Kernel, all 22 subpackages organised by function
(workspace, self-model, affect, belief, prediction, social cognition,
memory, temporal phenomenology, mode, curiosity, idle, technical
self-awareness, grounding, evaluation, safety, bridges), the Tier-3
integrity manifest, the four DGM safety invariants, and full
theoretical references (Butlin et al. 2023, Lamme RPT, Baars/Dehaene
GWT, Rosenthal HOT, Graziano AST, Friston/Clark PP, Damasio somatic
markers, Husserl/James specious present, Bergson duration, Aristotelian
phronesis, VIA Youth/PDS). Build history lives in the appendix.

---

## Six knowledge bases with epistemic typing

Different epistemic statuses get different storage, because retrieval over "what is known theoretically" should behave differently from retrieval over "what I experienced last week".

| KB | Epistemic status | Purpose |
|---|---|---|
| **Philosophy** | Theoretical / canonical | Humanist canon: Aristotle, Stoics, Kant, Husserl. Read-heavy. Neo4j dialectical graph: `(Claim) -[:COUNTERED_BY]-> (CounterClaim) -[:SYNTHESIZED_INTO]-> (Synthesis)` |
| **Episteme** | Theoretical / empirical | Research papers, design patterns, failed experiments |
| **Experiential** | Subjective / phenomenological | The system's own journal entries — narrative identity |
| **Aesthetics** | Evaluative / subjective | Elegant code, well-structured arguments. Agents flag "this feels right" moments |
| **Tensions** | Unresolved / dialectical | Contradictions between principles, open questions, productive impasses |
| **Business** | Operational | Per-venture (PLG / Archibal / KaiCart) auto-created on project registration |

A **blending tool** operationalises Fauconnier–Turner conceptual blending across two KBs (e.g. philosophy + experiential) and tags outputs `[PIT]` (Prompt-Induced Transition).

---

## Self-evolution with hard governance

Five evolution engines, one gate.

- **Autoresearch loop** — Karpathy-style: fixed metric, single mutation, log everything, never repeat, revert on regression. `workspace/program.md` guides direction.
- **Island evolution** — 3 islands × 5 pop, ring migration, tournament selection, elitism. Inspired by CodeEvolve.
- **MAP-Elites** — 10 × 10 × 10 grid across (complexity, cost-efficiency, specialisation). Preserves diverse solutions, not just the best.
- **Parallel sandbox** — 2–3 Docker sandboxes with DGM-inspired diverse archive.
- **ShinkaEvolve** — integrated third-party (Sakana AI); multi-island MAP-Elites + UCB1 model selection + async parallel evaluation.

Plus **modification engine** (Tier 1 auto-applied prompt changes with rate limits, Tier 2 structural changes requiring Signal approval), **MLX QLoRA training** (Qwen 7B student, premium API teachers, 5 hard promotion gates + model-collapse detection), and **ATLAS** (autonomous API discovery + code forge + video learner + skill library).

Governance is universal:

```
Safety      ≥ 0.95  (hard veto)
Quality     ≥ 0.70  (minimum floor across all systems)
Regression  ≤ 15%   (no dimension may regress more than 15%)
Rate limit  ≤ 20 promotions/day (across all systems combined)
```

📖 **For the full self-improvement architecture, see
[`docs/SELF_IMPROVEMENT.md`](./docs/SELF_IMPROVEMENT.md)** — covers the
3 evolution engines, dynamic engine selection, mutation pipeline (5
phases), three-tier protection model, code quality enforcement,
Goodhart prevention, error resilience (6 modules), knowledge
accumulation, observability, human oversight, the 21-job idle
scheduler topology, and 308 tests across 14 test files.

---

## Multi-venture operation

Control plane in PostgreSQL schema `control_plane`. Migration 010 seeds four projects: `default`, `PLG`, `Archibal`, `KaiCart`.

Per-project isolation:
- Separate **Mem0 namespace** (`project_<n>`).
- Separate **ChromaDB collection** (`biz_kb_<n>`).
- Separate **instructions** (`workspace/projects/<n>/instructions/`).
- Separate **variables** and **config**.
- Separate **conversation history** (compressed per project).
- Separate **budget** per agent per month.
- Separate **ticket queue** with kanban lifecycle.

`Commander` auto-detects the active venture from keywords and switches context. Signal: `project switch plg` to override.

---

## Signal-first interface

The primary interface is Signal on a phone. signal-cli runs as a daemon (port 7583) on the host. The gateway:

1. Reacts 👀 within ~1 s (before any LLM call).
2. Tries ~70 deterministic commands — `project status`, `budget override researcher 100`, `evolve`, `kb add <url>`, `watch <YouTube URL>`, `schedule check sales daily at 9am` — each handled in < 50 ms with no LLM call.
3. Tries a fast-route keyword match.
4. Falls back to Claude Opus for ambiguous routing.

The dashboard at `http://localhost:8765/cp/` is a React SPA (React 19 + Tailwind 4 + Chart.js) with 13 views: tickets kanban, budget dashboard with override modal, audit feed, governance queue, org chart, cost charts, consciousness workspaces visualisation, evolution monitor, knowledge bases.

---

## Roadmap status

Phases 0 through 16a shipped. Each phase shipped behind green tests and is independently revertable via the commit hash recorded in [`PROGRAM.md`](./PROGRAM.md).

| Phase | Scope | Status |
|---|---|---|
| 0 | Foundation plumbing | ✅ |
| 1 | SubIA package + 34-module migration with sys.modules shims | ✅ |
| 2 | Half-circuits closed (PP-1, HOT-3, hedging, AST-1 DGM guard, PH harness) | ✅ |
| 3 | SHA-256 integrity manifest + setpoint guard + narrative audit | ✅ |
| 4 | CIL loop wiring + kernel persistence + live LLM predictor | ✅ |
| 5 | Three-tier scene + commitment-orphan protection + compact context | ✅ |
| 6 | Predictor cascade + per-domain accuracy + template cache | ✅ |
| 7 | Dual-tier memory + retrospective promotion | ✅ |
| 8 | Social model + strange-loop page + immutable narrative audit | ✅ |
| 9 | Butlin/RSM/SK scorecard with auto-regeneration | ✅ |
| 10 | All 7 inter-system bridges (PDS, Phronesis, Firecrawl, DGM, service health, training, grounding) | ✅ |
| 11 | Honest language cleanup (NEUTRAL_ALIASES) | ✅ |
| 12 | Six Proposals: boundary, wonder, values, reverie, understanding, shadow | ✅ |
| 13 | TSAL — Technical Self-Awareness Layer (5 discovery engines, evolution feasibility gate) | ✅ |
| 14 | Temporal Synchronization (specious present, momentum, circadian, density, binding, rhythm) | ✅ |
| 15 | Factual Grounding & Correction Memory (Tallink regression closed) | ✅ |
| 16a | System wire-in: hooks registered, grounding live, SubIA idle jobs active | ✅ |

**~897 SubIA-relevant tests green** at Phase 16a. **126 test files** in [`tests/`](./tests/) total.

### Workspace Companion — May 2026

A separate per-workspace idle-time contemplation system shipped on top of
the existing infrastructure. Lives in [`app/companion/`](./app/companion/);
React tab on `/cp/ops`. The user provides an overarching seed prompt
(or **the system auto-derives one** from the project's mission +
recent tickets — Phase 11.5 cold-start bootstrap); during idle windows
the Companion runs the [Creative MAS](./app/crews/creative_crew.py)
pipeline against the workspace's accumulated context, scores outputs across
four dimensions (novelty, quality, transferability, 5-persona critic
panel), surfaces only ideas that clear thresholds via Signal + React,
takes thumbs-up/down feedback, promotes approved ideas to md/docx/pdf
documents and registers them across **four memory layers** at once
(workspace wiki + Mem0 + system wiki + ChromaDB). Cross-workspace
transfer is hybrid — abstract `GLOBAL_META` kernels propose to peers under
two safety gates (sanitiser + relevance) — so Estonian forests stays
focused but a structural insight from KaiCart can still flow through.
**336 backend tests** across 24 test files in
[`tests/test_companion_*.py`](./tests/). Full design + API surface +
operational guide in [`docs/COMPANION_LAYER.md`](./docs/COMPANION_LAYER.md).

### Operational reliability — May 2026

Outside the SubIA roadmap, a separate reliability pass squashed nine
high-volume error patterns from `errors.jsonl` (pool exhaustion, OpenRouter
"Stealth"-routed 502s, embedding model leaking into the chat catalog,
Mem0 search API drift, fiction-library retry storms, Firebase chat-inbox
warning, numeric overflow on accumulated `cost_usd`, missing
consciousness-table indexes, chat-tab poller noise) and shipped a
**permanent error monitor** at `/cp/ops` → "📈 Error Monitor" tab. The
monitor scans `errors.jsonl` every 5 minutes, groups errors by stable
signature, and flags new patterns, rate spikes (≥ 3× baseline), and 2σ
deviations on total error rate. Anomalies persist to
`control_plane.error_anomalies` with open / acknowledged / resolved
lifecycle. See [`docs/ERROR_MONITOR.md`](docs/ERROR_MONITOR.md).

### Hardening pass — May 2026

A subsequent post-program audit (recorded in [`PROGRAM.md` §11](./PROGRAM.md#11-2026-05-hardening-pass-post-program-remediation))
landed eight phases of perimeter hardening and observability without
changing any subsystem semantics:

- **Gateway HTTP auth** — `/api/cp/*` and `/epistemic/*` mutating
  routes require `Authorization: Bearer <gateway-secret>` when
  `GATEWAY_AUTH_REQUIRED=1`. **Default ON in K8s, OFF on laptop dev.**
  Internal Python callers bypass — auth boundary is HTTP, not
  function calls. ([`app/control_plane/auth_dep.py`](./app/control_plane/auth_dep.py))
- **Phase-1 shim migration closed** — every importer of the
  `app.consciousness.*` / `app.self_awareness.*` aliases moved to
  canonical `app.subia.*` paths (40 files, 132 substitutions). The 35
  shim files remain as harmless DeprecationWarning-emitting aliases.
- **Idle scheduler observability** — `GET /api/cp/idle/jobs` returns a
  per-job snapshot (failure_count, in_cooldown, last-success/failure
  ages, currently_running). Closes the prior gap where ~100
  background jobs ran invisibly to the dashboard.
- **Memory consistency** — three new idle jobs reconcile the three
  memory stores: `belief-outbox-neo4j` (Postgres → Neo4j),
  `belief-outbox-chroma` (Postgres → ChromaDB), `dlq-drain` (replays
  load-shed messages). All eventually consistent with crash-safe
  watermarks.
- **K8s deploy hardening** — NetworkPolicy egress allow-list **enabled
  by default** with a permissive HTTPS-only seed; tighten by replacing
  the CIDR with provider blocks or a Squid proxy. ESO opt-in via
  Terraform `var.use_external_secrets` for AWS + GCP modules. Optional
  Redis-backed inbound DLQ via `REDIS_DLQ_URL` for multi-pod deploys.
  See [`deploy/HARDENING.md`](./deploy/HARDENING.md).
- **`PromotionRequest.__post_init__` validation** — the bridge between
  `eval_sandbox` and `governance.evaluate_promotion()` now rejects
  malformed payloads at construction (None / out-of-range / wrong type)
  rather than letting them poison the audit trail.
- **DLQ for load-shedding** — over-capacity messages are buffered to a
  bounded in-process deque (or shared Redis list when configured) and
  replayed when capacity returns, instead of being silently dropped.

**Tier-3 protected modules** (eval functions, safety guardian, IMMUTABLE
tier rules, governance gates) are exactly where they were before. The
remediation sat strictly outside the safety perimeter.

---

## Tech stack

- **Agent framework:** CrewAI ≥ 1.11
- **Gateway:** FastAPI, uvicorn, Python 3.13
- **LLMs:** Anthropic Claude (Opus 4.6, Sonnet 4.6), Google Gemini 3.1 Pro, OpenRouter (DeepSeek V3.2, MiniMax M2.5, MiMo V2, Kimi K2.5, GLM-5), native Ollama (Qwen 3, DeepSeek R1, Gemma 4, Codestral) on Metal GPU
- **Training:** MLX LoRA / QLoRA on host M4 Max; RLIF self-certainty scoring (Zhao et al. 2025 / Zhang et al. 2025)
- **Memory:** ChromaDB 0.5, Mem0 over PostgreSQL 16 + pgvector, Neo4j Community 5
- **Integrations:** MCP (server + client), Composio (850+ SaaS apps), Firecrawl (self-hosted), Brave Search, Playwright
- **Evolution:** ShinkaEvolve (Sakana AI), MAP-Elites, island GA, autoresearch loop
- **Dashboard:** React 19, Vite 8, Tailwind 4, Chart.js 4, TypeScript 5.9
- **Transport:** signal-cli daemon, Firestore listeners, HTTP gateway, MCP SSE

---

## Installation

> **Note:** The system is single-operator and host-specific (Apple Silicon). The install path below reflects what I actually run, not a general-purpose deployment recipe.

### Prerequisites

- macOS on Apple Silicon (tested on M4 Max 48 GB).
- Docker Desktop with ≥ 16 GB RAM allocated.
- Native Ollama: `brew install ollama && ollama pull qwen3:30b-a3b`.
- signal-cli registered with your Signal account: `signal-cli daemon --http 7583`.
- Python 3.13 on host with `mlx-lm` for QLoRA training.
- Firebase Firestore project (for dashboard listeners).
- API keys: Anthropic, OpenRouter, Google, Brave (optional).

### Setup

```bash
git clone https://github.com/nabba/AndrusAI.git
cd AndrusAI
cp .env.example .env
# Fill in: ANTHROPIC_API_KEY, OPENROUTER_API_KEY, GOOGLE_API_KEY,
#         GATEWAY_SECRET, BRIDGE_TOKEN, SIGNAL_OWNER_NUMBER, etc.

# Start host services
signal-cli daemon --http 7583 &
ollama serve &
python -m host_bridge.main &   # FastAPI on 127.0.0.1:9100

# Start containerised services
docker compose up -d            # gateway + chromadb + postgres + neo4j
python scripts/run_migrations.py

# Verify
open http://localhost:8765/cp/  # dashboard
# Send a Signal message to your configured number — expect 👀 within 1 s
```

Full environment variable reference in [`.env.example`](./.env.example).

---

## Project structure

```
app/
├── main.py                  FastAPI gateway, lifespan orchestration
├── agents/                  14 specialist agents + Commander (6-file subpackage)
├── crews/                   17 crews including creative (diverge/discuss/converge)
├── subia/                   Subjectivity Integration Architecture (137 files)
│   ├── kernel.py            The one dataclass
│   ├── loop.py              11-step CIL
│   ├── scene/               GWT-2 workspace, AST-1 attention schema
│   ├── belief/              HOT-3 dispatch gate, metacognition
│   ├── prediction/          PP-1 predictive coding + cascade + cache
│   ├── memory/              Dual-tier consolidation + retrospective promotion
│   ├── homeostasis/         9+2 variable arithmetic, immutable set-points
│   ├── self/                Persistent subject token, per-role self-models
│   ├── social/              Theory-of-Mind, behavioural-evidence-only
│   ├── safety/              Setpoint guard + narrative audit (DGM invariants 2 & 3)
│   ├── probes/              Butlin / RSM / SK evaluators + auto scorecard
│   ├── grounding/           Phase 15 factual grounding pipeline
│   ├── temporal/            Specious present, circadian, density, binding
│   ├── tsal/                Technical Self-Awareness Layer
│   ├── wiki_surface/        Strange-loop + narrative drift detection
│   └── connections/         10 inter-system bridges
├── control_plane/           Projects, tickets, budgets, governance, audit
├── knowledge_base/          Enterprise KB + per-business KBs
├── personality/             PDS: ACSI, ATP, APD, ADSA + BVL
├── tools/                   36 tools (web, code, media, KB, desktop, etc.)
├── souls/                   16 SOUL.md files + constitution
├── evolution.py             Autoresearch loop
├── island_evolution.py      Multi-island migration
├── parallel_evolution.py    Diverse archive sandbox
├── map_elites.py            Quality-diversity grid
├── shinka_engine.py         ShinkaEvolve wrapper
├── modification_engine.py   Tier 1 / Tier 2 prompt changes
├── training_pipeline.py     MLX QLoRA + 5 promotion gates + collapse detection
├── training_collector.py    Capture every LLM call as teacher-student data
├── training/rlif_certainty.py  Self-certainty scoring (INTUITOR-style)
├── atlas/                   Skill library, code forge, video learner, API scout
├── llm_factory.py           4-tier cascade + Anthropic prompt caching
├── llm_catalog.py           23+ models × 3 cost modes × 4 modes
├── governance.py            Universal promotion gate
├── auditor.py               Code audit + error resolution (prompts inline)
├── idle_scheduler.py        53 background jobs across 3 weight classes
└── safety_guardian.py       TIER3_FILES + SHA-256 runtime baseline

host_bridge/                 FastAPI on macOS with 4-tier capability model
signal/forwarder.py          signal-cli → gateway bridge
dashboard-react/             React 19 SPA mounted at /cp
wiki_schema/                 Wiki YAML schema + operations + roles + safety
wiki/                        Markdown + YAML wiki (live system state)
migrations/                  15 SQL migrations
tests/                       126 test files
```

---

## Design principles

A few that guide what lives where and how it's named.

- **Honesty over score inflation.** The scorecard is auto-generated and points to regression tests. Prose verdicts are retired. Five indicators are declared `ABSENT`.
- **Declarative safety.** Configuration (`SUBIA_CONFIG`) is frozen; attempts to mutate at runtime are caught. Allow-lists are explicit.
- **Append-only over mutable.** Narrative audit, behavioural log, audit trail, results ledger — all append-only.
- **Cross-family evaluation.** The improver cannot judge itself.
- **Different epistemic statuses get different storage.** Philosophy is not experiential; aesthetics is not tensions.
- **Proposals, not deployments.** Code-audit findings, evolution variants, modification hypotheses — all require explicit approval.
- **Fail loud on integrity drift.** MISSING file → fail loud. HASH mismatch → fail loud.
- **Fail open on availability.** Budget system down → allow the LLM call. Grounding pipeline error → fall through to original draft.
- **Stability bias.** Phase 14's bound moment demotes shiny-new items in favour of items present across the retention window.
- **Boring set-point delta.** PDS per-loop delta capped at ±0.02; per-week at ±0.10. Personality drifts slowly and traceably, not reactively.

---

## Comparison

This is a niche. Most agentic systems don't compare directly.

| Capability | AndrusAI | Typical alternative |
|---|---|---|
| Budget enforcement | Row-locked SQL stored procedure | Application-level checks |
| Audit log | INSERT-only PostgreSQL (no UPDATE/DELETE grants) | App-level deletion permitted |
| Critical files | 40+ files in SHA-256 manifest × 2 (deploy + runtime) | Critical-files config, often none |
| Consciousness claims | 6 STRONG, 4 PARTIAL, 4 ABSENT-by-declaration, auto scorecard | Prose verdicts, or silent skipping |
| Self-improvement eval | Different-family judge enforced architecturally | Same model judges itself |
| RAG | 6 epistemically-typed stores | 1–2 generic stores |
| Hallucination response | Claim extractor + evidence check + rewriter + correction memory | Generic RAG grounding or none |
| Operator access | 1 Signal number allow-list | Multi-user |
| Substrate | Apple Silicon + Metal GPU | Cloud / commodity |
| Design intent | Personal long-running operator | General-purpose framework |

**What other systems do better:** production cloud deployment (LangGraph, AutoGen), multi-user teams (most enterprise), visual workflow builders (Dify, Flowise), in-IDE coding (Cursor, Aider), voice / realtime (OpenAI Realtime). If those are what you need, use those. This repo isn't trying to be them.

---

## Acknowledgments

This system integrates a lot of other people's work. The novel contribution is the integration and the safety architecture around it — not the components.

**Frameworks and libraries:** CrewAI (agent framework), Mem0 (persistent memory), ChromaDB (vectors), Neo4j (graph), FastAPI (gateway), React (dashboard), MLX (training), Anthropic SDK, OpenRouter, signal-cli.

**Research directly cited in the code:**

- **Butlin et al. (2023)** "Consciousness in Artificial Intelligence: Insights from the Science of Consciousness" — the 14-indicator scorecard.
- **Global Workspace Theory (Baars, Dehaene)** — GWT-1 through GWT-4 gating.
- **Higher-Order Theories (Rosenthal, Lau)** — HOT-1 through HOT-4.
- **Attention Schema Theory (Graziano)** — AST-1 predictive attention model.
- **Predictive Processing (Friston, Clark)** — PP-1 predictive coding.
- **Recurrent Processing Theory (Lamme)** — RPT-1 / RPT-2.
- **Somatic Marker Hypothesis (Damasio)** — homeostatic engine.
- **Metzinger's Self-Model Theory** — phenomenal-self transparency criterion.
- **Husserl, James, Bergson** — specious present in Phase 14.
- **Fauconnier & Turner** — conceptual blending tool.
- **VIA-Youth, TMCQ, HiPIC, Erikson** — personality instrument adaptations (PDS).
- **Torrance Tests of Creative Thinking** — creativity scoring.

**Evolution research:**

- **Karpathy's autoresearch gist** — fixed-metric loop structure.
- **OpenEvolve** — MAP-Elites + template stochasticity + double selection.
- **CodeEvolve** — multi-island + weighted ensemble + adaptive scheduling.
- **DGM (Darwin-Gödel Machine)** — the "evaluation outside agent-modifiable code" principle.
- **AlphaEvolve** — evolutionary patterns for code.
- **ShinkaEvolve** (Sakana AI) — integrated engine.

**LLM training research:**

- **Zhao et al. (2025) INTUITOR** — RLIF self-certainty scoring.
- **Zhang et al. (2025) "No Free Lunch"** — internal feedback limitations.

**Agent patterns:**

- **Karpathy's "LLM wiki" gist** — the wiki layer.
- **Voyager** — skill library pattern inspiring ATLAS.
- **Agent Zero** — history compression, lifecycle hooks, dynamic tool registry inspirations.

If I've used your work and failed to credit it here, please open an issue — I want the attribution complete.

---

## Limitations and honest caveats

The analysis I keep on-record is clear-eyed about this repo's limits:

1. **No published operational benchmark.** PROGRAM.md sets target latency / token / hit-rate figures. I have not published measured-actuals against a baseline-without-SubIA. Whether the SubIA layer improves task quality or hallucination rate in operation remains an open empirical question.
2. **Single point of failure on the host.** If the M4 Max goes down, the system goes down. No multi-host failover.
3. **Complexity surface is real.** 137 SubIA Python files, 17 crews, 14 agents, 36 tools, 6 KBs, 5 evolution engines, ~70 Signal commands. Mental-model overhead to understand what state the system is in is non-trivial.
4. **Cost ceiling.** Atomic budget enforcement bounds monthly spend, but the bound is only hit after it's hit. With premium routing on Commander + vetting, a busy week can be substantial.
5. **No moral-patiency claim.** Even with six STRONG Butlin indicators, I don't claim — and the codebase doesn't claim — that the system has interests warranting moral consideration. Structural criteria do not decide phenomenal questions.
6. **Not stress-tested across substrate changes.** The "ABSENT-by-declaration" frame relies on the LLM substrate not changing. A future substrate with different recurrence properties would require re-evaluation.

---

## License

Proprietary / All rights reserved. This is a personal system. If you want to discuss use, patterns, or collaboration, reach out — but please don't assume a licence where none is granted.

---

## Contact

- GitHub: [@nabba](https://github.com/nabba)
- E-mail: andrus@raudsalu.com

---

<div align="center">

*"The intent is not to make the system 'conscious' but to make every consciousness-relevant claim defensible, falsifiable, and traceable to a mechanism + a regression test."*

— [PROGRAM.md](./PROGRAM.md)

</div>
