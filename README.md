# AndrusAI Agent Team

An autonomous, self-improving AI agent team built on [CrewAI](https://www.crewai.com/). Control it from your iPhone via Signal. Monitor it in real time from a Firebase-hosted dashboard.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [System Diagram](#system-diagram)
  - [Agent Roles](#agent-roles)
  - [Crew System](#crew-system)
  - [Tool Inventory](#tool-inventory)
- [LLM System](#llm-system)
  - [Multi-Tier Model Catalog](#multi-tier-model-catalog)
  - [LLM Modes](#llm-modes)
  - [Cost Modes](#cost-modes)
  - [Model Selection Algorithm](#model-selection-algorithm)
  - [Risk-Based Vetting](#risk-based-vetting)
- [Self-Aware AI System](#self-aware-ai-system)
  - [Phase 1: Functional Self-Awareness](#phase-1-functional-self-awareness)
  - [Phase 2: Shared Memory & Cooperation](#phase-2-shared-memory--cooperation)
  - [Phase 3: Proactive Cooperation](#phase-3-proactive-cooperation)
  - [Phase 4: Meta-Cognitive Self-Improvement](#phase-4-meta-cognitive-self-improvement)
- [SOUL.md Personality Framework](#soulmd-personality-framework)
- [Memory System](#memory-system)
  - [ChromaDB Vector Memory](#chromadb-vector-memory)
  - [Scoped Memory Hierarchy](#scoped-memory-hierarchy)
  - [Mem0 Persistent Memory](#mem0-persistent-memory)
  - [Semantic Result Cache](#semantic-result-cache)
  - [Conversation History](#conversation-history)
- [Background Systems](#background-systems)
  - [Idle Scheduler](#idle-scheduler)
  - [Evolution Loop](#evolution-loop)
  - [Self-Healing](#self-healing)
  - [Circuit Breaker](#circuit-breaker)
- [Knowledge Base](#knowledge-base)
- [Monitoring Dashboard](#monitoring-dashboard)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Docker Architecture](#docker-architecture)
- [Security](#security)
- [Testing](#testing)
- [Cost Estimate](#cost-estimate)
- [Development](#development)

---

## Overview

This project implements an autonomous AI agent team that you interact with entirely through Signal messages from your phone. A **Commander** agent receives your requests, classifies them by type and difficulty (1-10), and dispatches specialist **crews** (Research, Coding, Writing, Media) to handle them — in parallel when multiple requests arrive simultaneously. The system continuously improves itself by learning new topics, diagnosing its own errors, and running an evolution loop inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

The system features a **self-aware AI architecture** built on current research (Li et al. 2025, ProAgent AAAI 2024, Park et al. 2023) with four capability layers: functional self-awareness, shared memory with belief tracking, proactive cooperation, and a meta-cognitive self-improvement loop. Agent personalities are defined through a **SOUL.md framework** following the SoulSpec standard — each agent has a distinct identity, values, and communication style.

Key capabilities:

- **Natural language task dispatch** — send any request via Signal; Commander routes to the right crew
- **Concurrent message handling** — dedicated thread pool processes multiple requests in parallel
- **Web research** — search the web, read articles, extract YouTube transcripts
- **Code execution** — write, test, and debug code in a sandboxed Docker container
- **Content creation** — summaries, reports, documentation, emails
- **Media analysis** — YouTube videos, images, audio/podcasts, documents with multimodal models
- **Multi-tier LLM fleet** — 5 tiers (local/free/budget/mid/premium) with 18+ models across Ollama, OpenRouter, and Anthropic
- **4 runtime LLM modes** — local, cloud, hybrid, insane — switchable from Signal or dashboard
- **Risk-based vetting** — 4-tier verification (none/schema/cheap/full) calibrated by task risk
- **Semantic result caching** — skips redundant work when similar questions arrive (cosine similarity ≥ 0.92)
- **Knowledge base** — ingest PDFs, DOCX, XLSX, URLs; semantic search across all documents
- **Self-awareness** — agents assess confidence, report blockers, store post-task reflections
- **Shared belief states** — agents track teammates' progress, needs, and current tasks
- **Proactive cooperation** — automatic detection of low confidence, unfulfilled needs, quality drift
- **Adversarial quality review** — Critic agent challenges research outputs for accuracy
- **Meta-cognitive policies** — Retrospective crew generates improvement policies from execution traces
- **Idle scheduler** — self-improvement, evolution, and retrospective work fills dead time between user requests
- **Self-improvement** — learns new topics, auto-discovers skill gaps, saves skill files
- **Self-healing** — automatically diagnoses errors and creates fixes
- **Autonomous evolution** — experiments on itself, keeps improvements, discards regressions
- **SOUL.md personalities** — each agent has a distinct identity, personality, and expertise
- **File attachment support** — PDFs, DOCX, XLSX, images via Signal for analysis
- **Real-time dashboard** — Firebase-hosted with crew status, task tracking, LLM mode control, background task toggle
- **Persistent memory** — dual memory: ChromaDB for operational state, Mem0 (PostgreSQL + Neo4j) for cross-session facts
- **Conversation history** — remembers recent exchanges for contextual follow-ups
- **Cloud backup** — optional git-based workspace sync

---

## Architecture

### System Diagram

```
iPhone (Signal) --> signal-cli daemon --> Forwarder --> FastAPI Gateway (127.0.0.1:8765)
                                                              |
                                                     Commander (Router + Soul)
                                                     [ThreadPoolExecutor: concurrent dispatch]
                                                     |       |       |       |
                                                Research  Coding  Writing  Media
                                                  Crew     Crew    Crew    Crew
                                                   |
                                                Critic Review
                                                   |
                                                Proactive Scan
                                                   |
                                              Vetting Pipeline (4-tier)
                                                   |
                                              Semantic Cache
                                                   |
                                              Response to User

LLM Fleet:                          Background Services (cron + idle):
  LOCAL:  Ollama on Metal GPU         Self-improvement (daily 3 AM + idle)
  FREE:   OpenRouter :free models     Evolution loop (every 6h + idle)
  BUDGET: DeepSeek V3.2, MiniMax      Retrospective (daily 4 AM + idle)
  MID:    Kimi K2.5, GLM-5            Topic discovery (idle)
  PREMIUM: Claude Opus/Sonnet,        Improvement scan (idle)
           Gemini 3.1 Pro             Benchmark snapshot (daily 5 AM)
                                      Code auditor (every 4h)
External Services:                    Error resolution (every 30 min)
  Brave API (search)                  Workspace sync (hourly)
  Web Fetch (articles)                Firebase heartbeat (60s)
  YouTube (transcripts)
  Docker Sandbox (code)
  ChromaDB + Mem0 (memory)
  Firebase (dashboard)
```

### Agent Roles

| Agent | Model | Role | Key Tools |
|-------|-------|------|-----------|
| **Commander** | Claude Opus 4.6 | Routes requests, handles commands, proactive scanning | Memory, belief state |
| **Researcher** | DeepSeek V3.2 / Kimi K2.5 | Web search, article reading, YouTube transcripts | web_search, web_fetch, youtube_transcript |
| **Coder** | MiniMax M2.5 / Gemini 3.1 Pro | Code generation and sandbox execution | execute_code, file_manager |
| **Writer** | Claude Sonnet 4.6 | Summaries, reports, docs, emails | file_manager, web_search |
| **Media Analyst** | Nemotron Nano 2 VL / Kimi K2.5 | YouTube videos, images, audio, document analysis | web_search, web_fetch, youtube_transcript, read_attachment |
| **Self-Improver** | DeepSeek V3.2 | Learns topics, extracts knowledge, proposes improvements | web_search, youtube_transcript |
| **Critic** | Gemini 3.1 Pro / DeepSeek V3.2 | Adversarial review of research for accuracy and gaps | scoped_memory |
| **Introspector** | DeepSeek V3.2 | Analyzes execution traces, generates improvement policies | scoped_memory |

All agents share: scoped memory tools, team memory, self-report, reflection, knowledge base search.

### Crew System

| Crew | Description | Key Features |
|------|-------------|--------------|
| **ResearchCrew** | Multi-source research with synthesis | Parallel sub-agents, critic review, belief tracking, benchmarks |
| **CodingCrew** | Sandbox code execution | Policy loading, iterative debugging, benchmarking |
| **WritingCrew** | Destination-adapted content | Audience-aware formatting, policy injection |
| **MediaCrew** | Multimodal content analysis | YouTube, images, audio, OCR; simple/full templates by difficulty |
| **SelfImprovementCrew** | Continuous learning | Learning queue, YouTube extraction, improvement proposals, gap scanning |
| **RetrospectiveCrew** | Meta-cognitive retrospective | Gathers execution traces, generates TRIGGER/ACTION/EVIDENCE policies |

The **parallel runner** provides concurrent crew execution with error isolation (configurable thread pool, default 3 workers).

### Tool Inventory

| Tool | Description |
|------|-------------|
| `web_search` | Brave Search API — top 5 results |
| `web_fetch` | URL content extraction with SSRF protection |
| `get_youtube_transcript` | YouTube transcript extraction with multi-strategy fallback |
| `execute_code` | Docker sandbox execution (Python, Bash, Node.js, Ruby) |
| `file_manager` | Workspace file read/write with path traversal protection |
| `read_attachment` | Text extraction from PDF, DOCX, XLSX, images |
| `knowledge_search` | Semantic search across ingested knowledge base documents |
| `memory_store` / `memory_retrieve` | Per-crew ChromaDB storage and retrieval |
| `team_memory_store` / `team_memory_retrieve` | Shared cross-crew memory |
| `scoped_memory_store` / `scoped_memory_retrieve` | Hierarchical scoped memory with dual retrieval profiles |
| `team_decision` | Records team-level decisions and shared conclusions |
| `update_team_belief` | Updates belief state about teammate progress |
| `team_state` | Views current state of all team members |
| `self_report` | Agent self-assessment: confidence, completeness, blockers, risks |
| `store_reflection` | Post-task reflection: what went well/wrong, lessons learned |
| `mem0_add_memory` | Persistent cross-session fact storage (Mem0) |
| `mem0_search` | Entity relationship graph search (Mem0) |

---

## LLM System

### Multi-Tier Model Catalog

The system manages 18+ models across 5 tiers, optimized for Apple M4 Max (48GB unified memory) for local inference.

**Local Tier** (free, Metal GPU, ~15-25 tok/s):

| Model | Size | Context | Strengths |
|-------|------|---------|-----------|
| qwen3:30b-a3b | 18GB | 32K | MoE — best local all-rounder |
| deepseek-r1:32b | 19GB | 32K | Strong reasoning, architecture |
| codestral:22b | 13GB | 32K | Mistral code specialist |
| gemma3:27b | 17GB | 128K | Large context, research |
| llama3.1:8b | 5GB | 128K | Small fallback |

**Free Tier** (OpenRouter `:free` variants, $0):

| Model | Context | Strengths |
|-------|---------|-----------|
| Nemotron Nano 2 VL | 32K | Multimodal, OCR, media analysis |
| Nemotron 3 Super | 1M | MoE 120B, long context |
| Trinity Large | 128K | 400B sparse, writing |
| Step 3.5 Flash | 256K | Multi-language, reasoning |
| MiniMax M2.5 Free | 196K | Code, SWE-bench |

**Budget Tier** (<$1.50/M output tokens):

| Model | Cost (in/out per M) | Strengths |
|-------|---------------------|-----------|
| DeepSeek V3.2 | $0.28 / $0.42 | Sparse attention, RL-trained, top value |
| MiniMax M2.5 | $0.30 / $1.20 | 80.2% SWE-bench, strong coding |

**Mid Tier** ($1-4/M output tokens):

| Model | Cost (in/out per M) | Strengths |
|-------|---------------------|-----------|
| Kimi K2.5 | $0.60 / $3.00 | 1T MoE, 256K context, multimodal |
| GLM-5 | $0.80 / $4.00 | 744B MoE, #1 open-weight |

**Premium Tier** (highest reliability):

| Model | Cost (in/out per M) | Strengths |
|-------|---------------------|-----------|
| Claude Sonnet 4.6 | $1.00 / $5.00 | #1 GDPval-AA, reliable tool use |
| Claude Opus 4.6 | $5.00 / $25.00 | 0.98 tool-use reliability |
| Gemini 3.1 Pro | $2.00 / $12.00 | #1 on 13/16 benchmarks |

### LLM Modes

Switchable at runtime via Signal (`mode <name>`) or dashboard toggle:

| Mode | Behavior |
|------|----------|
| **local** | Ollama only, Claude fallback for commander/vetting |
| **cloud** | API tier + Anthropic, skip Ollama |
| **hybrid** | Try local first → API → Claude (default) |
| **insane** | Premium only — Opus/Gemini/Sonnet for everything |

Mode state persists in Firestore and syncs in real time between backend and dashboard.

### Cost Modes

Control model selection across all roles:

| Mode | Strategy |
|------|----------|
| **budget** | DeepSeek V3.2 everywhere, Sonnet for commander |
| **balanced** | Best cost/quality per role (default) |
| **quality** | Kimi/Gemini for specialists, Opus for vetting |

### Model Selection Algorithm

1. Check environment variable override (`ROLE_MODEL_RESEARCH=kimi-k2.5`)
2. Get default for role + cost_mode from catalog
3. Detect task type from hint, apply specialist overrides
4. Apply benchmark-driven adjustments
5. Check availability (local RAM, API key presence, circuit breaker state)
6. Return fallback if primary unavailable

### Risk-Based Vetting

A 4-tier verification system calibrated by crew type, task difficulty, and model tier:

| Tier | When Used | Cost |
|------|-----------|------|
| **none** | Premium model + easy task, direct answers | Free |
| **schema** | Budget/mid model + easy writing/research | Free (pattern matching) |
| **cheap** | Budget/mid model + moderate tasks | 1 budget LLM call |
| **full** | All local/free output, all code, difficulty ≥ 8 | 1 Claude Sonnet call |

Escalation chain: schema → cheap → full. If a lighter check fails, the next tier runs automatically.

---

## Self-Aware AI System

The system implements a four-phase self-awareness architecture grounded in current research on functional self-awareness (Li et al. 2025), proactive multi-agent cooperation (ProAgent, AAAI 2024), generative agent memory (Park et al. 2023), and meta-cognitive self-improvement (Evers et al. 2025).

### Phase 1: Functional Self-Awareness

Each agent carries a **structured self-model** (`app/self_awareness/self_model.py`) describing its capabilities, limitations, operating principles, failure modes, and metacognitive triggers. This is injected into every agent's backstory.

Two awareness tools are available to all agents:
- **SelfReportTool** — After completing work, agents assess confidence (high/medium/low), completeness, blockers, risks, and needs from teammates. Reports stored in ChromaDB.
- **ReflectionTool** — Agents record post-task lessons: what went well, what went wrong, what to change. Stored in both agent-specific and shared team memory.

### Phase 2: Shared Memory & Cooperation

**Scoped Memory** (`app/memory/scoped_memory.py`) provides hierarchical memory on top of ChromaDB with two retrieval profiles:
- **Operational** — Recency-weighted for active tasks (boosts items from last 24 hours)
- **Strategic** — Importance-weighted for policies and lessons (filters by high/critical importance)

**Belief State Tracking** (`app/memory/belief_state.py`) implements ProAgent-style intention inference. Each crew updates its agent's state (idle/working/blocked/completed/failed) with current task, confidence, and needs. Commander sees team state in routing context.

**Critic Agent** (`app/agents/critic.py`) provides adversarial quality review on research outputs, checking for unsupported claims, gaps, unjustified confidence, and contradictions.

### Phase 3: Proactive Cooperation

After every crew execution, the Commander runs a **proactive trigger scanner** (`app/proactive/trigger_scanner.py`) checking for:
1. **Low confidence** — Recent self-reports with low confidence trigger verification recommendations
2. **Unfulfilled needs** — Agents with unmet teammate needs get flagged
3. **Quality drift** — Confidence trending downward triggers an alert

Up to 2 proactive notes are appended to each response.

### Phase 4: Meta-Cognitive Self-Improvement

**RetrospectiveCrew** (daily 4 AM + idle time) gathers execution traces and generates improvement policies in TRIGGER/ACTION/EVIDENCE format.

**Policy Loader** loads relevant policies before each crew execution, injecting them into task descriptions.

**Benchmarks** (`app/benchmarks.py`) tracks: task completion time, quality scores, proactive intervention rates, and period-over-period trends. View via `benchmarks` command.

---

## SOUL.md Personality Framework

Following the SoulSpec standard and research on agent personality (arXiv:2510.21413), each agent has a distinct identity defined in markdown soul files. Research shows structured persona files reduce runtime by ~29% and token consumption by ~17%.

```
app/souls/
├── constitution.md      # Shared: Safety > Honesty > Compliance > Helpfulness
├── style.md             # Shared: communication conventions, forbidden patterns
├── agents_protocol.md   # Shared: routing flow, escalation, quality gates
├── commander.md         # Calm, decisive operations manager
├── researcher.md        # Methodical, skeptical, source-obsessed analyst
├── coder.md             # Precise, pragmatic, complexity-allergic engineer
├── writer.md            # Clear, audience-aware communicator
├── self_improver.md     # Curious, systematic, constructively critical
├── media_analyst.md     # Detail-oriented multimodal specialist
└── loader.py            # Loads and composes soul files into backstories
```

Backstory composition: `Agent backstory = CONSTITUTION + SOUL (per-role) + STYLE + Self-Model block`

---

## Memory System

### ChromaDB Vector Memory

Per-crew collections plus shared `team_shared`. Local `all-MiniLM-L6-v2` embeddings (no API calls). Persistent at `/app/workspace/memory/`.

### Scoped Memory Hierarchy

| Scope | Purpose |
|-------|---------|
| `scope_team` | Team-wide decisions and shared context |
| `scope_agent_{name}` | Per-agent private working memory |
| `scope_beliefs` | Belief state tracking |
| `scope_policies` | Improvement policies from retrospective |
| `scope_ecology` | Resource consumption tracking |
| `scope_reflexion_lessons` | Retry loop lessons from past failures |
| `scope_project_{name}` | Per-project knowledge |
| `self_reports` | Agent self-assessment history |
| `reflections_{role}` | Per-agent post-task reflections |
| `result_cache` | Semantic result caching |

Two retrieval profiles: **Operational** (recency-boosted) and **Strategic** (importance-filtered).

### Mem0 Persistent Memory

Cross-session fact extraction and entity relationship tracking:
- **Backend**: PostgreSQL (pgvector) + Neo4j (graph)
- **Fact Extraction**: LLM-based automatic learning from conversations
- **Graph**: Entity relationships via Neo4j
- **Models**: ollama/qwen3:30b-a3b (extraction), all-MiniLM-L6-v2 (embedding)
- **Limits**: 10KB per fact, 50KB per conversation, 2KB per query

### Semantic Result Cache

Avoids redundant crew execution for similar questions:
- ChromaDB `result_cache` collection with cosine similarity ≥ 0.92
- TTL: 1 hour (configurable)
- Max 500 cached entries with auto-pruning
- Checked before crew dispatch — cache hit skips the entire crew pipeline

### Conversation History

SQLite with HMAC-hashed sender IDs. Last N exchanges (default 10) injected for contextual follow-ups.

### Skill Files

Markdown in `workspace/skills/`. Structure: Key Concepts, Best Practices, Code Patterns, Sources. Loaded as context for matching tasks.

---

## Background Systems

### Idle Scheduler

When no user tasks are active, the system fills dead time with background work:

- **Trigger**: 30 seconds after last user task completes
- **Behavior**: Cooperative multitasking — yields immediately when a user request arrives
- **Kill switch**: Toggle from dashboard or Firestore `config/background_tasks`
- **Does NOT replace cron** — cron is the guaranteed baseline; idle scheduling is opportunistic

Job rotation (round-robin, evolution at 2x frequency):

| Job | What It Does |
|-----|--------------|
| **learn-queue** | Process topics from the learning queue |
| **evolution** | Run 5 evolution experiment iterations |
| **discover-topics** | LLM analyzes recent failures → suggests 1-3 new learning topics |
| **retrospective** | Analyze recent performance, generate policies |
| **evolution-2** | Another 5 evolution iterations (doubled frequency) |
| **improvement-scan** | Analyze skill gaps, propose improvements |

### Evolution Loop

Inspired by Karpathy's autoresearch — single-mutation experiments with metric tracking:

1. Measure baseline composite score
2. LLM proposes one focused change (mutation)
3. Apply change, run test tasks, measure new score
4. Keep improvement or revert regression
5. Log everything to `workspace/results.tsv`

**Composite score** (weighted): task success (30%), error rate (20%), self-heal rate (15%), output quality (15%), skill breadth (10%), response time (10%).

Runs every 6 hours via cron + additional iterations during idle time.

### Self-Healing

On crew failure:
1. Log error with full context to `workspace/error_journal.json` (max 100 entries)
2. Spawn background diagnosis agent (non-blocking)
3. Create auto-fix proposal (skill or code change)
4. Track error frequency for pattern detection

Code proposals are validated via AST (blocks dangerous imports: subprocess, socket, pickle, etc.).

### Circuit Breaker

Provider-level resilience with state machine:

| Provider | Failure Threshold | Cooldown |
|----------|-------------------|----------|
| Ollama | 3 failures | 60s |
| OpenRouter | 3 failures | 60s |
| Anthropic | 5 failures | 120s |

States: CLOSED → (failures) → OPEN → (cooldown) → HALF_OPEN → (success) → CLOSED

---

## Knowledge Base

Document ingestion and semantic search system:

- **Supported formats**: PDF, DOCX, XLSX, PPTX, Markdown, TXT, URLs, images
- **Chunking**: Semantic chunking (max 2000 chars per chunk)
- **Storage**: ChromaDB with sentence-transformers embeddings
- **Search**: Query expansion for better retrieval, filtering by document type/date/relevance
- **Tools**: `ingest_document`, `search_knowledge_base`, `list_sources`
- **Dashboard upload**: KB upload queue via Firestore (`kb_queue` collection)

---

## Monitoring Dashboard

Firebase-hosted real-time dashboard with:

- **System status** — health, uptime, last-seen, error trends
- **Crew cards** — per-crew status, current task, completion ETA
- **Task table** — individual task records with timing
- **Activity feed** — rolling log of last 50 actions
- **LLM mode toggle** — switch between local/cloud/hybrid/insane
- **Background tasks toggle** — enable/disable idle scheduler
- **Scheduled jobs** — upcoming cron work visibility

---

## Project Structure

```
crewai-team/
├── app/
│   ├── main.py                       # FastAPI gateway, lifespan, scheduler, thread pool
│   ├── config.py                     # Pydantic settings with validation
│   ├── llm_catalog.py                # Multi-tier model registry (18+ models)
│   ├── llm_factory.py                # Role-based LLM provider with cascading fallback
│   ├── llm_selector.py               # Cost-aware model selection algorithm
│   ├── llm_mode.py                   # Runtime-mutable LLM mode state
│   ├── vetting.py                    # 4-tier risk-based verification pipeline
│   ├── result_cache.py               # Semantic result caching (ChromaDB)
│   ├── idle_scheduler.py             # Background work during idle time
│   ├── evolution.py                  # Autonomous evolution loop (autoresearch)
│   ├── self_heal.py                  # Error diagnosis and auto-fix
│   ├── circuit_breaker.py            # Provider resilience state machine
│   ├── rate_throttle.py              # Token bucket rate limiting
│   ├── benchmarks.py                 # Automated benchmarking system
│   ├── metrics.py                    # Composite scoring system
│   ├── firebase_reporter.py          # Real-time Firestore updates
│   ├── signal_client.py              # Signal messaging client
│   ├── conversation_store.py         # Context and history management
│   ├── sanitize.py                   # Input/output sanitization
│   ├── security.py                   # Auth and rate limiting
│   ├── workspace_sync.py             # Git-based cloud backup
│   ├── auto_deployer.py              # Hot-reload for code fixes
│   ├── auditor.py                    # Code quality auditing
│   ├── proposals.py                  # Improvement proposal management
│   ├── agents/
│   │   ├── commander.py              # Router + proactive scanning + concurrent dispatch
│   │   ├── researcher.py             # Web research specialist
│   │   ├── coder.py                  # Sandbox code execution
│   │   ├── writer.py                 # Content creation
│   │   ├── media_analyst.py          # Multimodal content analysis
│   │   ├── self_improver.py          # Learning and improvement
│   │   ├── critic.py                 # Adversarial quality reviewer
│   │   └── introspector.py           # Meta-cognitive policy generator
│   ├── crews/
│   │   ├── research_crew.py          # Parallel research + critic review
│   │   ├── coding_crew.py            # Sandbox code execution crew
│   │   ├── writing_crew.py           # Destination-adapted content
│   │   ├── media_crew.py             # Multimodal analysis crew
│   │   ├── self_improvement_crew.py  # Learning and skill acquisition
│   │   ├── retrospective_crew.py     # Meta-cognitive retrospective
│   │   └── parallel_runner.py        # Concurrent crew execution
│   ├── tools/
│   │   ├── web_search.py             # Brave Search API
│   │   ├── web_fetch.py              # URL content extraction (SSRF-protected)
│   │   ├── youtube_transcript.py     # Multi-strategy transcript extraction
│   │   ├── code_executor.py          # Docker sandbox execution
│   │   ├── file_manager.py           # Workspace file I/O
│   │   ├── attachment_reader.py      # PDF/DOCX/XLSX/image extraction
│   │   ├── memory_tool.py            # Per-crew ChromaDB tools
│   │   ├── scoped_memory_tool.py     # Hierarchical scoped memory tools
│   │   ├── self_report_tool.py       # Agent self-assessment
│   │   └── reflection_tool.py        # Post-task reflection storage
│   ├── memory/
│   │   ├── chromadb_manager.py       # ChromaDB manager + extensions
│   │   ├── scoped_memory.py          # Hierarchical scoped memory
│   │   ├── belief_state.py           # ProAgent belief tracking
│   │   └── mem0_manager.py           # Mem0 persistent memory integration
│   ├── knowledge_base/
│   │   ├── ingestion.py              # Document chunking and ingestion
│   │   ├── vectorstore.py            # Semantic search with query expansion
│   │   └── tools.py                  # KB tools for agents
│   ├── self_awareness/
│   │   └── self_model.py             # Structured self-models per role
│   ├── souls/                        # SOUL.md personality framework
│   │   ├── loader.py                 # Backstory composition engine
│   │   ├── constitution.md           # Shared values and safety rules
│   │   ├── style.md                  # Communication conventions
│   │   ├── agents_protocol.md        # Routing and escalation flow
│   │   ├── commander.md, researcher.md, coder.md, writer.md
│   │   ├── media_analyst.md, self_improver.md
│   │   └── critic.md
│   ├── proactive/
│   │   ├── trigger_scanner.py        # Post-execution trigger detection
│   │   └── proactive_behaviors.py    # Proactive action definitions
│   └── policies/
│       └── policy_loader.py          # Policy storage and injection
├── dashboard/
│   ├── public/index.html             # Firebase-hosted dashboard UI
│   └── firestore.rules               # Firestore security rules
├── tests/
│   ├── test_security.py              # Security and unit tests
│   └── test_self_awareness.py        # Self-awareness integration tests
├── signal/                           # Signal integration helpers
├── sandbox/                          # Docker sandbox configuration
├── scripts/                          # Installation and utility scripts
├── workspace/                        # Runtime data (Docker volume)
│   ├── skills/                       # Learned skill files
│   ├── memory/                       # ChromaDB persistence
│   ├── output/                       # Generated file output
│   └── error_journal.json            # Error tracking
├── docker-compose.yml                # Multi-service orchestration
├── Dockerfile                        # Container image
├── entrypoint.sh                     # Container startup script
├── requirements.txt                  # Python dependencies
└── .env.example                      # Environment variable template
```

---

## Prerequisites

- **Python 3.11+**, **Docker**, **Docker Compose**
- **signal-cli** (Java 17+), **Tailscale**
- **Anthropic API key**, **Brave Search API key**
- **Dedicated phone number** for the Signal bot
- (Optional) OpenRouter API key for frontier models
- (Optional) Firebase project for dashboard
- (Optional) Ollama for local LLM inference
- (Optional) GitHub repo for workspace backup

---

## Installation

```bash
cd ~/crewai-team
bash scripts/install.sh
```

---

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure. Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Anthropic API key (required) |
| `BRAVE_API_KEY` | — | Brave Search API key (required) |
| `OPENROUTER_API_KEY` | — | OpenRouter API key (frontier models) |
| `SIGNAL_BOT_NUMBER` / `SIGNAL_OWNER_NUMBER` | — | Phone numbers (required) |
| `GATEWAY_SECRET` | — | Auth token (required) |
| `LLM_MODE` | `hybrid` | LLM mode: local, cloud, hybrid, insane |
| `COST_MODE` | `balanced` | Cost mode: budget, balanced, quality |
| `LOCAL_LLM_ENABLED` | `true` | Use local Ollama models |
| `API_TIER_ENABLED` | `true` | Use OpenRouter frontier models |
| `VETTING_ENABLED` | `true` | Enable output verification |
| `MEM0_POSTGRES_PASSWORD` | — | Mem0 PostgreSQL password |
| `MEM0_NEO4J_PASSWORD` | — | Mem0 Neo4j password |
| `SELF_IMPROVE_CRON` | `0 3 * * *` | Self-improvement schedule |
| `EVOLUTION_CRON` | `0 */6 * * *` | Evolution loop schedule |
| `RETROSPECTIVE_CRON` | `0 4 * * *` | Retrospective analysis schedule |
| `BENCHMARK_CRON` | `0 5 * * *` | Benchmark snapshot schedule |
| `MAX_PARALLEL_CREWS` | `3` | Concurrent crew limit |
| `PROACTIVE_SCAN_ENABLED` | `true` | Enable proactive trigger scanning |
| `POLICY_LOADING_ENABLED` | `true` | Enable policy injection |
| `FIREBASE_SERVICE_ACCOUNT_JSON` | — | Firebase service account path |
| `WORKSPACE_BACKUP_REPO` | — | Git remote URL for backup |

See `.env.example` for the full list.

### Signal Setup

Register signal-cli, start daemon with HTTP mode, configure forwarder.

### Tailscale Setup

Zero-trust private networking. Never use `tailscale funnel`.

### Firebase Dashboard Setup

Deploy Firestore rules and hosting. Set service account path in `.env`.

---

## Usage

### Signal Commands Reference

| Command | Description |
|---------|-------------|
| *Any request* | Commander routes to appropriate crew |
| `status` | System health, proposals, LLM mode |
| `mode <local\|cloud\|hybrid\|insane>` | Change LLM mode |
| `learn <topic>` | Add to learning queue |
| `watch <youtube_url>` | Extract and learn from video |
| `improve` | Run improvement scan |
| `proposals` / `approve <id>` / `reject <id>` | Manage proposals |
| `evolve` / `evolve deep` | Trigger evolution (5 or 15 iterations) |
| `experiments` | Show experiment journal |
| `errors` / `diagnose` | Show recent errors / run diagnosis |
| `audit` | Run code audit |
| `fleet` / `models` | LLM fleet status |
| `memory` | Team state summary |
| `retrospective` | Run meta-cognitive retrospective |
| `benchmarks` | Show performance benchmarks and trends |
| `policies` | Show stored improvement policies |
| `skills` | List learned skill files |

### File Attachments

Send PDFs, DOCX, XLSX, images via Signal. Text is extracted and included as task context. The Media crew handles rich multimodal analysis (YouTube videos, images, audio).

---

## Docker Architecture

### Services

| Service | Purpose |
|---------|---------|
| `gateway` | FastAPI app, agents, scheduler, idle loop |
| `chromadb` | Vector database (internal network, no internet) |
| `docker-proxy` | Limited Docker API for sandbox execution |
| `postgres` | Mem0 backend with pgvector (512m limit) |
| `neo4j` | Mem0 entity relationship graph (256m heap) |

### Sandbox Security

Network disabled, read-only FS, all capabilities dropped, memory/CPU/timeout limits (512m/0.5 CPU/30s), non-root user.

### Network Isolation

- **external** — Gateway exposed to host
- **internal** — No internet access (ChromaDB, docker-proxy, postgres, neo4j)

---

## Security

- **Network**: Tailscale encrypted tunnel, `127.0.0.1` bind only
- **Auth**: HMAC Bearer token, owner-only sender authorization
- **Prompt injection**: Pattern filtering, XML wrapping, content labeling
- **SSRF**: Private IP blocking, DNS resolution, redirect validation
- **Sandbox**: Docker isolation with full security hardening
- **Code proposals**: AST validation, blocks dangerous imports (subprocess, socket, pickle)
- **Audit**: JSON rotating log, phone numbers always redacted
- **Rate limiting**: Token bucket (3 RPM default), litellm retry with exponential backoff

---

## Testing

```bash
.venv/bin/python -m pytest tests/ -v          # All tests
.venv/bin/python -m pytest tests/test_security.py -v         # Security tests
.venv/bin/python -m pytest tests/test_self_awareness.py -v   # Self-awareness tests
```

Tests run without live dependencies by stubbing heavy imports. Coverage: sanitization, SSRF, path traversal, rate limiting, config validation, self-models, self-report/reflection tools, scoped memory, belief states, ChromaDB extensions, proactive scanner, policy loader, benchmarks, retrospective crew, and cross-phase integration.

---

## Cost Estimate

~$20-70/month for moderate use (10-20 tasks/day). Budget mode with DeepSeek V3.2 is significantly cheaper (~$5-15/month). Local Ollama models reduce API costs to near zero for routine tasks — vetting calls to Claude Sonnet are the primary remaining cost. The idle scheduler adds background LLM calls but respects the kill switch when resources are needed elsewhere.

---

## Development

### Key Source Files

| File | Responsibility |
|------|---------------|
| `app/main.py` | Entry point, lifespan, scheduler, thread pool, idle scheduler |
| `app/agents/commander.py` | Core routing, concurrent dispatch, proactive scanning |
| `app/llm_catalog.py` | Multi-tier model registry with costs and capabilities |
| `app/llm_factory.py` | Role-based LLM provider with cascading fallback |
| `app/llm_selector.py` | Cost-aware model selection algorithm |
| `app/vetting.py` | 4-tier risk-based verification pipeline |
| `app/idle_scheduler.py` | Background work scheduling during idle time |
| `app/evolution.py` | Autonomous evolution loop |
| `app/result_cache.py` | Semantic result caching |
| `app/souls/loader.py` | Soul file loading and backstory composition |
| `app/self_awareness/self_model.py` | Structured self-models for all roles |
| `app/memory/scoped_memory.py` | Hierarchical scoped memory |
| `app/memory/belief_state.py` | ProAgent belief state tracking |
| `app/proactive/trigger_scanner.py` | Post-execution proactive detection |
| `app/policies/policy_loader.py` | Policy storage, loading, injection |
| `app/benchmarks.py` | Automated benchmarking with trends |

### Adding a New Tool

1. Create file in `app/tools/` with `@tool` decorator or `BaseTool` subclass
2. Import and add to relevant agent(s) in `app/agents/`
3. Add SSRF protection if accessing external resources
4. Restrict file paths if writing files

### Adding a New Crew

1. Create `app/crews/my_crew.py` with `crew_started/completed/failed` reporting
2. Add `update_belief()` calls for belief state tracking
3. Add `load_relevant_policies()` for policy injection
4. Add `record_metric()` for benchmarking
5. Register in `Commander._run_crew()` and `ROUTING_PROMPT`
6. Create soul file in `app/souls/` and self-model in `app/self_awareness/self_model.py`
7. Add crew card to the Firebase dashboard
