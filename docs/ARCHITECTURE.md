# System Architecture

## Three Control Planes

The system operates across three parallel control planes:

| Plane | Trigger | Latency | Entry Point |
|-------|---------|---------|-------------|
| **Request** | User Signal message | Synchronous (~1-20s) | `main.py:receive_signal()` → `Commander.handle()` |
| **Control** | Cron + idle timer | Background | APScheduler + `idle_scheduler.py` |
| **Adaptation** | Feedback patterns / scheduled | Background | evolution, modification, training, ATLAS |

These planes are independent: user requests never block background work, and background work yields when a user request arrives.

## System Lifecycle States

```
INIT → RESTORE → SCHEDULE → INFRASTRUCTURE → PARALLEL_INIT → STATE_RESTORE → LISTENERS → IDLE_SCHEDULER → KNOWLEDGE → READY → SHUTDOWN
```

| State | What happens | Key function |
|-------|-------------|-------------|
| **INIT** | Load config, validate safety constraints | `lifespan()` |
| **RESTORE** | Fetch workspace from git backup | `setup_workspace_repo()` |
| **SCHEDULE** | Register 8 APScheduler cron jobs + heartbeat | `scheduler.add_job()` |
| **INFRASTRUCTURE** | Ollama check, auditor, health monitor, hooks, projects | Various init calls |
| **PARALLEL_INIT** | Philosophy KB, system chronicle, system monitor (async) | `asyncio.gather()` |
| **STATE_RESTORE** | Clean stale tasks, read LLM mode from Firestore | `cleanup_stale_tasks()` |
| **LISTENERS** | Start 5 Firestore queue pollers | `start_*_poller()` |
| **IDLE_SCHEDULER** | Read kill switch, start cooperative job loop | `idle_scheduler.start()` |
| **KNOWLEDGE** | Init prompt registry, create version manifest | `init_registry()` |
| **READY** | Accept user requests | `uvicorn` running |
| **SHUTDOWN** | Stop scheduler, sync workspace, report offline | `lifespan()` exit |

## Request Processing Flow

```
ARRIVE → REACT → INTROSPECT → COMMANDS → ROUTE → DISPATCH → VET → POST_PROCESS → DELIVER
```

| State | Duration | What happens |
|-------|----------|-------------|
| **ARRIVE** | <1ms | Auth (HMAC), rate limit, sanitize input |
| **REACT** | 0ms (fire-and-forget) | Send 👀 emoji via Signal (non-blocking) |
| **INTROSPECT** | <1ms | Fuzzy keyword match for identity questions → answer from chronicle |
| **COMMANDS** | <50ms | 50+ deterministic command handlers (learn, skills, fleet, etc.) |
| **ROUTE** | 0-2.5s | Fast-path (instant/pattern) or LLM router (Opus with history + Mem0) |
| **DISPATCH** | 2-15s | Crew execution with parallel context injection, reflexion retry |
| **VET** | 0-5s | Risk-based vetting (none/schema/cheap/full by difficulty + tier) |
| **POST_PROCESS** | <10ms | Strip metadata, truncate for Signal, epistemic humility check |
| **DELIVER** | ~200ms | Send via Signal, attach .md if >1400 chars |

## Idle Scheduler Jobs

Jobs run cooperatively when no user tasks are active. Round-robin ordering with `should_yield()` check between each job.

| Job | What it does | Category |
|-----|-------------|----------|
| learn-queue | Process learning_queue.md topics | Learning |
| evolution | Run evolution session (max 5 iterations) | Adaptation |
| discover-topics | Auto-discover learning topics | Learning |
| retrospective | Performance meta-analysis crew | Learning |
| evolution-2 | Second evolution pass | Adaptation |
| improvement-scan | Code improvement scanning | Learning |
| feedback-aggregate | Aggregate reaction patterns | Adaptation |
| safety-health-check | Post-promotion health monitoring | Safety |
| modification-engine | Process triggered patterns into prompt changes | Adaptation |
| health-evaluate | Dimensional health scoring | Operations |
| version-snapshot | Create version manifest | Operations |
| personality-development | PDS session | Self-awareness |
| cogito-cycle | Metacognitive self-reflection | Self-awareness |
| self-knowledge-ingest | AST-based code → ChromaDB | Self-awareness |
| training-curate | Curate interaction data for MLX LoRA | Adaptation |
| fiction-ingest | Ingest fiction inspiration library | Learning |
| map-elites-maintain | MAP-Elites grid persistence | Adaptation |
| island-evolution | Population-based island evolution | Adaptation |
| parallel-evolution | Diverse archive exploration | Adaptation |
| atlas-competence-sync | Sync skill competence levels | Learning |
| atlas-stale-check | Detect stale skills | Learning |
| system-monitor | Dashboard health report | Operations |
| tech-radar | Technology scouting | Learning |

## Modification Tiers

| Tier | Authorization | Examples | Rate Limit |
|------|-------------|----------|------------|
| **Tier 1** (auto) | Autonomous — no approval needed | Temperature tweak, example injection, simple prompt refinement | 10/day per role |
| **Tier 2** (approved) | Requires owner approval via Signal | Role strategy changes, complex prompt rewrites | 3/day per role |
| **Tier 3** (protected) | Cannot be modified by agents | Soul files, constitution, security modules, bridge code | N/A |

## Promotion Gates (Unified Governance)

All improvement systems (evolution, modification, training, ATLAS) pass through `app/governance.py`:

| Gate | Threshold | Behavior |
|------|-----------|----------|
| **Safety** | ≥ 0.95 | Hard veto — any safety regression blocks promotion |
| **Quality** | ≥ 0.70 | Minimum quality floor across all systems |
| **Regression** | ≤ 15% drop | No dimension can regress more than 15% from baseline |
| **Rate Limit** | ≤ 20/day | Prevents runaway promotion loops |

## Safety Boundaries

### Protected Files (`auto_deployer.py`)
Agents cannot modify: `sanitize.py`, `security.py`, `vetting.py`, `auto_deployer.py`, `config.py`, `main.py`, `signal_client.py`, all `souls/*.md`, all `self_awareness/*.py`, `Dockerfile`, `docker-compose.yml`, `firestore.rules`.

### Lifecycle Hooks (`lifecycle_hooks.py`)
Priority 0-1 hooks are **immutable** — registered at startup, cannot be overridden:
- `humanist_safety` (priority 0): Constitutional safety enforcement
- `block_dangerous` (priority 1): Block dangerous tool usage

### DGM Safety Constraint
Evaluation functions MUST be external to the entity being evaluated:
- Training pipeline: MLX Qwen adapter → judged by Claude Sonnet (different family)
- Evolution: Proposals → judged by independent `evo_critic` role
- Modification: DeepSeek hypothesis → evaluated by Sonnet sandbox

## Package Boundaries

```
app/request/     → Synchronous request path (Commander, crews, vetting)
app/control/     → Background scheduling (idle_scheduler, APScheduler)
app/adaptation/  → Improvement systems (governance, evolution, feedback, training)
```

These are facade packages providing clean import boundaries. All existing import paths continue working.
