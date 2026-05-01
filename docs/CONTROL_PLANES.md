# Control Planes Reference

## Request Path

User message → response delivery. Synchronous, latency-critical.

```
Signal CLI → forwarder.py → /signal/inbound → handle_task() → Commander.handle()
                                                                    │
                ┌───────────────────────────────────────────────────┤
                │                                                   │
        Introspective?                                       Special command?
        (identity/memory)                                    (learn/skills/fleet/...)
                │                                                   │
        Answer from                                          Deterministic
        system chronicle                                     response (no LLM)
                │                                                   │
                └───────────────────┐    ┌──────────────────────────┘
                                    │    │
                              ┌─────▼────▼─────┐
                              │  LLM Routing   │
                              │  (Opus/fast)   │
                              └───────┬────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                  │
              ┌─────▼───┐      ┌─────▼───┐       ┌─────▼───┐
              │Research │      │ Coding  │       │Writing │  ...
              │  Crew   │      │  Crew   │       │  Crew  │
              └─────┬───┘      └─────┬───┘       └─────┬───┘
                    │                │                  │
                    └────────────────┼──────────────────┘
                                     │
                              ┌──────▼──────┐
                              │  Vetting    │
                              │  (optional) │
                              └──────┬──────┘
                                     │
                              ┌──────▼──────┐
                              │Post-process │
                              │ + Signal    │
                              └─────────────┘
```

**Key files:** `app/agents/commander/orchestrator.py`, `app/agents/commander/routing.py`, `app/agents/commander/commands.py`, `app/crews/*.py`, `app/vetting.py`

### Deterministic command surface

`try_command()` in `commands.py` runs *before* LLM routing. Matched
input gets a deterministic answer with no LLM call — both faster and
unhallucinable. Authoritative list (grep `commands.py` for the
canonical patterns):

| Command(s) | Action | Notes |
|---|---|---|
| `mode <name>` | Switch LLM runtime mode (free/budget/balanced/quality/insane/anthropic) | See LLM_SUBSYSTEM §16.1 |
| `pin <model> to <role>` / `unpin` | Hand-pin LLM for role | See LLM_SUBSYSTEM §16.2 |
| `refresh catalog` | Force LLM catalog rebuild | See LLM_SUBSYSTEM §16.3 |
| `learn <topic>` / `watch <topic>` | Queue self-improvement work | |
| `improve` / `evolve [deep]` | Trigger evolution session | See SELF_IMPROVEMENT §3 |
| `retrospective` / `benchmarks` / `policies` | Meta-cognitive runs | |
| `status` / `proposals` | Health + open governance | |
| `tickets` / `kanban` | Show ticket board | |
| `force recover` / `try harder` | Bypass refusal-loop budget | See RECOVERY_LOOP |
| **`workspaces` / `project list`** | **List available workspaces** | Workspace == project (DB term) |
| **`workspace` / `project` / `where am I`** | **Show current workspace + ticket counts** | Also matches "what is the current workspace" etc. |
| **`switch [to] {project\|workspace} [to] <name>`** | **Change active workspace** | Multi-word names OK ("eesti mets") |

The agent's commander backstory ([`app/souls/commander.md`](../app/souls/commander.md))
explicitly lists workspace commands so the LLM doesn't hallucinate
"UI-only" answers when a question falls through to `direct` routing.

## Control Path

Background jobs. Non-blocking, cooperative, interruptible.

```
┌─────────────────────────────────────────────────────────────┐
│                    APScheduler (cron)                        │
├─────────────────────────────────────────────────────────────┤
│  self_improve     │ 0 3 * * *   │ Learn from queue          │
│  code_audit       │ 0 */4 * * * │ Code quality scan         │
│  error_resolution │ */30 * * * *│ Auto-fix errors           │
│  evolution        │ 0 */6 * * * │ Prompt evolution          │
│  retrospective    │ 0 4 * * *   │ Performance review        │
│  benchmark        │ 0 5 * * *   │ Daily metrics             │
│  workspace_sync   │ 0 * * * *   │ Git backup                │
│  heartbeat        │ every 60s   │ Dashboard + anomalies     │
│  error_monitor    │ every 5 min │ errors.jsonl scan, anomalies → /cp/ops │
│  stuck_ticket_jan │ every 5 min │ Fail in_progress > 15 min │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│               Idle Scheduler (cooperative)                   │
├─────────────────────────────────────────────────────────────┤
│  Activates: 30s after last user task completes              │
│  Yields: immediately when user task arrives                 │
│  Kill switch: Firestore config/background_tasks.enabled     │
│  Jobs: 23 background tasks in round-robin order             │
│  Pause: 5s between jobs                                     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│             Firestore Listeners (real-time)                  │
├─────────────────────────────────────────────────────────────┤
│  mode_listener    │ LLM mode changes (local/hybrid/cloud)   │
│  kb_queue         │ Knowledge base document uploads (10s)   │
│  phil_queue       │ Philosophy document uploads (10s)        │
│  fiction_queue    │ Fiction library uploads (10s)             │
│  chat_inbox       │ Dashboard chat messages (10s)            │
│  bg_listener      │ Background tasks kill switch (15s)       │
└─────────────────────────────────────────────────────────────┘
```

**Key files:** `app/idle_scheduler.py`, `app/main.py` (lifespan), `app/firebase/listeners.py`

## Adaptation Path

Continuous improvement. Triggered by feedback, scheduled, or manual.

```
                    ┌─────────────┐
                    │  Feedback   │  (Signal reactions, corrections)
                    │  Pipeline   │
                    └──────┬──────┘
                           │ patterns
                    ┌──────▼──────┐
                    │Modification │  (Tier 1 auto / Tier 2 approval)
                    │  Engine     │
                    └──────┬──────┘
                           │
         ┌─────────────────┼─────────────────────┐
         │                 │                      │
  ┌──────▼──────┐   ┌─────▼──────┐   ┌──────────▼──────────┐
  │  Evolution  │   │  Training  │   │       ATLAS          │
  │  (prompt    │   │  (MLX LoRA │   │  (tool learning,     │
  │   variants) │   │   adapter) │   │   skill acquisition) │
  └──────┬──────┘   └──────┬─────┘   └──────────┬──────────┘
         │                 │                      │
         └─────────────────┼──────────────────────┘
                           │
                    ┌──────▼──────┐
                    │ Governance  │  (unified safety/quality/regression gates)
                    │   Gates     │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Promotion  │  (prompt version, adapter deploy, skill activate)
                    │  + Manifest │
                    └─────────────┘
```

**Key files:** `app/governance.py`, `app/feedback_pipeline.py`, `app/modification_engine.py`, `app/evolution.py`, `app/training_pipeline.py`, `app/atlas/`
