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
