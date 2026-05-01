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

### Workspace auto-detection (sticky-user pick + 👍/👎 ask)

Beyond the explicit commands, every inbound Signal message runs through
``project_isolation.detect_project(text)`` — keyword-based
substring matching against ``PROJECT_KEYWORDS`` (e.g. ``"estonia"`` /
``"baltic"`` / ``"event"`` → PLG). Pre-2026-05-02 a positive detection
called ``get_projects().switch(...)`` silently. That overrode the
user's explicit Signal-command picks on every subsequent message
(forest-related keywords like ``"estonia"`` matched PLG, so any
work in **Eesti mets** got mis-attributed to PLG immediately).

The current behaviour has three modes, gated on
``ProjectManager._active_project_source``:

| State | Detection result | Action |
|---|---|---|
| **No explicit pick yet** (`source is None`) | matches a project | **Auto-switch** (`source="auto"`) — seed the session |
| **Sticky-user pick** (`source == "user"`) | matches the *current* project | No action |
| **Sticky-user pick** | matches a *different* project | **Ask via Signal** — see below |
| (any) | no detection | No action |

The **ask flow** (added 2026-05-02 — [`app/workspace_switch_proposals.py`](../app/workspace_switch_proposals.py))
sends:

> 💡 This message looks like it might belong to *PLG* (currently on *Eesti mets*).
>
> React 👍 to switch to *PLG*, or 👎 to stay on *Eesti mets*.

Reaction routing in `main.py` (same dispatcher as `proposals` / `human_gate`):

- **👍** → `accept(proposal_id)` → `switch(name, source="user")` (the new pick is also sticky)
- **👎** → `decline(proposal_id)` → suppresses re-asking the same `(sender, detected_name)` for 24 h
- **No reaction** → `expire_stale` background job marks pending entries expired after 30 min

Anti-spam: `has_recent_decision(detected_name, sender)` short-circuits
new asks while a pending or recently-declined entry exists for the
same pair.

**`ProjectManager.switch(name, *, source=...)`** is the single source
of truth for stickiness. Call sites:

| Caller | `source` |
|---|---|
| Signal `switch workspace ...` command | `"user"` (default) |
| Dashboard / API project-switch | `"user"` |
| `main.py` auto-detect (no explicit pick yet) | `"auto"` |
| `workspace_switch_proposals.accept()` (after 👍) | `"user"` |

A `"user"` value latches `_active_project_source` so subsequent
`source="auto"` calls return without changing state — guaranteeing
the auto-detector can never silently override an explicit pick.

**State persistence**: `_active_project_id` and `_active_project_source`
are in-memory (singleton `ProjectManager`). They reset to `None` on
gateway restart, at which point auto-detect can seed the session
again. *Follow-up: persist to DB so the user doesn't fall back to
`default` on every restart.*

**Operator queries** (when something looks misrouted):

```bash
# Who's the current pick and how was it set?
docker exec gateway python -c \
  "from app.control_plane.projects import get_projects; pm = get_projects(); \
   print(pm.get_active_project_id(), pm._active_project_source)"

# Pending workspace asks
cat /app/workspace/workspace_switch_proposals.json | jq '.[]|select(.decision=="pending")'

# Was anything misrouted? Audit log by project, recent window:
psql -c "SELECT p.name, COUNT(*) FROM control_plane.audit_log a
         LEFT JOIN control_plane.projects p ON a.project_id=p.id
         WHERE timestamp >= NOW() - INTERVAL '24 hours'
         GROUP BY p.name ORDER BY MIN(timestamp);"
```

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
