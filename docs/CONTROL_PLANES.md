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

**Specialist tools worth knowing about** (full inventory lives in
each crew's agent file under `app/agents/`):

| Tool | Crew | Reference |
|---|---|---|
| `gee_run_script` (Google Earth Engine) | coder | [`docs/GEE.md`](GEE.md) |
| `forge_register_tool` (sandboxed tool generation) | coder | [`docs/FORGE.md`](FORGE.md) |
| `tool_search` (capability-typed registry discovery) | coder + writer | [`docs/TOOL_REGISTRY.md`](TOOL_REGISTRY.md) |
| `pdf_compose` + `signal_send_attachment` (PDF render → Signal delivery) | coder + writer | [`docs/PDF_AND_SIGNAL_DELIVERY.md`](PDF_AND_SIGNAL_DELIVERY.md) |
| `rank_emails` / `check_email` (PIM) | pim | inline in [`app/tools/email_tools.py`](../app/tools/email_tools.py) |
| `recovery_loop` strategies (refusal handling) | all | [`docs/RECOVERY_LOOP.md`](RECOVERY_LOOP.md) |

### Tool Registry control surface

`/api/cp/tools/*` — read-only HTTP browser of the capability-typed
tool registry. Auth via the same `require_gateway_auth` dep as the
rest of the control plane.

| Route | Purpose |
|---|---|
| `GET /api/cp/tools` | List tools. Query params: `capability`, `tier`, `loadable_only`, `workspace`. |
| `GET /api/cp/tools/{name}` | Full detail for one tool. 404 if absent. |
| `GET /api/cp/tools/stats` | Counts by tier + lifecycle + capability coverage. |
| `GET /api/cp/tools/capabilities` | Bounded vocabulary, grouped by category (`data`, `knowledge`, `memory`, `compute`, `delivery`, `governance`). |
| `GET /api/cp/tools/drift` | Description-hash + tier drift vs prior Postgres snapshot. |
| `GET /api/cp/tools/flags` | Phase 4d diagnostic — which migrated agents are running on legacy vs LoadableAgent right now. Reads the same env vars that the agent factories themselves consult; what it shows IS what the agents do. |

Use cases: React control plane "tool catalog" + "agent flag matrix" widgets; pre-deploy sanity checks (`curl … | jq`); post-incident triage (which agents were on which path when the regression happened). Full reference: [`docs/TOOL_REGISTRY.md`](TOOL_REGISTRY.md).

### Watchdog (request-path timeouts)

Every `handle_task` call runs under a progressive timeout. A single
hard wall-clock timeout couldn't distinguish "still making progress"
from "stalled" — d=9 deep research legitimately takes 15–30 min, but a
stuck retry-loop that cycles for 45 min is pure waste. The watchdog
keeps a task alive as long as it's *demonstrably* doing useful work
and kills it as soon as that signal goes flat.

**Phase flow** (in [`app/main.py`](../app/main.py), around `handle_task` /
`_evaluate_stall`):

| Phase | Trigger | Action |
|---|---|---|
| **Soft checkpoint** @ 900s (15 min) | Task hasn't returned | Run `_evaluate_stall`. If alive, send a "still working" Signal note and enter extension mode |
| **Progress-gated extension** | Every 30s while extending | Run `_evaluate_stall`; finished → return; stalled → kill; otherwise wait another window |
| **Hard cap** @ 2700s (45 min) | Cap reached | Abandon thread; user gets a "hit hard cap" message |

`_evaluate_stall(task_id, elapsed_secs)` is a tiered read over three
process-wide heartbeats. **First match wins** — tiers are ordered by
strictness:

| # | Tier | Fires when | Threshold | Source heartbeat |
|---|---|---|---|---|
| 1 | `output-stall` | A tool recorded a partial, then nothing for ≥ 5 min | 300s | `task_progress.seconds_since_last_output_progress` |
| 2 | `crew-zero-progress` | Elapsed ≥ 10 min, **never** any partial, AND tool-activity heartbeat is stale (≥ 4 min, or never seen) | 600s + 240s tool-quiet | `tools_timeout.seconds_since_last_tool_activity` |
| 3 | `zero-output` | Elapsed ≥ 20 min, **never** any partial (regardless of tool state) | 1200s | `task_progress.output_progress_count == 0` |
| 4 | `llm-stall` | LLM heartbeat stale ≥ 4 min | 240s | `rate_throttle.seconds_since_last_llm_activity` |

**Heartbeats** ticked from three independent sources, each detecting a
different failure mode:

| Heartbeat | Ticked by | Catches | Misses |
|---|---|---|---|
| Output progress | Tools that call `record_output_progress` (research orchestrator, search-result tools) | Slow draining, hallucination loops with no deliverable | Un-instrumented tools — falls back to tool-activity |
| Tool activity | Every `BaseTool.run` entry/exit, via the `tools_timeout` monkey-patch | Stuck retry-loops (provider 401/429 hammering — no tools fire), hung MCP / external calls (single tool wedged > 4 min) | A genuinely slow single-call (heartbeat reads stale during the call) — tier-2's 240s threshold leaves margin past the longest default tool budget (180s) |
| LLM activity | Every LiteLLM completion (success **or** failure) | Hung threads, provider-side outages | Retry-loops — failures still tick the heartbeat warm; tier 2 is the answer |

**Tier 2 (`crew-zero-progress`) was added 2026-05-02** after a
20-minute coding-crew dispatch produced zero output, masked by an
Anthropic credit-exhausted retry-loop that kept the LLM heartbeat warm
the whole time. The tool-activity gate is what distinguishes "LLM
genuinely cycling through tools" from "LLM stuck without external
action." Strictly tighter than tier 3 — the 1200s zero-output
backstop remains in place for the slow-but-cycling case (tools firing
every minute but no deliverable, e.g. sequential GEE `getInfo()`
calls).

**Inertness guarantees** (the watchdog must not kill legitimate work):

* High-difficulty research that ticks tool-activity every minute or two
  → tier 2 stays inert; tier 3's 20-min budget is the relevant gate.
* Tasks that emit partials → tier 2 never reachable (tier 1 supersedes
  on the output-progress branch).
* Sub-10-min tasks → tier 2 never reachable.

**Operator messaging** (kill-message branches in `handle_task`):

| Tier | User-facing message style |
|---|---|
| `output-stall` | "Stopped producing partial results — delivering what's been streamed so far" |
| `crew-zero-progress` | "10+ min, no partial, no tool activity — likely stuck retry-loop or hung external call" |
| `zero-output` | "20+ min, no partial — researcher likely looping on a blocked source" |
| `llm-stall` | "No LLM activity for several minutes — provider outage or stuck retry" |

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

### Idle scheduler snapshot

`GET /api/cp/idle/jobs` (added 2026-05) returns a JSON snapshot of every
known idle job — `failure_count`, `in_cooldown`, `cooldown_until_ts`,
`seconds_since_last_success`, `seconds_since_last_failure`,
`currently_running` — plus the inbound DLQ backend (memory or Redis)
and depth. Read-only; calling it never affects scheduling. Closes the
prior gap where ~100 background jobs ran invisibly to the dashboard.

```bash
curl -s -H "Authorization: Bearer $GATEWAY_SECRET" \
     http://localhost:8765/api/cp/idle/jobs | jq .
# { "scheduler_enabled": true,
#   "scheduler_idle": false,
#   "jobs": { "evolution": {"failure_count":0, "in_cooldown":false, ...}, ... },
#   "inbound_dlq": {"backend":"memory","depth":0, ...} }
```

Tunables (lifted to module constants in `app/idle_scheduler.py` so
operators don't have to read the body of `_run_single_job`):

| Constant | Default | Purpose |
|---|---|---|
| `IDLE_DELAY_SECONDS` | 30 | Wait after last user task before background work resumes |
| `INTER_JOB_PAUSE_SECONDS` | 2 | Cool-down between consecutive idle-job iterations |
| `MAX_CONSECUTIVE_FAILURES` | 3 | Skip-cooldown trigger |
| `JOB_COOLDOWN_AFTER_FAILURES_S` | 3600 | How long a failing job is parked |
| `TRAINING_LOOP_INTERVAL_S` | 3600 | Cadence of the training-pipeline loop |

### Inbound load-shed DLQ

When `handle_task` is at capacity (inflight ≥ `load_shed_threshold`),
the message is buffered in `app/dead_letter_inbound.py` instead of
dropped. The `dlq-drain` LIGHT idle job replays buffered messages
when capacity returns; messages older than 30 min are dropped.

Two backends, swappable at module-import time:

* **In-process deque** (default) — bounded at 200 messages, lost on
  pod restart. Correct for single-pod deploys.
* **Redis-backed list** — opt-in via `REDIS_DLQ_URL` (e.g.
  `redis://botarmy-redis:6379/0`). Multi-pod deployments share one
  queue; pod restarts no longer lose buffered messages. Falls back
  to in-process silently if Redis is unreachable.

Backend status is surfaced via `/api/cp/idle/jobs.inbound_dlq`.

## Gateway HTTP auth (perimeter)

`/api/cp/*` and `/epistemic/*` mutating routes attach a router-level
FastAPI dependency from `app/control_plane/auth_dep.py`:

```python
router = APIRouter(prefix="/api/cp",
                   dependencies=[Depends(require_gateway_auth)])
```

The dependency reads `GATEWAY_AUTH_REQUIRED` (a typed Pydantic field
in `app/config.py`, also readable via `os.environ.get`):

| State | Behaviour |
|---|---|
| Unset / `0` / `false` (laptop dev) | Pass-through; preserves the local-dev UX |
| `1` / `true` / `yes` / `on` (K8s default) | `Authorization: Bearer <gateway_secret>` required, validated via constant-time `hmac.compare_digest()` |
| Enforced + secret empty | 503 `auth misconfigured` (operator error, not request error) |

The React dashboard reads `VITE_GATEWAY_SECRET` at build time and
attaches the header to every request. Internal Python callers of
`record_override()`, `evaluate_promotion()`, etc., DO NOT pass through
the dependency — the auth boundary is HTTP, not function calls.

**Operational defaults**: Helm sets `gateway.authRequired: "true"` so
K8s deployments are always authenticated. See
[`deploy/HARDENING.md`](../deploy/HARDENING.md) §1 for the full
rollout playbook.

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
