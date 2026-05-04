# System State — Phase 5.1

A read-side service that returns deployment context. Single source of
truth for "what does the system look like right now?" across:

* Routing decisions (Phase 5.2 — Commander grounds itself in factual
  state instead of inferring from conversation history).
* Change-request review (Phase 5.3 — React UI shows deployment
  context next to the proposed diff so operators can decide).
* Agent reasoning (any agent can call `get_system_state()` before
  acting on potentially-stale conversation context).

Shipped after the 2026-05-04 PIM incident exposed a class of bug
where Commander LLM keeps refusing to dispatch a crew because
conversation history shows past failures, without checking whether
those failures have been fixed. The fix is in two parts: this
package (state-as-truth, Phase 5.1) and the routing override that
consumes it (Phase 5.2).

---

## 1. What it returns

`get_system_state(window_hours=24, crew_runs_limit=10, use_cache=True)` →

```json
{
  "ts": "2026-05-04T16:09:41.323230+00:00",
  "window_hours": 24,
  "git": {
    "available": true,
    "source": "host bridge (source repo)",
    "head_sha": "971d9f37...",
    "head_message": "fix(agents): add missing optional_tool_group import",
    "head_age_min": 22,
    "head_committed_at": "2026-05-04T13:42:00Z",
    "uncommitted_changes": false,
    "files_changed_last_24h": ["app/agents/pim_agent.py", "app/agents/devops_agent.py", ...]
  },
  "gateway": {
    "available": true,
    "uptime_min": 88,
    "started_at": "2026-05-04T14:41:32Z"
  },
  "tier_immutable": {"available": true, "count": 111},
  "tools": {"available": true, "count": 20, "names": [...]},
  "recent_crew_runs": {
    "available": true,
    "by_crew": {
      "pim":      [{"ts": "...", "ok": true, "duration_s": 1.8, "task_id": "..."}, ...],
      "coding":   [{"ts": "...", "ok": false, "error": "...", "duration_s": 245}],
      "research": [...]
    },
    "buffer_sizes": {"pim": 12, "coding": 3, ...}
  }
}
```

**Critical convention**: every section has an `available: bool`.
Consumers MUST check it before reading other fields — sources can
be transiently unreachable and we degrade gracefully rather than
fail.

---

## 2. Where the data comes from

| Section | Sources tried (in order) | Fallback |
|---------|--------------------------|----------|
| `git` | host bridge (source repo) → in-container `subprocess /app` → `subprocess /app/workspace` (workspace state — clearly labeled) → `BUILD_SHA` env | `available: false` |
| `gateway` | `/proc/1/stat` for PID 1 start time + `/proc/uptime` for boot epoch | `available: false` (non-Linux) |
| `tier_immutable` | `app.auto_deployer.TIER_IMMUTABLE` | `available: false` |
| `tools` | `app.tool_registry.ToolRegistry.instance()` | `available: false` |
| `recent_crew_runs` | In-memory ring buffer (50 per crew, populated by `base_crew.run_single_agent_crew`) | `available: false` (gateway just restarted, buffer empty) |

The git source resolution is intentional. Inside the container,
`/app` is the deployed Python tree without `.git`; `/app/workspace`
has its own `.git` for workspace state (skill files, journals) but
that's not the source-repo state. The host bridge (`bridge.execute(['git', ...], working_dir=HOST_REPO_PATH)`) gives source-repo state when reachable.

`HOST_REPO_PATH` defaults to `/Users/andrus/BotArmy/crewai-team`
and is overridable via `HOST_REPO_PATH` env var.

---

## 3. The crew-run ring buffer

In-memory (`app/system_state/crew_runs.py`). Bounded at 50 entries
per crew via `collections.deque(maxlen=50)`. Older entries fall off.

Tap-in: `app/crews/base_crew.py::run_single_agent_crew` calls
`record_crew_run(crew_name, ok=True/False, error=..., duration_s=...,
task_id=...)` on both success and failure paths. Wrapped in
try/except — observational telemetry must never break the request
path.

Buffer empties on gateway restart. The Phase 5.2 routing-override
logic correctly interprets "no recent runs visible" as "try the
crew" rather than "the crew is broken."

Why in-memory not persisted: minutes-to-hours scope only. The
journal (`workspace/self_awareness_data/journal/JOURNAL.jsonl`)
already captures every crew run with full context; this is the
fast lookup layer in front of it. The two are complementary.

---

## 4. Caching

Cache key: `(window_hours, crew_runs_limit)`. TTL: 5 seconds.
Different parameters get different cache entries; same parameters
hit the cache. Phase 5.2's routing layer can call
`get_system_state()` freely on every routing decision; the actual
work runs ~once every 5 seconds.

`use_cache=False` forces a fresh read.

`reset_cache_for_tests()` clears the entire cache (test helper).

---

## 5. Public surfaces

### 5.1 Agent tool (`get_system_state`)

Registered in `app/tools/system_state_tool.py` with capability
`reads-deployment-state` (new in `app/tool_registry/capabilities.py`,
under category `observability`).

Description steers agents to call this **before assuming a bug
exists or a crew is broken**. Example pattern (in the tool's
own description):

```python
state = get_system_state()
pim_runs = state['recent_crew_runs']['by_crew'].get('pim', [])
if any(r['ok'] for r in pim_runs[:3]):
    # PIM has succeeded recently — the 'broken' claim is stale.
    ...
```

### 5.2 Control plane endpoint

`GET /api/cp/system-state` — same auth as the rest of `/cp/`
(`require_gateway_auth` dep). Query params:

| Param | Default | Range | Effect |
|-------|---------|-------|--------|
| `window_hours` | 24 | 1–168 | Hours of history in `git.files_changed_last_24h` and `recent_crew_runs` |
| `use_cache` | `true` | bool | Pass `false` to force a fresh read (rarely needed; cache is short) |

Use cases:
* React control plane "system state" widget — operators see git
  head, gateway uptime, recent crew outcomes at a glance.
* Phase 5.3 React change-request review UI — shows deployment
  context alongside the proposed diff.
* Pre-deploy sanity check: `curl …/api/cp/system-state | jq`.

### 5.3 Programmatic API

```python
from app.system_state import get_system_state, record_crew_run

state = get_system_state(window_hours=12)
record_crew_run("custom_crew", ok=True, duration_s=2.4)
```

---

## 6. Failure modes (all non-fatal)

| Where | Behavior |
|-------|----------|
| Bridge not reachable | Falls back to `subprocess` git → workspace git → `BUILD_SHA` env |
| `/proc` unavailable (non-Linux) | `gateway.available=false`, other sections still populate |
| Tool registry not booted | `tools.available=false`, others fine |
| Crew-run buffer empty (post-restart) | `recent_crew_runs.by_crew={}` — interpreted by 5.2 as "try the crew" |
| Postgres / ChromaDB / external services | Not consulted at all — system_state is purely process-local plus git-via-bridge |
| `record_crew_run` called with bad input | Silent drop (try/except wraps the body) |

---

## 7. What this enables

Phase 5.1 is the foundation. The two consumers built on top:

* **Phase 5.2** — Commander routing fix. Reads
  `recent_crew_runs.by_crew[<crew>]`. If the crew has succeeded
  recently, override "X is broken" hallucinations and force an
  actual dispatch. If it hasn't run since the alleged failure,
  also force a dispatch — let the real error or success surface.

* **Phase 5.3** — Change request system. Reads `git.head_sha` to
  branch from the right base; reads
  `tier_immutable.count` to validate the count hasn't drifted;
  reads `tools.count` to surface in the React review UI.

---

## 8. What's NOT in Phase 5.1

* **Routing-prompt injection.** Auto-injecting state into the
  Commander's routing prompt is Phase 5.2.
* **Persistent crew-run history.** Buffer is in-memory only;
  journal already persists. Don't double-store.
* **Real-time pub/sub of state changes.** Cache + 5s TTL is the
  current sync model. Adequate for routing.
* **Multi-process state sharing.** Single-gateway deployment.
  Multi-pod K8s deployments would need a Postgres-backed snapshot;
  not in scope here.

---

## 9. Test surface (`tests/test_system_state.py`)

21 tests across:

* Crew-run buffer — record/read, newest-first ordering, bounded
  eviction (push 100 → keep 50), empty-crew-string drop, never
  raises on bad input.
* State composition — all sections present, every section has
  `available`, window_hours param, caching, cache bypass, degrades
  when git unavailable, crew runs flow through.
* Agent tool — registered with right capability, builds, `_run`
  returns valid JSON.
* HTTP endpoint — 200, expected shape, query params (window_hours,
  use_cache), bound validation (422 for out-of-range).
* Vocabulary — `reads-deployment-state` is in
  `capabilities.py` under `observability` category.
