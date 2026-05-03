# Tool Registry — Phase 3: Forge integration

**Date:** 2026-05-03  •  **Status:** read-only bridge, periodic reconciler

Phase 3 closes the loop Forge was designed for. Today, Forge can
generate sandboxed tools and graduate them through SHADOW → CANARY →
ACTIVE, but **agents have no way to find a Forge-generated tool** —
`tool_search` only knows about `@register_tool`-annotated entries.

This phase bridges Forge's Postgres-backed tool catalog into the
in-memory `ToolRegistry`. The bridge is **read-only on Forge's side**:
Forge's state machine, schema, audit pipeline, and TIER_IMMUTABLE
files are untouched. The bridge READS from `forge.registry.list_tools`
and WRITES into our in-memory `ToolRegistry`.

---

## 1. Components

| Path | Role |
|------|------|
| `app/tool_registry/forge_bridge.py` | The whole bridge — `sync_forge_tools()`, the BaseTool wrapper around `forge.runtime.dispatcher.invoke_tool`, and the periodic `reconciliation_loop()` coroutine. |
| `app/tool_registry/registry.py` | Gains `replace_spec(spec)` and `unregister(name)` so the bridge can update tier on Forge promotion or remove KILLED tools. |
| `app/tool_registry/boot.py` | `boot_registry(sync_forge=True)` calls the bridge after the static `@register_tool` pass. |
| `app/main.py` | Spawns `reconciliation_loop()` via `asyncio.create_task` inside `lifespan`. |
| `tests/test_forge_bridge.py` | 16 tests covering status mapping, no-op-when-disabled, transition replays, BaseTool wrapper safety, async loop. |

---

## 2. Status → Tier mapping

Forge has 7 statuses (`DRAFT`, `QUARANTINED`, `SHADOW`, `CANARY`,
`ACTIVE`, `DEPRECATED`, `KILLED`). The bridge maps three to our
registry tiers:

| Forge ToolStatus | Registry Tier | Visible in `tool_search`? |
|------------------|---------------|---------------------------|
| `DRAFT` | (NOT bridged) | No — pre-validation, hidden |
| `QUARANTINED` | (NOT bridged) | No — failed audit |
| **`SHADOW`** | `Tier.SHADOW` | Only to crews authorized for SHADOW |
| **`CANARY`** | `Tier.CANARY` | Crews authorized for CANARY+ |
| **`ACTIVE`** | `Tier.PRODUCTION` | All production crews |
| `DEPRECATED` | (NOT bridged) | No — phasing out |
| `KILLED` | (NOT bridged) | No — explicitly killed |

The "NOT bridged" statuses still exist in Forge's DB; they're just
invisible to agent discovery. Same as their behavior pre-Phase-3.

---

## 3. The BaseTool wrapper

When the registry builds a Forge tool's instance, it constructs a
CrewAI `BaseTool` whose `_run` method calls
`forge.runtime.dispatcher.invoke_tool`. This preserves **every**
Forge safety property:

| Forge safety property | How the wrapper preserves it |
|----------------------|------------------------------|
| Three-layer killswitch | Wrapper goes through `dispatcher.invoke_tool` which checks killswitch first; refusal returned to agent as "invocation refused". |
| Per-tool budget | Same path — dispatcher enforces. |
| Capability audit | Same path — dispatcher logs `capabilities_declared` vs `capabilities_used`. |
| SHADOW-tier result discard | Wrapper detects `mode=SHADOW` or `shadow_result` in response → returns a stub message describing the SHADOW execution; **never exposes the actual result to the agent**. |
| Hash-chained audit log | Untouched — Forge logs every invocation regardless of caller. |

The wrapper is purely additive. An agent that calls a wrapped Forge
tool through the new registry sees exactly the same response shape
as a direct `dispatcher.invoke_tool` call.

---

## 4. Sync semantics

`sync_forge_tools()` does four things:

1. **Add** any Forge tool in `{SHADOW, CANARY, ACTIVE}` not yet in
   the registry → registers a `ToolSpec` at the matching tier.
2. **Update** any tool whose Forge status changed → calls
   `registry.replace_spec(spec)` which:
   - Replaces the spec at the new tier.
   - Drops the cached SINGLETON instance (so the next agent call
     gets a wrapper at the new tier — important when SHADOW → ACTIVE
     because SHADOW discards results, ACTIVE doesn't).
3. **Remove** any tool that was bridged before but no longer appears
   in `{SHADOW, CANARY, ACTIVE}` → calls `registry.unregister(name)`.
   This handles `DEPRECATED`/`KILLED` transitions and tools that
   simply got dropped from Forge's DB.
4. **No-op** when nothing changed (idempotent).

Identification: the bridge tags every Forge tool's
`spec.source_module = "app.forge.tools.<name>"` so the removal
detection can scope to "only forge-bridged entries" without touching
`@register_tool` entries.

---

## 5. The reconciliation loop

Forge has no built-in pub/sub for status transitions — promotions
happen via the `POST /api/forge/tools/{tool_id}/promote` endpoint;
demotions happen via the anomaly detector running on a cron. So the
bridge polls.

Cadence: 5 minutes (`_RECONCILE_INTERVAL_SEC = 300`). At 300s we
pay one Postgres round-trip per period when Forge is enabled and
zero work when it's not. Loop is `asyncio.to_thread(sync_forge_tools)`
so the synchronous DB call doesn't block the event loop.

Spawned in `lifespan` startup (after the gateway has its event loop
ready). Survives transient Postgres outages — each iteration is
independent and the bridge's failure modes are non-fatal.

Cancellation: standard asyncio. The loop catches `CancelledError`
and exits cleanly during graceful shutdown.

---

## 6. Failure modes (all non-fatal)

| Where | Failsafe |
|-------|---------|
| `TOOL_FORGE_ENABLED` unset / `0` | Bridge no-op, returns 0. Registry contains only `@register_tool` entries. |
| Postgres unreachable | Each `list_tools` call wrapped in try/except → logged, skipped. Loop continues to next iteration in 5 min. |
| `forge.registry` import fails | Bridge no-op (cached check via `_is_forge_enabled`). |
| Per-row `_row_to_spec` fails | Logged, skipped. Other rows in the same status batch still process. |
| `_make_forge_basetool` fails (crewai/pydantic missing) | Factory returns a no-op error string → agent sees "cannot build BaseTool" instead of crashing. |
| `dispatcher.invoke_tool` raises | Wrapper catches, returns `forge_bridge ERROR: ...`. Agent gets a string, can decide to retry or move on. |
| Reconciler iteration raises | Logged at debug, loop continues. |

---

## 7. Live verification (running container)

The container has 7 actual Forge-generated tools in `SHADOW` status
from prior development sessions. Booting the registry with the
bridge active:

```
$ docker exec crewai-team-gateway-1 python -c "
from app.tool_registry.boot import boot_registry
boot_registry(snapshot_to_postgres=False, index_to_chromadb=False, sync_forge=True)
"

INFO app.tool_registry.boot: tool_registry boot: imported 27 modules across 1 roots;
                              replayed 12 cached decorator specs; bridged 7 Forge tools;
                              total 19 (was 0).
```

Calling one through the wrapper:

```
$ docker exec crewai-team-gateway-1 python -c "
from app.tool_registry import ToolRegistry
inst = ToolRegistry.instance().build_instance('post_to_external_test')
print(inst._run(params={}))
"

forge_bridge: SHADOW-tier execution OK. Result was computed and logged
for operator review, but not returned to the agent (per SHADOW-tier
semantics). elapsed_ms=575, capability_used=http.internet.https_post
```

The dispatcher ran, logged the invocation, applied SHADOW-tier
result-discard, and returned a safe message to the calling code.

---

## 8. What the agent experience looks like

Phase 1b's `tool_search` already supports SHADOW tools — only crews
authorized for SHADOW see them. With Phase 3, those tools are
**actually loadable**:

```
Agent: tool_search(intent="post a JSON payload to an external HTTP endpoint")

Result: Found 1 matching tool.
  post_to_external_test  [shadow]  score=0.65
    capabilities: ['registers-tool']
    why: semantic match (d=0.35)
    Forge-generated tool (shadow tier). tool_id=...

Agent: load_tool(name="post_to_external_test")
Result: OK — 'post_to_external_test' loaded. Schema announced next step.

Agent: post_to_external_test(params={"url": "https://example.com", "body": {...}})
Result: forge_bridge: SHADOW-tier execution OK. Result was computed
        and logged for operator review, but not returned to the agent
        (per SHADOW-tier semantics).
```

The agent gets the tool through standard registry plumbing; Forge's
safety properties are preserved end-to-end.

---

## 9. What's NOT in Phase 3

Deferred to later phases or operator workflow:

* **Real-time tier-change hooks.** The bridge polls every 5 min.
  If you promote a SHADOW tool to CANARY via the Forge UI, agents
  see the new tier within 5 min. Phase 4 may add a webhook from
  `POST /api/forge/tools/{tool_id}/promote` for instant propagation.
* **Capability tag enrichment.** Forge tools get tagged with
  `registers-tool` (the closest fit in our current vocabulary).
  Phase 4 may extend the vocabulary with Forge-specific tags
  (`http-call`, `fs-read`, etc) so semantic search can rank them
  better.
* **Forge composition audit integration.** Forge has its own audit
  for combinations of generated tools (`app/forge/composition.py`).
  When an agent loads multiple Forge tools mid-task, that audit
  doesn't currently fire. Phase 4 will wire it in.
* **PER_AGENT-lifecycle Forge tools.** Some Forge tools may be
  agent-scoped (e.g., per-collection memory shapes). Currently the
  bridge treats every Forge tool as `Lifecycle.SINGLETON`. Phase 4
  may extend.

---

## 10. Migration progress

| Phase | What | Status |
|-------|------|--------|
| 0 | Spike + measurement + LoadableAgent prototype | DONE |
| 1a | Registry foundation + 11 tools annotated + `/cp/tools` endpoint | DONE (#39) |
| 1b | `tool_search` discovery primitive + 4-layer defense | DONE (#40) |
| 1c | Empirical cache-cost gate; verdict GO @ 33.4% | DONE (#41) |
| 2 | LoadableAgent opt-in pilot on introspector | DONE (#42) |
| **3** | **Forge → registry bridge + reconciler** | **THIS PR** |
| 4 | Production agent migrations (one/week) | After 3 |
| 5 | Drop legacy factories | After 4 |
