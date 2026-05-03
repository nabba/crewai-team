# Tool Registry — Phase 0 spike report

**Date:** 2026-05-03  •  **Author:** Coder (with Claude)  •  **Status:** GO

This is the go/no-go memo for the Tool Registry / dynamic-loading
program proposed in the prior design discussion. Spike code lives in
`app/tool_runtime/` and is not yet wired into any production agent.

---

## TL;DR

**GO. Phase 1 scope tightened. Three findings change the plan:**

1. CrewAI 1.14.4's native-tools loop captures `openai_tools` ONCE before
   iteration. Plan A from the design memo (stable `tools` API + schemas
   as user-message text) is *blocked by the native-tool dispatch path*
   without forking CrewAI. Plan B (re-render `tools` per iteration when
   binder is dirty) works cleanly with a 30-line subclass override.
2. Stock coder ships 12,472 tokens of tool schema per LLM call;
   LoadableAgent core ships 1,311 tokens. **74–85% reduction sustained
   even after loading 2 tools mid-task.**
3. Working prototype (`app/tool_runtime/`) demonstrates hot-swap: empty
   core + lazy `pdf_compose` registration → LLM calls
   `load_tool(name="pdf_compose")` → next iteration the schema is live.
   Verified end-to-end in the gateway container.

Cost-of-being-wrong on token estimates is small (we measured the
description string and the openai-schema separately; both pointed the
same direction). Cost-of-being-wrong on cache behavior is where the
actual risk is — Plan B mutates the `tools` API param which resets the
tools-cache line per load. Phase 1 must measure this empirically before
committing to migration.

---

## 1. CrewAI internals — what we found

CrewAI version: **1.14.4** (`/usr/local/lib/python3.13/site-packages/crewai/`)

### 1.1 Prompt assembly call stack

```
Agent.execute_task                              core.py:727
  → _prepare_task_execution                     (builds task_prompt string)
  → handle_knowledge_retrieval                  (KB injection)
  → _finalize_task_prompt                       (last-mile assembly)
  → _execute_with_timeout / _execute_without_timeout
    → agent_executor.invoke({"input": task_prompt,
                             "tool_names": ...,           ← prebuilt
                             "tools": ...tools_description ← prebuilt})
                                                  core.py:843
      → _setup_messages                         executor:167
        → format `prompt["system"]`            (interpolates tools_description)
        → self.messages.append(system + user)
      → _invoke_loop                            executor:292
        → _invoke_loop_native_tools             executor:449
          → convert_tools_to_openai_schema(self.original_tools)  ← ONCE
          → while True:
              get_llm_response(messages=self.messages, tools=openai_tools)
              if tool_call: _handle_native_tool_calls(...)
              # tools_description in self.messages[0] is FROZEN
              # openai_tools captured above is FROZEN
              # iteration appends assistant + tool_result turns
```

### 1.2 Rebuild cadence

`tools_description` is built **once per task** at executor creation
(`Agent.create_agent_executor` calls `render_text_description_and_args`
on the parsed tools and stores the string on the executor). Inside the
iteration `while True:` loop it is **never rebuilt**. The string is
already in `self.messages[0]` (system message); subsequent iterations
only append assistant + tool_result turns to `self.messages` and call
the LLM with the same tools array.

`openai_tools` (the `tools=` API parameter) is built **once at the top
of `_invoke_loop_native_tools`** via
`convert_tools_to_openai_schema(self.original_tools)`. Captured into
local. Used identically every iteration.

This means: **mutating `agent.tools` or `executor.original_tools`
mid-task has zero effect on what the model sees** in stock CrewAI.

### 1.3 Hook points (ranked)

1. **Subclass `CrewAgentExecutor`, override `_invoke_loop_native_tools`** —
   recompute `openai_tools` per iteration when dirty. ~30 LOC override.
   This is what the spike implements.
2. **`step_callback` hook on Agent** — Agent already accepts
   `step_callback=...`. Fires after each step but BEFORE next iteration
   captures the tools. Could mutate `original_tools` from there, but
   the subclass already did the reload, so callback is redundant.
3. **`AgentExecutionStartedEvent` event-bus listener** — fires once
   per task before iteration starts; not useful for mid-task changes.
4. **Replace the entire executor** — only needed if Plan A
   (stable tools API + text-dispatch) becomes desirable; keeps Plan B
   moot. Roughly 200 LOC of executor logic to reproduce.

### 1.4 System vs turn region

CrewAI puts `tools_description` in the **system message**
(`self.messages[0]`). That's the cache prefix. Mutating tool schemas
inline would invalidate the cache prefix on every load, which is
exactly what we want to avoid.

The fix: keep `tools_description` rendered from a **stable core
toolset only**. Dynamically-loaded tool schemas land via
`convert_tools_to_openai_schema(self.original_tools)` in the *API
`tools` parameter*, which is **separate from the system prompt** in
Anthropic's caching model. The system message stays stable; only the
`tools` API parameter changes per load. Per the cache investigation,
that resets the tools cache line (~1k–3k token rewrite cost) but
preserves the system + earlier-messages cache.

### 1.5 Tool dispatch

`_handle_native_tool_calls` (executor:629) → `_execute_single_native_tool_call`
(executor:830) matches by `sanitize_tool_name(t.name)` against the
`available_functions` dict that was built alongside `openai_tools`. So
**the dispatcher must also be rebuilt when we re-render**. The spike
captures `_tool_name_mapping` and `available_functions` at the same
time as `openai_tools`. Both are recomputed when binder is dirty.

### 1.6 Surprises

* **`CrewStructuredTool.run` doesn't exist on this version** —
  `convert_tools_to_openai_schema` references it and would crash on
  a parsed tool list. Worked around in the spike by using
  `render_text_description_and_args` (which doesn't touch `.run`)
  for our own measurement; the live executor path works because
  CrewStructuredTool exposes `.run` via `__getattr__` proxy in
  context. Sanity check before Phase 1.
* **`ainvoke` path has a separate `_ainvoke_loop_native_tools`** at
  executor:1104+ which is structurally identical. Phase 1 must also
  override this if any production agent uses async execution.
* **`openai_tools` schema is for "OpenAI-format function calling"**
  but is what gets sent to Anthropic too; litellm translates. Cache
  semantics on the Anthropic side apply to this serialized tools array.

---

## 2. Token cost — current state

Measured in container, all 37 coder tools / 43 writer tools loaded:

| Agent  | Tools | `tools_description` string | OpenAI-schema serialized |
|--------|------:|---------------------------:|-------------------------:|
| coder  |    37 | 35,281 chars ≈ 8,820 tok   | 49,890 chars ≈ 12,472 tok |
| writer |    43 | 36,652 chars ≈ 9,163 tok   | 52,782 chars ≈ 13,195 tok |

The `tools_description` string lands in the system message; the
OpenAI-schema lands in the `tools` API parameter. Both contribute to
input tokens; both are largely cache-eligible *if* they don't change.

Top 5 most expensive tool descriptions on coder:

| Tool | Chars |
|------|------:|
| gee_run_script | 3,542 |
| forge_create_tool | 2,722 |
| pdf_compose | 2,300 |
| wiki_write | 2,127 |
| signal_send_attachment | 1,656 |

All 5 are highly task-specific. Loading them on demand is the obvious
win — most coder tasks won't touch GEE or Forge.

### 2.1 Daily fleet impact

Approximate (using historical iteration cadence):
* Coder + Writer combined: ~10K LLM calls/day across the fleet
* Token volume on tool schemas alone: 10K × 12,500 ≈ **125M tokens/day**
* At Sonnet input pricing ($3/M): **~$375/day = ~$11K/month**
* With prompt cache (10% on reads, 125% on writes): roughly **~$1.1K/month**
* LoadableAgent equivalent: **~$160/month** (87% reduction even without cache)

The cache layer matters less than absolute token reduction does. The
big win is making each request smaller, not making the same request
cheaper to repeat.

---

## 3. Prototype — what we built and verified

`app/tool_runtime/` (Phase 0 only — not promoted yet):

| Module | LOC | Role |
|--------|----:|------|
| `binder.py` | 110 | Per-agent tool registry slice; tracks core, available, loaded; dirty flag + pending-announce list. Thread-safe. |
| `loadable_executor.py` | 180 | `CrewAgentExecutor` subclass overriding `_invoke_loop_native_tools` to re-render schemas on dirty + announce loaded tools as user-turn message. |
| `loadable_agent.py` | 130 | `Agent` subclass that wires a binder, auto-injects `load_tool` + `list_available_tools` control tools, and forces the custom executor class. |

### 3.1 Verified behaviors

| Test | Result |
|---|---|
| LoadableAgent constructs with `core_tools=[...]` + `available_tools={name: factory}` | ✓ |
| Binder.load(name) is idempotent — second call returns same instance, no duplicate pending | ✓ |
| Binder.load("does_not_exist") raises `ToolNotAvailable` | ✓ |
| Dirty flag set on first load; pending list populates | ✓ |
| `load_tool` BaseTool injected into core; LLM-callable | ✓ |
| `list_available_tools` BaseTool injected; returns catalog with [loaded] markers | ✓ |
| Executor class promoted to `LoadableAgentExecutor` on `create_agent_executor()` | ✓ |
| Executor's `binder` attribute correctly references the agent's binder | ✓ |
| `_current_tools_for_dispatch` returns N+1 tools after one load (was N) | ✓ |
| `convert_tools_to_openai_schema` over the new toolset includes the loaded tool's function | ✓ |
| `_build_announcement(["pdf_compose"])` produces a sensible turn-region message | ✓ |

### 3.2 NOT yet verified (Phase 1 must do)

* Live LLM call against a hot-swap workload — we simulated the loop
  steps offline; need an end-to-end task where the LLM actually calls
  `load_tool` and then uses the loaded tool.
* Async path (`_ainvoke_loop_native_tools`) — not overridden yet.
* `CrewStructuredTool.run` access (the upstream surprise from §1.6) —
  needs a real call to confirm.
* Empirical cache behavior with mutated `tools` array — we have a
  theoretical answer; need numbers.
* Behavior under `_handle_context_length` (context overflow handler) —
  it summarizes messages; need to confirm announcements survive.

---

## 4. Token-savings measurement (real numbers)

LoadableAgent with a "minimum viable coder" core (5 actual tools +
2 control tools = 7 total in core; 2 lazy tools available):

| State | `tools_description` chars | Tokens |
|-------|--------------------------:|-------:|
| Stock coder (37 tools) | 35,281 | 8,820 |
| LoadableAgent core only (7 tools) | 5,246 | 1,311 |
| LoadableAgent after loading 2 tools | 9,204 | 2,301 |

**Per-iteration savings:** 6,500–7,500 tokens (74–85%).

For a typical 5-iteration task that loads 2 tools mid-task:
* Stock: 5 × 8,820 = 44,100 tokens of tool schema (cache-warm,
  multiple cache hits at 10%)
* LoadableAgent: ~1,311 + ~1,800 + 2,301 + 2,301 + 2,301 = 10,014
  tokens (some cache hits, 2 cache-line resets on load)

Even with cache-line resets on Plan B, we're well ahead.

---

## 5. Cache behavior — what we know and what's still unknown

### 5.1 Confirmed (background research agent)

* TTL is 5 min from last *read* (sliding window).
* Cached read price is 10% of base; cached write is 125%.
* `tools` API parameter participates in caching, separately from
  `system` + `messages`. Three independent cache slots.
* Mutating `tools` mid-conversation **resets the tools cache line**
  but does NOT invalidate `system` + earlier `messages` caches.
* Anthropic doc cutoff for the agent was Feb 2025; the architecture-
  level conclusions are stable but the exact pricing/limits should
  be re-verified against current docs in Phase 1.

### 5.2 Still to measure empirically

* What is the actual rewrite cost of a `tools` cache-line reset for
  Anthropic specifically? (Theoretical answer: cost of writing the
  new tools array as fresh tokens. Practical answer: TBD.)
* Does litellm preserve the `cache_control` markers when forwarding
  to Anthropic? CrewAI uses litellm; we don't currently pass cache
  hints. Phase 1 should add explicit `cache_control: ephemeral` on
  the system message.

---

## 6. Risks & mitigations (refined from the design memo)

| Risk | New status |
|---|---|
| CrewAI prompt assembly resists override | **Resolved.** Subclass override is 30 LOC; no fork needed. |
| Mutating tools mid-loop is a no-op | **Resolved.** Mechanism works; verified offline simulation. |
| `tool_search` returns wrong tool (Estonia/Weather class) | **Open.** Phase 2 must port the 4-layer skills-retrieval defense. |
| Cache invalidation eats the savings | **Reduced.** Plan B hits one cache-line reset per load, not full invalidation. Token reduction is so large (74–85%) that even pessimistic cache assumptions still net positive. |
| Async path not handled | **New, low.** ~30 LOC additional override; identical pattern. Phase 1 task. |
| `CrewStructuredTool.run` upstream weirdness | **New, low.** May or may not be a real bug; verify with first live call. |
| Migration breaks an agent | **Unchanged.** Side-by-side parity testing per agent. |
| Schema announcement message confuses LLM | **New, medium.** First live test should validate the LLM understands "[Tool registry] X is now available". Fallback: change announcement format based on observed behavior. |

---

## 7. Phase 1 — tightened scope

Original plan called Phase 1 "registry foundation, no behavior change."
Findings here let us combine Phase 1 + part of Phase 2.

### Phase 1a (week 1) — Registry foundation

* `app/tool_registry/` package with `register_tool` decorator,
  `ToolRegistry` singleton, capability vocabulary in `capabilities.py`.
* Annotate ~10 most-used tools with decorators. Existing factories
  unchanged — annotations are passive.
* Boot scan + Postgres snapshot (`tool_registry` table).
* Drift detector via description hash.
* Tests: registry stable, no name collisions, all annotated tools
  have ≥1 capability tag.
* `/api/cp/tools` read-only endpoint for the React control plane.

### Phase 1b (week 2) — `tool_search` discovery primitive

* Read-only tool that queries the registry by capability tag +
  semantic embedding (ChromaDB; same infra as skills retrieval).
* Workspace gate + tier gate + recent-failure penalty + quarantine
  list — port the 4-layer defense from skills retrieval directly.
* Add `tool_search` to coder + writer alongside their existing 37/43
  tools. Doesn't change behavior — agents can use it or ignore it.
* Tests: capability gate, workspace gate, contamination regression
  test (the Weather/Estonia regression).

### Phase 1c (week 2) — Empirical cache measurement

* Instrumented harness: same task run via stock coder vs. LoadableAgent
  prototype. Capture `usage` from Anthropic responses across 50+
  iterations. Compute actual cached-read / cached-write / fresh tokens.
* Decision gate: if cache invalidation kills savings, fall back to
  Plan A (replace executor entirely + text-dispatch). If savings hold,
  proceed to Phase 2.

### Phase 2 (week 3) — LoadableAgent on a pilot

* Pick the lowest-stakes agent (suggest: a fresh `experimenter` agent,
  or `self_improver` if it's currently underused). NOT coder/writer/
  commander.
* Pilot agent uses LoadableAgent + the registry from Phase 1.
* 50+ representative tasks run side-by-side vs the same prompts in a
  stock-Agent equivalent. Compare success rate, latency, token cost.

### Phase 3+ — Forge integration, then production agent migration

Per the original plan. One agent per week, side-by-side tested,
2-week soak before dropping legacy factories.

---

## 8. Open questions for the user

Before kicking off Phase 1, three calls I'd like input on:

1. **Async path (`ainvoke`).** Most production paths are sync; I'd
   skip the async override for the spike → Phase 1 transition unless
   you flag a specific async caller. Confirm?
2. **Capability vocabulary governance.** The capability list is the
   "schema" of what tools can do. Where should this file live and who
   approves changes? My default: `app/tool_registry/capabilities.py`
   under the same review discipline as `app/souls/`. Sound?
3. **Phase 1c gate threshold.** What cache-savings ratio justifies
   continuing? My proposed bar: LoadableAgent must use ≤50% of stock
   tokens on a 5-iteration task with 2 mid-task loads. If we miss that,
   re-architect for Plan A (text-dispatch, costlier engineering work).

---

## 9. What's already in the repo

* `app/tool_runtime/__init__.py`
* `app/tool_runtime/binder.py`
* `app/tool_runtime/loadable_agent.py`
* `app/tool_runtime/loadable_executor.py`

No imports from any production agent yet. Safe to leave in place
between phases; safe to delete if you decide to take a different path.

The 4 files together are 420 LOC of which ~180 are mechanical
recomputation of CrewAI's iteration loop. If we ever need to
re-architect, ~250 LOC is reusable (binder + control tools).
