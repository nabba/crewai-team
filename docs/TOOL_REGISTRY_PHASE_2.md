# Tool Registry — Phase 2 pilot rollout

**Date:** 2026-05-03  •  **Status:** **OPT-IN PILOT** (introspector)  •  **Flag:** `LOADABLE_AGENT_EXPERIMENTAL=1`

Phase 2 promotes the Phase 0 spike (`LoadableAgent` + `ToolBinder`)
from prototype to production-grade by:

1. Wiring `ToolRegistry.instance()` into the `LoadableAgent`'s
   `available_tools` instead of a static dict (the spike's
   placeholder).
2. Capturing real cache hit/miss tokens per LLM call so the Phase 1c
   analytical model can be validated against Anthropic's actual
   behavior.
3. Migrating one low-stakes pilot agent (introspector) behind an
   env flag — default OFF so production behavior is unchanged.
4. Providing a parity harness (dry + live modes) operators can run
   to compare stock vs LoadableAgent before promoting the flag to
   default-on.

This is the riskiest phase in the program because it actually
changes agent behavior. The mitigations: opt-in flag, automatic
fallback to legacy on construction error, low-stakes pilot, dry
parity verdict required before live runs.

---

## 1. What ships

| Component | Role |
|-----------|------|
| `app/tool_runtime/factory.py` | `build_loadable_agent(...)` — hybrid constructor that mixes raw `core_tools` (per-agent state) with capability/name-driven discoverable tools from the registry. Tier + workspace gates applied at construction. |
| `app/tool_runtime/telemetry.py` | `record_call_usage(...)` writes one JSONL row per LLM call to `workspace/observability/loadable_agent_usage.jsonl` with `cache_creation_input_tokens` + `cache_read_input_tokens`. `analyze_telemetry()` summarizes hit/miss + computes effective tokens under Anthropic's pricing. |
| `app/tool_runtime/parity.py` | `run_parity_panel(mode="dry"\|"live", runs=N)` runs a 5-task panel against both stock + LoadableAgent introspector. Dry mode uses the Phase 1c analytical model; live mode makes real calls + reads telemetry. CLI: `python -m app.tool_runtime.parity`. |
| `app/agents/introspector.py` | Migrated to dispatch between `_legacy_create_introspector()` (stock CrewAI) and `_build_loadable_introspector()` (Phase 2) based on `LOADABLE_AGENT_EXPERIMENTAL` env. Failsafe: factory exception → fallback to legacy + warn. |
| `tests/test_tool_runtime_phase_2.py` | 16 tests across factory, telemetry, flag dispatch, parity harness. |

---

## 2. The hybrid factory

`build_loadable_agent` accepts FOUR ways to populate the agent's tool
list, all optional:

```python
agent = build_loadable_agent(
    role="Introspector",
    goal="...",
    backstory="...",
    llm=llm,
    agent_id="introspector",

    # 1. Raw eager tools (typical for agent-specific state — the
    #    introspector's per-collection memory tools land here).
    core_tools=[memory_tool_a, memory_tool_b, ...],

    # 2. Eager capability-driven (registry resolves; tool factories
    #    are called once at agent construction).
    core_capabilities=["reads-knowledge-base"],

    # 3. Lazy capability-driven (registry collects matching tools;
    #    factories wrapped in callables added to binder.available;
    #    the agent loads them on demand via load_tool / tool_search).
    discoverable_capabilities=["renders-pdf", "sends-signal"],

    # 4. Lazy name-driven (specific tool names the agent should be
    #    able to load).
    discoverable_names=["forge_create_tool"],

    agent_tier=Tier.PRODUCTION,
    workspace=None,
)
```

The hybrid surface lets pilot agents migrate **incrementally**: they
keep their existing per-agent tools as raw eager core (no annotation
required yet), and gain registry-driven discoverable access on top.
Phase 4's full migration will sweep the agent-specific tools into
`PER_AGENT`-lifecycle registry entries; until then, both paths work
in the same agent.

### Tier + workspace gates at construction

Same gates as `discovery.search_tools`:

* `agent_tier=Tier.PRODUCTION` (default) → only PRODUCTION + IMMUTABLE
  tools surface. SHADOW + CANARY filtered out.
* `workspace="eesti-mets"` → only tools scoped to `*` or
  `eesti-mets` surface.

A tool that fails the gate at construction simply doesn't appear in
`binder.catalog_names()`; the agent never knows it existed for this
session.

---

## 3. The introspector pilot

Why introspector: 56 lines, single factory function, narrow tool
surface (memory + scoped_memory + self_report + reflection),
operator-facing not user-facing, failure mode is "doesn't try a new
policy this cycle" — recoverable.

### Dispatch logic

```python
def create_introspector() -> Agent:
    if _is_loadable_experimental():       # LOADABLE_AGENT_EXPERIMENTAL=1
        try:
            return _build_loadable_introspector()
        except Exception as exc:
            logger.warning("LoadableAgent path failed (%s) — fallback", exc)
    return _legacy_create_introspector()  # stock CrewAI
```

**Default behavior is unchanged.** Production stays on
`_legacy_create_introspector` until operators opt in.

### What changes when the flag is on

* Agent class: `crewai.Agent` → `LoadableAgent`.
* Tool surface: same 11 introspector-specific tools as eager core,
  PLUS `load_tool`, `list_available_tools` (auto-injected), PLUS a
  catalog of ~3 discoverable tools (`file_manager`, `web_search`,
  others tagged `reads-knowledge-base`).
* Behavior: agent can now `tool_search(...)` to discover tools or
  `load_tool(name=...)` to pull one into its active toolset
  mid-task.

### Rollback

```bash
unset LOADABLE_AGENT_EXPERIMENTAL
# or
export LOADABLE_AGENT_EXPERIMENTAL=0
```

Restart the gateway. Done.

---

## 4. Telemetry — validating the Phase 1c model

The Phase 1c gate report predicted 33.4% of stock tokens with the
analytical model. Phase 2 validates this against real cache behavior.

Each LoadableAgent LLM call writes a JSONL row to
`workspace/observability/loadable_agent_usage.jsonl`:

```json
{
  "ts": "2026-05-03T14:23:01Z",
  "agent_id": "introspector",
  "iteration": 3,
  "input_tokens": 234,
  "output_tokens": 89,
  "cache_creation_input_tokens": 1500,
  "cache_read_input_tokens": 8200,
  "model": "claude-3-5-sonnet-..."
}
```

`analyze_telemetry(agent_id="introspector")` summarizes:

* `effective_input_tokens` — cost under Anthropic's cache pricing
  (1.0× fresh + 1.25× write + 0.10× read).
* `cache_read_pct` — fraction of input tokens served from cache.
* `vs_uncached.savings_ratio` — what the cache saved overall.

After 50+ live calls, compare `effective_input_tokens` against the
Phase 1c model's prediction. The gate threshold ±15%:

* Within ±15%: model validated, Phase 4 migrations proceed against
  the analytical model.
* Outside ±15%: recalibrate the model (likely the
  `tools-line-independence` assumption needs adjustment) and re-run
  the Phase 1c gate before Phase 4.

---

## 5. The parity harness

`app/tool_runtime/parity.py` — runs the same task panel against both
agent implementations, reports a comparison.

### Default 5-task panel

| Task | Iterations | Loads | Tests |
|------|-----------:|------:|-------|
| `trivial_self_report` | 1 | 0 | Cold-start cost (loadable wins big) |
| `memory_lookup` | 2 | 0 | Cache-warm second iter |
| `policy_synthesis` | 3 | 0 | Multi-iter cache-warm |
| `knowledge_assisted` | 4 | 1 | First mid-task load |
| `web_grounded` | 4 | 1 | Single-load workload |

Operators add their own tasks by passing `panel=[...]` to
`run_parity_panel`.

### Two execution modes

**Dry** (default): no API calls, uses Phase 1c analytical model.
Free, deterministic, fast. Used in CI to gate against regression.

```
$ docker exec crewai-team-gateway-1 python -m app.tool_runtime.parity
* Stock total:    127,373.7 effective tokens
* Loadable total: 24,461.8 effective tokens
* Ratio:          19.2%
* Verdict: GO
```

**Live**: makes real LLM calls, captures actual cache fields via
telemetry. Costs money. Used for the post-merge validation pass.

```
$ docker exec crewai-team-gateway-1 python -m app.tool_runtime.parity --live --runs 5
```

### Verdict semantics

`run_parity_panel(...)` returns a dict with `"verdict": "GO" | "NO-GO"`.
GO requires:

* `ratio <= 0.50` (Phase 1c gate threshold).
* `loadable_success_rate >= stock_success_rate * 0.90` — LoadableAgent
  must not fail tasks that stock would have completed.

The success-rate check matters most in live mode; dry mode trivially
passes both.

---

## 6. Failsafe behavior

Phase 2 is opt-in and failsafe everywhere it can be:

| Where | Failsafe |
|-------|---------|
| Env flag default | `LOADABLE_AGENT_EXPERIMENTAL` unset → legacy path. |
| LoadableAgent factory exception | Falls back to legacy path with a logged warning. |
| Telemetry write error | Swallowed, agent continues. |
| Telemetry read error | Returns `{}`, analysis returns zeros. |
| Registry not booted | Catalog empty, agent has only `core_tools` + control tools — still functional, just no discovery. |
| Live parity API errors | Per-task try/except → success=False, panel continues across other tasks. |
| Postgres / ChromaDB unreachable | All registry features degrade gracefully (Phase 1a invariant carried through). |

---

## 7. What's NOT in Phase 2

* **Production migration of any agent.** Introspector's flag is
  default-OFF. No production behavior changes.
* **Coder / Writer / Commander migrations.** Those are Phase 4 with
  per-agent parity panels.
* **Full annotation pass.** ~30 tools still aren't in the registry
  (memory, fiction, wiki, bridge, philosophy, aesthetic, tensions).
  Phase 4 sweeps them.
* **Live parity verdict against the gate.** Operator-driven —
  requires API budget. Plan: 50 live calls on the introspector pilot
  in the next operational cycle, results recorded as a Phase 2.5
  follow-up memo.
* **`load_tool` activation pattern in production.** The introspector
  has the tool in its inventory but no telemetry yet on whether the
  LLM actually uses it under realistic conditions.

---

## 8. Phase 2.5 — the post-merge validation cycle

After this PR merges, the operator-side workflow is:

1. **Set `LOADABLE_AGENT_EXPERIMENTAL=1`** in a non-production gateway
   (laptop dev or staging).
2. **Run `parity.py --live --runs 10`** on a representative panel.
   Captures ~100 LLM calls of telemetry.
3. **Read `analyze_telemetry(agent_id="loadable_introspector")`.**
   Compare `effective_input_tokens` against Phase 1c's prediction.
4. **If within ±15%**: model validated. Promote the flag to
   default-on for introspector; flag stays opt-in for other agents.
5. **If outside ±15%**: investigate, recalibrate Phase 1c, regate.

The 50-task parity panel mentioned in the Phase 1c memo is
operator-driven — operators decide which tasks exercise the pilot's
real workload best.

---

## 9. Migration progress

| Phase | What | Status |
|-------|------|--------|
| 0 | Spike + measurement + LoadableAgent prototype | DONE |
| 1a | Registry foundation + 11 tools annotated + `/cp/tools` endpoint | DONE (#39) |
| 1b | `tool_search` discovery primitive + 4-layer defense | DONE (#40) |
| 1c | Empirical cache-cost gate; verdict GO @ 33.4% | DONE (#41) |
| **2** | **Pilot agent (introspector) opt-in; factory + telemetry + parity harness** | **THIS PR** |
| 2.5 | Live parity validation on introspector (operator-driven) | After this PR |
| 3 | Forge → registry integration | After 2.5 |
| 4 | Production agent migration, one per week | After 3 |
| 5 | Drop `optional_tool_group` + legacy factories | After 4 |
