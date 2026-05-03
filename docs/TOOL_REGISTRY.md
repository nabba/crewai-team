# Tool Registry

A capability-typed registry of every agent-callable tool in the
system, plus the mechanism to load tools dynamically into running
agents. Replaces the static `tools=[...]` list in agent factories
with a discoverable catalog. Phase 1a (foundation) shipped 2026-05-03;
later phases land per the roadmap below.

The companion document `docs/TOOL_REGISTRY_PHASE_0.md` is the spike
report that motivated this design; read it first if you want the
"why" + measurements.

---

## 1. The shape of the system

```
┌──────────────────────────────────────────────────────────────────┐
│ @register_tool(name=..., capabilities=[...], ...)  ← decoration  │
│   │                                                              │
│   ▼                                                              │
│ ToolSpec  (frozen dataclass, name + capabilities + tier +        │
│            lifecycle + description + factory + guard +           │
│            workspace_scope + source_module + description_hash)   │
│   │                                                              │
│   ├─→ ToolRegistry singleton (in-memory primary)                 │
│   └─→ _DECORATED_SPECS side-table (survives reset_for_tests)     │
│                                                                  │
│ ToolRegistry.replay_decorations()                                │
│   ↓ on boot or test setup, copies side-table into singleton      │
│                                                                  │
│ tool_registry.boot.boot_registry()                               │
│   - imports every module under TOOL_MODULE_ROOTS                 │
│   - replays decorator side-table (covers sys.modules cache)      │
│   - snapshots singleton to Postgres `tool_registry` table        │
│   - runs drift detection vs prior snapshot                       │
│                                                                  │
│ /api/cp/tools                                                    │
│   - GET /            list tools (filter by capability/tier/...)  │
│   - GET /{name}      detail for one tool                         │
│   - GET /stats       counts by tier / lifecycle                  │
│   - GET /capabilities the bounded vocabulary                     │
│   - GET /drift       hash + tier diff vs last snapshot           │
└──────────────────────────────────────────────────────────────────┘
```

Phase 2 will add `tool_search` (semantic discovery) and a `LoadableAgent`
that wires the registry into CrewAI's iteration loop. Both prerequisites
shipped in Phase 0 (`app/tool_runtime/`) and are usable for spikes.

---

## 2. Components

| Path | Role |
|------|------|
| `app/tool_registry/capabilities.py` | The bounded capability vocabulary. **TIER_IMMUTABLE** — Self-Improver cannot grow it. |
| `app/tool_registry/types.py` | `Tier`, `Lifecycle`, `ToolSpec` dataclasses. |
| `app/tool_registry/decorator.py` | `@register_tool` + the `_DECORATED_SPECS` side-table. |
| `app/tool_registry/registry.py` | `ToolRegistry` singleton. Read API + lifecycle-aware `build_instance`. |
| `app/tool_registry/persistence.py` | Postgres snapshot of the in-memory registry. |
| `app/tool_registry/drift.py` | Description-hash + tier drift detection vs prior snapshot. |
| `app/tool_registry/boot.py` | One-shot bootstrap: walk `TOOL_MODULE_ROOTS`, replay, snapshot, detect drift. |
| `app/control_plane/tools_api.py` | Read-only HTTP endpoints under `/api/cp/tools`. |
| `tests/test_tool_registry.py` | 22 tests across capabilities / decorator / filter / boot / governance. |
| `app/tool_runtime/` | Phase 0 spike — LoadableAgent + binder + custom executor (not yet promoted; lives separately). |

---

## 3. Capability vocabulary

The vocabulary is the **typed contract** between tool authors and
agents that search for capabilities. It is bounded: ~35 tags grouped
into 6 categories (`data`, `knowledge`, `memory`, `compute`,
`delivery`, `governance`).

Why bounded:

* The May 2026 incident where stale "Weather Forecast" skills hijacked
  the Estonia deforestation request was a surface-keyword failure —
  unbounded synonym space lets stale entries steal relevant queries.
  A finite vocabulary lets the matcher use exact-tag-match first,
  embedding-similarity second.
* Adding a tag is a deliberate act (PR review). Renaming or removing
  one is forbidden once tools reference it; deprecate via
  `DEPRECATED_CAPABILITIES` if needed.

Why governance-grade:

* `app/tool_registry/capabilities.py` is in `TIER_IMMUTABLE` (see
  `app/auto_deployer.py`). The Self-Improver agent cannot edit it.
  Same review discipline as `app/souls/`.

How tools use it:

```python
@register_tool(
    name="pdf_compose",
    capabilities=["renders-pdf", "renders-chart"],   # validated at decoration
    description="...",
    tier=Tier.PRODUCTION,
    lifecycle=Lifecycle.SINGLETON,
)
def _pdf_compose_factory(agent_id: str = "coder"):
    return PdfComposeTool()
```

If a tool declares an unknown tag, `@register_tool` raises `ValueError`
at module import time — startup fails loudly rather than producing a
silently-undiscoverable tool.

---

## 4. The decorator and side-table pattern

`@register_tool` is **passive** on the decorated function:

```python
def decorator(fn):
    spec = ToolSpec(...)
    _DECORATED_SPECS[name] = spec       # side-table (survives reset)
    ToolRegistry.instance().register(spec)
    return fn                            # untouched — legacy callers still work
```

Phase 1a's invariant: annotating an existing tool factory does **not**
change how that factory is called by existing agents. The legacy
`create_pdf_tools(agent_id)` keeps working unchanged; the annotation
just *also* registers the tool in the catalog.

### Why a side-table

Module-level decorators fire **once per Python process**: importlib
caches modules in `sys.modules`, so re-importing doesn't re-run the
body. That's a problem for:

* Tests that want a clean ToolRegistry singleton via
  `reset_for_tests()` — without the side-table, all decorator-
  registered specs would vanish permanently.
* Multi-process boot (rare today; possible later) where the gateway
  process imports tools but a sidecar wants to read the registry.

The decorator stores every spec in a process-global side-table that
**never clears**. `ToolRegistry.replay_decorations()` copies the
side-table into the current singleton on demand. Tests do
`reset_for_tests()` then `boot_registry()` which calls
`replay_decorations()` automatically.

---

## 5. Lifecycle

Most tools are `SINGLETON` — one instance per process, cached by the
registry. Heavy imports (matplotlib, mem0 client, Earth Engine) only
happen once. `build_instance("pdf_compose")` always returns the same
object.

```python
class Lifecycle(Enum):
    SINGLETON   # one per process, shared across agents.
    PER_AGENT   # one per (name, agent_id). For tools with per-agent state.
    PER_CREW    # fresh per crew run. Caller manages scope.
    PER_CALL    # fresh on every invocation. Rare.
```

Factories that take `agent_id` get it injected; factories that don't,
get called bare. The registry handles the introspection.

---

## 6. Tier model

`Tier` mirrors `auto_deployer.py`'s tier hierarchy:

| Tier | Meaning |
|------|---------|
| `SHADOW` | Newly forged, not yet validated. Hidden from production crews. |
| `CANARY` | Validated in soak. Visible to canary crews. |
| `PRODUCTION` | Live and trusted. |
| `IMMUTABLE` | Pinned. Cannot be replaced or unloaded. |

Discovery filters by tier: a `PRODUCTION`-only crew never sees
`SHADOW` tools, so Forge-generated tools don't bleed into prod
without going through the audit + graduation pipeline.

---

## 7. Workspace scope

Each tool declares one or more workspace IDs in its
`workspace_scope`, with `("*",)` meaning "any workspace." The
discovery layer filters per request:

```python
# In an eesti-mets workspace request:
reg.filter(workspace="eesti-mets")
# returns tools scoped to "*" OR explicitly to "eesti-mets"
```

This is the per-workspace allowlist that prevents domain-specific
tools (e.g. "fetch ECB rates") from being suggested to a forest-
research crew.

---

## 8. Postgres snapshot + drift detection

After `boot_registry` populates the in-memory registry, we snapshot
to a `tool_registry` Postgres table:

```sql
CREATE TABLE tool_registry (
    name             TEXT PRIMARY KEY,
    capabilities     TEXT[] NOT NULL,
    tier             TEXT NOT NULL,
    lifecycle        TEXT NOT NULL,
    description      TEXT NOT NULL,
    description_hash TEXT NOT NULL,
    workspace_scope  TEXT[] NOT NULL,
    source_module    TEXT NOT NULL,
    is_loadable      BOOLEAN NOT NULL,
    snapshot_ts      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

On boot we **first** run drift detection against the prior snapshot,
**then** rewrite the snapshot. Drift kinds:

| Kind | Meaning |
|------|---------|
| `new` | First registration this boot. |
| `changed` | Description hash differs from prior snapshot. **Logged as warning** — bypasses LLM-contract review. |
| `tier_changed` | Same tool, different tier (e.g. SHADOW → CANARY promotion). |
| `removed` | Prior snapshot has a tool not in current registry. |

Drift is **observational**, not enforcing. A CI test could be added
to assert no `changed` entries against a committed snapshot.

If Postgres is unreachable, the registry still works — snapshot +
drift detection are non-fatal.

---

## 9. /api/cp/tools — the React-facing browser

| Route | Purpose |
|-------|---------|
| `GET /api/cp/tools` | List tools. Query params: `capability`, `tier`, `loadable_only`, `workspace`. |
| `GET /api/cp/tools/{name}` | Full detail for one tool. 404 if absent. |
| `GET /api/cp/tools/stats` | Counts by tier + lifecycle + capability coverage. |
| `GET /api/cp/tools/capabilities` | The bounded vocabulary, grouped by category. |
| `GET /api/cp/tools/drift` | Drift entries vs last snapshot. |

Auth: same `require_gateway_auth` dependency as the rest of the
control plane. Default OFF on laptop dev, ON in K8s.

Source-of-truth precedence:
1. In-process ToolRegistry singleton (current state).
2. Postgres snapshot (cross-process visibility).

---

## 10. What ships in Phase 1a

| File | Lines | Status |
|------|------:|--------|
| `app/tool_registry/__init__.py` | 30 | new |
| `app/tool_registry/capabilities.py` | 165 | new (TIER_IMMUTABLE) |
| `app/tool_registry/types.py` | 110 | new |
| `app/tool_registry/decorator.py` | 130 | new |
| `app/tool_registry/registry.py` | 195 | new |
| `app/tool_registry/persistence.py` | 165 | new |
| `app/tool_registry/drift.py` | 100 | new |
| `app/tool_registry/boot.py` | 80 | new |
| `app/control_plane/tools_api.py` | 130 | new |
| `app/main.py` | +20 | boot-registry + router include |
| `app/auto_deployer.py` | +5 | capabilities.py → TIER_IMMUTABLE |
| `app/tool_runtime/loadable_executor.py` | +130 | async path override (closes Phase 0) |
| `tests/test_tool_registry.py` | 350 | new (22 tests) |
| `docs/TOOL_REGISTRY.md` | this | new |
| `docs/TOOL_REGISTRY_PHASE_0.md` | (existing) | from Phase 0 |
| `app/tools/{pdf_compose,signal_attachment,file_manager,web_search,attachment_reader,code_executor,gee_tool,geodata_tool,currency_tools}.py` | annotations only | 11 tools annotated |

Annotated tools: `pdf_compose`, `signal_send_attachment`,
`file_manager`, `web_search`, `read_attachment`, `execute_code`,
`gee_run_script`, `geodata_discover`, `geodata_fetch`,
`currency_convert`, `currency_rates`.

Verified: `boot_registry()` produces 11 specs, all loadable,
description hashes stable across boots, no name collisions, all
capabilities valid.

---

## 11. How to annotate a new tool

```python
# In app/tools/my_tool.py
from app.tool_registry import register_tool, Tier, Lifecycle

# ... existing tool definition (BaseTool subclass or @tool function) ...

# Append at the bottom:
@register_tool(
    name="my_tool",
    capabilities=["renders-pdf"],   # must be in capabilities.py
    description=(
        "What the tool does. When to use it. Worked example. "
        "Anti-patterns. Same prose as today's tool descriptions — "
        "this is what surfaces to the LLM."
    ),
    tier=Tier.PRODUCTION,
    lifecycle=Lifecycle.SINGLETON,
    guard=lambda: True,             # or env-check function
    workspace_scope=("*",),
)
def _my_tool_registry_factory(agent_id: str = "coder"):
    return MyToolInstance()
```

If `MyToolInstance` is already a module-level singleton (typical for
`@tool`-decorated functions), the factory just returns it — zero
overhead.

For new capability tags: open a PR against
`app/tool_registry/capabilities.py` first; it's TIER_IMMUTABLE.

---

## 12. Migration path (Phase 1b → 2 → ...)

| Phase | What | Status |
|-------|------|--------|
| 0 | Spike + measurement + LoadableAgent prototype | DONE — `docs/TOOL_REGISTRY_PHASE_0.md` |
| 1a | Registry foundation + 11 tools annotated + `/cp/tools` endpoint | **THIS PR** |
| 1b | `tool_search` discovery primitive (capability + ChromaDB embedding ranking) | Next |
| 1c | Empirical cache measurement; ≤50% of stock tokens gate | After 1b |
| 2 | LoadableAgent on a low-stakes pilot agent | Gated on 1c |
| 3 | Forge → registry integration (SHADOW → CANARY → PRODUCTION via existing tier-graduation) | After 2 |
| 4 | Production agent migration, one per week | After 3 |
| 5 | Drop `optional_tool_group` + legacy factories | After 4 |

The Phase 0 spike (`app/tool_runtime/`) shipped before this Phase 1a
package. It is not yet wired into any agent — it stays in tree as a
working prototype that Phase 2 will promote.

---

## 13. What changed in adjacent files

* `app/auto_deployer.py` — added `app/tool_registry/capabilities.py`
  to `TIER_IMMUTABLE`.
* `app/main.py` — added the `boot_registry()` call + `tools_api`
  router include. Both inside try/except so startup failures don't
  bring down the gateway.
* The 9 annotated tool files (`app/tools/...`) — appended a
  `try: from app.tool_registry import ... ; @register_tool ...`
  block at the bottom of each. Pure addition; existing factories
  unchanged.

---

## 14. Open follow-ups (Phase 1b territory)

* **Annotate the remaining ~30 tools.** Phase 1a covered the 11
  most-used. The rest (memory tools, fiction tools, wiki tools,
  bridge tools, philosophy tools, aesthetic tools, etc.) get
  annotated in Phase 1b alongside `tool_search`.
* **`tool_search` primitive.** Read-only discovery tool that ranks by
  capability tag (exact match) + ChromaDB embedding similarity +
  workspace gate + tier gate + recent-failure penalty + quarantine
  list. Same 4-layer defense as skills retrieval.
* **Schema-aware diff.** Right now drift detects description hash
  changes but not args_schema changes. Phase 1b adds a hash over the
  pydantic schema's JSON form.
* **CI gate on PRODUCTION tier description drift.** A test that
  asserts no `changed` entries against a committed snapshot would
  catch silently-modified descriptions in PR review.

---

# Phase 1b — `tool_search` discovery primitive

Shipped 2026-05-03 (this section appended to the same doc). Adds the
agent-callable read-only discovery primitive that ranks the catalog
by capability tag + ChromaDB semantic match, with the same 4-layer
contamination defense as the skills-retrieval system.

## 1b.1 Components added in this phase

| Path | Role |
|------|------|
| `app/tool_registry/discovery.py` | `search_tools(intent, capabilities, ...)` — the ranker. Implements the 4 hard gates + 3 soft signals. |
| `app/tool_registry/quarantine.py` | Read-only quarantine list at `workspace/tool_registry/quarantine.json`. Operator-edited; runtime filter for known-bad tools. |
| `app/tool_registry/indexer.py` | ChromaDB indexer over tool descriptions. Cosine-space collection (matches the skills-retrieval calibration). Idempotent — re-embeds only on `description_hash` change. |
| `app/tools/tool_search.py` | The agent-callable BaseTool. Calls `search_tools`, formats ranked output. Self-registered in the catalog. |
| `app/agents/coder.py` + `app/agents/writer.py` | Wired in via `optional_tool_group("...", "tool_search")` — purely additive, agents can use it or ignore it. |
| `tests/test_tool_search.py` | 22 unit tests including the Weather/Estonia regression analogue. |

## 1b.2 The 4-layer defense (ported from skills retrieval)

The May 2026 incident where a stale "Weather Forecast" skill matched
on a subjectless "execute the plan" query and hijacked an Estonia
deforestation request was a surface-keyword failure. The skills layer
hardened against it with four independent gates; this phase ports the
same discipline to tools.

The 4 layers in `discovery.search_tools`, applied in order:

| # | Layer | What it filters | Source of truth |
|---|-------|-----------------|-----------------|
| 0 | **Subjectless detection** | Bare imperative tokens ("ok", "go", "execute the plan") with no capability tags → empty list. | `_is_subjectless` token list, mirrors `app/agents/commander/context.py:53-74`. |
| 1 | **Quarantine** | Tools listed in `workspace/tool_registry/quarantine.json` are filtered out entirely (no rank, no surface). | Operator-edited JSON file; mirrors `workspace/skills/_quarantine/`. |
| 2 | **Tier gate** | Tools below the agent's authorized tier are filtered. PRODUCTION crews see PRODUCTION + IMMUTABLE only; SHADOW crews see everything. | `_tier_passes_gate` against `_TIER_RANK`. |
| 3 | **Workspace gate** | Tools whose `workspace_scope` doesn't include `*` or the active workspace ID are filtered. | `spec.workspace_scope` declared at `@register_tool` time. |
| 4 | **Distance gate** | ChromaDB cosine distance > `_DISTANCE_CEILING` (0.55, calibrated to match skills-retrieval) → match dropped. | `query_index` in `indexer.py`; cosine space pinned via `hnsw:space=cosine`. |

Plus three **soft signals** that affect ranking, not membership:

* **Capability exact-match boost** — tools declaring at least one of
  the requested tags get a `+0.30` score bonus. Tuned so a perfect
  tag match outranks a `0.40`-distance prose-only match.
* **Tier rank** — within authorized tier, PRODUCTION outranks CANARY
  outranks SHADOW.
* **Loadability** — tools whose `guard()` returns False are visible
  (the agent knows they exist) but ranked at `score * 0.5`.

## 1b.3 ChromaDB indexing

Collection: `tool_registry`. One document per tool. Doc text:

```
<name> [<capabilities>]
<description>
```

Capability tags are inlined so a query like "render forest report PDF"
hits both the prose AND the structured `renders-pdf` tag. Metadata
includes `tier`, `lifecycle`, `capabilities_csv`, `workspace_scope_csv`,
`description_hash`, `is_loadable`, `source_module`.

The collection is created with `metadata={"hnsw:space": "cosine"}` so
the `_DISTANCE_CEILING = 0.55` is calibrated correctly. Default L2
distances would run 100×–1000× larger and never fall below the ceiling
— this is the failure mode caught during the spike (initial L2-space
collection had the right architecture but wrong space, all queries
returned empty).

Re-indexing is idempotent: at boot, `index_tools(specs)` compares each
spec's `description_hash` against the stored value and skips unchanged
entries. Cold start is cheap on warm ChromaDB caches.

## 1b.4 The `tool_search` BaseTool

Added to coder + writer alongside their existing toolsets via the
standard `optional_tool_group` pattern. Args:

```python
tool_search(
    intent: str = "",                 # natural-language query
    capabilities: list[str] = [],     # optional tag filter / boost
    limit: int = 5,                   # max ranked candidates
)
```

Returns formatted ranked output — name, tier, score, capabilities,
one-line `reason`, truncated description, loadability note. Example:

```
Found 3 matching tool(s) (highest-ranked first):

  pdf_compose  [production]  score=0.92
    capabilities: ['renders-pdf', 'renders-chart']
    why: matches capability renders-pdf; semantic match (d=0.31)
    Render a PDF report locally (matplotlib + reportlab) ...

  signal_send_attachment  [production]  score=0.51
    capabilities: ['sends-signal', 'delivers-attachment']
    why: semantic match (d=0.49)
    Send a Signal message with one or more file attachments ...
```

Phase 1b is **read-only** — `tool_search` reports candidates but does
not auto-load them. Auto-load lands in Phase 2 with `LoadableAgent`
(prototyped in `app/tool_runtime/`).

## 1b.5 Quarantine list mechanism

File path: `workspace/tool_registry/quarantine.json`. Schema:

```json
{
  "quarantined": [
    {"name": "broken_tool", "reason": "produces wrong CSV values", "since": "2026-05-..."},
    ...
  ]
}
```

Re-read on every `quarantined_names()` call (small file, lookup is rare).
Operator edits are picked up immediately on the next discovery query.
Malformed JSON is treated as empty — operator-side bugs don't crash
discovery.

There is **no programmatic API** for agents to quarantine each other's
tools. Quarantine is operator-only, same governance as `app/souls/`.

## 1b.6 The Weather/Estonia regression test

`tests/test_tool_search.py::TestWeatherEstoniaRegression` mirrors
`tests/test_skill_retrieval_contamination.py::TestProductionRegression`
for the tool layer. Three independent assertions:

1. Subjectless query ("execute the plan") returns empty when no
   capabilities are specified — Layer 0 defense.
2. Explicit capability request (`renders-pdf`) still works — the
   defense disables UNINTENTIONAL retrieval, not deliberate
   capability lookups.
3. A weak distance match (above the 0.55 ceiling) is filtered even
   when the intent is non-subjectless — Layer 4 defense.

## 1b.7 Future work (Phase 1c onwards)

* **Annotate the remaining ~30 tools.** Phase 1a covered 11; this PR
  doesn't expand the annotated set. Phase 1c includes a sweep of the
  rest (memory, fiction, wiki, bridge, philosophy, aesthetic, tensions).
* **Cache measurement gate (Phase 1c).** Empirical token-cost
  measurement of LoadableAgent vs stock agent on a 5-iteration task
  with 2 mid-task loads. Must hit ≤50% of stock tokens before Phase 2.
* **Workspace propagation through the executor.** Phase 1b's tool_search
  accepts a `workspace=` kwarg but the agent-side `_run` defaults it to
  None — the workspace ID isn't yet propagated through CrewAI's tool
  dispatch. Phase 2's LoadableAgent will close this loop.
* **Recent-failure penalty.** Designed but not implemented — couples to
  the error monitor. Tools with a high recent failure rate would be
  ranked lower; lands when error_monitor exposes per-tool stats.
