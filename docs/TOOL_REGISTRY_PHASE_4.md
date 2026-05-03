# Tool Registry — Phase 4: production agent migrations

**Status:** in progress  •  **Cadence:** one agent per PR  •  **Default:** OFF for all migrated agents

Phase 4 takes the LoadableAgent path from "opt-in pilot on a low-stakes
agent" (Phase 2 — introspector) to "opt-in on production agents one at
a time, validated against a parity panel before flag goes default-on."

---

## 1. Migration order + status

| Order | Agent | PR | Flag | Default | Status |
|------:|-------|----|------|--------:|--------|
| 1 (pilot) | introspector | #42 | `LOADABLE_INTROSPECTOR` | OFF | Phase 2 — opt-in shipped |
| 2 | researcher | #44 | `LOADABLE_RESEARCHER` | OFF | Phase 4a — opt-in shipped |
| 3 | writer | #45 | `LOADABLE_WRITER` | OFF | Phase 4b — opt-in shipped |
| 4 | coder | #46 | `LOADABLE_CODER` | OFF | Phase 4c — opt-in shipped |
| 5 | commander (does NOT migrate — see §4d) | **THIS PR** | (n/a) | (n/a) | **Phase 4d — diagnostics endpoint** |

Order rationale: stakes ascend. Researcher's failure mode is "missed
information"; writer's is "stilted prose"; coder's is "broken code";
commander's is "wrong agent picked." Earlier agents are easier to
revert if the LoadableAgent path misbehaves.

---

## 2. The per-agent flag pattern

Phase 4 introduces granular per-agent flags so operators can:

* Run an A/B comparison of one agent: `LOADABLE_RESEARCHER=1` while
  keeping coder/writer/commander on stock.
* Roll back one agent without disrupting the others.
* Default-off everywhere by leaving both vars unset.

Resolution (most-specific wins):

```
LOADABLE_<AGENT>=1     → loadable path (overrides master)
LOADABLE_<AGENT>=0     → legacy path (overrides master)
LOADABLE_<AGENT> unset → master decides
```

Master:
```
LOADABLE_AGENT_EXPERIMENTAL=1 → loadable path for any agent
                                without explicit per-agent flag
LOADABLE_AGENT_EXPERIMENTAL    → legacy default
unset
```

The helper at `app/tool_runtime/feature_flags.py` is the single
source of truth — every migrated agent's factory calls
`is_loadable_for("<agent>")` instead of duplicating env-var parsing.

Diagnostics endpoint: `GET /api/cp/tools/flags` (Phase 4b — not yet
implemented) will list every migrated agent with its current effective
flag value + source (per-agent override / inherited from master /
default).

---

## 3. Per-agent migration template

Each agent migration in Phase 4 follows the same shape:

```python
# app/agents/<agent>.py — after Phase 4 migration

def create_<agent>(...) -> Agent:
    """Dispatches between legacy and LoadableAgent based on flag."""
    from app.tool_runtime.feature_flags import is_loadable_for
    if is_loadable_for("<agent>"):
        try:
            return _build_loadable_<agent>(...)
        except Exception as exc:
            logger.warning("LoadableAgent path failed (%s) — fallback", exc)
    return _legacy_create_<agent>(...)


def _legacy_create_<agent>(...) -> Agent:
    """Stock CrewAI factory — kept as the default path during soak."""
    ...


def _build_loadable_<agent>(...) -> Agent:
    """Phase 4 path — LoadableAgent backed by the tool registry."""
    from app.tool_runtime.factory import build_loadable_agent
    return build_loadable_agent(
        role="...", goal="...", backstory="...",
        agent_id="<agent>",
        core_tools=...,                         # mirror legacy eager set
        discoverable_capabilities=[...],        # registry-driven discovery
        agent_tier=Tier.PRODUCTION,
        ...
    )
```

Properties of this template:

* **Default behavior unchanged.** Legacy factory runs unless flag set.
* **Failsafe.** Any exception in the loadable path falls back to legacy
  with a logged warning — operator sees the issue but the agent still works.
* **High behavior parity by construction.** `core_tools` for the
  loadable path mirrors the legacy full path's tool list; the only
  delta is two control tools (`load_tool`, `list_available_tools`) and
  the discoverable catalog. Tests assert this parity (see
  `TestEagerToolsetParity` in each phase's test file).

---

## 4. Phase 4a — researcher

Researcher's two paths:

* **Light** (`light=True`) — 5-10 tools, compact backstory. Used for
  difficulty ≤ 3 simple factual questions. Always legacy regardless
  of flag — its small tool surface doesn't benefit from registry
  overhead.
* **Full** (`light=False`) — 30+ tools across web search, KB,
  Mem0, episteme, experiential, firecrawl, composio, bridge, wiki,
  blackboard, tensions, OCR, research_orchestrator. **Migration
  target.**

The full path's loadable build mirrors the legacy eager toolset
exactly (tested via `TestEagerToolsetParity::test_loadable_eager_count_matches_legacy_full`).
Discoverable capabilities added on top:

```python
discoverable_capabilities=[
    "renders-pdf",       # synthesize report PDFs
    "sends-signal",       # deliver findings to the user
    "renders-chart",      # visualize comparisons
    "fetches-geodata",    # map / region queries
    "executes-code",      # numeric processing of findings
]
```

This means the researcher can `tool_search(intent="…")` or
`load_tool(name="pdf_compose")` mid-task to pull in `pdf_compose`,
`signal_send_attachment`, `geodata_discover`, `geodata_fetch`, or
`execute_code` without an agent rewrite.

Activation:

```bash
# Just this agent
export LOADABLE_RESEARCHER=1

# All migrated agents at once (4a alone for now; 4b+ as they ship)
export LOADABLE_AGENT_EXPERIMENTAL=1
```

---

## 4b. Phase 4b — writer

Writer is single-path (no `light` variant). 30-44 tools across
memory, scoped_memory, mem0, knowledge_base, philosophy_rag, fiction,
experiential, aesthetics, document_generator, pdf_compose,
signal_send_attachment, tool_search, bridge, wiki, blend, dialectics,
tensions.

The migration mirrors Phase 4a's pattern exactly: extract the legacy
body to `_legacy_create_writer`, add `_build_loadable_writer` whose
eager toolset mirrors legacy by construction, add a dispatcher
`create_writer` that routes via `is_loadable_for("writer")` with
failsafe fallback.

Discoverable capabilities for writer:

```python
discoverable_capabilities=[
    "executes-code",        # numeric/data appendices in reports
    "fetches-geodata",      # geographic context for writing
    "reads-satellite",      # factual basis for nature/place writing
    "converts-currency",    # financial reports
    "renders-chart",        # visualize data alongside prose
]
```

Activation:

```bash
export LOADABLE_WRITER=1                    # writer only
export LOADABLE_AGENT_EXPERIMENTAL=1        # all migrated agents
```

---

## 4c. Phase 4c — coder

Coder is the highest-stakes Phase 4 migration. It executes code in
the sandbox, calls Forge to generate new tools, and produces
user-deliverable artifacts (PDFs over Signal). A migration
regression here would be more visible than researcher / writer.

The migration mirrors the Phase 4a/4b pattern exactly: dispatcher
+ legacy + loadable factories + failsafe fallback. Eager toolset
mirrors legacy by construction; the only delta is the 2 binder
control tools.

Discoverable capabilities for coder (catalog tools NOT already
eager):

```python
discoverable_capabilities=[
    "fetches-geodata",      # → geodata_discover/fetch
    "converts-currency",    # → currency_convert
    "fetches-finance",      # → currency_rates / future
]
```

Forge-bridged SHADOW/CANARY tools tagged `registers-tool` also
surface here automatically (Phase 3 bridge), so the coder can
pick up any operator-promoted Forge tool without an agent
rewrite — closing a long-standing gap in the Forge → coder
loop.

Activation:

```bash
export LOADABLE_CODER=1                    # coder only
export LOADABLE_AGENT_EXPERIMENTAL=1       # all migrated agents
```

Failsafe is especially important here: a Phase 4c bug must not
break the user-facing PDF/Signal flow. The dispatcher's
try/except catches any exception in `_build_loadable_coder` and
falls back to `_legacy_create_coder`, so the worst case is "the
experimental path is wasted this cycle" not "user can't get a
report."

---

## 4d. Phase 4d — Commander (architectural exception) + flags endpoint

### Why Commander does NOT migrate

The original Phase 0 plan listed Commander as the final Phase 4
migration ("highest stakes — the routing brain"). On inspection,
`Commander` (in `app/agents/commander/orchestrator.py`) is a
**class-based orchestrator**, not a CrewAI Agent factory:

* It has its own `self.llm` for routing decisions, used directly
  via the LLM client — not as a `crewai.Agent` instance.
* It holds `self.memory_tools` but doesn't pass them to a CrewAI
  Agent's `tools=` list — they're invoked directly when the
  orchestrator needs to record routing decisions.
* The actual `crewai.Agent` instances Commander interacts with are
  the OTHER agents (researcher / writer / coder), constructed by
  the factories already migrated in Phases 4a-c.

There is one `crewai.Agent` built in commander's territory — the
"Commander (Synthesizer)" agent in `app/crews/creative_crew.py`
— but it's constructed with `tools=[]`. LoadableAgent solves
tool-cost overhead; with zero tools there's nothing to optimize.

**Therefore Phase 4d does NOT migrate Commander.** The dispatcher
+ legacy + loadable factory pattern doesn't apply here. Documenting
this explicitly closes the "commander migration" item from the
Phase 0 plan.

### What Phase 4d ships instead

A **diagnostics endpoint** that lets operators see at a glance
which migrated agents are running on which path:

```
GET /api/cp/tools/flags

{
  "master_flag": true,
  "migrated_agents": [
    {"agent": "introspector", "loadable": true,  "source": "master flag", "explicit_flag": null},
    {"agent": "researcher",   "loadable": false, "source": "per-agent override", "explicit_flag": "0"},
    {"agent": "writer",       "loadable": true,  "source": "master flag", "explicit_flag": null},
    {"agent": "coder",        "loadable": true,  "source": "master flag", "explicit_flag": null}
  ],
  "count_loadable": 3,
  "count_legacy": 1
}
```

The endpoint reads the same env vars that the agent factories
themselves consult, so what it shows IS what the agents do — no
drift possible.

This was deferred from Phase 4a as "Phase 4b territory"; it lands
here as Phase 4d's deliverable.

Use cases:
* React control plane renders an "agent flag matrix" widget so
  operators don't need `kubectl exec` to read env vars.
* Pre-deploy sanity check: `curl /api/cp/tools/flags | jq` confirms
  expected flag state before flipping master to default-on.
* Post-incident triage: was researcher on legacy or loadable when
  the regression happened? The endpoint answers in one query.

### What's still left

The original Phase 0 plan called for a "live parity verdict"
on each migrated agent before flipping its flag default to ON.
This is operator-driven (Phase 2.5 / Phase 4-X.5 cycles) and
deferred to operator workflow, not blocking Phase 5.

**Phase 5** (drop legacy factories) is now the next step. It
consolidates: remove `_legacy_create_<agent>` once parity is
validated, drop `optional_tool_group` (replaced by registry guards),
clean up unused imports. Phase 5 is purely cleanup — no behavior
change.

---

## 5. Phase 4-X validation cycle

Each agent migration in Phase 4 follows this operator-driven
validation cycle before the flag goes default-on:

1. **Merge the migration PR.** Default behavior unchanged.
2. **Set per-agent flag in staging.** E.g. `LOADABLE_RESEARCHER=1`.
3. **Run a 25-50 task panel** representative of the agent's actual
   workload. Collect:
   * Success rate vs. legacy.
   * Token usage via `analyze_telemetry(agent_id="<agent>")`.
   * Behavioral artifacts that reveal weird LLM responses to the
     dynamic-loading flow (e.g. agent doesn't call `tool_search` even
     when it would help).
4. **Compare against acceptance criteria:**
   * Success rate ≥ 0.90× legacy success rate.
   * Token usage ≤ Phase 1c prediction × 1.15 (allowing 15% slack
     beyond the 33% analytical estimate).
   * No new failure modes vs. legacy (manual review of failed tasks).
5. **If pass:** flip the flag default to ON for that agent.
   Update this doc's status table. Move to next agent.
6. **If fail:** investigate, file follow-ups, regate. The legacy
   factory keeps running — there's no production impact while we fix.

The exact 25-50 task panel is operator-curated per agent — it should
exercise that agent's representative workload, not a synthetic toy
panel.

---

## 6. Failsafe behavior recap

Same matrix as Phase 2, applied per-agent:

| Where | Failsafe |
|-------|---------|
| Default flag state | Both vars unset → legacy path |
| LoadableAgent factory exception | Legacy path runs, warning logged |
| Registry unreachable / unbooted | Catalog empty, agent functional |
| Forge bridge fails | Phase 3 invariant — non-fatal, registry still works |
| ChromaDB / Postgres offline | Phase 1a invariant — degrades gracefully |

Rolling back one agent: `unset LOADABLE_<AGENT>` (or set to `0`).
Restart gateway. Done.

Rolling back all migrated agents: `unset LOADABLE_AGENT_EXPERIMENTAL`
and unset every per-agent flag set to `1`.

---

## 7. What's NOT in Phase 4a

* **Live parity panel results.** Operator-driven, post-merge.
  Plan: 25-50 tasks against staging with `LOADABLE_RESEARCHER=1`,
  results recorded as a Phase 4a.5 follow-up note.
* **Writer / coder / commander migrations.** Each gets its own PR
  (4b / 4c / 4d). Order reflects ascending stakes.
* **`/api/cp/tools/flags` endpoint.** Diagnostics view of every
  migrated agent's flag state. Phase 4b territory.
* **The 50-task panel mentioned in Phase 1c.** Operator-driven —
  operators decide which tasks exercise each agent's real workload
  best. The harness (`app/tool_runtime/parity.py`) supports any task
  list passed via `panel=[...]`.

---

## 8. Migration progress

| Phase | Status |
|-------|--------|
| 0 — Spike + measurement | DONE |
| 1a — Registry foundation (#39) | DONE |
| 1b — `tool_search` (#40) | DONE |
| 1c — Cache-cost gate (#41) | DONE |
| 2 — Pilot (introspector) (#42) | DONE |
| 3 — Forge bridge (#43) | DONE |
| 4a — Researcher migration (#44) | DONE |
| 4b — Writer migration (#45) | DONE |
| 4c — Coder migration (#46) | DONE |
| **4d — Commander n/a + flags endpoint** | **THIS PR** |
| 5 — Drop `optional_tool_group` + legacy factories | Next (post-soak) |
