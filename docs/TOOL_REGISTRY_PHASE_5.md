# Tool Registry — Phase 5: cleanup readiness + rollout sequence

**Date:** 2026-05-04  •  **Status:** readiness work shipped; cleanup gated on operator validation

Phase 5 was originally scoped as "drop the legacy factories + `optional_tool_group`." On reflection that's too aggressive for a single PR — it removes the failsafe path that every Phase 4 migration depends on. Without per-agent live parity validation (the Phase 2.5 / 4-X.5 cycles), deleting legacy is unjustified production risk.

Phase 5 here is the **readiness work** instead:

* A smoke-test CLI (`phase5_check`) that verifies each migrated agent can construct on both paths and that eager-toolset parity holds.
* This rollout-sequence doc that codifies the per-agent progression and the eventual cleanup steps.

The actual default flips and code deletion happen in **future per-agent PRs**, gated on each agent's parity panel passing in operator-managed staging. Phase 5 just makes that work cheap to execute when the time comes.

---

## 1. What `phase5_check` does

```bash
docker exec crewai-team-gateway-1 python -m app.tool_runtime.phase5_check
```

For each migrated agent (`introspector`, `researcher`, `writer`, `coder`):

1. Constructs the legacy path (`LOADABLE_<AGENT>=0`) — should succeed.
2. Constructs the loadable path (`LOADABLE_<AGENT>=1`) — should succeed.
3. Verifies eager-toolset parity: loadable's tool list should equal legacy's PLUS the 2 binder control tools (`load_tool`, `list_available_tools`), with no missing tools.
4. Checks the discoverable catalog: capability resolution must produce ≥1 tool (otherwise something's misconfigured).

Exit code 0 if all 4 agents pass; non-zero if any fails. Markdown output by default; `--json` for programmatic consumption.

Sample output (current state, all 4 agents READY):

```
# Phase 5 readiness check

**Verdict: READY**

| Agent | Ready | Legacy | Loadable | Eager parity | Catalog |
|-------|:-----:|:------:|:--------:|:------------:|--------:|
| introspector | ✓ | ✓ (11) | ✓ (13) | ✓ | 2 |
| researcher | ✓ | ✓ (38) | ✓ (40) | ✓ | 5 |
| writer | ✓ | ✓ (44) | ✓ (46) | ✓ | 5 |
| coder | ✓ | ✓ (38) | ✓ (40) | ✓ | 4 |

* 4 of 4 agents READY
* 0 NOT-READY
```

Use it as a CI gate, a pre-deploy sanity check, or a post-incident triage tool. It does NOT exercise live behavior — that's still operator-driven.

---

## 2. Per-agent rollout sequence

Each migrated agent moves through these stages independently. The order is up to operators; the suggested order matches ascending stakes (introspector → researcher → writer → coder).

```
[Stage 0] Legacy default (current state for all 4 agents)
    LOADABLE_<AGENT> unset; legacy path runs
    ↓
[Stage 1] Operator validation in staging
    Set LOADABLE_<AGENT>=1 in staging
    Run 25-50 representative task panel
    Compare success rate (≥0.90× legacy), token usage (≤Phase 1c × 1.15)
    ↓ pass
[Stage 2] Default flipped to ON in code
    Modify is_loadable_for() default, OR
    Add LOADABLE_<AGENT>=1 to deployment env
    Soak for ≥7 days; legacy still available via LOADABLE_<AGENT>=0
    ↓ stable
[Stage 3] Legacy factory deleted
    Drop _legacy_create_<agent> from app/agents/<agent>.py
    Drop dispatcher (or simplify to `return _build_loadable_<agent>(...)`)
    Per-agent cleanup PR
    ↓
[Stage 4] optional_tool_group consolidation (post-stage-3 sweep)
    Remove optional_tool_group(...) wrappers in <agent>.py
    Replace with registry guards (already enforced by build_loadable_agent)
    Trim unused imports
```

Stages can run in parallel across agents (introspector at Stage 3 while coder is at Stage 1 is fine), but each agent must traverse them in order.

---

## 3. Acceptance criteria for each stage

### Stage 1 → Stage 2 (default flip)

* `phase5_check` reports the agent as READY.
* Live parity: ≥0.90× legacy success rate on a 25-50 task panel representative of the agent's actual workload.
* Token usage: `analyze_telemetry(agent_id="<agent>")` reports `effective_input_tokens` ≤ Phase 1c prediction × 1.15 (i.e. ≤38% of stock).
* No new failure modes vs. legacy (manual review of failed tasks).

### Stage 2 → Stage 3 (legacy deletion)

* ≥7 days at default-on in production with no rollback.
* `/api/cp/tools/flags` shows the agent on `master flag` or `default` (not on a per-agent override) — meaning operators have stopped explicitly setting the flag because they trust the default.
* Per-agent error monitor shows no spike in agent-specific failures.

### Stage 3 → Stage 4 (optional_tool_group cleanup)

* All migrated agents at Stage 3.
* Replaced `optional_tool_group("<agent>", "<group>")` blocks have equivalent capability tags in `app/tool_registry/capabilities.py`.
* Tool's runtime `guard()` is the source of truth for env-config-based gating (rather than the import-time `optional_tool_group` block).

---

## 4. Why we're not doing all of this now

Three concrete reasons:

1. **No live parity data yet.** Phase 1c's 33% cache-cost win is analytical. The operator-driven 4-X.5 validation cycles haven't run because they require staging time + LLM budget. Without that data, flipping defaults assumes the analytical model is right within ±15% — plausible but unverified.

2. **Failsafe value is high right now, low later.** The dispatcher's try/except → legacy fallback is what makes Phase 4 zero-risk. Removing it before parity validation trades that safety for code tidiness — bad ratio. After parity validation, the legacy factory has earned its retirement.

3. **Per-agent independence.** Deleting all 4 legacy factories in one PR creates a single point of failure: any cleanup bug affects every agent. Agent-by-agent deletion (one PR per agent, per Stage 3) limits blast radius and gives operators the rollback granularity they already use via per-agent flags.

The readiness CLI + this doc make the eventual cleanup mechanical rather than judgment-heavy. That's the right output for Phase 5.

---

## 5. What ships in this PR

| Path | Role |
|------|------|
| `app/tool_runtime/phase5_check.py` | The readiness CLI. Public API: `check_agent(name)`, `run_full_check()`, `render_report(report)`. |
| `tests/test_phase_5_readiness.py` | 12 tests across per-agent checks, aggregate verdict, failure detection, env restoration. |
| `docs/TOOL_REGISTRY_PHASE_5.md` | This doc. |

What does NOT ship:

* No agent-default flips. All four migrated agents stay at `default OFF` until their parity panels pass.
* No legacy-factory deletion. Each `_legacy_create_<agent>` stays as the failsafe fallback.
* No `optional_tool_group` removal. That's Stage 4 cleanup, after all four agents complete Stage 3.

---

## 6. Operator workflow examples

### Pre-deploy sanity check

```bash
# Before promoting to staging or production:
docker exec crewai-team-gateway-1 python -m app.tool_runtime.phase5_check
# → exits 0 if everything's fine; 1 if a regression slipped in.
```

Use as a CI gate. If `phase5_check` returns non-zero, don't deploy.

### Post-incident triage

```bash
# Was the agent on legacy or loadable when the incident happened?
curl -s http://localhost:8000/api/cp/tools/flags | jq

# What does the readiness check say now?
docker exec crewai-team-gateway-1 python -m app.tool_runtime.phase5_check --json | \
  jq '.agents[] | select(.agent == "researcher")'
```

### Promoting an agent to default-on (Stage 1 → Stage 2)

After your parity panel passes for, say, the introspector:

1. Open a PR that changes `is_loadable_for("introspector")` to default-on (or adds `LOADABLE_INTROSPECTOR=1` to deployment env).
2. Title: `feat(tool-registry): Phase 5 — promote introspector to default-on`.
3. Tests: re-run `phase5_check` (should still pass), verify `tests/test_tool_runtime_phase_2.py::test_default_is_legacy_path` is updated to reflect the new default.
4. Soak ≥7 days. If anything goes wrong, set `LOADABLE_INTROSPECTOR=0` in the deployment env to roll back.

### Deleting an agent's legacy factory (Stage 2 → Stage 3)

After ≥7 days at default-on with no rollback:

1. Open a PR that removes `_legacy_create_<agent>` from the agent's file.
2. Simplify the dispatcher: `def create_<agent>(...) -> Agent: return _build_loadable_<agent>(...)`.
3. Drop the failsafe try/except — at this point the loadable path is the only path.
4. Update the dispatcher tests (the `test_default_is_legacy_path` and `test_loadable_failure_falls_back_to_legacy` cases no longer apply).
5. Title: `cleanup(tool-registry): Phase 5 — drop introspector legacy factory`.

---

## 7. Migration program — final state

| # | Phase | Status |
|--:|---|---|
| 0 | Spike + measurement | DONE |
| 1a | Registry foundation (#39) | DONE |
| 1b | `tool_search` (#40) | DONE |
| 1c | Cache-cost gate (#41) | DONE |
| 2 | Pilot — introspector (#42) | DONE |
| 3 | Forge bridge (#43) | DONE |
| 4a | Researcher migration (#44) | DONE |
| 4b | Writer migration (#45) | DONE |
| 4c | Coder migration (#46) | DONE |
| 4d | Commander n/a + flags endpoint (#47) | DONE |
| **5** | **Readiness work — phase5_check + rollout doc (this PR)** | **THIS PR** |
| 5b+ | Per-agent default flips + legacy deletion | operator-driven, post parity |

The migration program is **complete in terms of architecture**. What remains is operator-managed rollout per the sequence above.

---

## 8. The shape of the system at the end

When all four agents have completed Stages 1–4, the codebase has:

* **One agent factory per agent**, returning a `LoadableAgent`. Roughly 30 lines instead of today's 90-line dual-path dispatcher.
* **No `optional_tool_group`**. Every conditional toolset is expressed as a registry capability + a `guard()` callable on the tool's `@register_tool` decoration.
* **Universal `tool_search` access**. Every agent can discover Forge-bridged SHADOW/CANARY tools, registry-annotated tools, and (eventually) any new tool an operator adds — without an agent rewrite.
* **One source of truth** for tool descriptions: the `description` field on each `@register_tool` annotation. No duplicate prose between `souls/` and tool files.

The token-cost math from Phase 1c says this should cost ~33% of stock. The behavior parity from Phase 4 says the LLM gets the same toolset to work with. The registry from Phase 1a says new tools land in the catalog automatically.

Everything else after Phase 5 readiness is operator workflow, not engineering.
