# Code-health continuous observation (Phase 1 of elegance plan)

Status: shipped 2026-05-18, default ON, observational.

## Why this exists

`app/code_quality.py` and `app/architectural_review.py` are mutation
**gates**. They score new code before AVO ships it and reject regressions.
They do nothing about code that quietly rots after merge: a hand-edit
that drops type hints, a refactor that leaves a 25-McCabe branch in
place, a new module that re-implements something that already existed.

This phase adds the **continuous-observation loops** that surface those
post-merge drifts. It also closes the meta-gap behind CLAUDE.md drifting
from actual capabilities: `system_inventory` is the auto-generated
catalogue agents query at runtime, so the AVO planner and the operator
both reason from the live truth instead of a stale narrative.

The mutation gates and the new observation loops are complementary —
the gate is the *pre-merge* check, the loops are the *post-merge*
sweep. They share the same primitives (`code_quality.measure_file_quality`,
`architectural_review`-equivalent capability/cycle detection) so there
is exactly one definition of "regression."

## What ships

| Subsystem | Role | Master switch |
|---|---|---|
| `app/system_inventory/` | Auto-catalogue at `workspace/system_inventory/snapshot.json` | `system_inventory_enabled` (default ON) |
| `app/healing/monitors/elegance_drift.py` | Weekly per-file `QualityScore` regression detector | `elegance_drift_monitor_enabled` (default ON) |
| `app/healing/monitors/architectural_drift.py` | Weekly full-graph cycle / capability-overlap / centrality-spike detector | `architectural_drift_monitor_enabled` (default ON) |

All three default ON, observational, and route any signal through
`app.notify.notify` plus the identity continuity ledger event kind
**`architectural_debt_drift`** (auto-surfaced by `summarise_drift` into
the annual reflection).

## `system_inventory`

`app/system_inventory/scanner.py` does an AST-only walk of `app/**/*.py`:
no imports, no side effects, so a daemon scanner can't accidentally
boot a subsystem. Per-module it extracts:

- `path` — repo-relative
- `kind` — `package` (an `__init__.py`) or `module`
- `summary` — first non-blank line of the module docstring
- `public_symbols` — top-level `def`/`async def`/`class` whose names
  don't start with `_`
- `capabilities` — tags pulled from `@register_tool(capabilities=[…])`
  decorators (structural regex on the AST)
- `loc` — non-blank, non-comment line count
- `has_tests` — sibling `test_<stem>.py` exists under `tests/`

`app/system_inventory/store.py` persists the snapshot at
`workspace/system_inventory/snapshot.json` and exposes:

```python
from app.system_inventory import (
    build_snapshot, get_snapshot, query_inventory, inventory_summary,
)
```

Typical agent use:

```python
inventory_summary()
# → "system_inventory@2026-05-18: 1206 modules (136 packages) ·
#    251,887 LOC · 20% with tests · 18 registered-tool capabilities"

query_inventory(capability="reads-attachment")
# → list[ModuleEntry] for tools that claim this capability

query_inventory(keyword="evolution", limit=5)
# → list[ModuleEntry] for modules matching by path or docstring
```

The snapshot is the **live source of truth** going forward. CLAUDE.md
stays a stable narrative — the inventory catches every new file the
moment it lands without operator effort.

## `elegance_drift` monitor (41st)

- Daily probe, weekly internal cadence (`_INTERNAL_CADENCE_S = 7 days`).
- Iterates `app/**/*.py`, calls `code_quality.measure_file_at_path` per
  file, appends a `{ts, composite}` sample to a rolling history at
  `workspace/code_quality/elegance_history.json`.
- History capped at 26 samples per file (≈6 months at weekly cadence).
- A file is **regressed** when its current composite is below
  `median(last_8_samples) - 0.10`. This matches the threshold used by
  the mutation gate (`code_quality.QUALITY_REGRESSION_THRESHOLD`).
- First-encounter files need at least 4 prior samples before regression
  classification — `verdict = "baseline"` until then.
- Per-pass alert: a single Signal notification listing the top-5 worst
  regressors (`📉 Code-elegance regression`); one continuity-ledger
  `architectural_debt_drift` event with the full list.
- Master switch and cadence-gate both honored; alert routing is
  failure-isolated.

## `architectural_drift` monitor (42nd)

Companion observer focused on the *shape* of the codebase rather than
per-file quality. Probes once a week.

- Builds the forward import graph from sources (no `self_model`
  dependency). Three import shapes handled: `import app.x.y`, `from
  app.x import y` (emits both `app/x/y.py` and `app/x/__init__.py`),
  and `from app import x` (emits both `app/x.py` and `app/x/__init__.py`).
- Walks Tarjan's iterative SCC to find every cycle in one O(V+E) pass.
- Detects three drift kinds versus a persisted baseline:
  1. **New small cycles** (size ≤ `_MAX_ALERTABLE_CYCLE_SIZE = 20`) —
     systemic SCCs are tracked but excluded from the actionable list
     (they're coupling shapes, not isolated refactors).
  2. **New parallel capabilities** — a `@register_tool` capability now
     claimed by ≥3 distinct files that wasn't already parallel.
  3. **Centrality spikes** — a file's reverse-import-degree grew ≥5×
     versus baseline AND the baseline was ≥3 (so noise from "1 importer
     became 5" doesn't fire).
- Bonus signal: **systemic growth** — when the largest oversized SCC
  grows by ≥10 files AND ≥10%, that gets its own alert line. A
  monotonically-growing systemic SCC is the slow tell that the
  codebase's overall coupling is worsening, not just one cycle.
- First run never alerts — it just records the baseline at
  `workspace/code_quality/architectural_baseline.json`.

## Compose with existing primitives

- **Mutation gate** (`code_quality.evaluate_mutation_quality`): unchanged.
  AVO continues to reject regressions at submission time. The
  `elegance_drift` monitor catches what slipped through (operator
  hand-edits, multi-file refactors that diluted single-file scores).
- **`architectural_review.review_mutation`**: unchanged. It still fires
  per-mutation. The `architectural_drift` monitor catches drift that
  accumulates across many mutations none of which were individually
  large enough to flag.
- **Identity continuity ledger**: new event kind
  `architectural_debt_drift` — `summarise_drift.by_kind` is a dynamic
  Counter, so this auto-surfaces in the annual reflection without
  composer changes.
- **`auto_revert`** (60-min watcher for change-requests) is unaffected —
  these monitors observe, they don't propose CRs at this phase.

## What this phase deliberately does NOT do

- **No refactor-CR producer.** Phase 2 of the elegance plan adds a 4th
  producer in `proposal_bridge/` that consumes these signals and files
  refactor proposals. Phase 1 is "see clearly" — Phase 2 is "act."
- **No mutation of the existing `code_quality` thresholds.** The
  `QUALITY_REGRESSION_THRESHOLD = 0.10` constant is the same one the
  mutation gate uses; we deliberately don't drift the meaning of
  "regression" between the gate and the loop.
- **No TIER_IMMUTABLE edits.** Everything additive.
- **No alerts from systemic SCCs by default.** A 500-file SCC isn't
  actionable in one refactor pass; only the *growth* of one is alerted.

## Operator visibility

- `workspace/system_inventory/snapshot.json` — live catalogue
- `workspace/code_quality/elegance_history.json` — per-file 26-sample
  rolling history
- `workspace/code_quality/architectural_baseline.json` — last week's
  SCC + capability-owner + reverse-degree snapshot
- `workspace/healing/elegance_drift_state.json` —
  `last_run` + last summary
- `workspace/healing/architectural_drift_state.json` — same shape

Disable any one without affecting the others: each monitor checks its
own switch independently.

## Tests

- `tests/test_system_inventory.py` — 8 tests covering scan, docstring
  extraction, public-symbol filtering, capability extraction (list +
  singular kwargs), persist round-trip, query filters, summary shape.
- `tests/healing/test_elegance_drift.py` — 8 tests covering baseline
  fill, cadence gate, disabled short-circuit, classification thresholds,
  history cap, end-to-end regression detection.
- `tests/healing/test_architectural_drift.py` — 12 tests covering
  Tarjan SCC on 2/3-node cycles, singleton ignore, capability owner
  dedup, systemic-size exclusion, first-run silence, second-run alert,
  cadence gate, systemic growth detection.

Total: 28 new tests, all passing.
