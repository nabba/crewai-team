# Evolution Subsystems — Cheat Sheet

The eight evolution-related modules each have a distinct responsibility.
This file exists so newcomers (and the Self-Improver itself) can answer
"which one should I touch?" without reading 12,000 lines of code.

| Module | Tier | Owns |
|---|---|---|
| [`evolution.py`](evolution.py) | GATED | The continuous improvement **loop**. Karpathy autoresearch: propose → eval → measure → record. Single-population. The default entry point most callers reach. |
| [`evolution_roi.py`](evolution_roi.py) | IMMUTABLE | **Cost / value tracking + auto-throttle.** Per-experiment USD spend, productive-cycle ratio, 14-day ROI floor that slows the loop when nothing's landing. Read-only signal — does not propose. |
| [`evolve_blocks.py`](evolve_blocks.py) | GATED | **FREEZE-BLOCK / EVOLVE-BLOCK markers** inside prompt and soul files. The mutation engine refuses any diff that touches a FREEZE region. The marker infrastructure, not a loop. |
| [`island_evolution.py`](island_evolution.py) | GATED | **Island-population variant.** Multiple populations evolve in parallel; top performers migrate between islands every N generations. Different exploration strategy from `evolution.py`'s single-population loop — they coexist, not replace. |
| [`parallel_evolution.py`](parallel_evolution.py) | GATED | **Diverse-archive variant** that explores several mutation strategies in parallel against 2–3 Docker sandboxes. MAP-Elites-style stepping-stone archive. Sibling to island, different shape. |
| [`meta_evolution.py`](meta_evolution.py) | IMMUTABLE | **Second-order:** evolves the *evolution engine's* prompts (avo_planning_prompt.md, avo_critique_prompt.md). Cannot modify itself (cycle prevention). |
| [`evolution_db/`](evolution_db/) | mixed | **Persistence layer.** `archive_db.py` (variant archive), `eval_sets.py` (reference task lists), `judge.py` (LLM-as-judge scoring helpers). Pure storage + I/O — no orchestration. |
| [`evolution_suite/`](evolution_suite/) | facade | Re-exports from `evolution_db` so callers can `from app.evolution_suite import ArchiveDB` without coupling to the storage layout. |

## When to touch which

- **Add a new mutation strategy** → `parallel_evolution.py` (strategy registry).
- **Tune the gating thresholds** for which proposals get tried → `evolution.py`.
- **Add a cost gate** that blocks expensive proposals → `evolution_roi.py` (IMMUTABLE — operator only).
- **Mark a soul-file region as frozen** → annotate with `<!-- FREEZE-BLOCK -->` markers; `evolve_blocks.py` enforces.
- **Run multiple populations** with periodic migration → `island_evolution.py`.
- **Evolve the evolver's prompts** → `meta_evolution.py` (IMMUTABLE — operator only).
- **Read past variants from the archive** → `evolution_db.archive_db`.

## Cross-references

- Eval scoring lives in `app/cascade_evaluator.py` and `app/eval_sandbox.py` (IMMUTABLE).
- Promotion gating lives in `app/governance.py` and `app/tier_graduation.py` (IMMUTABLE).
- Canary deployment lives in `app/canary_deploy.py` (GATED).
- See `PROGRAM.md` for phase-by-phase status of the evolution roadmap.

> *Phase G6: this README replaces the prior "eight evolution-* siblings,
> no cheat sheet" tech-debt finding.*
