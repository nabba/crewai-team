"""Drift test for the structural inventory — WP G Phase 0.

The structural inventory (``app.tools.structural_inventory``) captures
the public surface of the codebase: every FastAPI route, every public
top-level callable/class in the five choke-point modules, every idle
job. This test fails if anything in that snapshot changes without an
explicit baseline update.

Why this exists:
  Refactor PRs (especially WP G Phase 1+) move code between files. The
  inventory tool's stable sort makes "moved a route between modules"
  show up as a clean diff. A route that vanishes — by accident — is
  caught here before merge.

To update the baseline (when changes are intentional):

    .venv/bin/python -m app.tools.structural_inventory --write-baseline

Or rerun the inventory and pipe to the baseline path manually if you
need to inspect the diff first:

    .venv/bin/python -m app.tools.structural_inventory > /tmp/new.json
    diff -u tests/baselines/structural_inventory.json /tmp/new.json
    cp /tmp/new.json tests/baselines/structural_inventory.json
"""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.tools.structural_inventory import collect_inventory  # noqa: E402


BASELINE = Path(__file__).parent / "baselines" / "structural_inventory.json"


def _diff_summary(baseline: dict, current: dict) -> str:
    """Return a human-friendly diff. Empty string means no drift."""
    lines: list[str] = []

    # Routes
    base_routes = {(r["method"], r["path"], r["handler"]) for r in baseline.get("routes", [])}
    cur_routes = {(r["method"], r["path"], r["handler"]) for r in current.get("routes", [])}
    added_routes = sorted(cur_routes - base_routes)
    removed_routes = sorted(base_routes - cur_routes)
    if added_routes:
        lines.append(f"ROUTES ADDED ({len(added_routes)}):")
        for r in added_routes[:20]:
            lines.append(f"  + {r[0]} {r[1]}  ({r[2]})")
        if len(added_routes) > 20:
            lines.append(f"  ... and {len(added_routes) - 20} more")
    if removed_routes:
        lines.append(f"ROUTES REMOVED ({len(removed_routes)}):")
        for r in removed_routes[:20]:
            lines.append(f"  - {r[0]} {r[1]}  ({r[2]})")
        if len(removed_routes) > 20:
            lines.append(f"  ... and {len(removed_routes) - 20} more")

    # Module reassignment (route still exists, just moved files).
    # We don't fail on this — moving routes between modules is the point
    # of WP G — but we surface it for the PR description.
    base_by_key = {
        (r["method"], r["path"], r["handler"]): r["module"]
        for r in baseline.get("routes", [])
    }
    cur_by_key = {
        (r["method"], r["path"], r["handler"]): r["module"]
        for r in current.get("routes", [])
    }
    moved = [
        (k, base_by_key[k], cur_by_key[k])
        for k in cur_by_key
        if k in base_by_key and cur_by_key[k] != base_by_key[k]
    ]
    if moved:
        lines.append(f"ROUTES RE-HOMED ({len(moved)}) — not a failure, just for the record:")
        for k, src, dst in moved[:10]:
            lines.append(f"  ~ {k[0]} {k[1]}  ({k[2]}): {src} → {dst}")
        if len(moved) > 10:
            lines.append(f"  ... and {len(moved) - 10} more")

    # Idle jobs
    base_jobs = {(j["name"], j["weight"]) for j in baseline.get("idle_jobs", [])}
    cur_jobs = {(j["name"], j["weight"]) for j in current.get("idle_jobs", [])}
    added_jobs = sorted(cur_jobs - base_jobs)
    removed_jobs = sorted(base_jobs - cur_jobs)
    if added_jobs:
        lines.append(f"IDLE JOBS ADDED ({len(added_jobs)}):")
        for j in added_jobs:
            lines.append(f"  + {j[0]} ({j[1]})")
    if removed_jobs:
        lines.append(f"IDLE JOBS REMOVED ({len(removed_jobs)}):")
        for j in removed_jobs:
            lines.append(f"  - {j[0]} ({j[1]})")

    # Choke-point surface
    base_cp = baseline.get("choke_points", {})
    cur_cp = current.get("choke_points", {})
    for mod in sorted(set(base_cp) | set(cur_cp)):
        b = base_cp.get(mod, {})
        c = cur_cp.get(mod, {})
        if not b or not c:
            continue
        b_defs = set(b.get("public_defs", []))
        c_defs = set(c.get("public_defs", []))
        b_cls = set(b.get("classes", []))
        c_cls = set(c.get("classes", []))
        added_d = sorted(c_defs - b_defs)
        removed_d = sorted(b_defs - c_defs)
        added_c = sorted(c_cls - b_cls)
        removed_c = sorted(b_cls - c_cls)
        if added_d or removed_d or added_c or removed_c:
            lines.append(f"CHOKE POINT {mod}:")
            for n in added_d:
                lines.append(f"  + def {n}")
            for n in removed_d:
                lines.append(f"  - def {n}")
            for n in added_c:
                lines.append(f"  + class {n}")
            for n in removed_c:
                lines.append(f"  - class {n}")

    return "\n".join(lines)


def _drift_is_only_moved_routes(baseline: dict, current: dict) -> bool:
    """True iff the only delta is route module-reassignment.

    Route counts and identities (method+path+handler) must match exactly.
    Moves between modules are tolerated — WP G Phase 1 deliberately
    re-homes routes from dashboard_api.py into topic submodules.
    """
    # Compare every section except "routes"
    base_no_routes = {k: v for k, v in baseline.items() if k != "routes"}
    cur_no_routes = {k: v for k, v in current.items() if k != "routes"}
    if base_no_routes != cur_no_routes:
        return False
    # Routes: identity must match, module may move
    base_routes = {(r["method"], r["path"], r["handler"]) for r in baseline.get("routes", [])}
    cur_routes = {(r["method"], r["path"], r["handler"]) for r in current.get("routes", [])}
    return base_routes == cur_routes


def test_baseline_exists():
    """The baseline file must be checked in."""
    assert BASELINE.exists(), (
        f"baseline missing — recreate with: "
        f"python -m app.tools.structural_inventory --write-baseline"
    )


def test_inventory_matches_baseline():
    """Fail loudly on any structural drift not yet acknowledged."""
    baseline = json.loads(BASELINE.read_text())
    current = collect_inventory()
    if baseline == current:
        return
    # Soft case: only difference is module-of-origin for routes that
    # otherwise match exactly. That's an intentional re-home; the
    # baseline still needs updating but we report it informationally
    # rather than failing the test.
    if _drift_is_only_moved_routes(baseline, current):
        diff = _diff_summary(baseline, current)
        pytest.fail(
            "Inventory drift detected (route re-homing only).\n"
            "If intentional, update the baseline:\n"
            "  python -m app.tools.structural_inventory --write-baseline\n\n"
            + diff,
        )
    diff = _diff_summary(baseline, current)
    pytest.fail(
        "Structural inventory has drifted from baseline.\n"
        "If the change is intentional, update the baseline with:\n"
        "  python -m app.tools.structural_inventory --write-baseline\n\n"
        + diff,
    )
