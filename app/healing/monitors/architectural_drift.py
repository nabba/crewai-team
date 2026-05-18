"""architectural_drift — continuous architectural-shape observation.

Companion to ``elegance_drift``. Where elegance_drift watches per-file
quality, this monitor watches the *shape* of the codebase: import cycles,
capability-ownership clusters, centrality spikes. Built from the same
primitives as ``app.architectural_review`` but pointed at the full
codebase instead of one mutation.

The existing ``review_mutation`` is mutation-focused — it DFS-walks
forward from the touched files and consults ``self_model``. For the
continuous case we want all SCCs in the import graph, computed without
needing the self-model to be live. Tarjan's iterative SCC gives both,
in one O(V+E) pass.

What we alert on (weekly)
-------------------------
1. A new SCC (cycle) that wasn't in last week's baseline.
2. A capability now claimed by ≥3 files (parallel implementation smell).
3. A file whose reverse-degree (number of importers) jumped >5× since
   its baseline — a load-bearing module appeared without intent.

Baseline lives in workspace/code_quality/architectural_baseline.json.
First run files no alerts — it just records the baseline.
"""
from __future__ import annotations

import ast
import json
import logging
import os
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


NAME = "architectural_drift"
CADENCE_SECONDS = 24 * 3600
MASTER_SWITCH_KEY = "architectural_drift_monitor_enabled"
_INTERNAL_CADENCE_S = 7 * 24 * 3600

# Capability is "parallel implementation" smell when ≥ this many files
# claim it. Matches ``architectural_review.NEW_FILE_OVERLAP_HARD_THRESHOLD``.
_CAPABILITY_PARALLEL_THRESHOLD = 3
# Centrality jump factor that triggers an alert (post/baseline).
_CENTRALITY_JUMP_FACTOR = 5.0
# Below this floor, a 5× jump (e.g. 1→5) is noise — only watch growing
# files that already have a meaningful presence.
_CENTRALITY_MIN_BASELINE = 3
# SCCs larger than this are *systemic* coupling, not an isolated cycle
# the operator can refactor. We still persist them so size-growth alerts
# are possible, but we omit them from the actionable "new cycle" body.
_MAX_ALERTABLE_CYCLE_SIZE = 20
_SKIP_PARTS: frozenset[str] = frozenset({"__pycache__", ".pytest_cache"})


def _workspace_root() -> Path:
    env = os.environ.get("WORKSPACE_ROOT")
    if env:
        return Path(env)
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT)
    except Exception:
        return Path("/app/workspace")


def _baseline_path() -> Path:
    return _workspace_root() / "code_quality" / "architectural_baseline.json"


def _state_path() -> Path:
    return _workspace_root() / "healing" / "architectural_drift_state.json"


def _app_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _enabled() -> bool:
    try:
        from app.runtime_settings import get_architectural_drift_monitor_enabled
        return get_architectural_drift_monitor_enabled()
    except Exception:
        return True


def _read_state() -> dict[str, Any]:
    p = _state_path()
    if not p.exists():
        return {"last_run": 0.0}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"last_run": 0.0}


def _write_state(state: dict[str, Any]) -> None:
    p = _state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(p)


def _cadence_due(state: dict[str, Any]) -> bool:
    return (time.time() - float(state.get("last_run") or 0)) >= _INTERNAL_CADENCE_S


# ── graph build ────────────────────────────────────────────────────────


def _local_imports(source: str) -> list[str]:
    """Return repo-relative paths for every ``app.*`` import.

    Three input shapes are handled:

      1. ``import app.healing.runbooks`` → ``app/healing/runbooks.py``
      2. ``from app.healing import runbooks`` → both ``app/healing/runbooks.py``
         (likely a submodule) and ``app/healing/__init__.py`` (the
         containing package — covers ``from app.healing import some_symbol``).
      3. ``from app import healing`` → both ``app/healing.py`` (module)
         and ``app/healing/__init__.py`` (package). The graph builder
         only follows the path that actually exists.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if not node.module:
                continue
            if node.module == "app":
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    out.append(f"app/{alias.name}.py")
                    out.append(f"app/{alias.name}/__init__.py")
            elif node.module.startswith("app."):
                pkg_path = node.module.replace(".", "/")
                # The package itself (covers `from app.X import some_symbol`).
                out.append(f"{pkg_path}/__init__.py")
                out.append(f"{pkg_path}.py")
                # Each imported name may be a submodule.
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    out.append(f"{pkg_path}/{alias.name}.py")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("app."):
                    out.append(alias.name.replace(".", "/") + ".py")
    return out


# Match @register_tool(... capabilities=[...] ...) without importing.
_CAPABILITIES_RE = re.compile(
    r"@register_tool\([^)]*capabilities\s*=\s*\[([^\]]*)\]",
    re.DOTALL,
)
_STRING_LITERAL_RE = re.compile(r'"([^"]+)"|\'([^\']+)\'')


def _capabilities(source: str) -> list[str]:
    out: list[str] = []
    for match in _CAPABILITIES_RE.finditer(source):
        for a, b in _STRING_LITERAL_RE.findall(match.group(1)):
            out.append(a or b)
    return out


def _build_graph(app_root: Path) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Return (forward_graph, capability_owners).

    Owners are deduped per capability — a file with multiple
    ``@register_tool`` decorators for the same capability counts once.
    """
    forward: dict[str, list[str]] = {}
    owners_per_cap: dict[str, set[str]] = defaultdict(set)
    repo_root = app_root.parent
    for path in sorted(app_root.rglob("*.py")):
        if any(part in _SKIP_PARTS for part in path.parts):
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except OSError:
            continue
        rel = str(path.relative_to(repo_root))
        forward[rel] = _local_imports(source)
        for cap in _capabilities(source):
            owners_per_cap[cap].add(rel)
    capability_owners = {cap: sorted(files) for cap, files in owners_per_cap.items()}
    return forward, capability_owners


# ── Tarjan SCC (iterative) ─────────────────────────────────────────────


def _strongly_connected_components(
    graph: dict[str, list[str]],
) -> list[list[str]]:
    """All SCCs. Bookkeeping is iterative so deep graphs don't blow stack.

    Returns each component in deterministic order (sorted by min member),
    with components themselves sorted internally — so a cycle's "name"
    is stable across runs and diffs cleanly.
    """
    index_counter = [0]
    stack: list[str] = []
    on_stack: dict[str, bool] = {}
    index: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    result: list[list[str]] = []

    def strongconnect(start: str) -> None:
        work: list[tuple[str, int]] = [(start, 0)]
        call_stack: list[str] = []
        while work:
            node, pi = work[-1]
            if pi == 0:
                index[node] = index_counter[0]
                lowlink[node] = index_counter[0]
                index_counter[0] += 1
                stack.append(node)
                on_stack[node] = True
            neighbors = graph.get(node, [])
            if pi < len(neighbors):
                work[-1] = (node, pi + 1)
                child = neighbors[pi]
                if child not in index:
                    work.append((child, 0))
                    call_stack.append(node)
                elif on_stack.get(child, False):
                    lowlink[node] = min(lowlink[node], index[child])
                continue
            if lowlink[node] == index[node]:
                comp: list[str] = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    comp.append(w)
                    if w == node:
                        break
                if len(comp) > 1:  # only proper cycles
                    result.append(sorted(comp))
            work.pop()
            if call_stack:
                parent = call_stack.pop()
                lowlink[parent] = min(lowlink[parent], lowlink[node])

    for v in graph:
        if v not in index:
            strongconnect(v)
    result.sort(key=lambda c: c[0])
    return result


# ── classification ─────────────────────────────────────────────────────


def _reverse_degree(forward: dict[str, list[str]]) -> Counter:
    rev: Counter = Counter()
    for _, imports in forward.items():
        for target in imports:
            rev[target] += 1
    return rev


def _new_cycles(
    cycles: list[list[str]], baseline_cycles: list[list[str]],
) -> list[list[str]]:
    """Cycles in the current scan that aren't in the baseline.

    Two cycles match if they have identical member sets (we sort members
    when extracting the SCC, so direct list comparison is canonical).
    Systemic SCCs (size > _MAX_ALERTABLE_CYCLE_SIZE) are excluded — they
    are coupling shapes, not isolated cycles the operator can refactor
    in one pass, and their size growth is tracked separately via
    ``_systemic_growth``.
    """
    seen = {tuple(c) for c in baseline_cycles}
    return [
        c for c in cycles
        if tuple(c) not in seen and len(c) <= _MAX_ALERTABLE_CYCLE_SIZE
    ]


def _systemic_growth(
    cycles: list[list[str]], baseline_cycles: list[list[str]],
) -> dict[str, Any] | None:
    """Detect when the largest systemic SCC grows past the baseline.

    Returns a small dict when growth crosses ≥10% AND ≥10 members, else
    None. We deliberately don't fire on every member churn — only on
    sustained growth that signals the codebase is becoming *more*
    entangled, not less.
    """
    largest_now = max((len(c) for c in cycles), default=0)
    largest_prior = max((len(c) for c in baseline_cycles), default=0)
    if largest_now <= _MAX_ALERTABLE_CYCLE_SIZE:
        return None
    if largest_prior == 0:
        return None  # first run handled by first_run gate
    delta = largest_now - largest_prior
    if delta < 10 or largest_now < largest_prior * 1.1:
        return None
    return {
        "prior_size": largest_prior,
        "current_size": largest_now,
        "delta": delta,
    }


def _new_parallel_capabilities(
    owners: dict[str, list[str]], baseline_owners: dict[str, list[str]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for cap, files in owners.items():
        if len(files) < _CAPABILITY_PARALLEL_THRESHOLD:
            continue
        prior = baseline_owners.get(cap) or []
        if len(prior) >= _CAPABILITY_PARALLEL_THRESHOLD:
            continue  # already known parallel
        out.append({
            "capability": cap,
            "owners": sorted(files),
            "prior_count": len(prior),
        })
    return out


def _new_centrality_spikes(
    rev: Counter, baseline_rev: dict[str, int],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path, dependents in rev.items():
        prior = int(baseline_rev.get(path, 0))
        if prior < _CENTRALITY_MIN_BASELINE:
            continue
        if dependents >= _CENTRALITY_JUMP_FACTOR * prior:
            out.append({
                "path": path,
                "prior_dependents": prior,
                "current_dependents": dependents,
                "factor": round(dependents / max(1, prior), 2),
            })
    out.sort(key=lambda d: -d["factor"])
    return out


# ── persistence ────────────────────────────────────────────────────────


def _read_baseline() -> dict[str, Any]:
    p = _baseline_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_baseline(baseline: dict[str, Any]) -> None:
    p = _baseline_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(baseline, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(p)


# ── alert ──────────────────────────────────────────────────────────────


def _emit_alert(
    new_cycles: list[list[str]],
    new_parallels: list[dict[str, Any]],
    new_spikes: list[dict[str, Any]],
    systemic: dict[str, Any] | None,
) -> None:
    body_lines: list[str] = []
    if new_cycles:
        body_lines.append(f"⟳ {len(new_cycles)} new import cycle(s):")
        for c in new_cycles[:3]:
            body_lines.append(f"  • {' → '.join(c)}")
    if new_parallels:
        body_lines.append(f"⫝ {len(new_parallels)} new parallel capability:")
        for p in new_parallels[:3]:
            owners = ", ".join(Path(o).name for o in p["owners"][:3])
            body_lines.append(f"  • '{p['capability']}' owned by {owners}")
    if new_spikes:
        body_lines.append(f"↑ {len(new_spikes)} centrality spike(s):")
        for s in new_spikes[:3]:
            body_lines.append(
                f"  • {s['path']}: {s['prior_dependents']} → "
                f"{s['current_dependents']} dependents (×{s['factor']})"
            )
    if systemic:
        body_lines.append(
            f"⊙ Largest systemic SCC grew "
            f"{systemic['prior_size']} → {systemic['current_size']} "
            f"(+{systemic['delta']} files)"
        )
    try:
        from app.notify import notify
        notify(
            title="🏛 Architectural drift",
            body="\n".join(body_lines) or "Drift detected (details in state file)",
            url="/cp/code-health",
            topic="architectural_drift",
            arbitrate=True,
        )
    except Exception:
        logger.debug("architectural_drift: notify failed", exc_info=True)
    try:
        from app.identity.continuity_ledger import record_event
        record_event(
            kind="architectural_debt_drift",
            actor="architectural_drift_monitor",
            summary=(
                f"new_cycles={len(new_cycles)} "
                f"new_parallel_caps={len(new_parallels)} "
                f"new_centrality_spikes={len(new_spikes)}"
                + (f" systemic_growth=+{systemic['delta']}" if systemic else "")
            ),
            detail={
                "new_cycles": new_cycles[:5],
                "new_parallel_capabilities": new_parallels[:5],
                "new_centrality_spikes": new_spikes[:5],
                "systemic_growth": systemic,
            },
        )
    except Exception:
        logger.debug("architectural_drift: ledger emit failed", exc_info=True)


# ── public entry ────────────────────────────────────────────────────────


def run() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "checked": False,
        "n_files": 0,
        "n_cycles": 0,
        "n_new_cycles": 0,
        "n_new_parallels": 0,
        "n_new_spikes": 0,
    }
    if not _enabled():
        summary["disabled"] = True
        return summary
    state = _read_state()
    if not _cadence_due(state):
        summary["skipped_cadence"] = True
        return summary

    try:
        app_root = _app_root()
        forward, capability_owners = _build_graph(app_root)
        summary["n_files"] = len(forward)
        cycles = _strongly_connected_components(forward)
        rev = _reverse_degree(forward)
        summary["n_cycles"] = len(cycles)

        baseline = _read_baseline()
        baseline_cycles = [list(c) for c in baseline.get("cycles", [])]
        baseline_owners = baseline.get("capability_owners", {}) or {}
        baseline_rev = baseline.get("reverse_degree", {}) or {}

        new_cycles = _new_cycles(cycles, baseline_cycles)
        new_parallels = _new_parallel_capabilities(capability_owners, baseline_owners)
        new_spikes = _new_centrality_spikes(rev, baseline_rev)
        systemic = _systemic_growth(cycles, baseline_cycles)
        summary["n_new_cycles"] = len(new_cycles)
        summary["n_new_parallels"] = len(new_parallels)
        summary["n_new_spikes"] = len(new_spikes)
        summary["systemic_growth"] = systemic

        # First run = no baseline → no alerts, just record.
        first_run = not baseline
        if not first_run and (new_cycles or new_parallels or new_spikes or systemic):
            _emit_alert(new_cycles, new_parallels, new_spikes, systemic)

        _write_baseline({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "cycles": cycles,
            "capability_owners": capability_owners,
            "reverse_degree": dict(rev),
        })
        state["last_run"] = time.time()
        state["last_summary"] = {
            k: v for k, v in summary.items()
            if k not in ("top_regressors",)  # parity placeholder
        }
        _write_state(state)
        summary["checked"] = True
        summary["first_run"] = first_run
    except Exception:
        logger.debug("architectural_drift: probe failed", exc_info=True)
        summary.setdefault("errors", 0)
        summary["errors"] += 1
    return summary
