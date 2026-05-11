"""L4.3 — Bridge / cut-vertex identification.

PROGRAM §42 (2026-05-11) — Q4.2 Level 4.3.

Tarjan's algorithm:
  * **Bridges**: edges whose removal disconnects the graph
                  (structural connections)
  * **Cut-vertices** (articulation points): nodes whose removal
                                              disconnects clusters
                                              (structural intermediaries)

UI discipline: surfaced with explicit caveat — *"This is what the
algorithm sees from co-appearance patterns. It is not necessarily
what your relationships actually are."*

Master switches:
  * L1 + L4 master + ``graph_bridges_enabled`` (L4.3 master)

Persists to ``workspace/companion/social_graph_structural.json``
(also caught by social_graph DR denylist fragment).
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _enabled() -> bool:
    try:
        from app.runtime_settings import (
            get_person_correlation_enabled,
            get_person_correlation_social_graph_enabled,
            get_graph_bridges_enabled,
        )
        return (
            get_person_correlation_enabled()
            and get_person_correlation_social_graph_enabled()
            and get_graph_bridges_enabled()
        )
    except Exception:
        return False


def _default_structural_path() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT) / "companion" / "social_graph_structural.json"
    except Exception:
        return Path("/app/workspace/companion/social_graph_structural.json")


# ── Tarjan's bridges + articulation points (iterative, pure-Python) ─────


def _find_bridges_and_articulations(
    adj: dict[str, dict[str, float]],
) -> tuple[list[tuple[str, str]], set[str]]:
    """Iterative Tarjan. Returns (bridges, articulation_points).

    Iterative to avoid Python recursion-limit issues on large graphs.
    """
    if not adj:
        return [], set()

    # Bump recursion limit minimally; we use iterative DFS, but some
    # helper computations may go deep.
    prev_limit = sys.getrecursionlimit()
    if prev_limit < 10000:
        sys.setrecursionlimit(10000)

    nodes = list(adj.keys())
    disc: dict[str, int] = {}
    low: dict[str, int] = {}
    parent: dict[str, str | None] = {}
    children_of_root: dict[str, int] = {}
    counter = [0]

    bridges: list[tuple[str, str]] = []
    articulations: set[str] = set()

    for start in nodes:
        if start in disc:
            continue
        # Iterative DFS using a stack of (node, parent, neighbor_iter).
        parent[start] = None
        stack: list[tuple[str, str | None, iter]] = [
            (start, None, iter(adj[start].keys())),
        ]
        disc[start] = low[start] = counter[0]
        counter[0] += 1
        children_of_root[start] = 0

        while stack:
            node, par, neigh_iter = stack[-1]
            try:
                v = next(neigh_iter)
            except StopIteration:
                stack.pop()
                # Backtrack: update parent's low + check bridge/articulation.
                if par is not None:
                    low[par] = min(low[par], low[node])
                    if low[node] > disc[par]:
                        bridges.append(tuple(sorted([par, node])))
                    # Articulation: non-root with a child whose low[v] >= disc[par]
                    if parent.get(par) is not None and low[node] >= disc[par]:
                        articulations.add(par)
                continue

            if v == par:
                continue
            if v in disc:
                low[node] = min(low[node], disc[v])
                continue

            # Tree edge.
            parent[v] = node
            disc[v] = low[v] = counter[0]
            counter[0] += 1
            if node == start:
                children_of_root[start] += 1
            stack.append((v, node, iter(adj[v].keys())))

        # Root is articulation iff it has ≥2 DFS-children.
        if children_of_root[start] >= 2:
            articulations.add(start)

    sys.setrecursionlimit(prev_limit)
    return bridges, articulations


# ── Public ───────────────────────────────────────────────────────────────


def compute_structural() -> dict[str, Any]:
    """Find bridges + cut-vertices. Persist. Returns dict."""
    if not _enabled():
        return {"bridges": [], "cut_vertices": [], "enabled": False}

    try:
        from app.companion.social_graph import adjacency
        adj = adjacency()
    except Exception:
        return {"bridges": [], "cut_vertices": [], "enabled": True, "error": "graph unavailable"}

    if not adj:
        return {"bridges": [], "cut_vertices": [], "enabled": True}

    try:
        bridges, cut_vertices = _find_bridges_and_articulations(adj)
    except Exception:
        logger.exception("bridges: Tarjan raised")
        return {"bridges": [], "cut_vertices": [], "enabled": True, "error": "algorithm failed"}

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bridges": [list(b) for b in bridges],
        "cut_vertices": sorted(cut_vertices),
        "caveat": (
            "Bridges and cut-vertices are computed from co-appearance "
            "patterns alone. They identify structural roles in the "
            "observation graph — NOT necessarily real-world importance. "
            "A 'bridge' may just be someone who happens to attend two "
            "unrelated meetings."
        ),
    }

    try:
        p = _default_structural_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(p)
    except OSError:
        logger.debug("bridges: persist failed", exc_info=True)

    return payload


def is_bridge_or_cut(person_id: str) -> bool:
    """Quick lookup for the arbiter — is this person structurally
    important right now? Returns False when L4.3 off."""
    if not _enabled():
        return False
    pid = (person_id or "").strip().lower()
    if not pid:
        return False
    try:
        p = _default_structural_path()
        if not p.exists():
            return False
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if pid in (data.get("cut_vertices") or []):
        return True
    for b in (data.get("bridges") or []):
        if pid in b:
            return True
    return False
