"""L4.1 — Shortest-path queries (operator-initiated).

PROGRAM §42 (2026-05-11) — Q4.2 Level 4.1.

BFS from Andrus to a target person through the co-appearance graph.
Returns the chain ``Andrus → Maria → John → target`` with each hop's
edge weight + last-appearance date.

CRITICAL: operator-initiated only. Never auto-surfaced. Path queries
are logged at ``workspace/companion/social_graph_query_log.jsonl``
for operator transparency.

Per-person opt-out: people in ``social_graph_path_optouts.json`` are
NEVER returned as intermediate hops. They can be path-source or
path-target, but never a conduit.

Master switches:
  * ``person_correlation_enabled`` (L1)
  * ``person_correlation_social_graph_enabled`` (L4 master)
  * ``graph_shortest_path_enabled`` (L4.1 master)
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Any

logger = logging.getLogger(__name__)


def _enabled() -> bool:
    try:
        from app.runtime_settings import (
            get_person_correlation_enabled,
            get_person_correlation_social_graph_enabled,
            get_graph_shortest_path_enabled,
        )
        return (
            get_person_correlation_enabled()
            and get_person_correlation_social_graph_enabled()
            and get_graph_shortest_path_enabled()
        )
    except Exception:
        return False


def find_path(source: str, target: str, max_hops: int = 6) -> dict[str, Any]:
    """BFS from ``source`` to ``target``. Returns:
        {
            "ok": bool,
            "path": [person_id, ...] or None,
            "hops": int,
            "edge_weights": [float, ...],
            "skipped_opt_outs": [person_id, ...],   # people not used
            "error": str | None,
        }
    Excludes path-opt-out people from intermediate positions but
    allows them as start/end."""
    if not _enabled():
        return {
            "ok": False, "path": None, "hops": 0,
            "edge_weights": [], "skipped_opt_outs": [],
            "error": "shortest_path_disabled",
        }

    src = (source or "").strip().lower()
    tgt = (target or "").strip().lower()
    if not src or not tgt:
        return {
            "ok": False, "path": None, "hops": 0,
            "edge_weights": [], "skipped_opt_outs": [],
            "error": "source/target required",
        }
    if src == tgt:
        return {
            "ok": True, "path": [src], "hops": 0,
            "edge_weights": [], "skipped_opt_outs": [],
            "error": None,
        }

    try:
        from app.companion.social_graph import adjacency, load_path_opt_outs, log_query
    except Exception:
        return {
            "ok": False, "path": None, "hops": 0,
            "edge_weights": [], "skipped_opt_outs": [],
            "error": "social_graph unavailable",
        }
    adj = adjacency()
    opt_outs = load_path_opt_outs()

    if src not in adj:
        return {
            "ok": False, "path": None, "hops": 0,
            "edge_weights": [], "skipped_opt_outs": [],
            "error": f"source {src!r} not in graph",
        }

    # BFS with hop-cap.
    # parent[v] = predecessor node on shortest path from src.
    parent: dict[str, str | None] = {src: None}
    frontier: deque[tuple[str, int]] = deque([(src, 0)])
    found = False
    while frontier:
        node, depth = frontier.popleft()
        if depth >= max_hops:
            continue
        for nb in adj.get(node, {}):
            if nb in parent:
                continue
            # Opt-out check: skip if `nb` is opt-out AND not the target
            # (target is allowed; only intermediate hops are excluded).
            if nb in opt_outs and nb != tgt:
                continue
            parent[nb] = node
            if nb == tgt:
                found = True
                break
            frontier.append((nb, depth + 1))
        if found:
            break

    if not found or tgt not in parent:
        result = {
            "ok": False, "path": None, "hops": 0,
            "edge_weights": [], "skipped_opt_outs": sorted(opt_outs),
            "error": "no path found within hop limit",
        }
        log_query("shortest_path", {"source": src, "target": tgt, "found": False})
        return result

    # Reconstruct path.
    path: list[str] = []
    cur: str | None = tgt
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    path.reverse()

    edge_weights: list[float] = []
    for a, b in zip(path, path[1:]):
        edge_weights.append(float(adj.get(a, {}).get(b, 0.0)))

    log_query("shortest_path", {
        "source": src, "target": tgt, "found": True, "hops": len(path) - 1,
    })

    return {
        "ok": True,
        "path": path,
        "hops": len(path) - 1,
        "edge_weights": edge_weights,
        "skipped_opt_outs": sorted(opt_outs),
        "error": None,
    }
