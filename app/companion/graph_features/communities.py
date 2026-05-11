"""L4.2 — Community detection (label propagation).

PROGRAM §42 (2026-05-11) — Q4.2 Level 4.2.

Label-propagation clustering on the social graph. Simpler than Louvain;
deterministic-enough for repeatable surfaces; clearly an
"interpretation" rather than a "discovery."

UI discipline: clusters are surfaced UNNAMED ("Cluster A: 5 nodes,
density 0.7"). The system never describes what a cluster is "about."
No per-cluster aggregate stats. The modularity score is surfaced so
the operator can see whether the clustering is mathematically strong.

Operator can dissolve any cluster — hidden from future surfaces.

Master switches:
  * L1 + L4 master + ``graph_communities_enabled`` (L4.2 master)

Persists to ``workspace/companion/social_graph_communities.json``
(also caught by the social_graph DR denylist fragment).
"""
from __future__ import annotations

import json
import logging
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_MAX_ITERATIONS = 30   # label propagation usually converges in <20


def _enabled() -> bool:
    try:
        from app.runtime_settings import (
            get_person_correlation_enabled,
            get_person_correlation_social_graph_enabled,
            get_graph_communities_enabled,
        )
        return (
            get_person_correlation_enabled()
            and get_person_correlation_social_graph_enabled()
            and get_graph_communities_enabled()
        )
    except Exception:
        return False


def _default_communities_path() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT) / "companion" / "social_graph_communities.json"
    except Exception:
        return Path("/app/workspace/companion/social_graph_communities.json")


def _default_dissolved_path() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT) / "companion" / "social_graph_dissolved.json"
    except Exception:
        return Path("/app/workspace/companion/social_graph_dissolved.json")


# ── Dissolved clusters (operator-hidden) ─────────────────────────────────


def _load_dissolved() -> set[frozenset[str]]:
    p = _default_dissolved_path()
    if not p.exists():
        return set()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return set(
            frozenset(c) for c in (data.get("dissolved") or [])
            if isinstance(c, list)
        )
    except (OSError, json.JSONDecodeError):
        return set()


def dissolve_cluster(member_emails: list[str]) -> bool:
    """Hide a cluster (by its member set) from future surfaces.
    Returns True if newly dissolved."""
    if not member_emails:
        return False
    fs = frozenset(e.lower() for e in member_emails)
    dissolved = _load_dissolved()
    if fs in dissolved:
        return False
    dissolved.add(fs)
    p = _default_dissolved_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({
        "dissolved": [sorted(c) for c in dissolved],
    }, indent=2), encoding="utf-8")
    tmp.replace(p)
    return True


# ── Label propagation ────────────────────────────────────────────────────


def _label_propagation(
    adj: dict[str, dict[str, float]], seed: int = 42,
) -> dict[str, int]:
    """Run label propagation. Returns ``{node: cluster_id}``.

    Each node starts in its own cluster. At each iteration, each node
    adopts the most-common label among its neighbors (weighted by
    edge weight). Iterate until stable or max iterations."""
    if not adj:
        return {}
    rng = random.Random(seed)
    nodes = list(adj.keys())
    labels = {n: i for i, n in enumerate(nodes)}

    for _ in range(_MAX_ITERATIONS):
        changes = 0
        rng.shuffle(nodes)
        for node in nodes:
            neighbors = adj.get(node, {})
            if not neighbors:
                continue
            counts: dict[int, float] = defaultdict(float)
            for nb, weight in neighbors.items():
                counts[labels[nb]] += weight
            if not counts:
                continue
            # Pick the most-weighted label; ties broken by lowest label id.
            best_label = min(
                counts.items(),
                key=lambda kv: (-kv[1], kv[0]),
            )[0]
            if labels[node] != best_label:
                labels[node] = best_label
                changes += 1
        if changes == 0:
            break

    return labels


def _compute_modularity(
    adj: dict[str, dict[str, float]],
    labels: dict[str, int],
) -> float:
    """Newman's modularity Q on a weighted graph.

    Q = (1 / 2m) * Σ_ij [ A_ij - (k_i × k_j / 2m) ] × δ(c_i, c_j)
    """
    if not adj or not labels:
        return 0.0
    # m = total edge weight (undirected, so we count each edge once)
    seen: set[tuple[str, str]] = set()
    m2 = 0.0  # 2m
    k: dict[str, float] = defaultdict(float)
    for node, neighbors in adj.items():
        for nb, w in neighbors.items():
            m2 += w
            k[node] += w
            seen.add(tuple(sorted([node, nb])))
    if m2 == 0:
        return 0.0
    m = m2 / 2.0

    Q = 0.0
    for node, neighbors in adj.items():
        c_i = labels.get(node)
        for nb, w in neighbors.items():
            c_j = labels.get(nb)
            if c_i is None or c_j is None or c_i != c_j:
                continue
            expected = (k[node] * k[nb]) / m2
            Q += (w - expected)
    Q /= m2
    return round(Q, 4)


# ── Public ───────────────────────────────────────────────────────────────


def compute_communities() -> dict[str, Any]:
    """Run label propagation + compute modularity. Persist.
    Filters dissolved clusters from output."""
    if not _enabled():
        return {"clusters": [], "modularity": 0.0, "enabled": False}

    try:
        from app.companion.social_graph import adjacency
        adj = adjacency()
    except Exception:
        return {"clusters": [], "modularity": 0.0, "enabled": True, "error": "graph unavailable"}

    if not adj:
        return {"clusters": [], "modularity": 0.0, "enabled": True}

    labels = _label_propagation(adj)
    if not labels:
        return {"clusters": [], "modularity": 0.0, "enabled": True}

    # Group by label.
    groups: dict[int, list[str]] = defaultdict(list)
    for node, lbl in labels.items():
        groups[lbl].append(node)

    dissolved = _load_dissolved()
    clusters_out: list[dict[str, Any]] = []
    # Stable cluster letter IDs.
    sorted_groups = sorted(
        groups.values(), key=lambda g: (-len(g), g[0] if g else ""),
    )
    for i, members in enumerate(sorted_groups):
        if len(members) < 2:
            continue  # singletons aren't a cluster
        fs = frozenset(members)
        if fs in dissolved:
            continue
        cluster_id = chr(ord("A") + i) if i < 26 else f"C{i}"
        # Density: edges-within / max-possible-edges.
        n = len(members)
        max_edges = n * (n - 1) / 2
        actual_edges = 0
        member_set = set(members)
        for m1 in members:
            for nb in adj.get(m1, {}):
                if nb in member_set and nb > m1:
                    actual_edges += 1
        density = round(actual_edges / max_edges, 3) if max_edges > 0 else 0.0
        clusters_out.append({
            "id": cluster_id,
            "members": sorted(members),
            "size": n,
            "density": density,
        })

    modularity = _compute_modularity(adj, labels)

    # Persist (also covered by social_graph DR denylist fragment).
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "modularity": modularity,
        "clusters": clusters_out,
        "caveat": (
            "Clusters are computed via label propagation from co-appearance "
            "patterns. Different algorithms produce different clusterings. "
            "Modularity > 0.3 = mathematically strong clustering. "
            "The system never names or describes what a cluster is 'about'."
        ),
    }
    try:
        p = _default_communities_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(p)
    except OSError:
        logger.debug("communities: persist failed", exc_info=True)

    return payload
