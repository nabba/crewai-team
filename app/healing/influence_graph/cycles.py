"""Tarjan's SCC algorithm over the curated influence graph.

PROGRAM §49 — Q14.2 (year-2+ resilience §10.2). Given the
:data:`EDGES` list from :mod:`app.healing.influence_graph.edges`,
return every strongly-connected component with > 1 node. Each
such SCC IS a feedback loop in the subsystem topology.

Pure-Python, no third-party deps. Stable output ordering so test
fixtures don't churn on unrelated graph changes.

Why Tarjan's specifically: linear-time O(V+E), iterative
implementation possible, returns SCCs in reverse topological
order (deterministic enough to test against). The graph is small
(~50 nodes, ~50 edges) so any algorithm would do; Tarjan's is
the textbook choice and easiest to maintain.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Cycle:
    """One feedback loop in the topology."""

    nodes: tuple[str, ...]   # sorted alphabetically for stable output
    n_edges_internal: int    # edges that stay inside this SCC


def find_cycles(edges: Iterable | None = None) -> list[Cycle]:
    """Return every SCC with > 1 node. Uses Tarjan's algorithm.

    ``edges`` defaults to the curated :data:`EDGES` list. Tests pass
    a synthetic edge list to exercise specific topologies.

    Returns a list of :class:`Cycle` objects, sorted by node-set so
    output is stable across runs."""
    if edges is None:
        from app.healing.influence_graph.edges import EDGES
        edges = EDGES

    # Build adjacency.
    adj: dict[str, set[str]] = {}
    edge_pairs: set[tuple[str, str]] = set()
    nodes: set[str] = set()
    for e in edges:
        adj.setdefault(e.producer, set()).add(e.consumer)
        edge_pairs.add((e.producer, e.consumer))
        nodes.add(e.producer)
        nodes.add(e.consumer)

    # Tarjan's iterative SCC.
    index_of: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    counter = [0]
    sccs: list[list[str]] = []

    def strongconnect(start: str) -> None:
        # Iterative implementation with explicit work stack.
        work: list[tuple[str, iter]] = [(start, iter(sorted(adj.get(start, ()))))]
        index_of[start] = counter[0]
        lowlink[start] = counter[0]
        counter[0] += 1
        stack.append(start)
        on_stack.add(start)
        while work:
            v, it = work[-1]
            try:
                w = next(it)
            except StopIteration:
                work.pop()
                if work:
                    parent = work[-1][0]
                    lowlink[parent] = min(lowlink[parent], lowlink[v])
                if lowlink[v] == index_of[v]:
                    component: list[str] = []
                    while True:
                        x = stack.pop()
                        on_stack.discard(x)
                        component.append(x)
                        if x == v:
                            break
                    if len(component) > 1:
                        sccs.append(sorted(component))
                continue
            if w not in index_of:
                index_of[w] = counter[0]
                lowlink[w] = counter[0]
                counter[0] += 1
                stack.append(w)
                on_stack.add(w)
                work.append((w, iter(sorted(adj.get(w, ())))))
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], index_of[w])

    for node in sorted(nodes):
        if node not in index_of:
            strongconnect(node)

    # Build Cycle objects.
    result: list[Cycle] = []
    for component in sccs:
        cs = set(component)
        n_internal = sum(
            1 for (a, b) in edge_pairs
            if a in cs and b in cs
        )
        result.append(Cycle(
            nodes=tuple(component),  # already sorted
            n_edges_internal=n_internal,
        ))
    result.sort(key=lambda c: c.nodes)
    return result


def summary(cycles: list[Cycle]) -> dict:
    """One-shot summary dict for the REST endpoint."""
    return {
        "n_cycles": len(cycles),
        "cycles": [
            {
                "nodes": list(c.nodes),
                "n_edges_internal": c.n_edges_internal,
            }
            for c in cycles
        ],
    }
