"""influence_graph — subsystem-to-subsystem topology + cycle detection.

PROGRAM §49 — Q14.2 (year-2+ resilience §10.2). The system has 30+
idle jobs, multiple knowledge bases, multiple decision surfaces. The
operator's stated concern:

  > meta-agent recipes affect agent selection which affects what
  > enters the lessons KB which affects what the meta-agent learns
  > from. With 30+ idle jobs, an emergent loop could optimize for
  > the wrong thing without any single component being broken.

The instrument here is **observability**, not control:

  * **Curated edge list** (:mod:`edges`) — one place the operator
    can see and revise the system's actual signal topology. Each
    edge is ``(producer_subsystem, consumer_subsystem, signal_name,
    edge_kind)``. Hand-curated; small enough to maintain by reading
    diffs.
  * **Cycle detection** (:mod:`cycles`) — Tarjan's SCC algorithm on
    the curated graph. Reports every closed loop with > 1 node.
    Operator can then ask "is this loop intended? Does it have a
    pressure-release valve?"
  * **Drift probe** (:mod:`drift_probe`) — for the concrete named
    loop (meta-agent → recipe → outcome → lessons KB → meta-agent
    selection), watch the Gini coefficient of meta-agent recipe
    selection over a 30-day rolling window. If Gini trends
    monotonically up (concentration increasing → loop converging
    on a fixed point), file a Signal alert.

Compose with what's already there:

  * The drift probe runs as a healing monitor (weekly cadence).
  * Alerts route through canonical ``notify`` with topic
    ``feedback_loop_drift`` for arbiter dedup.
  * Landmark transitions emit ``feedback_loop_drift`` continuity-
    ledger events so annual reflection picks them up.
  * The cycle report is exposed via a REST endpoint
    ``GET /api/cp/health/influence-graph`` for the React dashboard.

What this module deliberately does NOT do:

  * No automatic graph extraction (would require code analysis +
    runtime tracing infrastructure that doesn't exist; the curated
    edge list is cheaper and more honest about what we know).
  * No automatic loop intervention (that's the operator's job).
  * No SubIA-internal-graph edges. SubIA is observational; its
    internal modules are out of scope here. The graph covers the
    OUTER substrate (idle jobs, KBs, REST endpoints, governance
    surfaces).

Master switch: ``influence_graph_monitor_enabled`` (default ON).
"""
from app.healing.influence_graph.edges import (
    EDGES,
    EdgeKind,
    InfluenceEdge,
    nodes,
)
from app.healing.influence_graph.cycles import (
    Cycle,
    find_cycles,
    summary as cycle_summary,
)

__all__ = [
    "EDGES",
    "Cycle",
    "EdgeKind",
    "InfluenceEdge",
    "cycle_summary",
    "find_cycles",
    "nodes",
]
