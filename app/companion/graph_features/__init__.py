"""Social-graph analysis features — Q4.2 Level 4.x.

Four operator-opt-in sub-features layered on app.companion.social_graph:

  * shortest_path    — L4.1, operator-initiated BFS
  * communities      — L4.2, label-propagation clustering
  * bridges          — L4.3, Tarjan's bridges + cut-vertices
  * graph_suggestions — L4.4, prescriptive nudges based on topology
                        (requires SECOND typed-phrase gate)

Each sub-feature gates on its own runtime_settings flag and ALSO
requires the L4 master switch (``person_correlation_social_graph_enabled``).
"""
from __future__ import annotations

__all__ = ["shortest_path", "communities", "bridges", "graph_suggestions"]
