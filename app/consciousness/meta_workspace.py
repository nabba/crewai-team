"""
meta_workspace.py — Global meta-workspace for cross-project consciousness.

Hierarchical workspace architecture:
  - Each project has a local workspace (capacity=3)
  - "Generic" workspace for non-project tasks (capacity=5)
  - Meta-workspace aggregates top items from all projects (capacity=7)

Promotion rule: after each cycle, the highest-salience active item
from each project workspace gets promoted to the meta-workspace.
This enables cross-project insights without cross-contamination.

DGM Safety: promotion is infrastructure-level, agents cannot control it.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class MetaWorkspace:
    """Manages promotion from project workspaces to global meta-workspace."""

    def promote_from_project(self, project_id: str) -> bool:
        """Promote top item from a project workspace to the meta-workspace.

        Returns True if an item was promoted.
        """
        try:
            from app.consciousness.workspace_buffer import (
                get_workspace_gate, META_WORKSPACE,
            )
            project_gate = get_workspace_gate(project_id)
            meta_gate = get_workspace_gate(META_WORKSPACE)

            items = project_gate.active_items
            if not items:
                return False

            top_item = max(items, key=lambda x: x.salience_score)
            # Create a copy for meta-workspace (don't remove from project)
            from app.consciousness.workspace_buffer import WorkspaceItem
            meta_item = WorkspaceItem(
                content=f"[{project_id}] {top_item.content}",
                content_embedding=top_item.content_embedding,
                source_agent=top_item.source_agent,
                source_channel=f"project:{project_id}",
                goal_relevance=top_item.goal_relevance,
                novelty_score=top_item.novelty_score,
                agent_urgency=top_item.agent_urgency,
                surprise_signal=top_item.surprise_signal,
                salience_score=top_item.salience_score,
                metadata={**top_item.metadata, "promoted_from": project_id},
            )

            result = meta_gate.evaluate(meta_item)
            if result.transition_type != "rejected":
                logger.debug(
                    f"meta_workspace: promoted from {project_id} "
                    f"sal={top_item.salience_score:.2f} → meta"
                )
                return True
            return False
        except Exception:
            logger.debug("meta_workspace: promotion failed", exc_info=True)
            return False

    def promote_all(self) -> dict:
        """Run promotion for all known project workspaces.

        Returns dict of {project_id: promoted (bool)}.
        """
        from app.consciousness.workspace_buffer import (
            list_workspaces, GENERIC_WORKSPACE, META_WORKSPACE,
        )
        results = {}
        for project_id in list_workspaces():
            if project_id == META_WORKSPACE:
                continue  # Don't promote meta into itself
            promoted = self.promote_from_project(project_id)
            results[project_id] = promoted
        return results

    def get_cross_project_snapshot(self) -> dict:
        """Dashboard: what's in the meta-workspace from which projects."""
        from app.consciousness.workspace_buffer import (
            get_workspace_gate, list_workspaces, META_WORKSPACE,
        )
        meta_gate = get_workspace_gate(META_WORKSPACE)
        snapshot = meta_gate.get_snapshot()

        # Group by source project
        by_project: dict[str, list] = {}
        for item in meta_gate.active_items:
            project = item.metadata.get("promoted_from", item.source_channel)
            by_project.setdefault(project, []).append({
                "content": item.content[:100],
                "salience": round(item.salience_score, 3),
            })

        return {
            "meta_workspace": snapshot,
            "by_project": by_project,
            "project_count": len(list_workspaces()) - 1,  # Exclude meta
        }


# ── Module-level singleton ──────────────────────────────────────────────────

_meta: MetaWorkspace | None = None


def get_meta_workspace() -> MetaWorkspace:
    global _meta
    if _meta is None:
        _meta = MetaWorkspace()
    return _meta
