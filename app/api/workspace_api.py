"""
workspace_api.py — REST API for hierarchical consciousness workspaces.

Endpoints for React dashboard consumption:
  GET  /api/workspaces          — list all workspaces with snapshots
  POST /api/workspaces          — create new workspace
  GET  /api/workspaces/{id}/items — active + peripheral items
  GET  /api/workspaces/meta     — global meta-workspace snapshot
  GET  /api/workspaces/meta/cross-project — cross-project insight view

All endpoints return JSON. No authentication (internal Docker network only).
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["workspaces"])


class CreateWorkspaceRequest(BaseModel):
    project_id: str
    capacity: int = 3
    display_name: str = ""


# ── List all workspaces ──────────────────────────────────────────────────────

@router.get("/workspaces")
def list_workspaces():
    """List all workspace gates with their current snapshots."""
    try:
        from app.consciousness.workspace_buffer import list_workspaces as _list_ws
        workspaces = _list_ws()
        return {
            "workspaces": [
                {"project_id": pid, **snapshot}
                for pid, snapshot in workspaces.items()
            ],
            "count": len(workspaces),
        }
    except Exception as e:
        logger.warning(f"workspace_api: list failed: {e}")
        return {"workspaces": [], "count": 0}


# ── Create new workspace ─────────────────────────────────────────────────────

@router.post("/workspaces")
def create_workspace(req: CreateWorkspaceRequest):
    """Create a new project workspace (from dashboard)."""
    try:
        from app.consciousness.workspace_buffer import create_workspace as _create_ws
        gate = _create_ws(req.project_id, capacity=req.capacity)
        return {
            "project_id": req.project_id,
            "capacity": gate.capacity,
            "created": True,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Get workspace items ──────────────────────────────────────────────────────

@router.get("/workspaces/{project_id}/items")
def get_workspace_items(project_id: str):
    """Get active and peripheral items for a workspace."""
    try:
        from app.consciousness.workspace_buffer import get_workspace_gate
        gate = get_workspace_gate(project_id)
        return {
            "project_id": project_id,
            "active": [
                {
                    "item_id": item.item_id[:12],
                    "content": item.content[:200],
                    "salience": round(item.salience_score, 3),
                    "source_agent": item.source_agent,
                    "source_channel": item.source_channel,
                    "cycles": item.cycles_in_workspace,
                    "consumed": item.consumed,
                }
                for item in gate.active_items
            ],
            "peripheral": [
                {
                    "item_id": item.item_id[:12],
                    "content": item.content[:100],
                    "salience": round(item.salience_score, 3),
                }
                for item in list(gate.peripheral_items)[:20]
            ],
            "capacity": gate.capacity,
            "cycle": gate._cycle,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Meta-workspace snapshot ──────────────────────────────────────────────────

@router.get("/workspaces/meta")
def get_meta_workspace():
    """Get global meta-workspace snapshot (cross-project view)."""
    try:
        from app.consciousness.meta_workspace import get_meta_workspace as _get_meta
        return _get_meta().get_cross_project_snapshot()
    except Exception as e:
        logger.warning(f"workspace_api: meta snapshot failed: {e}")
        return {"meta_workspace": {}, "by_project": {}, "project_count": 0}
