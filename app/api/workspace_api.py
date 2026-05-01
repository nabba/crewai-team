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

def _project_display_name(pid: str, name_cache: dict[str, str]) -> str:
    """Resolve a project_id to a human-friendly label.

    UUIDs are looked up in the Control Plane projects table; legacy
    name-keyed gates (e.g. "__meta__", "generic", "eesti mets") pass
    through unchanged. Cached per call to avoid N+1 PG hits.
    """
    if pid in name_cache:
        return name_cache[pid]
    label = pid
    try:
        # Only look up values that look like UUIDs.
        if len(pid) == 36 and pid.count("-") == 4:
            from app.control_plane.projects import get_projects
            row = get_projects().get_by_id(pid)
            if row and row.get("name"):
                label = row["name"]
    except Exception:
        pass
    name_cache[pid] = label
    return label


@router.get("/workspaces")
def list_workspaces():
    """List all workspace gates with their current snapshots.

    Reconciles against ``control_plane.projects``: every persistent CP
    project is guaranteed to have a gate in the response, even if nothing
    has touched it since the gateway restarted (gates are in-memory and
    wipe on restart, whereas CP projects live in PostgreSQL).
    """
    try:
        from app.subia.scene.buffer import (
            list_workspaces as _list_ws,
            get_workspace_gate,
        )
    except Exception as e:
        logger.warning(f"workspace_api: buffer import failed: {e}")
        return {"workspaces": [], "count": 0}

    # 1. Ensure a gate exists for every CP project (lazy create on first call).
    try:
        from app.control_plane.projects import get_projects
        for proj in get_projects().list_all() or []:
            pid = proj.get("id")
            if pid:
                # get_workspace_gate is the public constructor — it creates
                # a default-capacity gate if one doesn't exist yet.
                get_workspace_gate(pid)
    except Exception as e:
        logger.debug("workspace_api: CP reconcile skipped: %s", e)

    # 2. Snapshot the registry and enrich with display names.
    try:
        workspaces = _list_ws()
        name_cache: dict[str, str] = {}
        return {
            "workspaces": [
                {
                    "project_id": pid,
                    "display_name": _project_display_name(pid, name_cache),
                    **snapshot,
                }
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
    """Create a new project workspace (from dashboard).

    Creates both halves of the project at once so the dashboard doesn't
    end up with split-brain state:

      1. The Control Plane project row (PostgreSQL — drives the project
         switcher, tickets, budgets, audit).
      2. The Consciousness gate (in-memory — drives the attention / salience
         competition for cognitive workspaces).

    If a Control Plane project with the same name already exists the
    creation is a no-op and the existing project is reused (idempotent).
    """
    raw_name = (req.display_name or req.project_id or "").strip()
    if not raw_name:
        raise HTTPException(status_code=400, detail="project_id / display_name required")

    # Normalise the id used everywhere downstream (telemetry, gates, etc.)
    # Prefer the UUID from the PG row once it exists — aligns with
    # control_plane.projects.get_active_project_id(), which the existing
    # /api/cp/projects list uses.
    cp_project: dict | None = None
    cp_error: str | None = None
    try:
        from app.control_plane.projects import get_projects
        projects = get_projects()
        # Idempotent lookup by name first so a second submit doesn't bomb.
        cp_project = projects.get_by_name(raw_name)
        if not cp_project:
            cp_project = projects.create(raw_name, mission="", description="")
    except Exception as exc:
        cp_error = str(exc)
        logger.warning("workspace_api: control-plane project create failed: %s", exc)

    project_key = (cp_project or {}).get("id") or req.project_id or raw_name.lower()

    # 2. Consciousness gate — use the CP id when available so both halves
    #    attribute telemetry to the same project_id. Preserve the legacy
    #    lowercase-name key as a fallback so pre-existing gates still work.
    try:
        from app.subia.scene.buffer import create_workspace as _create_ws
        gate = _create_ws(project_key, capacity=req.capacity)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"consciousness gate create failed: {exc}")

    return {
        "project_id": project_key,
        "display_name": raw_name,
        "capacity": gate.capacity,
        "control_plane_project": cp_project,
        "control_plane_error": cp_error,
        "created": True,
    }


# ── Get workspace items ──────────────────────────────────────────────────────

@router.get("/workspaces/{project_id}/items")
def get_workspace_items(project_id: str):
    """Get active and peripheral items for a workspace."""
    try:
        from app.subia.scene.buffer import get_workspace_gate
        gate = get_workspace_gate(project_id)
        return {
            "project_id": project_id,
            "display_name": _project_display_name(project_id, {}),
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
        from app.subia.scene.meta_workspace import get_meta_workspace as _get_meta
        return _get_meta().get_cross_project_snapshot()
    except Exception as e:
        logger.warning(f"workspace_api: meta snapshot failed: {e}")
        return {"meta_workspace": {}, "by_project": {}, "project_count": 0}
