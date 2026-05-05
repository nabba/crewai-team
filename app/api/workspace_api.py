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
    # Companion Layer (Phase 0+): optional seed for the workspace's idle
    # contemplation loop. None means the Companion still runs in
    # synthesis-mode (grand-task auto-derived from conversation/tasks).
    seed_prompt: str | None = None


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
    #
    # Defensive dedup: in some pre-restart states the in-memory ``_gates``
    # registry ends up with two snapshots whose serialized project_ids
    # collide — typically one freshly-reconciled gate (cycle 0, capacity
    # 3, fallback display) plus one runtime-populated gate (cycle > 0,
    # actual capacity, display from CP). The dashboard would render both
    # as separate tabs, which is confusing.
    #
    # The fix: collapse by ``(project_id, display_name_resolved_to_real_name)``
    # and keep the snapshot with the most signal — higher cycle wins,
    # ties broken by larger active_count, then capacity. The "real name"
    # is what _project_display_name resolves the pid to (e.g. "default"
    # for the UUID 676a8f70-…); two snapshots that resolve to the same
    # real name AND share their project_id are the same logical gate.
    try:
        workspaces = _list_ws()
        name_cache: dict[str, str] = {}

        # Collect (key, display, snapshot) once per dict entry.
        rows: list[tuple[str, str, dict]] = [
            (pid, _project_display_name(pid, name_cache), snap)
            for pid, snap in workspaces.items()
        ]

        # Dedup key: the resolved display name is the canonical
        # identity. Two entries resolving to the same display (e.g.
        # both "default") are the same project regardless of how they
        # got into the registry.
        best: dict[str, tuple[str, str, dict]] = {}
        for pid, display, snap in rows:
            key = display  # canonical identity
            existing = best.get(key)
            if existing is None:
                best[key] = (pid, display, snap)
                continue
            # Pick the snapshot with more signal.
            ex_pid, _ex_display, ex_snap = existing
            new_score = (
                int(snap.get("cycle", 0)),
                int(snap.get("active_count", 0)),
                int(snap.get("capacity", 0)),
            )
            ex_score = (
                int(ex_snap.get("cycle", 0)),
                int(ex_snap.get("active_count", 0)),
                int(ex_snap.get("capacity", 0)),
            )
            if new_score > ex_score:
                best[key] = (pid, display, snap)
            # If the existing pid is a UUID but the new pid is also a
            # UUID with same display, prefer whichever scored higher.
            # We don't try to mutate _gates here — that's a separate
            # cleanup concern; this endpoint is read-only.

        deduped = [
            {
                "project_id": pid,
                "display_name": display,
                **snap,
            }
            for pid, display, snap in best.values()
        ]

        if len(deduped) < len(rows):
            logger.info(
                "workspace_api: deduped %d snapshots → %d workspaces "
                "(stale duplicate gates in memory; restart gateway "
                "or call /api/_workspaces_dedup to GC).",
                len(rows), len(deduped),
            )

        return {
            "workspaces": deduped,
            "count": len(deduped),
        }
    except Exception as e:
        logger.warning(f"workspace_api: list failed: {e}")
        return {"workspaces": [], "count": 0}


@router.post("/workspaces/_dedup")
def dedup_workspace_registry():
    """Prune duplicate-resolving gates from the in-memory ``_gates`` dict.

    Operator-facing GC: when two gates resolve to the same display name
    (e.g. an empty UUID-keyed gate from the API reconcile and a
    populated name-keyed gate from a legacy code path), keep the one
    with more signal and drop the other. Returns a summary of what
    was removed.

    Read-only API endpoints stay defensive (the list endpoint dedupes
    in its serializer) — but this lets the operator actually clean
    up the underlying registry without a gateway restart.
    """
    try:
        from app.subia.scene.buffer import _gates, _gates_lock
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"buffer import failed: {exc}"
        )

    name_cache: dict[str, str] = {}
    removed: list[dict] = []

    with _gates_lock:
        # Build groups keyed on resolved display name.
        groups: dict[str, list[tuple[str, dict]]] = {}
        for pid, gate in _gates.items():
            display = _project_display_name(pid, name_cache)
            snap = gate.get_snapshot()
            groups.setdefault(display, []).append((pid, snap))

        for display, members in groups.items():
            if len(members) <= 1:
                continue
            # Keep the highest-signal entry; drop the rest.
            members.sort(
                key=lambda kv: (
                    int(kv[1].get("cycle", 0)),
                    int(kv[1].get("active_count", 0)),
                    int(kv[1].get("capacity", 0)),
                ),
                reverse=True,
            )
            keeper_pid, keeper_snap = members[0]
            for pid, snap in members[1:]:
                _gates.pop(pid, None)
                removed.append({
                    "removed_pid": pid,
                    "kept_pid": keeper_pid,
                    "display": display,
                    "removed_cycle": snap.get("cycle"),
                    "kept_cycle": keeper_snap.get("cycle"),
                })

    if removed:
        logger.info(
            "workspace_api: deduped %d stale gates: %s",
            len(removed),
            ", ".join(
                f"{r['display']!r} dropped {r['removed_pid']!r}"
                for r in removed
            ),
        )
    return {"removed": removed, "removed_count": len(removed)}


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
    # Build initial config_json so the seed prompt lands atomically with
    # project creation. If the project already exists this is ignored —
    # editing the seed afterwards uses the dedicated /api/cp/reverie endpoint
    # (Phase 4); we don't silently overwrite an existing seed here.
    initial_config: dict = {}
    if req.seed_prompt:
        try:
            from app.companion.config import CompanionConfig
            initial_config["companion"] = CompanionConfig(
                seed_prompt=req.seed_prompt
            ).clamp().to_dict()
        except Exception as exc:
            logger.warning("workspace_api: companion config build failed: %s", exc)
    try:
        from app.control_plane.projects import get_projects
        projects = get_projects()
        # Idempotent lookup by name first so a second submit doesn't bomb.
        cp_project = projects.get_by_name(raw_name)
        if not cp_project:
            cp_project = projects.create(
                raw_name, mission="", description="",
                config=initial_config or None,
            )
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
        "seed_prompt": req.seed_prompt,
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
