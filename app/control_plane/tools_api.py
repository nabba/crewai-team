"""Control Plane API — read-only tool registry browser.

Exposes ``GET /api/cp/tools`` for the React control plane to surface
the catalog. Phase 1a: read-only inspection. Phase 2 will add
``POST /api/cp/tools/load`` for explicit pre-loading from the UI.

Source of truth precedence:
  1. In-process ToolRegistry (fast, current state).
  2. Postgres snapshot fallback (cross-pod visibility — tickets / dashboard
     can run in a different process from the gateway).

Routes
------
GET /api/cp/tools           List all registered tools.
GET /api/cp/tools/{name}    Detail for one tool.
GET /api/cp/tools/stats     Counts by tier / lifecycle / capability coverage.
GET /api/cp/tools/capabilities
                            The bounded capability vocabulary.
GET /api/cp/tools/drift     Description-hash / tier drift vs last snapshot.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from app.control_plane.auth_dep import require_gateway_auth
from app.tool_registry import ToolRegistry, Tier
from app.tool_registry.capabilities import CAPABILITIES, all_capability_tags

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/cp/tools",
    tags=["control-plane", "tools"],
    dependencies=[Depends(require_gateway_auth)],
)


def _registry_snapshot() -> list[dict[str, Any]]:
    """Prefer in-process registry; fall back to Postgres snapshot."""
    reg = ToolRegistry.instance()
    specs = reg.all()
    if specs:
        return [s.to_dict() for s in specs]

    # Fallback path: gateway hasn't booted the registry, but Postgres
    # has the snapshot. Used by sidecar processes / multi-pod setups.
    try:
        from app.tool_registry.persistence import load_snapshot
        snap = load_snapshot()
        return snap or []
    except Exception as exc:  # noqa: BLE001
        logger.debug("tools_api: postgres fallback failed: %s", exc)
        return []


# ── Routes ──────────────────────────────────────────────────────────


@router.get("")
def list_tools(
    capability: str | None = Query(None, description="Filter to tools declaring this capability tag"),
    tier: str | None = Query(None, description="Filter by tier (shadow|canary|production|immutable)"),
    loadable_only: bool = Query(False, description="Drop tools whose runtime guard is failing"),
    workspace: str | None = Query(None, description="Filter to tools allowed in this workspace ID"),
):
    """List all registered tools, optionally filtered."""
    rows = _registry_snapshot()

    if capability:
        rows = [r for r in rows if capability in r.get("capabilities", [])]
    if tier:
        rows = [r for r in rows if r.get("tier") == tier]
    if loadable_only:
        rows = [r for r in rows if r.get("is_loadable")]
    if workspace:
        rows = [
            r for r in rows
            if "*" in r.get("workspace_scope", [])
            or workspace in r.get("workspace_scope", [])
        ]

    return {"count": len(rows), "tools": rows}


@router.get("/stats")
def tool_stats():
    """Counts by tier, lifecycle, and capability-tag coverage."""
    reg = ToolRegistry.instance()
    if reg.all():
        return reg.stats()
    # Fallback to snapshot-derived stats.
    rows = _registry_snapshot()
    out: dict[str, int] = {"total": len(rows)}
    for r in rows:
        out[f"tier:{r['tier']}"] = out.get(f"tier:{r['tier']}", 0) + 1
        out[f"lifecycle:{r['lifecycle']}"] = (
            out.get(f"lifecycle:{r['lifecycle']}", 0) + 1
        )
    return out


@router.get("/capabilities")
def list_capabilities():
    """Return the bounded capability vocabulary, grouped by category."""
    return {
        "categories": CAPABILITIES,
        "total_tags": len(all_capability_tags()),
    }


@router.get("/drift")
def tool_drift():
    """Compare current in-memory registry against last Postgres snapshot."""
    from app.tool_registry.drift import detect_drift

    reg = ToolRegistry.instance()
    drift = detect_drift(reg.all())
    return {
        "count": len(drift),
        "entries": [d.to_dict() for d in drift],
    }


@router.get("/{name}")
def get_tool(name: str):
    """Return full detail for a single tool."""
    reg = ToolRegistry.instance()
    spec = reg.get(name)
    if spec is not None:
        return spec.to_dict()
    # Fallback to snapshot.
    rows = _registry_snapshot()
    for r in rows:
        if r.get("name") == name:
            return r
    raise HTTPException(status_code=404, detail=f"tool {name!r} not in registry")
