"""Control plane — workflow_template endpoints at /api/cp/workflows.

PROGRAM §46.3 (Q8.3). Operators (via React or curl) can:

  GET  /api/cp/workflows                       list templates
  POST /api/cp/workflows                       save a template
  GET  /api/cp/workflows/{id}                  get one template
  DELETE /api/cp/workflows/{id}                remove a template
  POST /api/cp/workflows/{id}/run              enqueue a run; returns run_id
  GET  /api/cp/workflows/runs                  list runs (optional template_id)
  GET  /api/cp/workflows/runs/{run_id}         get one run's status

Auth: ``require_gateway_auth`` dependency.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.control_plane.auth_dep import require_gateway_auth
from app.workflows import (
    InvalidWorkflow,
    WorkflowNode,
    WorkflowTemplate,
    enqueue,
    get,
    get_run,
    list_all,
    list_runs,
    save,
    validate_template,
)
from app.workflows.queue import cancel_run

logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/api/cp/workflows",
    tags=["control-plane", "workflows"],
    dependencies=[Depends(require_gateway_auth)],
)


class _NodeBody(BaseModel):
    id: str = Field(..., min_length=1)
    tool_name: str = Field(..., min_length=1)
    args: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    description: str = Field(default="")


class _CreateBody(BaseModel):
    id: str | None = Field(default=None)  # server-fills uuid when None
    name: str = Field(..., min_length=1)
    description: str = Field(default="")
    nodes: list[_NodeBody] = Field(default_factory=list)
    inputs: list[str] = Field(default_factory=list)


class _RunBody(BaseModel):
    inputs: dict[str, Any] = Field(default_factory=dict)


def _serialize_template(t: WorkflowTemplate) -> dict[str, Any]:
    return t.to_dict()


def _serialize_run(r) -> dict[str, Any]:
    if r is None:
        return {}
    d = r.to_dict()
    d["is_terminal"] = r.is_terminal
    return d


# ── Templates ───────────────────────────────────────────────────────


@router.get("")
def list_templates(limit: int = Query(default=200, ge=1, le=500)):
    items = list_all(limit=limit)
    return {
        "count": len(items),
        "templates": [_serialize_template(t) for t in items],
    }


@router.get("/{template_id}")
def get_template(template_id: str):
    t = get(template_id)
    if t is None:
        raise HTTPException(
            status_code=404,
            detail=f"workflow template {template_id!r} not found",
        )
    return _serialize_template(t)


@router.post("")
def create_template(body: _CreateBody):
    tid = (body.id or "").strip() or str(uuid.uuid4())
    template = WorkflowTemplate(
        id=tid,
        name=body.name.strip(),
        description=body.description.strip(),
        nodes=[
            WorkflowNode(
                id=n.id, tool_name=n.tool_name, args=n.args,
                depends_on=list(n.depends_on), description=n.description,
            ) for n in body.nodes
        ],
        inputs=list(body.inputs),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    try:
        validate_template(template)
    except InvalidWorkflow as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    save(template)
    return {"ok": True, "template": _serialize_template(template)}


@router.delete("/{template_id}")
def delete_template(template_id: str):
    from app.workflows.store import delete
    if not delete(template_id):
        raise HTTPException(
            status_code=404,
            detail=f"workflow template {template_id!r} not found",
        )
    return {"ok": True, "deleted": template_id}


# ── Runs ────────────────────────────────────────────────────────────


@router.post("/{template_id}/run")
def start_run(template_id: str, body: _RunBody):
    t = get(template_id)
    if t is None:
        raise HTTPException(
            status_code=404,
            detail=f"workflow template {template_id!r} not found",
        )
    # Validate required inputs are supplied
    missing = [name for name in t.inputs if name not in body.inputs]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"missing required inputs: {missing}",
        )
    run = enqueue(t, inputs=body.inputs)
    return {"ok": True, "run": _serialize_run(run)}


@router.get("/runs")
def list_all_runs(
    template_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
):
    runs = list_runs(template_id=template_id, limit=limit)
    return {
        "count": len(runs),
        "runs": [_serialize_run(r) for r in runs],
    }


@router.get("/runs/{run_id}")
def get_run_status(run_id: str):
    r = get_run(run_id)
    if r is None:
        raise HTTPException(
            status_code=404,
            detail=f"workflow run {run_id!r} not found",
        )
    return _serialize_run(r)


@router.post("/runs/{run_id}/cancel")
def cancel_run_route(run_id: str):
    if not cancel_run(run_id):
        raise HTTPException(
            status_code=404,
            detail=f"workflow run {run_id!r} not found or already terminal",
        )
    return {"ok": True, "cancelled": run_id}
