"""Control plane — architecture-request endpoints at /api/cp/architecture-requests.

Operators (via React or curl) can:

  GET  /api/cp/architecture-requests                          list (filtered by status)
  GET  /api/cp/architecture-requests/{id}                     detail (incl. is_terminal)
  GET  /api/cp/architecture-requests/{id}/audit               audit-log entries for one request
  POST /api/cp/architecture-requests/{id}/approve             approve (PROPOSED → APPROVED)
  POST /api/cp/architecture-requests/{id}/reject              reject  (PROPOSED → REJECTED)
  POST /api/cp/architecture-requests/{id}/scaffold            scaffold + transition (APPROVED → SCAFFOLDED)
  POST /api/cp/architecture-requests/{id}/abandon             abandon (SCAFFOLDED|IMPLEMENTING → ABANDONED)
  POST /api/cp/architecture-requests/{id}/record-child-cr     record a per-file child CR id
  POST /api/cp/architecture-requests/{id}/mark-complete       all children APPLIED (IMPLEMENTING → COMPLETED)
  GET  /api/cp/architecture-requests/{id}/scaffold/manifest   read the staged MANIFEST.md

Auth: same ``require_gateway_auth`` dependency as the rest of /cp/.

These endpoints are the React surface. The Signal surface
(👍/👎 reactions) is wired in main.py and dispatches through
the same lifecycle module — the audit log records which source
made each decision.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.architecture_requests import (
    ArchStatus,
    DecisionSource,
    InvalidTransition,
    abandon,
    approve,
    get,
    is_protected_path,
    list_all,
    mark_complete,
    record_child_change_request,
    reject,
    scaffold as scaffold_lifecycle,
)
from app.architecture_requests.scaffolder import scaffold as scaffold_files
from app.control_plane.auth_dep import require_gateway_auth

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/cp/architecture-requests",
    tags=["control-plane", "architecture-requests"],
    dependencies=[Depends(require_gateway_auth)],
)


# ── Request bodies ──────────────────────────────────────────────────


class _ApproveBody(BaseModel):
    operator: str = Field(default="react-operator")
    reason: str | None = Field(default=None)


class _RejectBody(BaseModel):
    operator: str = Field(default="react-operator")
    reason: str | None = Field(default=None)


class _AbandonBody(BaseModel):
    operator: str = Field(default="react-operator")
    reason: str = Field(..., min_length=1)


class _RecordChildBody(BaseModel):
    child_change_request_id: str = Field(..., min_length=1)


# ── Helpers ─────────────────────────────────────────────────────────


def _serialize(req) -> dict[str, Any]:
    if req is None:
        return {}
    d = req.to_dict()
    d["is_terminal"] = req.is_terminal
    d["is_decided"] = req.is_decided
    d["package_is_protected"] = is_protected_path(req.package_path)
    return d


def _require(request_id: str):
    req = get(request_id)
    if req is None:
        raise HTTPException(status_code=404, detail=f"architecture-request {request_id!r} not found")
    return req


# ── Routes ──────────────────────────────────────────────────────────


@router.get("")
def list_architecture_requests(
    status: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
):
    status_enum: ArchStatus | None = None
    if status:
        try:
            status_enum = ArchStatus(status)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"invalid status {status!r}. Valid: "
                    f"{[s.value for s in ArchStatus]}"
                ),
            )
    items = list_all(status=status_enum, limit=limit)
    return {
        "count": len(items),
        "architecture_requests": [_serialize(r) for r in items],
    }


@router.get("/{request_id}")
def get_architecture_request(request_id: str):
    return _serialize(_require(request_id))


@router.get("/{request_id}/audit")
def get_audit_entries(request_id: str, limit: int = Query(default=200, ge=1, le=2000)):
    """Return audit-log entries scoped to one request, oldest first."""
    from app.architecture_requests import store
    _require(request_id)  # 404 if unknown
    out = [
        p for p in store.iter_audit_entries()
        if p.get("request_id") == request_id
    ]
    return {"count": len(out), "entries": out[-limit:]}


@router.post("/{request_id}/approve")
def approve_architecture_request(request_id: str, body: _ApproveBody):
    req = _require(request_id)
    if req.status is ArchStatus.TIER_IMMUTABLE_REFUSED:
        raise HTTPException(
            status_code=403,
            detail=(
                "TIER_IMMUTABLE / consciousness-layer architecture requests "
                "cannot be approved through this path. Operator must use "
                "Tier-3 amendment."
            ),
        )
    if req.status is ArchStatus.APPROVED:
        return {"ok": True, "architecture_request": _serialize(req), "note": "already approved"}
    if req.status is not ArchStatus.PROPOSED:
        raise HTTPException(
            status_code=409,
            detail=f"cannot approve in status {req.status.value}",
        )
    updated = approve(
        request_id,
        source=DecisionSource.REACT_APPROVE,
        decision_reason=body.reason,
    )
    return {"ok": True, "architecture_request": _serialize(updated)}


@router.post("/{request_id}/reject")
def reject_architecture_request(request_id: str, body: _RejectBody):
    req = _require(request_id)
    if req.status is not ArchStatus.PROPOSED:
        raise HTTPException(
            status_code=409,
            detail=f"cannot reject in status {req.status.value}",
        )
    updated = reject(
        request_id,
        source=DecisionSource.REACT_REJECT,
        decision_reason=body.reason,
    )
    return {"ok": True, "architecture_request": _serialize(updated)}


@router.post("/{request_id}/scaffold")
def scaffold_architecture_request(request_id: str):
    """Run the scaffolder + record SCAFFOLDED transition.

    Writes stubs to ``workspace/architecture_requests/<id>/scaffold/``
    plus a MANIFEST.md. Operator reviews staged files, then files
    per-file change-requests through the standard /cp/changes gate.
    """
    req = _require(request_id)
    if req.status is ArchStatus.SCAFFOLDED:
        return {
            "ok": True,
            "architecture_request": _serialize(req),
            "scaffold_dir": req.scaffold_dir,
            "note": "already scaffolded",
        }
    if req.status is not ArchStatus.APPROVED:
        raise HTTPException(
            status_code=409,
            detail=f"cannot scaffold in status {req.status.value}; expected APPROVED",
        )
    try:
        out_dir = scaffold_files(req)
        updated = scaffold_lifecycle(request_id, scaffold_dir=str(out_dir))
    except InvalidTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.warning("scaffold failed for %s: %s", request_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"scaffold failed: {exc}")
    return {
        "ok": True,
        "architecture_request": _serialize(updated),
        "scaffold_dir": str(out_dir),
    }


@router.get("/{request_id}/scaffold/manifest")
def get_scaffold_manifest(request_id: str):
    req = _require(request_id)
    if not req.scaffold_dir:
        raise HTTPException(
            status_code=409,
            detail="not yet scaffolded; call /scaffold first",
        )
    from pathlib import Path
    manifest = Path(req.scaffold_dir) / "MANIFEST.md"
    if not manifest.exists():
        raise HTTPException(status_code=404, detail="manifest file missing on disk")
    return {
        "request_id": request_id,
        "manifest_path": str(manifest),
        "manifest_text": manifest.read_text(encoding="utf-8"),
    }


@router.post("/{request_id}/abandon")
def abandon_architecture_request(request_id: str, body: _AbandonBody):
    _require(request_id)
    try:
        updated = abandon(request_id, reason=body.reason)
    except InvalidTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"ok": True, "architecture_request": _serialize(updated)}


@router.post("/{request_id}/record-child-cr")
def record_child(request_id: str, body: _RecordChildBody):
    _require(request_id)
    try:
        updated = record_child_change_request(request_id, body.child_change_request_id)
    except InvalidTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"ok": True, "architecture_request": _serialize(updated)}


@router.post("/{request_id}/mark-complete")
def mark_arch_complete(request_id: str):
    _require(request_id)
    try:
        updated = mark_complete(request_id)
    except InvalidTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"ok": True, "architecture_request": _serialize(updated)}
