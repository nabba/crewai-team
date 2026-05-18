"""REST API for vacation mode.

PROGRAM §51 — Q16 Theme 3 follow-on. Endpoints under
``/api/cp/vacation/*``:

  GET  /state                — current state (engaged + staged_allowlist)
  GET  /allowlist            — current allowlist (frozen if engaged)
  POST /allowlist/stage      — stage a new allowlist (refused if engaged)
  POST /engage               — engage vacation mode
  POST /disengage            — disengage immediately
  GET  /digests              — list end-of-vacation digest filenames
  GET  /digests/{name}       — read one digest's content
  GET  /audit-log?limit=N    — recent rows from auto_apply_log.jsonl

All mutating endpoints require Bearer auth via the existing gateway
auth dependency (the same one ``/api/cp/changes`` uses).
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cp/vacation", tags=["vacation"])


# ── Request bodies ───────────────────────────────────────────────────────


class StageAllowlistRequest(BaseModel):
    requestor_allowlist: list[str] = Field(default_factory=list)
    path_prefix_allowlist: list[str] = Field(default_factory=list)
    max_diff_lines: int = 10


class EngageRequest(BaseModel):
    hours: float = Field(gt=0, le=24 * 30, description="duration in hours, ≤ 30 days")
    reason: str = ""
    engaged_by: str = "operator"
    confirmation_phrase: str = Field(
        default="",
        description=(
            "Must equal 'ENGAGE VACATION MODE' exactly. Defends against "
            "accidental engagement via a bare REST POST; the React UI "
            "carries the same gate purely as UX."
        ),
    )


class DisengageRequest(BaseModel):
    disengaged_by: str = "operator"


# ── Read endpoints ───────────────────────────────────────────────────────


@router.get("/state")
def get_state() -> dict:
    """Full state blob, suitable for direct rendering. Auto-expires a
    stale engagement on read."""
    try:
        from app import vacation_mode as vm
        state = vm.current_state()
        return {
            "engaged": state.engaged,
            "engagement": (
                state.engagement.to_dict() if state.engagement else None
            ),
            "staged_allowlist": state.staged_allowlist.to_dict(),
            "is_active": vm.is_active(),
            "now": time.time(),
        }
    except Exception as exc:
        raise HTTPException(500, f"state read failed: {type(exc).__name__}: {exc}")


@router.get("/allowlist")
def get_allowlist() -> dict:
    """Currently-applicable allowlist (frozen if engaged, staged if not)."""
    try:
        from app import vacation_mode as vm
        al = vm.current_allowlist()
        return {
            "is_frozen": vm.is_active(),
            "allowlist": al.to_dict(),
        }
    except Exception as exc:
        raise HTTPException(500, f"allowlist read failed: {type(exc).__name__}: {exc}")


@router.get("/digests")
def list_digests() -> dict:
    """List all on-disk end-of-vacation digests by filename."""
    try:
        from app import vacation_mode as vm
        paths = vm.list_digests()
        return {
            "digests": [p.name for p in paths],
            "n": len(paths),
        }
    except Exception as exc:
        raise HTTPException(500, f"digests list failed: {type(exc).__name__}: {exc}")


@router.get("/digests/{name}")
def read_digest_endpoint(name: str) -> dict:
    """Read one digest's content. ``name`` is the filename returned by
    ``GET /digests``; we refuse anything containing a path separator
    to prevent traversal."""
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(400, "invalid digest name")
    try:
        from app import vacation_mode as vm
        for p in vm.list_digests():
            if p.name == name:
                body = vm.read_digest(p)
                if not body:
                    raise HTTPException(404, "digest body empty")
                return {"name": p.name, "body": body}
        raise HTTPException(404, "digest not found")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"digest read failed: {type(exc).__name__}: {exc}")


@router.get("/audit-log")
def audit_log(limit: int = 100) -> dict:
    """Recent rows from ``workspace/vacation_mode/auto_apply_log.jsonl``."""
    if not 1 <= limit <= 1000:
        raise HTTPException(400, "limit must be in [1, 1000]")
    try:
        from app.vacation_mode.sweep import _log_path  # internal helper
        p = _log_path()
        if not p.exists():
            return {"rows": [], "n": 0}
        rows: list[dict] = []
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        # Newest last → return newest first.
        rows = rows[-limit:][::-1]
        return {"rows": rows, "n": len(rows)}
    except Exception as exc:
        raise HTTPException(500, f"audit-log read failed: {type(exc).__name__}: {exc}")


# ── Mutating endpoints ───────────────────────────────────────────────────


@router.post("/allowlist/stage")
def stage_allowlist_endpoint(req: StageAllowlistRequest) -> dict:
    """Stage a new allowlist. Refused while engaged."""
    try:
        from app import vacation_mode as vm
        al = vm.stage_allowlist(
            requestor_allowlist=req.requestor_allowlist,
            path_prefix_allowlist=req.path_prefix_allowlist,
            max_diff_lines=req.max_diff_lines,
        )
        return {"ok": True, "allowlist": al.to_dict()}
    except Exception as exc:
        # vm.VacationModeError → 409 (state conflict).
        msg = str(exc)
        try:
            from app.vacation_mode import VacationModeError
            if isinstance(exc, VacationModeError):
                raise HTTPException(409, msg)
        except HTTPException:
            raise
        except Exception:
            pass
        raise HTTPException(500, msg)


@router.post("/engage")
def engage_endpoint(req: EngageRequest) -> dict:
    """Engage vacation mode.

    Requires ``confirmation_phrase == 'ENGAGE VACATION MODE'``. The
    React UI prompts the operator to type this; bare REST callers must
    pass it too.
    """
    try:
        from app import vacation_mode as vm
        engagement = vm.engage(
            until_ts=time.time() + req.hours * 3600,
            engaged_by=req.engaged_by,
            reason=req.reason,
            confirmation_phrase=req.confirmation_phrase,
        )
        return {"ok": True, "engagement": engagement.to_dict()}
    except Exception as exc:
        msg = str(exc)
        try:
            from app.vacation_mode import VacationModeError
            if isinstance(exc, VacationModeError):
                raise HTTPException(409, msg)
        except HTTPException:
            raise
        except Exception:
            pass
        raise HTTPException(500, msg)


@router.post("/disengage")
def disengage_endpoint(req: Optional[DisengageRequest] = None) -> dict:
    """Disengage vacation mode immediately. Idempotent."""
    try:
        from app import vacation_mode as vm
        actor = (req.disengaged_by if req else "operator") or "operator"
        state = vm.disengage(disengaged_by=actor)
        latest_digest: Optional[str] = None
        try:
            paths = vm.list_digests()
            if paths:
                latest_digest = paths[-1].name
        except Exception:
            pass
        return {
            "ok": True,
            "engaged": state.engaged,
            "latest_digest": latest_digest,
        }
    except Exception as exc:
        raise HTTPException(500, f"disengage failed: {type(exc).__name__}: {exc}")
