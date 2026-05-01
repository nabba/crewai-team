"""REST API for the Companion Layer — feedback + ideas + sources.

Endpoints under ``/api/cp/companion/*``:

  Phase 4 (feedback + ideas)
    POST /feedback                       — record thumbs up/down + comment
    GET  /ideas/{workspace_id}           — list ideas with current state

  Phase 6 (sources)
    GET    /sources/{workspace_id}             — list workspace sources
    POST   /sources/{workspace_id}             — add a source
    DELETE /sources/{workspace_id}/{source_id} — remove a source
    GET    /sources/{workspace_id}/suggestions — LLM-proposed sources

Phase 9 (Wiki) will add document/wiki endpoints; Phase 13 (cross-workspace)
adds inbox endpoints. To enable in production:
``app.include_router(companion_router)`` next to the other CP routers in
``app/main.py``.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cp/companion", tags=["companion"])


class FeedbackRequest(BaseModel):
    idea_id: str
    workspace_id: str
    polarity: str  # "up" | "down"
    comment: str = ""


@router.post("/feedback")
def post_feedback(req: FeedbackRequest):
    """Record thumbs up/down (and optional comment) on an idea."""
    if req.polarity not in ("up", "down"):
        raise HTTPException(status_code=400,
                            detail="polarity must be 'up' or 'down'")
    try:
        from app.companion import feedback as _fb
        ev_id = _fb.record(
            idea_id=req.idea_id,
            workspace_id=req.workspace_id,
            polarity=_fb.Polarity(req.polarity),
            comment=req.comment,
            source=_fb.Source.REACT,
        )
    except Exception as exc:
        logger.warning("companion_api: feedback record failed: %s", exc)
        raise HTTPException(status_code=500,
                            detail="failed to record feedback")
    return {"event_id": ev_id, "ok": True}


@router.get("/ideas/{workspace_id}")
def list_ideas(workspace_id: str, limit: int = 50):
    """Return recent ideas for a workspace with their CURRENT state.

    State is computed by folding the event log forward over the original
    creation record — see ``app.companion.idea_store.current_state``.
    """
    try:
        from app.companion import idea_store as _is
        records = _is.find_by_workspace(workspace_id, limit=limit)
    except Exception as exc:
        logger.warning("companion_api: list ideas failed: %s", exc)
        raise HTTPException(status_code=500, detail="failed to list ideas")
    out = []
    for r in records:
        try:
            cur = _is.current_state(workspace_id, r.idea_id)
        except Exception:
            cur = r.state
        out.append({
            "idea_id": r.idea_id,
            "cycle_id": r.cycle_id,
            "text": r.text[:1000],
            "role": r.role,
            "created_state": r.state.value if hasattr(r.state, "value")
            else str(r.state),
            "current_state": cur.value if cur is not None else None,
            "lineage_parents": r.lineage_parents,
            "novelty": r.novelty,
            "quality": r.quality,
            "transferability": r.transferability,
            "created_at": r.created_at,
        })
    return {"workspace_id": workspace_id, "count": len(out), "ideas": out}


# ── Phase 6: sources ───────────────────────────────────────────────────────

class AddSourceRequest(BaseModel):
    type: str
    config: dict
    enabled: bool = True


@router.get("/sources/{workspace_id}")
def list_workspace_sources(workspace_id: str):
    try:
        from app.companion import sources as _s
        items = _s.list_sources(workspace_id)
    except Exception as exc:
        logger.warning("companion_api: list sources failed: %s", exc)
        raise HTTPException(status_code=500, detail="failed to list sources")
    return {
        "workspace_id": workspace_id,
        "count": len(items),
        "sources": [
            {
                "source_id": s.source_id,
                "type": s.type,
                "config": s.config,
                "enabled": s.enabled,
                "added_at": s.added_at,
                "last_ingested_at": s.last_ingested_at,
                "last_ingest_status": s.last_ingest_status,
            }
            for s in items
        ],
    }


@router.post("/sources/{workspace_id}")
def add_workspace_source(workspace_id: str, req: AddSourceRequest):
    from app.companion import sources as _s
    if req.type not in _s.ALLOWED_TYPES:
        raise HTTPException(status_code=400,
                            detail=f"type must be one of {_s.ALLOWED_TYPES}")
    src = _s.add_source(workspace_id, req.type, req.config,
                        enabled=req.enabled)
    if src is None:
        raise HTTPException(status_code=400, detail="source rejected")
    return {"source_id": src.source_id, "ok": True}


@router.delete("/sources/{workspace_id}/{source_id}")
def delete_workspace_source(workspace_id: str, source_id: str):
    from app.companion import sources as _s
    removed = _s.remove_source(workspace_id, source_id)
    if not removed:
        raise HTTPException(status_code=404, detail="source not found")
    return {"ok": True}


@router.get("/sources/{workspace_id}/suggestions")
def get_source_suggestions(workspace_id: str, limit: int = 5):
    try:
        from app.companion import source_suggester as _ss
        proposals = _ss.propose(workspace_id, max_count=max(1, min(limit, 10)))
    except Exception as exc:
        logger.warning("companion_api: source suggester failed: %s", exc)
        raise HTTPException(status_code=500,
                            detail="failed to generate suggestions")
    return {
        "workspace_id": workspace_id,
        "count": len(proposals),
        "suggestions": proposals,
    }
