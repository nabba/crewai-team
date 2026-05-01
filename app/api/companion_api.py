"""REST API for the Companion Layer — feedback intake + idea browsing.

Adds endpoints under ``/api/cp/companion/*``. Phase 4 ships:
  POST /api/cp/companion/feedback  — record thumbs up/down/comment from React
  GET  /api/cp/companion/ideas/<workspace_id>  — list recent ideas with state

Phase 9 (Wiki) will add document/wiki endpoints; Phase 13 (cross-workspace)
will add inbox endpoints. To enable: add ``app.include_router(companion_router)``
to ``app/main.py`` alongside the other CP routers.
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
