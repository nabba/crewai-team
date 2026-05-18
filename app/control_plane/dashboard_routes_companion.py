"""Control-plane dashboard routes — companion topic.

Workspace Companion + person-correlation stack + tensions + cross-modal patterns.

Extracted from app/control_plane/dashboard_api.py as part of WP G
Phase 1 (2026-05-17); wired into the parent router via
``include_router`` in Phase 2 (2026-05-18). The parent router in
``dashboard_api.py`` carries the ``/api/cp`` prefix and the
``require_gateway_auth`` dependency, both of which propagate to
every route here — so the URL surface and auth boundary are
identical to the pre-Phase-1 monolith.
"""
import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# No prefix or dependencies here — the parent router in dashboard_api.py
# carries those, so every path below is identical to the original.
router = APIRouter()


@router.get("/companion/tensions")
def companion_tensions(status: str = Query("OPEN"), min_freshness: float = Query(0.0, ge=0.0, le=1.0)):
    """Q4#16 (PROGRAM §41) — open questions the companion tracks on
    the operator's behalf. Read-only; mutations go through the
    POST endpoints below.
    """
    try:
        from app.companion.tensions import list_tensions
        tensions = list_tensions(status=status, min_freshness=min_freshness) or []
    except Exception as exc:
        logger.debug("companion/tensions failed: %s", exc, exc_info=True)
        return {"tensions": [], "error": str(exc)}
    return {
        "tensions": [
            {
                **t.to_dict(),
                "freshness": round(t.freshness(), 4),
            }
            for t in tensions
        ],
        "as_of": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/notify/fatigue")
def notify_fatigue(window_hours: float = Query(168.0, ge=1.0, le=720.0)):
    """Q4#17 (PROGRAM §41) — recent notification arbiter decisions.
    Operator-facing review surface. Default window = 1 week.
    """
    try:
        from app.notify.fatigue import list_recent, daily_suppression_rate
        events = list_recent(window_hours=window_hours) or []
        suppressed, total, rate = daily_suppression_rate()
    except Exception as exc:
        logger.debug("notify/fatigue failed: %s", exc, exc_info=True)
        return {"events": [], "error": str(exc)}
    return {
        "events": events,
        "today_summary": {
            "suppressed": suppressed,
            "total_decisions": total,
            "suppression_rate": round(rate, 4),
        },
        "as_of": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/companion/people")
def companion_people():
    """L1 — current person profile snapshot. Returns counts only.
    Muted people are filtered out at the read path."""
    try:
        from app.companion.person_model import current_profile
        return current_profile()
    except Exception as exc:
        logger.debug("companion/people failed: %s", exc, exc_info=True)
        return {"people": [], "muted": [], "enabled": False, "error": str(exc)}


class PersonAction(BaseModel):
    person_id: str


@router.post("/companion/people/mute")
def companion_people_mute(body: PersonAction):
    try:
        from app.companion.person_model import mute
        ok = mute(body.person_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"ok": ok}


@router.post("/companion/people/unmute")
def companion_people_unmute(body: PersonAction):
    try:
        from app.companion.person_model import unmute
        ok = unmute(body.person_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"ok": ok}


@router.post("/companion/people/forget")
def companion_people_forget(body: PersonAction):
    try:
        from app.companion.person_model import forget
        ok = forget(body.person_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"ok": ok}


@router.post("/companion/people/forget-all")
def companion_people_forget_all():
    try:
        from app.companion.person_model import forget_all
        n = forget_all()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"forgotten": n}


@router.get("/companion/people/centrality")
def companion_people_centrality():
    """L2 — centrality scores (when L2 enabled). Sorted by last_seen,
    NEVER by score (deliberate Goodhart guard)."""
    try:
        from app.companion.person_centrality import compute_centrality
        return compute_centrality()
    except Exception as exc:
        return {"scores": [], "enabled": False, "error": str(exc)}


@router.get("/companion/people/suggestions")
def companion_people_suggestions(limit: int = Query(50, ge=1, le=500)):
    """L3 + L4.4 — recent emitted suggestions log."""
    try:
        from app.companion.person_suggestions import recent_emitted
        return {"suggestions": recent_emitted(limit=limit)}
    except Exception as exc:
        return {"suggestions": [], "error": str(exc)}


@router.post("/companion/people/mute-suggestions")
def companion_people_mute_sug(body: PersonAction):
    try:
        from app.companion.person_suggestions import mute_suggestions_for
        ok = mute_suggestions_for(body.person_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"ok": ok}


@router.get("/companion/people/graph")
def companion_people_graph():
    """L4 — co-appearance graph (when L4 enabled). NEVER exported in
    DR backups (excluded by social_graph denylist fragment)."""
    try:
        from app.companion.social_graph import current_graph
        return current_graph()
    except Exception as exc:
        return {"edges": [], "nodes": [], "enabled": False, "error": str(exc)}


@router.post("/companion/people/graph/forget")
def companion_people_graph_forget():
    """L4 — delete the entire graph file. Preserves L1 profiles."""
    try:
        from app.companion.social_graph import forget_graph
        n = forget_graph()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"edges_deleted": n}


class PairMuteBody(BaseModel):
    a: str
    b: str


@router.post("/companion/people/graph/mute-pair")
def companion_people_graph_mute_pair(body: PairMuteBody):
    try:
        from app.companion.social_graph import mute_pair
        ok = mute_pair(body.a, body.b)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"ok": ok}


class PathQueryBody(BaseModel):
    source: str
    target: str
    max_hops: int = 6


@router.post("/companion/people/graph/path")
def companion_people_graph_path(body: PathQueryBody):
    """L4.1 — shortest path query (operator-initiated)."""
    try:
        from app.companion.graph_features.shortest_path import find_path
        return find_path(body.source, body.target, body.max_hops)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/companion/people/graph/path-opt-out")
def companion_people_graph_path_optout(body: PersonAction):
    try:
        from app.companion.social_graph import opt_out_of_paths
        ok = opt_out_of_paths(body.person_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"ok": ok}


@router.get("/companion/people/graph/communities")
def companion_people_graph_communities():
    """L4.2 — community detection result."""
    try:
        from app.companion.graph_features.communities import compute_communities
        return compute_communities()
    except Exception as exc:
        return {"clusters": [], "modularity": 0.0, "enabled": False, "error": str(exc)}


class DissolveBody(BaseModel):
    member_emails: list[str]


@router.post("/companion/people/graph/dissolve-cluster")
def companion_people_graph_dissolve(body: DissolveBody):
    try:
        from app.companion.graph_features.communities import dissolve_cluster
        ok = dissolve_cluster(body.member_emails)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"ok": ok}


@router.get("/companion/people/graph/structural")
def companion_people_graph_structural():
    """L4.3 — bridges + cut-vertices."""
    try:
        from app.companion.graph_features.bridges import compute_structural
        return compute_structural()
    except Exception as exc:
        return {"bridges": [], "cut_vertices": [], "enabled": False, "error": str(exc)}


@router.get("/companion/cross-modal-patterns")
def companion_cross_modal_patterns(n: int = Query(20, ge=1, le=100), min_strength: float = Query(0.7, ge=0.0, le=1.0)):
    """Q4#15 (PROGRAM §41) — recent convergence patterns the system
    detected across modalities (conversations, emails, calendar,
    feedback, affect, tickets). Read-only.
    """
    try:
        from app.companion.cross_modal_patterns import list_recent_patterns
        patterns = list_recent_patterns(n=n, min_strength=min_strength) or []
    except Exception as exc:
        logger.debug("companion/cross-modal-patterns failed: %s", exc, exc_info=True)
        return {"patterns": [], "error": str(exc)}
    return {
        "patterns": patterns,
        "as_of": datetime.now(timezone.utc).isoformat(),
    }


class TensionResolve(BaseModel):
    resolution: str


@router.post("/companion/tensions/{tid}/resolve")
def companion_tensions_resolve(tid: str, body: TensionResolve):
    """Operator marks a tension RESOLVED with a short resolution note."""
    try:
        from app.companion.tensions import resolve_tension
        t = resolve_tension(tid, body.resolution)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    if t is None:
        raise HTTPException(status_code=404, detail=f"tension {tid!r} not found")
    return t.to_dict()


class TensionCreate(BaseModel):
    question: str
    workspace_id: str | None = None


@router.post("/companion/tensions")
def companion_tensions_create(body: TensionCreate):
    """Operator manually files a tension. Use case: operator says
    'I'm wondering about X' via /cp/companion rather than letting
    the regex detector pick it up from conversation."""
    try:
        from app.companion.tensions import create_tension
        t = create_tension(
            question=body.question,
            workspace_id=body.workspace_id,
            detection_source="manual:operator",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    if t is None:
        raise HTTPException(
            status_code=400,
            detail="tension rejected (question length or OPEN cap reached)",
        )
    return t.to_dict()


