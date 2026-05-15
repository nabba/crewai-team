"""Control plane — long-term goal review endpoints at /api/cp/goals.

PROGRAM §46.9 (Q9.6). Operator-facing surface for the quarterly
long-term goal review:

  GET  /api/cp/goals/reviews           list past reviews
  POST /api/cp/goals/review            trigger review now (force)
  GET  /api/cp/goals/state             current goals + last-run state

Auth: ``require_gateway_auth``.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from app.control_plane.auth_dep import require_gateway_auth

logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/api/cp/goals",
    tags=["control-plane", "goals"],
    dependencies=[Depends(require_gateway_auth)],
)


@router.get("/state")
def get_state() -> dict[str, Any]:
    """Return current_goals + last-review metadata."""
    try:
        from app.identity.long_term_goal_review import (
            _current_goals, list_recent_reviews,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"goal review subsystem unavailable: {exc}",
        )
    goals = _current_goals()
    recent = list_recent_reviews(limit=1)
    return {
        "current_goals_count": len(goals),
        "current_goals": goals,
        "last_review": recent[0] if recent else None,
    }


@router.get("/reviews")
def list_reviews(limit: int = 12) -> dict[str, Any]:
    try:
        from app.identity.long_term_goal_review import list_recent_reviews
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"goal review subsystem unavailable: {exc}",
        )
    items = list_recent_reviews(limit=max(1, min(limit, 50)))
    return {"count": len(items), "reviews": items}


@router.post("/review")
def trigger_review(force: bool = True) -> dict[str, Any]:
    """Operator-triggered review pass. Bypasses cadence by default."""
    try:
        from app.identity.long_term_goal_review import run_review
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"goal review subsystem unavailable: {exc}",
        )
    result = run_review(force=force)
    return {
        "status": result.status,
        "quarter_label": result.quarter_label,
        "written_to": result.written_to,
        "failure_reason": result.failure_reason,
        "attempts": result.attempts,
    }
