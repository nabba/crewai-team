"""REST API for the self-improvement velocity surface.

PROGRAM §51 — Q16 Theme 4. Endpoints under
``/api/cp/self-improvement/*``:

  GET  /velocity?window_days=N   — full velocity rollup

The endpoint reads only — it never mutates state. Failure-isolated:
when an upstream source breaks, that section reports
``{"available": false}`` but the whole call still succeeds.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cp/self-improvement", tags=["self-improvement"])


@router.get("/velocity")
def get_velocity(
    window_days: int = Query(365, ge=30, le=3650),
) -> dict:
    """Velocity rollup. Window 30..3650 days; default 365."""
    try:
        from app.self_improvement.velocity import velocity_summary
        return velocity_summary(window_days=window_days)
    except Exception as exc:
        raise HTTPException(
            500,
            f"velocity aggregation failed: {type(exc).__name__}: {exc}",
        )
