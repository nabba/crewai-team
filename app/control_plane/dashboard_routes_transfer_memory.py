"""Control-plane dashboard routes — transfer_memory topic.

Transfer Insight Layer (Phase 17d) — cross-domain memory promotion.

Extracted from app/control_plane/dashboard_api.py as part of WP G
Phase 1 (productization plan, 2026-05-17). Pure code movement —
routes, classes, and helpers are verbatim. The parent router in
``dashboard_api.py`` re-attaches the ``/api/cp`` prefix and the
``require_gateway_auth`` dependency via ``include_router``, so the
URL surface and auth boundary are unchanged.
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


@router.get("/transfer-memory/overview")
def transfer_memory_overview():
    """Top-line metrics for the Transfer Insight Layer."""
    from app.transfer_memory.dashboard import get_overview
    return get_overview()


@router.get("/transfer-memory/by-source-kind")
def transfer_memory_by_source_kind():
    """Per-source-kind compile + outcome counters."""
    from app.transfer_memory.dashboard import get_by_source_kind
    return get_by_source_kind()


@router.get("/transfer-memory/recent")
def transfer_memory_recent(days: int = Query(7, ge=1, le=90)):
    """Compile + retrieval activity over the trailing N days."""
    from app.transfer_memory.dashboard import get_recent_activity
    return get_recent_activity(days=days)


@router.get("/transfer-memory/top-performers")
def transfer_memory_top(n: int = Query(10, ge=1, le=100)):
    """Most-surfaced records with no negative-transfer entries."""
    from app.transfer_memory.dashboard import get_top_performers
    return get_top_performers(n=n)


@router.get("/transfer-memory/worst-performers")
def transfer_memory_worst(n: int = Query(10, ge=1, le=100)):
    """Records with the most negative-transfer entries."""
    from app.transfer_memory.dashboard import get_worst_performers
    return get_worst_performers(n=n)


@router.get("/transfer-memory/sanitizer-stats")
def transfer_memory_sanitizer_stats():
    """Hard-reject + demotion totals across all compiled rows."""
    from app.transfer_memory.dashboard import get_sanitizer_stats
    return get_sanitizer_stats()


@router.get("/transfer-memory/promotion-candidates")
def transfer_memory_promotion_candidates():
    """Eligible records waiting for promotion (operator review)."""
    from app.transfer_memory.dashboard import get_promotion_candidates
    return get_promotion_candidates()


@router.get("/transfer-memory/negative-transfer")
def transfer_memory_negative_transfer():
    """Tag distribution and recent negative-transfer events."""
    from app.transfer_memory.dashboard import get_negative_transfer_stats
    return get_negative_transfer_stats()


@router.get("/transfer-memory/source-target-matrix")
def transfer_memory_source_target_matrix():
    """Source-domain × target-domain co-occurrence from shadow retrievals."""
    from app.transfer_memory.dashboard import get_source_to_target_matrix
    return get_source_to_target_matrix()


@router.post("/transfer-memory/promote/{record_id}")
def transfer_memory_promote(record_id: str):
    """Operator-driven manual promotion of a single shadow record.

    Bypasses the cadence guard but still requires the record to pass
    eligibility checks (age, surface count, no negative attribution).
    """
    from app.transfer_memory.promotion import manual_promote
    ok = manual_promote(record_id)
    if not ok:
        raise HTTPException(409, "Promotion rejected — record ineligible or update failed")
    return {"promoted": True, "skill_record_id": record_id}


