"""Control-plane endpoints for browser-history ingestion.

Mounted at ``/api/cp/browse``. All endpoints honor the same Bearer
auth as the rest of the control plane.

Surface
-------

  GET  /api/cp/browse/state                 master switch + stats
  GET  /api/cp/browse/recent                last N days of events (limited)
  GET  /api/cp/browse/categories            top topic clusters from daily batch
  GET  /api/cp/browse/blocklist             seeded + operator entries
  POST /api/cp/browse/mute                  add a domain to the operator file
  POST /api/cp/browse/forget                purge everything OR per-domain OR per-day

The two write endpoints emit identity-continuity ledger events via
the underlying store/blocklist helpers (already wired in Phase A).
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.control_plane.auth_dep import require_gateway_auth

logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/api/cp/browse",
    tags=["control-plane", "browse"],
    dependencies=[Depends(require_gateway_auth)],
)


def _lazy_imports():
    """Module imports inside the request handler.

    Two reasons:
      1. Module-load-time side effects (workspace dir creation, env
         var checks) only run when the endpoint is actually invoked.
      2. A broken browse module shouldn't keep the rest of /cp/ from
         booting — the registration in ``main.py`` is itself wrapped.
    """
    from app.browse import blocklist, store
    from app.browse.topic_extraction import topics_for_day
    return store, blocklist, topics_for_day


# ── State + stats ─────────────────────────────────────────────────────


@router.get("/state")
def get_state(window_days: int = Query(default=7, ge=1, le=30)) -> dict[str, Any]:
    """Master-switch state + recent-window stats for the React card."""
    store, _bl, _topics = _lazy_imports()
    return {
        "enabled": store.enabled(),
        "stats": store.event_counts(days=window_days),
    }


# ── Recent events ─────────────────────────────────────────────────────


@router.get("/recent")
def get_recent(
    days: int = Query(default=3, ge=1, le=14),
    limit: int = Query(default=200, ge=1, le=2000),
) -> dict[str, Any]:
    """Last N days of canonical events. Returns at most ``limit`` rows."""
    store, _bl, _topics = _lazy_imports()
    events = store.list_events_window(days=days)
    # Newest-first.
    events.sort(key=lambda e: e.visit_ts, reverse=True)
    return {
        "count": len(events),
        "events": [e.to_dict() for e in events[:limit]],
        "truncated": len(events) > limit,
    }


# ── Topic clusters ────────────────────────────────────────────────────


@router.get("/categories")
def get_categories(
    days: int = Query(default=7, ge=1, le=30),
) -> dict[str, Any]:
    """Aggregated topic-cluster counts from the daily LLM batch."""
    store, _bl, topics_for_day = _lazy_imports()
    from collections import Counter
    cur = datetime.now(timezone.utc).date()
    agg: Counter[str] = Counter()
    sample_map: dict[str, list[str]] = {}
    available_days: list[str] = []
    for i in range(days):
        day = cur - timedelta(days=i)
        result = topics_for_day(day)
        if result is None:
            continue
        available_days.append(day.isoformat())
        for t in result.topics:
            if not t.label:
                continue
            agg[t.label] += int(t.title_count or 0)
            if t.label not in sample_map and t.sample_titles:
                sample_map[t.label] = list(t.sample_titles[:3])
    rows = [
        {
            "label": label,
            "count": count,
            "sample_titles": sample_map.get(label, []),
        }
        for label, count in agg.most_common(50)
    ]
    return {
        "categories": rows,
        "window_days": days,
        "days_with_data": available_days,
    }


# ── Blocklist ─────────────────────────────────────────────────────────


@router.get("/blocklist")
def get_blocklist() -> dict[str, Any]:
    _store, blocklist, _topics = _lazy_imports()
    return {
        "seeded": blocklist.list_seed_entries(),
        "operator": blocklist.list_operator_entries(),
    }


class _MuteBody(BaseModel):
    domain: str = Field(..., min_length=1, max_length=253)


@router.post("/mute")
def post_mute(body: _MuteBody) -> dict[str, Any]:
    _store, blocklist, _topics = _lazy_imports()
    added = blocklist.mute_domain(body.domain)
    return {"added": added, "domain": body.domain.lower().strip()}


# ── Forget paths ──────────────────────────────────────────────────────


class _ForgetBody(BaseModel):
    """One of (all, domain, day) must be set. Strict — no mixed
    semantics."""
    scope: str = Field(..., pattern="^(all|domain|day)$")
    domain: str | None = Field(default=None)
    day: str | None = Field(default=None)


@router.post("/forget")
def post_forget(body: _ForgetBody) -> dict[str, Any]:
    store, _bl, _topics = _lazy_imports()
    if body.scope == "all":
        n = store.forget_all()
        return {"scope": "all", "files_removed": n}
    if body.scope == "domain":
        if not body.domain:
            raise HTTPException(status_code=400, detail="domain required")
        n = store.forget_domain(body.domain)
        return {
            "scope": "domain",
            "domain": body.domain,
            "rows_removed": n,
        }
    if body.scope == "day":
        if not body.day:
            raise HTTPException(status_code=400, detail="day required")
        try:
            d = date.fromisoformat(body.day)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        removed = store.forget_day(d)
        return {"scope": "day", "day": body.day, "removed": removed}
    raise HTTPException(status_code=400, detail=f"unknown scope: {body.scope}")
