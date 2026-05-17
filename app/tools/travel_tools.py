"""
travel_tools.py — agent-callable surface over the TripIt travel monitor.

Wraps :mod:`app.life_companion.travel` (Q9.3, PROGRAM §46.6) so PIM-style
chat questions like "what are my next flights?" route to a real tool
instead of returning the generic "missing tool" answer.

Two CrewAI tools:

    list_upcoming_flights    flight segments only, with optional live-status
    list_upcoming_trips      every segment (flight / ferry / train / hotel
                             / car / other) in the window

Data source is the on-disk snapshot the travel idle job writes to
``workspace/life_companion/travel/tripit_trips.json``. We never hit the
TripIt feed inline — the idle job owns refresh on a 6h cadence so the
chat path stays fast and predictable.

Empty-state behaviour: when the snapshot is missing or empty the tools
return a JSON object with ``status="no_data"`` plus a short hint at
``hint`` so the LLM can explain to the user *why* there's nothing to
show (URL not configured / idle job hasn't run yet / no trips in
window).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────


def _hint_when_empty() -> str:
    """Diagnose *why* upcoming_trips is empty so the LLM can tell the user."""
    try:
        from app.life_companion.travel import _get_tripit_url
    except Exception:
        return "travel monitor unavailable in this build"
    if not _get_tripit_url().strip():
        return (
            "TripIt iCal URL is not configured — set it in React "
            "/cp/settings → Travel card"
        )
    return (
        "no trips in window — either there's nothing scheduled or the "
        "travel idle job hasn't refreshed yet (6h cadence)"
    )


def _segments_in_window(window_days: int) -> list[dict[str, Any]]:
    try:
        from app.life_companion.travel import upcoming_trips, _read_flight_status
    except Exception:
        return []
    segs = upcoming_trips(window_days=window_days)
    out: list[dict[str, Any]] = []
    for s in segs:
        row: dict[str, Any] = {
            "kind": s.kind,
            "starts_at": s.starts_at,
            "ends_at": s.ends_at,
            "summary": s.summary,
            "location": s.location,
        }
        if s.kind == "flight" and s.flight_number:
            row["flight_number"] = s.flight_number
            status = _read_flight_status(s.flight_number)
            if status is not None:
                row["live_status"] = {
                    "status": status.status,
                    "delay_minutes": status.delay_minutes,
                    "gate": status.gate,
                    "terminal": status.terminal,
                    "fetched_at": status.fetched_at,
                }
        out.append(row)
    return out


# ── CrewAI tool factory ────────────────────────────────────────────────


def create_travel_tools(agent_id: str = "pim") -> list:
    """Build CrewAI BaseTool instances for the travel monitor.

    Returns ``[]`` if CrewAI tooling isn't available in this build (e.g.
    a test runner without the optional dep). Returns the tools even
    when TripIt is unconfigured — they return an informative
    ``status="no_data"`` payload in that case.
    """
    try:
        from crewai.tools import tool
    except ImportError:
        return []

    @tool("list_upcoming_flights")
    def flights_tool(window_days: int = 14) -> str:
        """List the operator's upcoming flights from the TripIt snapshot.

        Use this when the user asks "what flights do I have", "when am I
        flying next", or similar. Returns JSON.

        Args:
            window_days: how far ahead to look (1–60, default 14).

        Returns:
            JSON. On success:
                {"status": "ok", "count": N, "flights": [...]}
            Each flight row has ``starts_at`` (ISO UTC), ``ends_at``,
            ``flight_number`` (IATA e.g. "LH881"), ``summary``,
            ``location``, and — when an Aviationstack key is configured
            and the flight is within 24 h — a ``live_status`` block
            with delay / gate / terminal.

            When no data: {"status": "no_data", "hint": "..."}.
        """
        try:
            window = max(1, min(60, int(window_days)))
        except (TypeError, ValueError):
            window = 14
        segs = _segments_in_window(window)
        flights = [s for s in segs if s.get("kind") == "flight"]
        if not flights:
            return json.dumps({
                "status": "no_data",
                "window_days": window,
                "hint": _hint_when_empty(),
            }, ensure_ascii=False)
        return json.dumps({
            "status": "ok",
            "window_days": window,
            "count": len(flights),
            "flights": flights,
        }, ensure_ascii=False)

    @tool("list_upcoming_trips")
    def trips_tool(window_days: int = 14, kind: str = "") -> str:
        """List the operator's upcoming trip segments (every kind, not
        only flights).

        Use this for "am I traveling on <date>", "what's my hotel for
        <city>", or "what's on my travel calendar". For flights-only
        questions prefer ``list_upcoming_flights``.

        Args:
            window_days: how far ahead to look (1–60, default 14).
            kind: optional filter — one of "flight", "ferry", "train",
                "hotel", "car", "other". Empty = all kinds.

        Returns:
            JSON. On success:
                {"status": "ok", "count": N, "segments": [...]}
            Each segment has ``kind``, ``starts_at`` (ISO UTC),
            ``ends_at``, ``summary``, ``location`` (and
            ``flight_number`` / ``live_status`` when applicable).

            When no data: {"status": "no_data", "hint": "..."}.
        """
        try:
            window = max(1, min(60, int(window_days)))
        except (TypeError, ValueError):
            window = 14
        segs = _segments_in_window(window)
        kind_filter = (kind or "").strip().lower()
        if kind_filter:
            segs = [s for s in segs if s.get("kind") == kind_filter]
        if not segs:
            return json.dumps({
                "status": "no_data",
                "window_days": window,
                "kind_filter": kind_filter,
                "hint": _hint_when_empty(),
            }, ensure_ascii=False)
        return json.dumps({
            "status": "ok",
            "window_days": window,
            "kind_filter": kind_filter,
            "count": len(segs),
            "segments": segs,
        }, ensure_ascii=False)

    return [flights_tool, trips_tool]
