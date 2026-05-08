"""
gcal_tools.py — agent-callable Google Calendar operations.

Two CrewAI tools:

    list_google_calendar_events   upcoming/past events with start/end + summary
    create_google_calendar_event  create a new event with start, end, title, attendees

Sits alongside ``app.tools.calendar_tools`` (macOS Calendar.app via AppleScript).
The macOS path keeps working when the Mac is awake; this Google-native path
keeps working when the Mac is asleep or the agent runs in the cloud.

All times use ISO 8601 strings; the helper accepts a naive `YYYY-MM-DD HH:MM`
form too and stamps the system's default timezone (Europe/Helsinki) to keep
the agent prompt simple.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def _service():
    from app.google_workspace import get_service
    return get_service("calendar")


def _default_tz() -> str:
    from app.config import get_settings
    return get_settings().default_timezone or "Europe/Helsinki"


def _parse_time(value: str) -> dict[str, str]:
    """Convert various input shapes into a Calendar API ``EventDateTime`` dict."""
    value = (value or "").strip()
    if not value:
        return {}
    # All-day form: YYYY-MM-DD only.
    if len(value) == 10 and value[4] == "-" and value[7] == "-":
        return {"date": value}
    # ISO 8601 with timezone — pass through verbatim.
    if "T" in value and (value.endswith("Z") or "+" in value[10:] or value[-6] == "-"):
        return {"dateTime": value}
    # Tolerant fallback: parse "YYYY-MM-DD HH:MM[:SS]" with the default tz.
    fmt_candidates = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S",
                      "%Y-%m-%dT%H:%M")
    for fmt in fmt_candidates:
        try:
            dt = datetime.strptime(value, fmt)
            return {
                "dateTime": dt.strftime("%Y-%m-%dT%H:%M:%S"),
                "timeZone": _default_tz(),
            }
        except ValueError:
            continue
    # Last-ditch: hand the raw string to Google and let it complain.
    return {"dateTime": value, "timeZone": _default_tz()}


def _list_events(
    calendar_id: str = "primary",
    max_results: int = 10,
    time_min: str = "",
    time_max: str = "",
    query: str = "",
) -> list[dict]:
    svc = _service()
    if svc is None:
        return []
    params: dict[str, Any] = {
        "calendarId": calendar_id,
        "maxResults": max(1, min(50, max_results)),
        "singleEvents": True,
        "orderBy": "startTime",
    }
    if not time_min:
        time_min = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    params["timeMin"] = time_min
    if time_max:
        params["timeMax"] = time_max
    if query:
        params["q"] = query

    resp = svc.events().list(**params).execute()
    events = []
    for ev in resp.get("items", []) or []:
        events.append({
            "id": ev.get("id"),
            "summary": ev.get("summary", ""),
            "start": ev.get("start", {}).get("dateTime") or ev.get("start", {}).get("date"),
            "end": ev.get("end", {}).get("dateTime") or ev.get("end", {}).get("date"),
            "location": ev.get("location", ""),
            "attendees": [a.get("email") for a in ev.get("attendees", []) or []],
            "html_link": ev.get("htmlLink", ""),
        })
    return events


def _create_event(
    summary: str,
    start: str,
    end: str,
    *,
    description: str = "",
    location: str = "",
    attendees: list[str] | None = None,
    calendar_id: str = "primary",
) -> dict:
    svc = _service()
    if svc is None:
        return {"error": "Calendar not configured"}
    body: dict[str, Any] = {
        "summary": summary,
        "start": _parse_time(start),
        "end": _parse_time(end),
    }
    if description:
        body["description"] = description
    if location:
        body["location"] = location
    if attendees:
        body["attendees"] = [{"email": e} for e in attendees if e]

    ev = svc.events().insert(calendarId=calendar_id, body=body).execute()
    return {
        "id": ev.get("id"),
        "html_link": ev.get("htmlLink", ""),
        "status": ev.get("status", ""),
        "summary": ev.get("summary", ""),
    }


# ── CrewAI tool factory ────────────────────────────────────────────────────

def create_gcal_tools(agent_id: str = "pim") -> list:
    """Build CrewAI BaseTool instances for Google Calendar. Returns [] if not configured."""
    try:
        from app.google_workspace import is_configured
        if not is_configured():
            return []
    except Exception:
        return []
    if _service() is None:
        return []

    try:
        from crewai.tools import tool
    except ImportError:
        return []

    @tool("list_google_calendar_events")
    def list_tool(
        max_results: int = 10,
        time_min: str = "",
        time_max: str = "",
        query: str = "",
        calendar_id: str = "primary",
    ) -> str:
        """List upcoming Google Calendar events. Returns JSON.

        Args:
            max_results: 1–50 events.
            time_min: ISO 8601 lower bound (default: now).
            time_max: ISO 8601 upper bound (default: unbounded).
            query: free-text search across event titles/descriptions.
            calendar_id: calendar to query (default: "primary").
        """
        import json
        events = _list_events(
            calendar_id=calendar_id, max_results=max_results,
            time_min=time_min, time_max=time_max, query=query,
        )
        return json.dumps(events, ensure_ascii=False)

    @tool("create_google_calendar_event")
    def create_tool(
        summary: str,
        start: str,
        end: str,
        description: str = "",
        location: str = "",
        attendees: str = "",
        calendar_id: str = "primary",
    ) -> str:
        """Create a Google Calendar event. Returns JSON {id, html_link, status, summary}.

        Time formats accepted for `start`/`end`:
          - "YYYY-MM-DD" (all-day)
          - "YYYY-MM-DD HH:MM" (naive, stamped with default timezone)
          - ISO 8601 with offset/Z

        attendees: comma-separated email list.
        """
        import json
        att_list = [a.strip() for a in attendees.split(",") if a.strip()]
        return json.dumps(_create_event(
            summary=summary, start=start, end=end,
            description=description, location=location,
            attendees=att_list, calendar_id=calendar_id,
        ))

    return [list_tool, create_tool]
