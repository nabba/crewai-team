"""Pre-meeting prep nudge (Phase B #2, 2026-05-09).

For every Google Calendar event 30 minutes ahead of "now", send the
operator a structured Signal prep message. Pulls:

  * Title, start time, location, attendees
  * Event description (agenda)
  * Recent inbox messages from any attendee (last 7 days, top 3)
  * Recent Mem0 facts about any attendee (top 3)

Sticks to a small surface — heavy briefings live in
``daily_briefing``; this module is the tactical "your meeting starts
in 30 min, here's the context" nudge. Dedup is per-event-id so we
never double-prep the same event even if the cadence ticks twice
inside the prep window.

Cadence: 5 min via the idle scheduler. The 30-min trigger window is
[28, 32] min — that absorbs cadence drift (any 5-min slot inside the
window catches the event exactly once).

State at ``workspace/life_companion/calendar_prep.json``:
    {"prepped_event_ids": [<id>, ...]}    # capped to last 200
    {"last_run_at": <ts>}

The LIFE_COMPANION_ENABLED master switch already gates the broader
life-companion subsystem; no separate kill switch.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from app.life_companion._common import (
    audit_event,
    background_enabled,
    read_state_json,
    send_signal_alert,
    write_state_json,
)

logger = logging.getLogger(__name__)


_STATE_FILE = "calendar_prep.json"
_RUN_CADENCE_S = 300        # 5 min — idle scheduler tick
_PREP_WINDOW_MIN = 28       # event start ≥ now + 28 min
_PREP_WINDOW_MAX = 32       # event start ≤ now + 32 min
_DEDUP_KEEP = 200           # last N event IDs to remember as "prepped"


def _list_upcoming_events_with_description(
    minutes_ahead_min: int, minutes_ahead_max: int,
) -> list[dict[str, Any]]:
    """Pull events whose start is in [now+min, now+max] minutes.

    Calls Calendar API directly so we get the ``description`` field
    (the existing ``app.tools.gcal_tools._list_events`` helper drops
    it). Returns [] on any error so the prep job never crashes the
    idle scheduler.
    """
    try:
        from app.tools.gcal_tools import _service
    except Exception:
        return []
    svc = _service()
    if svc is None:
        return []
    now = datetime.now(timezone.utc)
    time_min = now.isoformat().replace("+00:00", "Z")
    time_max = (now + timedelta(minutes=minutes_ahead_max + 1)).isoformat().replace("+00:00", "Z")
    try:
        resp = svc.events().list(
            calendarId="primary",
            timeMin=time_min, timeMax=time_max,
            maxResults=20, singleEvents=True, orderBy="startTime",
        ).execute()
    except Exception:
        logger.debug("calendar_prep: API call failed", exc_info=True)
        return []

    events: list[dict[str, Any]] = []
    cutoff_min_dt = now + timedelta(minutes=minutes_ahead_min)
    cutoff_max_dt = now + timedelta(minutes=minutes_ahead_max)
    for ev in resp.get("items", []) or []:
        start_str = (ev.get("start") or {}).get("dateTime") or (ev.get("start") or {}).get("date")
        if not start_str:
            continue
        try:
            start_dt = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            continue
        # All-day events have no tzinfo on a date string — assume UTC.
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc)
        if not (cutoff_min_dt <= start_dt <= cutoff_max_dt):
            continue
        events.append({
            "id": ev.get("id") or "",
            "summary": ev.get("summary") or "(no title)",
            "start_dt": start_dt,
            "location": ev.get("location") or "",
            "attendees": [
                a.get("email") for a in ev.get("attendees", []) or []
                if a.get("email")
            ],
            "description": ev.get("description") or "",
            "html_link": ev.get("htmlLink") or "",
        })
    return events


# ── Context enrichment ────────────────────────────────────────────────────


def _recent_inbox_from(attendee: str, n: int = 3) -> list[str]:
    """Top-N recent inbox subject lines from this attendee. Empty on miss."""
    try:
        from app.tools.gmail_tools import _list_messages  # type: ignore[import-not-found]
    except Exception:
        return []
    try:
        msgs = _list_messages(query=f"from:{attendee} newer_than:7d", max_results=n)
    except Exception:
        return []
    out: list[str] = []
    for m in msgs or []:
        subject = (m.get("subject") or "").strip()
        if subject:
            out.append(f"  • {subject[:80]}")
    return out


def _mem0_facts_about(attendee: str, n: int = 3) -> list[str]:
    """Top-N Mem0 facts mentioning this attendee. Empty on miss."""
    try:
        from app.memory.mem0_manager import get_manager
    except Exception:
        return []
    try:
        mgr = get_manager()
        facts = mgr.search(query=attendee, limit=n)
    except Exception:
        return []
    out: list[str] = []
    for f in facts or []:
        text = ""
        if isinstance(f, dict):
            text = (f.get("memory") or f.get("text") or "")
        elif isinstance(f, str):
            text = f
        text = text.strip()
        if text:
            out.append(f"  • {text[:120]}")
    return out


# ── Composition ───────────────────────────────────────────────────────────


def _format_prep(event: dict, now: datetime) -> str:
    minutes_ahead = max(0, int((event["start_dt"] - now).total_seconds() // 60))
    start_local = event["start_dt"].astimezone()
    start_str = start_local.strftime("%H:%M")

    lines = [
        f"📅 Prep: \"{event['summary']}\"",
        f"In {minutes_ahead} min · {start_str} local"
        + (f" · {event['location']}" if event["location"] else ""),
    ]
    attendees = [a for a in event["attendees"] if a]
    if attendees:
        lines.append(
            "Attendees: " + ", ".join(attendees[:6])
            + (f" (+{len(attendees) - 6})" if len(attendees) > 6 else "")
        )

    if event["description"]:
        agenda = event["description"].strip()
        if len(agenda) > 280:
            agenda = agenda[:280].rsplit(" ", 1)[0] + "…"
        lines.append("\nAgenda:")
        lines.append(agenda)

    # Per-attendee enrichment — first 2 only to keep the message short.
    enriched_blocks: list[str] = []
    for attendee in attendees[:2]:
        sub_lines: list[str] = []
        recent = _recent_inbox_from(attendee, n=2)
        if recent:
            sub_lines.append("Recent emails:")
            sub_lines.extend(recent)
        facts = _mem0_facts_about(attendee, n=2)
        if facts:
            sub_lines.append("From memory:")
            sub_lines.extend(facts)
        if sub_lines:
            enriched_blocks.append(f"\n— {attendee}\n" + "\n".join(sub_lines))
    if enriched_blocks:
        lines.append("".join(enriched_blocks))

    if event["html_link"]:
        lines.append(f"\n{event['html_link']}")
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────


def run() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "ran": False, "events_in_window": 0, "sent": 0,
    }
    if not background_enabled():
        return summary

    state = read_state_json(_STATE_FILE, {
        "last_run_at": 0.0, "prepped_event_ids": [],
    })
    now_ts = time.time()
    if now_ts - float(state.get("last_run_at", 0)) < _RUN_CADENCE_S:
        return summary
    state["last_run_at"] = now_ts
    summary["ran"] = True

    events = _list_upcoming_events_with_description(_PREP_WINDOW_MIN, _PREP_WINDOW_MAX)
    summary["events_in_window"] = len(events)
    if not events:
        write_state_json(_STATE_FILE, state)
        return summary

    prepped: list[str] = list(state.get("prepped_event_ids", []))
    prepped_set = set(prepped)
    now = datetime.now(timezone.utc)
    for ev in events:
        if not ev["id"] or ev["id"] in prepped_set:
            continue
        body = _format_prep(ev, now)
        try:
            send_signal_alert(body, tag=f"calendar_prep:{ev['id']}")
            summary["sent"] += 1
            prepped.append(ev["id"])
        except Exception:
            logger.debug("calendar_prep: send failed for %s", ev["id"], exc_info=True)

    # Cap dedup memory.
    if len(prepped) > _DEDUP_KEEP:
        prepped = prepped[-_DEDUP_KEEP:]
    state["prepped_event_ids"] = prepped
    write_state_json(_STATE_FILE, state)

    audit_event(
        "calendar_prep_pass",
        events_in_window=summary["events_in_window"],
        sent=summary["sent"],
    )
    return summary
