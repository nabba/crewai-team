"""72-hour calendar horizon scan + conflict detection (Phase G #1).

Companion to ``calendar_prep.py`` (Phase B #2). The prep module
fires 30 min before each event with tactical context. This module
runs once per morning and scans the next 72 h for two scope-specific
problems:

  1. **Time conflicts** — overlapping events. ("9-10 AM gov briefing
     overlaps with 9:30-10:30 standup.")
  2. **Density warnings** — N events back-to-back with <15 min
     buffer. (Operator gets brunt of "5 calls in a row" without
     warning otherwise.)

Cadence: daily, 8:00 local (configurable). The morning briefing also
sees calendar events — but that briefing scans ONLY the next 24 h,
not the full 72 h horizon. Conflicts spanning today→Wednesday are
invisible to the morning brief; this module catches them.

Output: one Signal message per pass, only if conflicts/density-warns
exist. Otherwise quiet.

Master switch: ``CALENDAR_HORIZON_ENABLED`` (default ON).
"""
from __future__ import annotations

import logging
import os
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


_STATE_FILE = "calendar_horizon.json"
_RUN_CADENCE_S = 3600  # hourly probe; internal once-per-day gate
_HORIZON_HOURS = 72
_DENSITY_THRESHOLD_MIN = 15  # back-to-back if next event starts within this
_DEFAULT_TARGET_HOUR = 8


def _enabled() -> bool:
    return os.getenv("CALENDAR_HORIZON_ENABLED", "true").lower() in (
        "true", "1", "yes", "on",
    )


def _target_hour() -> int:
    raw = os.getenv("CALENDAR_HORIZON_HOUR", str(_DEFAULT_TARGET_HOUR)).strip()
    try:
        return max(0, min(23, int(raw)))
    except ValueError:
        return _DEFAULT_TARGET_HOUR


def _now_local() -> datetime:
    return datetime.now().astimezone()


def _list_events_72h() -> list[dict[str, Any]]:
    """Pull events from now to now+72 h. Returns parsed records."""
    try:
        from app.tools.gcal_tools import _service
    except Exception:
        return []
    svc = _service()
    if svc is None:
        return []
    now = datetime.now(timezone.utc)
    time_min = now.isoformat().replace("+00:00", "Z")
    time_max = (now + timedelta(hours=_HORIZON_HOURS)).isoformat().replace("+00:00", "Z")
    try:
        resp = svc.events().list(
            calendarId="primary",
            timeMin=time_min, timeMax=time_max,
            maxResults=80, singleEvents=True, orderBy="startTime",
        ).execute()
    except Exception:
        logger.debug("calendar_horizon: API call failed", exc_info=True)
        return []

    events: list[dict[str, Any]] = []
    for ev in resp.get("items", []) or []:
        start_str = (ev.get("start") or {}).get("dateTime") or (ev.get("start") or {}).get("date")
        end_str = (ev.get("end") or {}).get("dateTime") or (ev.get("end") or {}).get("date")
        if not start_str or not end_str:
            continue
        try:
            start_dt = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            continue
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc)
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=timezone.utc)
        events.append({
            "id": ev.get("id") or "",
            "summary": ev.get("summary") or "(no title)",
            "start": start_dt,
            "end": end_dt,
            "all_day": ev.get("start", {}).get("date") is not None,
        })
    return events


# ── Conflict + density detection ─────────────────────────────────────────


def _detect_conflicts(events: list[dict]) -> list[tuple[dict, dict]]:
    """Pairs of overlapping events. All-day events excluded — they
    overlap everything and would noise the alert."""
    pairs: list[tuple[dict, dict]] = []
    timed = [e for e in events if not e["all_day"]]
    for i, a in enumerate(timed):
        for b in timed[i + 1:]:
            if b["start"] >= a["end"]:
                # Sorted ascending — once b starts after a ends, no
                # later event can overlap a.
                break
            if a["start"] < b["end"] and b["start"] < a["end"]:
                pairs.append((a, b))
    return pairs


def _detect_density(events: list[dict]) -> list[list[dict]]:
    """Groups of 3+ events with <15 min buffer between consecutive ones.

    Returns the densely-packed clusters. Single back-to-back pairs
    aren't surfaced — they're normal scheduling. Three or more
    consecutive events with <15 min gaps is the noise the operator
    actually wants to know about.
    """
    timed = [e for e in events if not e["all_day"]]
    if len(timed) < 3:
        return []
    timed.sort(key=lambda e: e["start"])
    threshold = timedelta(minutes=_DENSITY_THRESHOLD_MIN)
    clusters: list[list[dict]] = []
    current: list[dict] = [timed[0]]
    for nxt in timed[1:]:
        gap = nxt["start"] - current[-1]["end"]
        if gap <= threshold:
            current.append(nxt)
        else:
            if len(current) >= 3:
                clusters.append(current)
            current = [nxt]
    if len(current) >= 3:
        clusters.append(current)
    return clusters


# ── Composition ──────────────────────────────────────────────────────────


def _format_horizon_alert(
    conflicts: list[tuple[dict, dict]],
    clusters: list[list[dict]],
) -> str:
    lines = ["📅 72h calendar horizon\n"]
    if conflicts:
        lines.append(f"⚠️ {len(conflicts)} time conflict(s):")
        for a, b in conflicts[:5]:
            a_label = f"{a['start'].astimezone():%a %H:%M}–{a['end'].astimezone():%H:%M}"
            b_label = f"{b['start'].astimezone():%H:%M}–{b['end'].astimezone():%H:%M}"
            lines.append(
                f"  • {a_label} \"{a['summary'][:40]}\" overlaps "
                f"{b_label} \"{b['summary'][:40]}\""
            )
        if len(conflicts) > 5:
            lines.append(f"  …and {len(conflicts) - 5} more")
        lines.append("")
    if clusters:
        lines.append(f"⏱ {len(clusters)} dense cluster(s) (<15 min buffer):")
        for c in clusters[:3]:
            day_label = c[0]["start"].astimezone().strftime("%a")
            duration = c[-1]["end"] - c[0]["start"]
            n = len(c)
            lines.append(
                f"  • {day_label} {c[0]['start'].astimezone():%H:%M} — "
                f"{n} events back-to-back over "
                f"{int(duration.total_seconds() / 60)} min"
            )
        lines.append("")
    return "\n".join(lines).rstrip()


# ── Main ──────────────────────────────────────────────────────────────────


def run() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "ran": False, "events_in_window": 0,
        "conflicts": 0, "dense_clusters": 0, "sent": False,
    }
    if not _enabled() or not background_enabled():
        return summary

    state = read_state_json(_STATE_FILE, {
        "last_run_at": 0.0, "last_sent_date": "",
    })
    now_ts = time.time()
    if now_ts - float(state.get("last_run_at", 0)) < _RUN_CADENCE_S:
        return summary
    state["last_run_at"] = now_ts
    summary["ran"] = True

    now_local = _now_local()
    target_hour = _target_hour()
    today_key = now_local.strftime("%Y-%m-%d")

    # Once-per-day gate. Fires at-or-after target_hour, dedup by date.
    if now_local.hour < target_hour:
        write_state_json(_STATE_FILE, state)
        return summary
    if state.get("last_sent_date") == today_key:
        write_state_json(_STATE_FILE, state)
        return summary

    events = _list_events_72h()
    summary["events_in_window"] = len(events)

    conflicts = _detect_conflicts(events)
    clusters = _detect_density(events)
    summary["conflicts"] = len(conflicts)
    summary["dense_clusters"] = len(clusters)

    if conflicts or clusters:
        body = _format_horizon_alert(conflicts, clusters)
        if body:
            try:
                send_signal_alert(body, tag="calendar_horizon")
                summary["sent"] = True
                state["last_sent_date"] = today_key
            except Exception:
                logger.debug("calendar_horizon: send failed", exc_info=True)
    else:
        # Mark date even on quiet days so we don't re-scan after restart.
        state["last_sent_date"] = today_key

    write_state_json(_STATE_FILE, state)
    audit_event(
        "calendar_horizon_pass",
        events_in_window=summary["events_in_window"],
        conflicts=summary["conflicts"],
        dense_clusters=summary["dense_clusters"],
        sent=summary["sent"],
    )
    return summary
