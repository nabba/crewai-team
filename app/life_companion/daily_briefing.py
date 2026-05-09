"""Daily briefing — proactive Signal digest at fixed local times.

Three flavours, all opt-out per env var:

  * **morning** (default 07:00) — calendar (next 24 h) + top-3 urgent
    unread + open project tickets + companion ideas.
  * **evening** (default 18:00) — wrap of completed events / unhandled
    flagged mail / open ticket reminders.
  * **weekly** (default Mon 09:00) — last week's highlights + this
    week's calendar density + workspace activity.

Cadence guards inside ``run()`` ensure each flavour fires at most once
per scheduled window per local day. State at
``workspace/life_companion/daily_briefing.json``::

    {
      "last_morning_at": "2026-05-09",
      "last_evening_at": "2026-05-09",
      "last_weekly_at": "2026-W19",
    }

All data sources fail soft: a missing Calendar token, an empty inbox,
or a down ticket DB just gets a "(none)" line in the digest.
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
    feature_enabled,
    read_state_json,
    send_signal_alert,
    user_email_address,
    write_state_json,
)

logger = logging.getLogger(__name__)

_STATE_FILE = "daily_briefing.json"

# Default local-clock windows. The cadence check in run() picks the
# flavour whose window the wall clock is in (with a ±15 min tolerance).
_TOLERANCE_MIN = 15


def _parse_hhmm(value: str, default: tuple[int, int]) -> tuple[int, int]:
    try:
        h, m = value.split(":")
        return (int(h), int(m))
    except Exception:
        return default


def _morning_time() -> tuple[int, int]:
    return _parse_hhmm(os.getenv("LIFE_COMPANION_BRIEFING_MORNING", "07:00"), (7, 0))


def _evening_time() -> tuple[int, int]:
    return _parse_hhmm(os.getenv("LIFE_COMPANION_BRIEFING_EVENING", "18:00"), (18, 0))


def _weekly_dow() -> int:
    """0=Mon ... 6=Sun"""
    table = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4, "SAT": 5, "SUN": 6}
    raw = os.getenv("LIFE_COMPANION_BRIEFING_WEEKLY_DOW", "MON").upper().strip()
    return table.get(raw, 0)


def _weekly_time() -> tuple[int, int]:
    return _parse_hhmm(os.getenv("LIFE_COMPANION_BRIEFING_WEEKLY_TIME", "09:00"), (9, 0))


def _now_local() -> datetime:
    """Local-clock datetime. The system already runs in the operator's
    timezone; ``datetime.now()`` (no tzinfo) is the right call.
    """
    return datetime.now()


def _within_window(now: datetime, target_h: int, target_m: int) -> bool:
    """Within ±_TOLERANCE_MIN of the target HH:MM."""
    target = now.replace(hour=target_h, minute=target_m, second=0, microsecond=0)
    delta = abs((now - target).total_seconds()) / 60
    return delta <= _TOLERANCE_MIN


def _which_flavour(now: datetime) -> str | None:
    """Decide which (if any) flavour should fire right now."""
    # Weekly takes priority when it's the configured day + window.
    if now.weekday() == _weekly_dow() and _within_window(now, *_weekly_time()):
        return "weekly"
    if _within_window(now, *_morning_time()):
        return "morning"
    if _within_window(now, *_evening_time()):
        return "evening"
    return None


# ── Data collectors (each fail-soft) ──────────────────────────────────────


def _gather_calendar_24h() -> list[str]:
    """Lines for the next 24 h of calendar events. Empty list on failure."""
    try:
        from app.tools.gcal_tools import _list_events
    except Exception:
        return []
    try:
        now = datetime.now(timezone.utc)
        events = _list_events(
            max_results=15,
            time_min=now.isoformat().replace("+00:00", "Z"),
            time_max=(now + timedelta(hours=24)).isoformat().replace("+00:00", "Z"),
        ) or []
    except Exception:
        return []

    lines = []
    for ev in events[:10]:
        start = ev.get("start") or ""
        summary = ev.get("summary") or "(untitled)"
        loc = ev.get("location") or ""
        loc_part = f" @ {loc[:30]}" if loc else ""
        # Strip date component when it's the same day as now (already implicit).
        try:
            t = start.split("T", 1)[1][:5] if "T" in start else start[:10]
        except Exception:
            t = start[:16]
        lines.append(f"  • {t} — {summary[:60]}{loc_part}")
    return lines


def _gather_top_emails(n: int = 3) -> list[str]:
    """Top-N urgent unread bullets. Empty on failure."""
    try:
        from app.tools.gmail_tools import _list_recent
        from app.tools.email_importance import EmailHeaders, score_email
    except Exception:
        return []
    try:
        stubs = _list_recent(limit=20, query="in:inbox is:unread") or []
    except Exception:
        return []
    if not stubs:
        return []

    user_addr = user_email_address()
    important_senders_raw = os.getenv("EMAIL_IMPORTANT_SENDERS", "")
    senders = tuple(
        p.strip().lower() for p in important_senders_raw.split(",") if p.strip()
    )

    scored = []
    for stub in stubs:
        h = EmailHeaders(
            from_=stub.get("from", ""),
            subject=stub.get("subject", ""),
            unread=True,
        )
        try:
            r = score_email(h, user_address=user_addr, important_senders=senders)
            scored.append((r.score, stub))
        except Exception:
            continue

    scored.sort(key=lambda x: x[0], reverse=True)
    lines = []
    for score, stub in scored[:n]:
        sender = (stub.get("from") or "(unknown)")[:50]
        subj = (stub.get("subject") or "(no subject)")[:60]
        lines.append(f"  • [{score:.1f}] {sender}: {subj}")
    return lines


def _gather_open_tickets(n: int = 5) -> list[str]:
    """Open tickets across the active venture. Soft fail."""
    try:
        from app.control_plane import db as cp_db
    except Exception:
        return []
    try:
        rows = cp_db.execute(
            """
            SELECT title, status, project_id
              FROM control_plane.tickets
             WHERE status NOT IN ('done', 'cancelled', 'archived')
             ORDER BY updated_at DESC
             LIMIT %s
            """,
            (n,),
            fetch=True,
        ) or []
    except Exception:
        return []

    lines = []
    for row in rows:
        title = (row.get("title") or "")[:60]
        status = row.get("status") or ""
        project = row.get("project_id") or ""
        lines.append(f"  • [{status}] {title}  ({project})")
    return lines


def _gather_top_interests(n: int = 5) -> list[str]:
    """Top-N topics from the interest_model profile (Phase F #6).

    Empty list when the profile hasn't been generated yet — not an
    error, just means interest_model hasn't run.
    """
    try:
        from app.companion.interest_model import current_profile
    except Exception:
        return []
    try:
        profile = current_profile()
    except Exception:
        return []
    topics = profile.get("topics") or []
    out: list[str] = []
    for t in topics[:n]:
        if not isinstance(t, dict):
            continue
        name = (t.get("name") or "").strip()
        score = t.get("score")
        if name and score is not None:
            out.append(f"  • {name} ({score:.2f})")
    return out


def _gather_companion_surfaced() -> list[str]:
    """Recent companion ideas surfaced to the user (last 24 h). Soft fail."""
    try:
        from app.companion import idea_store as _idea_store
    except Exception:
        return []
    try:
        # The idea_store API varies by version; fall back gracefully if a
        # ``recent_surfaced`` accessor isn't available.
        if hasattr(_idea_store, "recent_surfaced"):
            ideas = _idea_store.recent_surfaced(hours=24) or []
        elif hasattr(_idea_store, "list_recent"):
            ideas = _idea_store.list_recent(hours=24) or []
        else:
            return []
    except Exception:
        return []

    lines = []
    for idea in ideas[:5]:
        if isinstance(idea, dict):
            txt = (idea.get("text") or idea.get("title") or "")[:80]
            ws = idea.get("workspace_id") or ""
            lines.append(f"  • {txt}  ({ws})")
    return lines


# ── Compose ──────────────────────────────────────────────────────────────


def _compose_morning() -> str:
    cal = _gather_calendar_24h()
    mail = _gather_top_emails(n=3)
    tickets = _gather_open_tickets(n=5)

    parts = ["☀️  Morning briefing\n"]
    parts.append("📅 Today's events:")
    parts.extend(cal or ["  • (none scheduled)"])
    parts.append("\n📬 Urgent unread:")
    parts.extend(mail or ["  • (inbox clean)"])
    parts.append("\n🎯 Open tickets:")
    parts.extend(tickets or ["  • (no open tickets)"])
    return "\n".join(parts)


def _compose_evening() -> str:
    cal = _gather_calendar_24h()  # also covers tonight + tomorrow morning
    mail = _gather_top_emails(n=3)
    surfaced = _gather_companion_surfaced()

    parts = ["🌙 Evening wrap\n"]
    parts.append("📅 Tomorrow:")
    parts.extend(cal or ["  • (no events)"])
    parts.append("\n📬 Still flagged:")
    parts.extend(mail or ["  • (inbox clean)"])
    parts.append("\n💡 Companion surfaced today:")
    parts.extend(surfaced or ["  • (no surfaced ideas)"])
    return "\n".join(parts)


def _compose_weekly() -> str:
    cal = _gather_calendar_24h()
    tickets = _gather_open_tickets(n=8)
    surfaced = _gather_companion_surfaced()
    interests = _gather_top_interests(n=5)

    parts = ["🗓 Weekly review\n"]
    parts.append("📅 Next 24h:")
    parts.extend(cal or ["  • (no events)"])
    parts.append("\n🎯 Open tickets:")
    parts.extend(tickets or ["  • (no open tickets)"])
    parts.append("\n💡 Companion surfaced last week:")
    parts.extend(surfaced or ["  • (none)"])
    if interests:
        # Phase F #6: only surface interests in the weekly digest —
        # daily morning/evening cadences don't need this noise.
        parts.append("\n🧭 Topics you've cared about:")
        parts.extend(interests)
    return "\n".join(parts)


# ── Cadence-aware entry point ─────────────────────────────────────────────


def _key_for(flavour: str, now: datetime) -> str:
    """Idempotency key. Daily for morning/evening; weekly for weekly."""
    if flavour == "weekly":
        # ISO week token like "2026-W19" — same week shares the key.
        iso = now.isocalendar()
        return f"{iso[0]}-W{iso[1]:02d}"
    return now.strftime("%Y-%m-%d")


def _last_key(state: dict, flavour: str) -> str:
    return state.get(f"last_{flavour}_at", "") or ""


def _set_last_key(state: dict, flavour: str, key: str) -> None:
    state[f"last_{flavour}_at"] = key


_COMPOSERS = {
    "morning": _compose_morning,
    "evening": _compose_evening,
    "weekly": _compose_weekly,
}


def run() -> None:
    """Cadence-aware tick. Idempotent within each scheduled window."""
    if not feature_enabled("briefing"):
        return
    if not background_enabled():
        return

    now = _now_local()
    flavour = _which_flavour(now)
    if flavour is None:
        return

    state = read_state_json(_STATE_FILE, {})
    key = _key_for(flavour, now)
    if _last_key(state, flavour) == key:
        return  # already sent this window

    composer = _COMPOSERS.get(flavour)
    if composer is None:
        return

    try:
        body = composer()
    except Exception:
        logger.debug("daily_briefing: composer %s raised", flavour, exc_info=True)
        return

    audit_event(
        "daily_briefing_send",
        flavour=flavour,
        key=key,
        body_chars=len(body),
    )
    sent = send_signal_alert(body, tag=f"daily_briefing_{flavour}")
    if sent:
        _set_last_key(state, flavour, key)
        write_state_json(_STATE_FILE, state)
