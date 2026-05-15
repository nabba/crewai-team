"""Calendar-invite action handler — PROGRAM §46.8 (Q9.5).

The agent proposes a Google Calendar event; the operator approves;
this handler creates it via the existing
:mod:`app.tools.gcal_tools` surface backed by
:mod:`app.google_workspace`.

Data payload shape::

    {
        "summary":      "Coffee with Ave",
        "start_iso":    "2026-06-01T10:00:00+02:00",
        "end_iso":      "2026-06-01T10:30:00+02:00",
        "location":     "Helsinki",            # optional
        "description":  "Bring the PIM notes", # optional
        "attendees":    ["a@x", "b@x"],        # optional
        "calendar_id":  "primary",             # optional, default "primary"
        "all_day":      false,                  # optional, default False
        "timezone":     "Europe/Helsinki",      # optional, default UTC
    }

Failure modes (validate):
  * Missing/empty summary / start_iso / end_iso → INVALID before gate.
  * Non-ISO timestamps → INVALID.
  * end <= start → INVALID.
  * Any attendee fails RFC-5322 sanity check → INVALID.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any

from app.action_requests.handlers.base import ActionHandler, ApplyResult
from app.action_requests.models import ActionType

logger = logging.getLogger(__name__)


_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
_MAX_SUMMARY_CHARS = 300
_MAX_DESCRIPTION_CHARS = 8000


def _parse_iso(s: Any) -> datetime | None:
    if not isinstance(s, str) or not s.strip():
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


class CalendarInviteHandler(ActionHandler):
    @property
    def action_type(self):
        return ActionType.CALENDAR_INVITE

    def validate(self, data: dict[str, Any]) -> tuple[bool, str | None]:
        summary = data.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            return False, "summary is required and non-empty"
        if len(summary) > _MAX_SUMMARY_CHARS:
            return False, f"summary exceeds {_MAX_SUMMARY_CHARS} chars"

        start = _parse_iso(data.get("start_iso"))
        end = _parse_iso(data.get("end_iso"))
        if start is None:
            return False, "start_iso must be ISO 8601 datetime"
        if end is None:
            return False, "end_iso must be ISO 8601 datetime"
        if end <= start:
            return False, "end_iso must be strictly after start_iso"

        description = data.get("description")
        if description is not None and not isinstance(description, str):
            return False, "description must be a string or omitted"
        if isinstance(description, str) and len(description) > _MAX_DESCRIPTION_CHARS:
            return False, f"description exceeds {_MAX_DESCRIPTION_CHARS} chars"

        attendees = data.get("attendees")
        if attendees is not None:
            if not isinstance(attendees, list):
                return False, "attendees must be a list or omitted"
            for a in attendees:
                if not isinstance(a, str) or not _EMAIL_RE.match(a):
                    return False, f"invalid attendee email: {a!r}"

        for opt in ("location", "calendar_id", "timezone"):
            v = data.get(opt)
            if v is not None and not isinstance(v, str):
                return False, f"{opt} must be a string or omitted"

        all_day = data.get("all_day", False)
        if not isinstance(all_day, bool):
            return False, "all_day must be a boolean"

        return True, None

    def apply(self, data: dict[str, Any]) -> ApplyResult:
        try:
            from app.google_workspace import get_service
        except Exception as exc:  # noqa: BLE001
            return ApplyResult(
                ok=False, error=f"google_workspace import failed: {exc}",
            )
        try:
            service = get_service("calendar")
        except Exception as exc:  # noqa: BLE001
            return ApplyResult(
                ok=False, error=f"calendar service unavailable: {exc}",
            )
        if service is None:
            return ApplyResult(
                ok=False,
                error=(
                    "calendar service is None — run "
                    "`python -m app.google_workspace.bootstrap` first"
                ),
            )

        event_body: dict[str, Any] = {
            "summary": data["summary"],
        }
        if data.get("location"):
            event_body["location"] = data["location"]
        if data.get("description"):
            event_body["description"] = data["description"]

        tz = data.get("timezone") or "UTC"
        if data.get("all_day"):
            # Google expects date-only strings for all-day events
            start_d = _parse_iso(data["start_iso"]).date().isoformat()
            end_d = _parse_iso(data["end_iso"]).date().isoformat()
            event_body["start"] = {"date": start_d}
            event_body["end"] = {"date": end_d}
        else:
            event_body["start"] = {
                "dateTime": data["start_iso"], "timeZone": tz,
            }
            event_body["end"] = {
                "dateTime": data["end_iso"], "timeZone": tz,
            }

        attendees = data.get("attendees") or []
        if attendees:
            event_body["attendees"] = [{"email": a} for a in attendees]

        calendar_id = data.get("calendar_id") or "primary"
        try:
            created = service.events().insert(
                calendarId=calendar_id,
                body=event_body,
                sendUpdates="all" if attendees else "none",
            ).execute()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "calendar_invite: events.insert raised: %s",
                exc, exc_info=True,
            )
            return ApplyResult(ok=False, error=f"insert raised: {exc}")

        return ApplyResult(
            ok=True,
            artifact={
                "event_id": created.get("id"),
                "html_link": created.get("htmlLink"),
                "calendar_id": calendar_id,
                "attendee_count": len(attendees),
            },
        )

    def render_summary(self, data: dict[str, Any]) -> str:
        summary = (data.get("summary") or "")[:80]
        start = data.get("start_iso", "")[:16]
        loc = data.get("location") or ""
        loc_suf = f" @ {loc[:40]}" if loc else ""
        n_attendees = len(data.get("attendees") or [])
        people_suf = f" — {n_attendees} attendee(s)" if n_attendees else ""
        return f"📅 {summary} on {start}{loc_suf}{people_suf}"
