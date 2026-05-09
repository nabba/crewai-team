"""Finland-seasonal proactive nudges (Phase G #4, 2026-05-09).

The operator lives in Helsinki (60°N). The annual rhythm there is
strong enough that several events warrant a once-a-year heads-up:

  * **First-frost watch** (mid-October–early November) — practical
    triggers for outdoor projects, plant-cover, car winter-tyres.
  * **Kaamos onset** (Nov 22 — civil-twilight only at noon) —
    daylight-hour collapse begins; SAD prevention becomes salient.
  * **Winter solstice / shortest day** (Dec 21–22) — ~5 h 50 min of
    daylight in Helsinki; symbolic turning point.
  * **Vappu** (Apr 30–May 1) — Finnish public holiday; commitments
    over the long weekend.
  * **Midsummer / Juhannus** (Saturday between Jun 20–26) — Finnish
    high holiday; two-week absence pattern starts.
  * **Polar-night-ends marker** (mid-January) — daylight returns
    above 6 h; energy returns.

Each trigger fires once per year. State at
``workspace/life_companion/seasonal_nudges.json`` records ``{key:
year}`` so a re-run inside the same year is a no-op.

Only fires when ``temporal_context`` reports the operator's location
is in Finland — operator may travel; the season prompts wouldn't
make sense from Bali.

Cadence: daily.
Master switch: ``SEASONAL_NUDGES_ENABLED`` (default ON).
"""
from __future__ import annotations

import logging
import os
import time
from datetime import date, timedelta
from typing import Any, Optional

from app.life_companion._common import (
    audit_event,
    background_enabled,
    read_state_json,
    send_signal_alert,
    write_state_json,
)

logger = logging.getLogger(__name__)


_STATE_FILE = "seasonal_nudges.json"
_RUN_CADENCE_S = 24 * 3600


def _enabled() -> bool:
    return os.getenv("SEASONAL_NUDGES_ENABLED", "true").lower() in (
        "true", "1", "yes", "on",
    )


def _today() -> date:
    return date.today()


# ── Location gate ─────────────────────────────────────────────────────────


def _operator_in_finland() -> bool:
    """True when ``temporal_context`` reports Finland location.

    Falls through to True when location resolution fails — the
    operator's primary location is Helsinki and most deployments are
    static. Better to fire a nudge in the wrong country occasionally
    than silently never fire.
    """
    try:
        from app.spatial_context import get_location
        loc = get_location()
        country = (loc.get("country") or "").strip().lower()
        if not country:
            return True  # unresolvable → assume default location (Finland)
        return country in ("finland", "fi", "suomi")
    except Exception:
        return True


# ── Trigger calendar ─────────────────────────────────────────────────────


def _midsummer_saturday(year: int) -> date:
    """Finnish Juhannus = Saturday between June 20 and 26 inclusive."""
    for day in range(20, 27):
        d = date(year, 6, day)
        if d.weekday() == 5:  # Saturday
            return d
    return date(year, 6, 24)  # fallback


def _trigger_for_today(today: date) -> Optional[tuple[str, str]]:
    """Return (key, body) if today matches a trigger; else None.

    Each trigger fires on a SPECIFIC date or in a narrow window.
    Window-based triggers (frost-watch, kaamos onset) check across
    a few-day window so a missed run doesn't silently miss the year.
    """
    year = today.year

    # First-frost watch — fires once when entering the Oct-15 → Nov-5 window.
    if date(year, 10, 15) <= today <= date(year, 11, 5):
        return (
            f"first_frost_{year}",
            "❄️ First-frost watch: Helsinki nights now cold enough for "
            "ground frost in the next 2-3 weeks. Reminders worth queuing:\n"
            "  • Winter tyres mounted by mid-November\n"
            "  • Outdoor plumbing / hose drainage\n"
            "  • Plant covers on tender perennials\n"
            "  • Outdoor furniture / grill stowed",
        )

    # Kaamos onset — Nov 22 ± 3 days.
    if date(year, 11, 19) <= today <= date(year, 11, 25):
        return (
            f"kaamos_{year}",
            "🌑 Kaamos approaching: Helsinki daylight drops below 6 h "
            "this week and stays there until late January. Salient "
            "this year:\n"
            "  • Light therapy lamp mornings\n"
            "  • Vitamin D — daily through April\n"
            "  • Calendar lighter on social commitments\n"
            "  • Companion will go gentler on activation cycles",
        )

    # Winter solstice — Dec 21-22.
    if today in (date(year, 12, 21), date(year, 12, 22)):
        return (
            f"solstice_winter_{year}",
            "🌒 Winter solstice today — Helsinki has ~5 h 50 min of "
            "daylight, the shortest of the year. From here on, every "
            "day adds 1-2 minutes. Polar-night ends ~Jan 18.",
        )

    # Polar-night ends marker — Jan 18 ± 2 days.
    if date(year, 1, 16) <= today <= date(year, 1, 20):
        return (
            f"polar_night_ends_{year}",
            "🌅 Polar-night ends: Helsinki daylight crosses 6.5 h again "
            "this week. Energy returns over the next 4-6 weeks; pacing "
            "demanding work toward Feb-Mar is realistic again.",
        )

    # Vappu — April 30 → May 1.
    if today in (date(year, 4, 30), date(year, 5, 1)):
        return (
            f"vappu_{year}",
            "🎓 Vappu: Finnish workers' day + spring carnival. Most of "
            "your network offline today + tomorrow. Schedule any "
            "deliverables around it.",
        )

    # Midsummer (Juhannus) — the Saturday between Jun 20-26 ± 2 days
    # of warning so the operator has time to plan.
    juhannus = _midsummer_saturday(year)
    warn_start = juhannus - timedelta(days=10)
    if warn_start <= today <= juhannus - timedelta(days=8):
        return (
            f"juhannus_warn_{year}",
            f"🌞 Midsummer ({juhannus:%A %B %d}) is in ~10 days. "
            "Two-week absence pattern starts in Finland for many of "
            "your contacts. Confirm any deliverables landing in late "
            "June + first week of July before they go dark.",
        )

    return None


# ── Main ──────────────────────────────────────────────────────────────────


def run() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "ran": False, "trigger_key": "", "sent": False,
    }
    if not _enabled() or not background_enabled():
        return summary
    if not _operator_in_finland():
        return summary

    state = read_state_json(_STATE_FILE, {
        "last_run_at": 0.0, "fired": {},
    })
    now_ts = time.time()
    if now_ts - float(state.get("last_run_at", 0)) < _RUN_CADENCE_S:
        return summary
    state["last_run_at"] = now_ts
    summary["ran"] = True

    today = _today()
    fired: dict = state.setdefault("fired", {})
    trigger = _trigger_for_today(today)
    if trigger is None:
        write_state_json(_STATE_FILE, state)
        return summary

    key, body = trigger
    summary["trigger_key"] = key
    if key in fired:
        write_state_json(_STATE_FILE, state)
        return summary

    try:
        send_signal_alert(body, tag=f"seasonal:{key}")
        summary["sent"] = True
        fired[key] = today.isoformat()
    except Exception:
        logger.debug("seasonal_nudges: send failed", exc_info=True)

    write_state_json(_STATE_FILE, state)
    audit_event(
        "seasonal_nudges_pass",
        trigger_key=key,
        sent=summary["sent"],
    )
    return summary
