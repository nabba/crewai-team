"""Life-companion subsystem — proactive features that watch your life,
not your code:

  * ``email_monitor``    — proactive triage of unread inbox.
  * ``daily_briefing``   — morning / evening / weekly digest.
  * ``routine_detector`` — DOW + time-of-day pattern surfacing.

Wired into the existing idle scheduler via
``app.companion.loop.get_idle_jobs()`` — each module exposes a ``run()``
function that's idempotent, cadence-aware, and respects the
``LIFE_COMPANION_ENABLED`` master switch + the global
``idle_scheduler.is_enabled()`` kill switch.

Distinct from ``app.healing`` (system-health observability) and
``app.companion`` (per-workspace ideation). Lives at the user-life
abstraction layer.
"""
from __future__ import annotations

from typing import Any, Callable


def get_idle_jobs() -> list[tuple[str, Callable[[], None], Any]]:
    """Return the three life-companion jobs as
    ``(name, fn, JobWeight)`` tuples for the idle scheduler.

    All three are LIGHT — they cadence-check internally (≥10 min for
    email, scheduled-window for briefing, ≥20 h for routine detection)
    so a chatty idle scheduler doesn't cause expensive work.
    """
    from app.idle_scheduler import JobWeight
    from app.life_companion import (
        daily_briefing,
        email_monitor,
        routine_detector,
    )
    return [
        ("life-companion-email", email_monitor.run, JobWeight.LIGHT),
        ("life-companion-briefing", daily_briefing.run, JobWeight.LIGHT),
        ("life-companion-routines", routine_detector.run, JobWeight.LIGHT),
    ]
