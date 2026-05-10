"""Life-companion subsystem — proactive features that watch your life,
not your code:

  * ``email_monitor``        — proactive triage of unread inbox.
  * ``daily_briefing``       — morning / evening / weekly digest.
  * ``routine_detector``     — DOW + time-of-day pattern surfacing.
  * ``long_arc_follow_up``   — multi-week commitment check-ins
                               (Phase B #4, 2026-05-09).
  * ``calendar_prep``        — per-event 30-min pre-meeting briefing
                               (Phase B #2, 2026-05-09).

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
    """Return the life-companion jobs as
    ``(name, fn, JobWeight)`` tuples for the idle scheduler.

    All are LIGHT — they cadence-check internally (≥10 min for
    email, scheduled-window for briefing, ≥20 h for routine detection,
    ≥6 h for long-arc, 5 min for calendar prep) so a chatty idle
    scheduler doesn't cause expensive work.
    """
    from app.idle_scheduler import JobWeight
    from app.life_companion import (
        daily_briefing,
        email_monitor,
        routine_detector,
    )
    jobs: list[tuple[str, Callable[[], None], Any]] = [
        ("life-companion-email", email_monitor.run, JobWeight.LIGHT),
        ("life-companion-briefing", daily_briefing.run, JobWeight.LIGHT),
        ("life-companion-routines", routine_detector.run, JobWeight.LIGHT),
    ]
    # Long-arc + calendar-prep are best-effort: any import failure
    # (e.g. kernel persistence broken, Google libs missing on a slim
    # deploy) must not block the other life-companion jobs.
    try:
        from app.life_companion import long_arc_follow_up
        jobs.append(
            ("life-companion-long-arc", long_arc_follow_up.run, JobWeight.LIGHT),
        )
    except Exception:
        pass
    try:
        from app.life_companion import calendar_prep
        jobs.append(
            ("life-companion-calendar-prep", calendar_prep.run, JobWeight.LIGHT),
        )
    except Exception:
        pass
    # Phase D #5 (2026-05-09) — weekly personalized digest.
    try:
        from app.life_companion import personalized_digest
        jobs.append(
            ("life-companion-personalized-digest",
             personalized_digest.run, JobWeight.LIGHT),
        )
    except Exception:
        pass
    # Phase G #1 (2026-05-09) — daily 72h calendar horizon scan.
    try:
        from app.life_companion import calendar_horizon
        jobs.append(
            ("life-companion-calendar-horizon",
             calendar_horizon.run, JobWeight.LIGHT),
        )
    except Exception:
        pass
    # Phase G #3 (2026-05-09) — topic-dormancy long-arc nudge.
    try:
        from app.life_companion import topic_dormancy
        jobs.append(
            ("life-companion-topic-dormancy",
             topic_dormancy.run, JobWeight.LIGHT),
        )
    except Exception:
        pass
    # Phase G #4 (2026-05-09) — Finland-seasonal nudges.
    try:
        from app.life_companion import seasonal_nudges
        jobs.append(
            ("life-companion-seasonal-nudges",
             seasonal_nudges.run, JobWeight.LIGHT),
        )
    except Exception:
        pass
    # 2026-05-10 — act-now email digest.  LLM-graded synthesis of
    # last 48h unread inbox; fires every 3h between 07:00–22:00
    # local; surfaces top-7 act-now items with why / action /
    # Gmail-link.  Sibling of email_monitor (which is heuristic +
    # real-time); the two co-exist deliberately.
    try:
        from app.life_companion import act_now_digest
        jobs.append(
            ("life-companion-act-now-digest",
             act_now_digest.run, JobWeight.LIGHT),
        )
    except Exception:
        pass
    return jobs
