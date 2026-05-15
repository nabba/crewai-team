"""drill_staleness — proactive monitor for past-due resilience drills.

PROGRAM §44.2 — Q6.2. Bridges the resilience-drill subsystem into the
existing healing-monitor cadence. Alerts when any registered drill
has not had a successful run within ``cadence + grace`` days.

This is the gateway-side awareness of whether the operator is
actually running drills. The scheduler also fires "due" notifications,
but this monitor catches the further-along case where a drill is
significantly past-due — typically because the operator hasn't acted
on the scheduler's prior notifications.

Master switch: ``drill_staleness_monitor_enabled`` (default ON).
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# Monitor metadata — read by the healing-monitor driver to set
# cadence + master-switch gate.
NAME = "drill_staleness"
CADENCE_SECONDS = 86_400          # daily probe
MASTER_SWITCH_KEY = "drill_staleness_monitor_enabled"


def run() -> dict[str, Any]:
    """One monitor probe. Walks the drill registry; reports stale
    drills via Signal alert (rate-limited by the notify arbiter).

    Returns a summary suitable for the monitor driver's logging."""
    summary: dict[str, Any] = {
        "checked": 0,
        "stale": 0,
        "alerts": 0,
        "errors": 0,
    }
    try:
        from app.runtime_settings import get_drill_staleness_monitor_enabled
        if not get_drill_staleness_monitor_enabled():
            summary["skipped"] = True
            return summary
    except Exception:
        # Fall through — default ON behavior.
        pass

    # Force-import drills package so registry is populated.
    try:
        import app.resilience_drills.drills  # noqa: F401
    except Exception:
        logger.debug("drill_staleness: drills package import failed", exc_info=True)
        summary["errors"] = 1
        return summary

    try:
        from app.resilience_drills.audit import days_since_last_success
        from app.resilience_drills.protocol import (
            drill_enabled, get_registry,
        )
    except Exception:
        logger.debug(
            "drill_staleness: resilience_drills import failed",
            exc_info=True,
        )
        summary["errors"] = 1
        return summary

    registry = get_registry()
    stale_names: list[tuple[str, float | None]] = []

    for spec in registry.list_specs():
        summary["checked"] += 1
        if not drill_enabled(spec):
            continue
        days = days_since_last_success(spec.name)
        stale_threshold = spec.cadence_days + spec.grace_days
        # Stale if never run OR if last successful run is past threshold.
        is_stale = (
            days is None
            or days > stale_threshold
        )
        if is_stale:
            summary["stale"] += 1
            stale_names.append((spec.name, days))

    if stale_names:
        try:
            from app.notify import notify
            names_summary = ", ".join(
                f"{n} ({d:.0f}d)" if d is not None else f"{n} (never)"
                for n, d in stale_names[:5]
            )
            notify(
                title="🛡 Resilience drills past-due",
                body=(
                    f"{len(stale_names)} drill"
                    f"{'s' if len(stale_names) != 1 else ''} past their "
                    f"cadence+grace window: {names_summary}. See "
                    f"/cp/drills for details."
                ),
                url="/cp/drills",
                topic="resilience_drill_stale",
                critical=False,
                arbitrate=True,
            )
            summary["alerts"] = 1
        except Exception:
            logger.debug("drill_staleness: notify failed", exc_info=True)

    return summary
