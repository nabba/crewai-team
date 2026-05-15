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

Q6.4 P1#5 — boot-grace. On a fresh deploy, every drill is "never
run" so without grace the monitor would fire 4 alerts on first
probe. The grace period suppresses alerts for the first
``_BOOT_GRACE_DAYS`` after the resilience subsystem first appeared
(detected via the audit file's mtime; absent file = brand-new
install).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Monitor metadata — read by the healing-monitor driver to set
# cadence + master-switch gate.
NAME = "drill_staleness"
CADENCE_SECONDS = 86_400          # daily probe
MASTER_SWITCH_KEY = "drill_staleness_monitor_enabled"

# Q6.4 — fresh-install grace window. On a brand-new deploy the
# operator needs time to schedule first runs; staleness alerts in
# the first week would be noise.
_BOOT_GRACE_DAYS = 7


def _in_boot_grace_window() -> bool:
    """True when the resilience-drill subsystem is in its initial
    grace window. Heuristic: if the audit file doesn't exist yet OR
    is younger than ``_BOOT_GRACE_DAYS``, we're brand new.

    Once the audit file ages past the grace window, this returns
    False permanently — alerts resume."""
    try:
        from app.resilience_drills.audit import _default_audit_path
        path = _default_audit_path()
    except Exception:
        # If we can't even import the audit module, we definitely
        # don't want noisy alerts — keep grace on.
        return True
    if not path.exists():
        # No audit history at all → brand new install → grace.
        return True
    try:
        age_s = time.time() - path.stat().st_mtime
        return age_s < (_BOOT_GRACE_DAYS * 86_400)
    except OSError:
        return True


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

    # Q6.4 P1#5 — fresh-install grace window. Suppress staleness
    # alerts for the first 7 days after the audit file first appears
    # (or while it doesn't exist at all).
    if _in_boot_grace_window():
        summary["skipped"] = True
        summary["reason"] = "boot_grace"
        return summary

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
