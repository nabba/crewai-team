"""Idle-job entry point for daily health summary + anomaly detection.

Computes a 7-day :class:`HealthSummary` and walks anomaly detection
once per day. Outputs are observational — they're written to the
companion stream, never auto-routed.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


def _last_run_path() -> Path:
    """Marker file path. Derived from the same base directory as the
    JSONL store, so :func:`app.health.store._reset_for_tests` and the
    ``HEALTH_BASE_DIR`` env override apply uniformly."""
    from app.health.store import resolve_base
    return resolve_base() / ".last_summary_at"


def _enabled() -> bool:
    return os.getenv("HEALTH_INGESTION_ENABLED", "false").lower() in (
        "true", "1", "yes", "on",
    )


def _due(min_interval_h: float = 23.5) -> bool:
    """True if it's been long enough since the last summary."""
    p = _last_run_path()
    if not p.exists():
        return True
    try:
        age = time.time() - p.stat().st_mtime
    except OSError:
        return True
    return age >= min_interval_h * 3600.0


def _touch_last_run() -> None:
    p = _last_run_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            datetime.now(timezone.utc).isoformat(), encoding="utf-8",
        )
    except OSError:
        logger.debug("health idle: could not touch last_run marker", exc_info=True)


def run_health_summary_tick() -> None:
    """One daily tick. Computes summary + anomalies; logs structured
    output the daily-briefing composer reads.

    No-op when health ingestion is disabled or the daily window has
    already been logged.
    """
    if not _enabled():
        return
    if not _due():
        return
    try:
        from app.health.anomaly import detect_anomalies
        from app.health.summary import summarise_window
    except Exception:
        logger.debug("health idle: imports failed", exc_info=True)
        return
    try:
        summary = summarise_window(days=7)
    except Exception:
        logger.debug("health idle: summarise_window raised", exc_info=True)
        return
    if not summary.record_counts or all(
        v == 0 for v in summary.record_counts.values()
    ):
        # No data yet; the user hasn't imported anything.
        return
    try:
        anomalies = detect_anomalies()
    except Exception:
        logger.debug("health idle: detect_anomalies raised", exc_info=True)
        anomalies = []
    logger.info(
        "health summary: 7d steps/day=%.0f sleep_hours/night=%s "
        "active_kcal/day=%.0f anomalies=%d",
        summary.steps_per_day_mean,
        f"{summary.sleep_hours_per_night_mean:.1f}"
            if summary.sleep_hours_per_night_mean is not None else "n/a",
        summary.active_kcal_per_day_mean,
        len(anomalies),
    )
    for a in anomalies:
        logger.info("health anomaly: %s", a.description)
    _touch_last_run()


def get_idle_jobs() -> list[tuple[str, Callable[[], None], str]]:
    """One job tuple — ``health-summary`` at LIGHT weight."""
    from app.idle_scheduler import JobWeight
    return [("health-summary", run_health_summary_tick, JobWeight.LIGHT)]
