"""Idle-job entry point for browser-history ingestion.

LIGHT-cadence job. Internally gates at 30-minute cadence via a marker
file so the underlying SQLite reads don't run on every idle tick;
matches the cadence-guarded pattern used by ``app.health.idle_job``.

Disabled by default — ``BROWSE_INGESTION_ENABLED=false``. The gate
short-circuits before any disk I/O.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from app.browse import store

logger = logging.getLogger(__name__)


_MIN_INTERVAL_S = 30 * 60  # 30 minutes


def _last_run_path() -> Path:
    return store.resolve_base() / ".last_pass_at"


def _due() -> bool:
    p = _last_run_path()
    if not p.exists():
        return True
    try:
        age = time.time() - p.stat().st_mtime
    except OSError:
        return True
    return age >= _MIN_INTERVAL_S


def _touch() -> None:
    p = _last_run_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            datetime.now(timezone.utc).isoformat(), encoding="utf-8",
        )
    except OSError:
        logger.debug("browse idle: marker touch failed", exc_info=True)


def run_browse_tick() -> None:
    """One LIGHT idle pass. Failure-isolated; no-op when disabled or
    not yet due."""
    if not store.enabled():
        return
    if not _due():
        return
    try:
        from app.browse.aggregator import run_one_pass
    except Exception:
        logger.debug("browse idle: aggregator import failed", exc_info=True)
        return
    try:
        result = run_one_pass()
    except Exception:
        logger.debug("browse idle: pass raised", exc_info=True)
        return
    _touch()
    if result.status != "ok":
        return
    if result.total_events == 0 and not result.errors:
        return
    logger.info(
        "browse: events=%d written=%d skipped_blocklisted=%d errors=%d",
        result.total_events,
        result.written,
        result.total_skipped_blocklisted,
        len(result.errors),
    )
    for browser, profile, err in result.errors:
        logger.info("browse error: %s/%s — %s", browser, profile, err)


def get_idle_jobs() -> list[tuple[str, Callable[[], None], str]]:
    """One job tuple — ``browse-tick`` at LIGHT weight."""
    from app.idle_scheduler import JobWeight
    return [("browse-tick", run_browse_tick, JobWeight.LIGHT)]
