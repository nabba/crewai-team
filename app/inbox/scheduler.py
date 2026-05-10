"""Idle-job entry point for the inbox watcher.

Registered with ``app.companion.loop.get_idle_jobs`` as a LIGHT job.
The job is cadence-resilient: enabled or not, it returns quickly
when there's nothing to do.
"""
from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)


def run_inbox_tick() -> None:
    """One idle-tick of the inbox watcher. Failure-isolated."""
    try:
        from app.inbox.router import scan_and_route
    except Exception:
        logger.debug("inbox scheduler: router import failed", exc_info=True)
        return
    try:
        result = scan_and_route()
    except Exception:
        logger.debug("inbox scheduler: scan_and_route raised", exc_info=True)
        return
    if result.status != "ok":
        return
    if result.processed or result.failed or result.skipped_unknown:
        logger.info(
            "inbox: processed=%d failed=%d unknown=%d deferred=%d dedup=%d",
            len(result.processed), len(result.failed),
            len(result.skipped_unknown), len(result.deferred),
            len(result.skipped_dedup),
        )


def get_idle_jobs() -> list[tuple[str, Callable[[], None], str]]:
    """One job tuple — ``inbox-tick`` at LIGHT weight."""
    from app.idle_scheduler import JobWeight
    return [("inbox-tick", run_inbox_tick, JobWeight.LIGHT)]
