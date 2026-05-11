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
    """One idle-tick of the inbox watcher. Failure-isolated.

    Notifies the user when something needs attention:

      * Successful processing of an Apple Health export — the user
        wants to see "your last sync landed."
      * Failures or unrecognised files — silent failure was the
        explicit anti-pattern §31's forest-age work cured. Tell the
        user why a file didn't process so they can rename / drop a
        different format / file feature work.

    Routine successes for already-supported kinds (e.g. text drop) are
    logged but not pushed — push spam is worse than no push.
    """
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
    if not (result.processed or result.failed or result.skipped_unknown):
        return
    logger.info(
        "inbox: processed=%d failed=%d unknown=%d deferred=%d dedup=%d",
        len(result.processed), len(result.failed),
        len(result.skipped_unknown), len(result.deferred),
        len(result.skipped_dedup),
    )
    _maybe_notify(result)


def _maybe_notify(result) -> None:
    """Fire a single ``notify`` ping summarising notable events."""
    notable_processed = [
        p for p in result.processed
        if p.get("kind") == "apple_health_export"
    ]
    if not (notable_processed or result.failed or result.skipped_unknown):
        return
    try:
        from app.notify import notify
    except Exception:
        logger.debug("inbox scheduler: notify import failed", exc_info=True)
        return

    body_lines: list[str] = []
    for p in notable_processed:
        body_lines.append(f"✓ {p['name']}: {p['outcome']}")
    for f in result.failed:
        body_lines.append(
            f"✗ {f['name']} ({f.get('kind', 'unknown')}): {f['reason']}"
        )
    for name in result.skipped_unknown:
        body_lines.append(f"? {name}: unrecognised file type")

    title = "Inbox"
    if result.failed or result.skipped_unknown:
        title = f"Inbox — {len(result.failed) + len(result.skipped_unknown)} need attention"

    try:
        # Q4.1 (PROGRAM §41.4) — inbox events arbitrate when they're
        # purely informational (successful imports, unrecognised file
        # types). Failures are operator-actionable and bypass via
        # critical=True so they always reach Signal.
        has_failures = bool(result.failed)
        notify(
            title,
            "\n".join(body_lines[:10]),  # cap to keep the push readable
            url="/cp/files",
            arbitrate=not has_failures,
            topic="inbox",
            critical=has_failures,
        )
    except Exception:
        logger.debug("inbox scheduler: notify failed", exc_info=True)


def get_idle_jobs() -> list[tuple[str, Callable[[], None], str]]:
    """One job tuple — ``inbox-tick`` at LIGHT weight."""
    from app.idle_scheduler import JobWeight
    return [("inbox-tick", run_inbox_tick, JobWeight.LIGHT)]
