"""Monthly VACUUM for ``workspace/conversations.db``.

Wave 0/1 closure (#A6, 2026-05-09). The existing
``prune_old_inbound`` / ``prune_old_outbound`` helpers delete rows on
the inbound/outbound queues, but SQLite doesn't release the freed
pages back to the filesystem without an explicit VACUUM. Over years
that accumulates as latent disk usage even though the row count stays
bounded.

Cadence: ~30 days (internal guard). The daemon driver pings us daily;
the cadence guard keeps real work to once per month.
"""
from __future__ import annotations

import logging
import time

from app.life_companion._common import (
    audit_event,
    background_enabled,
    read_state_json,
    send_signal_alert,
    write_state_json,
)

logger = logging.getLogger(__name__)

_STATE_FILE = "db_vacuum.json"
_RUN_CADENCE_S = 30 * 24 * 3600
_BIG_FREE_THRESHOLD_BYTES = 50 * 1024 * 1024   # alert when we free >50 MB

# Discipline (post-2026-05-16): consecutive-failure tracking so a
# chronically-locked conversations.db surfaces as an alert instead of
# silently never reclaiming. conversations.db has a single writer (the
# Signal client), so lock contention is rare — but if it happens
# consistently it means the Signal client is holding the connection
# open through the VACUUM probe window, which is a real bug worth
# surfacing.
_FAILURE_ALERT_THRESHOLD = 4   # alert after this many consecutive misses


def run() -> None:
    """One pass — cadence-guarded internally."""
    if not background_enabled():
        return

    state = read_state_json(_STATE_FILE, {"last_run_at": 0.0})
    now = time.time()
    if now - float(state.get("last_run_at", 0)) < _RUN_CADENCE_S:
        return
    state["last_run_at"] = now

    try:
        from app.conversation_store import vacuum
    except Exception:
        logger.debug("db_vacuum: conversation_store import failed", exc_info=True)
        write_state_json(_STATE_FILE, state)
        return

    try:
        summary = vacuum()
    except Exception:
        logger.debug("db_vacuum: vacuum() raised", exc_info=True)
        write_state_json(_STATE_FILE, state)
        return

    # Consecutive-failure tracking. State shape:
    #   state["consecutive_failures"]: int
    prev_failures = int(state.get("consecutive_failures", 0) or 0)
    if summary.get("ok"):
        if prev_failures >= _FAILURE_ALERT_THRESHOLD:
            # Recovery transition — log but don't alert.
            logger.info(
                "db_vacuum: VACUUM recovered after %d consecutive failures",
                prev_failures,
            )
        state["consecutive_failures"] = 0
        newly_chronic = False
    else:
        state["consecutive_failures"] = prev_failures + 1
        newly_chronic = state["consecutive_failures"] == _FAILURE_ALERT_THRESHOLD

    audit_event(
        "db_vacuum_pass",
        ok=summary.get("ok"),
        bytes_before=summary.get("bytes_before"),
        bytes_after=summary.get("bytes_after"),
        freed_bytes=summary.get("freed_bytes"),
        duration_s=summary.get("duration_s"),
        consecutive_failures=state["consecutive_failures"],
    )

    if summary.get("ok") and summary.get("freed_bytes", 0) >= _BIG_FREE_THRESHOLD_BYTES:
        send_signal_alert(
            f"🗜 Self-heal: monthly VACUUM on conversations.db freed "
            f"{summary['freed_bytes'] / 1024 / 1024:.1f} MB "
            f"({summary['duration_s']:.1f}s).",
            tag="db_vacuum",
        )

    if newly_chronic:
        # One-shot alert when the failure streak crosses the threshold.
        # State tracking prevents re-alerting until recovery resets it.
        send_signal_alert(
            f"⚠️ Self-heal: VACUUM on conversations.db has failed "
            f"{_FAILURE_ALERT_THRESHOLD} consecutive monthly passes. "
            f"The SQLite freelist isn't being reclaimed. conversations.db "
            f"has a single writer (Signal client) so this usually means "
            f"the connection is being held through the VACUUM window — "
            f"check for long-running queries or transactions in "
            f"app.conversation_store.",
            tag="db_vacuum_chronic_failure",
        )

    state["last_summary"] = summary
    write_state_json(_STATE_FILE, state)
