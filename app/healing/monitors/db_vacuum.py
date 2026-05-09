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

    audit_event(
        "db_vacuum_pass",
        ok=summary.get("ok"),
        bytes_before=summary.get("bytes_before"),
        bytes_after=summary.get("bytes_after"),
        freed_bytes=summary.get("freed_bytes"),
        duration_s=summary.get("duration_s"),
    )

    if summary.get("ok") and summary.get("freed_bytes", 0) >= _BIG_FREE_THRESHOLD_BYTES:
        send_signal_alert(
            f"🗜 Self-heal: monthly VACUUM on conversations.db freed "
            f"{summary['freed_bytes'] / 1024 / 1024:.1f} MB "
            f"({summary['duration_s']:.1f}s).",
            tag="db_vacuum",
        )

    state["last_summary"] = summary
    write_state_json(_STATE_FILE, state)
