"""Runbook A — db_pool_reset.

Triggers on the ``connection pool exhausted`` anomaly emitted by
``app/control_plane/db.py`` when ``ThreadedConnectionPool.getconn()``
runs out of slots. Calls ``_reset_pool()`` which closes all open
connections and lets the next request lazily recreate the pool.

Severity is high — this is the dominant production error in this
codebase by ~5× (48k+ occurrences in the rolling errors.jsonl). Pool
exhaustion typically reflects a connection leak from an aborted
worker that never returned its connection. Reset releases the leaked
slots; the next ``getconn()`` rebuilds. Idempotent.

Rate limit: ``_reset_pool()`` is cheap (~10 ms), but doing it under
sustained load can drop genuinely in-flight queries. Cap to 2 resets
per hour. The dispatcher's per-handler concurrency cap (max 1) plus
this rate limiter combine to prevent reset storms.
"""
from __future__ import annotations

import logging

from app.healing.handlers._common import (
    RateLimiter,
    audit_event,
    compute_signature,
    sample_contains,
    send_signal_alert,
)
from app.healing.runbooks import RunbookResult, register_runbook

logger = logging.getLogger(__name__)

# Canonical message used by ``app/control_plane/db.py`` line 103 in the
# generic SQL-error catch. The pool-exhausted text comes from psycopg2's
# ``pool.PoolError("connection pool exhausted")`` which lands in that
# generic except clause, then logged via ``logger.error("control_plane
# SQL error: %s", e)``.
_LOGGER = "app.control_plane.db"
_MESSAGE = "control_plane SQL error: connection pool exhausted"

# Computed once at import time. If the canonical message drifts the
# integration test ``test_db_pool_reset_signature_matches`` will catch it.
_SIGNATURE = compute_signature(_LOGGER, _MESSAGE)

# 2 resets per hour ceiling (1800 s spacing).
_LIMITER = RateLimiter(cooldown_seconds=1800)


def _handle_db_pool_reset(anomaly: dict) -> RunbookResult:
    """Reset the control-plane DB pool. Idempotent.

    The handler is gated four ways before any work runs:
      1. Master env flag ``ERROR_RUNBOOKS_ENABLED`` (dispatcher).
      2. Per-runbook ``enabled`` flag in ``runbook_settings.json``.
      3. Recurrence threshold (dispatcher; default 1).
      4. Local 30-min rate limit so sustained spikes don't storm.
    """
    # Defensive: if the message subtly drifted, the hash collided with
    # something else — refuse to act.
    if not sample_contains(anomaly, "connection pool exhausted"):
        return RunbookResult(
            name="db_pool_reset",
            success=False,
            detail="sample-text mismatch — refused",
            error="sample_mismatch",
        )

    if not _LIMITER.allow():
        return RunbookResult(
            name="db_pool_reset",
            success=False,
            detail="rate-limited; skipped",
            error="rate_limited",
        )

    try:
        from app.control_plane import db as cp_db
    except Exception as exc:
        return RunbookResult(
            name="db_pool_reset",
            success=False,
            detail="cp_db import failed",
            error=f"{type(exc).__name__}: {exc}",
        )

    try:
        cp_db._reset_pool()
    except Exception as exc:
        logger.warning("db_pool_reset: _reset_pool raised: %s", exc)
        return RunbookResult(
            name="db_pool_reset",
            success=False,
            detail="_reset_pool raised",
            error=f"{type(exc).__name__}: {exc}",
        )

    audit_event(
        "db_pool_reset",
        pattern_signature=anomaly.get("pattern_signature"),
        hourly_rate=anomaly.get("hourly_rate"),
        baseline=anomaly.get("baseline"),
    )
    logger.info("db_pool_reset: pool reset; next getconn() will rebuild")

    # If the spike is severe, also surface to the operator. Keep it dry —
    # one Signal per limiter window is enough.
    if (anomaly.get("severity") or "").lower() == "critical":
        send_signal_alert(
            "🩹 Self-heal: control_plane connection pool was exhausted "
            f"({anomaly.get('hourly_rate', '?')}/h vs baseline "
            f"{anomaly.get('baseline', '?')}/h). Pool reset; investigating "
            "whether a connection leak is the cause.",
            tag="db_pool_reset",
        )

    return RunbookResult(
        name="db_pool_reset",
        success=True,
        detail=f"pool reset for sig={_SIGNATURE}",
    )


def register() -> None:
    """Register this runbook with the dispatcher. Idempotent."""
    register_runbook("db_pool_reset", _SIGNATURE, _handle_db_pool_reset)
