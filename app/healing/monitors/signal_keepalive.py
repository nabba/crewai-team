"""Signal 120-day re-registration keepalive (Phase H #2, 2026-05-10).

Signal-cli registrations silently expire after ~4 months of zero
activity from this device. After expiration the client appears to
work locally but messages aren't delivered. Detection only happens
the next time the operator notices replies missing — which can be
days.

Most operators chat regularly enough that this never triggers, but
the gateway's outbound traffic can dip to zero during quiet weeks
+ vacations. This module keeps the registration warm:

  * **Probe** — every 30 days, send a heartbeat message to the
    operator's own number. Signal-cli treats outbound-to-self as a
    valid keepalive.
  * **Body** — short, dated, prefixed with a unique tag so the
    operator's client can filter / mute these.

Cadence: daily probe with internal 30-day gate. Master switch:
``SIGNAL_KEEPALIVE_ENABLED`` (default ON).

If signal-cli is unreachable when the keepalive fires, alert
escalation goes through the existing ``signal_heartbeat`` monitor
chain (Wave 2 #3) — Signal alert → PWA push → email after 7 fails.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

from app.healing.handlers._common import (
    audit_event,
    read_state_json,
    send_signal_alert,
    write_state_json,
)

logger = logging.getLogger(__name__)


_STATE_FILE = "signal_keepalive.json"
_RUN_CADENCE_S = 24 * 3600              # daily probe
_KEEPALIVE_INTERVAL_DAYS = 30           # send keepalive every 30 days
_KEEPALIVE_TAG = "[andrusai-keepalive]"


def _enabled() -> bool:
    return os.getenv("SIGNAL_KEEPALIVE_ENABLED", "true").lower() in (
        "true", "1", "yes", "on",
    )


def _send_keepalive() -> bool:
    """Send a heartbeat to the operator's own number. Returns True on success.

    The message lands in the operator's Signal as a "Note to Self"
    style entry (signal-cli supports send-to-self when sender ==
    recipient). It's tagged so the operator can filter or mute the
    thread.
    """
    try:
        from app.config import get_settings
        from app.signal_client import send_message_blocking
    except Exception:
        logger.debug("signal_keepalive: imports failed", exc_info=True)
        return False
    s = get_settings()
    recipient = (getattr(s, "signal_owner_number", "") or "").strip()
    if not recipient:
        return False

    body = (
        f"{_KEEPALIVE_TAG} {datetime.now(timezone.utc).isoformat(timespec='seconds')}"
        " — registration keepalive (~30d)."
    )
    try:
        ts = send_message_blocking(recipient, body)
        return ts is not None and ts > 0
    except Exception:
        logger.debug("signal_keepalive: send failed", exc_info=True)
        return False


def run() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "ran": False, "sent": False, "days_since_last": None,
    }
    if not _enabled():
        return summary

    state = read_state_json(_STATE_FILE, {
        "last_run_at": 0.0, "last_keepalive_at": 0.0,
        "consecutive_failures": 0,
    })
    now = time.time()
    if now - float(state.get("last_run_at", 0)) < _RUN_CADENCE_S:
        return summary
    state["last_run_at"] = now
    summary["ran"] = True

    last_keepalive = float(state.get("last_keepalive_at", 0))
    if last_keepalive:
        summary["days_since_last"] = int((now - last_keepalive) / 86400)
    else:
        summary["days_since_last"] = None

    # Send only when the interval has elapsed (or never sent yet).
    if last_keepalive and (now - last_keepalive) < _KEEPALIVE_INTERVAL_DAYS * 86400:
        write_state_json(_STATE_FILE, state)
        return summary

    if _send_keepalive():
        state["last_keepalive_at"] = now
        state["consecutive_failures"] = 0
        summary["sent"] = True
    else:
        state["consecutive_failures"] = int(state.get("consecutive_failures", 0)) + 1
        # After 3 consecutive failed keepalives (≥90 days of probe
        # window) escalate — registration may already be lost.
        if state["consecutive_failures"] >= 3:
            try:
                send_signal_alert(
                    f"📡 Signal keepalive failed {state['consecutive_failures']}× — "
                    f"registration may be lost. Check `signal-cli` device "
                    f"status; re-link if needed.",
                    tag="signal_keepalive_failed",
                )
            except Exception:
                pass

    write_state_json(_STATE_FILE, state)
    audit_event(
        "signal_keepalive_pass",
        sent=summary["sent"],
        days_since_last=summary["days_since_last"],
        consecutive_failures=int(state.get("consecutive_failures", 0)),
    )
    return summary
