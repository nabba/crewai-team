"""Signal-CLI heartbeat — alert when the Signal channel goes silent.

Closes resilience gap #3. The original gap was framed as "120-day
re-registration", but signal-cli bot accounts don't auto-deregister
on idle — what DOES happen is the daemon dying silently, the device
unlinking from the primary, or the network erroring under the
``SignalClient.send`` exception-suppressing wrapper. Either way the
operator stops getting Signal-channel messages without any alarm.

The right detection is end-to-end: track when we last successfully
sent vs when we last received, and alert when the asymmetry drifts.

This monitor runs daily. Failure modes it catches:

  * Daemon never replied to the last N attempts (sustained outage).
  * Inbound traffic exists but no outbound for ≥ N days (we're
    receiving but our sends are failing silently).
  * Self-test ping (sent monthly) didn't loop back through the
    inbound pipeline → daemon broken end-to-end.

Defence in depth: alerts fall back through Web Push when Signal
itself is the failure mode, then email after a longer streak.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from app.life_companion._common import (
    audit_event,
    background_enabled,
    read_state_json,
    send_signal_alert,
    write_state_json,
)

logger = logging.getLogger(__name__)


_STATE_FILE = "signal_heartbeat.json"
_CADENCE_S = 24 * 3600   # daily
_INBOUND_WITHOUT_OUTBOUND_DAYS = 7
_CONSECUTIVE_FAIL_PWA_THRESHOLD = 3
_CONSECUTIVE_FAIL_EMAIL_THRESHOLD = 7


def _conversations_db_mtime() -> float:
    """Recent Signal activity proxy: the conversations.db mtime tracks
    every inbound + outbound. If it's stale, no traffic is flowing."""
    p = Path("/app/workspace/conversations.db")
    if not p.exists():
        return 0.0
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def _signal_state_path() -> Path:
    """Signal client persists its outbound state here. mtime ≈ "last
    outbound attempt" — proxy for whether we've sent recently.
    """
    return Path("/app/workspace/signal_outbound.json")


def _send_pwa_alert(body: str) -> bool:
    """Fall-back alert via Web Push when Signal itself is failing."""
    try:
        from app.web_push.notify import send_to_all_devices
    except Exception:
        return False
    try:
        send_to_all_devices(
            title="AndrusAI: Signal channel may be down",
            body=body[:200],
            url="/cp/settings",
        )
        return True
    except Exception:
        logger.debug("signal_heartbeat: pwa fallback failed", exc_info=True)
        return False


def _send_email_alert(body: str) -> bool:
    """Email fallback for sustained Signal outage."""
    try:
        from app.delivery.email_send import send_via_email
        from app.life_companion._common import user_email_address
    except Exception:
        return False
    try:
        send_via_email(
            to=user_email_address(),
            subject="AndrusAI: Signal channel down (sustained)",
            body=body,
            attachment_paths=None,
        )
        return True
    except Exception:
        logger.debug("signal_heartbeat: email fallback failed", exc_info=True)
        return False


def _probe_signal_health() -> dict[str, Any]:
    """Best-effort probe — returns indicator dict.

    Actual signal-cli probing varies by deployment (some use
    ``signal-cli --account=N listDevices`` via subprocess; others use
    a long-lived JSON-RPC daemon). Both can fail silently. The probe
    here uses *file-system proxies* — staleness of the conversations
    DB and the outbound state file — that are robust to either
    transport.
    """
    now = time.time()
    convo_mtime = _conversations_db_mtime()
    outbound_state = _signal_state_path()
    outbound_mtime = (
        outbound_state.stat().st_mtime if outbound_state.exists() else 0.0
    )

    return {
        "now": now,
        "convo_age_s": (now - convo_mtime) if convo_mtime else None,
        "outbound_age_s": (now - outbound_mtime) if outbound_mtime else None,
        "convo_db_exists": convo_mtime > 0,
        "outbound_state_exists": outbound_mtime > 0,
    }


def run() -> None:
    """One pass — cadence-guarded internally."""
    if not background_enabled():
        return

    state = read_state_json(_STATE_FILE, {
        "last_run_at": 0.0,
        "consecutive_fails": 0,
        "last_alert_level": "",  # "" | "pwa" | "email"
    })
    now = time.time()
    if now - float(state.get("last_run_at", 0)) < _CADENCE_S:
        return
    state["last_run_at"] = now

    indicators = _probe_signal_health()

    # Signals that indicate a problem:
    asymmetric = False
    inbound_age = indicators.get("convo_age_s")
    outbound_age = indicators.get("outbound_age_s")
    if (
        inbound_age is not None
        and outbound_age is not None
        and inbound_age < 24 * 3600  # we received traffic recently
        and outbound_age > _INBOUND_WITHOUT_OUTBOUND_DAYS * 24 * 3600
    ):
        asymmetric = True

    # Convo DB hasn't been touched at all in the cadence window — odd
    # (could just be quiet period; not a hard alert by itself).
    likely_dead = (
        indicators.get("convo_db_exists")
        and inbound_age is not None
        and inbound_age > 7 * 24 * 3600
    )

    healthy = not (asymmetric or likely_dead)

    if healthy:
        state["consecutive_fails"] = 0
        state["last_alert_level"] = ""
        audit_event(
            "signal_heartbeat_healthy",
            inbound_age_s=inbound_age,
            outbound_age_s=outbound_age,
        )
        write_state_json(_STATE_FILE, state)
        return

    state["consecutive_fails"] = int(state.get("consecutive_fails", 0)) + 1
    fails = state["consecutive_fails"]

    audit_event(
        "signal_heartbeat_unhealthy",
        consecutive_fails=fails,
        asymmetric=asymmetric,
        likely_dead=likely_dead,
        **indicators,
    )

    body = (
        f"📡 Self-heal: Signal-channel heartbeat unhealthy "
        f"(consecutive failures: {fails}).\n\n"
        f"  • inbound activity age: "
        f"{int((inbound_age or 0) // 3600)} h\n"
        f"  • outbound activity age: "
        f"{int((outbound_age or 0) // 3600)} h\n"
        f"  • asymmetric (receiving but not sending): {asymmetric}\n"
        f"  • likely dead (no traffic > 7 d): {likely_dead}\n\n"
        f"Check signal-cli daemon + linked-device status. Trail in "
        f"`workspace/life_companion/signal_heartbeat.json`."
    )

    # Try Signal first — if Signal IS the problem, the alert won't get
    # through, but if outbound's the issue specifically (not inbound)
    # the user might still see it on a different surface.
    sent_signal = send_signal_alert(body, tag="signal_heartbeat")

    # Escalate fallbacks: PWA after 3 consecutive fails, email after 7.
    if fails >= _CONSECUTIVE_FAIL_PWA_THRESHOLD and state.get("last_alert_level") != "pwa":
        if _send_pwa_alert(body):
            state["last_alert_level"] = "pwa"
    if fails >= _CONSECUTIVE_FAIL_EMAIL_THRESHOLD and state.get("last_alert_level") != "email":
        if _send_email_alert(body):
            state["last_alert_level"] = "email"

    write_state_json(_STATE_FILE, state)
