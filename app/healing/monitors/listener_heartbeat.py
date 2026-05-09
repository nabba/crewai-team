"""Listener-heartbeat monitor — detect Firestore polling threads gone silent.

The Firestore listener pollers in ``app/firebase/listeners.py`` swallow
all exceptions in their inner loops (deliberately — they're meant to be
robust). Side effect: if the polling itself stops doing useful work
(e.g. authentication broke and every poll returns nothing), the system
silently degrades — KB queue uploads, mode changes from the dashboard,
chat-inbox messages all stop reaching the gateway with no error log.

This monitor checks ``status/heartbeat`` Firestore-like activity proxies
on disk. The system already publishes activity timestamps to several
state files; we read the most recently-touched one as the proxy for
"the gateway is alive and listeners are firing".

Tunable via env: ``HEALING_LISTENER_STALE_MIN`` (default 30 min).
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from app.healing.handlers._common import (
    audit_event,
    read_state_json,
    send_signal_alert,
    write_state_json,
)

logger = logging.getLogger(__name__)

_STATE_FILE = "listener_heartbeat_alerts.json"
_ALERT_WINDOW_S = 4 * 3600  # 4 h cooldown — staleness rarely flickers fast

# Files whose mtime is a reliable proxy that the gateway is doing work.
# We pick high-frequency writes from different subsystems so any one
# of them being fresh proves liveness.
_LIVENESS_PROBES = [
    "workspace/audit.log",                  # any audit write
    "workspace/conversations.db",           # any user message
    "workspace/error_journal.json",         # error_diagnosis activity
    "workspace/audit_journal.json",         # auditor cron
    "workspace/control_plane/heartbeat.json",  # explicit heartbeat
]


def _stale_min() -> int:
    try:
        return int(os.getenv("HEALING_LISTENER_STALE_MIN", "30"))
    except ValueError:
        return 30


def run() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    now = time.time()
    stale_threshold = _stale_min() * 60

    freshest_age = float("inf")
    freshest_probe = None
    for rel in _LIVENESS_PROBES:
        p = repo_root / rel
        if not p.exists():
            continue
        try:
            age = now - p.stat().st_mtime
        except Exception:
            continue
        if age < freshest_age:
            freshest_age = age
            freshest_probe = rel

    audit_event(
        "listener_heartbeat_check",
        freshest_probe=freshest_probe,
        freshest_age_s=int(freshest_age) if freshest_age != float("inf") else None,
        stale_threshold_s=stale_threshold,
    )

    if freshest_probe is None or freshest_age > stale_threshold:
        # No probe seen recently → likely the gateway / its listeners
        # have stopped. De-dup alerts.
        state = read_state_json(_STATE_FILE, {"last_alert_at": 0})
        if now - state.get("last_alert_at", 0) >= _ALERT_WINDOW_S:
            state["last_alert_at"] = now
            state["last_freshest_probe"] = freshest_probe
            state["last_freshest_age_s"] = (
                int(freshest_age) if freshest_age != float("inf") else None
            )
            write_state_json(_STATE_FILE, state)
            human_age = (
                f"{int(freshest_age // 60)} min"
                if freshest_age != float("inf")
                else "∞"
            )
            send_signal_alert(
                f"🔌 Self-heal: listener / heartbeat staleness — newest "
                f"workspace activity is {human_age} old "
                f"(probe `{freshest_probe or 'none'}`). Threshold "
                f"{_stale_min()} min. Check that pollers and idle "
                f"scheduler are running.",
                tag="listener_heartbeat",
            )
