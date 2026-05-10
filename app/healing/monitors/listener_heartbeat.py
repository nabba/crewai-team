"""Listener-heartbeat monitor — detect Firestore polling threads gone silent.

Wave 0/1 closure (#A3, 2026-05-09). Two-layer design:

  1. **Per-listener heartbeats** — preferred path. Each poller in
     ``app/firebase/listeners.py`` touches a file under
     ``workspace/heartbeats/<thread-name>.heartbeat`` at every loop
     iteration. We walk those files and alert on any whose mtime is
     older than ``HEALING_LISTENER_STALE_MIN``. This catches the case
     where ONE listener silently dies (e.g. mode_listener) while others
     stay healthy — invisible to the workspace-wide proxy below.

  2. **Workspace-wide activity proxy** — fallback when no per-listener
     heartbeats exist (e.g. fresh boot before first touch, or
     ``FIREBASE_ENABLED=0`` on laptop dev). Reads the freshest mtime
     across a handful of high-frequency state files; if NONE of them
     are recent, the gateway is likely dead.

Per-listener alerts are de-duped per listener (separate cooldown each
under ``listener_heartbeat_alerts.json``); the fallback alert uses the
legacy single-key cooldown.

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
from app.healing.listener_heartbeats import KNOWN_LISTENERS, list_heartbeats

logger = logging.getLogger(__name__)

_STATE_FILE = "listener_heartbeat_alerts.json"
_ALERT_WINDOW_S = 4 * 3600  # 4 h cooldown — staleness rarely flickers fast

# Files whose mtime is a reliable proxy that the gateway is doing work.
# Used only as a fallback when no per-listener heartbeats are present.
_LIVENESS_PROBES = [
    "workspace/audit.log",                  # any audit write
    "workspace/conversations.db",           # any user message
    "workspace/error_journal.json",         # error_diagnosis activity
    "workspace/audit_journal/current.jsonl",  # auditor cron (rolled-segment active)
    "workspace/control_plane/heartbeat.json",  # explicit heartbeat
]


def _stale_min() -> int:
    try:
        return int(os.getenv("HEALING_LISTENER_STALE_MIN", "30"))
    except ValueError:
        return 30


def _check_per_listener(now: float, threshold_s: int) -> tuple[list[dict], int]:
    """Walk per-listener heartbeats. Returns (stale_records, total_seen).

    Each ``stale_record`` is ``{name, age_s}``. ``age_s`` is ``None``
    when the listener is in ``KNOWN_LISTENERS`` but never produced a
    heartbeat (Phase F #9, 2026-05-09: previously such cases were
    invisible — listener crashed before first touch went unnoticed).
    Listeners NOT in ``KNOWN_LISTENERS`` are checked for staleness only
    so an experimental new poller doesn't fire absent-warnings.
    """
    seen = list_heartbeats()
    seen_names = {h["name"] for h in seen}
    stale = [
        {"name": h["name"], "age_s": int(h["age_s"])}
        for h in seen
        if h["age_s"] > threshold_s
    ]
    # Only fire missing-heartbeat alerts when SOME heartbeats exist —
    # otherwise the entire heartbeat subsystem may be off (laptop dev,
    # FIREBASE_ENABLED=0) and every known listener would alert
    # spuriously.
    if seen:
        for known in KNOWN_LISTENERS:
            if known not in seen_names:
                stale.append({"name": known, "age_s": None})
    return stale, len(seen)


def _check_workspace_proxy(now: float, threshold_s: int) -> tuple[float, str | None]:
    """Fallback proxy: freshest mtime across known state files."""
    repo_root = Path(__file__).resolve().parents[3]
    freshest_age = float("inf")
    freshest_probe: str | None = None
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
    return freshest_age, freshest_probe


def _maybe_alert_per_listener(
    stale: list[dict], state: dict, now: float,
) -> list[str]:
    """De-dup per-listener alerts. Returns list of names actually alerted.

    Two alert flavours: stale (``age_s`` is an int) and missing
    (``age_s`` is None — listener in KNOWN_LISTENERS but never
    produced a heartbeat). Both share the per-listener dedup window.
    """
    alerted_names: list[str] = []
    last_alerts: dict = state.setdefault("per_listener_last_alert", {})
    for record in stale:
        name = record["name"]
        last = float(last_alerts.get(name, 0))
        if now - last < _ALERT_WINDOW_S:
            continue
        last_alerts[name] = now
        alerted_names.append(name)
        if record["age_s"] is None:
            send_signal_alert(
                f"🔌 Self-heal: listener `{name}` has produced NO "
                f"heartbeat — known listener never started, or "
                f"crashed before its first loop iteration. Other "
                f"listeners are healthy (heartbeat subsystem on).",
                tag=f"listener_heartbeat:{name}",
            )
        else:
            human_age = f"{record['age_s'] // 60} min"
            send_signal_alert(
                f"🔌 Self-heal: listener `{name}` heartbeat stale "
                f"— {human_age} since last loop iteration "
                f"(threshold {_stale_min()} min). The listener thread "
                f"may have crashed or hung.",
                tag=f"listener_heartbeat:{name}",
            )
    return alerted_names


def run() -> None:
    now = time.time()
    threshold_s = _stale_min() * 60

    state = read_state_json(_STATE_FILE, {"last_alert_at": 0})

    # --- Path 1: per-listener heartbeats ----------------------------------
    stale, total_seen = _check_per_listener(now, threshold_s)

    if total_seen > 0:
        # Per-listener path is active. Audit and (maybe) alert.
        alerted_names = _maybe_alert_per_listener(stale, state, now)
        audit_event(
            "listener_heartbeat_check",
            mode="per_listener",
            seen=total_seen,
            stale_count=len(stale),
            stale=[r["name"] for r in stale],
            alerted=alerted_names,
            threshold_s=threshold_s,
            known=list(KNOWN_LISTENERS),
        )
        write_state_json(_STATE_FILE, state)
        return

    # --- Path 2: fallback workspace proxy ---------------------------------
    freshest_age, freshest_probe = _check_workspace_proxy(now, threshold_s)
    audit_event(
        "listener_heartbeat_check",
        mode="workspace_proxy",
        freshest_probe=freshest_probe,
        freshest_age_s=int(freshest_age) if freshest_age != float("inf") else None,
        threshold_s=threshold_s,
    )

    if freshest_probe is None or freshest_age > threshold_s:
        if now - state.get("last_alert_at", 0) >= _ALERT_WINDOW_S:
            state["last_alert_at"] = now
            state["last_freshest_probe"] = freshest_probe
            state["last_freshest_age_s"] = (
                int(freshest_age) if freshest_age != float("inf") else None
            )
            human_age = (
                f"{int(freshest_age // 60)} min"
                if freshest_age != float("inf")
                else "∞"
            )
            send_signal_alert(
                f"🔌 Self-heal: listener / heartbeat staleness — newest "
                f"workspace activity is {human_age} old "
                f"(probe `{freshest_probe or 'none'}`). Threshold "
                f"{_stale_min()} min. No per-listener heartbeats found "
                f"(workspace/heartbeats/ empty). Check that pollers and "
                f"idle scheduler are running.",
                tag="listener_heartbeat",
            )
    write_state_json(_STATE_FILE, state)
