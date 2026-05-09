"""Proactive monitors — close the silent-failure gaps for years-of-uptime.

Five monitors run periodically in a daemon thread started at import time.
None of them mutate code or schedule, all just observe and alert on
conditions the existing reactive runbooks can't see (because nothing
throws):

  * ``disk_quota``         — free disk below threshold → Signal alert.
  * ``listener_heartbeat`` — Firestore polling threads gone silent.
  * ``cron_liveness``      — scheduled jobs that haven't fired on time.
  * ``vendor_sunset``      — provider models flagged deprecated.
  * ``idle_cooldown``      — idle-scheduler job stuck in cooldown >24 h.

The driver runs each monitor on its own cadence inside a single daemon
thread. Failure in one monitor never breaks the others — every step is
wrapped in try/except. The thread waits for a generous warm-up after
process start so it doesn't fight boot.

Master switch: ``HEALING_MONITORS_ENABLED`` (defaults ON; set ``false``
to disable the entire driver). The runbook framework's master switch
(``ERROR_RUNBOOKS_ENABLED``) is independent — monitors can run without
runbooks being on.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Callable

logger = logging.getLogger(__name__)


def _enabled() -> bool:
    return os.getenv("HEALING_MONITORS_ENABLED", "true").lower() in (
        "true", "1", "yes",
    )


# Per-monitor cadence. Slow defaults: monitors are alerts, not real-time.
_DEFAULT_CADENCE_S = {
    "disk_quota": 300,           # 5 min — disk fills slow
    "listener_heartbeat": 600,   # 10 min — match the dashboard refresh
    "cron_liveness": 1800,       # 30 min — cron jobs are O(hour) cadence
    "vendor_sunset": 7 * 86400,  # weekly
    "idle_cooldown": 3600,       # hourly — cooldowns last hours
    "audit_chain_check": 23 * 3600,  # ~daily — Wave 1 (#5)
    "lock_housekeeper": 6 * 3600,    # 6h — Wave 1 (#9)
}

_WARMUP_S = 120  # don't run anything in the first 2 min after import.

_driver_started = False
_driver_lock = threading.Lock()
_stop_event = threading.Event()


def _run_one(name: str, fn: Callable[[], None]) -> None:
    started = time.monotonic()
    try:
        fn()
    except Exception:
        logger.debug("healing.monitors[%s]: raised", name, exc_info=True)
    elapsed = time.monotonic() - started
    if elapsed > 30:
        logger.info("healing.monitors[%s]: slow run %.1fs", name, elapsed)


def _driver() -> None:
    """Single daemon loop. Each monitor runs at its own cadence; the loop
    sleeps 30 s between checks so any individual monitor can fire on time
    without the driver being a tight CPU spinner.
    """
    # Warm-up so monitors don't compete with boot.
    if _stop_event.wait(_WARMUP_S):
        return

    # Lazy import each monitor so a broken one never prevents the others
    # from running. The handler bodies themselves are pure-Python with no
    # framework dependencies past os/pathlib/json/time — they should
    # always import successfully.
    monitors: list[tuple[str, Callable[[], None], int, float]] = []
    try:
        from app.healing.monitors import disk_quota
        monitors.append(("disk_quota", disk_quota.run, _DEFAULT_CADENCE_S["disk_quota"], 0.0))
    except Exception:
        logger.debug("monitors: disk_quota import failed", exc_info=True)
    try:
        from app.healing.monitors import listener_heartbeat
        monitors.append(("listener_heartbeat", listener_heartbeat.run, _DEFAULT_CADENCE_S["listener_heartbeat"], 0.0))
    except Exception:
        logger.debug("monitors: listener_heartbeat import failed", exc_info=True)
    try:
        from app.healing.monitors import cron_liveness
        monitors.append(("cron_liveness", cron_liveness.run, _DEFAULT_CADENCE_S["cron_liveness"], 0.0))
    except Exception:
        logger.debug("monitors: cron_liveness import failed", exc_info=True)
    try:
        from app.healing.monitors import vendor_sunset
        monitors.append(("vendor_sunset", vendor_sunset.run, _DEFAULT_CADENCE_S["vendor_sunset"], 0.0))
    except Exception:
        logger.debug("monitors: vendor_sunset import failed", exc_info=True)
    try:
        from app.healing.monitors import idle_cooldown
        monitors.append(("idle_cooldown", idle_cooldown.run, _DEFAULT_CADENCE_S["idle_cooldown"], 0.0))
    except Exception:
        logger.debug("monitors: idle_cooldown import failed", exc_info=True)
    try:
        from app.healing.monitors import audit_chain_check
        monitors.append(("audit_chain_check", audit_chain_check.run, _DEFAULT_CADENCE_S["audit_chain_check"], 0.0))
    except Exception:
        logger.debug("monitors: audit_chain_check import failed", exc_info=True)
    try:
        from app.healing.monitors import lock_housekeeper
        monitors.append(("lock_housekeeper", lock_housekeeper.run, _DEFAULT_CADENCE_S["lock_housekeeper"], 0.0))
    except Exception:
        logger.debug("monitors: lock_housekeeper import failed", exc_info=True)

    if not monitors:
        logger.warning("healing.monitors: no monitors loaded; driver exiting")
        return

    logger.info("healing.monitors: driver running %d monitors", len(monitors))

    # Mutable cadence + last-run state. Each tuple is replaced on each tick.
    state = [
        {"name": n, "fn": fn, "cadence": cadence, "last_run": 0.0}
        for n, fn, cadence, _ in monitors
    ]

    while not _stop_event.is_set():
        now = time.monotonic()
        for entry in state:
            if now - entry["last_run"] >= entry["cadence"]:
                _run_one(entry["name"], entry["fn"])
                entry["last_run"] = time.monotonic()
        # Sleep 30 s; lets cadences finer than 30 s run jittery but fine
        # for the slowest cadence (weekly) without burning CPU.
        if _stop_event.wait(30):
            return


def start() -> None:
    """Start the daemon driver. Idempotent."""
    global _driver_started
    if not _enabled():
        logger.info("healing.monitors: disabled via HEALING_MONITORS_ENABLED")
        return
    with _driver_lock:
        if _driver_started:
            return
        _stop_event.clear()
        thread = threading.Thread(
            target=_driver, name="healing-monitors", daemon=True,
        )
        thread.start()
        _driver_started = True
        logger.info("healing.monitors: daemon started (warm-up=%ds)", _WARMUP_S)


def stop() -> None:
    """Signal the driver to exit. Mostly used in tests."""
    _stop_event.set()


# Eager start on import. The warm-up + thread isolation make this safe
# even when the surrounding process is still booting.
start()
