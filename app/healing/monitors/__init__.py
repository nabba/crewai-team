"""Proactive monitors — close the silent-failure gaps for years-of-uptime.

Monitors run periodically in a daemon thread started at import time.
None of them mutate code or schedule (with the exception of opt-in
gated runners like ``db_backup`` and ``db_vacuum``); they observe and
alert on conditions the existing reactive runbooks can't see (because
nothing throws).

Currently registered monitors:

  * ``disk_quota``               — free disk below threshold → Signal.
  * ``listener_heartbeat``       — Firestore polling threads gone silent.
  * ``cron_liveness``            — scheduled jobs that haven't fired on time.
  * ``vendor_sunset``            — provider models flagged deprecated; files CRs.
  * ``idle_cooldown``            — idle-scheduler job stuck in cooldown >24 h.
  * ``audit_chain_check``        — re-verify the hash-chained audit JSONL daily.
  * ``lock_housekeeper``         — sweep stale lock files weekly.
  * ``adapter_lifecycle``        — orphan / dead-pointer / bloat in workspace/lora.
  * ``retention_chromadb`` etc.  — bounded growth on KB indices, worktrees, attachments.
  * ``signal_heartbeat``         — multi-channel escalation if Signal is wedged.
  * ``db_vacuum``                — monthly conversations.db VACUUM (Wave 0/1 #A6).
  * ``log_archival``             — daily errors.jsonl + audit_journal rotate (Wave 0/1 #A5).
  * ``db_backup``                — opt-in weekly Postgres+Neo4j+ChromaDB (Wave 0/1 #A1).

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
    "audit_chain_check": 23 * 3600,    # ~daily — Wave 1 (#5)
    "lock_housekeeper": 6 * 3600,      # 6h — Wave 1 (#9)
    "adapter_lifecycle": 24 * 3600,    # daily probe — internal cadence 30 d; Wave 2 (#4)
    "retention_chromadb": 24 * 3600,   # daily probe — internal cadence weekly; Wave 2 (#8)
    "retention_worktrees": 12 * 3600,  # twice daily probe — internal cadence daily; Wave 2 (#8)
    "retention_attachments": 12 * 3600,  # twice daily probe — internal cadence daily; Wave 2 (#8)
    "signal_heartbeat": 12 * 3600,     # twice daily probe — internal cadence daily; Wave 2 (#3)
    "db_vacuum": 24 * 3600,            # daily probe — internal cadence 30 d; Wave 0/1 (#A6)
    "log_archival": 6 * 3600,          # 6 h probe — internal cadence daily; Wave 0/1 (#A5)
    "db_backup": 6 * 3600,             # 6 h probe — internal cadence weekly; Wave 0/1 (#A1)
    "silent_regression_detector": 4 * 3600,  # 4 h probe — Phase C #2 (2026-05-09)
    "pattern_learner": 24 * 3600,            # daily probe — Phase C #4 (2026-05-09)
    "llm_output_drift": 24 * 3600,           # daily probe — Phase D #6 (2026-05-09)
    "signal_keepalive": 24 * 3600,           # daily probe — internal 30-day gate; Phase H #2
    "restore_drill": 24 * 3600,              # daily probe — alerts at 100d stale; Phase H #1
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
    try:
        from app.training import adapter_lifecycle
        monitors.append(("adapter_lifecycle", adapter_lifecycle.run, _DEFAULT_CADENCE_S["adapter_lifecycle"], 0.0))
    except Exception:
        logger.debug("monitors: adapter_lifecycle import failed", exc_info=True)
    try:
        from app.healing.monitors import retention
        monitors.append(("retention_chromadb", retention.run_chromadb, _DEFAULT_CADENCE_S["retention_chromadb"], 0.0))
        monitors.append(("retention_worktrees", retention.run_worktrees, _DEFAULT_CADENCE_S["retention_worktrees"], 0.0))
        monitors.append(("retention_attachments", retention.run_attachments, _DEFAULT_CADENCE_S["retention_attachments"], 0.0))
    except Exception:
        logger.debug("monitors: retention import failed", exc_info=True)
    try:
        from app.healing.monitors import signal_heartbeat
        monitors.append(("signal_heartbeat", signal_heartbeat.run, _DEFAULT_CADENCE_S["signal_heartbeat"], 0.0))
    except Exception:
        logger.debug("monitors: signal_heartbeat import failed", exc_info=True)
    try:
        from app.healing.monitors import db_vacuum
        monitors.append(("db_vacuum", db_vacuum.run, _DEFAULT_CADENCE_S["db_vacuum"], 0.0))
    except Exception:
        logger.debug("monitors: db_vacuum import failed", exc_info=True)
    try:
        from app.healing.monitors import log_archival
        monitors.append(("log_archival", log_archival.run, _DEFAULT_CADENCE_S["log_archival"], 0.0))
    except Exception:
        logger.debug("monitors: log_archival import failed", exc_info=True)
    try:
        from app.healing.monitors import db_backup
        monitors.append(("db_backup", db_backup.run, _DEFAULT_CADENCE_S["db_backup"], 0.0))
    except Exception:
        logger.debug("monitors: db_backup import failed", exc_info=True)
    try:
        from app.healing import silent_regression_detector
        monitors.append((
            "silent_regression_detector", silent_regression_detector.run,
            _DEFAULT_CADENCE_S["silent_regression_detector"], 0.0,
        ))
    except Exception:
        logger.debug("monitors: silent_regression_detector import failed", exc_info=True)
    try:
        from app.healing import pattern_learner
        monitors.append((
            "pattern_learner", pattern_learner.run,
            _DEFAULT_CADENCE_S["pattern_learner"], 0.0,
        ))
    except Exception:
        logger.debug("monitors: pattern_learner import failed", exc_info=True)
    try:
        from app.healing import llm_output_drift
        monitors.append((
            "llm_output_drift", llm_output_drift.run,
            _DEFAULT_CADENCE_S["llm_output_drift"], 0.0,
        ))
    except Exception:
        logger.debug("monitors: llm_output_drift import failed", exc_info=True)
    try:
        from app.healing.monitors import signal_keepalive
        monitors.append((
            "signal_keepalive", signal_keepalive.run,
            _DEFAULT_CADENCE_S["signal_keepalive"], 0.0,
        ))
    except Exception:
        logger.debug("monitors: signal_keepalive import failed", exc_info=True)
    try:
        from app.healing.monitors import restore_drill
        monitors.append((
            "restore_drill", restore_drill.run,
            _DEFAULT_CADENCE_S["restore_drill"], 0.0,
        ))
    except Exception:
        logger.debug("monitors: restore_drill import failed", exc_info=True)

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


_DAEMON_THREAD_NAME = "healing-monitors"


def _is_running() -> bool:
    """True iff a thread named ``_DAEMON_THREAD_NAME`` is currently alive."""
    return any(
        t.name == _DAEMON_THREAD_NAME and t.is_alive()
        for t in threading.enumerate()
    )


def start() -> None:
    """Start the daemon driver. Truly idempotent — checks thread liveness
    on every call, so the watchdog can call this to re-spawn after death
    and the call is safe even if another thread already restarted us.

    The previous implementation gated on a ``_driver_started`` flag that
    drifted out of sync when the thread died (flag stayed True, no restart
    was possible). The new path detects death directly via
    ``threading.enumerate()``.
    """
    global _driver_started
    if not _enabled():
        logger.info("healing.monitors: disabled via HEALING_MONITORS_ENABLED")
        return
    with _driver_lock:
        if _is_running():
            return  # already alive — nothing to do
        if _driver_started:
            logger.warning(
                "healing.monitors: previous daemon thread is dead, re-spawning"
            )
        _stop_event.clear()
        thread = threading.Thread(
            target=_driver, name=_DAEMON_THREAD_NAME, daemon=True,
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
