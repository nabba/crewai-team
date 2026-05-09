"""Daemon-thread reaper — re-spawn known daemons that vanished.

Defends against silent thread death: an uncaught exception inside a
daemon's loop, an OOM-killed worker, or a container restart race that
left some thread stuck. Without this, the system silently degrades —
we ship more daemons every PR (Wave 1+2 add several), and any one of
them dying takes its observability surface offline.

## How it works

A single watchdog thread (``healing-watchdog``) wakes every 60 s and
walks ``threading.enumerate()``. For each daemon name in
``_REGISTERED_DAEMONS``, if no thread by that name is alive, the
watchdog calls the registered ``start_fn``. Each registered ``start``
function is required to be idempotent + thread-liveness-aware — it
checks ``threading.enumerate()`` itself before re-spawning, so a
second caller won't create a duplicate.

## Backoff

A daemon that crashes 3+ times within an hour is *given up on*. The
watchdog stops re-spawning it and emits a Signal alert so the operator
can intervene. Without the cap, a daemon that crashes on import
(e.g. broken dependency) would re-spawn at the watchdog's full
60 s cadence indefinitely.

## Liveness footprint

Every loop iteration touches ``workspace/healing/watchdog_heartbeat``
so the existing ``cron_liveness`` monitor can detect the watchdog
itself going silent. The watchdog dies → footprint goes stale →
``cron_liveness`` alerts.

## Master switch

``HEALING_WATCHDOG_ENABLED`` (default ``true``). Set ``false`` to
disable the watchdog entirely; existing daemons keep running but
won't be re-spawned on death.
"""
from __future__ import annotations

import importlib
import logging
import os
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


# ── Registry: thread name → (module, function name) ──────────────────────


_REGISTERED_DAEMONS: dict[str, tuple[str, str]] = {
    "healing-monitors": ("app.healing.monitors", "start"),
    "healing-auditor-bridge": ("app.healing.auditor_bridge", "start"),
}


def register_daemon(thread_name: str, *, module: str, start_fn: str) -> None:
    """Register a daemon for watchdog supervision. ``start_fn`` MUST be
    idempotent (safe to call when the thread already runs).
    """
    _REGISTERED_DAEMONS[thread_name] = (module, start_fn)


# ── Constants ────────────────────────────────────────────────────────────


_WATCHDOG_THREAD_NAME = "healing-watchdog"
_CHECK_INTERVAL_S = 60
_WARMUP_S = 90
_MAX_CRASHES_PER_HOUR = 3
_GIVEUP_RESET_HOURS = 24  # after a 24h quiet streak, reset give-up state


_HEARTBEAT_PATH = (
    Path(__file__).resolve().parents[2] / "workspace" / "healing"
    / "watchdog_heartbeat"
)


# ── Internal state ───────────────────────────────────────────────────────


_started = False
_start_lock = threading.Lock()
_stop_event = threading.Event()

# Per-daemon crash timestamps (sliding 1-h window).
_crash_history: dict[str, deque[float]] = defaultdict(
    lambda: deque(maxlen=_MAX_CRASHES_PER_HOUR + 5)
)

# Daemons we've alerted-and-stopped on; cleared after a 24h quiet streak.
_given_up: dict[str, float] = {}  # name → ts of give-up


# ── Master switch ────────────────────────────────────────────────────────


def _enabled() -> bool:
    return os.getenv("HEALING_WATCHDOG_ENABLED", "true").lower() in (
        "true", "1", "yes",
    )


# ── Helpers ──────────────────────────────────────────────────────────────


def _is_alive(thread_name: str) -> bool:
    return any(
        t.name == thread_name and t.is_alive()
        for t in threading.enumerate()
    )


def _attempt_start(daemon_name: str) -> bool:
    """Import the module and call its start function. Returns success."""
    module_path, fn_name = _REGISTERED_DAEMONS[daemon_name]
    try:
        mod = importlib.import_module(module_path)
        start_fn: Callable[[], None] = getattr(mod, fn_name)
    except Exception:
        logger.warning(
            "watchdog: cannot import %s.%s", module_path, fn_name,
            exc_info=True,
        )
        return False
    try:
        start_fn()
    except Exception:
        logger.warning(
            "watchdog: %s.%s() raised", module_path, fn_name, exc_info=True,
        )
        return False
    # The start_fn may have been a no-op if the daemon was already alive
    # (race). Re-check liveness to confirm a thread by the right name
    # exists.
    return _is_alive(daemon_name)


def _touch_heartbeat() -> None:
    """Update mtime on the heartbeat file so cron_liveness can see we're
    alive. Best-effort — failure is logged at debug level only.
    """
    try:
        _HEARTBEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
        _HEARTBEAT_PATH.touch(exist_ok=True)
    except Exception:
        logger.debug("watchdog: heartbeat touch failed", exc_info=True)


def _send_giveup_alert(daemon_name: str, crashes: int) -> None:
    """Best-effort Signal alert when a daemon hits the crash cap."""
    try:
        from app.life_companion._common import send_signal_alert
        send_signal_alert(
            f"💀 Self-heal: daemon `{daemon_name}` has crashed {crashes} times "
            f"in the last hour. Watchdog has given up — operator intervention "
            f"required. The daemon will not be re-spawned for at least "
            f"{_GIVEUP_RESET_HOURS} hours.",
            tag="healing_watchdog",
        )
    except Exception:
        logger.debug("watchdog: giveup signal alert failed", exc_info=True)


def _audit(action: str, **detail) -> None:
    try:
        from app.life_companion._common import audit_event
        audit_event(action, **detail)
    except Exception:
        logger.debug("watchdog: audit event failed", exc_info=True)


# ── Loop body (testable in isolation) ────────────────────────────────────


def _check_and_respawn() -> dict:
    """One reaper pass. Returns a summary dict for tests + audit.

    Side effects: re-spawns dead daemons, updates crash history, fires
    give-up alerts when needed.
    """
    now = time.time()
    summary = {
        "alive": [],
        "respawned": [],
        "given_up": [],
        "still_in_giveup": [],
    }

    for daemon_name in list(_REGISTERED_DAEMONS.keys()):
        # Reset give-up state after 24h of quiet — a long-stable
        # daemon shouldn't be permanently locked out.
        if daemon_name in _given_up:
            given_up_at = _given_up[daemon_name]
            if now - given_up_at >= _GIVEUP_RESET_HOURS * 3600:
                _given_up.pop(daemon_name, None)
                _crash_history[daemon_name].clear()

        if _is_alive(daemon_name):
            summary["alive"].append(daemon_name)
            continue

        if daemon_name in _given_up:
            summary["still_in_giveup"].append(daemon_name)
            continue

        # Drop crashes older than 1 h from the window.
        history = _crash_history[daemon_name]
        while history and now - history[0] > 3600:
            history.popleft()

        if len(history) >= _MAX_CRASHES_PER_HOUR:
            _given_up[daemon_name] = now
            summary["given_up"].append(daemon_name)
            _audit(
                "watchdog_giveup",
                daemon=daemon_name,
                crashes_in_window=len(history),
            )
            _send_giveup_alert(daemon_name, len(history))
            continue

        # Re-spawn attempt.
        history.append(now)
        if _attempt_start(daemon_name):
            summary["respawned"].append(daemon_name)
            _audit(
                "watchdog_respawn",
                daemon=daemon_name,
                crashes_in_window=len(history),
            )
            logger.warning("watchdog: re-spawned %s", daemon_name)
        else:
            # Failed to spawn — counts as a crash.
            logger.warning("watchdog: spawn of %s failed", daemon_name)

    return summary


# ── Loop driver ──────────────────────────────────────────────────────────


def _watchdog_loop() -> None:
    """The watchdog's own loop. Wrapped so any exception in
    ``_check_and_respawn`` doesn't kill us — except KeyboardInterrupt
    which we propagate.
    """
    if _stop_event.wait(_WARMUP_S):
        return
    while not _stop_event.is_set():
        try:
            _check_and_respawn()
            _touch_heartbeat()
        except Exception:
            logger.debug("watchdog: loop iteration raised", exc_info=True)
        if _stop_event.wait(_CHECK_INTERVAL_S):
            return


# ── Public API ───────────────────────────────────────────────────────────


def start() -> None:
    """Start the watchdog daemon. Truly idempotent — uses thread-liveness
    detection so a dead watchdog is replaced (rather than the start being
    silently skipped because of a stale ``_started`` flag).
    """
    global _started
    if not _enabled():
        logger.info("watchdog: disabled via HEALING_WATCHDOG_ENABLED")
        return
    with _start_lock:
        if _is_alive(_WATCHDOG_THREAD_NAME):
            return  # already running
        if _started:
            logger.warning("watchdog: previous watchdog dead, re-spawning")
        _stop_event.clear()
        thread = threading.Thread(
            target=_watchdog_loop, name=_WATCHDOG_THREAD_NAME, daemon=True,
        )
        thread.start()
        _started = True
        logger.info("watchdog: daemon started (warm-up=%ds)", _WARMUP_S)


def stop() -> None:
    _stop_event.set()


# Eager start on import (matches the pattern of other healing modules).
start()
