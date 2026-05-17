"""boot_state — single shared signal for "lifespan setup is complete".

Background:
  At gateway boot, 11+ subsystems start daemon threads or schedule
  background work. Historically each picked its own warm-up duration
  (30 / 60 / 90 / 120 / 300 s) and slept that long before doing anything
  useful. These local guesses do not coordinate, can be too short on a
  slow boot, and waste time on a fast one.

  This module exposes one explicit signal. ``mark_boot_complete()`` is
  called by the FastAPI lifespan immediately before ``yield`` — the point
  at which the gateway is about to start accepting HTTP requests.
  Subsystems that want to defer background work until the system is
  ready consult ``is_boot_complete()`` or block on
  ``wait_for_boot_complete(timeout)``.

  The first consumer is ``idle_scheduler.is_idle()``, which previously
  returned True at boot (because ``_last_task_end == 0``) and let dozens
  of LIGHT jobs storm into a still-initializing gateway — the proximate
  cause of the 2026-05-17 ``/signal/inbound`` starvation incident.

Contract:
  - Set-once. Repeated ``mark_boot_complete()`` calls are no-ops.
  - Passive. No timer, no auto-set. Consumers decide their own fallback.
  - Inert when unset. ``is_boot_complete()`` returns False until the
    lifespan calls ``mark_boot_complete()``.

Safety for out-of-band callers:
  - Tests and other harnesses that don't run the real lifespan SHOULD
    either call ``mark_boot_complete()`` themselves OR have their own
    fallback. The idle scheduler implements such a fallback locally.
  - No module under ``app/`` imports ``boot_state`` at import time except
    when it's about to consume the signal — no circular-import risk.
"""
from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)

_boot_event = threading.Event()
_boot_completed_at: float | None = None  # time.monotonic() at signal moment
_lock = threading.Lock()


def mark_boot_complete() -> None:
    """Signal that the lifespan has finished startup setup.

    Called by the FastAPI lifespan immediately before ``yield``.
    Idempotent — repeated calls are no-ops and do not move the
    timestamp.
    """
    global _boot_completed_at
    with _lock:
        if _boot_event.is_set():
            return
        _boot_completed_at = time.monotonic()
        _boot_event.set()
    logger.info("boot_state: boot complete")


def is_boot_complete() -> bool:
    """Non-blocking check: has ``mark_boot_complete()`` been called?"""
    return _boot_event.is_set()


def wait_for_boot_complete(timeout: float | None = None) -> bool:
    """Block until the signal arrives, or ``timeout`` seconds elapse.

    Returns True if the signal arrived in time, False on timeout. A
    caller that proceeds anyway on False SHOULD log a warning and
    explain its own fallback policy — a missing boot signal is unusual.
    """
    return _boot_event.wait(timeout)


def boot_completed_at() -> float | None:
    """Return ``time.monotonic()`` at the moment the signal was set, or None."""
    return _boot_completed_at


def _reset_for_tests() -> None:
    """Test-only: clear state so each test starts fresh.

    Do NOT call this in production code — there is no legitimate reason
    to un-mark a completed boot.
    """
    global _boot_completed_at
    with _lock:
        _boot_event.clear()
        _boot_completed_at = None
