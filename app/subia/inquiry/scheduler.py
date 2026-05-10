"""Daemon-thread scheduler for the weekly inquiry pass.

Pattern mirrors :mod:`app.healing.monitors` (eager-start at import
time, env-gated, idempotent restart, watchdog-friendly liveness check
via ``threading.enumerate()``). The reason for a daemon thread instead
of a registration in ``app/idle_scheduler.py`` is the same as for the
healing monitors: the inquiry pass has its own cadence (weekly) that
the round-robin idle scheduler doesn't naturally model.

Cadence: the daemon wakes once per hour and checks
"have ``MIN_INTERVAL_DAYS`` (default 7) passed since the most recent
file under ``wiki/self/inquiries/``?" If yes, runs one pass. Otherwise
sleeps another hour. This keeps the math simple and makes "skipped a
week" behaviour transparent (operator can read mtimes).

Two env switches:

  ``INQUIRY_PASS_ENABLED``       master switch; default ``true``.
                                 ``false`` → daemon never starts.
  ``INQUIRY_MIN_INTERVAL_DAYS``  override the 7-day spacing for tests
                                 / faster iteration in dev.

The daemon NEVER raises into the runtime — every error path is
caught + logged. Failure modes are surfaced via :class:`PassResult`
returned by :func:`run_once`.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


_DAEMON_THREAD_NAME = "subia-inquiry-scheduler"
_WARMUP_S = 30
_POLL_INTERVAL_S = 3600  # 1 hour
_DEFAULT_MIN_INTERVAL_DAYS = 7
_DEFAULT_INQUIRIES_DIR = Path("/app/wiki/self/inquiries")

_driver_lock = threading.Lock()
_driver_started = False
_stop_event = threading.Event()


def _enabled() -> bool:
    return os.getenv("INQUIRY_PASS_ENABLED", "true").lower() in (
        "true", "1", "yes", "on",
    )


def _min_interval_days() -> int:
    raw = os.getenv("INQUIRY_MIN_INTERVAL_DAYS", str(_DEFAULT_MIN_INTERVAL_DAYS))
    try:
        return max(1, int(raw))
    except ValueError:
        return _DEFAULT_MIN_INTERVAL_DAYS


def _is_running() -> bool:
    return any(
        t.name == _DAEMON_THREAD_NAME and t.is_alive()
        for t in threading.enumerate()
    )


def _newest_inquiry_mtime(inquiries_dir: Path) -> float | None:
    if not inquiries_dir.exists():
        return None
    newest: float | None = None
    for f in inquiries_dir.glob("*.md"):
        try:
            m = f.stat().st_mtime
        except OSError:
            continue
        if newest is None or m > newest:
            newest = m
    return newest


def _due_to_run(inquiries_dir: Path | None = None) -> bool:
    """True iff at least ``_min_interval_days`` have elapsed since the most
    recent inquiry file (or no files exist yet)."""
    src = inquiries_dir if inquiries_dir is not None else _DEFAULT_INQUIRIES_DIR
    newest = _newest_inquiry_mtime(src)
    if newest is None:
        return True
    age_days = (time.time() - newest) / 86400.0
    return age_days >= _min_interval_days()


def _resolve_llm_call():
    """Wire the production LLM call. Failure-isolated: returns None
    on any wiring failure so the daemon defers the run."""
    try:
        from app.llm_factory import create_specialist_llm
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "inquiry scheduler: llm_factory import failed: %s", exc,
        )
        return None
    try:
        llm = create_specialist_llm(role="research", max_tokens=4096)
    except Exception as exc:  # noqa: BLE001
        logger.warning("inquiry scheduler: LLM construction failed: %s", exc)
        return None

    def call(system: str, user: str) -> str:
        # crewai LLM objects expose .call(messages); we adapt.
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            response = llm.call(messages=messages)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"LLM call failed: {exc}") from exc
        if isinstance(response, str):
            return response
        if isinstance(response, dict) and "content" in response:
            return str(response["content"])
        return str(response)

    return call


def _driver() -> None:
    if _stop_event.wait(_WARMUP_S):
        return
    while not _stop_event.is_set():
        try:
            if _due_to_run():
                logger.info("inquiry scheduler: due — running one pass")
                from app.subia.inquiry.idle_registration import run_once
                llm_call = _resolve_llm_call()
                if llm_call is None:
                    logger.info(
                        "inquiry scheduler: LLM unavailable; deferring",
                    )
                else:
                    result = run_once(llm_call=llm_call)
                    logger.info(
                        "inquiry scheduler: pass result=%s slug=%r",
                        result.status, result.question_slug,
                    )
        except Exception:
            logger.debug("inquiry scheduler: pass raised", exc_info=True)
        if _stop_event.wait(_POLL_INTERVAL_S):
            return


def start() -> None:
    """Idempotent daemon launch. Safe under watchdog re-spawn."""
    global _driver_started
    if not _enabled():
        logger.info("inquiry scheduler: disabled via INQUIRY_PASS_ENABLED")
        return
    with _driver_lock:
        if _is_running():
            return
        if _driver_started:
            logger.warning(
                "inquiry scheduler: previous daemon thread is dead, re-spawning",
            )
        _stop_event.clear()
        thread = threading.Thread(
            target=_driver, name=_DAEMON_THREAD_NAME, daemon=True,
        )
        thread.start()
        _driver_started = True
        logger.info(
            "inquiry scheduler: daemon started "
            "(warm-up=%ds, poll=%ds, min-interval=%dd)",
            _WARMUP_S, _POLL_INTERVAL_S, _min_interval_days(),
        )


def stop() -> None:
    """Signal the daemon to exit. Mostly used in tests."""
    _stop_event.set()


# Eager start at import time. The warm-up + thread isolation make this
# safe even when the surrounding process is still booting. The boot
# trigger is the import line in app/subia/__init__.py — wherever the
# kernel loads, the inquiry daemon comes up alongside.
start()
