"""Tests for the idle_scheduler iterator-level TimeoutError catch.

PR 1 (2026-05-16). Before this PR, the light-job phase wrapped only
``future.result()`` in ``try/except Exception``. ``as_completed(...,
timeout=N)`` itself raises ``concurrent.futures.TimeoutError`` at the
*generator* level when at least one future hasn't completed within N
seconds — that exception escaped the for-loop and killed the
idle-scheduler daemon thread. One stuck job silently stopped ALL
background work until the next gateway restart.

The fix extracts the drain logic into ``_drain_futures_with_timeout``
which:
  1. Catches the iterator-level TimeoutError
  2. Cancels still-pending futures so the calling pool isn't permanently
     starved
  3. Logs the names of stuck jobs for operator visibility

These tests pin all three properties.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import pytest


def test_drain_normal_completion():
    """Happy path: all futures finish before timeout; drain returns cleanly."""
    from app.idle_scheduler import _drain_futures_with_timeout

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(lambda: 1): "a",
            pool.submit(lambda: 2): "b",
            pool.submit(lambda: 3): "c",
        }
        _drain_futures_with_timeout(futures, timeout_s=5.0)
        # All should be done with no exceptions left lying around
        for f in futures:
            assert f.done()


def test_drain_swallows_per_future_exceptions():
    """A future raising an exception must NOT propagate out of drain."""
    from app.idle_scheduler import _drain_futures_with_timeout

    def boom():
        raise RuntimeError("job failed")

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(boom): "broken",
            pool.submit(lambda: 1): "ok",
        }
        # Must not raise — both futures resolve, the broken one's
        # exception is swallowed.
        _drain_futures_with_timeout(futures, timeout_s=5.0)


def test_drain_catches_iterator_level_timeout():
    """A stuck future that never completes must NOT crash drain.

    This is the load-bearing assertion for the whole PR 1 fix — if
    this raises, the scheduler thread dies in production.
    """
    from app.idle_scheduler import _drain_futures_with_timeout

    stuck_event = __import__("threading").Event()

    def stuck():
        # Block until released — simulates a job that's hung forever
        stuck_event.wait(timeout=10.0)

    with ThreadPoolExecutor(max_workers=2) as pool:
        try:
            futures = {
                pool.submit(stuck): "stuck-job",
                pool.submit(lambda: 1): "fast-job",
            }
            t0 = time.monotonic()
            # If iterator-level TimeoutError escapes, this raises.
            _drain_futures_with_timeout(futures, timeout_s=0.5)
            elapsed = time.monotonic() - t0
            # Must return promptly (drain timeout + small overhead)
            assert elapsed < 2.0, f"drain took {elapsed}s, expected < 2s"
        finally:
            stuck_event.set()


def test_drain_cancels_pending_on_timeout():
    """On iterator timeout, drain must cancel still-pending futures."""
    from app.idle_scheduler import _drain_futures_with_timeout

    stuck_event = __import__("threading").Event()
    started_count = {"n": 0}

    def stuck():
        started_count["n"] += 1
        stuck_event.wait(timeout=10.0)

    # Use a 1-worker pool so the second submit() stays queued (not yet
    # running) — that's the future that drain should cancel.
    with ThreadPoolExecutor(max_workers=1) as pool:
        try:
            f1 = pool.submit(stuck)        # Will start running
            f2 = pool.submit(lambda: 99)   # Will stay queued
            futures = {f1: "stuck", f2: "queued"}

            _drain_futures_with_timeout(futures, timeout_s=0.5)

            # f1 is running and cannot be cancelled (already started),
            # but f2 must be cancelled (still in queue).
            assert f2.cancelled() or f2.done(), \
                "queued future should be cancelled when drain times out"
        finally:
            stuck_event.set()


def test_drain_logs_stuck_job_names(monkeypatch, caplog):
    """Operator visibility: drain must log the names of stuck jobs."""
    import logging
    from app.idle_scheduler import _drain_futures_with_timeout

    stuck_event = __import__("threading").Event()

    def stuck():
        stuck_event.wait(timeout=10.0)

    with ThreadPoolExecutor(max_workers=2) as pool:
        try:
            futures = {
                pool.submit(stuck): "stuck-job-alpha",
                pool.submit(lambda: 1): "fast-job",
            }
            with caplog.at_level(logging.WARNING, logger="app.idle_scheduler"):
                _drain_futures_with_timeout(futures, timeout_s=0.5)
            assert any(
                "stuck-job-alpha" in record.message
                for record in caplog.records
            ), "drain should log the name of the stuck job"
        finally:
            stuck_event.set()


def test_drain_empty_futures_is_noop():
    """drain on an empty dict should return immediately, not iterate at all."""
    from app.idle_scheduler import _drain_futures_with_timeout
    # No exception, no crash, no hang
    _drain_futures_with_timeout({}, timeout_s=1.0)
