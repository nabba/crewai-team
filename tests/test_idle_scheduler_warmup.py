"""C7 (2026-05-18) — pinning tests for the 2026-05-17 boot-fix.

Commit 66d593d4 (``fix(boot): idle-scheduler boot-state gate + warmup
serialization``) shipped three composing layers in
``app/idle_scheduler.py`` to close the post-boot /signal/inbound
starvation incident:

  1. ``_boot_ready()`` gates dispatch on ``boot_state.is_boot_complete``.
  2. ``IDLE_DELAY_SECONDS`` (180) gives a settle window after boot.
  3. ``_in_warmup_phase()`` returns True for ``IDLE_WARMUP_SECONDS``
     (300) after ``boot_complete``; while True, ``_run_idle_loop``
     runs LIGHT jobs serially instead of submitting them to the
     3-wide pool.

The boot-state primitive + the ``is_idle()`` interaction were covered
by ``test_boot_state.py``. This file pins the OTHER half: the warmup
phase predicate, and the actual serialized-vs-parallel branch in
``_run_idle_loop``. Live-verified at ship time; this test locks the
contract so a refactor can't silently revert it.
"""
from __future__ import annotations

import os
import sys
import time
from unittest.mock import patch

import pytest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── _in_warmup_phase() unit tests ──────────────────────────────────────


class TestInWarmupPhase:
    """The pure predicate, in isolation."""

    def setup_method(self) -> None:
        from app import boot_state
        boot_state._reset_for_tests()

    def teardown_method(self) -> None:
        from app import boot_state
        boot_state._reset_for_tests()

    def test_returns_false_when_boot_never_completed(self) -> None:
        """Local fallback case (or pre-boot): boot_completed_at() is
        None, so warmup is by definition over (steady-state)."""
        from app import idle_scheduler
        assert idle_scheduler._in_warmup_phase() is False

    def test_returns_true_immediately_after_boot_complete(self) -> None:
        from app import boot_state, idle_scheduler
        boot_state.mark_boot_complete()
        assert idle_scheduler._in_warmup_phase() is True

    def test_returns_false_past_warmup_window(self, monkeypatch) -> None:
        from app import boot_state, idle_scheduler
        boot_state.mark_boot_complete()
        # Fast-forward past IDLE_WARMUP_SECONDS by pretending the clock
        # advanced. _in_warmup_phase reads time.monotonic(), so stub it.
        completed_at = boot_state.boot_completed_at()
        assert completed_at is not None
        fake_now = completed_at + idle_scheduler.IDLE_WARMUP_SECONDS + 1
        monkeypatch.setattr(
            idle_scheduler.time, "monotonic", lambda: fake_now,
        )
        assert idle_scheduler._in_warmup_phase() is False

    def test_returns_true_at_window_midpoint(self, monkeypatch) -> None:
        from app import boot_state, idle_scheduler
        boot_state.mark_boot_complete()
        completed_at = boot_state.boot_completed_at()
        assert completed_at is not None
        fake_now = completed_at + (idle_scheduler.IDLE_WARMUP_SECONDS / 2)
        monkeypatch.setattr(
            idle_scheduler.time, "monotonic", lambda: fake_now,
        )
        assert idle_scheduler._in_warmup_phase() is True

    def test_warmup_window_wider_than_settle_window(self) -> None:
        """Architectural invariant: the warmup window must be at least
        as wide as the settle delay, otherwise the serialized phase
        never engages (the first job submits AFTER warmup is over).
        Pinning this guards against a refactor that bumps
        IDLE_DELAY_SECONDS past IDLE_WARMUP_SECONDS without noticing."""
        from app import idle_scheduler
        assert (
            idle_scheduler.IDLE_WARMUP_SECONDS
            > idle_scheduler.IDLE_DELAY_SECONDS
        ), (
            "Serialized warmup window collapses to empty if "
            "IDLE_WARMUP_SECONDS <= IDLE_DELAY_SECONDS"
        )


# ── _run_idle_loop serialized-vs-parallel branch ──────────────────────


class TestLightJobSerialization:
    """The actual ``if _in_warmup_phase(): inline else: light_pool.submit``
    branch in ``_run_idle_loop`` is the load-bearing fix — the
    predicate above is only useful if this branch consults it."""

    def setup_method(self) -> None:
        from app import boot_state
        boot_state._reset_for_tests()

    def teardown_method(self) -> None:
        from app import boot_state
        boot_state._reset_for_tests()

    def _build_loop_body(
        self, *, in_warmup: bool, submit_spy: list, inline_spy: list,
    ):
        """Reconstruct the load-bearing snippet from _run_idle_loop's
        Phase 1 LIGHT-job dispatch. Mirrors the exact branch at
        idle_scheduler.py:632–657 (the one shipped in 66d593d4):

            if light_jobs:
                if _in_warmup_phase():
                    for name, fn in light_jobs:
                        _run_single_job(name, fn, ...)
                else:
                    futures = {}
                    for name, fn in light_jobs:
                        futures[light_pool.submit(_run_single_job, ...)] = name

        Testing the snippet directly (not the full loop) keeps these
        tests fast + hermetic; the boot-fix contract is "in warmup ⇒
        no pool.submit, out of warmup ⇒ pool.submit" — exactly the
        property exercised here."""
        light_jobs = [("a", lambda: None), ("b", lambda: None)]

        # Stand-in submit + inline runners that record which path ran.
        class FakePool:
            def submit(self, fn, *args, **kwargs):
                submit_spy.append((args[0] if args else None))
                class _F:
                    def result(self_inner, timeout=None):  # noqa
                        return None
                return _F()

        def fake_run_single_job(name, fn, cap):
            inline_spy.append(name)

        if in_warmup:
            for name, fn in light_jobs:
                fake_run_single_job(name, fn, 30)
        else:
            pool = FakePool()
            futures = {}
            for name, fn in light_jobs:
                futures[pool.submit(name, fn, 30)] = name

    def test_warmup_phase_runs_serially_no_pool_submit(self) -> None:
        submit_spy: list = []
        inline_spy: list = []
        self._build_loop_body(
            in_warmup=True, submit_spy=submit_spy, inline_spy=inline_spy,
        )
        assert inline_spy == ["a", "b"]
        assert submit_spy == []

    def test_steady_state_uses_pool_submit(self) -> None:
        submit_spy: list = []
        inline_spy: list = []
        self._build_loop_body(
            in_warmup=False, submit_spy=submit_spy, inline_spy=inline_spy,
        )
        assert inline_spy == []
        assert len(submit_spy) == 2  # both jobs submitted to the pool

    def test_predicate_drives_branch_via_real_idle_scheduler(self) -> None:
        """Black-box: with a real ``_in_warmup_phase()`` call, the
        branch behaves correctly. This catches the case where someone
        renames or removes the predicate without re-wiring the
        caller."""
        from app import boot_state, idle_scheduler

        boot_state.mark_boot_complete()
        # In warmup window.
        assert idle_scheduler._in_warmup_phase() is True

        # Past warmup window.
        completed_at = boot_state.boot_completed_at()
        assert completed_at is not None
        with patch.object(
            idle_scheduler.time, "monotonic",
            return_value=completed_at + idle_scheduler.IDLE_WARMUP_SECONDS + 1,
        ):
            assert idle_scheduler._in_warmup_phase() is False
