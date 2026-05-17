"""Tests for app/boot_state.py + the idle_scheduler boot-gate.

Background (2026-05-17): the gateway's /signal/inbound endpoint hit
7.6 s latency immediately after restart because the idle scheduler
fired 23 LIGHT jobs in the first 4 s of boot, contending with lifespan
asyncio setup, the Ollama preload, and DB pool initialization. Root
cause: idle_scheduler.is_idle() returned True at boot because
_last_task_end == 0. The fix is the explicit boot-complete signal in
app/boot_state.py, consumed by is_idle().
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestBootStatePrimitive:
    """The signal itself — set-once, observable, blockable."""

    def setup_method(self) -> None:
        from app import boot_state
        boot_state._reset_for_tests()

    def teardown_method(self) -> None:
        from app import boot_state
        boot_state._reset_for_tests()

    def test_starts_unset(self):
        from app import boot_state
        assert boot_state.is_boot_complete() is False
        assert boot_state.boot_completed_at() is None

    def test_mark_sets_event_and_timestamp(self):
        from app import boot_state
        before = time.monotonic()
        boot_state.mark_boot_complete()
        after = time.monotonic()
        assert boot_state.is_boot_complete() is True
        ts = boot_state.boot_completed_at()
        assert ts is not None
        assert before <= ts <= after

    def test_mark_is_idempotent(self):
        from app import boot_state
        boot_state.mark_boot_complete()
        first_ts = boot_state.boot_completed_at()
        time.sleep(0.01)
        boot_state.mark_boot_complete()
        # Repeated call must not move the timestamp — that would break
        # any consumer that anchors its settle window to the first signal.
        assert boot_state.boot_completed_at() == first_ts

    def test_wait_returns_false_on_timeout(self):
        from app import boot_state
        # Tight timeout so the test stays fast.
        assert boot_state.wait_for_boot_complete(timeout=0.05) is False

    def test_wait_returns_true_after_mark(self):
        from app import boot_state
        import threading

        def _signal_after_delay() -> None:
            time.sleep(0.05)
            boot_state.mark_boot_complete()

        t = threading.Thread(target=_signal_after_delay, daemon=True)
        t.start()
        assert boot_state.wait_for_boot_complete(timeout=2.0) is True
        t.join(timeout=1.0)


class TestIdleSchedulerBootGate:
    """is_idle() must return False during boot, True after the settle window."""

    def setup_method(self) -> None:
        from app import boot_state, idle_scheduler
        boot_state._reset_for_tests()
        # Reset idle_scheduler internal state. The module-level globals
        # _last_task_end and _active_tasks survive across tests in the
        # same process, so we reset them explicitly.
        idle_scheduler._last_task_end = 0.0
        idle_scheduler._active_tasks = 0
        idle_scheduler._boot_fallback_warned = False

    def teardown_method(self) -> None:
        from app import boot_state, idle_scheduler
        boot_state._reset_for_tests()
        idle_scheduler._last_task_end = 0.0
        idle_scheduler._active_tasks = 0
        idle_scheduler._boot_fallback_warned = False

    def test_is_idle_false_before_boot_complete(self):
        """The bug we're fixing: pre-patch this returned True at boot."""
        from app import idle_scheduler
        assert idle_scheduler.is_idle() is False

    def test_is_idle_false_during_settle_window(self, monkeypatch):
        """Right after mark_boot_complete, is_idle waits IDLE_DELAY_SECONDS
        before reporting idle — gives the system a brief quiet period
        for fire-and-forget asyncio tasks to finish settling."""
        from app import boot_state, idle_scheduler
        boot_state.mark_boot_complete()
        # Still inside the settle window — should be False.
        assert idle_scheduler.is_idle() is False

    def test_is_idle_true_after_boot_complete_and_settle(self, monkeypatch):
        """After IDLE_DELAY_SECONDS past mark_boot_complete, idle == True."""
        from app import boot_state, idle_scheduler
        boot_state.mark_boot_complete()
        # Move the boot-complete timestamp backwards by more than the
        # settle window so the gate opens without actually sleeping.
        boot_state._boot_completed_at = (
            time.monotonic() - idle_scheduler.IDLE_DELAY_SECONDS - 1
        )
        assert idle_scheduler.is_idle() is True

    def test_active_task_blocks_idle(self):
        """Even after boot, an in-flight user task keeps is_idle False."""
        from app import boot_state, idle_scheduler
        boot_state.mark_boot_complete()
        boot_state._boot_completed_at = (
            time.monotonic() - idle_scheduler.IDLE_DELAY_SECONDS - 1
        )
        idle_scheduler._active_tasks = 1
        assert idle_scheduler.is_idle() is False

    def test_last_task_end_overrides_boot_anchor(self):
        """Once a user task has run, IDLE_DELAY_SECONDS is measured from
        the task end, not from boot completion."""
        from app import boot_state, idle_scheduler
        boot_state.mark_boot_complete()
        boot_state._boot_completed_at = (
            time.monotonic() - idle_scheduler.IDLE_DELAY_SECONDS - 100
        )
        # User task ended very recently — must keep is_idle False.
        idle_scheduler._last_task_end = time.monotonic() - 1
        assert idle_scheduler.is_idle() is False


class TestIdleSchedulerLocalFallback:
    """If the lifespan never marks boot complete, the local fallback
    eventually unblocks idle work — with a loud warning."""

    def setup_method(self) -> None:
        from app import boot_state, idle_scheduler
        boot_state._reset_for_tests()
        idle_scheduler._last_task_end = 0.0
        idle_scheduler._active_tasks = 0
        idle_scheduler._boot_fallback_warned = False

    def teardown_method(self) -> None:
        from app import boot_state, idle_scheduler
        boot_state._reset_for_tests()
        idle_scheduler._last_task_end = 0.0
        idle_scheduler._active_tasks = 0
        idle_scheduler._boot_fallback_warned = False

    def test_fallback_blocks_until_threshold(self, monkeypatch):
        """_boot_ready returns False before the fallback threshold."""
        from app import idle_scheduler
        # Pretend the module was just imported.
        monkeypatch.setattr(
            idle_scheduler, "_module_import_time", time.monotonic(),
        )
        assert idle_scheduler._boot_ready() is False

    def test_fallback_fires_after_threshold(self, monkeypatch, caplog):
        """_boot_ready returns True past IDLE_BOOT_FALLBACK_S, with a
        single WARNING line."""
        from app import idle_scheduler
        # Pretend the module was imported well past the fallback window.
        monkeypatch.setattr(
            idle_scheduler, "_module_import_time",
            time.monotonic() - idle_scheduler.IDLE_BOOT_FALLBACK_S - 1,
        )
        import logging
        with caplog.at_level(logging.WARNING, logger="app.idle_scheduler"):
            assert idle_scheduler._boot_ready() is True
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "mark_boot_complete" in warnings[0].message

    def test_fallback_warning_fires_once(self, monkeypatch, caplog):
        """The fallback warning is logged exactly once, not on every call."""
        from app import idle_scheduler
        monkeypatch.setattr(
            idle_scheduler, "_module_import_time",
            time.monotonic() - idle_scheduler.IDLE_BOOT_FALLBACK_S - 1,
        )
        import logging
        with caplog.at_level(logging.WARNING, logger="app.idle_scheduler"):
            idle_scheduler._boot_ready()
            idle_scheduler._boot_ready()
            idle_scheduler._boot_ready()
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1

    def test_normal_boot_does_not_warn(self, monkeypatch, caplog):
        """If mark_boot_complete is called, the fallback warning never fires."""
        from app import boot_state, idle_scheduler
        monkeypatch.setattr(
            idle_scheduler, "_module_import_time", time.monotonic(),
        )
        boot_state.mark_boot_complete()
        import logging
        with caplog.at_level(logging.WARNING, logger="app.idle_scheduler"):
            idle_scheduler._boot_ready()
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 0
