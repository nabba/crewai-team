"""Tests for app.subia.inquiry.scheduler."""

from __future__ import annotations

import os
import time
from pathlib import Path

from app.subia.inquiry import scheduler


def test_disabled_short_circuits_start(monkeypatch, caplog) -> None:
    """When INQUIRY_PASS_ENABLED=false, start() logs + returns without
    spawning. We can't reliably check thread liveness because the
    eager-import already spawned a thread; we verify the disabled
    branch fires by capturing the log line."""
    import logging
    monkeypatch.setenv("INQUIRY_PASS_ENABLED", "false")
    assert scheduler._enabled() is False
    with caplog.at_level(logging.INFO, logger="app.subia.inquiry.scheduler"):
        scheduler.start()
    assert any(
        "disabled via INQUIRY_PASS_ENABLED" in r.message
        for r in caplog.records
    )


def test_due_to_run_when_no_inquiries(tmp_path: Path) -> None:
    assert scheduler._due_to_run(inquiries_dir=tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.mkdir()
    assert scheduler._due_to_run(inquiries_dir=empty)


def test_not_due_when_recent_inquiry_exists(tmp_path: Path) -> None:
    d = tmp_path / "inquiries"
    d.mkdir()
    (d / "2026-05-10-x.md").write_text("essay")
    # mtime is now → less than 7 days ago.
    assert not scheduler._due_to_run(inquiries_dir=d)


def test_due_when_oldest_recent_is_past_min_interval(tmp_path: Path, monkeypatch) -> None:
    d = tmp_path / "inquiries"
    d.mkdir()
    f = d / "2026-04-01-x.md"
    f.write_text("essay")
    old = time.time() - 10 * 86400
    os.utime(f, (old, old))
    assert scheduler._due_to_run(inquiries_dir=d)


def test_min_interval_days_overridable(monkeypatch) -> None:
    monkeypatch.setenv("INQUIRY_MIN_INTERVAL_DAYS", "1")
    assert scheduler._min_interval_days() == 1
    monkeypatch.setenv("INQUIRY_MIN_INTERVAL_DAYS", "garbage")
    assert scheduler._min_interval_days() == 7


def test_resolve_llm_call_returns_none_on_factory_failure(monkeypatch) -> None:
    """When LLM construction fails, scheduler defers the run rather than crashing."""
    import app.subia.inquiry.scheduler as s

    def boom(*a, **kw):
        raise RuntimeError("no models configured")

    # Must patch llm_factory.create_specialist_llm where scheduler imports it.
    import app.llm_factory as llm_factory
    monkeypatch.setattr(llm_factory, "create_specialist_llm", boom)
    assert s._resolve_llm_call() is None


def test_stop_clears_running_event(monkeypatch) -> None:
    scheduler.stop()
    assert scheduler._stop_event.is_set()


def test_start_is_idempotent(monkeypatch) -> None:
    """Calling start() twice while a thread is alive is a no-op (no duplicate)."""
    monkeypatch.setenv("INQUIRY_PASS_ENABLED", "true")
    scheduler.stop()
    scheduler._driver_started = False
    # First start
    scheduler.start()
    # Second start — should NOT spawn a duplicate.
    n_first = sum(
        1 for t in scheduler.threading.enumerate()
        if t.name == scheduler._DAEMON_THREAD_NAME
    )
    scheduler.start()
    n_second = sum(
        1 for t in scheduler.threading.enumerate()
        if t.name == scheduler._DAEMON_THREAD_NAME
    )
    assert n_second == n_first
    scheduler.stop()
