"""Smoke tests for app.companion.loop — idle-job registration shape + tick safety."""

from pathlib import Path
from unittest.mock import patch

import pytest

from app.companion import loop as _loop
from app.companion import state as _state


@pytest.fixture
def tmp_state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_state, "_STATE_DIR", tmp_path)
    return tmp_path


def test_get_idle_jobs_returns_one_medium_job():
    jobs = _loop.get_idle_jobs()
    assert len(jobs) == 1
    name, fn, weight = jobs[0]
    assert name == "companion-tick"
    assert callable(fn)

    from app.idle_scheduler import JobWeight
    assert weight == JobWeight.MEDIUM


def test_idle_scheduler_default_jobs_includes_companion():
    """Verifies the wiring at idle_scheduler._default_jobs() picks us up."""
    from app import idle_scheduler

    jobs = idle_scheduler._default_jobs()
    names = [j[0] for j in jobs]
    assert "companion-tick" in names


def test_tick_with_no_workspaces_is_noop(tmp_state_dir):
    """No projects → no tick, no exception, no state written."""
    with patch("app.companion.scheduler._list_projects", lambda: []):
        _loop.companion_tick()  # must not raise

    assert list(tmp_state_dir.iterdir()) == []


def test_tick_records_state_for_selected_workspace(tmp_state_dir):
    rows = [{
        "id": "a",
        "config_json": {"companion": {"enabled": True, "daily_budget_usd": 1.0}},
    }]
    with patch("app.companion.scheduler._list_projects", lambda: rows):
        _loop.companion_tick()

    s = _state.load("a")
    assert s.cycles_total == 1
    assert s.last_tick_at > 0


def test_tick_select_next_failure_does_not_raise(tmp_state_dir):
    """Defensive: a flaky DB must not crash the idle thread."""
    def _broken():
        raise RuntimeError("db down")

    with patch("app.companion.scheduler.select_next", lambda **_: _broken()):
        _loop.companion_tick()  # must not raise
