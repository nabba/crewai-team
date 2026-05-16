"""Tests for app.browse.idle_job — cadence + disabled short-circuit."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from app.browse import idle_job, store


def test_get_idle_jobs_returns_one_tuple() -> None:
    jobs = idle_job.get_idle_jobs()
    assert len(jobs) == 1
    name, fn, weight = jobs[0]
    assert name == "browse-tick"
    assert callable(fn)


def test_run_tick_disabled_is_silent(
    _reset_browse_state: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PRIVACY PIN: disabled → no marker file, no events dir."""
    monkeypatch.setenv("BROWSE_INGESTION_ENABLED", "false")
    idle_job.run_browse_tick()
    assert not (_reset_browse_state / ".last_pass_at").exists()
    assert not (_reset_browse_state / "events").exists()


def test_run_tick_creates_marker_when_due(_reset_browse_state: Path) -> None:
    idle_job.run_browse_tick()
    assert (_reset_browse_state / ".last_pass_at").exists()


def test_run_tick_respects_cadence(_reset_browse_state: Path) -> None:
    """Second call within 30 min should be a no-op (marker mtime unchanged)."""
    idle_job.run_browse_tick()
    marker = _reset_browse_state / ".last_pass_at"
    first_mtime = marker.stat().st_mtime

    # Sleep a tiny bit so mtime resolution can distinguish.
    time.sleep(0.05)
    idle_job.run_browse_tick()
    assert marker.stat().st_mtime == first_mtime


def test_run_tick_re_fires_after_cadence_elapsed(_reset_browse_state: Path) -> None:
    """Force the marker to look old; expect a re-run."""
    idle_job.run_browse_tick()
    marker = _reset_browse_state / ".last_pass_at"
    old = time.time() - 60 * 60 * 2  # 2 hours ago
    import os
    os.utime(marker, (old, old))

    idle_job.run_browse_tick()
    assert marker.stat().st_mtime >= time.time() - 5
