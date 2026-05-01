"""Tests for app.companion.scheduler — fairness selector + temporal floor."""

import time
from pathlib import Path
from unittest.mock import patch

import pytest

from app.companion import budget as _budget
from app.companion import scheduler as _scheduler
from app.companion import state as _state
from app.companion.config import CompanionConfig


@pytest.fixture
def tmp_state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_state, "_STATE_DIR", tmp_path)
    return tmp_path


def _stub_projects(rows):
    return patch("app.companion.scheduler._list_projects", lambda: rows)


def _row(pid, *, enabled=True, seed=None, budget=1.0, qstart=2, qend=6):
    return {
        "id": pid,
        "config_json": {
            "companion": {
                "enabled": enabled,
                "seed_prompt": seed,
                "daily_budget_usd": budget,
                "quiet_hours_start": qstart,
                "quiet_hours_end": qend,
            }
        },
    }


def test_no_candidates_when_disabled(tmp_state_dir):
    with _stub_projects([_row("a", enabled=False)]):
        assert _scheduler.select_next(now_local_hour=12) is None


def test_starved_workspace_jumps_queue(tmp_state_dir):
    rows = [_row("a"), _row("b")]
    now = time.time()
    # 'a' was last ticked 24 h ago and has a HUGE vruntime; 'b' has tiny vruntime
    # but only 1 h since its last tick. Despite vruntime favouring 'b', the
    # 12 h temporal floor must surface 'a'.
    _state.save(_state.WorkspaceState(
        project_id="a", last_tick_at=now - 86400, vruntime_s=10000.0))
    _state.save(_state.WorkspaceState(
        project_id="b", last_tick_at=now - 3600, vruntime_s=1.0))
    with _stub_projects(rows):
        cand = _scheduler.select_next(now_unix=now, now_local_hour=12)
    assert cand is not None
    assert cand.project_id == "a"


def test_lowest_vruntime_wins_when_no_starvation(tmp_state_dir):
    rows = [_row("a"), _row("b")]
    now = time.time()
    _state.save(_state.WorkspaceState(
        project_id="a", last_tick_at=now - 60, vruntime_s=50.0))
    _state.save(_state.WorkspaceState(
        project_id="b", last_tick_at=now - 60, vruntime_s=10.0))
    with _stub_projects(rows):
        cand = _scheduler.select_next(now_unix=now, now_local_hour=12)
    assert cand is not None
    assert cand.project_id == "b"


def test_fresh_workspace_with_zero_last_tick_wins(tmp_state_dir):
    """A workspace that's never ticked counts as starved (last_tick_at == 0)."""
    rows = [_row("a"), _row("brand-new")]
    now = time.time()
    _state.save(_state.WorkspaceState(
        project_id="a", last_tick_at=now - 60, vruntime_s=1.0))
    # 'brand-new' has no saved state; load() will return default with last_tick_at=0
    with _stub_projects(rows):
        cand = _scheduler.select_next(now_unix=now, now_local_hour=12)
    assert cand is not None
    assert cand.project_id == "brand-new"


def test_quiet_hours_skip(tmp_state_dir):
    rows = [_row("a", qstart=2, qend=6)]
    with _stub_projects(rows):
        # 03:00 is within quiet hours → no candidate.
        assert _scheduler.select_next(now_local_hour=3) is None
        # 12:00 is outside → candidate returned.
        assert _scheduler.select_next(now_local_hour=12) is not None


def test_budget_exhausted_skip(tmp_state_dir):
    rows = [_row("a", budget=0.1)]
    _budget.charge("a", 0.5)
    with _stub_projects(rows):
        assert _scheduler.select_next(now_local_hour=12) is None


def test_record_tick_advances_vruntime(tmp_state_dir):
    _scheduler.record_tick("a", cycle_cost_s=10.0, weight=1.0)
    s = _state.load("a")
    assert s.vruntime_s == pytest.approx(10.0)
    assert s.last_tick_at > 0
    assert s.cycles_total == 1


def test_record_tick_weight_scales_vruntime(tmp_state_dir):
    _scheduler.record_tick("a", cycle_cost_s=10.0, weight=2.0)
    s = _state.load("a")
    # weight=2.0 → vruntime grows half as fast (cost / weight)
    assert s.vruntime_s == pytest.approx(5.0)


def test_record_tick_clamps_weight(tmp_state_dir):
    _scheduler.record_tick("a", cycle_cost_s=10.0, weight=99.0)
    s = _state.load("a")
    # weight clamped to WEIGHT_CEIL (2.0) → vruntime = 10/2 = 5
    assert s.vruntime_s == pytest.approx(5.0)


def test_select_returns_none_when_no_projects(tmp_state_dir):
    with _stub_projects([]):
        assert _scheduler.select_next(now_local_hour=12) is None


def test_skip_reason_recorded(tmp_state_dir):
    rows = [_row("a", qstart=0, qend=24)]  # always quiet
    with _stub_projects(rows):
        _scheduler.select_next(now_local_hour=10)
    s = _state.load("a")
    assert s.last_skip_reason == "quiet_hours"


def test_record_tick_clears_skip_reason(tmp_state_dir):
    s = _state.WorkspaceState(project_id="a", last_skip_reason="budget_exhausted")
    _state.save(s)
    _scheduler.record_tick("a", cycle_cost_s=1.0, weight=1.0)
    assert _state.load("a").last_skip_reason is None
