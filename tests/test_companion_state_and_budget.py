"""Tests for app.companion.state + budget — sidecar JSON I/O on a tmp dir."""

from pathlib import Path

import pytest

from app.companion import budget as _budget
from app.companion import state as _state
from app.companion.config import CompanionConfig


@pytest.fixture
def tmp_state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_state, "_STATE_DIR", tmp_path)
    return tmp_path


def test_load_missing_returns_default(tmp_state_dir):
    s = _state.load("ws-1")
    assert s.project_id == "ws-1"
    assert s.vruntime_s == 0.0
    assert s.daily_cost_usd == 0.0
    assert s.cycles_total == 0


def test_save_and_load_round_trip(tmp_state_dir):
    s = _state.WorkspaceState(
        project_id="ws-1", vruntime_s=12.5, last_tick_at=1234.0,
        cycles_total=3, last_skip_reason="quiet_hours",
    )
    _state.save(s)
    s2 = _state.load("ws-1")
    assert s2.vruntime_s == 12.5
    assert s2.last_tick_at == 1234.0
    assert s2.cycles_total == 3
    assert s2.last_skip_reason == "quiet_hours"


def test_path_sanitises_project_id(tmp_state_dir):
    s = _state.WorkspaceState(project_id="../../etc/passwd")
    _state.save(s)
    # File lands inside tmp_state_dir, no traversal.
    written = list(tmp_state_dir.iterdir())
    assert len(written) == 1
    assert written[0].parent == tmp_state_dir


def test_path_sanitises_empty_id(tmp_state_dir):
    s = _state.WorkspaceState(project_id="!!!")
    _state.save(s)
    assert (tmp_state_dir / "default.json").exists()


def test_utc_day_key_format():
    key = _state.utc_day_key(0)  # 1970-01-01
    assert key == "1970-01-01"


def test_atomic_write_no_temp_files_left(tmp_state_dir):
    s = _state.WorkspaceState(project_id="ws-1", vruntime_s=1.0)
    _state.save(s)
    files = list(tmp_state_dir.iterdir())
    assert len(files) == 1
    assert not files[0].name.startswith(".tmp")


def test_budget_remaining_full_when_no_state(tmp_state_dir):
    cfg = CompanionConfig(daily_budget_usd=1.0).clamp()
    assert _budget.remaining_usd("ws-1", cfg) == 1.0


def test_budget_charge_reduces_remaining(tmp_state_dir):
    cfg = CompanionConfig(daily_budget_usd=1.0).clamp()
    _budget.charge("ws-1", 0.3)
    assert _budget.remaining_usd("ws-1", cfg) == pytest.approx(0.7)


def test_budget_resets_on_new_day(tmp_state_dir):
    cfg = CompanionConfig(daily_budget_usd=1.0).clamp()
    _budget.charge("ws-1", 0.5)
    s = _state.load("ws-1")
    s.cost_day_key = "1900-01-01"
    _state.save(s)
    new_total = _budget.charge("ws-1", 0.2)
    assert new_total == pytest.approx(0.2)
    assert _budget.remaining_usd("ws-1", cfg) == pytest.approx(0.8)


def test_budget_exhausted(tmp_state_dir):
    cfg = CompanionConfig(daily_budget_usd=0.5).clamp()
    _budget.charge("ws-1", 0.6)
    assert _budget.is_exhausted("ws-1", cfg) is True


def test_budget_charge_negative_ignored(tmp_state_dir):
    _budget.charge("ws-1", -1.0)
    s = _state.load("ws-1")
    assert s.daily_cost_usd == 0.0


def test_budget_charge_none_ignored(tmp_state_dir):
    result = _budget.charge("ws-1", None)
    assert result == 0.0


def test_budget_isolated_per_workspace(tmp_state_dir):
    cfg = CompanionConfig(daily_budget_usd=1.0).clamp()
    _budget.charge("ws-1", 0.6)
    assert _budget.remaining_usd("ws-1", cfg) == pytest.approx(0.4)
    assert _budget.remaining_usd("ws-2", cfg) == pytest.approx(1.0)
