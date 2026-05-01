"""Tests for app.companion.config — CompanionConfig dataclass + clamp + load."""

from unittest.mock import patch

import pytest

from app.companion.config import (
    DEFAULT_DAILY_BUDGET_USD,
    MAX_DAILY_BUDGET_USD,
    MIN_DAILY_BUDGET_USD,
    CompanionConfig,
    load,
)


def test_defaults():
    cfg = CompanionConfig()
    assert cfg.enabled is True
    assert cfg.daily_budget_usd == DEFAULT_DAILY_BUDGET_USD
    assert cfg.seed_prompt is None
    assert cfg.sources == []


def test_clamp_high_budget():
    cfg = CompanionConfig(daily_budget_usd=9999.0).clamp()
    assert cfg.daily_budget_usd == MAX_DAILY_BUDGET_USD


def test_clamp_negative_budget():
    cfg = CompanionConfig(daily_budget_usd=-5.0).clamp()
    assert cfg.daily_budget_usd == MIN_DAILY_BUDGET_USD


def test_clamp_thresholds_in_range():
    cfg = CompanionConfig(novelty_threshold=2.0, surface_threshold=0.0).clamp()
    assert 0.3 <= cfg.novelty_threshold <= 0.95
    assert 0.5 <= cfg.surface_threshold <= 0.95


def test_clamp_quiet_hours_bound():
    cfg = CompanionConfig(quiet_hours_start=99, quiet_hours_end=-3).clamp()
    assert 0 <= cfg.quiet_hours_start <= 23
    assert 0 <= cfg.quiet_hours_end <= 23


def test_quiet_hours_simple_window():
    cfg = CompanionConfig(quiet_hours_start=2, quiet_hours_end=6).clamp()
    assert cfg.is_quiet_hour(3) is True
    assert cfg.is_quiet_hour(2) is True
    assert cfg.is_quiet_hour(6) is False
    assert cfg.is_quiet_hour(7) is False


def test_quiet_hours_wraps_midnight():
    cfg = CompanionConfig(quiet_hours_start=22, quiet_hours_end=6).clamp()
    assert cfg.is_quiet_hour(23) is True
    assert cfg.is_quiet_hour(2) is True
    assert cfg.is_quiet_hour(6) is False
    assert cfg.is_quiet_hour(12) is False


def test_quiet_hours_disabled_when_equal():
    cfg = CompanionConfig(quiet_hours_start=5, quiet_hours_end=5).clamp()
    for h in range(24):
        assert cfg.is_quiet_hour(h) is False


def test_from_dict_drops_unknown_keys():
    raw = {
        "enabled": True,
        "daily_budget_usd": 0.5,
        "seed_prompt": "Estonian forests",
        "ghost_field": "ignored",
    }
    cfg = CompanionConfig.from_dict(raw)
    assert cfg.daily_budget_usd == 0.5
    assert cfg.seed_prompt == "Estonian forests"
    assert not hasattr(cfg, "ghost_field")


def test_from_dict_none_returns_defaults():
    cfg = CompanionConfig.from_dict(None)
    assert cfg.daily_budget_usd == DEFAULT_DAILY_BUDGET_USD
    assert cfg.enabled is True


def test_round_trip_dict():
    cfg = CompanionConfig(seed_prompt="x", daily_budget_usd=0.7).clamp()
    cfg2 = CompanionConfig.from_dict(cfg.to_dict())
    assert cfg2.seed_prompt == "x"
    assert cfg2.daily_budget_usd == pytest.approx(0.7)


def test_load_returns_none_when_project_missing():
    with patch("app.companion.config._get_project_by_id", lambda pid: None):
        assert load("does-not-exist") is None


def test_load_default_when_no_companion_key():
    with patch("app.companion.config._get_project_by_id",
               lambda pid: {"id": pid, "config_json": {}}):
        cfg = load("p1")
    assert cfg is not None
    assert cfg.daily_budget_usd == DEFAULT_DAILY_BUDGET_USD


def test_load_reads_seed_prompt():
    row = {
        "id": "p1",
        "config_json": {
            "companion": {
                "seed_prompt": "Estonian forests",
                "daily_budget_usd": 0.5,
            }
        },
    }
    with patch("app.companion.config._get_project_by_id", lambda pid: row):
        cfg = load("p1")
    assert cfg is not None
    assert cfg.seed_prompt == "Estonian forests"
    assert cfg.daily_budget_usd == 0.5


def test_load_handles_get_failure():
    def _broken(pid):
        raise RuntimeError("db down")

    with patch("app.companion.config._get_project_by_id", _broken):
        assert load("p1") is None
