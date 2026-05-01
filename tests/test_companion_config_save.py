"""Tests for app.companion.config.save — CP config_json read-modify-write."""

from unittest.mock import patch

import pytest

from app.companion.config import (
    CompanionConfig,
    DEFAULT_DAILY_BUDGET_USD,
    save,
)


def test_save_writes_companion_key_into_config_json():
    captured: list[tuple[str, dict]] = []

    def _save_full(pid, cfg_json):
        captured.append((pid, cfg_json))
        return True

    with patch("app.companion.config._get_full_config_json",
               lambda pid: {}), \
         patch("app.companion.config._save_full_config_json", _save_full):
        ok = save("ws-1",
                  CompanionConfig(seed_prompt="forests",
                                   daily_budget_usd=0.5))

    assert ok is True
    assert len(captured) == 1
    pid, written = captured[0]
    assert pid == "ws-1"
    assert "companion" in written
    assert written["companion"]["seed_prompt"] == "forests"
    assert written["companion"]["daily_budget_usd"] == 0.5


def test_save_preserves_other_top_level_keys():
    """A foreign top-level key in config_json must survive the merge."""
    captured: list[dict] = []

    with patch("app.companion.config._get_full_config_json",
               lambda pid: {"other_module": {"setting": True}}), \
         patch("app.companion.config._save_full_config_json",
               lambda pid, cfg: captured.append(cfg) or True):
        save("ws-1", CompanionConfig(seed_prompt="x"))

    assert len(captured) == 1
    written = captured[0]
    assert written["other_module"] == {"setting": True}
    assert "companion" in written


def test_save_clamps_out_of_range_values():
    captured: list[dict] = []

    with patch("app.companion.config._get_full_config_json",
               lambda pid: {}), \
         patch("app.companion.config._save_full_config_json",
               lambda pid, cfg: captured.append(cfg) or True):
        save("ws-1", CompanionConfig(daily_budget_usd=999_999.0))

    assert captured[0]["companion"]["daily_budget_usd"] == 100.0


def test_save_returns_false_when_project_missing():
    with patch("app.companion.config._get_full_config_json",
               lambda pid: None):
        ok = save("missing", CompanionConfig(seed_prompt="x"))
    assert ok is False


def test_save_returns_false_on_read_failure():
    def _broken(pid):
        raise RuntimeError("DB down")

    with patch("app.companion.config._get_full_config_json", _broken):
        ok = save("ws-1", CompanionConfig(seed_prompt="x"))
    assert ok is False


def test_save_returns_false_on_write_failure():
    def _broken(pid, cfg):
        raise RuntimeError("DB write failed")

    with patch("app.companion.config._get_full_config_json",
               lambda pid: {}), \
         patch("app.companion.config._save_full_config_json", _broken):
        ok = save("ws-1", CompanionConfig(seed_prompt="x"))
    assert ok is False


def test_save_handles_non_dict_existing_config_json():
    """Defensive: if config_json was somehow stored as a non-dict, treat as
    empty rather than crashing."""
    captured: list[dict] = []

    with patch("app.companion.config._get_full_config_json",
               lambda pid: "weird-string-blob"), \
         patch("app.companion.config._save_full_config_json",
               lambda pid, cfg: captured.append(cfg) or True):
        ok = save("ws-1", CompanionConfig(seed_prompt="x"))
    assert ok is True
    assert captured[0] == {"companion": captured[0]["companion"]}


def test_save_default_config_uses_default_budget():
    captured: list[dict] = []

    with patch("app.companion.config._get_full_config_json",
               lambda pid: {}), \
         patch("app.companion.config._save_full_config_json",
               lambda pid, cfg: captured.append(cfg) or True):
        save("ws-1", CompanionConfig())

    assert captured[0]["companion"]["daily_budget_usd"] == DEFAULT_DAILY_BUDGET_USD
