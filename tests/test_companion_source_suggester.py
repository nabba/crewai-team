"""Tests for app.companion.source_suggester — LLM proposals + JSON parsing."""

from unittest.mock import patch

import pytest

from app.companion import source_suggester as _ss
from app.companion.config import CompanionConfig


def _stub_config(seed):
    return CompanionConfig(seed_prompt=seed).clamp()


def test_propose_no_seed_returns_empty():
    with patch("app.companion.config.load",
               lambda ws: _stub_config(None)):
        assert _ss.propose("ws-1") == []


def test_propose_workspace_unknown_returns_empty():
    with patch("app.companion.config.load", lambda ws: None):
        assert _ss.propose("ws-1") == []


def test_propose_parses_json_array():
    seed_cfg = _stub_config("Estonian forests")
    raw = """[
        {"type": "web_search", "config": {"query": "Estonian forest ecology 2025"}, "reason": "current research"},
        {"type": "web_search", "config": {"query": "boreal forest carbon"}, "reason": "carbon angle"}
    ]"""
    with patch("app.companion.config.load", lambda ws: seed_cfg), \
         patch("app.companion.source_suggester._invoke_suggester",
               lambda p: raw):
        out = _ss.propose("ws-1")
    assert len(out) == 2
    assert out[0]["type"] == "web_search"
    assert "ecology" in out[0]["config"]["query"]


def test_propose_strips_markdown_fences():
    seed_cfg = _stub_config("forests")
    raw = """```json
    [{"type": "web_search", "config": {"query": "x"}, "reason": "r"}]
    ```"""
    with patch("app.companion.config.load", lambda ws: seed_cfg), \
         patch("app.companion.source_suggester._invoke_suggester",
               lambda p: raw):
        out = _ss.propose("ws-1")
    assert len(out) == 1


def test_propose_rejects_non_array_response():
    seed_cfg = _stub_config("forests")
    raw = "I'm sorry, I can't help with that."
    with patch("app.companion.config.load", lambda ws: seed_cfg), \
         patch("app.companion.source_suggester._invoke_suggester",
               lambda p: raw):
        out = _ss.propose("ws-1")
    assert out == []


def test_propose_filters_unknown_types():
    seed_cfg = _stub_config("forests")
    raw = """[
        {"type": "web_search", "config": {"query": "x"}, "reason": "r"},
        {"type": "telepathy", "config": {"channel": 7}, "reason": "r"}
    ]"""
    with patch("app.companion.config.load", lambda ws: seed_cfg), \
         patch("app.companion.source_suggester._invoke_suggester",
               lambda p: raw):
        out = _ss.propose("ws-1")
    assert len(out) == 1
    assert out[0]["type"] == "web_search"


def test_propose_caps_at_max_count():
    seed_cfg = _stub_config("forests")
    raw = "[" + ", ".join(
        '{"type": "web_search", "config": {"query": "q' + str(i) + '"}, "reason": "r"}'
        for i in range(20)
    ) + "]"
    with patch("app.companion.config.load", lambda ws: seed_cfg), \
         patch("app.companion.source_suggester._invoke_suggester",
               lambda p: raw):
        out = _ss.propose("ws-1", max_count=3)
    assert len(out) == 3


def test_propose_handles_llm_failure():
    seed_cfg = _stub_config("forests")
    def _broken(p):
        raise RuntimeError("LLM down")

    with patch("app.companion.config.load", lambda ws: seed_cfg), \
         patch("app.companion.source_suggester._invoke_suggester", _broken):
        out = _ss.propose("ws-1")
    assert out == []


def test_propose_skips_malformed_items():
    seed_cfg = _stub_config("forests")
    raw = """[
        {"type": "web_search", "config": {"query": "good"}, "reason": "r"},
        "not an object",
        {"type": "web_search", "config": "not a dict", "reason": "r"},
        {"type": "", "config": {"query": "blank type"}, "reason": "r"}
    ]"""
    with patch("app.companion.config.load", lambda ws: seed_cfg), \
         patch("app.companion.source_suggester._invoke_suggester",
               lambda p: raw):
        out = _ss.propose("ws-1")
    assert len(out) == 1
    assert out[0]["config"]["query"] == "good"
