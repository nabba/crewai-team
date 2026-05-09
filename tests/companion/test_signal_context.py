"""Tests for ``app.personality.signal_context`` (Phase B #5, 2026-05-09)."""
from __future__ import annotations

import pytest


def test_default_context_when_everything_fails(monkeypatch):
    from app.personality import signal_context

    def boom(*a, **k):
        raise RuntimeError("nope")
    monkeypatch.setattr(signal_context, "_mood_from_affect", boom)
    monkeypatch.setattr(signal_context, "_time_of_day", boom)
    monkeypatch.setattr(signal_context, "_top_interests", boom)

    ctx = signal_context.build_context()
    assert ctx.mood == "steady"
    assert ctx.time_of_day == "day"
    assert ctx.top_interests == []


def test_mood_quadrants(monkeypatch):
    from app.personality import signal_context

    class _S:
        valence = 0.0
        arousal = 0.0
    state = _S()
    # default → steady
    monkeypatch.setattr("app.affect.core.latest_affect", lambda: state)
    assert signal_context._mood_from_affect() == "steady"

    state.valence, state.arousal = 0.5, 0.7
    assert signal_context._mood_from_affect() == "energized"

    state.valence, state.arousal = -0.5, 0.7
    assert signal_context._mood_from_affect() == "agitated"

    state.valence, state.arousal = -0.5, 0.3
    assert signal_context._mood_from_affect() == "tired"


def test_time_of_day_segments(monkeypatch):
    from app.personality import signal_context
    import datetime as _dt

    class _Frozen:
        @classmethod
        def now(cls, tz=None):
            return _dt.datetime(2026, 5, 9, cls._hour, 0, 0)

    for hour, expected in [
        (8, "morning"), (14, "afternoon"), (19, "evening"), (1, "night"),
    ]:
        _Frozen._hour = hour
        monkeypatch.setattr(signal_context, "datetime", _Frozen)
        # Disable the SubIA circadian helper path so we hit the fallback.
        monkeypatch.setitem(__import__("sys").modules, "app.subia.temporal.circadian", None)
        assert signal_context._time_of_day() == expected


def test_top_interests_reads_profile(monkeypatch):
    from app.personality import signal_context

    monkeypatch.setattr(
        "app.companion.interest_model.current_profile",
        lambda: {"topics": [
            {"name": "forest carbon", "score": 1.0},
            {"name": "mes-team", "score": 0.8},
            {"name": "kaicart", "score": 0.6},
            {"name": "plg", "score": 0.4},
        ]},
    )
    out = signal_context._top_interests(n=3)
    assert out == ["forest carbon", "mes-team", "kaicart"]


def test_to_prompt_line():
    from app.personality.signal_context import ToneContext
    ctx = ToneContext(mood="energized", time_of_day="morning",
                      top_interests=["forest", "kaicart"])
    line = ctx.to_prompt_line()
    assert "morning" in line
    assert "energized" in line
    assert "forest" in line
