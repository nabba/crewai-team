"""Tests for app.life_companion.routine_detector."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    from app.life_companion import _common
    monkeypatch.setattr(_common, "_STATE_DIR", tmp_path)

    sent: list[str] = []
    monkeypatch.setattr(_common, "send_signal_alert",
                        lambda body, **kw: sent.append(body) or True)
    monkeypatch.setattr(_common, "audit_event", lambda *a, **k: None)
    monkeypatch.setenv("LIFE_COMPANION_ENABLED", "true")
    monkeypatch.setenv("LIFE_COMPANION_ROUTINES_ENABLED", "true")
    monkeypatch.setattr(_common, "background_enabled", lambda: True)
    yield tmp_path, sent


def _make_episode(when: datetime, *, crew: str, preview: str) -> dict:
    return {
        "ts": when.astimezone(timezone.utc).isoformat(),
        "crew": crew,
        "agent_id": crew,
        "task_preview": preview,
        "success": True,
    }


def test_detector_finds_weekday_pattern(isolated, monkeypatch):
    tmp_path, sent = isolated
    from app.life_companion import routine_detector

    monkeypatch.setattr(routine_detector, "send_signal_alert",
                        lambda body, **kw: sent.append(body) or True)
    monkeypatch.setattr(routine_detector, "audit_event",
                        lambda *a, **k: None)

    # Synthesise 6 episodes — all on Friday afternoons (16-20 bucket),
    # all with crew=coder, spread across 6 different ISO weeks.
    episodes = []
    base = datetime(2026, 3, 6, 17, 0)  # 2026-03-06 was a Friday
    for week in range(6):
        ts = base + timedelta(weeks=week)
        episodes.append(_make_episode(ts, crew="coder", preview="PR review"))
    # Plus a few off-day distractors with the same crew so the
    # concentration check has a baseline.
    for tue_week in range(2):
        ts = datetime(2026, 3, 3, 11, 0) + timedelta(weeks=tue_week)
        episodes.append(_make_episode(ts, crew="coder", preview="random Tue work"))

    monkeypatch.setattr(routine_detector, "_read_episodes", lambda: episodes)

    routine_detector.run()
    # First detection should produce a "new routines" Signal alert.
    assert any("new routine" in body.lower() for body in sent)
    assert any("Fri" in body for body in sent)


def test_detector_no_pattern_no_alert(isolated, monkeypatch):
    tmp_path, sent = isolated
    from app.life_companion import routine_detector

    monkeypatch.setattr(routine_detector, "send_signal_alert",
                        lambda body, **kw: sent.append(body) or True)
    monkeypatch.setattr(routine_detector, "audit_event",
                        lambda *a, **k: None)

    # Random episodes scattered across days — no concentration.
    episodes = []
    for day in range(20):
        episodes.append(_make_episode(
            datetime(2026, 4, 1, 10, 0) + timedelta(days=day),
            crew=("coder" if day % 2 else "research"),
            preview="random work",
        ))
    monkeypatch.setattr(routine_detector, "_read_episodes", lambda: episodes)

    routine_detector.run()
    # No clean weekly pattern — should not crow about routines.
    assert not any("new routine" in body.lower() for body in sent)


def test_detector_dedups_existing_routines(isolated, monkeypatch):
    """A routine already in state shouldn't be re-alerted."""
    tmp_path, sent = isolated
    from app.life_companion import routine_detector

    monkeypatch.setattr(routine_detector, "send_signal_alert",
                        lambda body, **kw: sent.append(body) or True)
    monkeypatch.setattr(routine_detector, "audit_event",
                        lambda *a, **k: None)

    base = datetime(2026, 3, 6, 17, 0)
    episodes = [
        _make_episode(base + timedelta(weeks=w),
                      crew="coder", preview="PR review")
        for w in range(6)
    ]
    monkeypatch.setattr(routine_detector, "_read_episodes", lambda: episodes)

    routine_detector.run()
    n1 = len(sent)
    # Force a re-detection by clearing the cadence guard.
    from app.life_companion._common import read_state_json, write_state_json
    state = read_state_json("routines.json", {})
    state["last_detect_at"] = 0
    write_state_json("routines.json", state)

    routine_detector.run()
    # Same routines already in state → no NEW-routine alert.
    new_alerts = [b for b in sent[n1:] if "new routine" in b.lower()]
    assert new_alerts == []
