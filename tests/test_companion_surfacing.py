"""Tests for app.companion.surfacing — threshold + cooldown + state event."""

import time
from pathlib import Path
from unittest.mock import patch

import pytest

from app.companion import events as _ev
from app.companion import surfacing as _surf
from app.companion.config import CompanionConfig
from app.companion.idea_store import IdeaRecord, IdeaState


@pytest.fixture
def tmp_events_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_ev, "_EVENTS_DIR", tmp_path)
    return tmp_path


def _idea(*, novelty=0.9, quality=0.9, text="A solid idea text body."):
    return IdeaRecord(
        idea_id="i_test", workspace_id="ws-1",
        text=text, state=IdeaState.CONVERGED,
        novelty=novelty, quality=quality, transferability=0.5,
    )


def _cfg(*, novelty_t=0.7, surface_t=0.7, seed="forests"):
    return CompanionConfig(
        seed_prompt=seed,
        novelty_threshold=novelty_t,
        surface_threshold=surface_t,
    ).clamp()


# ── should_surface ──────────────────────────────────────────────────────────

def test_should_surface_eligible(tmp_events_dir):
    d = _surf.should_surface(_idea(), _cfg())
    assert d.eligible is True
    assert d.reason == "ok"


def test_should_surface_below_novelty(tmp_events_dir):
    d = _surf.should_surface(_idea(novelty=0.4), _cfg())
    assert d.eligible is False
    assert d.reason == "below_novelty"


def test_should_surface_below_quality(tmp_events_dir):
    d = _surf.should_surface(_idea(quality=0.4), _cfg())
    assert d.eligible is False
    assert d.reason == "below_quality"


def test_should_surface_no_text(tmp_events_dir):
    d = _surf.should_surface(_idea(text="   "), _cfg())
    assert d.eligible is False
    assert d.reason == "no_text"


def test_should_surface_blocked_by_cooldown(tmp_events_dir):
    # Pre-existing surface 1 hour ago.
    _ev.append(_ev.Event(workspace_id="ws-1", idea_id="other",
                          type=_ev.EventType.SURFACED,
                          ts=time.time() - 3600))
    d = _surf.should_surface(_idea(), _cfg())
    assert d.eligible is False
    assert d.reason == "cooldown"


def test_cooldown_clears_after_window(tmp_events_dir):
    # Old surface — outside cooldown window.
    old = time.time() - (_surf.SURFACE_COOLDOWN_S + 1000)
    _ev.append(_ev.Event(workspace_id="ws-1", idea_id="other",
                          type=_ev.EventType.SURFACED, ts=old))
    d = _surf.should_surface(_idea(), _cfg())
    assert d.eligible is True


# ── surface ─────────────────────────────────────────────────────────────────

def test_surface_writes_event_and_calls_send(tmp_events_dir):
    sent_messages: list[tuple[str, str]] = []

    def _capture(text, ws):
        sent_messages.append((text, ws))
        return True

    with patch("app.companion.surfacing._send_signal", _capture):
        ok = _surf.surface(_idea(), _cfg())

    assert ok is True
    assert len(sent_messages) == 1
    text, ws = sent_messages[0]
    assert ws == "ws-1"
    assert "Companion" in text
    assert "forests" in text

    events = _ev.read_for_idea("ws-1", "i_test")
    assert len(events) == 1
    assert events[0].type == _ev.EventType.SURFACED
    assert events[0].payload["signal_sent"] is True


def test_surface_records_event_even_when_send_fails(tmp_events_dir):
    with patch("app.companion.surfacing._send_signal", lambda t, w: False):
        ok = _surf.surface(_idea(), _cfg())
    assert ok is False
    events = _ev.read_for_idea("ws-1", "i_test")
    assert len(events) == 1
    assert events[0].payload["signal_sent"] is False


def test_surface_records_event_when_send_raises(tmp_events_dir):
    def _broken(t, w):
        raise RuntimeError("network down")

    with patch("app.companion.surfacing._send_signal", _broken):
        ok = _surf.surface(_idea(), _cfg())
    assert ok is False
    events = _ev.read_for_idea("ws-1", "i_test")
    assert len(events) == 1
    assert "network down" in events[0].payload.get("send_error", "")


def test_compose_card_shape(tmp_events_dir):
    text = _surf.compose_card(_idea(text="The idea body"), _cfg())
    assert "Companion" in text
    assert "novelty 0.90" in text
    assert "quality 0.90" in text
    assert "The idea body" in text
    assert "Reply Y" in text


def test_compose_card_truncates_long_text(tmp_events_dir):
    long = "x" * 4000
    text = _surf.compose_card(_idea(text=long), _cfg())
    assert len(text) < len(long) + 500
    assert text.count("...") >= 1
