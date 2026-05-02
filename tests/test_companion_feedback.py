"""Tests for app.companion.feedback — thumbs / comment recording + summary."""

from pathlib import Path

import pytest

from app.companion import events as _ev
from app.companion import feedback as _fb


@pytest.fixture
def tmp_events_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_ev, "_EVENTS_DIR", tmp_path)
    return tmp_path


def test_record_writes_feedback_event(tmp_events_dir):
    eid = _fb.record("i1", "ws-1", polarity=_fb.Polarity.UP,
                      comment="great", source=_fb.Source.REACT)
    events = _ev.read_for_idea("ws-1", "i1")
    assert len(events) == 1
    e = events[0]
    assert e.type == _ev.EventType.FEEDBACK
    assert e.payload["polarity"] == "up"
    assert e.payload["comment"] == "great"
    assert e.payload["source"] == "react"
    assert e.event_id == eid


def test_record_clamps_long_comments(tmp_events_dir):
    long = "x" * 5000
    _fb.record("i1", "ws-1", polarity=_fb.Polarity.DOWN, comment=long)
    e = _ev.read_for_idea("ws-1", "i1")[0]
    assert len(e.payload["comment"]) <= 2000


def test_for_idea_filters_to_feedback_only(tmp_events_dir):
    # mix of feedback + surfaced events
    _ev.append(_ev.Event(workspace_id="ws-1", idea_id="i1",
                          type=_ev.EventType.SURFACED))
    _fb.record("i1", "ws-1", polarity=_fb.Polarity.UP)
    _fb.record("i1", "ws-1", polarity=_fb.Polarity.DOWN, comment="meh")
    _ev.append(_ev.Event(workspace_id="ws-1", idea_id="i1",
                          type=_ev.EventType.ARCHIVED))

    out = _fb.for_idea("ws-1", "i1")
    assert len(out) == 2
    assert all(e.type == _ev.EventType.FEEDBACK for e in out)


def test_summary_aggregates_polarity(tmp_events_dir):
    _fb.record("i1", "ws-1", polarity=_fb.Polarity.UP)
    _fb.record("i2", "ws-1", polarity=_fb.Polarity.UP)
    _fb.record("i3", "ws-1", polarity=_fb.Polarity.DOWN, comment="bad")
    _fb.record("i4", "ws-1", polarity=_fb.Polarity.DOWN)

    s = _fb.summary("ws-1")
    assert s["up"] == 2
    assert s["down"] == 2
    assert s["with_comment"] == 1
    assert "i3" in s["recent_negative_idea_ids"]
    assert "i4" in s["recent_negative_idea_ids"]


def test_summary_recent_negative_caps_at_5(tmp_events_dir):
    for i in range(10):
        _fb.record(f"i{i}", "ws-1", polarity=_fb.Polarity.DOWN)
    s = _fb.summary("ws-1")
    assert len(s["recent_negative_idea_ids"]) == 5


def test_summary_empty_workspace(tmp_events_dir):
    s = _fb.summary("ws-empty")
    assert s == {
        "up": 0, "down": 0, "with_comment": 0,
        "recent_negative_idea_ids": [],
        "recent_positive_idea_ids": [],
    }
