"""Tests for app.companion.events — append-only event log."""

from pathlib import Path

import pytest

from app.companion import events as _ev


@pytest.fixture
def tmp_events_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_ev, "_EVENTS_DIR", tmp_path)
    return tmp_path


def test_append_writes_jsonl(tmp_events_dir):
    e = _ev.Event(workspace_id="ws-1", idea_id="i1",
                   type=_ev.EventType.SURFACED,
                   payload={"signal_sent": True})
    eid = _ev.append(e)
    assert eid == e.event_id
    body = (tmp_events_dir / "ws-1.jsonl").read_text().strip()
    assert e.event_id in body
    assert "surfaced" in body


def test_event_id_auto_generated():
    e = _ev.Event(workspace_id="ws-1", idea_id="i1")
    assert e.event_id.startswith("ev_")


def test_read_all_returns_events_in_order(tmp_events_dir):
    for i, t in enumerate([_ev.EventType.SURFACED, _ev.EventType.FEEDBACK,
                            _ev.EventType.ARCHIVED]):
        _ev.append(_ev.Event(workspace_id="ws-1", idea_id=f"i{i}",
                              type=t, ts=float(i)))
    out = _ev.read_all("ws-1")
    assert len(out) == 3
    assert out[0].type == _ev.EventType.SURFACED
    assert out[2].type == _ev.EventType.ARCHIVED


def test_read_for_idea_filters(tmp_events_dir):
    _ev.append(_ev.Event(workspace_id="ws-1", idea_id="i1",
                          type=_ev.EventType.SURFACED))
    _ev.append(_ev.Event(workspace_id="ws-1", idea_id="i2",
                          type=_ev.EventType.SURFACED))
    _ev.append(_ev.Event(workspace_id="ws-1", idea_id="i1",
                          type=_ev.EventType.FEEDBACK,
                          payload={"polarity": "up"}))

    only_i1 = _ev.read_for_idea("ws-1", "i1")
    assert len(only_i1) == 2
    assert all(e.idea_id == "i1" for e in only_i1)


def test_read_unknown_workspace_returns_empty(tmp_events_dir):
    assert _ev.read_all("never") == []
    assert _ev.read_for_idea("never", "i1") == []


def test_path_sanitises_workspace_id(tmp_events_dir):
    _ev.append(_ev.Event(workspace_id="../../etc/passwd", idea_id="i1"))
    files = list(tmp_events_dir.iterdir())
    assert len(files) == 1
    assert files[0].parent == tmp_events_dir


def test_event_type_round_trip(tmp_events_dir):
    _ev.append(_ev.Event(workspace_id="ws-1", idea_id="i1",
                          type=_ev.EventType.APPROVED,
                          payload={"by": "user"}))
    out = _ev.read_all("ws-1")
    assert out[0].type == _ev.EventType.APPROVED
    assert out[0].payload == {"by": "user"}
