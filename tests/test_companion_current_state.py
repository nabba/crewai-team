"""Tests for app.companion.idea_store.current_state — event-folded state."""

from pathlib import Path

import pytest

from app.companion import events as _ev
from app.companion import idea_store as _is


@pytest.fixture
def tmp_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ideas = tmp_path / "ideas"
    events = tmp_path / "events"
    ideas.mkdir()
    events.mkdir()
    monkeypatch.setattr(_is, "_IDEAS_DIR", ideas)
    monkeypatch.setattr(_is, "_index_chromadb", lambda r: None)
    monkeypatch.setattr(_ev, "_EVENTS_DIR", events)
    return tmp_path


def _idea(text="x", state=_is.IdeaState.CONVERGED, idea_id=None):
    rec = _is.IdeaRecord(workspace_id="ws-1", text=text, state=state)
    if idea_id:
        rec.idea_id = idea_id
    return rec


def test_current_state_unknown_idea_returns_none(tmp_dirs):
    assert _is.current_state("ws-1", "does-not-exist") is None


def test_current_state_no_events_returns_original(tmp_dirs):
    rec = _idea(state=_is.IdeaState.CONVERGED)
    _is.persist(rec)
    assert _is.current_state("ws-1", rec.idea_id) == _is.IdeaState.CONVERGED


def test_surfaced_event_promotes_state(tmp_dirs):
    rec = _idea(state=_is.IdeaState.CONVERGED)
    _is.persist(rec)
    _ev.append(_ev.Event(workspace_id="ws-1", idea_id=rec.idea_id,
                          type=_ev.EventType.SURFACED))
    assert _is.current_state("ws-1", rec.idea_id) == _is.IdeaState.SURFACED


def test_thumbs_down_archives_idea(tmp_dirs):
    rec = _idea(state=_is.IdeaState.CONVERGED)
    _is.persist(rec)
    _ev.append(_ev.Event(workspace_id="ws-1", idea_id=rec.idea_id,
                          type=_ev.EventType.SURFACED))
    _ev.append(_ev.Event(workspace_id="ws-1", idea_id=rec.idea_id,
                          type=_ev.EventType.FEEDBACK,
                          payload={"polarity": "down"}))
    assert _is.current_state("ws-1", rec.idea_id) == _is.IdeaState.ARCHIVED


def test_thumbs_up_keeps_surfaced(tmp_dirs):
    rec = _idea(state=_is.IdeaState.CONVERGED)
    _is.persist(rec)
    _ev.append(_ev.Event(workspace_id="ws-1", idea_id=rec.idea_id,
                          type=_ev.EventType.SURFACED))
    _ev.append(_ev.Event(workspace_id="ws-1", idea_id=rec.idea_id,
                          type=_ev.EventType.FEEDBACK,
                          payload={"polarity": "up"}))
    assert _is.current_state("ws-1", rec.idea_id) == _is.IdeaState.SURFACED


def test_explicit_archived_event_overrides(tmp_dirs):
    rec = _idea(state=_is.IdeaState.CONVERGED)
    _is.persist(rec)
    _ev.append(_ev.Event(workspace_id="ws-1", idea_id=rec.idea_id,
                          type=_ev.EventType.ARCHIVED))
    assert _is.current_state("ws-1", rec.idea_id) == _is.IdeaState.ARCHIVED


def test_find_by_id_returns_record(tmp_dirs):
    rec = _idea(text="findme")
    _is.persist(rec)
    found = _is.find_by_id("ws-1", rec.idea_id)
    assert found is not None
    assert found.text == "findme"


def test_find_by_id_missing_returns_none(tmp_dirs):
    assert _is.find_by_id("ws-1", "no-such-id") is None
