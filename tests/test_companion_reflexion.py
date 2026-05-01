"""Tests for app.companion.reflexion — feedback-driven prompt block."""

from pathlib import Path

import pytest

from app.companion import events as _ev
from app.companion import feedback as _fb
from app.companion import idea_store as _is
from app.companion import reflexion as _refl


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


def _persist_idea(workspace_id, text, idea_id=None):
    rec = _is.IdeaRecord(workspace_id=workspace_id, text=text)
    if idea_id:
        rec.idea_id = idea_id
    _is.persist(rec)
    return rec.idea_id


def test_empty_workspace_returns_empty_block(tmp_dirs):
    assert _refl.build_block("ws-1") == ""


def test_negative_feedback_appears_in_block(tmp_dirs):
    iid = _persist_idea("ws-1", "Use blockchain for paperclips")
    _fb.record(iid, "ws-1", polarity=_fb.Polarity.DOWN)
    block = _refl.build_block("ws-1")
    assert "Lessons from past feedback" in block
    assert "did NOT resonate" in block
    assert "blockchain for paperclips" in block


def test_positive_feedback_appears_in_block(tmp_dirs):
    iid = _persist_idea("ws-1", "Cycles in mycorrhizal networks")
    _fb.record(iid, "ws-1", polarity=_fb.Polarity.UP)
    block = _refl.build_block("ws-1")
    assert "DID resonate" in block
    assert "mycorrhizal" in block


def test_both_negative_and_positive_in_block(tmp_dirs):
    pos_iid = _persist_idea("ws-1", "Resonant positive idea body")
    neg_iid = _persist_idea("ws-1", "Bad negative idea body")
    _fb.record(pos_iid, "ws-1", polarity=_fb.Polarity.UP)
    _fb.record(neg_iid, "ws-1", polarity=_fb.Polarity.DOWN)
    block = _refl.build_block("ws-1")
    assert "Bad negative" in block
    assert "Resonant positive" in block
    # Negative section comes first, positive after.
    assert block.index("did NOT") < block.index("DID resonate")


def test_max_negative_respected(tmp_dirs):
    for i in range(10):
        iid = _persist_idea("ws-1", f"bad idea number {i}")
        _fb.record(iid, "ws-1", polarity=_fb.Polarity.DOWN)
    block = _refl.build_block("ws-1", max_negative=2, max_positive=0)
    # Each "- bad idea ..." line counts; cap at 2.
    bad_lines = [line for line in block.splitlines()
                 if line.startswith("- bad idea")]
    assert len(bad_lines) == 2


def test_truncation_caps_long_text(tmp_dirs):
    long = "x" * 1000
    iid = _persist_idea("ws-1", long)
    _fb.record(iid, "ws-1", polarity=_fb.Polarity.DOWN)
    block = _refl.build_block("ws-1")
    # Truncated to ~200 chars + "..." at most.
    long_lines = [line for line in block.splitlines() if line.startswith("- xxx")]
    assert len(long_lines) == 1
    assert len(long_lines[0]) < 250
    assert "..." in long_lines[0]


def test_collapses_newlines_in_snippet(tmp_dirs):
    iid = _persist_idea("ws-1", "line one\nline two\nline three")
    _fb.record(iid, "ws-1", polarity=_fb.Polarity.DOWN)
    block = _refl.build_block("ws-1")
    bullet_lines = [line for line in block.splitlines() if line.startswith("- ")]
    assert len(bullet_lines) == 1
    assert "\n" not in bullet_lines[0][2:]
    assert "line one" in bullet_lines[0]
    assert "line two" in bullet_lines[0]


def test_missing_idea_record_skipped(tmp_dirs):
    # Feedback for an idea that was never persisted.
    _fb.record("ghost-idea", "ws-1", polarity=_fb.Polarity.DOWN)
    block = _refl.build_block("ws-1")
    # No real idea text → block is empty, not a malformed bullet.
    assert block == ""


def test_summary_unavailable_returns_empty(tmp_dirs, monkeypatch):
    def _broken(ws):
        raise RuntimeError("storage down")

    monkeypatch.setattr(_fb, "summary", _broken)
    assert _refl.build_block("ws-1") == ""


def test_summary_returns_recent_positive_idea_ids(tmp_dirs):
    iid = _persist_idea("ws-1", "good")
    _fb.record(iid, "ws-1", polarity=_fb.Polarity.UP)
    s = _fb.summary("ws-1")
    assert iid in s["recent_positive_idea_ids"]
