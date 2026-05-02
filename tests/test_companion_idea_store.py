"""Tests for app.companion.idea_store — JSONL + ChromaDB best-effort persistence."""

from pathlib import Path
from unittest.mock import patch

import pytest

from app.companion import idea_store as _is


@pytest.fixture
def tmp_ideas_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_is, "_IDEAS_DIR", tmp_path)
    # Block all ChromaDB writes by default — most tests don't need them.
    monkeypatch.setattr(_is, "_index_chromadb", lambda r: None)
    return tmp_path


def test_idea_id_auto_generated():
    rec = _is.IdeaRecord(workspace_id="ws-1", text="hi")
    assert rec.idea_id.startswith("idea_")
    assert len(rec.idea_id) > len("idea_")


def test_persist_writes_jsonl(tmp_ideas_dir):
    rec = _is.IdeaRecord(workspace_id="ws-1", text="abc",
                          state=_is.IdeaState.CONVERGED)
    idea_id = _is.persist(rec)
    assert idea_id == rec.idea_id

    files = list(tmp_ideas_dir.iterdir())
    assert len(files) == 1
    assert files[0].name == "ws-1.jsonl"
    line = files[0].read_text().strip()
    assert rec.idea_id in line
    assert "abc" in line
    assert "converged" in line  # enum serialised as value


def test_persist_appends_multiple(tmp_ideas_dir):
    for i in range(3):
        _is.persist(_is.IdeaRecord(workspace_id="ws-1",
                                    text=f"idea {i}"))
    body = (tmp_ideas_dir / "ws-1.jsonl").read_text().splitlines()
    assert len(body) == 3


def test_find_by_workspace_returns_all(tmp_ideas_dir):
    _is.persist(_is.IdeaRecord(workspace_id="ws-1", text="a"))
    _is.persist(_is.IdeaRecord(workspace_id="ws-1", text="b"))

    found = _is.find_by_workspace("ws-1")
    assert [r.text for r in found] == ["a", "b"]


def test_find_by_workspace_filters_by_state(tmp_ideas_dir):
    _is.persist(_is.IdeaRecord(workspace_id="ws-1", text="frag",
                                state=_is.IdeaState.FRAGMENT))
    _is.persist(_is.IdeaRecord(workspace_id="ws-1", text="conv",
                                state=_is.IdeaState.CONVERGED))

    only_conv = _is.find_by_workspace("ws-1",
                                       state=_is.IdeaState.CONVERGED)
    assert len(only_conv) == 1
    assert only_conv[0].text == "conv"
    assert only_conv[0].state == _is.IdeaState.CONVERGED


def test_find_by_workspace_unknown_returns_empty(tmp_ideas_dir):
    assert _is.find_by_workspace("never-existed") == []


def test_find_respects_limit(tmp_ideas_dir):
    for i in range(20):
        _is.persist(_is.IdeaRecord(workspace_id="ws-1", text=f"i{i}"))
    found = _is.find_by_workspace("ws-1", limit=5)
    # Most-recent 5 returned
    assert len(found) == 5
    assert found[-1].text == "i19"


def test_path_sanitises_workspace_id(tmp_ideas_dir):
    _is.persist(_is.IdeaRecord(workspace_id="../../etc/passwd", text="x"))
    files = list(tmp_ideas_dir.iterdir())
    assert len(files) == 1
    assert files[0].parent == tmp_ideas_dir


def test_path_falls_back_to_default_for_empty_id(tmp_ideas_dir):
    _is.persist(_is.IdeaRecord(workspace_id="!!!", text="x"))
    assert (tmp_ideas_dir / "default.jsonl").exists()


def test_state_round_trip(tmp_ideas_dir):
    rec = _is.IdeaRecord(workspace_id="ws-1", text="x",
                          state=_is.IdeaState.DEVELOPED)
    _is.persist(rec)
    found = _is.find_by_workspace("ws-1")
    assert found[0].state == _is.IdeaState.DEVELOPED


def test_lineage_round_trip(tmp_ideas_dir):
    rec = _is.IdeaRecord(workspace_id="ws-1", text="x",
                          lineage_parents=["a", "b", "c"])
    _is.persist(rec)
    found = _is.find_by_workspace("ws-1")
    assert found[0].lineage_parents == ["a", "b", "c"]


def test_search_similar_empty_text_returns_empty():
    assert _is.search_similar("ws-1", "") == []
    assert _is.search_similar("ws-1", "   ") == []


def test_search_similar_absorbs_chroma_failure():
    """ChromaDB unreachable → return [], not raise."""
    def _broken(*a, **kw):
        raise RuntimeError("chroma down")

    with patch("app.companion.idea_store._chroma_query_for_workspace",
               _broken):
        assert _is.search_similar("ws-1", "query") == []


def test_search_similar_returns_distances():
    fake = [
        {"document": "alpha", "metadata": {"workspace_id": "ws-1"},
         "distance": 0.2},
        {"document": "beta", "metadata": {"workspace_id": "ws-1"},
         "distance": 0.5},
    ]

    with patch("app.companion.idea_store._chroma_query_for_workspace",
               lambda *a, **kw: fake):
        results = _is.search_similar("ws-1", "query", top_k=2)

    assert len(results) == 2
    assert results[0]["distance"] == 0.2
    assert results[1]["distance"] == 0.5
