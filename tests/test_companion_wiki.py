"""Tests for app.companion.wiki — workspace wiki + Mem0 + system wiki."""

from pathlib import Path
from unittest.mock import patch

import pytest

from app.companion import events as _ev
from app.companion import idea_store as _is
from app.companion import wiki as _w


@pytest.fixture
def tmp_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ws_wiki = tmp_path / "ws_wiki"
    sys_wiki = tmp_path / "sys_wiki"
    ideas = tmp_path / "ideas"
    events = tmp_path / "events"
    monkeypatch.setattr(_w, "_WORKSPACE_WIKI_DIR", ws_wiki)
    monkeypatch.setattr(_w, "_SYSTEM_WIKI_DIR", sys_wiki)
    monkeypatch.setattr(_is, "_IDEAS_DIR", ideas)
    monkeypatch.setattr(_is, "_index_chromadb", lambda r: None)
    monkeypatch.setattr(_ev, "_EVENTS_DIR", events)
    # Block real Mem0 imports — tests mock _invoke_mem0_add explicitly.
    monkeypatch.setattr(_w, "_invoke_mem0_add", lambda *a, **kw: None)
    return tmp_path


def _persist_idea(text="Mycorrhizal networks couple forests below ground.",
                   **kwargs):
    rec = _is.IdeaRecord(workspace_id="ws-1", text=text, **kwargs)
    _is.persist(rec)
    return rec


def test_publish_unknown_idea_returns_error(tmp_dirs):
    res = _w.publish_to_wiki("ws-1", "nonexistent")
    assert "idea not found" in res.errors
    assert res.wiki_page is None


def test_publish_writes_workspace_wiki_page(tmp_dirs):
    rec = _persist_idea()
    res = _w.publish_to_wiki("ws-1", rec.idea_id)
    assert res.wiki_page is not None
    p = Path(res.wiki_page)
    assert p.exists()
    body = p.read_text()
    assert "Mycorrhizal networks" in body
    assert f"idea_id: {rec.idea_id}" in body
    assert "workspace_id: ws-1" in body


def test_publish_includes_lineage_links(tmp_dirs):
    rec = _persist_idea(lineage_parents=["idea_aaaa1111", "idea_bbbb2222"])
    res = _w.publish_to_wiki("ws-1", rec.idea_id)
    body = Path(res.wiki_page).read_text()
    assert "## Lineage" in body
    assert "[[idea_aaaa1111]]" in body
    assert "[[idea_bbbb2222]]" in body


def test_publish_writes_system_wiki_page(tmp_dirs):
    rec = _persist_idea()
    res = _w.publish_to_wiki("ws-1", rec.idea_id)
    assert res.system_wiki_page is not None
    p = Path(res.system_wiki_page)
    assert p.exists()
    body = p.read_text()
    assert "section: meta/companion" in body
    assert "epistemic_status: companion-polished" in body
    assert "Mycorrhizal" in body
    # Path lands under sys_wiki/meta/companion/
    assert p.parent.name == "companion"
    assert p.parent.parent.name == "meta"


def test_publish_emits_wiki_registered_event(tmp_dirs):
    rec = _persist_idea()
    res = _w.publish_to_wiki("ws-1", rec.idea_id)
    events = _ev.read_for_idea("ws-1", rec.idea_id)
    registered = [e for e in events
                  if e.type == _ev.EventType.WIKI_REGISTERED]
    assert len(registered) == 1
    payload = registered[0].payload
    assert payload["wiki_page"] == res.wiki_page
    assert payload["system_wiki_page"] == res.system_wiki_page


def test_publish_calls_mem0_with_workspace_user_id(tmp_dirs):
    rec = _persist_idea()
    captured: dict = {}

    def _capture(workspace_id, idea, fact, metadata):
        captured["workspace_id"] = workspace_id
        captured["idea_id"] = idea.idea_id
        captured["fact"] = fact
        captured["metadata"] = metadata
        return "mem0_record_xyz"

    with patch("app.companion.wiki._invoke_mem0_add", _capture):
        res = _w.publish_to_wiki("ws-1", rec.idea_id)

    assert res.mem0_id == "mem0_record_xyz"
    assert captured["workspace_id"] == "ws-1"
    assert captured["idea_id"] == rec.idea_id
    assert "Workspace `ws-1`" in captured["fact"]
    assert captured["metadata"]["source"] == "companion"


def test_publish_absorbs_mem0_failure(tmp_dirs):
    rec = _persist_idea()

    def _broken(*a, **kw):
        raise RuntimeError("mem0 down")

    with patch("app.companion.wiki._invoke_mem0_add", _broken):
        res = _w.publish_to_wiki("ws-1", rec.idea_id)

    # Workspace wiki + system wiki still landed; mem0 listed in errors.
    assert res.wiki_page is not None
    assert res.system_wiki_page is not None
    assert any("mem0" in e for e in res.errors)
    assert res.mem0_id is None


def test_index_lists_pages(tmp_dirs):
    a = _persist_idea(text="First cool idea body content")
    b = _persist_idea(text="Second cool idea body content")
    _w.publish_to_wiki("ws-1", a.idea_id)
    _w.publish_to_wiki("ws-1", b.idea_id)

    index_path = tmp_dirs / "ws_wiki" / "ws-1" / "_index.md"
    assert index_path.exists()
    body = index_path.read_text()
    # Both ideas referenced in the index.
    assert "## Pages" in body
    assert a.idea_id in body
    assert b.idea_id in body
    assert "2 pages" in body


def test_workspaces_isolated(tmp_dirs):
    a = _is.IdeaRecord(workspace_id="ws-1", text="alpha idea")
    b = _is.IdeaRecord(workspace_id="ws-2", text="beta idea")
    _is.persist(a)
    _is.persist(b)
    _w.publish_to_wiki("ws-1", a.idea_id)
    _w.publish_to_wiki("ws-2", b.idea_id)

    ws1_dir = tmp_dirs / "ws_wiki" / "ws-1"
    ws2_dir = tmp_dirs / "ws_wiki" / "ws-2"
    assert ws1_dir.exists() and ws2_dir.exists()
    ws1_files = sorted(p.name for p in ws1_dir.iterdir())
    ws2_files = sorted(p.name for p in ws2_dir.iterdir())
    # Each workspace owns exactly one idea page + the index.
    assert any(a.idea_id in n for n in ws1_files)
    assert any(b.idea_id in n for n in ws2_files)
    assert not any(a.idea_id in n for n in ws2_files)


def test_extract_mem0_id_handles_dict_with_id():
    assert _w._extract_mem0_id({"id": "rec_42"}) == "rec_42"


def test_extract_mem0_id_handles_results_list():
    result = {"results": [{"id": "rec_99"}]}
    assert _w._extract_mem0_id(result) == "rec_99"


def test_extract_mem0_id_handles_object_with_attr():
    class _R:
        id = "rec_11"
    assert _w._extract_mem0_id(_R()) == "rec_11"


def test_extract_mem0_id_handles_none():
    assert _w._extract_mem0_id(None) is None
    assert _w._extract_mem0_id({"no_id": "x"}) is None


def test_slugify_safe():
    assert _w._slugify("Hello World!") == "hello-world"
    assert _w._slugify("A " * 100).startswith("a")
    assert len(_w._slugify("A " * 100)) <= 60
    assert _w._slugify("") == "idea"
    assert _w._slugify("!!!") == "idea"


def test_path_sanitises_workspace_id(tmp_dirs):
    rec = _is.IdeaRecord(workspace_id="../../etc/passwd", text="x" * 50)
    _is.persist(rec)
    res = _w.publish_to_wiki("../../etc/passwd", rec.idea_id)
    if res.wiki_page:
        wp = Path(res.wiki_page)
        # Stays under tmp_dirs/ws_wiki — no traversal.
        try:
            wp.relative_to(tmp_dirs / "ws_wiki")
        except ValueError:
            pytest.fail("workspace wiki path escaped sandbox")


def test_atomic_write_no_temp_files_left(tmp_dirs):
    rec = _persist_idea()
    _w.publish_to_wiki("ws-1", rec.idea_id)
    ws_dir = tmp_dirs / "ws_wiki" / "ws-1"
    leftovers = [p for p in ws_dir.iterdir() if p.name.startswith(".tmp")]
    assert leftovers == []
