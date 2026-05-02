"""Tests for app.companion.document_pipeline — markdown + pandoc render."""

from pathlib import Path
from unittest.mock import patch

import pytest

from app.companion import document_pipeline as _dp
from app.companion import events as _ev
from app.companion import idea_store as _is


@pytest.fixture
def tmp_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    docs = tmp_path / "docs"
    ideas = tmp_path / "ideas"
    events = tmp_path / "events"
    monkeypatch.setattr(_dp, "_DOCUMENTS_DIR", docs)
    monkeypatch.setattr(_is, "_IDEAS_DIR", ideas)
    monkeypatch.setattr(_is, "_index_chromadb", lambda r: None)
    monkeypatch.setattr(_ev, "_EVENTS_DIR", events)
    return tmp_path


def _persist_idea(workspace_id="ws-1", text="An interesting idea body",
                   **kwargs):
    rec = _is.IdeaRecord(workspace_id=workspace_id, text=text, **kwargs)
    _is.persist(rec)
    return rec


def test_promote_unknown_idea_returns_error(tmp_dirs):
    result = _dp.promote("ws-1", "nonexistent")
    assert result.error == "idea not found"
    assert result.formats == {}


def test_promote_writes_md_canonical(tmp_dirs):
    rec = _persist_idea(text="Mycorrhizal networks couple forests below ground.")
    result = _dp.promote("ws-1", rec.idea_id)
    assert result.error is None
    assert "md" in result.formats
    md_path = Path(result.formats["md"])
    assert md_path.exists()
    body = md_path.read_text()
    assert "Mycorrhizal networks" in body
    assert "idea_id:" in body
    assert "workspace_id: ws-1" in body


def test_promote_includes_scores_in_frontmatter(tmp_dirs):
    rec = _persist_idea(novelty=0.85, quality=0.72,
                         transferability=0.60, panel_score=0.91)
    result = _dp.promote("ws-1", rec.idea_id)
    body = Path(result.formats["md"]).read_text()
    assert "novelty: 0.850" in body
    assert "quality: 0.720" in body
    assert "transferability: 0.600" in body
    assert "panel_score: 0.910" in body


def test_promote_includes_lineage_section(tmp_dirs):
    rec = _persist_idea(lineage_parents=["idea_aaaa1111", "idea_bbbb2222"])
    result = _dp.promote("ws-1", rec.idea_id)
    body = Path(result.formats["md"]).read_text()
    assert "## Lineage" in body
    assert "idea_aaaa1111" in body
    assert "idea_bbbb2222" in body


def test_promote_emits_documented_event(tmp_dirs):
    rec = _persist_idea()
    _dp.promote("ws-1", rec.idea_id, formats=["md", "docx"])
    events = _ev.read_for_idea("ws-1", rec.idea_id)
    documented = [e for e in events if e.type == _ev.EventType.DOCUMENTED]
    assert len(documented) == 1
    assert "md" in documented[0].payload["formats"]


def test_extract_title_from_first_line(tmp_dirs):
    assert _dp.extract_title("# Forest dynamics\nrest of body") == "Forest dynamics"
    assert _dp.extract_title("Forest dynamics\nrest of body") == "Forest dynamics"


def test_extract_title_falls_back_to_first_chars(tmp_dirs):
    long_first_line = ("This is a very long single-line idea body that "
                        "should fall back to the first 80 chars of cropping "
                        "since it's too long to be a heading.")
    title = _dp.extract_title(long_first_line)
    assert len(title) <= 80
    assert title.startswith("This is a very long")


def test_extract_title_handles_empty():
    assert _dp.extract_title(None) == "(untitled)"
    assert _dp.extract_title("") == "(untitled)"
    assert _dp.extract_title("   ") == "(untitled)"


def test_promote_skips_unknown_formats(tmp_dirs):
    rec = _persist_idea()
    result = _dp.promote("ws-1", rec.idea_id,
                          formats=["md", "telepathy"])
    assert "telepathy" not in result.formats


def test_promote_skips_pandoc_when_unavailable(tmp_dirs):
    rec = _persist_idea()
    with patch("app.companion.document_pipeline._pandoc_executable",
               lambda: None):
        result = _dp.promote("ws-1", rec.idea_id, formats=["md", "docx", "pdf"])
    # md still landed; docx/pdf silently skipped.
    assert "md" in result.formats
    assert "docx" not in result.formats
    assert "pdf" not in result.formats


def test_promote_renders_docx_when_pandoc_succeeds(tmp_dirs):
    rec = _persist_idea()
    rendered: list[tuple[str, str]] = []

    def _fake_pandoc(pandoc_path, src, dst):
        rendered.append((src, dst))
        Path(dst).write_text("rendered docx bytes")

    with patch("app.companion.document_pipeline._pandoc_executable",
               lambda: "/usr/bin/pandoc"), \
         patch("app.companion.document_pipeline._invoke_pandoc", _fake_pandoc):
        result = _dp.promote("ws-1", rec.idea_id, formats=["md", "docx"])

    assert "docx" in result.formats
    assert Path(result.formats["docx"]).exists()
    assert len(rendered) == 1


def test_promote_render_failure_logged_not_raised(tmp_dirs):
    rec = _persist_idea()

    def _broken(pandoc_path, src, dst):
        raise RuntimeError("pandoc went sideways")

    with patch("app.companion.document_pipeline._pandoc_executable",
               lambda: "/usr/bin/pandoc"), \
         patch("app.companion.document_pipeline._invoke_pandoc", _broken):
        result = _dp.promote("ws-1", rec.idea_id, formats=["md", "pdf"])

    # md still landed; pdf absent; no exception bubbled up.
    assert "md" in result.formats
    assert "pdf" not in result.formats
    assert result.error is None


def test_list_formats_returns_existing_artifacts(tmp_dirs):
    rec = _persist_idea()
    _dp.promote("ws-1", rec.idea_id)
    out = _dp.list_formats("ws-1", rec.idea_id)
    assert "md" in out
    # docx was never rendered → not present.
    assert "docx" not in out


def test_list_formats_empty_when_never_promoted(tmp_dirs):
    rec = _persist_idea()
    assert _dp.list_formats("ws-1", rec.idea_id) == {}


def test_path_for_returns_none_for_unknown_format(tmp_dirs):
    assert _dp.path_for("ws-1", "idea_x", "telepathy") is None


def test_path_for_returns_path_for_known_format(tmp_dirs):
    p = _dp.path_for("ws-1", "idea_x", "md")
    assert p is not None
    assert p.suffix == ".md"


def test_path_sanitises_workspace_and_idea_ids(tmp_dirs):
    rec = _persist_idea(workspace_id="../../etc/passwd")
    rec.idea_id = "../../../escape"
    _is._append_jsonl(rec)
    result = _dp.promote("../../etc/passwd", "../../../escape")
    # md lands inside _DOCUMENTS_DIR — no traversal.
    if "md" in result.formats:
        md_path = Path(result.formats["md"])
        # Resolved path stays under tmp_dirs/docs.
        try:
            md_path.relative_to(tmp_dirs / "docs")
        except ValueError:
            pytest.fail("path escaped documents dir")


def test_atomic_write_no_temp_files_left(tmp_dirs):
    rec = _persist_idea()
    _dp.promote("ws-1", rec.idea_id)
    docs_dir = tmp_dirs / "docs" / "ws-1"
    leftovers = [p for p in docs_dir.iterdir() if p.name.startswith(".tmp")]
    assert leftovers == []


def test_promote_is_idempotent(tmp_dirs):
    """Re-promoting the same idea produces the same file without error."""
    rec = _persist_idea(text="stable idea content")
    a = _dp.promote("ws-1", rec.idea_id)
    b = _dp.promote("ws-1", rec.idea_id)
    assert a.error is None
    assert b.error is None
    assert a.formats["md"] == b.formats["md"]
    md_path = Path(a.formats["md"])
    assert md_path.exists()
    assert "stable idea content" in md_path.read_text()
