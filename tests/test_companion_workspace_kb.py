"""Tests for app.companion.workspace_kb — context composer."""

from unittest.mock import patch

import pytest

from app.companion import workspace_kb


class _FakeOrchResult:
    def __init__(self, text, score, *, collection="episteme"):
        self.text = text
        self.score = score
        self.metadata = {"collection": collection}
        self.provenance = {"collection": collection}


class _FakeOrch:
    def __init__(self, results):
        self._results = results

    def retrieve(self, **kwargs):
        return list(self._results)


def test_compose_includes_temporal_snippet():
    """Temporal context is always present, even with no KB results."""
    snippets = workspace_kb.compose("ws-1", "forests", top_k=3)
    sources = [s.source for s in snippets]
    assert "temporal_context" in sources


def test_compose_with_kb_v2_results():
    fake_results = [
        _FakeOrchResult("Forests regulate carbon", 0.85, collection="episteme"),
        _FakeOrchResult("Past sapling pruning failed", 0.72,
                        collection="experiential"),
    ]
    with patch("app.retrieval.orchestrator.RetrievalOrchestrator",
               lambda *a, **k: _FakeOrch(fake_results)):
        snippets = workspace_kb.compose("ws-1", "forests", top_k=3)

    bodies = [s.text for s in snippets]
    assert "Forests regulate carbon" in bodies
    assert "Past sapling pruning failed" in bodies


def test_compose_absorbs_helper_failure():
    """compose() must never raise — idle code must not crash the scheduler."""
    def _broken(*a, **k):
        raise RuntimeError("chroma down")

    with patch("app.companion.workspace_kb._kb_v2_snippets", _broken):
        snippets = workspace_kb.compose("ws-1", "x", top_k=3)
    # Temporal still present even when KB fails.
    assert any(s.source == "temporal_context" for s in snippets)


def test_compose_absorbs_internal_kb_v2_failure():
    """The compose function itself catches kb_v2 failure via the helper's
    own try/except — verify the helper returns []."""
    class _BrokenOrch:
        def retrieve(self, **kwargs):
            raise RuntimeError("chroma down")

    with patch("app.retrieval.orchestrator.RetrievalOrchestrator",
               lambda *a, **k: _BrokenOrch()):
        result = workspace_kb._kb_v2_snippets("forests", top_k=3)
    assert result == []


def test_kb_v2_snippets_empty_query_returns_empty():
    assert workspace_kb._kb_v2_snippets("", top_k=3) == []
    assert workspace_kb._kb_v2_snippets("   ", top_k=3) == []


def test_snippet_to_prompt_line_truncates_long_text():
    long_text = "x" * 1000
    s = workspace_kb.KBSnippet(text=long_text, score=0.5, source="episteme")
    line = s.to_prompt_line()
    assert "episteme" in line
    assert "0.50" in line
    assert len(line) < len(long_text)


def test_snippet_to_prompt_line_collapses_newlines():
    s = workspace_kb.KBSnippet(text="line1\nline2\nline3", score=0.5,
                                source="episteme")
    assert "\n" not in s.to_prompt_line()


def test_compose_passes_collection_provenance():
    fake_results = [_FakeOrchResult("a", 0.5, collection="experiential")]
    with patch("app.retrieval.orchestrator.RetrievalOrchestrator",
               lambda *a, **k: _FakeOrch(fake_results)):
        snippets = workspace_kb.compose("ws-1", "x", top_k=1)
    kb_snippets = [s for s in snippets if s.source != "temporal_context"]
    assert len(kb_snippets) == 1
    assert kb_snippets[0].source == "experiential"
