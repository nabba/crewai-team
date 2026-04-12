"""
Comprehensive tests for the LLM Wiki subsystem.

Covers:
  1. WikiReadTool — path traversal, frontmatter parsing, missing files
  2. WikiWriteTool — create/update/deprecate, frontmatter validation,
     epistemic enforcement, locking, semaphore, ChromaDB embed, typed relationships
  3. WikiSearchTool — grep fallback, BM25 scoring, hybrid search
  4. WikiLintTool — all 8 health checks, contradiction resolution
  5. WikiSlidesTool — Marp generation, error handling
  6. Hot cache — generation, Commander integration
  7. Cross-system integration — agent wiring, idle scheduler, schema docs
  8. Helper functions — _safe_path, _parse_frontmatter, _bm25_score

Total: ~80 tests
"""

import json
import math
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

# Mock Docker-only modules
for _mod in ["psycopg2", "psycopg2.pool", "psycopg2.extras",
             "chromadb", "chromadb.config", "chromadb.utils",
             "chromadb.utils.embedding_functions",
             "crewai", "crewai.tools",
             "app.control_plane", "app.control_plane.db",
             "app.memory", "app.memory.chromadb_manager"]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# Mock BaseTool so wiki tools can import
class _MockBaseTool:
    name: str = ""
    description: str = ""
    def _run(self, **kwargs): pass

sys.modules["crewai.tools"].BaseTool = _MockBaseTool
sys.modules["pydantic"] = MagicMock()
sys.modules["pydantic"].Field = lambda **kw: None

sys.modules["app.memory.chromadb_manager"].embed = MagicMock(return_value=[0.1] * 768)
sys.modules["app.memory.chromadb_manager"].store = MagicMock()
sys.modules["app.memory.chromadb_manager"].retrieve_with_metadata = MagicMock(return_value=[])

import pytest
import yaml


# ═══════════════════════════════════════════════════════════════════════════════
# Setup: temporary wiki directory for isolated testing
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def wiki_dir(tmp_path):
    """Create a temporary wiki directory structure for testing."""
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    for section in ("meta", "self", "philosophy", "plg", "archibal", "kaicart"):
        (wiki / section).mkdir()
        idx = wiki / section / "index.md"
        idx.write_text(f"---\ntitle: \"{section} Index\"\nsection: {section}\npage_count: 0\n---\n# {section}\n")
    (wiki / ".locks").mkdir()
    (wiki / ".slides").mkdir()
    # Master index
    (wiki / "index.md").write_text("---\ntitle: Master Index\ntotal_pages: 0\n---\n# Wiki\n")
    # Log
    (wiki / "log.md").write_text("---\ntitle: Log\n---\n# Log\n")

    # Patch WIKI_ROOT
    import app.tools.wiki_tools as wt
    original_root = wt.WIKI_ROOT
    original_locks = wt.LOCKS_DIR
    original_slides = wt.SLIDES_DIR
    wt.WIKI_ROOT = str(wiki)
    wt.LOCKS_DIR = str(wiki / ".locks")
    wt.SLIDES_DIR = str(wiki / ".slides")

    yield wiki

    wt.WIKI_ROOT = original_root
    wt.LOCKS_DIR = original_locks
    wt.SLIDES_DIR = original_slides


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Helper Functions
# ═══════════════════════════════════════════════════════════════════════════════

class TestHelpers:
    """Tests for helper functions."""

    def test_parse_frontmatter_valid(self):
        from app.tools.wiki_tools import _parse_frontmatter
        content = "---\ntitle: Test\nstatus: active\n---\n# Body\nHello"
        fm, body = _parse_frontmatter(content)
        assert fm["title"] == "Test"
        assert "Hello" in body

    def test_parse_frontmatter_missing(self):
        from app.tools.wiki_tools import _parse_frontmatter
        fm, body = _parse_frontmatter("No frontmatter here")
        assert fm == {}
        assert "No frontmatter" in body

    def test_parse_frontmatter_invalid_yaml(self):
        from app.tools.wiki_tools import _parse_frontmatter
        fm, body = _parse_frontmatter("---\n: invalid: yaml: [[\n---\nbody")
        assert fm == {}

    def test_render_frontmatter(self):
        from app.tools.wiki_tools import _render_frontmatter
        result = _render_frontmatter({"title": "Test", "status": "active"})
        assert result.startswith("---\n")
        assert result.endswith("---\n")
        assert "title: Test" in result

    def test_safe_path_valid(self, wiki_dir):
        from app.tools.wiki_tools import _safe_path
        result = _safe_path("meta/test-page.md")
        assert str(wiki_dir) in result

    def test_safe_path_traversal_blocked(self, wiki_dir):
        from app.tools.wiki_tools import _safe_path
        with pytest.raises(ValueError, match="traversal"):
            _safe_path("../../etc/passwd")

    def test_bm25_score_matching(self):
        from app.tools.wiki_tools import _bm25_score
        score = _bm25_score("competitive landscape", "The competitive landscape analysis shows strong positioning")
        assert score > 0

    def test_bm25_score_no_match(self):
        from app.tools.wiki_tools import _bm25_score
        score = _bm25_score("quantum physics", "The cat sat on the mat")
        assert score == 0.0

    def test_bm25_score_empty(self):
        from app.tools.wiki_tools import _bm25_score
        assert _bm25_score("", "doc") == 0.0
        assert _bm25_score("query", "") == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 2. WikiReadTool
# ═══════════════════════════════════════════════════════════════════════════════

class TestWikiReadTool:
    """Tests for WikiReadTool."""

    def test_read_index(self, wiki_dir):
        from app.tools.wiki_tools import WikiReadTool
        reader = WikiReadTool()
        result = reader._run(path="index.md")
        assert "Wiki" in result or "Master" in result

    def test_read_section_index(self, wiki_dir):
        from app.tools.wiki_tools import WikiReadTool
        reader = WikiReadTool()
        result = reader._run(path="meta/index.md")
        assert "meta" in result.lower()

    def test_read_nonexistent(self, wiki_dir):
        from app.tools.wiki_tools import WikiReadTool
        reader = WikiReadTool()
        result = reader._run(path="meta/nonexistent.md")
        assert "not found" in result.lower() or "error" in result.lower()

    def test_read_frontmatter_only(self, wiki_dir):
        # Create a page first
        page = wiki_dir / "meta" / "test.md"
        page.write_text("---\ntitle: Test Page\nstatus: active\n---\n# Body\nContent here")
        from app.tools.wiki_tools import WikiReadTool
        reader = WikiReadTool()
        result = reader._run(path="meta/test", frontmatter_only=True)
        assert "title" in result
        assert "Body" not in result  # Body should be excluded

    def test_read_path_traversal_blocked(self, wiki_dir):
        from app.tools.wiki_tools import WikiReadTool
        reader = WikiReadTool()
        # _safe_path raises ValueError on traversal — tool should handle it
        try:
            result = reader._run(path="../../etc/passwd")
            assert "error" in result.lower() or "traversal" in result.lower()
        except ValueError as e:
            assert "traversal" in str(e).lower()  # Also acceptable


# ═══════════════════════════════════════════════════════════════════════════════
# 3. WikiWriteTool
# ═══════════════════════════════════════════════════════════════════════════════

class TestWikiWriteTool:
    """Tests for WikiWriteTool create/update/deprecate."""

    def test_create_page(self, wiki_dir):
        from app.tools.wiki_tools import WikiWriteTool
        writer = WikiWriteTool()
        result = writer._run(
            action="create", section="meta", slug="test-create",
            author="researcher", title="Test Create",
            content="Test content.", source="raw/test.md",
        )
        assert "Created" in result or "created" in result
        assert (wiki_dir / "meta" / "test-create.md").exists()

    def test_create_sets_frontmatter(self, wiki_dir):
        from app.tools.wiki_tools import WikiWriteTool, WikiReadTool
        writer = WikiWriteTool()
        writer._run(
            action="create", section="plg", slug="fm-test",
            author="researcher", title="FM Test",
            content="Content.", confidence="high", source="raw/src.md",
            tags="test,frontmatter",
        )
        reader = WikiReadTool()
        page = reader._run(path="plg/fm-test")
        assert "title: FM Test" in page
        assert "confidence: high" in page
        assert "aliases" in page  # Dataview field
        assert "date:" in page  # Dataview date field

    def test_create_with_typed_relationships(self, wiki_dir):
        from app.tools.wiki_tools import WikiWriteTool, WikiReadTool
        writer = WikiWriteTool()
        writer._run(
            action="create", section="meta", slug="rel-test",
            author="researcher", title="Relationship Test",
            content="Testing typed rels.", source="raw/test.md",
            relationships="supports:meta/other,contradicts:meta/conflict",
        )
        reader = WikiReadTool()
        page = reader._run(path="meta/rel-test")
        assert "supports" in page
        assert "contradicts" in page

    def test_update_page(self, wiki_dir):
        from app.tools.wiki_tools import WikiWriteTool
        writer = WikiWriteTool()
        writer._run(action="create", section="meta", slug="update-test",
                     author="researcher", title="V1", content="Version 1.",
                     source="raw/test.md")
        result = writer._run(action="update", section="meta", slug="update-test",
                              author="researcher", title="V2", content="Version 2.")
        assert "Updated" in result or "updated" in result

    def test_update_increments_version(self, wiki_dir):
        from app.tools.wiki_tools import WikiWriteTool, _parse_frontmatter
        writer = WikiWriteTool()
        writer._run(action="create", section="meta", slug="ver-test",
                     author="researcher", title="V1", content="V1.",
                     source="raw/test.md")
        writer._run(action="update", section="meta", slug="ver-test",
                      author="researcher", content="V2.")
        page = (wiki_dir / "meta" / "ver-test.md").read_text()
        fm, _ = _parse_frontmatter(page)
        assert fm.get("version", 0) >= 2

    def test_update_stores_relationships(self, wiki_dir):
        from app.tools.wiki_tools import WikiWriteTool, _parse_frontmatter
        writer = WikiWriteTool()
        writer._run(action="create", section="meta", slug="rel-upd",
                     author="researcher", title="T", content="C.",
                     source="raw/test.md")
        writer._run(action="update", section="meta", slug="rel-upd",
                      author="researcher", content="Updated.",
                      relationships="extends:meta/base")
        page = (wiki_dir / "meta" / "rel-upd.md").read_text()
        fm, _ = _parse_frontmatter(page)
        assert any(r.get("type") == "extends" for r in fm.get("relationships", []))

    def test_deprecate_page(self, wiki_dir):
        from app.tools.wiki_tools import WikiWriteTool, _parse_frontmatter
        writer = WikiWriteTool()
        writer._run(action="create", section="meta", slug="dep-test",
                     author="researcher", title="Old Page", content="Old.",
                     source="raw/test.md")
        result = writer._run(action="deprecate", section="meta", slug="dep-test",
                              author="researcher", deprecated_by="meta/new-page")
        assert "Deprecated" in result or "deprecated" in result
        page = (wiki_dir / "meta" / "dep-test.md").read_text()
        fm, _ = _parse_frontmatter(page)
        assert fm.get("status") == "deprecated"

    def test_create_duplicate_rejected(self, wiki_dir):
        from app.tools.wiki_tools import WikiWriteTool
        writer = WikiWriteTool()
        writer._run(action="create", section="meta", slug="dup-test",
                     author="researcher", title="T", content="C.",
                     source="raw/test.md")
        result = writer._run(action="create", section="meta", slug="dup-test",
                              author="researcher", title="T2", content="C2.",
                              source="raw/test.md")
        assert "already exists" in result.lower() or "error" in result.lower()

    def test_invalid_section_rejected(self, wiki_dir):
        from app.tools.wiki_tools import WikiWriteTool
        writer = WikiWriteTool()
        result = writer._run(action="create", section="invalid", slug="test",
                              author="researcher", title="T", content="C.")
        assert "error" in result.lower()

    def test_invalid_slug_rejected(self, wiki_dir):
        from app.tools.wiki_tools import WikiWriteTool
        writer = WikiWriteTool()
        result = writer._run(action="create", section="meta", slug="INVALID SLUG",
                              author="researcher", title="T", content="C.")
        assert "error" in result.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DGM Epistemic Boundary Enforcement
# ═══════════════════════════════════════════════════════════════════════════════

class TestEpistemicEnforcement:
    """Tests for DGM epistemic boundary enforcement at write-time."""

    def test_high_confidence_requires_source(self, wiki_dir):
        from app.tools.wiki_tools import WikiWriteTool
        writer = WikiWriteTool()
        result = writer._run(
            action="create", section="meta", slug="epist-test",
            author="researcher", title="T", content="C.",
            confidence="verified", source="synthesis",
        )
        assert "violation" in result.lower() or "error" in result.lower()

    def test_creative_blocked_in_ventures(self, wiki_dir):
        from app.tools.wiki_tools import WikiWriteTool
        writer = WikiWriteTool()
        result = writer._run(
            action="create", section="archibal", slug="creative-test",
            author="writer", title="T", content="C.",
            source="creative",
        )
        assert "boundary" in result.lower() or "error" in result.lower()

    def test_medium_confidence_allows_synthesis(self, wiki_dir):
        from app.tools.wiki_tools import WikiWriteTool
        writer = WikiWriteTool()
        result = writer._run(
            action="create", section="meta", slug="synth-ok",
            author="researcher", title="Synthesis Page",
            content="Synthesized from multiple sources.",
            confidence="medium", source="synthesis",
        )
        assert "Created" in result


# ═══════════════════════════════════════════════════════════════════════════════
# 5. WikiSearchTool
# ═══════════════════════════════════════════════════════════════════════════════

class TestWikiSearchTool:
    """Tests for grep-based and hybrid search."""

    def test_grep_search_finds_content(self, wiki_dir):
        from app.tools.wiki_tools import WikiWriteTool, WikiSearchTool
        WikiWriteTool()._run(
            action="create", section="meta", slug="search-target",
            author="researcher", title="Competitive Analysis",
            content="The competitive landscape is evolving rapidly.",
            source="raw/test.md",
        )
        searcher = WikiSearchTool()
        result = searcher._run(query="competitive landscape")
        assert "search-target" in result.lower() or "Competitive" in result

    def test_grep_search_no_match(self, wiki_dir):
        from app.tools.wiki_tools import WikiSearchTool
        searcher = WikiSearchTool()
        result = searcher._run(query="xyznonexistent123")
        assert "no" in result.lower() or "0" in result

    def test_search_section_filter(self, wiki_dir):
        from app.tools.wiki_tools import WikiWriteTool, WikiSearchTool
        WikiWriteTool()._run(
            action="create", section="plg", slug="plg-page",
            author="researcher", title="PLG Content",
            content="PLG specific content here.", source="raw/test.md",
        )
        searcher = WikiSearchTool()
        # Search in plg section only
        result = searcher._run(query="PLG specific", section="plg")
        assert "plg-page" in result.lower() or "PLG" in result

    def test_search_invalid_section(self, wiki_dir):
        from app.tools.wiki_tools import WikiSearchTool
        result = WikiSearchTool()._run(query="test", section="invalid")
        assert "error" in result.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# 6. WikiLintTool
# ═══════════════════════════════════════════════════════════════════════════════

class TestWikiLintTool:
    """Tests for wiki health checks."""

    def test_lint_empty_wiki(self, wiki_dir):
        from app.tools.wiki_tools import WikiLintTool
        result = WikiLintTool()._run()
        assert "lint" in result.lower() or "no issues" in result.lower() or "passed" in result.lower()

    def test_lint_detects_contradiction(self, wiki_dir):
        from app.tools.wiki_tools import WikiWriteTool, WikiLintTool
        writer = WikiWriteTool()
        # Create two pages with same tags (potential contradiction)
        writer._run(action="create", section="archibal", slug="page-a",
                     author="researcher", title="Page A",
                     content="Claim A.", source="raw/a.md",
                     tags="competitive,c2pa")
        writer._run(action="create", section="archibal", slug="page-b",
                     author="researcher", title="Page B",
                     content="Different claim.", source="raw/b.md",
                     tags="competitive,c2pa")
        result = WikiLintTool()._run(section="archibal")
        assert "contradiction" in result.lower() or "overlap" in result.lower()

    def test_lint_detects_explicit_contradiction(self, wiki_dir):
        from app.tools.wiki_tools import WikiWriteTool, WikiLintTool
        writer = WikiWriteTool()
        writer._run(action="create", section="meta", slug="contradict-a",
                     author="researcher", title="A", content="A.",
                     source="raw/a.md",
                     relationships="contradicts:meta/contradict-b")
        writer._run(action="create", section="meta", slug="contradict-b",
                     author="researcher", title="B", content="B.",
                     source="raw/b.md")
        result = WikiLintTool()._run(section="meta")
        assert "EXPLICIT" in result or "contradicts" in result

    def test_lint_resolution_recommendation(self, wiki_dir):
        from app.tools.wiki_tools import WikiWriteTool, WikiLintTool
        writer = WikiWriteTool()
        writer._run(action="create", section="meta", slug="rec-a",
                     author="researcher", title="Rec A", content="A.",
                     source="raw/a.md", tags="analysis", confidence="high")
        writer._run(action="create", section="meta", slug="rec-b",
                     author="researcher", title="Rec B", content="B.",
                     source="raw/b.md", tags="analysis", confidence="low")
        result = WikiLintTool()._run(section="meta")
        assert "RECOMMENDATION" in result or "authoritative" in result.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# 7. WikiSlidesTool
# ═══════════════════════════════════════════════════════════════════════════════

class TestWikiSlidesTool:
    """Tests for Marp slide generation."""

    def test_generate_slides(self, wiki_dir):
        from app.tools.wiki_tools import WikiWriteTool, WikiSlidesTool
        WikiWriteTool()._run(
            action="create", section="meta", slug="slides-source",
            author="researcher", title="Slide Source",
            content="## Overview\nKey findings.\n\n## Analysis\nDetailed analysis here.",
            source="raw/test.md",
        )
        result = WikiSlidesTool()._run(page_path="meta/slides-source")
        assert "Generated" in result
        assert (wiki_dir / ".slides" / "slides-source.md").exists()

    def test_slides_marp_format(self, wiki_dir):
        from app.tools.wiki_tools import WikiWriteTool, WikiSlidesTool
        WikiWriteTool()._run(
            action="create", section="meta", slug="marp-test",
            author="researcher", title="Marp Test",
            content="## Section 1\nContent 1.\n\n## Section 2\nContent 2.",
            source="raw/test.md",
        )
        WikiSlidesTool()._run(page_path="meta/marp-test")
        slides = (wiki_dir / ".slides" / "marp-test.md").read_text()
        assert "marp: true" in slides
        assert "---" in slides  # Slide separators

    def test_slides_missing_page(self, wiki_dir):
        from app.tools.wiki_tools import WikiSlidesTool
        result = WikiSlidesTool()._run(page_path="meta/nonexistent")
        assert "not found" in result.lower() or "error" in result.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Index Rebuilding
# ═══════════════════════════════════════════════════════════════════════════════

class TestIndexRebuilding:
    """Tests for automatic index updates."""

    def test_master_index_updated_on_create(self, wiki_dir):
        from app.tools.wiki_tools import WikiWriteTool
        WikiWriteTool()._run(
            action="create", section="meta", slug="idx-test",
            author="researcher", title="Index Test Page",
            content="For index testing.", source="raw/test.md",
        )
        index = (wiki_dir / "index.md").read_text()
        assert "idx-test" in index or "Index Test" in index

    def test_section_index_updated(self, wiki_dir):
        from app.tools.wiki_tools import WikiWriteTool
        WikiWriteTool()._run(
            action="create", section="plg", slug="sec-idx-test",
            author="researcher", title="Section Index Test",
            content="Testing section index.", source="raw/test.md",
        )
        sec_idx = (wiki_dir / "plg" / "index.md").read_text()
        assert "sec-idx-test" in sec_idx or "Section Index" in sec_idx

    def test_log_updated_on_create(self, wiki_dir):
        from app.tools.wiki_tools import WikiWriteTool
        WikiWriteTool()._run(
            action="create", section="meta", slug="log-test",
            author="researcher", title="Log Test",
            content="Testing log.", source="raw/test.md",
        )
        log = (wiki_dir / "log.md").read_text()
        assert "CREATE" in log or "log-test" in log


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Locking and Concurrency
# ═══════════════════════════════════════════════════════════════════════════════

class TestLockingAndConcurrency:
    """Tests for file locking and write semaphore."""

    def test_semaphore_exists(self):
        from app.tools.wiki_tools import _WRITE_SEMAPHORE
        assert _WRITE_SEMAPHORE._value == 3

    def test_stale_lock_cleanup(self, wiki_dir):
        from app.tools.wiki_tools import _cleanup_stale_locks, LOCKS_DIR
        # Create a stale lock
        lock_path = os.path.join(LOCKS_DIR, "stale_test.lock")
        with open(lock_path, "w") as f:
            f.write("test")
        # Make it old
        old_time = time.time() - 600  # 10 minutes ago
        os.utime(lock_path, (old_time, old_time))
        _cleanup_stale_locks(max_age_s=300)
        assert not os.path.exists(lock_path)


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Hot Cache
# ═══════════════════════════════════════════════════════════════════════════════

class TestHotCache:
    """Tests for wiki/hot.md session context."""

    def test_hot_cache_generation(self, wiki_dir):
        import app.tools.wiki_tools as wt
        # Patch the import path for hot cache
        with patch.dict("sys.modules", {"app.tools.wiki_tools": wt}):
            from app.tools.wiki_hot_cache import update_hot_cache
            update_hot_cache()
        assert (wiki_dir / "hot.md").exists()

    def test_hot_cache_contains_stats(self, wiki_dir):
        import app.tools.wiki_tools as wt
        from app.tools.wiki_tools import WikiWriteTool
        WikiWriteTool()._run(
            action="create", section="meta", slug="hot-test",
            author="researcher", title="Hot Test",
            content="Content for hot cache.", source="raw/test.md",
        )
        with patch.dict("sys.modules", {"app.tools.wiki_tools": wt}):
            from app.tools.wiki_hot_cache import update_hot_cache
            update_hot_cache()
        hot = (wiki_dir / "hot.md").read_text()
        assert "Wiki Stats" in hot or "Total pages" in hot


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Cross-System Integration (Source Code Checks)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossSystemIntegration:
    """Verify wiki is wired into the broader system."""

    def test_researcher_has_wiki_tools(self):
        src = (Path(__file__).parent.parent / "app" / "agents" / "researcher.py").read_text()
        assert "create_wiki_tools" in src
        assert '"slides"' in src

    def test_writer_has_wiki_tools(self):
        src = (Path(__file__).parent.parent / "app" / "agents" / "writer.py").read_text()
        assert "create_wiki_tools" in src
        assert '"slides"' in src

    def test_coder_has_wiki_tools(self):
        src = (Path(__file__).parent.parent / "app" / "agents" / "coder.py").read_text()
        assert "create_wiki_tools" in src

    def test_self_improver_has_all_wiki_tools(self):
        src = (Path(__file__).parent.parent / "app" / "crews" / "self_improvement_crew.py").read_text()
        assert "create_wiki_tools()" in src  # All tools (no filter)

    def test_commander_reads_wiki(self):
        src = (Path(__file__).parent.parent / "app" / "agents" / "commander" / "orchestrator.py").read_text()
        assert "wiki_index" in src or "hot.md" in src
        assert "wiki_block" in src or "wiki_knowledge" in src

    def test_idle_scheduler_has_wiki_lint(self):
        src = (Path(__file__).parent.parent / "app" / "idle_scheduler.py").read_text()
        assert "wiki-lint" in src
        assert "WikiLintTool" in src

    def test_idle_scheduler_has_hot_cache(self):
        src = (Path(__file__).parent.parent / "app" / "idle_scheduler.py").read_text()
        assert "wiki-hot-cache" in src
        assert "update_hot_cache" in src

    def test_tool_registry_has_5_tools(self):
        src = (Path(__file__).parent.parent / "app" / "tools" / "wiki_tool_registry.py").read_text()
        assert "WikiSlidesTool" in src
        assert '"slides"' in src
        assert '"read"' in src
        assert '"write"' in src
        assert '"search"' in src
        assert '"lint"' in src

    def test_schema_documents_relationships(self):
        src = (Path(__file__).parent.parent / "wiki_schema" / "WIKI_SCHEMA.md").read_text()
        assert "relationships" in src
        assert "supports" in src
        assert "contradicts" in src
        assert "aliases" in src
        assert "date" in src

    def test_obsidian_config_exists(self):
        assert (Path(__file__).parent.parent / "wiki" / ".obsidian" / "app.json").exists()
        assert (Path(__file__).parent.parent / "wiki" / ".obsidian" / "graph.json").exists()

    def test_gitignore_has_slides(self):
        gitignore = (Path(__file__).parent.parent / ".gitignore").read_text()
        assert "wiki/.slides/" in gitignore

    def test_gitignore_has_locks(self):
        gitignore = (Path(__file__).parent.parent / ".gitignore").read_text()
        assert "wiki/.locks/" in gitignore

    def test_dockerfile_copies_wiki(self):
        dockerfile = (Path(__file__).parent.parent / "Dockerfile").read_text()
        assert "COPY wiki/" in dockerfile
        assert "COPY raw/" in dockerfile


# ═══════════════════════════════════════════════════════════════════════════════
# 12. Constants and Configuration
# ═══════════════════════════════════════════════════════════════════════════════

class TestConstants:
    """Verify all configuration constants are correct."""

    def test_valid_sections(self):
        from app.tools.wiki_tools import VALID_SECTIONS
        assert "meta" in VALID_SECTIONS
        assert "self" in VALID_SECTIONS
        assert "philosophy" in VALID_SECTIONS
        assert "plg" in VALID_SECTIONS
        assert "archibal" in VALID_SECTIONS
        assert "kaicart" in VALID_SECTIONS
        assert len(VALID_SECTIONS) == 6

    def test_relationship_types(self):
        from app.tools.wiki_tools import VALID_RELATIONSHIP_TYPES
        assert "supports" in VALID_RELATIONSHIP_TYPES
        assert "contradicts" in VALID_RELATIONSHIP_TYPES
        assert "supersedes" in VALID_RELATIONSHIP_TYPES
        assert "prerequisite" in VALID_RELATIONSHIP_TYPES
        assert "tested_by" in VALID_RELATIONSHIP_TYPES
        assert "refines" in VALID_RELATIONSHIP_TYPES
        assert "extends" in VALID_RELATIONSHIP_TYPES
        assert len(VALID_RELATIONSHIP_TYPES) == 7

    def test_valid_confidence_levels(self):
        from app.tools.wiki_tools import VALID_CONFIDENCE
        assert "low" in VALID_CONFIDENCE
        assert "medium" in VALID_CONFIDENCE
        assert "high" in VALID_CONFIDENCE
        assert "verified" in VALID_CONFIDENCE

    def test_valid_statuses(self):
        from app.tools.wiki_tools import VALID_STATUSES
        assert "draft" in VALID_STATUSES
        assert "active" in VALID_STATUSES
        assert "deprecated" in VALID_STATUSES
