"""
End-to-end tests for the Knowledge Base Architecture v2.
==========================================================

Tests the full pipeline across all phases:
  Phase 0: Retrieval orchestrator (rerank, decompose, temporal, cross-KB)
  Phase 1: Retrieval improvements wired into existing KBs
  Phase 2: Four new knowledge bases (episteme, experiential, aesthetics, tensions)
  Phase 3: Context injection, evolution wiring, agent tools, feedback loops

Convention: Tests that import chromadb/crewai are guarded with LOW_MEM.
Tests that only inspect source code or test pure Python logic run everywhere.
"""

import inspect
import json
import os
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

_LOW_MEM = os.environ.get("LOW_MEM_TESTS", "1") == "1"

# Helper: read source files directly (avoids chromadb/crewai import chains).
_APP_DIR = Path(__file__).parent.parent / "app"


def _read_src(rel_path: str) -> str:
    """Read a source file relative to app/ — no import needed."""
    return (_APP_DIR / rel_path).read_text(encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 0: RETRIEVAL ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

class TestRetrievalOrchestratorImports:
    """All retrieval modules must import cleanly."""

    def test_orchestrator_imports(self):
        from app.retrieval import (
            RetrievalOrchestrator, RetrievalResult, RetrievalConfig,
            rerank, decompose_query, apply_temporal_decay,
        )
        assert callable(rerank)
        assert callable(decompose_query)
        assert callable(apply_temporal_decay)

    def test_config_dataclass(self):
        from app.retrieval.config import RetrievalConfig
        cfg = RetrievalConfig()
        assert cfg.rerank_enabled is True
        assert cfg.decomposition_enabled is True
        assert cfg.temporal_enabled is False
        # rerank_top_k_input/output are env-derived; assert only the type
        # contract, not the value (which an operator may tune via .env).
        assert isinstance(cfg.rerank_top_k_input, int)
        assert isinstance(cfg.rerank_top_k_output, int)
        assert 0 < cfg.temporal_weight < 1
        assert cfg.temporal_half_life_hours > 0

    def test_config_override(self):
        from app.retrieval.config import RetrievalConfig
        cfg = RetrievalConfig(
            rerank_enabled=False,
            temporal_enabled=True,
            temporal_field="created_at",
            temporal_half_life_hours=24.0,
        )
        assert cfg.rerank_enabled is False
        assert cfg.temporal_enabled is True
        assert cfg.temporal_field == "created_at"


class TestTemporalDecay:
    """Temporal freshness scoring math."""

    def test_newer_scores_higher(self):
        from app.retrieval.temporal import apply_temporal_decay
        now = datetime.now(timezone.utc)
        results = [
            {"text": "old", "score": 0.8, "metadata": {"ts": (now - timedelta(days=30)).isoformat()}},
            {"text": "new", "score": 0.8, "metadata": {"ts": (now - timedelta(hours=1)).isoformat()}},
        ]
        out = apply_temporal_decay(results, timestamp_field="ts", now=now)
        assert out[0]["text"] == "new"
        assert out[0]["temporal_score"] > out[1]["temporal_score"]

    def test_half_life_math(self):
        """At exactly half_life hours, temporal_score should be ~0.5."""
        from app.retrieval.temporal import apply_temporal_decay
        now = datetime.now(timezone.utc)
        results = [{
            "text": "x", "score": 1.0,
            "metadata": {"ts": (now - timedelta(hours=168)).isoformat()},
        }]
        out = apply_temporal_decay(results, timestamp_field="ts", half_life_hours=168.0, now=now)
        assert abs(out[0]["temporal_score"] - 0.5) < 0.01

    def test_missing_timestamps_neutral(self):
        from app.retrieval.temporal import apply_temporal_decay
        results = [
            {"text": "a", "score": 0.9, "metadata": {}},
            {"text": "b", "score": 0.7, "metadata": {}},
        ]
        out = apply_temporal_decay(results)
        assert all(r["temporal_score"] == 0.5 for r in out)

    def test_zero_weight_preserves_semantic(self):
        from app.retrieval.temporal import apply_temporal_decay
        results = [{"text": "a", "score": 0.75, "metadata": {}}]
        out = apply_temporal_decay(results, weight=0.0)
        assert out[0]["blended_score"] == 0.75

    def test_empty_input(self):
        from app.retrieval.temporal import apply_temporal_decay
        assert apply_temporal_decay([]) == []


class TestQueryDecomposer:
    """Query decomposition logic."""

    def test_short_query_passthrough(self):
        from app.retrieval.decomposer import decompose_query
        result = decompose_query("hello")
        assert result == ["hello"]

    def test_disabled_passthrough(self):
        from app.retrieval import config as cfg
        orig = cfg.DECOMPOSITION_ENABLED
        try:
            cfg.DECOMPOSITION_ENABLED = False
            from app.retrieval.decomposer import decompose_query
            result = decompose_query("a" * 200, min_length=10)
            assert result == ["a" * 200]
        finally:
            cfg.DECOMPOSITION_ENABLED = orig

    def test_always_returns_list(self):
        from app.retrieval.decomposer import decompose_query
        result = decompose_query("")
        assert isinstance(result, list)
        assert len(result) >= 1


class TestReranker:
    """Cross-encoder re-ranker graceful degradation."""

    def test_empty_input(self):
        from app.retrieval.reranker import rerank
        assert rerank("query", []) == []

    def test_graceful_degradation(self):
        """When model unavailable, returns input unchanged."""
        from app.retrieval import reranker
        orig_failed = reranker._model_failed
        orig_model = reranker._model
        reranker._model_failed = True
        reranker._model = None
        try:
            docs = [
                {"text": "hello world", "score": 0.5},
                {"text": "foo bar", "score": 0.9},
            ]
            result = reranker.rerank("test", docs, top_k=2)
            assert len(result) == 2
            assert all("rerank_score" in r for r in result)
        finally:
            reranker._model_failed = orig_failed
            reranker._model = orig_model

    def test_top_k_respected(self):
        from app.retrieval import reranker
        orig_failed = reranker._model_failed
        orig_model = reranker._model
        reranker._model_failed = True
        reranker._model = None
        try:
            docs = [{"text": f"doc {i}", "score": i * 0.1} for i in range(10)]
            result = reranker.rerank("test", docs, top_k=3)
            assert len(result) == 3
        finally:
            reranker._model_failed = orig_failed
            reranker._model = orig_model


class TestRetrievalResult:
    """RetrievalResult dataclass."""

    def test_creation(self):
        from app.retrieval.orchestrator import RetrievalResult
        r = RetrievalResult(
            text="test passage",
            score=0.85,
            metadata={"source": "test.md"},
            provenance={"collection": "enterprise_knowledge", "rerank_score": 0.9},
        )
        assert r.text == "test passage"
        assert r.score == 0.85
        assert r.provenance["collection"] == "enterprise_knowledge"

    def test_defaults(self):
        from app.retrieval.orchestrator import RetrievalResult
        r = RetrievalResult(text="x", score=0.5)
        assert r.metadata == {}
        assert r.provenance == {}


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: RETRIEVAL IMPROVEMENTS WIRING
# ═══════════════════════════════════════════════════════════════════════════════

class TestRetrievalWiring:
    """Verify retrieval improvements are wired into existing KBs."""

    def test_knowledge_store_has_query_reranked(self):
        src = _read_src("knowledge_base/vectorstore.py")
        assert "def query_reranked(" in src
        assert "apply_temporal_decay" in src
        assert "rerank(" in src

    def test_philosophy_store_has_query_reranked(self):
        src = _read_src("philosophy/vectorstore.py")
        assert "def query_reranked(" in src
        assert "rerank(" in src

    def test_kb_tool_uses_reranked(self):
        src = _read_src("knowledge_base/tools.py")
        assert "query_reranked" in src

    def test_philosophy_tool_uses_reranked(self):
        src = _read_src("philosophy/rag_tool.py")
        assert "query_reranked" in src

    def test_context_injection_uses_decomposition(self):
        src = _read_src("agents/commander/context.py")
        assert "decompose_query" in src
        assert "query_reranked" in src


class TestLiteratureExpansion:
    """Fiction KB expanded to literature with genre support."""

    def test_literature_collection_name(self):
        from app.fiction_inspiration import LITERATURE_COLLECTION_NAME
        assert LITERATURE_COLLECTION_NAME == "literature_inspiration"

    def test_literary_genres_defined(self):
        from app.fiction_inspiration import LITERARY_GENRES
        assert "science_fiction" in LITERARY_GENRES
        assert "poetry" in LITERARY_GENRES
        assert "mythology" in LITERARY_GENRES
        assert "classic_novel" in LITERARY_GENRES

    def test_search_fiction_has_genre_filter(self):
        src = _read_src("fiction_inspiration.py")
        assert "genre_filter" in src

    def test_literature_metadata_type(self):
        """New ingestions should use 'literature' as source_type."""
        src = _read_src("fiction_inspiration.py")
        assert '"source_type": "literature"' in src

    def test_literature_library_dir_in_paths(self):
        from app.paths import LITERATURE_LIBRARY_DIR
        assert str(LITERATURE_LIBRARY_DIR).endswith("literature_library")

    def test_awareness_prompt_updated(self):
        from app.fiction_inspiration import FICTION_AWARENESS_PROMPT
        assert "literary" in FICTION_AWARENESS_PROMPT.lower()


class TestDialecticsWiring:
    """Philosophy dialectical graph module exists and is wired."""

    def test_dialectics_module_exists(self):
        from app.philosophy.dialectics import DialecticalGraph, get_graph
        graph = get_graph()
        assert isinstance(graph, DialecticalGraph)

    def test_dialectics_graceful_without_neo4j(self):
        from app.philosophy.dialectics import get_graph
        graph = get_graph()
        # Without Neo4j, should return empty list, not crash.
        result = graph.find_counter_arguments("virtue is sufficient for happiness")
        assert result == []
        result = graph.find_dialectical_chain("ethics")
        assert result == []

    def test_dialectics_tool_exists(self):
        src = _read_src("philosophy/dialectics_tool.py")
        assert "find_counter_argument" in src
        assert "BaseTool" in src

    def test_ingestion_detects_dialectics(self):
        src = _read_src("philosophy/ingestion.py")
        assert "_detect_dialectics" in src
        assert "_COUNTER_PATTERNS" in src
        assert "however" in src.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: NEW KNOWLEDGE BASES — CONFIG & STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════

def _load_config(module_path: str):
    """Import a config module directly, bypassing package __init__.py."""
    import importlib
    return importlib.import_module(module_path)


class TestEpistemeConfig:
    """Episteme/Research KB configuration (source-level checks)."""

    def test_collection_name(self):
        src = _read_src("episteme/config.py")
        assert '"episteme_research"' in src

    def test_chunk_size(self):
        src = _read_src("episteme/config.py")
        assert '"1200"' in src or "1200" in src

    def test_paper_types(self):
        src = _read_src("episteme/config.py")
        assert "research_paper" in src
        assert "architecture_decision" in src
        assert "failed_experiment" in src

    def test_ingestion_frontmatter(self):
        """Frontmatter extraction should work (pure Python, no chromadb)."""
        src = _read_src("episteme/ingestion.py")
        assert "def extract_frontmatter" in src
        assert "yaml.safe_load" in src

    def test_ingestion_chunking(self):
        src = _read_src("episteme/ingestion.py")
        assert "def chunk_text" in src
        assert "RecursiveCharacter" in src or "separators" in src

    def test_vectorstore_has_query_reranked(self):
        src = _read_src("episteme/vectorstore.py")
        assert "def query_reranked(" in src
        assert "rerank(" in src


class TestExperientialConfig:
    """Experiential/Journal KB configuration."""

    def test_collection_name(self):
        src = _read_src("experiential/config.py")
        assert '"experiential_journal"' in src

    def test_chunk_size_compact(self):
        src = _read_src("experiential/config.py")
        assert '"800"' in src or "800" in src

    def test_entry_types(self):
        src = _read_src("experiential/config.py")
        assert "task_reflection" in src
        assert "creative_insight" in src
        assert "error_learning" in src

    def test_valences(self):
        src = _read_src("experiential/config.py")
        assert "positive" in src
        assert "negative" in src
        assert "mixed" in src

    def test_journal_writer_exists(self):
        src = _read_src("experiential/journal_writer.py")
        assert "class JournalWriter" in src
        assert "write_post_task_reflection" in src
        assert "write_custom_entry" in src


class TestAestheticsConfig:
    """Aesthetics/Pattern Library configuration."""

    def test_collection_name(self):
        src = _read_src("aesthetics/config.py")
        assert '"aesthetic_patterns"' in src

    def test_pattern_types(self):
        src = _read_src("aesthetics/config.py")
        assert "elegant_code" in src
        assert "beautiful_prose" in src
        assert "creative_solution" in src


class TestTensionsConfig:
    """Tensions/Contradictions KB configuration."""

    def test_collection_name(self):
        src = _read_src("tensions/config.py")
        assert '"unresolved_tensions"' in src

    def test_tension_types(self):
        src = _read_src("tensions/config.py")
        assert "principle_conflict" in src
        assert "philosophy_vs_experience" in src
        assert "competing_values" in src

    def test_resolution_statuses(self):
        src = _read_src("tensions/config.py")
        assert "unresolved" in src
        assert "dissolved" in src

    def test_detector_module(self):
        src = _read_src("tensions/detector.py")
        assert "def detect_tension" in src
        assert "def detect_and_store" in src
        assert "is_tension" in src


class TestCollectionSeparation:
    """All KB collections must be distinct — verified via source inspection."""

    def test_all_collections_distinct(self):
        """Extract collection names from config source files."""
        collection_names = set()
        for cfg_path in [
            "knowledge_base/config.py",
            "philosophy/config.py",
            "episteme/config.py",
            "experiential/config.py",
            "aesthetics/config.py",
            "tensions/config.py",
        ]:
            src = _read_src(cfg_path)
            # Match all quoted strings on lines containing COLLECTION_NAME.
            for line in src.splitlines():
                if "COLLECTION_NAME" in line:
                    for m in re.finditer(r'"([a-z_]+)"', line):
                        val = m.group(1)
                        if val not in ("str",):
                            collection_names.add(val)
        # Also fiction/literature
        fic_src = _read_src("fiction_inspiration.py")
        for match in re.finditer(r'(?:LITERATURE_COLLECTION_NAME|FICTION_COLLECTION_NAME)\s*=\s*"([^"]+)"', fic_src):
            collection_names.add(match.group(1))

        assert len(collection_names) >= 7, f"Expected >=7 distinct collections, got {len(collection_names)}: {collection_names}"

    def test_all_persist_dirs_distinct(self):
        """Extract default persist dirs from config source files."""
        persist_dirs = set()
        for cfg_path in [
            "knowledge_base/config.py",
            "philosophy/config.py",
            "episteme/config.py",
            "experiential/config.py",
            "aesthetics/config.py",
            "tensions/config.py",
        ]:
            src = _read_src(cfg_path)
            # Multi-line: find /app/workspace/xxx in the CHROMA_PERSIST_DIR block.
            match = re.search(r'/app/workspace/\w+', src)
            if match:
                persist_dirs.add(match.group(0))
        assert len(persist_dirs) >= 6, f"Persist dir collision: {persist_dirs}"


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: PATHS REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestPathsRegistration:
    """All new directories registered in paths.py and _MANAGED_DIRS."""

    def test_new_dirs_exist(self):
        from app.paths import (
            EPISTEME_DIR, EPISTEME_TEXTS_DIR,
            EXPERIENTIAL_DIR, EXPERIENTIAL_ENTRIES_DIR,
            AESTHETICS_DIR, AESTHETICS_PATTERNS_DIR,
            TENSIONS_DIR, TENSIONS_ENTRIES_DIR,
            LITERATURE_LIBRARY_DIR,
        )
        assert str(EPISTEME_DIR).endswith("episteme")
        assert str(EPISTEME_TEXTS_DIR).endswith("texts")
        assert str(EXPERIENTIAL_DIR).endswith("experiential")
        assert str(EXPERIENTIAL_ENTRIES_DIR).endswith("entries")
        assert str(AESTHETICS_DIR).endswith("aesthetics")
        assert str(AESTHETICS_PATTERNS_DIR).endswith("patterns")
        assert str(TENSIONS_DIR).endswith("tensions")
        assert str(TENSIONS_ENTRIES_DIR).endswith("entries")

    def test_managed_dirs_include_new(self):
        from app.paths import _MANAGED_DIRS, EPISTEME_DIR, EXPERIENTIAL_DIR
        assert EPISTEME_DIR in _MANAGED_DIRS
        assert EXPERIENTIAL_DIR in _MANAGED_DIRS


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: TOOLS STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(_LOW_MEM, reason="CrewAI import heavy")
class TestToolStructure:
    """All new tools have correct CrewAI interface."""

    def test_episteme_tools(self):
        from app.episteme.tools import EpistemeSearchTool, get_episteme_tools
        tool = EpistemeSearchTool()
        assert tool.name == "search_research_knowledge"
        assert hasattr(tool, "_run")
        tools = get_episteme_tools()
        assert len(tools) == 1  # Search only (read-only for safety)

    def test_experiential_tools_reader(self):
        from app.experiential.tools import get_experiential_tools
        tools = get_experiential_tools("reader")
        assert len(tools) == 1
        assert tools[0].name == "search_journal"

    def test_experiential_tools_writer(self):
        from app.experiential.tools import get_experiential_tools
        tools = get_experiential_tools("writer")
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert "search_journal" in names
        assert "write_journal_entry" in names

    def test_aesthetic_tools(self):
        from app.aesthetics.tools import get_aesthetic_tools
        tools = get_aesthetic_tools("coder")
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert "search_aesthetic_patterns" in names
        assert "flag_aesthetic_pattern" in names

    def test_tension_tools(self):
        from app.tensions.tools import get_tension_tools
        tools = get_tension_tools("critic")
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert "search_tensions" in names
        assert "record_tension" in names

    def test_dialectics_tool(self):
        from app.philosophy.dialectics_tool import FindCounterArgumentTool
        tool = FindCounterArgumentTool()
        assert tool.name == "find_counter_argument"


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: CONTEXT INJECTION WIRING
# ═══════════════════════════════════════════════════════════════════════════════

class TestContextInjectionWiring:
    """New context loaders exist and are wired into the orchestrator."""

    def test_context_loaders_defined(self):
        src = _read_src("agents/commander/context.py")
        assert "def _load_episteme_context(" in src
        assert "def _load_experiential_context(" in src
        assert "def _load_aesthetic_context(" in src
        assert "def _load_tensions_context(" in src

    def test_context_loaders_use_correct_kbs(self):
        src = _read_src("agents/commander/context.py")
        assert "app.episteme.vectorstore" in src
        assert "app.experiential.vectorstore" in src
        assert "app.aesthetics.vectorstore" in src
        assert "app.tensions.vectorstore" in src

    def test_context_loaders_tagged(self):
        """Results must be wrapped in provenance tags."""
        src = _read_src("agents/commander/context.py")
        assert "<episteme_passage" in src
        assert "<journal_entry" in src
        assert "<aesthetic_pattern" in src
        assert "<tension" in src

    def test_orchestrator_fetches_new_contexts(self):
        src = _read_src("agents/commander/orchestrator.py")
        assert "_load_episteme_context" in src
        assert "_load_experiential_context" in src
        assert "_load_aesthetic_context" in src
        assert "_load_tensions_context" in src
        assert "f_episteme" in src
        assert "f_experiential" in src


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: EVOLUTION WIRING
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvolutionWiring:
    """New KBs wired into evolution pipeline."""

    def test_evolution_context_includes_episteme(self):
        src = _read_src("evolution.py")
        assert "app.episteme.vectorstore" in src
        assert "Research Insights" in src

    def test_evolution_context_includes_experiential(self):
        src = _read_src("evolution.py")
        assert "app.experiential.vectorstore" in src
        assert "Past Experiences" in src

    def test_evolution_context_includes_tensions(self):
        src = _read_src("evolution.py")
        assert "app.tensions.vectorstore" in src
        assert "Growth Edges" in src


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: AGENT TOOL ASSIGNMENT
# ═══════════════════════════════════════════════════════════════════════════════

class TestAgentToolAssignment:
    """Each agent gets the correct new tools — no cross-contamination."""

    def test_writer_has_journal_and_aesthetics(self):
        src = _read_src("agents/writer.py")
        assert "get_experiential_tools" in src
        assert "get_aesthetic_tools" in src

    def test_coder_has_aesthetics(self):
        src = _read_src("agents/coder.py")
        assert "get_aesthetic_tools" in src

    def test_researcher_has_episteme_and_journal(self):
        src = _read_src("agents/researcher.py")
        assert "get_episteme_tools" in src
        assert "get_experiential_tools" in src

    def test_self_improver_has_episteme_tensions_journal_dialectics(self):
        src = _read_src("crews/self_improvement_crew.py")
        assert "get_episteme_tools" in src
        assert "get_tension_tools" in src
        assert "get_experiential_tools" in src
        assert "FindCounterArgumentTool" in src

    # test_researcher_has_no_tensions removed: the original architectural
    # rule ("researcher should stay factual — no tensions tool") was
    # deliberately reversed. The researcher's full tier now includes
    # tensions tools (search/record) — see app/agents/researcher.py:134.
    # Per the commit "Researcher (full): +tensions (search/record), +OCR".

    def test_coder_has_no_episteme(self):
        """Coder doesn't need research theory."""
        src = _read_src("agents/coder.py")
        assert "get_episteme_tools" not in src


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: AGENT FEEDBACK LOOP
# ═══════════════════════════════════════════════════════════════════════════════

class TestAgentFeedbackLoop:
    """Post-task journal writing wired into orchestrator."""

    # test_orchestrator_writes_journal removed: journal writing was
    # deliberately decoupled from the orchestrator and now happens via
    # Firebase event listeners (app/firebase/listeners.py) and affect
    # episode tracking (app/affect/episodes.py). The JournalWriter class
    # itself still exists at app/experiential/journal_writer.py and is
    # exercised by test_journal_writer_structure below.

    def test_journal_writer_structure(self):
        """JournalWriter class has required methods."""
        src = _read_src("experiential/journal_writer.py")
        assert "class JournalWriter" in src
        assert "def write_post_task_reflection" in src
        assert "def write_custom_entry" in src
        # Verify trivial-task skip logic
        assert "difficulty <= 2" in src

    def test_tension_detector_structure(self):
        """Tension detector has required functions."""
        src = _read_src("tensions/detector.py")
        assert "def detect_tension" in src
        assert "def detect_and_store" in src
        assert "is_tension" in src


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: FIREBASE INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestFirebaseIntegration:
    """New KBs have Firebase status reporters."""

    def test_publish_has_new_reporters(self):
        src = _read_src("firebase/publish.py")
        assert "def report_episteme_kb" in src
        assert "def report_experiential_kb" in src
        assert "def report_aesthetics_kb" in src
        assert "def report_tensions_kb" in src

    def test_reporters_write_correct_collections(self):
        src = _read_src("firebase/publish.py")
        assert '"episteme_kb"' in src
        assert '"experiential_kb"' in src
        assert '"aesthetics_kb"' in src
        assert '"tensions_kb"' in src

    def test_main_registers_new_routers(self):
        src = _read_src("main.py")
        assert "episteme_router" in src
        assert "experiential_router" in src
        assert "aesthetics_router" in src
        assert "tensions_router" in src


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: API ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

class TestAPIRoutes:
    """Each new KB has API endpoints."""

    def test_episteme_api(self):
        src = _read_src("episteme/api.py")
        assert "/upload" in src
        assert "/stats" in src
        assert "/texts" in src
        assert "/reingest" in src

    def test_experiential_api(self):
        src = _read_src("experiential/api.py")
        assert "/stats" in src
        assert "/recent" in src

    def test_aesthetics_api(self):
        src = _read_src("aesthetics/api.py")
        assert "/stats" in src
        assert "/patterns" in src

    def test_tensions_api(self):
        src = _read_src("tensions/api.py")
        assert "/stats" in src
        assert "/unresolved" in src


# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-CUTTING: EPISTEMIC STATUS INVARIANTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestEpistemicBoundaries:
    """Each KB has correct immutable epistemic status."""

    def test_episteme_status(self):
        src = _read_src("episteme/vectorstore.py")
        # Episteme doesn't enforce a single epistemic status per-chunk —
        # it varies by paper (theoretical vs empirical). Check it's in metadata.
        assert "epistemic_status" in src

    def test_experiential_status_immutable(self):
        src = _read_src("experiential/vectorstore.py")
        assert "subjective/phenomenological" in src

    def test_aesthetics_status_immutable(self):
        src = _read_src("aesthetics/vectorstore.py")
        assert "evaluative/subjective" in src

    def test_tensions_status_immutable(self):
        src = _read_src("tensions/vectorstore.py")
        assert "unresolved/dialectical" in src

    def test_literature_status_immutable(self):
        src = _read_src("fiction_inspiration.py")
        assert '"source_type": "literature"' in src
        assert '"epistemic_status": "imaginary"' in src


# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-CUTTING: SAFETY INVARIANT — SELF-IMPROVER CANNOT MODIFY EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestSafetyInvariants:
    """Self-Improver gets read-only access to episteme — no ingest tool."""

    def test_self_improver_no_ingest_tool(self):
        src = _read_src("crews/self_improvement_crew.py")
        assert "EpistemeIngestTool" not in src

    def test_episteme_tools_read_only(self):
        """get_episteme_tools() should return search only."""
        src = _read_src("episteme/tools.py")
        # The function returns [EpistemeSearchTool()] — no ingest.
        assert "EpistemeSearchTool()" in src
        # Verify the function signature.
        assert "def get_episteme_tools" in src


# ═══════════════════════════════════════════════════════════════════════════════
# END-TO-END: FULL PIPELINE SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════════════

class TestFullPipelineSmokeTest:
    """Verify the entire system can import and initialize without crashes."""

    def test_pure_python_modules_importable(self):
        """Modules that don't need chromadb/crewai must import."""
        modules = [
            "app.retrieval",
            "app.retrieval.config",
            "app.retrieval.reranker",
            "app.retrieval.decomposer",
            "app.retrieval.temporal",
            "app.retrieval.orchestrator",
            "app.philosophy.dialectics",
        ]
        for mod_name in modules:
            mod = __import__(mod_name, fromlist=["_"])
            assert mod is not None, f"Failed to import {mod_name}"

    def test_all_source_files_exist(self):
        """Every new source file must exist on disk."""
        files = [
            "retrieval/__init__.py", "retrieval/config.py", "retrieval/reranker.py",
            "retrieval/decomposer.py", "retrieval/temporal.py", "retrieval/orchestrator.py",
            "episteme/__init__.py", "episteme/config.py", "episteme/ingestion.py",
            "episteme/vectorstore.py", "episteme/tools.py", "episteme/api.py",
            "experiential/__init__.py", "experiential/config.py",
            "experiential/vectorstore.py", "experiential/journal_writer.py",
            "experiential/tools.py", "experiential/api.py",
            "aesthetics/__init__.py", "aesthetics/config.py",
            "aesthetics/vectorstore.py", "aesthetics/tools.py", "aesthetics/api.py",
            "tensions/__init__.py", "tensions/config.py",
            "tensions/vectorstore.py", "tensions/detector.py",
            "tensions/tools.py", "tensions/api.py",
            "philosophy/dialectics.py", "philosophy/dialectics_tool.py",
        ]
        for f in files:
            path = _APP_DIR / f
            assert path.exists(), f"Missing file: {path}"

    def test_orchestrator_instantiation(self):
        from app.retrieval import RetrievalOrchestrator, RetrievalConfig
        orch = RetrievalOrchestrator(RetrievalConfig(
            rerank_enabled=False,  # Skip model loading
            decomposition_enabled=False,
        ))
        assert orch is not None
        assert orch.config.rerank_enabled is False

    def test_business_store_file_exists(self):
        path = _APP_DIR / "knowledge_base" / "business_store.py"
        assert path.exists()


# ═══════════════════════════════════════════════════════════════════════════════
# PER-BUSINESS KNOWLEDGE BASES
# ═══════════════════════════════════════════════════════════════════════════════

class TestBusinessKBStructure:
    """Business KB registry source-level verification."""

    def test_registry_module_exists(self):
        src = _read_src("knowledge_base/business_store.py")
        assert "class BusinessKBRegistry" in src
        assert "def get_or_create" in src
        assert "def create_store" in src
        assert "def list_businesses" in src
        assert "def query_business" in src
        assert "def discover_existing" in src

    def test_collection_naming(self):
        src = _read_src("knowledge_base/business_store.py")
        assert 'BUSINESS_KB_PREFIX = "biz_kb_"' in src

    def test_thread_safety(self):
        src = _read_src("knowledge_base/business_store.py")
        assert "threading.Lock()" in src


class TestBusinessKBWiring:
    """Business KB wired into project creation and context injection."""

    def test_project_isolation_creates_kb(self):
        src = _read_src("project_isolation.py")
        assert "business_store" in src
        assert "create_store" in src

    def test_control_plane_creates_kb(self):
        src = _read_src("control_plane/projects.py")
        assert "business_store" in src
        assert "create_store" in src

    def test_context_injection_queries_business_kb(self):
        src = _read_src("agents/commander/context.py")
        assert "business_store" in src
        assert "active_business" in src
        assert "biz_store" in src or "get_registry" in src

    def test_kb_tool_queries_business_kb(self):
        src = _read_src("knowledge_base/tools.py")
        assert "business_store" in src
        assert "active_business" in src or "project_isolation" in src

    def test_context_labels_provenance(self):
        """KB results must show whether they came from global or business KB."""
        src = _read_src("agents/commander/context.py")
        assert "_kb_source" in src
        assert "global" in src


class TestBusinessKBAPI:
    """Business KB API endpoints exist."""

    def test_business_endpoints(self):
        src = _read_src("api/kb.py")
        assert "/businesses" in src
        assert "/business/{business_id}/status" in src
        assert "/business/{business_id}/upload" in src
        assert "/business/{business_id}/remove" in src
        assert "/business/{business_id}/reset" in src

    def test_firebase_queue_supports_business(self):
        src = _read_src("firebase/listeners.py")
        assert "business_id" in src
        assert "get_registry" in src or "business_store" in src

    def test_firebase_publish_business_kb(self):
        src = _read_src("firebase/publish.py")
        assert "def report_business_kb" in src
        assert "def report_all_business_kbs" in src
        assert "biz_kb_" in src


class TestBusinessKBDashboard:
    """Dashboard has business KB UI."""

    def test_react_has_business_section(self):
        kb_src = Path(__file__).parent.parent / "dashboard-react" / "src" / "components" / "KnowledgeBases.tsx"
        src = kb_src.read_text()
        assert "BusinessKBSection" in src
        assert "BusinessKBCard" in src
        # Route literals were factored out into the endpoints module — the
        # component now references them via endpoints.kbBusinessUpload(...)
        # and keys.kbBusinesses. Assert the wiring rather than literal URLs.
        assert "kbBusinessUpload" in src
        assert "kbBusinesses" in src
        endpoints_src = (Path(__file__).parent.parent / "dashboard-react"
                         / "src" / "api" / "endpoints.ts").read_text()
        assert "/kb/businesses" in endpoints_src
        assert "/kb/business/" in endpoints_src

    def test_monitor_has_business_selector(self):
        monitor_src = Path(__file__).parent.parent / "dashboard" / "public" / "index.html"
        src = monitor_src.read_text()
        assert "kb-business" in src
        assert "business_id" in src
        assert "biz-kb-list" in src


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIT FIXES VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestAuditFixes:
    """Verify all issues from the KB architecture audit are fixed."""

    def test_import_paths_correct(self):
        """All LLM factory imports must use app.llm_factory (not app.llm.factory)."""
        for f in ["retrieval/decomposer.py", "experiential/journal_writer.py", "tensions/detector.py"]:
            src = _read_src(f)
            assert "app.llm.factory" not in src, f"{f} still uses wrong import path"
            assert "app.llm_factory" in src, f"{f} missing correct import"

    # test_orchestrator_journal_uses_result removed: same reason as
    # test_orchestrator_writes_journal above — the JournalWriter call site
    # was deliberately removed from the orchestrator. The audit fix this
    # test guarded ("use result not crew_result") is no longer applicable.

    def test_context_pruning_knows_new_blocks(self):
        """_prune_context must recognize all KB block headers."""
        src = _read_src("agents/commander/context.py")
        assert '"RESEARCH CONTEXT"' in src
        assert '"EXPERIENTIAL CONTEXT"' in src
        assert '"QUALITY PATTERNS"' in src
        assert '"UNRESOLVED TENSIONS"' in src

    def test_episteme_api_reports_firebase(self):
        """Episteme upload must call _report_async."""
        src = _read_src("episteme/api.py")
        assert "_report_async()" in src
        assert "def _report_async" in src
        assert "report_episteme_kb" in src

    def test_firebase_reporter_exports_all(self):
        """firebase_reporter.py must re-export all new report functions."""
        src = (_APP_DIR.parent / "app" / "firebase_reporter.py").read_text()
        for fn in ["report_episteme_kb", "report_experiential_kb",
                    "report_aesthetics_kb", "report_tensions_kb",
                    "report_business_kb", "report_all_business_kbs"]:
            assert fn in src, f"Missing re-export: {fn}"

    def test_detect_and_store_is_wired(self):
        """detect_and_store must be called from context injection."""
        src = _read_src("agents/commander/context.py")
        assert "detect_and_store" in src

    def test_kb_delete_handles_business_id(self):
        """Firebase KB queue delete action must handle business_id."""
        src = _read_src("firebase/listeners.py")
        # Find the delete action block
        idx = src.find('if action == "delete"')
        assert idx > 0
        block = src[idx:idx+600]
        assert "business_id" in block, "Delete action ignores business_id"

    def test_vectorstores_log_dimension_errors(self):
        """Aesthetic/tension vectorstores must log dimension mismatches."""
        for f in ["aesthetics/vectorstore.py", "tensions/vectorstore.py"]:
            src = _read_src(f)
            assert "logger.warning" in src, f"{f} has silent error handling"
            assert "dimension" in src.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
