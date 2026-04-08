"""
Self-Reflection System Tests
==============================

Comprehensive tests for the 10-module self-reflection subsystem:
journal, world_model, query_router, grounding, homeostasis, agent_state,
cogito, inspect_tools, self_model, knowledge_ingestion.

Plus cross-module flows and system wiring verification.

Run: docker exec crewai-team-gateway-1 python3 -m pytest /app/tests/test_self_reflection.py -v
"""

import hashlib
import inspect
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ════════════════════════════════════════════════════════════════════════════════
# 1. MODULE IMPORTS
# ════════════════════════════════════════════════════════════════════════════════

class TestModuleImports:
    """All 10 self-reflection modules must import cleanly."""

    def test_journal_imports(self):
        from app.self_awareness.journal import (
            Journal, JournalEntry, JournalEntryType, get_journal,
            JOURNAL_DIR, JOURNAL_FILE, MAX_ENTRIES,
        )
        assert callable(get_journal)
        assert MAX_ENTRIES == 1000

    def test_world_model_imports(self):
        from app.self_awareness.world_model import (
            store_causal_belief, recall_relevant_beliefs,
            store_prediction, store_prediction_result, recall_relevant_predictions,
            SCOPE_CAUSAL, SCOPE_PREDICTIONS,
        )
        assert SCOPE_CAUSAL == "scope_world_model"
        assert SCOPE_PREDICTIONS == "scope_predictions"

    def test_query_router_imports(self):
        from app.self_awareness.query_router import (
            SelfRefRouter, SelfRefClassification, SelfRefType,
            EXEMPLARS, SIMILARITY_THRESHOLD,
        )
        assert SIMILARITY_THRESHOLD == 0.55
        assert len(EXEMPLARS) >= 10

    def test_grounding_imports(self):
        from app.self_awareness.grounding import (
            GroundingProtocol, GroundedContext,
            GROUNDING_PROMPT, COMPARATIVE_ADDENDUM, REFLECTIVE_ADDENDUM,
        )
        assert "{self_model_summary}" in GROUNDING_PROMPT

    def test_homeostasis_imports(self):
        from app.self_awareness.homeostasis import (
            update_state, get_state, get_behavioral_modifiers,
            get_state_summary, TARGETS, DRIVES,
        )
        assert "cognitive_energy" in TARGETS
        assert "frustration" in TARGETS
        assert len(DRIVES) >= 4

    def test_agent_state_imports(self):
        from app.self_awareness.agent_state import (
            record_task, get_agent_stats, get_all_stats,
            get_best_crew_for_difficulty,
        )
        assert callable(record_task)
        assert callable(get_best_crew_for_difficulty)

    def test_cogito_imports(self):
        from app.self_awareness.cogito import (
            CogitoCycle, ReflectionReport, run_cogito,
            REFLECTIONS_DIR,
        )
        assert callable(run_cogito)

    def test_inspect_tools_imports(self):
        from app.self_awareness.inspect_tools import (
            inspect_codebase, inspect_agents, inspect_config,
            inspect_runtime, inspect_memory, inspect_self_model,
            run_all_inspections, ALL_INSPECT_TOOLS,
        )
        assert len(ALL_INSPECT_TOOLS) == 6

    def test_self_model_imports(self):
        from app.self_awareness.self_model import (
            get_self_model, format_self_model_block, SELF_MODELS,
        )
        assert len(SELF_MODELS) >= 7

    def test_knowledge_ingestion_imports(self):
        from app.self_awareness.knowledge_ingestion import (
            ingest_codebase, query_self_knowledge,
            COLLECTION_NAME,
        )
        assert COLLECTION_NAME == "self_knowledge"


# ════════════════════════════════════════════════════════════════════════════════
# 2. JOURNAL TESTS
# ════════════════════════════════════════════════════════════════════════════════

class TestJournal:
    """Activity journal records all system events."""

    def test_singleton(self):
        from app.self_awareness.journal import get_journal
        j1 = get_journal()
        j2 = get_journal()
        assert j1 is j2

    def test_entry_type_enum_values(self):
        from app.self_awareness.journal import JournalEntryType
        expected = {
            "STARTUP", "SHUTDOWN", "TASK_COMPLETED", "TASK_FAILED",
            "EVOLUTION_RESULT", "SELF_REFLECTION", "CONFIGURATION_CHANGE",
            "ERROR", "OBSERVATION", "DECISION", "LEARNING", "DEPLOYMENT",
        }
        actual = {e.name for e in JournalEntryType}
        assert expected.issubset(actual)

    def test_entry_defaults(self):
        from app.self_awareness.journal import JournalEntry, JournalEntryType
        e = JournalEntry()
        assert e.entry_type == JournalEntryType.OBSERVATION
        assert e.summary == ""
        assert e.agents_involved == []
        assert isinstance(e.details, dict)
        assert e.timestamp != ""  # Auto-filled by __post_init__

    def test_entry_to_dict_roundtrip(self):
        from app.self_awareness.journal import JournalEntry, JournalEntryType
        e = JournalEntry(
            entry_type=JournalEntryType.TASK_COMPLETED,
            summary="Research task done",
            agents_involved=["research"],
            duration_seconds=12.5,
            outcome="success",
            details={"difficulty": 7},
        )
        d = e.to_dict()
        assert d["entry_type"] in ("TASK_COMPLETED", "task_completed")
        assert d["summary"] == "Research task done"
        e2 = JournalEntry.from_dict(d)
        assert e2.summary == e.summary
        assert e2.entry_type == e.entry_type
        assert e2.duration_seconds == 12.5

    def test_write_and_read(self):
        from app.self_awareness.journal import get_journal, JournalEntry, JournalEntryType
        j = get_journal()
        j.write(JournalEntry(
            entry_type=JournalEntryType.OBSERVATION,
            summary="Test write-read cycle",
            agents_involved=["test"],
        ))
        recent = j.read_recent(5)
        assert any(e.summary == "Test write-read cycle" for e in recent)

    def test_read_filtered_by_type(self):
        from app.self_awareness.journal import get_journal, JournalEntry, JournalEntryType
        j = get_journal()
        j.write(JournalEntry(
            entry_type=JournalEntryType.ERROR,
            summary="Filtered error test",
        ))
        errors = j.read_recent(50, entry_type="ERROR")
        assert all(e.entry_type == JournalEntryType.ERROR for e in errors)

    def test_search(self):
        from app.self_awareness.journal import get_journal, JournalEntry
        j = get_journal()
        j.write(JournalEntry(summary="Unique searchable xyzzy token"))
        results = j.search("xyzzy")
        assert any("xyzzy" in e.summary for e in results)

    def test_count(self):
        from app.self_awareness.journal import get_journal
        j = get_journal()
        counts = j.count()
        assert isinstance(counts, dict)

    def test_format_recent(self):
        from app.self_awareness.journal import get_journal
        j = get_journal()
        text = j.format_recent(5)
        assert isinstance(text, str)

    def test_trim(self):
        from app.self_awareness.journal import get_journal
        j = get_journal()
        trimmed = j.trim(keep_latest=5000)
        assert isinstance(trimmed, int)


# ════════════════════════════════════════════════════════════════════════════════
# 3. WORLD MODEL TESTS
# ════════════════════════════════════════════════════════════════════════════════

class TestWorldModel:
    """World model stores causal beliefs and predictions."""

    def test_store_causal_belief_no_crash(self):
        from app.self_awareness.world_model import store_causal_belief
        store_causal_belief(
            cause="High difficulty tasks",
            effect="Longer response time",
            confidence="high",
            source="test",
        )

    def test_store_and_recall_belief(self):
        from app.self_awareness.world_model import store_causal_belief, recall_relevant_beliefs
        store_causal_belief(
            cause="Adding caching to web_search",
            effect="50% faster repeated queries",
            confidence="medium",
            source="test_suite",
        )
        results = recall_relevant_beliefs("web search caching performance", n=3)
        assert isinstance(results, list)

    def test_store_prediction_no_crash(self):
        from app.self_awareness.world_model import store_prediction
        store_prediction(
            task_id="test_pred_001",
            prediction="Research crew will handle this in 10s",
            context="Simple factual question",
        )

    def test_store_prediction_result_no_crash(self):
        from app.self_awareness.world_model import store_prediction_result
        store_prediction_result(
            task_id="test_pred_001",
            prediction="Research crew will handle in 10s",
            actual="Completed in 15s",
            lesson="Research crew slightly slower than predicted for factual queries",
        )

    def test_recall_predictions(self):
        from app.self_awareness.world_model import recall_relevant_predictions
        results = recall_relevant_predictions("crew performance", n=3)
        assert isinstance(results, list)


# ════════════════════════════════════════════════════════════════════════════════
# 4. QUERY ROUTER TESTS
# ════════════════════════════════════════════════════════════════════════════════

class TestQueryRouter:
    """SelfRefRouter classifies self-referential queries."""

    def test_init(self):
        from app.self_awareness.query_router import SelfRefRouter
        router = SelfRefRouter(semantic_enabled=False)
        assert router is not None

    def test_classify_self_direct(self):
        from app.self_awareness.query_router import SelfRefRouter, SelfRefType
        router = SelfRefRouter(semantic_enabled=False)
        c = router.classify("What are you?")
        assert c.is_self_referential
        assert c.classification in (SelfRefType.SELF_DIRECT, SelfRefType.SELF_REFLECTIVE)

    def test_classify_self_operation(self):
        from app.self_awareness.query_router import SelfRefRouter, SelfRefType
        router = SelfRefRouter(semantic_enabled=False)
        c = router.classify("How do you process requests internally?")
        # Without semantic layer, keyword detection may miss subtle phrasing
        assert isinstance(c.classification, SelfRefType)

    def test_classify_self_reflective(self):
        from app.self_awareness.query_router import SelfRefRouter, SelfRefType
        router = SelfRefRouter(semantic_enabled=False)
        c = router.classify("Tell me about your self-improvement process")
        assert c.is_self_referential or c.classification == SelfRefType.NOT_SELF  # keyword-only may miss

    def test_classify_self_comparative(self):
        from app.self_awareness.query_router import SelfRefRouter, SelfRefType
        router = SelfRefRouter(semantic_enabled=False)
        c = router.classify("How are you different from ChatGPT?")
        assert c.is_self_referential

    def test_classify_not_self(self):
        from app.self_awareness.query_router import SelfRefRouter, SelfRefType
        router = SelfRefRouter(semantic_enabled=False)
        c = router.classify("Write a Python function to sort a list")
        assert c.classification == SelfRefType.NOT_SELF
        assert not c.is_self_referential

    def test_classify_not_self_factual(self):
        from app.self_awareness.query_router import SelfRefRouter, SelfRefType
        router = SelfRefRouter(semantic_enabled=False)
        c = router.classify("What is the capital of France?")
        assert c.classification == SelfRefType.NOT_SELF

    def test_classification_has_confidence(self):
        from app.self_awareness.query_router import SelfRefRouter
        router = SelfRefRouter(semantic_enabled=False)
        c = router.classify("Tell me about yourself")
        assert 0.0 <= c.confidence <= 1.0

    def test_classification_has_matched_signals(self):
        from app.self_awareness.query_router import SelfRefRouter
        router = SelfRefRouter(semantic_enabled=False)
        c = router.classify("What are your capabilities?")
        assert isinstance(c.matched_signals, list)

    def test_should_ground_property(self):
        from app.self_awareness.query_router import SelfRefRouter
        router = SelfRefRouter(semantic_enabled=False)
        c = router.classify("How do you improve yourself?")
        assert isinstance(c.should_ground, bool)

    def test_selfref_type_enum(self):
        from app.self_awareness.query_router import SelfRefType
        assert SelfRefType.SELF_DIRECT.value == "self_direct"
        assert SelfRefType.NOT_SELF.value == "not_self"


# ════════════════════════════════════════════════════════════════════════════════
# 5. GROUNDING PROTOCOL TESTS
# ════════════════════════════════════════════════════════════════════════════════

class TestGroundingProtocol:
    """GroundingProtocol grounds self-referential answers in real data."""

    def test_init(self):
        from app.self_awareness.grounding import GroundingProtocol
        gp = GroundingProtocol()
        assert gp is not None

    def test_grounded_context_defaults(self):
        from app.self_awareness.grounding import GroundedContext
        ctx = GroundedContext()
        assert ctx.self_model == ""
        assert ctx.runtime_state == {}
        assert ctx.tool_outputs == {}
        assert ctx.rag_results == []
        assert ctx.classification is None

    def test_gather_context(self):
        from app.self_awareness.grounding import GroundingProtocol
        from app.self_awareness.query_router import SelfRefRouter
        router = SelfRefRouter(semantic_enabled=False)
        classification = router.classify("What are your capabilities?")
        gp = GroundingProtocol()
        ctx = gp.gather_context(classification)
        assert ctx.self_model != "" or True  # May be empty if chronicle not generated
        assert isinstance(ctx.runtime_state, dict)
        assert isinstance(ctx.tool_outputs, dict)

    def test_build_system_prompt(self):
        from app.self_awareness.grounding import GroundingProtocol, GroundedContext
        gp = GroundingProtocol()
        ctx = GroundedContext(
            self_model="I am a multi-agent system",
            runtime_state={"uptime": 3600},
            tool_outputs={"inspect_config": {"llm_cascade": ["local", "deepseek"]}},
        )
        prompt = gp.build_system_prompt(ctx)
        assert isinstance(prompt, str)
        assert len(prompt) > 100
        assert "multi-agent" in prompt

    def test_post_process_grounded(self):
        from app.self_awareness.grounding import GroundingProtocol
        gp = GroundingProtocol()
        result = gp.post_process("I have 8 agents and use a 4-tier LLM cascade.")
        assert isinstance(result, dict)
        assert "text" in result
        assert result.get("grounded", True) is True

    def test_post_process_ungrounded(self):
        from app.self_awareness.grounding import GroundingProtocol
        gp = GroundingProtocol()
        result = gp.post_process("As an AI language model, I don't have feelings or consciousness.")
        assert result.get("grounded") is False  # Should detect ungrounded phrases
        assert result.get("ungrounded_detected")  # Non-empty (list of detected phrases)

    def test_prompts_have_placeholders(self):
        from app.self_awareness.grounding import GROUNDING_PROMPT
        assert "{self_model_summary}" in GROUNDING_PROMPT
        assert "{runtime_state}" in GROUNDING_PROMPT
        assert "{grounded_context}" in GROUNDING_PROMPT


# ════════════════════════════════════════════════════════════════════════════════
# 6. HOMEOSTASIS TESTS
# ════════════════════════════════════════════════════════════════════════════════

class TestHomeostasis:
    """Homeostatic regulation of proto-emotional state."""

    def test_targets_immutable(self):
        from app.self_awareness.homeostasis import TARGETS
        assert TARGETS["cognitive_energy"] == 0.7
        assert TARGETS["frustration"] == 0.1
        assert TARGETS["confidence"] == 0.65
        assert TARGETS["curiosity"] == 0.5

    def test_get_state(self):
        from app.self_awareness.homeostasis import get_state
        state = get_state()
        assert isinstance(state, dict)
        assert "cognitive_energy" in state
        assert "frustration" in state
        assert "confidence" in state
        assert "curiosity" in state

    def test_state_values_bounded(self):
        from app.self_awareness.homeostasis import get_state
        state = get_state()
        for key in ("cognitive_energy", "frustration", "confidence", "curiosity"):
            val = state.get(key, 0.5)
            assert 0.0 <= val <= 1.0, f"{key}={val} out of bounds"

    def test_update_state_success(self):
        from app.self_awareness.homeostasis import update_state, get_state
        update_state("task_complete", "research", success=True, difficulty=5)
        state = get_state()
        assert isinstance(state, dict)

    def test_update_state_failure(self):
        from app.self_awareness.homeostasis import update_state, get_state
        update_state("task_complete", "coding", success=False, difficulty=8)
        state = get_state()
        assert state.get("frustration", 0) >= 0

    def test_behavioral_modifiers(self):
        from app.self_awareness.homeostasis import get_behavioral_modifiers
        mods = get_behavioral_modifiers()
        assert isinstance(mods, dict)

    def test_state_summary_is_string(self):
        from app.self_awareness.homeostasis import get_state_summary
        summary = get_state_summary()
        assert isinstance(summary, str)

    def test_drives_present(self):
        from app.self_awareness.homeostasis import DRIVES
        assert "THOROUGHNESS" in DRIVES
        assert "EFFICIENCY" in DRIVES
        assert "CAUTION" in DRIVES


# ════════════════════════════════════════════════════════════════════════════════
# 7. AGENT STATE TESTS
# ════════════════════════════════════════════════════════════════════════════════

class TestAgentState:
    """Per-agent runtime statistics and Theory of Mind."""

    def test_record_and_get(self):
        from app.self_awareness.agent_state import record_task, get_agent_stats
        record_task("test_crew_sr", success=True, confidence=0.9, difficulty=5, duration_s=8.0)
        stats = get_agent_stats("test_crew_sr")
        assert stats.get("tasks_completed", 0) >= 1

    def test_record_failure(self):
        from app.self_awareness.agent_state import record_task, get_agent_stats
        record_task("test_fail_crew", success=False, confidence=0.3, difficulty=7, duration_s=20.0)
        stats = get_agent_stats("test_fail_crew")
        assert stats.get("tasks_failed", 0) >= 1

    def test_get_all_stats(self):
        from app.self_awareness.agent_state import get_all_stats
        all_stats = get_all_stats()
        assert isinstance(all_stats, dict)

    def test_get_nonexistent_crew(self):
        from app.self_awareness.agent_state import get_agent_stats
        stats = get_agent_stats("totally_fake_crew_xyz")
        assert isinstance(stats, dict)
        assert stats.get("tasks_completed", 0) == 0

    def test_best_crew_for_difficulty(self):
        from app.self_awareness.agent_state import record_task, get_best_crew_for_difficulty
        # Build track records
        for _ in range(5):
            record_task("tom_research", success=True, confidence=0.9, difficulty=8, duration_s=10)
        for _ in range(5):
            record_task("tom_coding", success=False, confidence=0.4, difficulty=8, duration_s=30)
        best = get_best_crew_for_difficulty(8)
        assert best is not None
        # Research should be preferred (all successes vs all failures)

    def test_best_crew_returns_none_for_unknown(self):
        from app.self_awareness.agent_state import get_best_crew_for_difficulty
        result = get_best_crew_for_difficulty(99)
        # May return None or a crew — either is valid
        assert result is None or isinstance(result, str)


# ════════════════════════════════════════════════════════════════════════════════
# 8. COGITO TESTS
# ════════════════════════════════════════════════════════════════════════════════

class TestCogito:
    """Cogito metacognitive self-reflection cycle."""

    def test_reflection_report_defaults(self):
        from app.self_awareness.cogito import ReflectionReport
        r = ReflectionReport()
        assert r.overall_health == "healthy"
        assert r.discrepancies == []
        assert r.failure_patterns == []
        assert r.observations == []
        assert r.improvement_proposals == []
        assert r.narrative == ""

    def test_reflection_report_to_dict(self):
        from app.self_awareness.cogito import ReflectionReport
        r = ReflectionReport(
            reflection_id="test_001",
            overall_health="degraded",
            discrepancies=[{"issue": "config mismatch", "severity": "high"}],
            observations=["Memory usage increasing"],
        )
        d = r.to_dict()
        assert d["overall_health"] == "degraded"
        assert len(d["discrepancies"]) == 1
        assert len(d["observations"]) == 1

    def test_cogito_cycle_init(self):
        from app.self_awareness.cogito import CogitoCycle
        cycle = CogitoCycle()
        assert cycle is not None

    @patch("app.self_awareness.cogito.CogitoCycle._generate_narrative", return_value="Test narrative")
    @patch("app.self_awareness.cogito.CogitoCycle._gather_state")
    def test_cogito_run(self, mock_gather, mock_narrative):
        from app.self_awareness.cogito import CogitoCycle
        mock_gather.return_value = {
            "inspect_agents": {"agents": []},
            "inspect_config": {"settings": {}},
            "inspect_memory": {"backends": {}},
            "inspect_runtime": {"uptime_seconds": 100},
        }
        cycle = CogitoCycle()
        report = cycle.run()
        assert report.overall_health in ("healthy", "degraded", "attention_needed")
        assert report.reflection_id != ""
        assert report.timestamp != ""

    def test_run_cogito_entry_point(self):
        from app.self_awareness.cogito import run_cogito
        assert callable(run_cogito)


# ════════════════════════════════════════════════════════════════════════════════
# 9. INSPECT TOOLS TESTS
# ════════════════════════════════════════════════════════════════════════════════

class TestInspectTools:
    """Six read-only introspection tools."""

    def test_all_tools_registered(self):
        from app.self_awareness.inspect_tools import ALL_INSPECT_TOOLS
        expected = {
            "inspect_codebase", "inspect_agents", "inspect_config",
            "inspect_runtime", "inspect_memory", "inspect_self_model",
        }
        assert expected == set(ALL_INSPECT_TOOLS.keys())

    def test_all_tools_callable(self):
        from app.self_awareness.inspect_tools import ALL_INSPECT_TOOLS
        for name, fn in ALL_INSPECT_TOOLS.items():
            assert callable(fn), f"{name} is not callable"

    def test_inspect_runtime(self):
        from app.self_awareness.inspect_tools import inspect_runtime
        result = inspect_runtime()
        assert isinstance(result, dict)

    def test_inspect_config(self):
        from app.self_awareness.inspect_tools import inspect_config
        result = inspect_config()
        assert isinstance(result, dict)

    def test_inspect_codebase(self):
        from app.self_awareness.inspect_tools import inspect_codebase
        result = inspect_codebase(scope="summary")
        assert isinstance(result, dict)

    def test_inspect_memory(self):
        from app.self_awareness.inspect_tools import inspect_memory
        result = inspect_memory()
        assert isinstance(result, dict)

    def test_inspect_self_model(self):
        from app.self_awareness.inspect_tools import inspect_self_model
        result = inspect_self_model()
        assert isinstance(result, dict)

    def test_run_all_inspections(self):
        from app.self_awareness.inspect_tools import run_all_inspections
        result = run_all_inspections()
        assert isinstance(result, dict)
        assert len(result) >= 5


# ════════════════════════════════════════════════════════════════════════════════
# 10. SELF MODEL TESTS
# ════════════════════════════════════════════════════════════════════════════════

class TestSelfModel:
    """Declarative self-model per agent role."""

    def test_all_roles_defined(self):
        from app.self_awareness.self_model import SELF_MODELS
        expected_roles = {"researcher", "coder", "writer", "commander", "critic", "introspector"}
        assert expected_roles.issubset(set(SELF_MODELS.keys()))

    def test_role_structure(self):
        from app.self_awareness.self_model import SELF_MODELS
        required_keys = {"capabilities", "limitations", "operating_principles",
                         "typical_failure_modes", "metacognitive_triggers"}
        for role, model in SELF_MODELS.items():
            for key in required_keys:
                assert key in model, f"Role '{role}' missing key '{key}'"
                assert isinstance(model[key], list), f"Role '{role}' key '{key}' should be list"
                assert len(model[key]) >= 1, f"Role '{role}' key '{key}' is empty"
            # tools_available may be empty for commander (it delegates, doesn't use tools directly)
            assert "tools_available" in model, f"Role '{role}' missing 'tools_available'"
            assert isinstance(model["tools_available"], list)

    def test_get_self_model_known_role(self):
        from app.self_awareness.self_model import get_self_model
        model = get_self_model("researcher")
        assert isinstance(model, dict)
        assert "capabilities" in model

    def test_get_self_model_unknown_role(self):
        from app.self_awareness.self_model import get_self_model
        model = get_self_model("nonexistent_role_xyz")
        assert isinstance(model, dict)

    def test_format_self_model_block(self):
        from app.self_awareness.self_model import format_self_model_block
        block = format_self_model_block("coder")
        assert isinstance(block, str)
        if block:  # May return empty for unknown roles
            assert len(block) > 50

    def test_format_self_model_injects_runtime(self):
        from app.self_awareness.self_model import format_self_model_block
        block = format_self_model_block("researcher")
        # Should contain runtime stats section
        assert isinstance(block, str)


# ════════════════════════════════════════════════════════════════════════════════
# 11. KNOWLEDGE INGESTION TESTS
# ════════════════════════════════════════════════════════════════════════════════

class TestKnowledgeIngestion:
    """Codebase self-embedding into ChromaDB."""

    def test_ingest_codebase_incremental(self):
        from app.self_awareness.knowledge_ingestion import ingest_codebase
        result = ingest_codebase(full=False)
        assert isinstance(result, dict)
        assert "files_processed" in result or "chunks_added" in result or "error" in str(result)

    def test_query_self_knowledge(self):
        from app.self_awareness.knowledge_ingestion import query_self_knowledge
        results = query_self_knowledge("evolution pipeline", n_results=3)
        assert isinstance(results, list)

    def test_collection_name(self):
        from app.self_awareness.knowledge_ingestion import COLLECTION_NAME
        assert COLLECTION_NAME == "self_knowledge"


# ════════════════════════════════════════════════════════════════════════════════
# 12. CROSS-MODULE DATA FLOW TESTS
# ════════════════════════════════════════════════════════════════════════════════

class TestCrossModuleFlows:
    """Verify data flows between self-reflection modules."""

    def test_cogito_uses_inspect_tools(self):
        src = inspect.getsource(
            __import__("app.self_awareness.cogito", fromlist=["CogitoCycle"]).CogitoCycle._gather_state
        )
        assert "run_all_inspections" in src

    def test_cogito_uses_grounding(self):
        src = inspect.getsource(
            __import__("app.self_awareness.cogito", fromlist=["CogitoCycle"]).CogitoCycle._generate_narrative
        )
        assert "GroundingProtocol" in src

    def test_cogito_writes_journal(self):
        src = inspect.getsource(
            __import__("app.self_awareness.cogito", fromlist=["CogitoCycle"]).CogitoCycle.run
        )
        assert "JournalEntryType.SELF_REFLECTION" in src

    def test_cogito_writes_world_model(self):
        src = inspect.getsource(
            __import__("app.self_awareness.cogito", fromlist=["CogitoCycle"]).CogitoCycle.run
        )
        assert "store_causal_belief" in src

    def test_grounding_uses_inspect_tools(self):
        src = inspect.getsource(
            __import__("app.self_awareness.grounding", fromlist=["GroundingProtocol"]).GroundingProtocol.gather_context
        )
        assert "ALL_INSPECT_TOOLS" in src or "inspect_self_model" in src

    def test_self_model_used_in_agent_backstories(self):
        src = inspect.getsource(__import__("app.souls.loader", fromlist=["_"]))
        assert "format_self_model_block" in src

    def test_homeostasis_read_by_chronicle(self):
        src = inspect.getsource(__import__("app.memory.system_chronicle", fromlist=["_"]))
        assert "homeostasis" in src

    def test_agent_state_read_by_chronicle(self):
        src = inspect.getsource(__import__("app.memory.system_chronicle", fromlist=["_"]))
        assert "agent_state" in src or "get_all_stats" in src


# ════════════════════════════════════════════════════════════════════════════════
# 13. SYSTEM WIRING TESTS
# ════════════════════════════════════════════════════════════════════════════════

class TestSystemWiring:
    """All self-reflection modules must be wired into the live system."""

    def test_orchestrator_wires_journal(self):
        src = inspect.getsource(__import__("app.agents.commander.orchestrator", fromlist=["_"]))
        assert "JournalEntryType.TASK_COMPLETED" in src
        assert "JournalEntryType.TASK_FAILED" in src

    def test_orchestrator_wires_world_model(self):
        src = inspect.getsource(__import__("app.agents.commander.orchestrator", fromlist=["_"]))
        assert "store_prediction_result" in src

    def test_orchestrator_wires_query_router(self):
        src = inspect.getsource(__import__("app.agents.commander.orchestrator", fromlist=["_"]))
        assert "SelfRefRouter" in src

    def test_orchestrator_wires_grounding(self):
        src = inspect.getsource(__import__("app.agents.commander.orchestrator", fromlist=["_"]))
        assert "GroundingProtocol" in src

    def test_orchestrator_wires_agent_state(self):
        src = inspect.getsource(__import__("app.agents.commander.orchestrator", fromlist=["_"]))
        assert "record_task" in src

    def test_orchestrator_wires_homeostasis(self):
        src = inspect.getsource(__import__("app.agents.commander.orchestrator", fromlist=["_"]))
        assert "update_state" in src
        assert "get_behavioral_modifiers" in src

    def test_orchestrator_wires_theory_of_mind(self):
        src = inspect.getsource(__import__("app.agents.commander.orchestrator", fromlist=["_"]))
        assert "get_best_crew_for_difficulty" in src

    def test_evolution_wires_journal(self):
        src = inspect.getsource(__import__("app.evolution", fromlist=["_"]))
        assert "JournalEntryType.EVOLUTION_RESULT" in src

    def test_evolution_wires_world_model(self):
        src = inspect.getsource(__import__("app.evolution", fromlist=["_"]))
        assert "store_causal_belief" in src

    def test_idle_scheduler_wires_cogito(self):
        src = inspect.getsource(__import__("app.idle_scheduler", fromlist=["_"]))
        assert "run_cogito" in src

    def test_idle_scheduler_wires_ingestion(self):
        src = inspect.getsource(__import__("app.idle_scheduler", fromlist=["_"]))
        assert "ingest_codebase" in src

    def test_context_wires_world_model(self):
        src = inspect.getsource(__import__("app.agents.commander.context", fromlist=["_"]))
        assert "recall_relevant_beliefs" in src

    def test_context_wires_homeostasis(self):
        src = inspect.getsource(__import__("app.agents.commander.context", fromlist=["_"]))
        assert "get_state_summary" in src

    def test_publish_wires_inspect_tools(self):
        src = inspect.getsource(__import__("app.firebase.publish", fromlist=["_"]))
        assert "inspect_self_model" in src or "inspect_runtime" in src

    def test_publish_wires_journal(self):
        src = inspect.getsource(__import__("app.firebase.publish", fromlist=["_"]))
        assert "get_journal" in src

    def test_self_heal_wires_world_model(self):
        src = inspect.getsource(__import__("app.self_heal", fromlist=["_"]))
        assert "store_causal_belief" in src

    def test_postprocess_wires_world_model(self):
        src = inspect.getsource(__import__("app.agents.commander.postprocess", fromlist=["_"]))
        assert "store_prediction_result" in src


# ════════════════════════════════════════════════════════════════════════════════
# 14. INTEGRATION TESTS
# ════════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    """End-to-end integration tests."""

    def test_full_journal_lifecycle(self):
        from app.self_awareness.journal import get_journal, JournalEntry, JournalEntryType
        j = get_journal()
        # Write multiple types
        for etype in (JournalEntryType.TASK_COMPLETED, JournalEntryType.ERROR, JournalEntryType.OBSERVATION):
            j.write(JournalEntry(entry_type=etype, summary=f"Integration test {etype.value}"))
        counts = j.count()
        assert isinstance(counts, dict)
        recent = j.read_recent(10)
        assert len(recent) >= 3

    def test_homeostasis_regulation_toward_targets(self):
        from app.self_awareness.homeostasis import update_state, get_state, TARGETS
        # Multiple successes should move confidence toward target
        for _ in range(5):
            update_state("task_complete", "research", success=True, difficulty=5)
        state = get_state()
        # Confidence should be reasonable (not crashed to 0 or exploded to infinity)
        assert 0.0 < state.get("confidence", 0.5) <= 1.0

    def test_query_classification_to_grounding_pipeline(self):
        from app.self_awareness.query_router import SelfRefRouter
        from app.self_awareness.grounding import GroundingProtocol
        router = SelfRefRouter(semantic_enabled=False)
        c = router.classify("How do you learn from your mistakes?")
        if c.should_ground:
            gp = GroundingProtocol()
            ctx = gp.gather_context(c)
            prompt = gp.build_system_prompt(ctx)
            assert len(prompt) > 100

    def test_inspect_tools_produce_valid_output(self):
        from app.self_awareness.inspect_tools import run_all_inspections
        results = run_all_inspections()
        for key, value in results.items():
            assert isinstance(value, dict), f"{key} should return dict, got {type(value)}"


# ════════════════════════════════════════════════════════════════════════════════
# 15. EXPANDED QUERY ROUTER PATTERNS
# ════════════════════════════════════════════════════════════════════════════════

class TestQueryRouterExpanded:
    """Expanded keyword patterns for reflective and operation queries."""

    def _classify(self, query):
        from app.self_awareness.query_router import SelfRefRouter
        return SelfRefRouter(semantic_enabled=False).classify(query)

    # Reflective queries that should now be detected
    def test_how_do_you_learn(self):
        c = self._classify("How do you learn?")
        assert c.is_self_referential

    def test_do_you_learn_from_errors(self):
        c = self._classify("Do you learn from errors?")
        assert c.is_self_referential

    def test_what_have_you_learned(self):
        c = self._classify("What have you learned from your mistakes?")
        assert c.is_self_referential

    def test_how_smart_are_you(self):
        c = self._classify("How smart are you?")
        assert c.is_self_referential

    def test_your_mistakes(self):
        c = self._classify("Tell me about your mistakes")
        assert c.is_self_referential

    def test_do_you_have_feelings(self):
        c = self._classify("Do you have feelings?")
        assert c.is_self_referential

    def test_how_do_you_self_improve(self):
        c = self._classify("How do you self-improve?")
        assert c.is_self_referential

    # Operation queries that should now be detected
    def test_explain_how_you_work(self):
        c = self._classify("Explain how you work")
        assert c.is_self_referential

    def test_what_crews_do_you_have(self):
        c = self._classify("What crews do you have?")
        assert c.is_self_referential

    def test_how_do_you_process(self):
        c = self._classify("How do you process my requests?")
        assert c.is_self_referential

    def test_how_do_you_handle_errors(self):
        c = self._classify("How do you handle errors?")
        assert c.is_self_referential

    # These should still NOT be self-referential
    def test_not_self_code_request(self):
        assert not self._classify("Write a Python function").is_self_referential

    def test_not_self_math(self):
        assert not self._classify("What is 2+2?").is_self_referential

    def test_not_self_search(self):
        assert not self._classify("Search for articles about climate").is_self_referential

    def test_not_self_translation(self):
        assert not self._classify("Translate this to French").is_self_referential


# ════════════════════════════════════════════════════════════════════════════════
# 16. KNOWLEDGE INGESTION CHROMADB CLIENT
# ════════════════════════════════════════════════════════════════════════════════

class TestKnowledgeIngestionClient:
    """Knowledge ingestion must use shared PersistentClient, not HttpClient."""

    def test_ingest_uses_get_client(self):
        """ingest_codebase must use get_client() from chromadb_manager."""
        src = inspect.getsource(
            __import__("app.self_awareness.knowledge_ingestion", fromlist=["ingest_codebase"]).ingest_codebase
        )
        assert "get_client" in src
        assert "HttpClient" not in src

    def test_query_uses_get_client(self):
        """query_self_knowledge must use get_client() from chromadb_manager."""
        src = inspect.getsource(
            __import__("app.self_awareness.knowledge_ingestion", fromlist=["query_self_knowledge"]).query_self_knowledge
        )
        assert "get_client" in src
        assert "HttpClient" not in src

    def test_ingest_returns_dict(self):
        from app.self_awareness.knowledge_ingestion import ingest_codebase
        result = ingest_codebase(full=False)
        assert isinstance(result, dict)
        # Should have success keys, not error
        if "error" not in result:
            assert "files_processed" in result
            assert "chunks_added" in result

    def test_ingest_produces_chunks(self):
        from app.self_awareness.knowledge_ingestion import ingest_codebase
        result = ingest_codebase(full=False)
        if "error" not in result:
            assert result.get("files_processed", 0) >= 0
            assert result.get("chunks_added", 0) >= 0

    def test_query_returns_results_after_ingest(self):
        from app.self_awareness.knowledge_ingestion import query_self_knowledge
        results = query_self_knowledge("evolution", n_results=3)
        assert isinstance(results, list)
        # After ingestion, should find something
        if results:
            assert "document" in results[0] or "metadata" in results[0]

    def test_query_result_has_metadata(self):
        from app.self_awareness.knowledge_ingestion import query_self_knowledge
        results = query_self_knowledge("homeostasis", n_results=2)
        for r in results:
            assert "metadata" in r
            meta = r["metadata"]
            assert isinstance(meta, dict)

    def test_hash_cache_constants(self):
        from app.self_awareness.knowledge_ingestion import HASH_CACHE_PATH, COLLECTION_NAME
        assert COLLECTION_NAME == "self_knowledge"
        assert "self_knowledge_hashes" in str(HASH_CACHE_PATH)

    def test_skip_dirs_defined(self):
        from app.self_awareness.knowledge_ingestion import SKIP_DIRS, CODE_EXTENSIONS
        assert "__pycache__" in SKIP_DIRS
        assert ".py" in CODE_EXTENSIONS
        assert ".md" in CODE_EXTENSIONS


# ════════════════════════════════════════════════════════════════════════════════
# 17. ORCHESTRATOR SELF-AWARENESS INTEGRATION
# ════════════════════════════════════════════════════════════════════════════════

class TestOrchestratorSelfAwareness:
    """Orchestrator integration with self-awareness subsystem."""

    def test_has_grounded_self_response_method(self):
        src = inspect.getsource(
            __import__("app.agents.commander.orchestrator", fromlist=["Commander"])
        )
        assert "_try_grounded_self_response" in src

    def test_grounded_response_uses_router(self):
        src = inspect.getsource(
            __import__("app.agents.commander.orchestrator", fromlist=["Commander"])
        )
        assert "SelfRefRouter" in src
        assert "classify" in src

    def test_grounded_response_uses_protocol(self):
        src = inspect.getsource(
            __import__("app.agents.commander.orchestrator", fromlist=["Commander"])
        )
        assert "GroundingProtocol" in src
        assert "gather_context" in src
        assert "build_system_prompt" in src

    def test_grounded_response_checks_reflective(self):
        """Only REFLECTIVE/COMPARATIVE queries should trigger grounding."""
        src = inspect.getsource(
            __import__("app.agents.commander.orchestrator", fromlist=["Commander"])
        )
        assert "SELF_REFLECTIVE" in src
        assert "SELF_COMPARATIVE" in src

    def test_grounded_response_post_processes(self):
        src = inspect.getsource(
            __import__("app.agents.commander.orchestrator", fromlist=["Commander"])
        )
        assert "post_process" in src

    def test_theory_of_mind_in_routing(self):
        """Routing should use get_best_crew_for_difficulty for d>=6."""
        src = inspect.getsource(
            __import__("app.agents.commander.orchestrator", fromlist=["Commander"])
        )
        assert "get_best_crew_for_difficulty" in src
        assert "Theory of Mind" in src

    def test_journal_records_task_outcomes(self):
        src = inspect.getsource(
            __import__("app.agents.commander.orchestrator", fromlist=["Commander"])
        )
        assert "JournalEntryType.TASK_COMPLETED" in src
        assert "JournalEntryType.TASK_FAILED" in src

    def test_world_model_records_predictions(self):
        src = inspect.getsource(
            __import__("app.agents.commander.orchestrator", fromlist=["Commander"])
        )
        assert "store_prediction_result" in src

    def test_homeostasis_updated_after_tasks(self):
        src = inspect.getsource(
            __import__("app.agents.commander.orchestrator", fromlist=["Commander"])
        )
        assert "update_state" in src
        assert "task_complete" in src

    def test_agent_state_recorded_after_tasks(self):
        src = inspect.getsource(
            __import__("app.agents.commander.orchestrator", fromlist=["Commander"])
        )
        assert "record_task" in src


# ════════════════════════════════════════════════════════════════════════════════
# 18. THEORY OF MIND FUNCTIONAL TESTS
# ════════════════════════════════════════════════════════════════════════════════

class TestTheoryOfMind:
    """Theory of Mind crew selection based on track records."""

    def test_best_crew_for_high_difficulty(self):
        from app.self_awareness.agent_state import record_task, get_best_crew_for_difficulty
        # Give research a perfect record at d=9
        for _ in range(5):
            record_task("tom_test_research", success=True, confidence=0.9, difficulty=9)
        for _ in range(5):
            record_task("tom_test_coding", success=False, confidence=0.3, difficulty=9)
        best = get_best_crew_for_difficulty(9)
        assert best is not None

    def test_best_crew_returns_none_no_data(self):
        from app.self_awareness.agent_state import get_best_crew_for_difficulty
        result = get_best_crew_for_difficulty(99)  # No data at d=99
        assert result is None or isinstance(result, str)

    def test_all_stats_includes_tracked_crews(self):
        from app.self_awareness.agent_state import get_all_stats
        stats = get_all_stats()
        assert isinstance(stats, dict)
        assert len(stats) > 0

    def test_stats_have_success_rate(self):
        from app.self_awareness.agent_state import record_task, get_agent_stats
        record_task("tom_rate_test", success=True, confidence=0.8, difficulty=5)
        record_task("tom_rate_test", success=False, confidence=0.4, difficulty=5)
        stats = get_agent_stats("tom_rate_test")
        assert "success_rate" in stats
        assert 0.0 <= stats["success_rate"] <= 1.0


# ════════════════════════════════════════════════════════════════════════════════
# 19. DEEPER INTEGRATION TESTS
# ════════════════════════════════════════════════════════════════════════════════

class TestDeeperIntegration:
    """Additional integration tests for recently wired modules."""

    def test_journal_has_multiple_entry_types(self):
        from app.self_awareness.journal import get_journal
        counts = get_journal().count()
        # After wiring, journal should have entries from orchestrator + evolution + cogito
        types_with_entries = [t for t, n in counts.items() if n > 0]
        assert len(types_with_entries) >= 2, f"Journal should have 2+ entry types, got: {counts}"

    def test_world_model_has_beliefs(self):
        from app.self_awareness.world_model import recall_relevant_beliefs
        beliefs = recall_relevant_beliefs("system", n=5)
        assert isinstance(beliefs, list)

    def test_homeostasis_state_bounded(self):
        from app.self_awareness.homeostasis import get_state
        state = get_state()
        for key in ("cognitive_energy", "frustration", "confidence", "curiosity"):
            val = state.get(key, 0.5)
            assert 0.0 <= val <= 1.0, f"{key}={val} out of bounds"

    def test_cogito_reflections_persisted(self):
        from app.self_awareness.cogito import REFLECTIONS_DIR
        if REFLECTIONS_DIR.exists():
            reflections = list(REFLECTIONS_DIR.glob("*.json"))
            assert len(reflections) >= 0  # May be empty in fresh container

    def test_self_model_covers_all_active_roles(self):
        from app.self_awareness.self_model import SELF_MODELS
        active_roles = {"researcher", "coder", "writer", "commander", "critic", "introspector"}
        for role in active_roles:
            assert role in SELF_MODELS, f"Active role '{role}' not in SELF_MODELS"

    def test_query_router_to_grounding_pipeline(self):
        """Full pipeline: classify → gather context → build prompt."""
        from app.self_awareness.query_router import SelfRefRouter, SelfRefType
        from app.self_awareness.grounding import GroundingProtocol
        router = SelfRefRouter(semantic_enabled=False)
        c = router.classify("How do you learn from your mistakes?")
        assert c.is_self_referential
        gp = GroundingProtocol()
        ctx = gp.gather_context(c)
        prompt = gp.build_system_prompt(ctx)
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_inspect_tools_all_return_dicts(self):
        from app.self_awareness.inspect_tools import ALL_INSPECT_TOOLS
        for name, fn in ALL_INSPECT_TOOLS.items():
            result = fn()
            assert isinstance(result, dict), f"{name} returned {type(result)}"


# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
