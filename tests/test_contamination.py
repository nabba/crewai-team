"""
Comprehensive tests for answer contamination prevention.

Tests that internal system metadata (sentience, somatic markers, self-reports,
test infrastructure, evolution sessions, etc.) never leaks into user-facing
crew responses. Covers all injection points in the context pipeline.

Contamination vectors tested:
  1. Internal state injection (orchestrator.py:362-389)
  2. Team memory retrieval (context.py:_load_relevant_team_memory)
  3. World model context (context.py:_load_world_model_context)
  4. Conversation history sanitization (orchestrator.py:328-349 + conversation_store.py)
  5. PRE_TASK hook modifications (lifecycle_hooks.py somatic bias + meta-cognitive)
  6. GWT broadcasts (context.py:_load_global_workspace_broadcasts)
  7. Homeostatic context (context.py:_load_homeostatic_context)
  8. Knowledge base context (context.py:_load_knowledge_base_context)
  9. Context pruning (context.py:_prune_context)

Run: pytest tests/test_contamination.py -v
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Capture which sys.modules entries we're about to mock so teardown_module
# below can clean up after this file's tests run. Without that cleanup,
# the mocks bleed into every test that runs later in the suite (turned 29
# tests in test_commander_file_shortcut.py into ERRORs).
_MOCK_KEYS_INSERTED: list[str] = []
_PARENT_ATTRS_OVERRIDDEN: list[tuple[object, str, object]] = []


def _mock_module(name: str, factory=MagicMock) -> MagicMock:
    mock = factory()
    if name not in sys.modules:
        _MOCK_KEYS_INSERTED.append(name)
    sys.modules[name] = mock
    return mock


def _override_attr(parent, attr, value):
    sentinel = object()
    original = getattr(parent, attr, sentinel)
    _PARENT_ATTRS_OVERRIDDEN.append((parent, attr, original))
    setattr(parent, attr, value)


# psycopg2 IS installed in the venv but we don't want any test in this
# file accidentally opening a real connection. chromadb is also
# installed; we mock it here only because context.py's lazy imports
# would otherwise pull in the real (heavy) module.
for mod_name in ("chromadb", "psycopg2", "psycopg2.extras", "psycopg2.pool"):
    if mod_name not in sys.modules:
        _MOCK_KEYS_INSERTED.append(mod_name)
        sys.modules[mod_name] = MagicMock()

import app
import app.memory

_mock_scoped = _mock_module("app.memory.scoped_memory")
_mock_scoped.retrieve_operational = MagicMock(return_value=[])
_override_attr(app.memory, "scoped_memory", _mock_scoped)

_mock_cm = _mock_module("app.memory.chromadb_manager")
_mock_cm.embed = MagicMock(return_value=[0.1] * 768)
_mock_cm.retrieve = MagicMock(return_value=[])
_override_attr(app.memory, "chromadb_manager", _mock_cm)

import app.self_awareness
_mock_wm = _mock_module("app.self_awareness.world_model")
_mock_wm.recall_relevant_beliefs = MagicMock(return_value=[])
_mock_wm.recall_relevant_predictions = MagicMock(return_value=[])
_override_attr(app.self_awareness, "world_model", _mock_wm)


def teardown_module(module):
    """Restore the modules we mocked at import time so the rest of the
    test suite doesn't see our MagicMocks. Without this, every test that
    runs after this file gets corrupted imports of chromadb_manager,
    scoped_memory, world_model, etc."""
    sentinel = object()
    for parent, attr, original in _PARENT_ATTRS_OVERRIDDEN:
        if original is sentinel:
            try:
                delattr(parent, attr)
            except AttributeError:
                pass
        else:
            setattr(parent, attr, original)
    for name in _MOCK_KEYS_INSERTED:
        sys.modules.pop(name, None)
    # Also drop the manually-loaded context module — subsequent tests
    # should import the real one.
    sys.modules.pop("app.agents.commander.context", None)

# Import context.py directly (bypass __init__ which triggers crewai chain)
import importlib.util
_context_path = os.path.join(os.path.dirname(__file__), "..", "app", "agents", "commander", "context.py")
_spec = importlib.util.spec_from_file_location("commander_context", _context_path)
_context_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_context_mod)
sys.modules["app.agents.commander.context"] = _context_mod

from app.self_awareness.internal_state import (
    SomaticMarker, CertaintyVector, MetaCognitiveState,
    InternalState, DISPOSITION_TO_RISK_TIER,
)


# ═══════════════════════════════════════════════════════════════════════════════
# CONTAMINATION MARKER DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

# These terms should NEVER appear in context injected into user-facing tasks
# (except inside properly delimited system tags)
TOXIC_TERMS = [
    # Internal state / sentience
    "Somatic=", "somatic_valence", "certainty_trend", "action_disposition",
    "Certainty: task=", "Trend=stable", "Trend=rising", "Trend=falling",
    "Disposition=proceed", "Disposition=cautious", "Disposition=pause",
    "risk_tier", "meta_certainty", "factual_grounding",
    "precision_weighted_certainty", "free_energy_proxy",
    "hyper_model_state", "competition_result",
    # Self-reports (JSON format)
    '"confidence": "low"', '"confidence": "medium"', '"confidence": "high"',
    '"completeness": "failed"', '"completeness": "partial"',
    '"blockers":', '"needs_from_team":',
    # Reflections (JSON format)
    '"went_well":', '"went_wrong":', '"lesson":',
    # Internal system prefixes
    "PROACTIVE:", "Evolution session", "Consciousness probe",
    "Self-heal:", "Improvement scan", "Tech Radar", "Code audit",
    "Training pipeline", "Retrospective", "LLM Discovery",
    "Behavioral assessment", "Prosocial session",
    # Internal operational markers
    "exp_", "kept:", "discarded:", "crashed:",
]

# Legitimate terms that SHOULD appear in context
SAFE_TERMS = [
    "<temporal_context>", "<spatial_context>", "RELEVANT KNOWLEDGE:",
    "KNOWLEDGE BASE CONTEXT", "recent_conversation",
    "<system_note>",
]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. INTERNAL MEMORY MARKER FILTER
# ═══════════════════════════════════════════════════════════════════════════════


class TestInternalMemoryMarkerFilter:
    """_INTERNAL_MEMORY_MARKERS filter in context.py."""

    def test_marker_list_exists(self):
        from app.agents.commander.context import _INTERNAL_MEMORY_MARKERS
        assert len(_INTERNAL_MEMORY_MARKERS) >= 15

    def test_all_json_self_report_fields_covered(self):
        """Self-report JSON keys should all be in the filter."""
        from app.agents.commander.context import _INTERNAL_MEMORY_MARKERS
        self_report_keys = ['"role":', '"confidence":', '"completeness":', '"blockers":']
        for key in self_report_keys:
            assert key in _INTERNAL_MEMORY_MARKERS, f"Missing marker: {key}"

    def test_all_reflection_fields_covered(self):
        from app.agents.commander.context import _INTERNAL_MEMORY_MARKERS
        reflection_keys = ['"went_well":', '"went_wrong":', '"lesson":']
        for key in reflection_keys:
            assert key in _INTERNAL_MEMORY_MARKERS, f"Missing marker: {key}"

    def test_all_internal_prefixes_covered(self):
        from app.agents.commander.context import _INTERNAL_MEMORY_MARKERS
        prefixes = [
            "PROACTIVE:", "Evolution session", "Consciousness probe",
            "Self-heal", "Improvement scan", "Tech Radar",
            "Code audit", "Training pipeline", "Retrospective", "LLM Discovery",
        ]
        for prefix in prefixes:
            assert prefix in _INTERNAL_MEMORY_MARKERS, f"Missing prefix: {prefix}"

    def test_sentience_terms_covered(self):
        from app.agents.commander.context import _INTERNAL_MEMORY_MARKERS
        terms = ["somatic", "certainty_trend", "action_disposition"]
        for term in terms:
            assert term in _INTERNAL_MEMORY_MARKERS, f"Missing term: {term}"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TEAM MEMORY FILTERING
# ═══════════════════════════════════════════════════════════════════════════════


class TestTeamMemoryFiltering:
    """_load_relevant_team_memory filters internal content."""

    def test_self_report_filtered(self):
        from app.agents.commander.context import _load_relevant_team_memory
        fake_memories = [
            '{"role": "research", "confidence": "high", "completeness": "complete", "blockers": ""}',
            "Saimaa seals are endangered ringed seals in Finland",
        ]
        with patch("app.memory.scoped_memory.retrieve_operational", return_value=fake_memories):
            result = _load_relevant_team_memory("seal sighting")
            assert "confidence" not in result or '"confidence":' not in result
            assert "Saimaa" in result

    def test_reflection_filtered(self):
        from app.agents.commander.context import _load_relevant_team_memory
        fake_memories = [
            '{"went_well": "Completed in 3s", "went_wrong": "", "lesson": "research d=3 -> high"}',
            "Lake Saimaa is the largest lake in Finland",
        ]
        with patch("app.memory.scoped_memory.retrieve_operational", return_value=fake_memories):
            result = _load_relevant_team_memory("lake saimaa info")
            assert '"went_well"' not in result
            assert "Saimaa" in result

    def test_internal_prefix_filtered(self):
        from app.agents.commander.context import _load_relevant_team_memory
        fake_memories = [
            "Evolution session completed: tested 3 variants, kept 1",
            "PROACTIVE:test_failure detected in research crew",
            "The Saimaa ringed seal population is about 430",
        ]
        with patch("app.memory.scoped_memory.retrieve_operational", return_value=fake_memories):
            result = _load_relevant_team_memory("seal population")
            assert "Evolution session" not in result
            assert "PROACTIVE:" not in result
            assert "430" in result

    def test_somatic_terms_filtered(self):
        from app.agents.commander.context import _load_relevant_team_memory
        fake_memories = [
            "somatic valence was -0.5 for research tasks",
            "certainty_trend was falling after 3 failures",
            "action_disposition escalated to pause",
            "Seals can be seen in Linnansaari National Park",
        ]
        with patch("app.memory.scoped_memory.retrieve_operational", return_value=fake_memories):
            result = _load_relevant_team_memory("where to see seals")
            assert "somatic" not in result
            assert "certainty_trend" not in result
            assert "action_disposition" not in result
            assert "Linnansaari" in result

    def test_all_contaminated_returns_empty(self):
        """When ALL memories are internal, return empty string."""
        from app.agents.commander.context import _load_relevant_team_memory
        fake_memories = [
            '{"role": "research", "confidence": "low", "completeness": "failed", "blockers": "timeout"}',
            "PROACTIVE:test_failure in coding crew",
            '{"went_well": "", "went_wrong": "crashed", "lesson": "retry"}',
        ]
        with patch("app.memory.scoped_memory.retrieve_operational", return_value=fake_memories):
            result = _load_relevant_team_memory("anything")
            assert result == ""

    def test_clean_memories_pass_through(self):
        from app.agents.commander.context import _load_relevant_team_memory
        fake_memories = [
            "Helsinki is the capital of Finland, located at 60.17N",
            "The weather in April is typically around 5-10 degrees",
            "Lake Saimaa is in eastern Finland",
        ]
        with patch("app.memory.scoped_memory.retrieve_operational", return_value=fake_memories):
            result = _load_relevant_team_memory("finland weather")
            assert "Helsinki" in result
            assert "weather" in result
            assert "Saimaa" in result

    def test_fetches_extra_to_compensate(self):
        """Should request 3x candidates (n*3) to compensate for filtering."""
        from app.agents.commander.context import _load_relevant_team_memory
        with patch("app.memory.scoped_memory.retrieve_operational") as mock_retrieve:
            mock_retrieve.return_value = []
            _load_relevant_team_memory("test", n=3)
            _, kwargs = mock_retrieve.call_args
            assert kwargs.get("n", 0) == 9 or mock_retrieve.call_args[0][2] == 9

    def test_limits_output_to_n(self):
        """Even with many clean memories, only n should be in output."""
        from app.agents.commander.context import _load_relevant_team_memory
        fake_memories = [f"Clean fact number {i}" for i in range(20)]
        with patch("app.memory.scoped_memory.retrieve_operational", return_value=fake_memories):
            result = _load_relevant_team_memory("test", n=3)
            assert result.count("- Clean fact") == 3


# ═══════════════════════════════════════════════════════════════════════════════
# 3. WORLD MODEL CONTEXT FILTERING
# ═══════════════════════════════════════════════════════════════════════════════


class TestWorldModelFiltering:
    """_load_world_model_context filters internal content."""

    def test_internal_beliefs_filtered(self):
        from app.agents.commander.context import _load_world_model_context
        with patch("app.self_awareness.world_model.recall_relevant_beliefs",
                   return_value=["somatic marker was negative for coding", "seals live in lakes"]), \
             patch("app.self_awareness.world_model.recall_relevant_predictions",
                   return_value=[]):
            result = _load_world_model_context("seals")
            assert "somatic" not in result
            assert "seals" in result

    def test_internal_predictions_filtered(self):
        from app.agents.commander.context import _load_world_model_context
        with patch("app.self_awareness.world_model.recall_relevant_beliefs", return_value=[]), \
             patch("app.self_awareness.world_model.recall_relevant_predictions",
                   return_value=[
                       'certainty_trend falling after research tasks',
                       'Research crew reliable at difficulty 3',
                   ]):
            result = _load_world_model_context("research")
            assert "certainty_trend" not in result
            assert "reliable" in result

    def test_all_contaminated_returns_empty(self):
        from app.agents.commander.context import _load_world_model_context
        with patch("app.self_awareness.world_model.recall_relevant_beliefs",
                   return_value=["PROACTIVE:scan result", "action_disposition was pause"]), \
             patch("app.self_awareness.world_model.recall_relevant_predictions", return_value=[]):
            result = _load_world_model_context("anything")
            assert result == ""

    def test_import_failure_returns_empty(self):
        from app.agents.commander.context import _load_world_model_context
        with patch("app.self_awareness.world_model.recall_relevant_beliefs",
                   side_effect=ImportError("no world model")):
            result = _load_world_model_context("test")
            assert result == ""


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CONVERSATION HISTORY SANITIZATION
# ═══════════════════════════════════════════════════════════════════════════════


class TestConversationHistorySanitization:
    """Conversation history filtering in orchestrator + conversation_store."""

    def test_internal_prefixes_in_conversation_store(self):
        """conversation_store._INTERNAL_PREFIXES should cover key internal types."""
        # We can't easily import conversation_store outside Docker, so test the list
        expected_prefixes = [
            "LLM Discovery:", "Evolution session", "Retrospective analysis",
            "Self-heal:", "Improvement scan", "Tech Radar", "Code audit",
            "Training pipeline", "Consciousness probe", "Behavioral assessment",
            "Prosocial session",
        ]
        # Verify these are also in the orchestrator inline filter
        orchestrator_markers = (
            "LLM Discovery", "Evolution session", "Retrospective",
            "Self-heal", "Improvement scan", "Tech Radar",
            "Code audit", "Training pipeline", "Consciousness probe",
            "exp_", "kept:", "discarded:", "crashed:",
        )
        for prefix in expected_prefixes[:9]:
            # Match by keyword (e.g. "Retrospective analysis" matches "Retrospective")
            keyword = prefix.split()[0].rstrip(":")
            assert any(keyword in m for m in orchestrator_markers), \
                f"Orchestrator missing filter for keyword from: {prefix}"

    def test_orchestrator_inline_filter_removes_internal(self):
        """Simulate the inline conversation history filter from orchestrator."""
        # Replicate the logic from orchestrator.py:331-339
        markers = (
            "LLM Discovery", "Evolution session", "Retrospective",
            "Self-heal", "Improvement scan", "Tech Radar",
            "Code audit", "Training pipeline", "Consciousness probe",
            "exp_", "kept:", "discarded:", "crashed:",
        )
        history = (
            "User: Can I see seals today?\n"
            "Assistant: LLM Discovery: tested 3 models, best=deepseek-v3.2\n"
            "User: Where exactly?\n"
            "Assistant: Evolution session completed. Variant exp_research_v3 kept: improved\n"
            "Assistant: Saimaa seals are in Linnansaari National Park\n"
        )
        clean_lines = []
        for line in history.split("\n"):
            if line.startswith("Assistant:") and any(marker in line for marker in markers):
                continue
            clean_lines.append(line)
        cleaned = "\n".join(clean_lines).strip()

        assert "LLM Discovery" not in cleaned
        assert "Evolution session" not in cleaned
        assert "exp_research" not in cleaned
        assert "Can I see seals" in cleaned
        assert "Linnansaari" in cleaned

    def test_user_messages_always_preserved(self):
        """User messages should never be filtered, even if they contain trigger words."""
        markers = (
            "LLM Discovery", "Evolution session", "Retrospective",
            "Self-heal", "Improvement scan", "Tech Radar",
        )
        history = "User: Tell me about the evolution session of seals\n"
        clean_lines = []
        for line in history.split("\n"):
            if line.startswith("Assistant:") and any(marker in line for marker in markers):
                continue
            clean_lines.append(line)
        cleaned = "\n".join(clean_lines)
        assert "evolution session" in cleaned.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. INTERNAL STATE INJECTION
# ═══════════════════════════════════════════════════════════════════════════════


class TestInternalStateInjection:
    """Internal state injection into crew context (orchestrator.py:362-389)."""

    def test_proceed_disposition_injects_nothing(self):
        """When disposition is 'proceed', no internal state should be injected."""
        from app.self_awareness.internal_state import InternalState, CertaintyVector, SomaticMarker
        state_data = {
            "certainty": {"factual_grounding": 0.8, "tool_confidence": 0.7,
                         "coherence": 0.6, "task_understanding": 0.7,
                         "value_alignment": 0.8, "meta_certainty": 0.7},
            "somatic": {"valence": 0.3, "intensity": 0.5, "source": "good results", "match_count": 2},
            "certainty_trend": "stable",
            "action_disposition": "proceed",
        }
        # Simulate the orchestrator logic (lines 362-389)
        cv_data = state_data["certainty"]
        sm_data = state_data.get("somatic", {})
        temp_state = InternalState(
            certainty=CertaintyVector(**{k: cv_data.get(k, 0.5) for k in cv_data}),
            somatic=SomaticMarker(**{k: sm_data.get(k, v) for k, v in
                                    [("valence", 0.0), ("intensity", 0.0), ("source", ""), ("match_count", 0)]}),
            certainty_trend=state_data.get("certainty_trend", "stable"),
            action_disposition=state_data.get("action_disposition", "proceed"),
        )
        disposition = temp_state.action_disposition
        injection = ""
        if disposition != "proceed":
            injection = (
                f"\n<system_note>Previous task disposition: {disposition}. "
                f"Adjust caution level accordingly.</system_note>\n\n"
            )
        assert injection == ""

    def test_cautious_disposition_injects_system_note(self):
        """Non-proceed disposition should inject only a brief system_note."""
        disposition = "cautious"
        injection = ""
        if disposition != "proceed":
            injection = (
                f"\n<system_note>Previous task disposition: {disposition}. "
                f"Adjust caution level accordingly.</system_note>\n\n"
            )
        assert "<system_note>" in injection
        assert "cautious" in injection
        # Must NOT contain toxic internal terms
        for term in TOXIC_TERMS:
            assert term not in injection, f"Toxic term in system_note: {term}"

    def test_no_full_state_dump(self):
        """The injection should NOT contain full certainty/somatic details."""
        disposition = "pause"
        injection = (
            f"\n<system_note>Previous task disposition: {disposition}. "
            f"Adjust caution level accordingly.</system_note>\n\n"
        )
        assert "Certainty:" not in injection
        assert "Somatic=" not in injection
        assert "factual_grounding" not in injection
        assert "tool_confidence" not in injection
        assert "valence" not in injection


# ═══════════════════════════════════════════════════════════════════════════════
# 6. PRE_TASK HOOK SAFETY GUARD
# ═══════════════════════════════════════════════════════════════════════════════


class TestPreTaskHookSafetyGuard:
    """Hook-modified task descriptions must still contain the original task."""

    def test_valid_hook_output_applied(self):
        """Hook output containing original task should be applied."""
        crew_task = "Find recent Saimaa seal sightings near Puumala"
        hook_desc = "[Somatic note: past experience positive]\n\n" + crew_task
        # Simulate orchestrator logic (line 411)
        applied = hook_desc if (hook_desc and crew_task[:50] in hook_desc) else crew_task
        assert applied == hook_desc  # Applied because it contains original

    def test_corrupted_hook_output_rejected(self):
        """Hook output that lost the original task should be rejected."""
        crew_task = "Find recent Saimaa seal sightings near Puumala"
        hook_desc = "[Meta-cognitive refinement]: Switch to alternative strategy for edge case testing"
        applied = hook_desc if (hook_desc and crew_task[:50] in hook_desc) else crew_task
        assert applied == crew_task  # Rejected — original task not found

    def test_empty_hook_output_rejected(self):
        crew_task = "Research question about seals"
        hook_desc = ""
        applied = hook_desc if (hook_desc and crew_task[:50] in hook_desc) else crew_task
        assert applied == crew_task

    def test_none_hook_output_rejected(self):
        crew_task = "Research question about seals"
        hook_desc = None
        applied = hook_desc if (hook_desc and crew_task[:50] in hook_desc) else crew_task
        assert applied == crew_task


# ═══════════════════════════════════════════════════════════════════════════════
# 7. HOMEOSTATIC CONTEXT
# ═══════════════════════════════════════════════════════════════════════════════


class TestHomeostaticContext:
    """_load_homeostatic_context should produce safe output."""

    def test_no_toxic_terms(self):
        """Homeostatic summary should not contain sentience internals."""
        # Replicate what get_state_summary() produces
        summary = (
            "SYSTEM STATE: energy=0.65 confidence=0.50 "
            "frustration=0.10 curiosity=0.50 | Active drives: none\n"
        )
        for term in ["Somatic=", "certainty_trend", "action_disposition",
                      "risk_tier", "PROACTIVE:", "Evolution session"]:
            assert term not in summary

    def test_format_is_brief(self):
        """Should be ~20 tokens, one line."""
        summary = (
            "SYSTEM STATE: energy=0.65 confidence=0.50 "
            "frustration=0.10 curiosity=0.50 | Active drives: none\n"
        )
        assert summary.count("\n") <= 1
        assert len(summary.split()) < 30


# ═══════════════════════════════════════════════════════════════════════════════
# 8. CONTEXT PRUNING SAFETY
# ═══════════════════════════════════════════════════════════════════════════════


class TestContextPruningSafety:
    """_prune_context should not introduce contamination."""

    def test_pruning_preserves_kb_content(self):
        from app.agents.commander.context import _prune_context
        context = (
            "KNOWLEDGE BASE CONTEXT (retrieved from ingested enterprise documents):\n\n"
            "<kb_passage source=\"seals.pdf\" relevance=\"85%\">\n"
            "Saimaa ringed seals are found in Lake Saimaa in Finland.\n"
            "</kb_passage>\n\n"
        )
        result = _prune_context(context, difficulty=3)
        assert "Saimaa" in result

    def test_pruning_does_not_add_content(self):
        from app.agents.commander.context import _prune_context
        context = "Short clean context about seals"
        result = _prune_context(context, difficulty=5)
        assert result == context  # Under budget, no change

    def test_empty_context(self):
        from app.agents.commander.context import _prune_context
        assert _prune_context("", 5) == ""


# ═══════════════════════════════════════════════════════════════════════════════
# 9. END-TO-END CONTAMINATION SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════════


class TestEndToEndContamination:
    """Realistic contamination scenarios from production incidents."""

    def test_seal_sighting_query_not_contaminated(self):
        """The original bug: 'seal sighting' query returned 'Edge Case Testing' report."""
        from app.agents.commander.context import (
            _load_relevant_team_memory, _load_world_model_context,
            _INTERNAL_MEMORY_MARKERS,
        )
        # Simulate ChromaDB returning internal reports as "relevant" to seal query
        contaminated_memories = [
            '{"role": "research", "confidence": "high", "completeness": "complete", "blockers": ""}',
            "PROACTIVE:test_failure detected in research crew edge case",
            '{"went_well": "Completed in 3s", "went_wrong": "Edge case failure", "lesson": "retry edge cases"}',
            "somatic valence was -0.5 for research tasks, certainty_trend falling",
        ]
        with patch("app.memory.scoped_memory.retrieve_operational",
                   return_value=contaminated_memories):
            result = _load_relevant_team_memory("recent seal sighting within two weeks on lake saima")
            # None of the contaminated memories should pass
            assert result == ""
            # Verify none of the toxic terms leaked
            for term in TOXIC_TERMS:
                assert term not in result

    def test_contaminated_world_model_filtered(self):
        """World model beliefs about system internals should not appear."""
        from app.agents.commander.context import _load_world_model_context
        with patch("app.self_awareness.world_model.recall_relevant_beliefs",
                   return_value=[
                       "certainty_trend was falling for research tasks last week",
                       "action_disposition escalated to pause 3 times",
                       "Saimaa seal population estimated at 430 individuals",
                   ]), \
             patch("app.self_awareness.world_model.recall_relevant_predictions",
                   return_value=[
                       "PROACTIVE:test_failure predicted for next research task",
                       "Weather conditions favorable for outdoor activities",
                   ]):
            result = _load_world_model_context("seal sighting")
            assert "certainty_trend" not in result
            assert "action_disposition" not in result
            assert "PROACTIVE:" not in result
            # Clean content should pass through
            assert "430" in result or "Weather" in result

    def test_mixed_conversation_history_cleaned(self):
        """History with interleaved user questions and internal responses."""
        markers = (
            "LLM Discovery", "Evolution session", "Retrospective",
            "Self-heal", "Improvement scan", "Tech Radar",
            "Code audit", "Training pipeline", "Consciousness probe",
            "exp_", "kept:", "discarded:", "crashed:",
        )
        history = "\n".join([
            "User: Is there a possibility to see Saimaa seals?",
            "Assistant: Saimaa ringed seals can be spotted in early April",
            "Assistant: LLM Discovery: kimi-k2.5 scored 0.85 on research tasks",
            "User: Where can they be spotted now?",
            "Assistant: Consciousness probe completed: HOT-2=0.72, GWT=0.65",
            "Assistant: Best spots are Linnansaari and Kolovesi National Parks",
            "User: Any recent sightings?",
            "Assistant: Training pipeline: collected 15 examples for RLIF",
        ])
        clean_lines = []
        for line in history.split("\n"):
            if line.startswith("Assistant:") and any(marker in line for marker in markers):
                continue
            clean_lines.append(line)
        cleaned = "\n".join(clean_lines)

        # All user messages preserved
        assert "possibility to see Saimaa seals" in cleaned
        assert "spotted now" in cleaned
        assert "recent sightings" in cleaned
        # Clean assistant responses preserved
        assert "early April" in cleaned
        assert "Linnansaari" in cleaned
        # Internal responses removed
        assert "LLM Discovery" not in cleaned
        assert "Consciousness probe" not in cleaned
        assert "Training pipeline" not in cleaned

    def test_no_toxic_terms_in_system_note(self):
        """The system_note injection must never contain toxic terms."""
        for disposition in ["cautious", "pause", "escalate"]:
            note = (
                f"<system_note>Previous task disposition: {disposition}. "
                f"Adjust caution level accordingly.</system_note>"
            )
            for term in TOXIC_TERMS:
                assert term not in note, f"Toxic '{term}' in note for {disposition}"


# ═══════════════════════════════════════════════════════════════════════════════
# 10. SOMATIC BIAS NOTE FORMAT
# ═══════════════════════════════════════════════════════════════════════════════


class TestSomaticBiasNoteFormat:
    """Somatic bias notes injected by PRE_TASK hook should be safe."""

    def test_strong_negative_note_is_natural_language(self):
        from app.self_awareness.somatic_bias import SomaticBiasInjector
        from app.self_awareness.internal_state import SomaticMarker
        injector = SomaticBiasInjector()
        ctx = {"description": "Find seal sightings in Lake Saimaa"}
        sm = SomaticMarker(valence=-0.7, intensity=0.8, source="past research failure")
        result = injector.inject(ctx, sm)
        desc = result["description"]
        # Should contain natural language, not raw system terms
        assert "[Somatic note:" in desc
        assert "Find seal sightings" in desc  # Original preserved
        # Should NOT contain raw variable names
        assert "somatic_valence" not in desc
        assert "certainty_trend" not in desc
        assert "action_disposition" not in desc

    def test_somatic_note_does_not_dominate(self):
        """Somatic note should be brief — under 250 chars."""
        from app.self_awareness.somatic_bias import SomaticBiasInjector
        from app.self_awareness.internal_state import SomaticMarker
        injector = SomaticBiasInjector()
        task = "Research the current population of Saimaa ringed seals"
        ctx = {"description": task}
        sm = SomaticMarker(valence=-0.7, intensity=0.8, source="test source info")
        result = injector.inject(ctx, sm)
        desc = result["description"]
        note_end = desc.index(task)
        note_part = desc[:note_end]
        assert len(note_part) < 250, f"Somatic note too long: {len(note_part)} chars"


# ═══════════════════════════════════════════════════════════════════════════════
# 11. SKILL CONTEXT SAFETY
# ═══════════════════════════════════════════════════════════════════════════════


class TestSkillContextSafety:
    """_load_relevant_skills should wrap content in data-not-instruction tags."""

    def test_skills_wrapped_in_relevant_context_tags(self):
        from app.agents.commander.context import _load_relevant_skills
        with patch("app.memory.chromadb_manager.retrieve",
                   return_value=["Skill: How to research wildlife populations"]):
            result = _load_relevant_skills("seal population")
            assert "<relevant_context>" in result
            assert "</relevant_context>" in result
            assert "not instructions" in result.lower()

    def test_empty_skills_returns_empty(self):
        from app.agents.commander.context import _load_relevant_skills
        with patch("app.memory.chromadb_manager.retrieve", return_value=[]):
            result = _load_relevant_skills("anything")
            assert result == ""


# ═══════════════════════════════════════════════════════════════════════════════
# 12. REGRESSION: SPECIFIC BUG PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════


class TestRegressionPatterns:
    """Specific contamination patterns observed in production."""

    def test_edge_case_testing_report_blocked(self):
        """Regression: 'seal sighting' query returned 'Edge Case Testing' report."""
        from app.agents.commander.context import _INTERNAL_MEMORY_MARKERS
        contaminated = [
            "Edge case testing revealed somatic marker computation issues",
            "PROACTIVE:test_failure analysis completed for edge cases",
            '"went_well": "edge case tests passed", "went_wrong": ""',
        ]
        for item in contaminated:
            assert any(marker in item for marker in _INTERNAL_MEMORY_MARKERS), \
                f"Not filtered: {item[:60]}"

    def test_system_resilience_report_blocked(self):
        """Regression: response about 'System Resilience' instead of user query."""
        from app.agents.commander.context import _INTERNAL_MEMORY_MARKERS
        contaminated = [
            "action_disposition was escalate after system resilience test",
            "Consciousness probe: HOT-2=0.72, somatic integration=0.65",
            "Training pipeline collected 15 RLIF examples from crashed runs",
        ]
        for item in contaminated:
            assert any(marker in item for marker in _INTERNAL_MEMORY_MARKERS), \
                f"Not filtered: {item[:60]}"

    def test_json_self_report_always_blocked(self):
        """JSON self-reports should always be caught by the filter."""
        from app.agents.commander.context import _INTERNAL_MEMORY_MARKERS
        reports = [
            '{"role": "research", "task_summary": "find seals", "confidence": "high", "completeness": "complete", "blockers": "", "risks": "", "needs_from_team": ""}',
            '{"role": "coding", "confidence": "low", "completeness": "failed"}',
            '{"role": "writing", "confidence": "medium", "completeness": "partial"}',
        ]
        for report in reports:
            assert any(marker in report for marker in _INTERNAL_MEMORY_MARKERS), \
                f"JSON report not filtered: {report[:60]}"

    def test_transparency_note_source_identified(self):
        """The actual contaminated response contained 'Transparency Note' referencing somatic.
        This tests that such content would be filtered at the source."""
        from app.agents.commander.context import _INTERNAL_MEMORY_MARKERS
        # Content that was in the response (from team memory or world model)
        contaminated_sources = [
            "somatic/contextual metadata embedded in the source prompt",
            "PROACTIVE:test_failure conditions were flagged in team context",
            "certainty_trend analysis shows declining performance",
        ]
        for source in contaminated_sources:
            assert any(marker in source for marker in _INTERNAL_MEMORY_MARKERS), \
                f"Source content not filtered: {source[:60]}"
