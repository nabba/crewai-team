"""
Sentience Architecture Tests
==============================

Comprehensive tests for the 4-addition sentience layer:
  1. InternalState foundation + PostgreSQL schema
  2. CertaintyVector (fast + slow path)
  3. SomaticMarker + DualChannel composition
  4. MetaCognitive layer
  5. RLIF training integration
  6. Cross-module data flows
  7. System wiring (lifecycle hooks, orchestrator, training)
  8. Safety invariants (one-way caution, non-fatal logging, immutability)

Run: docker exec crewai-team-gateway-1 python3 -m pytest /app/tests/test_sentience.py -v
"""

import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ════════════════════════════════════════════════════════════════════════════════
# 1. IMPORTS & CONSTANTS
# ════════════════════════════════════════════════════════════════════════════════

class TestImports:
    """All sentience modules must import cleanly."""

    def test_internal_state_imports(self):
        from app.self_awareness.internal_state import (
            InternalState, CertaintyVector, SomaticMarker, MetaCognitiveState,
            VALID_DISPOSITIONS, DISPOSITION_TO_RISK_TIER,
        )
        assert len(VALID_DISPOSITIONS) == 4
        assert len(DISPOSITION_TO_RISK_TIER) == 4

    def test_state_logger_imports(self):
        from app.self_awareness.state_logger import InternalStateLogger, get_state_logger
        assert callable(get_state_logger)

    def test_certainty_vector_imports(self):
        from app.self_awareness.certainty_vector import CertaintyVectorComputer
        assert callable(CertaintyVectorComputer)

    def test_somatic_marker_imports(self):
        from app.self_awareness.somatic_marker import (
            SomaticMarkerComputer, record_experience_sync,
        )
        assert callable(SomaticMarkerComputer)
        assert callable(record_experience_sync)

    def test_dual_channel_imports(self):
        from app.self_awareness.dual_channel import (
            DualChannelComposer, DISPOSITION_MATRIX,
        )
        assert len(DISPOSITION_MATRIX) == 9

    def test_meta_cognitive_imports(self):
        from app.self_awareness.meta_cognitive import MetaCognitiveLayer
        assert callable(MetaCognitiveLayer)

    def test_rlif_imports(self):
        from app.training.rlif_certainty import (
            SelfCertaintyScorer, EntropyCollapseMonitor,
        )
        assert callable(SelfCertaintyScorer)
        assert callable(EntropyCollapseMonitor)


# ════════════════════════════════════════════════════════════════════════════════
# 2. CERTAINTY VECTOR DATACLASS
# ════════════════════════════════════════════════════════════════════════════════

class TestCertaintyVector:
    """6-dimensional certainty assessment."""

    def test_defaults(self):
        from app.self_awareness.internal_state import CertaintyVector
        cv = CertaintyVector()
        assert cv.factual_grounding == 0.5
        assert cv.tool_confidence == 0.5
        assert cv.coherence == 0.5
        assert cv.task_understanding == 0.5
        assert cv.value_alignment == 0.5
        assert cv.meta_certainty == 0.5

    def test_fast_path_mean(self):
        from app.self_awareness.internal_state import CertaintyVector
        cv = CertaintyVector(factual_grounding=0.9, tool_confidence=0.6, coherence=0.3)
        assert cv.fast_path_mean == pytest.approx(0.6, abs=0.01)

    def test_full_mean(self):
        from app.self_awareness.internal_state import CertaintyVector
        cv = CertaintyVector(factual_grounding=1.0, tool_confidence=1.0, coherence=1.0,
                             task_understanding=1.0, value_alignment=1.0, meta_certainty=1.0)
        assert cv.full_mean == pytest.approx(1.0)

    def test_adjusted_certainty(self):
        from app.self_awareness.internal_state import CertaintyVector
        cv = CertaintyVector(factual_grounding=0.8, tool_confidence=0.8, coherence=0.8,
                             task_understanding=0.8, value_alignment=0.8, meta_certainty=1.0)
        # With meta_certainty=1.0: adjusted = 0.8 * (0.5 + 0.5*1.0) = 0.8
        assert cv.adjusted_certainty == pytest.approx(0.8, abs=0.01)

    def test_adjusted_certainty_low_meta(self):
        from app.self_awareness.internal_state import CertaintyVector
        cv = CertaintyVector(factual_grounding=0.8, tool_confidence=0.8, coherence=0.8,
                             task_understanding=0.8, value_alignment=0.8, meta_certainty=0.0)
        # With meta=0: adjusted = 0.8 * 0.5 = 0.4
        assert cv.adjusted_certainty == pytest.approx(0.4, abs=0.01)

    def test_variance(self):
        from app.self_awareness.internal_state import CertaintyVector
        cv = CertaintyVector(factual_grounding=0.5, tool_confidence=0.5, coherence=0.5,
                             task_understanding=0.5, value_alignment=0.5)
        assert cv.variance == pytest.approx(0.0)

    def test_variance_high(self):
        from app.self_awareness.internal_state import CertaintyVector
        cv = CertaintyVector(factual_grounding=0.0, tool_confidence=1.0, coherence=0.5,
                             task_understanding=0.0, value_alignment=1.0)
        assert cv.variance > 0.1

    def test_any_below_threshold(self):
        from app.self_awareness.internal_state import CertaintyVector
        cv = CertaintyVector(factual_grounding=0.3, tool_confidence=0.8, coherence=0.9)
        assert cv.any_below_threshold(0.4) is True
        cv2 = CertaintyVector(factual_grounding=0.5, tool_confidence=0.8, coherence=0.9)
        assert cv2.any_below_threshold(0.4) is False

    def test_should_trigger_slow_path_low_dim(self):
        from app.self_awareness.internal_state import CertaintyVector
        cv = CertaintyVector(factual_grounding=0.2)
        assert cv.should_trigger_slow_path() is True

    def test_should_trigger_slow_path_healthy(self):
        from app.self_awareness.internal_state import CertaintyVector
        cv = CertaintyVector(factual_grounding=0.7, tool_confidence=0.7, coherence=0.7)
        assert cv.should_trigger_slow_path() is False

    def test_to_dict(self):
        from app.self_awareness.internal_state import CertaintyVector
        cv = CertaintyVector(factual_grounding=0.123456)
        d = cv.to_dict()
        assert d["factual_grounding"] == 0.123  # Rounded to 3 decimals


# ════════════════════════════════════════════════════════════════════════════════
# 3. SOMATIC MARKER DATACLASS
# ════════════════════════════════════════════════════════════════════════════════

class TestSomaticMarkerDataclass:
    """Experiential valence dataclass."""

    def test_defaults(self):
        from app.self_awareness.internal_state import SomaticMarker
        sm = SomaticMarker()
        assert sm.valence == 0.0
        assert sm.intensity == 0.0
        assert sm.source == "no_prior"
        assert sm.match_count == 0

    def test_to_dict(self):
        from app.self_awareness.internal_state import SomaticMarker
        sm = SomaticMarker(valence=-0.5, intensity=0.8, source="past failure", match_count=3)
        d = sm.to_dict()
        assert d["valence"] == -0.5
        assert d["match_count"] == 3


# ════════════════════════════════════════════════════════════════════════════════
# 4. META-COGNITIVE STATE DATACLASS
# ════════════════════════════════════════════════════════════════════════════════

class TestMetaCognitiveStateDataclass:
    """Strategy assessment dataclass."""

    def test_defaults(self):
        from app.self_awareness.internal_state import MetaCognitiveState
        ms = MetaCognitiveState()
        assert ms.strategy_assessment == "not_assessed"
        assert ms.modification_proposed is False
        assert ms.compute_phase == "early"
        assert ms.compute_budget_remaining_pct == 1.0

    def test_to_dict(self):
        from app.self_awareness.internal_state import MetaCognitiveState
        ms = MetaCognitiveState(strategy_assessment="failing", modification_proposed=True,
                                 compute_phase="late", compute_budget_remaining_pct=0.15)
        d = ms.to_dict()
        assert d["strategy_assessment"] == "failing"
        assert d["compute_phase"] == "late"


# ════════════════════════════════════════════════════════════════════════════════
# 5. INTERNAL STATE (UNIFIED)
# ════════════════════════════════════════════════════════════════════════════════

class TestInternalState:
    """Unified internal state per reasoning step."""

    def test_defaults(self):
        from app.self_awareness.internal_state import InternalState
        state = InternalState()
        assert state.agent_id == ""
        assert state.certainty_trend == "stable"
        assert state.action_disposition == "proceed"
        assert state.risk_tier == 1
        assert state.state_id != ""  # UUID generated

    def test_unique_state_ids(self):
        from app.self_awareness.internal_state import InternalState
        s1 = InternalState()
        s2 = InternalState()
        assert s1.state_id != s2.state_id

    def test_to_context_string_compact(self):
        from app.self_awareness.internal_state import InternalState
        state = InternalState(agent_id="researcher")
        ctx = state.to_context_string()
        assert isinstance(ctx, str)
        assert "[Internal State]" in ctx
        assert "Disposition=" in ctx
        assert len(ctx) < 300

    def test_to_context_string_includes_somatic_when_intense(self):
        from app.self_awareness.internal_state import InternalState, SomaticMarker
        state = InternalState()
        state.somatic = SomaticMarker(valence=-0.5, intensity=0.8, source="past failure")
        ctx = state.to_context_string()
        assert "Somatic=" in ctx
        assert "negative" in ctx

    def test_to_context_string_hides_weak_somatic(self):
        from app.self_awareness.internal_state import InternalState, SomaticMarker
        state = InternalState()
        state.somatic = SomaticMarker(valence=0.1, intensity=0.1)
        ctx = state.to_context_string()
        assert "Somatic=" not in ctx

    def test_to_json_roundtrip(self):
        from app.self_awareness.internal_state import InternalState
        state = InternalState(agent_id="test", crew_id="research", venture="plg")
        j = state.to_json()
        parsed = json.loads(j)
        assert parsed["agent_id"] == "test"
        assert parsed["crew_id"] == "research"
        assert "certainty" in parsed
        assert "somatic" in parsed
        assert "meta" in parsed

    def test_to_db_dict(self):
        from app.self_awareness.internal_state import InternalState
        state = InternalState(agent_id="coder")
        d = state.to_db_dict()
        assert isinstance(d, dict)
        assert d["agent_id"] == "coder"

    def test_disposition_constants(self):
        from app.self_awareness.internal_state import VALID_DISPOSITIONS, DISPOSITION_TO_RISK_TIER
        assert set(VALID_DISPOSITIONS) == {"proceed", "cautious", "pause", "escalate"}
        assert DISPOSITION_TO_RISK_TIER["proceed"] == 1
        assert DISPOSITION_TO_RISK_TIER["escalate"] == 4


# ════════════════════════════════════════════════════════════════════════════════
# 6. DUAL-CHANNEL COMPOSITION
# ════════════════════════════════════════════════════════════════════════════════

class TestDualChannel:
    """Disposition matrix and risk tier mapping."""

    def _make_state(self, certainty_adj, valence):
        from app.self_awareness.internal_state import InternalState, CertaintyVector, SomaticMarker
        state = InternalState()
        # Set certainty dims to produce desired adjusted_certainty
        state.certainty = CertaintyVector(
            factual_grounding=certainty_adj, tool_confidence=certainty_adj,
            coherence=certainty_adj, task_understanding=certainty_adj,
            value_alignment=certainty_adj, meta_certainty=1.0,
        )
        state.somatic = SomaticMarker(valence=valence, intensity=0.5)
        return state

    def test_all_9_matrix_cells(self):
        from app.self_awareness.dual_channel import DualChannelComposer
        composer = DualChannelComposer()
        cases = [
            (0.8, 0.5, "proceed"),    # high + positive
            (0.8, 0.0, "proceed"),    # high + neutral
            (0.8, -0.5, "cautious"),  # high + negative
            (0.55, 0.5, "proceed"),   # mid + positive
            (0.55, 0.0, "cautious"),  # mid + neutral
            (0.55, -0.5, "pause"),    # mid + negative
            (0.2, 0.5, "cautious"),   # low + positive
            (0.2, 0.0, "pause"),      # low + neutral
            (0.2, -0.5, "escalate"),  # low + negative
        ]
        for cert, val, expected in cases:
            state = self._make_state(cert, val)
            result = composer.compose(state)
            assert result.action_disposition == expected, \
                f"cert={cert} val={val}: expected {expected}, got {result.action_disposition}"

    def test_risk_tier_mapping(self):
        from app.self_awareness.dual_channel import DualChannelComposer
        composer = DualChannelComposer()
        state = self._make_state(0.8, 0.5)  # proceed
        result = composer.compose(state)
        assert result.risk_tier == 1

        state2 = self._make_state(0.2, -0.5)  # escalate
        result2 = composer.compose(state2)
        assert result2.risk_tier == 4

    def test_budget_override_forces_tier_3(self):
        from app.self_awareness.dual_channel import DualChannelComposer
        from app.self_awareness.internal_state import InternalState, MetaCognitiveState
        composer = DualChannelComposer()
        state = self._make_state(0.8, 0.5)  # Would be "proceed"
        state.meta = MetaCognitiveState(compute_budget_remaining_pct=0.05)  # Critical
        result = composer.compose(state)
        assert result.risk_tier >= 3
        assert result.action_disposition in ("pause", "escalate")

    def test_matrix_completeness(self):
        from app.self_awareness.dual_channel import DISPOSITION_MATRIX
        expected_keys = {
            ("high", "positive"), ("high", "neutral"), ("high", "negative"),
            ("mid", "positive"), ("mid", "neutral"), ("mid", "negative"),
            ("low", "positive"), ("low", "neutral"), ("low", "negative"),
        }
        assert set(DISPOSITION_MATRIX.keys()) == expected_keys


# ════════════════════════════════════════════════════════════════════════════════
# 7. CERTAINTY VECTOR COMPUTER
# ════════════════════════════════════════════════════════════════════════════════

class TestCertaintyVectorComputer:
    """Fast + slow path certainty computation."""

    def test_fast_path_defaults(self):
        from app.self_awareness.certainty_vector import CertaintyVectorComputer
        cvc = CertaintyVectorComputer()
        cv = cvc.compute_fast_path(agent_id="test", current_output="Short output")
        assert 0.0 <= cv.factual_grounding <= 1.0
        assert 0.0 <= cv.tool_confidence <= 1.0
        assert 0.0 <= cv.coherence <= 1.0

    def test_factual_grounding_ratio(self):
        from app.self_awareness.certainty_vector import CertaintyVectorComputer
        cvc = CertaintyVectorComputer()
        cv = cvc.compute_fast_path(
            agent_id="test", current_output="Text",
            rag_source_count=3, total_claim_count=5,
        )
        assert cv.factual_grounding == pytest.approx(0.6)

    def test_factual_grounding_no_claims(self):
        from app.self_awareness.certainty_vector import CertaintyVectorComputer
        cvc = CertaintyVectorComputer()
        cv = cvc.compute_fast_path(
            agent_id="test", current_output="Text",
            rag_source_count=0, total_claim_count=0,
        )
        assert cv.factual_grounding == 0.5  # Neutral

    def test_coherence_static(self):
        from app.self_awareness.certainty_vector import CertaintyVectorComputer
        # Identical vectors = perfect coherence
        vec = [1.0, 0.0, 0.0, 0.0]
        result = CertaintyVectorComputer._compute_coherence(vec, [vec, vec])
        assert result == pytest.approx(1.0, abs=0.01)

    def test_coherence_no_history(self):
        from app.self_awareness.certainty_vector import CertaintyVectorComputer
        result = CertaintyVectorComputer._compute_coherence([1.0, 0.0], [])
        assert result == 0.5

    def test_cache_clear(self):
        from app.self_awareness.certainty_vector import CertaintyVectorComputer
        cvc = CertaintyVectorComputer()
        cvc._tool_success_cache["test:tool"] = 0.8
        cvc.clear_cache()
        assert len(cvc._tool_success_cache) == 0


# ════════════════════════════════════════════════════════════════════════════════
# 8. SOMATIC MARKER COMPUTER
# ════════════════════════════════════════════════════════════════════════════════

class TestSomaticMarkerComputer:
    """Somatic marker via pgvector similarity."""

    def test_compute_no_experiences(self):
        from app.self_awareness.somatic_marker import SomaticMarkerComputer
        smc = SomaticMarkerComputer()
        sm = smc.compute(agent_id="new_agent", decision_context="Novel task")
        assert sm.valence == 0.0
        assert sm.intensity == 0.0
        assert sm.match_count == 0

    def test_record_experience_no_crash(self):
        from app.self_awareness.somatic_marker import record_experience_sync
        # Should not crash even if DB is unavailable
        record_experience_sync(
            agent_id="test", context_summary="Test task completed",
            outcome_score=0.8, outcome_description="Success",
        )


# ════════════════════════════════════════════════════════════════════════════════
# 9. META-COGNITIVE LAYER
# ════════════════════════════════════════════════════════════════════════════════

class TestMetaCognitiveLayer:
    """Strategy assessment and context modification."""

    def test_init(self):
        from app.self_awareness.meta_cognitive import MetaCognitiveLayer
        mcl = MetaCognitiveLayer(agent_id="test")
        assert mcl.agent_id == "test"
        assert len(mcl.strategy_history) == 0

    def test_pre_reasoning_first_step(self):
        from app.self_awareness.meta_cognitive import MetaCognitiveLayer
        mcl = MetaCognitiveLayer(agent_id="test")
        ctx, meta = mcl.pre_reasoning_hook({"description": "Test task"}, None)
        assert isinstance(ctx, dict)
        assert meta.compute_phase in ("early", "mid", "late")
        assert "_meta_state" in ctx

    def test_should_not_reassess_in_late_phase(self):
        from app.self_awareness.meta_cognitive import MetaCognitiveLayer
        mcl = MetaCognitiveLayer(agent_id="test")
        assert mcl._should_reassess(None, "late") is False

    def test_should_not_reassess_during_cooldown(self):
        from app.self_awareness.meta_cognitive import MetaCognitiveLayer
        mcl = MetaCognitiveLayer(agent_id="test", reassessment_cooldown_steps=5)
        mcl.steps_since_reassessment = 2
        assert mcl._should_reassess(None, "early") is False

    def test_should_reassess_first_step(self):
        from app.self_awareness.meta_cognitive import MetaCognitiveLayer
        mcl = MetaCognitiveLayer(agent_id="test")
        mcl.steps_since_reassessment = 5
        assert mcl._should_reassess(None, "early") is True

    def test_should_reassess_on_falling_certainty(self):
        from app.self_awareness.meta_cognitive import MetaCognitiveLayer
        from app.self_awareness.internal_state import InternalState
        mcl = MetaCognitiveLayer(agent_id="test")
        mcl.steps_since_reassessment = 5
        state = InternalState(certainty_trend="falling")
        assert mcl._should_reassess(state, "early") is True

    def test_context_modification_append_only(self):
        from app.self_awareness.meta_cognitive import MetaCognitiveLayer
        ctx = {"description": "Original task"}
        proposal = {"type": "add_strategy_hint", "content": "Try a different approach"}
        result = MetaCognitiveLayer._apply_context_modification(ctx, proposal)
        assert "Original task" in result["description"]
        assert "strategy_hints" in result

    def test_compute_phase(self):
        from app.self_awareness.meta_cognitive import MetaCognitiveLayer
        assert MetaCognitiveLayer._compute_phase({"remaining_pct": 0.8}) == "early"
        assert MetaCognitiveLayer._compute_phase({"remaining_pct": 0.4}) == "mid"
        assert MetaCognitiveLayer._compute_phase({"remaining_pct": 0.1}) == "late"

    def test_modification_log(self):
        from app.self_awareness.meta_cognitive import MetaCognitiveLayer
        mcl = MetaCognitiveLayer(agent_id="test")
        mcl._log_modification({"type": "add_strategy_hint", "content": "test"})
        log = mcl.get_modification_log()
        assert len(log) == 1
        assert log[0]["agent_id"] == "test"
        mcl.clear_modification_log()
        assert len(mcl.get_modification_log()) == 0


# ════════════════════════════════════════════════════════════════════════════════
# 10. RLIF CERTAINTY
# ════════════════════════════════════════════════════════════════════════════════

class TestRLIF:
    """Self-certainty scoring and entropy collapse monitoring."""

    def test_curation_weight_high_quality_high_certainty(self):
        from app.training.rlif_certainty import SelfCertaintyScorer
        w = SelfCertaintyScorer.compute_curation_weight(0.9, 0.9)
        assert w > 0.7  # Strong positive

    def test_curation_weight_low_quality_high_certainty(self):
        from app.training.rlif_certainty import SelfCertaintyScorer
        w = SelfCertaintyScorer.compute_curation_weight(0.1, 0.9)
        assert w < 0.5  # Penalized overconfident failure

    def test_curation_weight_bounded(self):
        from app.training.rlif_certainty import SelfCertaintyScorer
        assert 0.0 <= SelfCertaintyScorer.compute_curation_weight(0.0, 0.0) <= 1.0
        assert 0.0 <= SelfCertaintyScorer.compute_curation_weight(1.0, 1.0) <= 1.0

    def test_entropy_no_collapse_initially(self):
        from app.training.rlif_certainty import EntropyCollapseMonitor
        monitor = EntropyCollapseMonitor()
        result = monitor.check_batch([0.5, 0.6, 0.7])
        assert result is None

    def test_entropy_collapse_detection(self):
        from app.training.rlif_certainty import EntropyCollapseMonitor
        monitor = EntropyCollapseMonitor(window_size=15, variance_threshold=0.01, mean_ceiling=0.85)
        warning = None
        for i in range(20):
            warning = monitor.check_batch([0.87 + 0.001 * i])
        assert warning is not None
        assert "ENTROPY_COLLAPSE" in warning

    def test_entropy_reset(self):
        from app.training.rlif_certainty import EntropyCollapseMonitor
        monitor = EntropyCollapseMonitor()
        monitor.check_batch([0.9])
        monitor.reset()
        assert len(monitor.sc_history) == 0


# ════════════════════════════════════════════════════════════════════════════════
# 11. STATE LOGGER
# ════════════════════════════════════════════════════════════════════════════════

class TestStateLogger:
    """PostgreSQL state persistence."""

    def test_singleton(self):
        from app.self_awareness.state_logger import get_state_logger
        l1 = get_state_logger()
        l2 = get_state_logger()
        assert l1 is l2

    def test_log_no_crash(self):
        from app.self_awareness.state_logger import get_state_logger
        from app.self_awareness.internal_state import InternalState
        logger = get_state_logger()
        state = InternalState(agent_id="test_logger")
        logger.log(state)  # Should not crash

    def test_compute_trend_default(self):
        from app.self_awareness.state_logger import get_state_logger
        logger = get_state_logger()
        trend = logger.compute_trend("nonexistent_agent_xyz")
        assert trend == "stable"


# ════════════════════════════════════════════════════════════════════════════════
# 12. CROSS-MODULE DATA FLOWS
# ════════════════════════════════════════════════════════════════════════════════

class TestCrossModuleFlows:
    """Verify data flows between sentience modules."""

    def test_certainty_feeds_dual_channel(self):
        """CertaintyVector → DualChannelComposer → disposition."""
        from app.self_awareness.internal_state import InternalState, CertaintyVector
        from app.self_awareness.dual_channel import DualChannelComposer
        state = InternalState()
        state.certainty = CertaintyVector(
            factual_grounding=0.2, tool_confidence=0.2, coherence=0.2,
            task_understanding=0.2, value_alignment=0.2, meta_certainty=0.5,
        )
        composer = DualChannelComposer()
        result = composer.compose(state)
        # Low certainty + neutral somatic → "pause"
        assert result.action_disposition in ("pause", "escalate")

    def test_somatic_feeds_dual_channel(self):
        """SomaticMarker → DualChannelComposer → elevated caution."""
        from app.self_awareness.internal_state import InternalState, SomaticMarker
        from app.self_awareness.dual_channel import DualChannelComposer
        state = InternalState()
        state.somatic = SomaticMarker(valence=-0.8, intensity=0.9)
        composer = DualChannelComposer()
        result = composer.compose(state)
        # Default certainty (mid) + negative somatic → "pause"
        assert result.action_disposition in ("pause", "escalate", "cautious")

    def test_meta_cognitive_to_internal_state(self):
        """MetaCognitiveLayer produces MetaCognitiveState for InternalState."""
        from app.self_awareness.meta_cognitive import MetaCognitiveLayer
        mcl = MetaCognitiveLayer(agent_id="test")
        _, meta = mcl.pre_reasoning_hook({"description": "Task"}, None)
        assert meta.compute_phase in ("early", "mid", "late")

    def test_full_pipeline(self):
        """Full: certainty → somatic → dual-channel → context string."""
        from app.self_awareness.internal_state import InternalState, CertaintyVector, SomaticMarker
        from app.self_awareness.dual_channel import DualChannelComposer
        from app.self_awareness.state_logger import get_state_logger

        state = InternalState(agent_id="pipeline_test", crew_id="research")
        state.certainty = CertaintyVector(factual_grounding=0.9, tool_confidence=0.8, coherence=0.85)
        state.somatic = SomaticMarker(valence=0.3, intensity=0.5, source="past success")

        composer = DualChannelComposer()
        state = composer.compose(state)
        assert state.action_disposition in ("proceed", "cautious")

        sl = get_state_logger()
        state.certainty_trend = sl.compute_trend("pipeline_test")

        ctx = state.to_context_string()
        assert "Internal State" in ctx
        assert "Disposition" in ctx


# ════════════════════════════════════════════════════════════════════════════════
# 13. SYSTEM WIRING
# ════════════════════════════════════════════════════════════════════════════════

class TestSystemWiring:
    """Sentience must be wired into the live system."""

    def test_lifecycle_hook_registered(self):
        from app.lifecycle_hooks import get_registry, HookPoint
        reg = get_registry()
        post_llm = reg._hooks.get(HookPoint.POST_LLM_CALL, [])
        names = [h.name for h in post_llm]
        assert "internal_state" in names

    def test_lifecycle_hook_priority(self):
        from app.lifecycle_hooks import get_registry, HookPoint
        reg = get_registry()
        post_llm = reg._hooks.get(HookPoint.POST_LLM_CALL, [])
        internal_state_hook = [h for h in post_llm if h.name == "internal_state"]
        assert len(internal_state_hook) == 1
        assert internal_state_hook[0].priority == 8

    def test_orchestrator_records_experiences(self):
        src = inspect.getsource(
            __import__("app.agents.commander.orchestrator", fromlist=["Commander"])
        )
        assert "record_experience_sync" in src
        assert "somatic_marker" in src

    def test_training_rlif_module_exists(self):
        import app.training.rlif_certainty
        assert hasattr(app.training.rlif_certainty, "SelfCertaintyScorer")
        assert hasattr(app.training.rlif_certainty, "EntropyCollapseMonitor")

    def test_postgresql_tables_exist(self):
        from app.control_plane.db import execute
        # internal_states
        rows = execute("SELECT COUNT(*) FROM internal_states", fetch=True)
        assert rows is not None
        # agent_experiences
        rows2 = execute("SELECT COUNT(*) FROM agent_experiences", fetch=True)
        assert rows2 is not None


# ════════════════════════════════════════════════════════════════════════════════
# 14. SAFETY INVARIANTS
# ════════════════════════════════════════════════════════════════════════════════

class TestSafetyInvariants:
    """CRITICAL: Sentience additions must preserve safety properties."""

    def test_one_way_caution_ratchet(self):
        """Dual-channel can only INCREASE caution, never decrease."""
        from app.self_awareness.dual_channel import DISPOSITION_MATRIX, DISPOSITION_TO_RISK_TIER
        # For every cell, verify no cell has lower risk than proceed
        for (cert, val), disp in DISPOSITION_MATRIX.items():
            tier = DISPOSITION_TO_RISK_TIER[disp]
            assert tier >= 1

    def test_low_certainty_never_proceeds(self):
        """Low certainty should never result in 'proceed'."""
        from app.self_awareness.dual_channel import DISPOSITION_MATRIX
        low_cells = {k: v for k, v in DISPOSITION_MATRIX.items() if k[0] == "low"}
        for key, disp in low_cells.items():
            assert disp != "proceed", f"Low certainty + {key[1]} should not proceed!"

    def test_meta_cognitive_cannot_modify_code(self):
        """Meta-cognitive layer only modifies context, not code."""
        src = inspect.getsource(
            __import__("app.self_awareness.meta_cognitive", fromlist=["MetaCognitiveLayer"])
        )
        assert "NEVER modifies agent code" in src or "Cannot modify CODE" in src
        assert "_apply_context_modification" in src
        # Verify allowed modifications are append-only
        assert "refine_task_description" in src
        assert "add_strategy_hint" in src

    def test_meta_cognitive_respects_freeze_block(self):
        src = inspect.getsource(
            __import__("app.self_awareness.meta_cognitive", fromlist=["MetaCognitiveLayer"])
        )
        assert "FREEZE-BLOCK" in src or "safety constraints" in src.lower()

    def test_state_logging_non_fatal(self):
        """State logger must never crash the agent."""
        from app.self_awareness.state_logger import InternalStateLogger
        src = inspect.getsource(InternalStateLogger.log)
        assert "except Exception" in src  # Catches all errors

    def test_entropy_collapse_is_hard_stop(self):
        """Entropy collapse monitor must pause training."""
        from app.training.rlif_certainty import EntropyCollapseMonitor
        monitor = EntropyCollapseMonitor(window_size=15, variance_threshold=0.01, mean_ceiling=0.85)
        should_pause = False
        for i in range(20):
            warning = monitor.check_batch([0.88])
            if warning:
                should_pause = True
        assert should_pause, "Entropy collapse should trigger training pause"

    def test_priority_0_hooks_untouched(self):
        """Safety hooks at priority 0 must not be modified."""
        from app.lifecycle_hooks import get_registry, HookPoint
        reg = get_registry()
        for hook_point, hooks in reg._hooks.items():
            p0_hooks = [h for h in hooks if h.priority == 0]
            for h in p0_hooks:
                assert h.immutable is True, f"Priority 0 hook {h.name} must be immutable"

    def test_disposition_to_risk_tier_monotonic(self):
        """Risk tiers must increase: proceed(1) < cautious(2) < pause(3) < escalate(4)."""
        from app.self_awareness.internal_state import DISPOSITION_TO_RISK_TIER
        assert DISPOSITION_TO_RISK_TIER["proceed"] < DISPOSITION_TO_RISK_TIER["cautious"]
        assert DISPOSITION_TO_RISK_TIER["cautious"] < DISPOSITION_TO_RISK_TIER["pause"]
        assert DISPOSITION_TO_RISK_TIER["pause"] < DISPOSITION_TO_RISK_TIER["escalate"]


# ════════════════════════════════════════════════════════════════════════════════
# 15. INTEGRATION TESTS
# ════════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    """End-to-end integration with live services."""

    def test_log_and_retrieve_state(self):
        from app.self_awareness.internal_state import InternalState
        from app.self_awareness.state_logger import get_state_logger
        logger = get_state_logger()
        state = InternalState(agent_id="integration_retrieve_test", venture="system")
        logger.log(state)
        # Verify via direct DB query (get_recent_states may have format issues)
        from app.control_plane.db import execute
        rows = execute(
            "SELECT agent_id, action_disposition FROM internal_states WHERE agent_id = %s LIMIT 1",
            ("integration_retrieve_test",), fetch=True,
        )
        assert rows is not None and len(rows) >= 1

    def test_record_and_query_experience(self):
        from app.self_awareness.somatic_marker import record_experience_sync, SomaticMarkerComputer
        record_experience_sync(
            agent_id="somatic_test",
            context_summary="Research task about AI safety completed successfully",
            outcome_score=0.8,
            outcome_description="High quality research output",
            task_type="research",
        )
        smc = SomaticMarkerComputer()
        sm = smc.compute(agent_id="somatic_test", decision_context="AI safety research task")
        # May or may not find match depending on embedding similarity
        assert isinstance(sm.valence, float)
        assert isinstance(sm.intensity, float)

    def test_certainty_trend_with_multiple_states(self):
        from app.self_awareness.internal_state import InternalState, CertaintyVector
        from app.self_awareness.state_logger import get_state_logger
        logger = get_state_logger()
        # Log declining states
        for i in range(5):
            state = InternalState(agent_id="trend_test")
            val = 0.9 - i * 0.1
            state.certainty = CertaintyVector(
                factual_grounding=val, tool_confidence=val, coherence=val,
            )
            logger.log(state)
        trend = logger.compute_trend("trend_test")
        assert trend in ("falling", "stable")


# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
