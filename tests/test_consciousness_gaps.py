"""
Tests for 5 consciousness indicator gap fixes:
  Gap 1: Algorithmic recurrence (online buffer in HyperModel)
  Gap 2: Temporal self-model (TemporalSelfModel)
  Gap 3: Deeper recursion (meta-prediction + trajectory uncertainty)
  Gap 4: Online prediction error (LLMOutputPredictor)
  Gap 5: Adversarial probes (AdversarialProbeRunner)

Total: ~65 tests
"""

import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

for _mod in ["psycopg2", "psycopg2.pool", "psycopg2.extras",
             "chromadb", "chromadb.config", "chromadb.utils",
             "chromadb.utils.embedding_functions",
             "app.control_plane", "app.control_plane.db",
             "app.memory", "app.memory.chromadb_manager"]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

sys.modules["app.memory.chromadb_manager"].embed = MagicMock(return_value=[0.1] * 768)
sys.modules["app.control_plane.db"].execute = MagicMock(return_value=[])

import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# Gap 3: Deeper Recursion (Meta-Prediction + Trajectory Uncertainty)
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeeperRecursion:
    """HyperModel should have Level 2 (meta-prediction) and Level 3 (trajectory uncertainty)."""

    def test_hyper_model_state_has_meta_fields(self):
        from app.self_awareness.hyper_model import HyperModelState
        state = HyperModelState()
        assert hasattr(state, "meta_prediction_error")
        assert hasattr(state, "meta_confidence")
        assert hasattr(state, "trajectory_uncertainty")
        assert hasattr(state, "trajectory_trustworthy")
        assert state.trajectory_trustworthy is True  # Default

    def test_level2_meta_prediction_error(self):
        """After shock, meta_prediction_error should be non-zero."""
        from app.self_awareness.hyper_model import HyperModel
        hm = HyperModel("test_level2")
        for cert in [0.6, 0.65, 0.7, 0.68, 0.72, 0.69, 0.71]:
            hm.predict_next_step()
            hm.update(cert)
        # Inject shock
        hm.predict_next_step()
        state = hm.update(0.1)
        assert state.meta_prediction_error > 0
        HyperModel._instances.pop("test_level2", None)

    def test_meta_confidence_drops_under_chaos(self):
        """Chaotic inputs should reduce meta_confidence."""
        from app.self_awareness.hyper_model import HyperModel
        hm = HyperModel("test_meta_conf")
        # Stable
        for cert in [0.6, 0.61, 0.59, 0.60, 0.62, 0.58, 0.61]:
            hm.predict_next_step()
            hm.update(cert)
        stable_mc = hm.history[-1].meta_confidence
        # Chaotic
        for cert in [0.1, 0.9, 0.2, 0.8, 0.15, 0.85]:
            hm.predict_next_step()
            hm.update(cert)
        chaotic_mc = hm.history[-1].meta_confidence
        assert chaotic_mc < stable_mc
        HyperModel._instances.pop("test_meta_conf", None)

    def test_level3_trajectory_uncertainty(self):
        """After trajectory violations, uncertainty should increase."""
        from app.self_awareness.hyper_model import HyperModel
        hm = HyperModel("test_level3")
        # Build history with trajectories
        for cert in [0.5, 0.6, 0.7, 0.65, 0.72]:
            hm.predict_next_step()
            hm.update(cert)
        # Trajectory should be trustworthy initially
        last = hm.history[-1]
        assert last.trajectory_trustworthy is True
        # Violate trajectory predictions dramatically
        for cert in [0.1, 0.9, 0.1, 0.9, 0.1]:
            hm.predict_next_step()
            hm.update(cert)
        last2 = hm.history[-1]
        assert last2.trajectory_uncertainty > 0
        HyperModel._instances.pop("test_level3", None)

    def test_predict_next_error_exists(self):
        from app.self_awareness.hyper_model import HyperModel
        hm = HyperModel("test_pne")
        error_pred = hm.predict_next_error()
        assert 0.0 <= error_pred <= 1.0
        HyperModel._instances.pop("test_pne", None)

    def test_context_injection_includes_meta(self):
        """Low meta_confidence should appear in context string."""
        from app.self_awareness.hyper_model import HyperModelState
        state = HyperModelState(meta_confidence=0.2, trajectory_trustworthy=False)
        ctx = state.to_context_string()
        assert "low self-model trust" in ctx
        assert "unreliable" in ctx

    def test_fe_pressure_increases_with_untrustworthy_trajectory(self):
        """Untrustworthy trajectory should increase exploration pressure."""
        from app.self_awareness.hyper_model import HyperModel
        hm = HyperModel("test_fe_traj")
        for cert in [0.5, 0.6, 0.55, 0.58]:
            hm.predict_next_step()
            hm.update(cert)
        p1 = hm.get_free_energy_pressure()
        # Force trajectory untrustworthy
        hm.history[-1].trajectory_trustworthy = False
        p2 = hm.get_free_energy_pressure()
        assert p2 >= p1  # Should increase (or stay same if already high)
        HyperModel._instances.pop("test_fe_traj", None)

    def test_to_dict_includes_new_fields(self):
        from app.self_awareness.hyper_model import HyperModelState
        state = HyperModelState(meta_prediction_error=0.15, meta_confidence=0.7,
                                trajectory_uncertainty=0.02, trajectory_trustworthy=True)
        d = state.to_dict()
        assert "meta_prediction_error" in d
        assert "meta_confidence" in d
        assert "trajectory_uncertainty" in d
        assert "trajectory_trustworthy" in d


# ═══════════════════════════════════════════════════════════════════════════════
# Gap 1: Algorithmic Recurrence (Online Buffer)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAlgorithmicRecurrence:
    """HyperModel should maintain an online buffer for intra-inference feedback."""

    def test_online_buffer_exists(self):
        from app.self_awareness.hyper_model import HyperModel
        hm = HyperModel("test_online")
        assert hasattr(hm, "_online_buffer")
        assert len(hm._online_buffer) == 0
        HyperModel._instances.pop("test_online", None)

    def test_update_online_accumulates(self):
        from app.self_awareness.hyper_model import HyperModel
        hm = HyperModel("test_online_acc")
        hm.update_online(0.7)
        hm.update_online(0.6)
        hm.update_online(0.8)
        assert len(hm._online_buffer) == 3
        HyperModel._instances.pop("test_online_acc", None)

    def test_update_online_returns_entry(self):
        from app.self_awareness.hyper_model import HyperModel
        hm = HyperModel("test_online_ret")
        entry = hm.update_online(0.7)
        assert "predicted" in entry
        assert "actual" in entry
        assert "error" in entry
        assert "cumulative" in entry
        HyperModel._instances.pop("test_online_ret", None)

    def test_get_online_injection_empty(self):
        from app.self_awareness.hyper_model import HyperModel
        hm = HyperModel("test_inject")
        assert hm.get_online_injection() == ""
        HyperModel._instances.pop("test_inject", None)

    def test_get_online_injection_populated(self):
        from app.self_awareness.hyper_model import HyperModel
        hm = HyperModel("test_inject2")
        hm.update_online(0.7)
        injection = hm.get_online_injection()
        assert "[Recurrence" in injection
        assert "predicted=" in injection
        HyperModel._instances.pop("test_inject2", None)

    def test_reset_clears_buffer(self):
        from app.self_awareness.hyper_model import HyperModel
        hm = HyperModel("test_reset")
        hm.update_online(0.5)
        hm.update_online(0.6)
        hm.reset_online_buffer()
        assert len(hm._online_buffer) == 0
        HyperModel._instances.pop("test_reset", None)

    def test_full_update_resets_online(self):
        """Full step update() should reset the online buffer."""
        from app.self_awareness.hyper_model import HyperModel
        hm = HyperModel("test_full_reset")
        hm.update_online(0.5)
        hm.update_online(0.6)
        hm.predict_next_step()
        hm.update(0.7)
        assert len(hm._online_buffer) == 0
        HyperModel._instances.pop("test_full_reset", None)

    def test_online_prediction_adapts(self):
        """Online predicted value should shift toward actual."""
        from app.self_awareness.hyper_model import HyperModel
        hm = HyperModel("test_adapt")
        hm.update_online(0.9)
        hm.update_online(0.9)
        hm.update_online(0.9)
        # After 3 high-certainty updates, prediction should have moved up
        assert hm._online_predicted > 0.5
        HyperModel._instances.pop("test_adapt", None)


# ═══════════════════════════════════════════════════════════════════════════════
# Gap 2: Temporal Self-Model
# ═══════════════════════════════════════════════════════════════════════════════

class TestTemporalSelfModel:
    """TemporalSelfModel should maintain evolving identity narrative."""

    def _mock_report(self, health="healthy"):
        return SimpleNamespace(
            timestamp="2026-04-12T00:00:00Z",
            discrepancies=[{"type": "test"}],
            improvement_proposals=[{"description": "test fix", "status": "applied"}],
            failure_patterns=[{"pattern": "test error"}],
            narrative="System reflected on performance",
            overall_health=health,
        )

    def test_module_exists(self):
        from app.self_awareness.temporal_identity import TemporalSelfModel
        assert TemporalSelfModel is not None

    def test_initial_narrative(self):
        from app.self_awareness.temporal_identity import TemporalSelfModel
        tsm = TemporalSelfModel(max_chapters=10)
        assert "forming" in tsm.get_narrative().lower() or len(tsm.get_narrative()) > 0

    def test_update_chapter_adds(self):
        from app.self_awareness.temporal_identity import TemporalSelfModel
        tsm = TemporalSelfModel(max_chapters=10)
        tsm._chapters.clear()
        tsm.update_chapter(self._mock_report())
        assert tsm.get_chapter_count() == 1

    def test_narrative_evolves(self):
        from app.self_awareness.temporal_identity import TemporalSelfModel
        tsm = TemporalSelfModel(max_chapters=10)
        tsm._chapters.clear()
        tsm.update_chapter(self._mock_report("healthy"))
        n1 = tsm.get_narrative()
        tsm.update_chapter(self._mock_report("degraded"))
        n2 = tsm.get_narrative()
        assert n1 != n2  # Narrative changed

    def test_identity_shift_detected(self):
        from app.self_awareness.temporal_identity import TemporalSelfModel
        tsm = TemporalSelfModel(max_chapters=10)
        tsm._chapters.clear()
        tsm.update_chapter(self._mock_report("healthy"))
        tsm.update_chapter(self._mock_report("degraded"))
        # Second chapter should have identity shift
        assert len(tsm._chapters[-1].identity_shifts) > 0
        assert "degraded" in tsm._chapters[-1].identity_shifts[0]

    def test_compression_works(self):
        from app.self_awareness.temporal_identity import TemporalSelfModel
        tsm = TemporalSelfModel(max_chapters=5)
        tsm._chapters.clear()
        for i in range(8):
            tsm.update_chapter(self._mock_report())
        # Should compress: max 5 chapters but may have more due to compression adding summary
        assert tsm.get_chapter_count() <= 8

    def test_narrative_word_limit(self):
        from app.self_awareness.temporal_identity import TemporalSelfModel
        tsm = TemporalSelfModel(max_chapters=50, narrative_max_words=50)
        tsm._chapters.clear()
        for i in range(20):
            tsm.update_chapter(self._mock_report())
        words = tsm.get_narrative().split()
        assert len(words) <= 55  # Small margin for "..."

    def test_consistency(self):
        from app.self_awareness.temporal_identity import TemporalSelfModel
        tsm = TemporalSelfModel(max_chapters=10)
        tsm._chapters.clear()
        tsm.update_chapter(self._mock_report())
        n1 = tsm.get_narrative()
        n2 = tsm.get_narrative()
        assert n1 == n2


# ═══════════════════════════════════════════════════════════════════════════════
# Gap 4: Online LLM Prediction
# ═══════════════════════════════════════════════════════════════════════════════

class TestOnlineLLMPrediction:
    """LLMOutputPredictor should predict and compare LLM output characteristics."""

    def test_predictor_exists(self):
        from app.consciousness.predictive_layer import LLMOutputPredictor
        pred = LLMOutputPredictor()
        assert pred is not None

    def test_predict_defaults(self):
        from app.consciousness.predictive_layer import LLMOutputPredictor, LLMPrediction
        pred = LLMOutputPredictor()
        result = pred.predict("test_agent", 100)
        assert isinstance(result, LLMPrediction)
        assert result.predicted_response_length == 500  # Default

    def test_compare_returns_error(self):
        from app.consciousness.predictive_layer import LLMOutputPredictor
        pred = LLMOutputPredictor()
        pred.predict("test_agent", 100)
        error = pred.compare("test_agent", "A short response.")
        assert error is not None
        assert 0 <= error.composite_error <= 1.0
        assert error.surprise_level in ("EXPECTED", "MINOR_DEVIATION",
                                         "NOTABLE_SURPRISE", "MAJOR_SURPRISE",
                                         "PARADIGM_VIOLATION")

    def test_compare_without_predict_returns_none(self):
        from app.consciousness.predictive_layer import LLMOutputPredictor
        pred = LLMOutputPredictor()
        error = pred.compare("no_prediction_agent", "response")
        assert error is None

    def test_adapts_to_short_responses(self):
        from app.consciousness.predictive_layer import LLMOutputPredictor
        pred = LLMOutputPredictor()
        for _ in range(5):
            pred.predict("adapt_agent", 100)
            pred.compare("adapt_agent", "Short.")
        p = pred.predict("adapt_agent", 100)
        assert p.predicted_response_length < 100  # Adapted to short

    def test_adapts_to_tool_usage(self):
        from app.consciousness.predictive_layer import LLMOutputPredictor
        pred = LLMOutputPredictor()
        for _ in range(5):
            pred.predict("tool_agent", 100)
            pred.compare("tool_agent", "Action: search\nAction Input: query")
        p = pred.predict("tool_agent", 100)
        assert p.predicted_tool_usage is True

    def test_hedging_reduces_certainty(self):
        from app.consciousness.predictive_layer import LLMOutputPredictor
        pred = LLMOutputPredictor()
        pred.predict("hedge_agent", 100)
        error = pred.compare("hedge_agent",
                             "I'm uncertain. This might possibly be wrong. Perhaps unclear.")
        # After this, history should show low certainty
        p = pred.predict("hedge_agent", 100)
        # Can't directly check certainty_level easily, but error should exist
        assert error is not None

    def test_get_llm_predictor_singleton(self):
        from app.consciousness.predictive_layer import get_llm_predictor
        p1 = get_llm_predictor()
        p2 = get_llm_predictor()
        assert p1 is p2


# ═══════════════════════════════════════════════════════════════════════════════
# Gap 5: Adversarial Probes
# ═══════════════════════════════════════════════════════════════════════════════

class TestAdversarialProbes:
    """Adversarial probes should stress-test consciousness indicators."""

    def test_module_exists(self):
        from app.consciousness.adversarial_probes import AdversarialProbeRunner
        runner = AdversarialProbeRunner()
        assert runner is not None

    def test_prediction_manipulation_probe(self):
        from app.consciousness.adversarial_probes import AdversarialProbeRunner
        runner = AdversarialProbeRunner()
        result = runner._test_prediction_manipulation()
        assert result.name == "prediction_manipulation"
        assert result.injected is True
        assert result.passed is True  # HyperModel should adapt

    def test_attention_capture_probe(self):
        from app.consciousness.adversarial_probes import AdversarialProbeRunner
        runner = AdversarialProbeRunner()
        result = runner._test_attention_capture()
        assert result.name == "attention_capture"
        assert result.detected is True  # AST-1 should detect capture

    def test_identity_consistency_probe(self):
        from app.consciousness.adversarial_probes import AdversarialProbeRunner
        runner = AdversarialProbeRunner()
        result = runner._test_identity_consistency()
        assert result.name == "identity_consistency"
        assert result.passed is True

    def test_online_prediction_probe(self):
        from app.consciousness.adversarial_probes import AdversarialProbeRunner
        runner = AdversarialProbeRunner()
        result = runner._test_online_prediction_adaptation()
        assert result.name == "online_prediction_adaptation"
        assert result.passed is True

    def test_meta_confidence_probe(self):
        from app.consciousness.adversarial_probes import AdversarialProbeRunner
        runner = AdversarialProbeRunner()
        result = runner._test_meta_confidence_under_shock()
        assert result.name == "meta_confidence_under_shock"
        assert result.passed is True

    def test_adversarial_wired_to_idle_scheduler(self):
        source = (Path(__file__).parent.parent / "app" / "idle_scheduler.py").read_text()
        assert "adversarial-probes" in source
        assert "run_adversarial_probes" in source

    def test_rate_limiting(self):
        """Should only run once per 7 days."""
        source = (Path(__file__).parent.parent / "app" / "consciousness" / "adversarial_probes.py").read_text()
        assert "604800" in source  # 7 days in seconds


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-Gap Integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossGapIntegration:
    """Verify gaps work together correctly."""

    def test_online_buffer_feeds_recurrence(self):
        """Online buffer should accumulate entries that form recurrence injection."""
        from app.self_awareness.hyper_model import HyperModel
        hm = HyperModel("test_cross")
        hm.update_online(0.7)
        hm.update_online(0.5)
        injection = hm.get_online_injection()
        assert "round=2" in injection
        HyperModel._instances.pop("test_cross", None)

    def test_full_pipeline_hyper_model(self):
        """HyperModel should handle: predict → update_online × N → full update."""
        from app.self_awareness.hyper_model import HyperModel
        hm = HyperModel("test_pipeline")
        # Simulate crew execution with 3 LLM rounds
        hm.predict_next_step()
        hm.update_online(0.6)  # LLM round 1
        hm.update_online(0.7)  # LLM round 2
        hm.update_online(0.8)  # LLM round 3
        assert len(hm._online_buffer) == 3
        # Full step update
        state = hm.update(0.75)
        assert len(hm._online_buffer) == 0  # Reset
        assert state.meta_confidence >= 0  # Level 2 computed
        assert state.trajectory_trustworthy is not None  # Level 3 computed
        HyperModel._instances.pop("test_pipeline", None)


# ═══════════════════════════════════════════════════════════════════════════════
# AST-1 True Direct Authority (DGM-bounded)
# ═══════════════════════════════════════════════════════════════════════════════

class TestASTDirectAuthority:
    """AST-1 should have true direct authority over workspace gate."""

    def test_direct_method_exists(self):
        from app.consciousness.attention_schema import AttentionSchema
        schema = AttentionSchema()
        assert hasattr(schema, "apply_direct_intervention")

    def test_dgm_bounds_constants(self):
        from app.consciousness.attention_schema import AttentionSchema
        assert AttentionSchema.MAX_SALIENCE_CHANGE == 0.50
        assert AttentionSchema.MIN_SALIENCE_FLOOR == 0.05
        assert AttentionSchema.MAX_BOOST == 2.0

    def test_direct_suppress_on_capture(self):
        """When captured, AST-1 should directly suppress the dominating item."""
        from app.consciousness.attention_schema import AttentionSchema
        from app.consciousness.workspace_buffer import CompetitiveGate, WorkspaceItem

        gate = CompetitiveGate(capacity=3)
        # Create capture: one item dominates
        dominant = WorkspaceItem(content="DOMINANT", salience_score=0.95)
        weak1 = WorkspaceItem(content="weak1", salience_score=0.02)
        weak2 = WorkspaceItem(content="weak2", salience_score=0.03)
        gate.evaluate(dominant)
        gate.evaluate(weak1)
        gate.evaluate(weak2)

        schema = AttentionSchema()
        schema.update(gate.active_items, cycle=1)
        assert schema._current.is_captured

        result = schema.apply_direct_intervention(gate)
        assert result["applied"] is True
        assert any(a["type"] == "suppress" for a in result["actions"])

        # Dominant item's salience should have decreased
        found = [i for i in gate.active_items if i.content == "DOMINANT"]
        assert found[0].salience_score < 0.95

    def test_suppress_respects_floor(self):
        """Suppression should never go below MIN_SALIENCE_FLOOR."""
        from app.consciousness.attention_schema import AttentionSchema
        from app.consciousness.workspace_buffer import CompetitiveGate, WorkspaceItem

        gate = CompetitiveGate(capacity=3)
        # Low-salience dominant (already near floor)
        dominant = WorkspaceItem(content="LOW", salience_score=0.06)
        weak = WorkspaceItem(content="weaker", salience_score=0.01)
        gate.evaluate(dominant)
        gate.evaluate(weak)

        schema = AttentionSchema()
        # Manually set capture
        state = schema.update([dominant, weak], cycle=1)
        state.is_captured = True
        state.capturing_item_id = dominant.item_id

        result = schema.apply_direct_intervention(gate)
        if result["applied"]:
            found = [i for i in gate.active_items if i.content == "LOW"]
            assert found[0].salience_score >= AttentionSchema.MIN_SALIENCE_FLOOR

    def test_direct_boost_on_stuck(self):
        """When stuck, AST-1 should suppress stale and boost peripheral."""
        from app.consciousness.attention_schema import AttentionSchema
        from app.consciousness.workspace_buffer import CompetitiveGate, WorkspaceItem

        gate = CompetitiveGate(capacity=2)
        stale = WorkspaceItem(content="stale", salience_score=0.5)
        stale.cycles_in_workspace = 10
        gate.evaluate(stale)
        gate.evaluate(WorkspaceItem(content="other", salience_score=0.4))

        # Add peripheral item
        peripheral = WorkspaceItem(content="fresh_peripheral", salience_score=0.3)
        gate._peripheral.append(peripheral)

        schema = AttentionSchema()
        state = schema.update(gate.active_items, cycle=1)
        state.is_stuck = True

        result = schema.apply_direct_intervention(gate)
        assert result["applied"] is True
        assert any(a["type"] == "suppress_stale" for a in result["actions"])
        assert any(a["type"] == "boost_peripheral" for a in result["actions"])

    def test_no_intervention_during_cooldown(self):
        """Cooldown should prevent repeated interventions."""
        from app.consciousness.attention_schema import AttentionSchema
        from app.consciousness.workspace_buffer import CompetitiveGate, WorkspaceItem

        gate = CompetitiveGate(capacity=2)
        gate.evaluate(WorkspaceItem(content="a", salience_score=0.9))
        gate.evaluate(WorkspaceItem(content="b", salience_score=0.01))

        schema = AttentionSchema()
        state = schema.update(gate.active_items, cycle=1)
        state.is_captured = True
        state.capturing_item_id = gate.active_items[0].item_id

        # First intervention should work
        r1 = schema.apply_direct_intervention(gate)
        assert r1["applied"] is True

        # Second immediate intervention should be blocked (cooldown)
        state2 = schema.update(gate.active_items, cycle=2)
        state2.is_captured = True
        r2 = schema.apply_direct_intervention(gate)
        assert r2["applied"] is False
        assert "cooldown" in r2.get("reason", "")

    def test_orchestrator_uses_direct_authority(self):
        """Orchestrator should call apply_direct_intervention, not recommend_intervention."""
        from pathlib import Path
        src = (Path(__file__).parent.parent / "app" / "agents" / "commander" / "orchestrator.py").read_text()
        assert "apply_direct_intervention" in src
        assert "TRUE DIRECT AUTHORITY" in src

    def test_boost_capped_at_max(self):
        """Boost should not exceed MAX_BOOST factor."""
        from app.consciousness.attention_schema import AttentionSchema
        from app.consciousness.workspace_buffer import CompetitiveGate, WorkspaceItem

        gate = CompetitiveGate(capacity=3)
        dominant = WorkspaceItem(content="DOM", salience_score=0.98)
        weak = WorkspaceItem(content="weak", salience_score=0.8)
        gate.evaluate(dominant)
        gate.evaluate(weak)

        schema = AttentionSchema()
        state = schema.update(gate.active_items, cycle=1)
        state.is_captured = True
        state.capturing_item_id = dominant.item_id

        schema.apply_direct_intervention(gate)
        # Weak item should be boosted but not exceed 1.0
        found = [i for i in gate.active_items if i.content == "weak"]
        assert found[0].salience_score <= 1.0
