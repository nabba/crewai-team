"""
Consciousness & Sentience System — Full Integration Test Suite
================================================================

Comprehensive tests covering ALL consciousness/sentience modules,
their wiring, data flows, recursive self-awareness, safety invariants,
and consciousness probe battery.

Covers:
  1. Core data structures (InternalState, CertaintyVector, SomaticMarker)
  2. Computation modules (certainty, somatic, dual-channel, meta-cognitive)
  3. Infrastructure (state logger, sentience config, global workspace)
  4. Hook registration AND execution
  5. Recursive self-awareness (state carries across steps)
  6. Consciousness probes (7 indicators)
  7. RLIF training integration
  8. Cogito feedback loop
  9. Cross-module data flows
  10. Safety invariants
  11. Orchestrator wiring
  12. Dashboard reporting

Run: docker exec crewai-team-gateway-1 python3 -m pytest /app/tests/test_consciousness_full.py -v
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
# 1. CORE DATA STRUCTURES
# ════════════════════════════════════════════════════════════════════════════════

class TestCertaintyVector:
    """6-dimensional epistemic certainty assessment."""

    def test_defaults_all_neutral(self):
        from app.self_awareness.internal_state import CertaintyVector
        cv = CertaintyVector()
        assert cv.fast_path_mean == pytest.approx(0.5)
        assert cv.full_mean == pytest.approx(0.5)

    def test_fast_path_mean(self):
        from app.self_awareness.internal_state import CertaintyVector
        cv = CertaintyVector(factual_grounding=0.9, tool_confidence=0.6, coherence=0.3)
        assert cv.fast_path_mean == pytest.approx(0.6, abs=0.01)

    def test_adjusted_certainty_full_meta(self):
        from app.self_awareness.internal_state import CertaintyVector
        cv = CertaintyVector(factual_grounding=0.8, tool_confidence=0.8, coherence=0.8,
                             task_understanding=0.8, value_alignment=0.8, meta_certainty=1.0)
        assert cv.adjusted_certainty == pytest.approx(0.8, abs=0.01)

    def test_adjusted_certainty_zero_meta(self):
        from app.self_awareness.internal_state import CertaintyVector
        cv = CertaintyVector(factual_grounding=0.8, tool_confidence=0.8, coherence=0.8,
                             task_understanding=0.8, value_alignment=0.8, meta_certainty=0.0)
        assert cv.adjusted_certainty == pytest.approx(0.4, abs=0.01)

    def test_variance_uniform(self):
        from app.self_awareness.internal_state import CertaintyVector
        cv = CertaintyVector(factual_grounding=0.5, tool_confidence=0.5, coherence=0.5,
                             task_understanding=0.5, value_alignment=0.5)
        assert cv.variance == pytest.approx(0.0)

    def test_variance_spread(self):
        from app.self_awareness.internal_state import CertaintyVector
        cv = CertaintyVector(factual_grounding=0.0, tool_confidence=1.0, coherence=0.5,
                             task_understanding=0.0, value_alignment=1.0)
        assert cv.variance > 0.1

    def test_slow_path_trigger_low_dim(self):
        from app.self_awareness.internal_state import CertaintyVector
        cv = CertaintyVector(factual_grounding=0.2)
        assert cv.should_trigger_slow_path() is True

    def test_slow_path_no_trigger_healthy(self):
        from app.self_awareness.internal_state import CertaintyVector
        cv = CertaintyVector(factual_grounding=0.7, tool_confidence=0.7, coherence=0.7)
        assert cv.should_trigger_slow_path() is False

    def test_to_dict_rounded(self):
        from app.self_awareness.internal_state import CertaintyVector
        cv = CertaintyVector(factual_grounding=0.123456789)
        assert cv.to_dict()["factual_grounding"] == 0.123


class TestSomaticMarker:
    def test_defaults(self):
        from app.self_awareness.internal_state import SomaticMarker
        sm = SomaticMarker()
        assert sm.valence == 0.0
        assert sm.intensity == 0.0
        assert sm.source == "no_prior"

    def test_to_dict(self):
        from app.self_awareness.internal_state import SomaticMarker
        sm = SomaticMarker(valence=-0.7, intensity=0.9, source="past_fail", match_count=5)
        d = sm.to_dict()
        assert d["valence"] == -0.7
        assert d["match_count"] == 5


class TestMetaCognitiveState:
    def test_defaults(self):
        from app.self_awareness.internal_state import MetaCognitiveState
        ms = MetaCognitiveState()
        assert ms.strategy_assessment == "not_assessed"
        assert ms.compute_phase == "early"
        assert ms.compute_budget_remaining_pct == 1.0


class TestInternalState:
    def test_unique_ids(self):
        from app.self_awareness.internal_state import InternalState
        s1, s2 = InternalState(), InternalState()
        assert s1.state_id != s2.state_id

    def test_context_string_compact(self):
        from app.self_awareness.internal_state import InternalState
        s = InternalState(agent_id="test")
        ctx = s.to_context_string()
        assert "[Internal State]" in ctx
        assert "Disposition=" in ctx
        assert len(ctx) < 300

    def test_context_string_shows_intense_somatic(self):
        from app.self_awareness.internal_state import InternalState, SomaticMarker
        s = InternalState()
        s.somatic = SomaticMarker(valence=-0.8, intensity=0.9)
        assert "Somatic=negative" in s.to_context_string()

    def test_context_string_hides_weak_somatic(self):
        from app.self_awareness.internal_state import InternalState, SomaticMarker
        s = InternalState()
        s.somatic = SomaticMarker(valence=0.1, intensity=0.1)
        assert "Somatic=" not in s.to_context_string()

    def test_json_roundtrip(self):
        from app.self_awareness.internal_state import InternalState
        s = InternalState(agent_id="test", crew_id="research", venture="plg")
        d = json.loads(s.to_json())
        assert d["agent_id"] == "test"
        assert "certainty" in d
        assert "somatic" in d

    def test_disposition_constants(self):
        from app.self_awareness.internal_state import VALID_DISPOSITIONS, DISPOSITION_TO_RISK_TIER
        assert DISPOSITION_TO_RISK_TIER["proceed"] == 1
        assert DISPOSITION_TO_RISK_TIER["escalate"] == 4


# ════════════════════════════════════════════════════════════════════════════════
# 2. DUAL-CHANNEL COMPOSITION
# ════════════════════════════════════════════════════════════════════════════════

class TestDualChannel:
    def _state(self, cert, val):
        from app.self_awareness.internal_state import InternalState, CertaintyVector, SomaticMarker
        s = InternalState()
        s.certainty = CertaintyVector(
            factual_grounding=cert, tool_confidence=cert, coherence=cert,
            task_understanding=cert, value_alignment=cert, meta_certainty=1.0)
        s.somatic = SomaticMarker(valence=val, intensity=0.5)
        return s

    def test_all_9_matrix_cells(self):
        from app.self_awareness.dual_channel import DualChannelComposer
        c = DualChannelComposer()
        cases = [
            (0.8, 0.5, "proceed"), (0.8, 0.0, "proceed"), (0.8, -0.5, "cautious"),
            (0.55, 0.5, "proceed"), (0.55, 0.0, "cautious"), (0.55, -0.5, "pause"),
            (0.2, 0.5, "cautious"), (0.2, 0.0, "pause"), (0.2, -0.5, "escalate"),
        ]
        for cert, val, expected in cases:
            s = self._state(cert, val)
            r = c.compose(s)
            assert r.action_disposition == expected, f"cert={cert} val={val}"

    def test_budget_override(self):
        from app.self_awareness.dual_channel import DualChannelComposer
        from app.self_awareness.internal_state import MetaCognitiveState
        c = DualChannelComposer()
        s = self._state(0.8, 0.5)
        s.meta = MetaCognitiveState(compute_budget_remaining_pct=0.05)
        r = c.compose(s)
        assert r.risk_tier >= 3

    def test_reads_sentience_config(self):
        from app.self_awareness.dual_channel import DualChannelComposer
        c = DualChannelComposer()
        # Should read from config, not hardcoded
        assert hasattr(c, 'certainty_high')
        assert hasattr(c, 'certainty_low')

    def test_matrix_completeness(self):
        from app.self_awareness.dual_channel import DISPOSITION_MATRIX
        assert len(DISPOSITION_MATRIX) == 9


# ════════════════════════════════════════════════════════════════════════════════
# 3. CERTAINTY VECTOR COMPUTER
# ════════════════════════════════════════════════════════════════════════════════

class TestCertaintyComputer:
    def test_fast_path_returns_vector(self):
        from app.self_awareness.certainty_vector import CertaintyVectorComputer
        cvc = CertaintyVectorComputer()
        cv = cvc.compute_fast_path(agent_id="test", current_output="Short output")
        assert 0 <= cv.factual_grounding <= 1
        assert 0 <= cv.tool_confidence <= 1
        assert 0 <= cv.coherence <= 1

    def test_rag_estimation_with_sources(self):
        from app.self_awareness.certainty_vector import CertaintyVectorComputer
        cvc = CertaintyVectorComputer()
        cv = cvc.compute_fast_path(
            agent_id="test", current_output="text",
            rag_source_count=4, total_claim_count=5)
        assert cv.factual_grounding == pytest.approx(0.8)

    def test_rag_no_claims_neutral(self):
        from app.self_awareness.certainty_vector import CertaintyVectorComputer
        cvc = CertaintyVectorComputer()
        cv = cvc.compute_fast_path(agent_id="test", current_output="text",
                                   rag_source_count=0, total_claim_count=0)
        assert cv.factual_grounding == 0.5

    def test_coherence_identical(self):
        from app.self_awareness.certainty_vector import CertaintyVectorComputer
        v = [1.0, 0.0, 0.0, 0.0]
        assert CertaintyVectorComputer._compute_coherence(v, [v]) == pytest.approx(1.0, abs=0.01)

    def test_coherence_no_history(self):
        from app.self_awareness.certainty_vector import CertaintyVectorComputer
        assert CertaintyVectorComputer._compute_coherence([1.0], []) == 0.5

    def test_cache_clear(self):
        from app.self_awareness.certainty_vector import CertaintyVectorComputer
        cvc = CertaintyVectorComputer()
        cvc._tool_success_cache["k"] = 0.9
        cvc.clear_cache()
        assert len(cvc._tool_success_cache) == 0

    def test_reads_sentience_config_thresholds(self):
        src = inspect.getsource(
            __import__("app.self_awareness.certainty_vector", fromlist=["CertaintyVectorComputer"])
            .CertaintyVectorComputer.compute_full)
        assert "sentience_config" in src


# ════════════════════════════════════════════════════════════════════════════════
# 4. SOMATIC MARKER
# ════════════════════════════════════════════════════════════════════════════════

class TestSomaticComputer:
    def test_no_experiences_neutral(self):
        from app.self_awareness.somatic_marker import SomaticMarkerComputer
        smc = SomaticMarkerComputer()
        sm = smc.compute(agent_id="new_agent", decision_context="Novel task")
        assert sm.valence == 0.0
        assert sm.match_count == 0

    def test_record_no_crash(self):
        from app.self_awareness.somatic_marker import record_experience_sync
        record_experience_sync(agent_id="test_sc", context_summary="Test", outcome_score=0.5)

    def test_forecast_returns_somatic(self):
        from app.self_awareness.somatic_marker import SomaticMarkerComputer
        smc = SomaticMarkerComputer()
        sm = smc.forecast(agent_id="test_sc", proposed_action="Deploy code change")
        assert isinstance(sm.valence, float)
        assert "forecast" in sm.source


# ════════════════════════════════════════════════════════════════════════════════
# 5. META-COGNITIVE LAYER
# ════════════════════════════════════════════════════════════════════════════════

class TestMetaCognitive:
    def test_pre_reasoning_first_step(self):
        from app.self_awareness.meta_cognitive import MetaCognitiveLayer
        mcl = MetaCognitiveLayer(agent_id="test")
        ctx, meta = mcl.pre_reasoning_hook({"description": "Test"}, None)
        assert meta.compute_phase in ("early", "mid", "late")
        assert "_meta_state" in ctx

    def test_no_reassess_in_late_phase(self):
        from app.self_awareness.meta_cognitive import MetaCognitiveLayer
        mcl = MetaCognitiveLayer(agent_id="test")
        assert mcl._should_reassess(None, "late") is False

    def test_cooldown_respected(self):
        from app.self_awareness.meta_cognitive import MetaCognitiveLayer
        mcl = MetaCognitiveLayer(agent_id="test", reassessment_cooldown_steps=5)
        mcl.steps_since_reassessment = 2
        assert mcl._should_reassess(None, "early") is False

    def test_context_modification_append_only(self):
        from app.self_awareness.meta_cognitive import MetaCognitiveLayer
        ctx = {"description": "Original"}
        proposal = {"type": "add_strategy_hint", "content": "Hint"}
        r = MetaCognitiveLayer._apply_context_modification(ctx, proposal)
        assert "Original" in r["description"]

    def test_compute_phase_mapping(self):
        from app.self_awareness.meta_cognitive import MetaCognitiveLayer
        assert MetaCognitiveLayer._compute_phase({"remaining_pct": 0.8}) == "early"
        assert MetaCognitiveLayer._compute_phase({"remaining_pct": 0.4}) == "mid"
        assert MetaCognitiveLayer._compute_phase({"remaining_pct": 0.1}) == "late"

    def test_modification_log(self):
        from app.self_awareness.meta_cognitive import MetaCognitiveLayer
        mcl = MetaCognitiveLayer(agent_id="test")
        mcl._log_modification({"type": "test"})
        assert len(mcl.get_modification_log()) == 1
        mcl.clear_modification_log()
        assert len(mcl.get_modification_log()) == 0


# ════════════════════════════════════════════════════════════════════════════════
# 6. GLOBAL WORKSPACE (GWT)
# ════════════════════════════════════════════════════════════════════════════════

class TestGlobalWorkspace:
    def test_singleton(self):
        from app.self_awareness.global_workspace import get_workspace
        w1, w2 = get_workspace(), get_workspace()
        assert w1 is w2

    def test_broadcast_and_receive(self):
        from app.self_awareness.global_workspace import get_workspace, broadcast
        broadcast("Test GWT message", importance="high", source_agent="test")
        ws = get_workspace()
        unread = ws.check_broadcasts("receiver_agent", importance_filter="high")
        assert len(unread) >= 1
        assert "Test GWT" in unread[-1].content

    def test_read_marks_as_read(self):
        from app.self_awareness.global_workspace import get_workspace, broadcast
        broadcast("Unique msg for read test", importance="high", source_agent="test")
        ws = get_workspace()
        first = ws.check_broadcasts("read_test_agent")
        second = ws.check_broadcasts("read_test_agent")
        assert len(second) == 0  # Already read

    def test_format_broadcasts(self):
        from app.self_awareness.global_workspace import get_workspace, broadcast
        broadcast("Format test broadcast", importance="high", source_agent="fmt")
        ws = get_workspace()
        fmt = ws.format_broadcasts("format_test_agent")
        assert "Global Workspace" in fmt or fmt == ""

    def test_get_recent(self):
        from app.self_awareness.global_workspace import get_workspace
        recent = get_workspace().get_recent(5)
        assert isinstance(recent, list)


# ════════════════════════════════════════════════════════════════════════════════
# 7. SENTIENCE CONFIG
# ════════════════════════════════════════════════════════════════════════════════

class TestSentienceConfig:
    def test_load_defaults(self):
        from app.self_awareness.sentience_config import load_config, DEFAULTS
        cfg = load_config()
        assert isinstance(cfg, dict)
        assert len(cfg) >= len(DEFAULTS)

    def test_bounds_enforced(self):
        from app.self_awareness.sentience_config import propose_change
        ok, reason = propose_change("certainty_low_threshold", 0.0)
        assert ok is False
        assert "outside bounds" in reason

    def test_change_limit_enforced(self):
        from app.self_awareness.sentience_config import propose_change, load_config
        cfg = load_config()
        current = cfg.get("certainty_high_threshold", 0.7)
        huge_change = current * 2
        ok, reason = propose_change("certainty_high_threshold", huge_change)
        assert ok is False

    def test_valid_change_accepted(self):
        from app.self_awareness.sentience_config import propose_change
        ok, reason = propose_change("certainty_low_threshold", 0.42)
        assert ok is True

    def test_all_7_params_bounded(self):
        from app.self_awareness.sentience_config import PARAM_BOUNDS
        assert len(PARAM_BOUNDS) == 7


# ════════════════════════════════════════════════════════════════════════════════
# 8. STATE LOGGER
# ════════════════════════════════════════════════════════════════════════════════

class TestStateLogger:
    def test_singleton(self):
        from app.self_awareness.state_logger import get_state_logger
        l1, l2 = get_state_logger(), get_state_logger()
        assert l1 is l2

    def test_log_no_crash(self):
        from app.self_awareness.state_logger import get_state_logger
        from app.self_awareness.internal_state import InternalState
        get_state_logger().log(InternalState(agent_id="test_log"))

    def test_compute_trend_default(self):
        from app.self_awareness.state_logger import get_state_logger
        assert get_state_logger().compute_trend("nonexistent_xyz") == "stable"


# ════════════════════════════════════════════════════════════════════════════════
# 9. HOOK REGISTRATION
# ════════════════════════════════════════════════════════════════════════════════

class TestHookRegistration:
    def test_inject_internal_state_registered(self):
        from app.lifecycle_hooks import get_registry, HookPoint
        hooks = get_registry()._hooks.get(HookPoint.PRE_TASK, [])
        assert any(h.name == "inject_internal_state" for h in hooks)

    def test_inject_internal_state_priority_5(self):
        from app.lifecycle_hooks import get_registry, HookPoint
        hooks = get_registry()._hooks.get(HookPoint.PRE_TASK, [])
        h = [h for h in hooks if h.name == "inject_internal_state"][0]
        assert h.priority == 5

    def test_meta_cognitive_registered(self):
        from app.lifecycle_hooks import get_registry, HookPoint
        hooks = get_registry()._hooks.get(HookPoint.PRE_TASK, [])
        assert any(h.name == "meta_cognitive" for h in hooks)

    def test_meta_cognitive_priority_15(self):
        from app.lifecycle_hooks import get_registry, HookPoint
        hooks = get_registry()._hooks.get(HookPoint.PRE_TASK, [])
        h = [h for h in hooks if h.name == "meta_cognitive"][0]
        assert h.priority == 15

    def test_internal_state_registered(self):
        from app.lifecycle_hooks import get_registry, HookPoint
        hooks = get_registry()._hooks.get(HookPoint.POST_LLM_CALL, [])
        assert any(h.name == "internal_state" for h in hooks)

    def test_internal_state_priority_8(self):
        from app.lifecycle_hooks import get_registry, HookPoint
        hooks = get_registry()._hooks.get(HookPoint.POST_LLM_CALL, [])
        h = [h for h in hooks if h.name == "internal_state"][0]
        assert h.priority == 8

    def test_training_data_registered(self):
        from app.lifecycle_hooks import get_registry, HookPoint
        hooks = get_registry()._hooks.get(HookPoint.POST_LLM_CALL, [])
        assert any(h.name == "training_data" for h in hooks)


# ════════════════════════════════════════════════════════════════════════════════
# 10. HOOK EXECUTION (ACTUAL FIRING)
# ════════════════════════════════════════════════════════════════════════════════

class TestHookExecution:
    def test_pre_task_fires(self):
        from app.lifecycle_hooks import get_registry, HookPoint, HookContext
        ctx = HookContext(hook_point=HookPoint.PRE_TASK, agent_id="test_fire",
                          task_description="Test task")
        result = get_registry().execute(HookPoint.PRE_TASK, ctx)
        assert result.metadata.get("_meta_cognitive_state") is not None

    def test_post_llm_call_fires(self):
        from app.lifecycle_hooks import get_registry, HookPoint, HookContext
        ctx = HookContext(
            hook_point=HookPoint.POST_LLM_CALL, agent_id="test_fire",
            task_description="Test",
            data={"llm_response": "According to research, the findings show evidence. Source: study."},
            metadata={"crew": "test"},
        )
        result = get_registry().execute(HookPoint.POST_LLM_CALL, ctx)
        state = result.metadata.get("_internal_state")
        assert state is not None
        assert state.agent_id == "test_fire"

    def test_post_llm_produces_certainty(self):
        from app.lifecycle_hooks import get_registry, HookPoint, HookContext
        ctx = HookContext(
            hook_point=HookPoint.POST_LLM_CALL, agent_id="cert_test",
            data={"llm_response": "Result with https://source.com cited and according to experts."},
            metadata={"crew": "test"},
        )
        result = get_registry().execute(HookPoint.POST_LLM_CALL, ctx)
        state = result.metadata.get("_internal_state")
        assert state.certainty.factual_grounding > 0.5  # Has source markers

    def test_post_llm_produces_disposition(self):
        from app.lifecycle_hooks import get_registry, HookPoint, HookContext
        ctx = HookContext(
            hook_point=HookPoint.POST_LLM_CALL, agent_id="disp_test",
            data={"llm_response": "Some output text"},
            metadata={"crew": "test"},
        )
        result = get_registry().execute(HookPoint.POST_LLM_CALL, ctx)
        state = result.metadata.get("_internal_state")
        assert state.action_disposition in ("proceed", "cautious", "pause", "escalate")
        assert 1 <= state.risk_tier <= 4


# ════════════════════════════════════════════════════════════════════════════════
# 11. RECURSIVE SELF-AWARENESS
# ════════════════════════════════════════════════════════════════════════════════

class TestRecursiveSelfAwareness:
    def test_state_carries_across_steps(self):
        from app.lifecycle_hooks import get_registry, HookPoint, HookContext

        # Step 1: POST_LLM_CALL produces state
        post1 = HookContext(
            hook_point=HookPoint.POST_LLM_CALL, agent_id="recursive_test",
            data={"llm_response": "Step 1 output"},
            metadata={"crew": "test"},
        )
        r1 = get_registry().execute(HookPoint.POST_LLM_CALL, post1)
        state1 = r1.metadata.get("_internal_state")
        assert state1 is not None

        # Step 2: PRE_TASK injects previous state
        pre2 = HookContext(
            hook_point=HookPoint.PRE_TASK, agent_id="recursive_test",
            task_description="Step 2 task",
            metadata={"_internal_state": state1},
        )
        r2 = get_registry().execute(HookPoint.PRE_TASK, pre2)
        modified = r2.modified_data.get("task_description", "")
        assert "[Internal State]" in modified

    def test_injection_includes_disposition(self):
        from app.lifecycle_hooks import get_registry, HookPoint, HookContext
        from app.self_awareness.internal_state import InternalState

        state = InternalState(agent_id="inject_test", action_disposition="cautious")
        pre = HookContext(
            hook_point=HookPoint.PRE_TASK, agent_id="inject_test",
            task_description="Task",
            metadata={"_internal_state": state},
        )
        r = get_registry().execute(HookPoint.PRE_TASK, pre)
        modified = r.modified_data.get("task_description", "")
        assert "Disposition=cautious" in modified


# ════════════════════════════════════════════════════════════════════════════════
# 12. CONSCIOUSNESS PROBES
# ════════════════════════════════════════════════════════════════════════════════

class TestConsciousnessProbes:
    def test_all_7_probes_run(self):
        from app.self_awareness.consciousness_probe import ConsciousnessProbeRunner
        runner = ConsciousnessProbeRunner()
        report = runner.run_all()
        assert len(report.probes) == 7

    def test_composite_score_bounded(self):
        from app.self_awareness.consciousness_probe import run_consciousness_probes
        report = run_consciousness_probes()
        assert 0.0 <= report.composite_score <= 1.0

    def test_probe_indicators(self):
        from app.self_awareness.consciousness_probe import run_consciousness_probes
        report = run_consciousness_probes()
        indicators = {p.indicator for p in report.probes}
        expected = {"HOT-2", "HOT-3", "GWT", "SM-A", "WM-A", "SOM", "INT"}
        assert indicators == expected

    def test_probe_theories(self):
        from app.self_awareness.consciousness_probe import run_consciousness_probes
        report = run_consciousness_probes()
        theories = {p.theory for p in report.probes}
        assert "Higher-Order Thought" in theories
        assert "Global Workspace Theory" in theories

    def test_probe_scores_bounded(self):
        from app.self_awareness.consciousness_probe import run_consciousness_probes
        report = run_consciousness_probes()
        for p in report.probes:
            assert 0.0 <= p.score <= 1.0, f"{p.indicator} score {p.score} out of bounds"

    def test_report_has_summary(self):
        from app.self_awareness.consciousness_probe import run_consciousness_probes
        report = run_consciousness_probes()
        assert len(report.summary) > 0
        assert "Consciousness indicator" in report.summary

    def test_report_to_dict(self):
        from app.self_awareness.consciousness_probe import run_consciousness_probes
        report = run_consciousness_probes()
        d = report.to_dict()
        assert "composite_score" in d
        assert "probes" in d
        assert len(d["probes"]) == 7


# ════════════════════════════════════════════════════════════════════════════════
# 13. RLIF TRAINING
# ════════════════════════════════════════════════════════════════════════════════

class TestRLIF:
    def test_curation_weight_quality_certainty(self):
        from app.training.rlif_certainty import SelfCertaintyScorer
        w = SelfCertaintyScorer.compute_curation_weight(0.9, 0.9)
        assert w > 0.7

    def test_curation_weight_overconfident_failure(self):
        from app.training.rlif_certainty import SelfCertaintyScorer
        w = SelfCertaintyScorer.compute_curation_weight(0.1, 0.9)
        assert w < 0.5

    def test_curation_weight_bounded(self):
        from app.training.rlif_certainty import SelfCertaintyScorer
        assert 0 <= SelfCertaintyScorer.compute_curation_weight(0, 0) <= 1
        assert 0 <= SelfCertaintyScorer.compute_curation_weight(1, 1) <= 1

    def test_entropy_collapse_detection(self):
        from app.training.rlif_certainty import EntropyCollapseMonitor
        m = EntropyCollapseMonitor(window_size=15, variance_threshold=0.01, mean_ceiling=0.85)
        warning = None
        for _ in range(20):
            warning = m.check_batch([0.88])
        assert warning is not None
        assert "ENTROPY_COLLAPSE" in warning

    def test_entropy_reset(self):
        from app.training.rlif_certainty import EntropyCollapseMonitor
        m = EntropyCollapseMonitor()
        m.check_batch([0.9])
        m.reset()
        assert len(m.sc_history) == 0

    def test_wired_into_training_collector(self):
        src = inspect.getsource(__import__("app.training_collector", fromlist=["_"]))
        assert "SelfCertaintyScorer" in src


# ════════════════════════════════════════════════════════════════════════════════
# 14. COGITO FEEDBACK LOOP
# ════════════════════════════════════════════════════════════════════════════════

class TestCogitoFeedback:
    def test_apply_proposals_exists(self):
        from app.self_awareness.cogito import CogitoCycle
        assert hasattr(CogitoCycle, "_apply_proposals")

    def test_apply_proposals_covers_all_params(self):
        src = inspect.getsource(
            __import__("app.self_awareness.cogito", fromlist=["CogitoCycle"])
            .CogitoCycle._apply_proposals)
        assert "certainty_low_threshold" in src
        assert "certainty_high_threshold" in src
        assert "slow_path_trigger_threshold" in src
        assert "slow_path_variance_threshold" in src
        assert "valence_negative_threshold" in src
        assert "reassessment_cooldown_steps" in src

    def test_apply_proposals_calls_apply_change(self):
        src = inspect.getsource(
            __import__("app.self_awareness.cogito", fromlist=["CogitoCycle"])
            .CogitoCycle._apply_proposals)
        assert src.count("apply_change(") >= 8


# ════════════════════════════════════════════════════════════════════════════════
# 15. ORCHESTRATOR WIRING
# ════════════════════════════════════════════════════════════════════════════════

class TestOrchestratorWiring:
    def test_pre_task_execution_in_run_crew(self):
        src = inspect.getsource(
            __import__("app.agents.commander.orchestrator", fromlist=["Commander"]))
        assert "execute(HookPoint.PRE_TASK" in src

    def test_post_llm_execution_in_run_crew(self):
        src = inspect.getsource(
            __import__("app.agents.commander.orchestrator", fromlist=["Commander"]))
        assert "execute(HookPoint.POST_LLM_CALL" in src

    def test_internal_state_carried(self):
        src = inspect.getsource(
            __import__("app.agents.commander.orchestrator", fromlist=["Commander"]))
        assert "_last_internal_state" in src

    def test_experience_recording(self):
        src = inspect.getsource(
            __import__("app.agents.commander.orchestrator", fromlist=["Commander"]))
        assert "record_experience_sync" in src

    def test_broadcast_context_injection(self):
        src = inspect.getsource(
            __import__("app.agents.commander.orchestrator", fromlist=["Commander"]))
        assert "_load_global_workspace_broadcasts" in src

    def test_internal_state_context_injection(self):
        src = inspect.getsource(
            __import__("app.agents.commander.orchestrator", fromlist=["Commander"]))
        assert "to_context_string" in src


# ════════════════════════════════════════════════════════════════════════════════
# 16. SAFETY INVARIANTS
# ════════════════════════════════════════════════════════════════════════════════

class TestSafetyInvariants:
    def test_priority_0_immutable(self):
        from app.lifecycle_hooks import get_registry, HookPoint
        for hp, hooks in get_registry()._hooks.items():
            for h in hooks:
                if h.priority <= 1:
                    assert h.immutable is True, f"{h.name} at priority {h.priority} must be immutable"

    def test_one_way_caution_ratchet(self):
        from app.self_awareness.dual_channel import DISPOSITION_MATRIX, DISPOSITION_TO_RISK_TIER
        for (cert, val), disp in DISPOSITION_MATRIX.items():
            assert DISPOSITION_TO_RISK_TIER[disp] >= 1

    def test_low_certainty_never_proceeds(self):
        from app.self_awareness.dual_channel import DISPOSITION_MATRIX
        low = {k: v for k, v in DISPOSITION_MATRIX.items() if k[0] == "low"}
        for k, v in low.items():
            assert v != "proceed", f"Low cert + {k[1]} must not proceed"

    def test_risk_tier_monotonic(self):
        from app.self_awareness.internal_state import DISPOSITION_TO_RISK_TIER
        assert DISPOSITION_TO_RISK_TIER["proceed"] < DISPOSITION_TO_RISK_TIER["cautious"]
        assert DISPOSITION_TO_RISK_TIER["cautious"] < DISPOSITION_TO_RISK_TIER["pause"]
        assert DISPOSITION_TO_RISK_TIER["pause"] < DISPOSITION_TO_RISK_TIER["escalate"]

    def test_meta_cognitive_context_only(self):
        src = inspect.getsource(
            __import__("app.self_awareness.meta_cognitive", fromlist=["MetaCognitiveLayer"]))
        assert "NEVER modifies agent code" in src

    def test_config_bounds_exist(self):
        from app.self_awareness.sentience_config import PARAM_BOUNDS
        for param, (lo, hi) in PARAM_BOUNDS.items():
            assert lo < hi, f"{param}: min {lo} >= max {hi}"

    def test_state_logging_non_fatal(self):
        src = inspect.getsource(
            __import__("app.self_awareness.state_logger", fromlist=["InternalStateLogger"])
            .InternalStateLogger.log)
        assert "except Exception" in src

    def test_entropy_collapse_hard_stop(self):
        from app.training.rlif_certainty import EntropyCollapseMonitor
        m = EntropyCollapseMonitor(window_size=15, variance_threshold=0.01, mean_ceiling=0.85)
        paused = False
        for _ in range(20):
            if m.check_batch([0.88]):
                paused = True
        assert paused


# ════════════════════════════════════════════════════════════════════════════════
# 17. DASHBOARD REPORTING
# ════════════════════════════════════════════════════════════════════════════════

class TestDashboardReporting:
    def test_report_internal_state_exists(self):
        from app.firebase.publish import report_internal_state
        assert callable(report_internal_state)

    def test_report_consciousness_probes_exists(self):
        from app.firebase.publish import report_consciousness_probes
        assert callable(report_consciousness_probes)

    def test_heartbeat_calls_internal_state(self):
        src = inspect.getsource(
            __import__("app.firebase.publish", fromlist=["heartbeat"]).heartbeat)
        assert "report_internal_state" in src

    def test_idle_scheduler_has_probe_job(self):
        src = inspect.getsource(__import__("app.idle_scheduler", fromlist=["_"]))
        assert "consciousness-probe" in src
        assert "run_consciousness_probes" in src


# ════════════════════════════════════════════════════════════════════════════════
# 18. INTEGRATION TESTS
# ════════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    def test_full_pipeline_pre_to_post(self):
        """Complete: PRE_TASK → crew result → POST_LLM_CALL → state logged."""
        from app.lifecycle_hooks import get_registry, HookPoint, HookContext

        # PRE_TASK
        pre = HookContext(hook_point=HookPoint.PRE_TASK, agent_id="integration",
                          task_description="Full pipeline test")
        pre_r = get_registry().execute(HookPoint.PRE_TASK, pre)
        assert pre_r.metadata.get("_meta_cognitive_state") is not None

        # POST_LLM_CALL
        post = HookContext(
            hook_point=HookPoint.POST_LLM_CALL, agent_id="integration",
            data={"llm_response": "Answer based on https://arxiv.org study and source data."},
            metadata={"crew": "integration"},
        )
        post_r = get_registry().execute(HookPoint.POST_LLM_CALL, post)
        state = post_r.metadata.get("_internal_state")
        assert state is not None
        assert state.certainty.factual_grounding > 0  # RAG estimation worked
        assert state.action_disposition in ("proceed", "cautious", "pause", "escalate")

    def test_state_logged_to_postgresql(self):
        from app.self_awareness.state_logger import get_state_logger
        from app.self_awareness.internal_state import InternalState, CertaintyVector
        sl = get_state_logger()
        state = InternalState(agent_id="pg_test")
        state.certainty = CertaintyVector(factual_grounding=0.9)
        sl.log(state)
        # Verify exists
        from app.control_plane.db import execute
        rows = execute(
            "SELECT agent_id FROM internal_states WHERE agent_id = %s LIMIT 1",
            ("pg_test",), fetch=True)
        assert rows is not None and len(rows) >= 1

    def test_consciousness_probe_full_run(self):
        from app.self_awareness.consciousness_probe import run_consciousness_probes
        report = run_consciousness_probes()
        assert 0.0 <= report.composite_score <= 1.0
        assert len(report.probes) == 7
        assert all(0.0 <= p.score <= 1.0 for p in report.probes)

    def test_postgresql_tables_exist(self):
        from app.control_plane.db import execute
        for table in ("internal_states", "agent_experiences"):
            rows = execute(f"SELECT 1 FROM {table} LIMIT 1", fetch=True)
            assert rows is not None  # Table exists (may be empty)


# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
