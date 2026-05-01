"""
Phase 4: CIL loop sequencer regression tests.

Verifies the 11-step SubIALoop composes the Phase-2 gates correctly:

  - Step ordering matches the CIL diagram (1→2→3 then optional
    4→5→5b→6 for full loop; 8→9 then optional 10→11 for post-task).
  - Operation classification picks the right path per SUBIA_CONFIG.
  - Step-level failures are contained (the agent never sees an
    exception; the result records ok=False and keeps going).
  - Performance budget is recorded per call.
  - PP-1 auto-routing fires when a gate is attached to the
    predictive_layer (end-to-end).
  - HOT-3 dispatch decision is surfaced in the injectable context.
  - Cascade recommendation reflects prediction confidence.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from unittest.mock import MagicMock

# Stub DB/embedding layers that Phase-1 gates pull in transitively.
for _mod in ["psycopg2", "psycopg2.pool", "psycopg2.extras",
             "app.memory.chromadb_manager"]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
sys.modules["app.memory.chromadb_manager"].embed = MagicMock(return_value=[0.1] * 768)

import pytest

from app.subia.kernel import Prediction, SceneItem, SubjectivityKernel
from app.subia.loop import (
    CILResult,
    StepOutcome,
    SubIALoop,
    classify_operation,
)
from app.subia.scene.buffer import CompetitiveGate, WorkspaceItem


# ── Helpers ───────────────────────────────────────────────────────

@dataclass
class FakeBelief:
    belief_id: str = "b1"
    confidence: float = 0.8
    belief_status: str = "ACTIVE"


def _make_loop(**overrides) -> SubIALoop:
    kernel = SubjectivityKernel()
    gate = CompetitiveGate(capacity=5)

    def default_predict(ctx):
        return Prediction(
            id="p-fake",
            operation=f"{ctx['agent_role']}:{ctx['task_description'][:40]}",
            predicted_outcome={"wiki_pages_affected": []},
            predicted_self_change={"confidence_change": 0.1},
            predicted_homeostatic_effect={},
            confidence=0.75,
            created_at="2026-04-13T00:00:00Z",
        )

    def default_consult(task_description, crew_name, goal_context):
        return [FakeBelief(belief_id="b1", confidence=0.85)]

    base = dict(
        kernel=kernel,
        scene_gate=gate,
        predict_fn=default_predict,
        consult_fn=default_consult,
        predictive_layer=None,
        hierarchy=None,
    )
    base.update(overrides)
    return SubIALoop(**base)


# ── Operation classification ──────────────────────────────────────

class TestOperationClassification:
    def test_full_operations_route_to_full(self):
        assert classify_operation("task_execute") == "full"
        assert classify_operation("ingest") == "full"
        assert classify_operation("lint") == "full"

    def test_compressed_operations_route_to_compressed(self):
        assert classify_operation("wiki_read") == "compressed"
        assert classify_operation("wiki_search") == "compressed"
        assert classify_operation("routine_query") == "compressed"

    def test_unknown_defaults_to_compressed(self):
        """Unknown ops should be cheap by default, not expensive."""
        assert classify_operation("totally_new_thing") == "compressed"


# ── Full pre_task pipeline ────────────────────────────────────────

class TestFullPreTask:
    def test_runs_all_pre_task_steps_in_order(self):
        loop = _make_loop()
        result = loop.pre_task(
            agent_role="researcher",
            task_description="gather competitive intel",
            operation_type="task_execute",
            input_items=[
                WorkspaceItem(item_id="w1", content="x", salience_score=0.7),
            ],
        )

        assert result.loop_type == "full"
        assert result.phase == "pre_task"
        assert result.ok
        step_names = [s.step for s in result.steps]
        assert step_names == [
            "1_perceive", "2_feel", "3_attend",
            "4_own", "5_predict", "5b_cascade", "6_monitor",
        ]

    def test_context_contains_scene_and_prediction(self):
        loop = _make_loop()
        result = loop.pre_task(
            agent_role="researcher",
            task_description="x",
            operation_type="task_execute",
            input_items=[
                WorkspaceItem(item_id="w1", content="x", salience_score=0.7),
            ],
        )
        ctx = result.context_for_agent
        assert ctx["loop_type"] == "full"
        assert "scene_summary" in ctx
        assert "prediction" in ctx
        assert ctx["prediction"]["confidence"] == 0.75

    def test_dispatch_verdict_surfaces_in_context(self):
        loop = _make_loop()
        result = loop.pre_task(
            agent_role="coder",
            task_description="x",
            operation_type="task_execute",
        )
        assert result.context_for_agent["dispatch"]["verdict"] == "ALLOW"

    def test_suspended_belief_surfaces_as_block(self):
        """Wire a consult_fn that returns an ACTIVE plus we simulate a
        suspended candidate by injecting a custom dispatch_decider.
        """
        def custom_decider(consulted_beliefs, suspended_candidates,
                            task_description, crew_name):
            from app.subia.belief.dispatch_gate import DispatchDecision
            return DispatchDecision(
                verdict="BLOCK",
                reason="synthetic block",
                blocking_belief_ids=["sX"],
            )
        loop = _make_loop(dispatch_decider=custom_decider)
        result = loop.pre_task(
            agent_role="coder",
            task_description="run deprecated migration",
            operation_type="task_execute",
        )
        assert result.context_for_agent["dispatch"]["verdict"] == "BLOCK"


# ── Compressed pre_task ──────────────────────────────────────────

class TestCompressedPreTask:
    def test_skips_steps_4_through_6(self):
        loop = _make_loop()
        result = loop.pre_task(
            agent_role="any",
            task_description="look up",
            operation_type="wiki_read",
            input_items=[
                WorkspaceItem(item_id="w1", content="x", salience_score=0.5),
            ],
        )
        step_names = [s.step for s in result.steps]
        assert step_names == ["1_perceive", "2_feel", "3_attend"]
        assert result.loop_type == "compressed"
        assert result.context_for_agent["loop_type"] == "compressed"

    def test_compressed_budget_stricter(self):
        loop = _make_loop()
        result = loop.pre_task(
            agent_role="any",
            task_description="x",
            operation_type="wiki_read",
        )
        # Compressed budget is 100ms; full is 1200ms
        assert result.budget_ms == 100


# ── Post-task ────────────────────────────────────────────────────

class TestPostTask:
    def test_full_post_task_runs_all_steps(self):
        loop = _make_loop()
        result = loop.post_task(
            agent_role="researcher",
            task_description="x",
            operation_type="task_execute",
            task_result={"success": True, "summary": "done"},
        )
        step_names = [s.step for s in result.steps]
        assert step_names == [
            "8_compare", "9_update", "10_consolidate", "11_reflect",
        ]
        assert result.phase == "post_task"

    def test_compressed_post_task_skips_consolidate_reflect(self):
        loop = _make_loop()
        result = loop.post_task(
            agent_role="any",
            task_description="x",
            operation_type="wiki_read",
            task_result={"summary": "quick"},
        )
        step_names = [s.step for s in result.steps]
        assert step_names == ["8_compare", "9_update"]

    def test_post_task_records_agency(self):
        loop = _make_loop()
        assert loop.kernel.self_state.agency_log == []
        loop.post_task(
            agent_role="coder",
            task_description="y",
            operation_type="task_execute",
            task_result={"success": True, "summary": "shipped feature"},
        )
        log = loop.kernel.self_state.agency_log
        assert len(log) == 1
        assert log[0]["summary"] == "shipped feature"
        assert log[0]["success"] is True

    def test_post_task_increments_loop_count(self):
        loop = _make_loop()
        start_count = loop.kernel.loop_count
        loop.post_task(
            agent_role="any", task_description="x",
            operation_type="task_execute",
            task_result={"summary": "s"},
        )
        assert loop.kernel.loop_count == start_count + 1


# ── Failure containment ──────────────────────────────────────────

class TestFailureContainment:
    def test_predict_fn_crash_does_not_break_loop(self):
        def crashing(_ctx):
            raise RuntimeError("llm outage")
        loop = _make_loop(predict_fn=crashing)
        result = loop.pre_task(
            agent_role="any",
            task_description="x",
            operation_type="task_execute",
        )
        assert not result.ok
        predict_outcome = result.step("5_predict")
        assert predict_outcome is not None
        assert not predict_outcome.ok
        assert "llm outage" in (predict_outcome.error or "")
        # 5b and 6 must still have run (failure contained per step)
        assert any(s.step == "5b_cascade" for s in result.steps)
        assert any(s.step == "6_monitor" for s in result.steps)

    def test_consult_fn_crash_does_not_break_loop(self):
        def crash_consult(**_kw):
            raise RuntimeError("belief store down")
        loop = _make_loop(consult_fn=crash_consult)
        result = loop.pre_task(
            agent_role="any",
            task_description="x",
            operation_type="task_execute",
        )
        # 6 must have run and the step should be OK (the crash was
        # caught inside _step_monitor, which proceeds to the dispatch
        # gate with beliefs=[]).
        monitor = result.step("6_monitor")
        assert monitor is not None
        assert monitor.ok

    def test_no_gate_attached_yields_graceful_details(self):
        loop = _make_loop(scene_gate=None)
        result = loop.pre_task(
            agent_role="any",
            task_description="x",
            operation_type="task_execute",
            input_items=[WorkspaceItem(item_id="a")],
        )
        attend = result.step("3_attend")
        assert attend is not None
        assert attend.ok
        assert attend.details == {"gate": "not_attached"}


# ── Cascade modulation ──────────────────────────────────────────

class TestCascadeModulation:
    def test_high_confidence_maintains_tier(self):
        def high_conf(_ctx):
            return Prediction(id="p", operation="o",
                              predicted_outcome={}, predicted_self_change={},
                              predicted_homeostatic_effect={},
                              confidence=0.9, created_at="")
        loop = _make_loop(predict_fn=high_conf)
        result = loop.pre_task(
            agent_role="a", task_description="x",
            operation_type="task_execute",
        )
        assert result.context_for_agent["cascade_recommendation"] == "maintain"

    def test_low_confidence_escalates(self):
        def low_conf(_ctx):
            return Prediction(id="p", operation="o",
                              predicted_outcome={}, predicted_self_change={},
                              predicted_homeostatic_effect={},
                              confidence=0.3, created_at="")
        loop = _make_loop(predict_fn=low_conf)
        result = loop.pre_task(
            agent_role="a", task_description="x",
            operation_type="task_execute",
        )
        assert result.context_for_agent["cascade_recommendation"] == "escalate"

    def test_very_low_confidence_escalates_premium(self):
        def tiny_conf(_ctx):
            return Prediction(id="p", operation="o",
                              predicted_outcome={}, predicted_self_change={},
                              predicted_homeostatic_effect={},
                              confidence=0.05, created_at="")
        loop = _make_loop(predict_fn=tiny_conf)
        result = loop.pre_task(
            agent_role="a", task_description="x",
            operation_type="task_execute",
        )
        assert result.context_for_agent["cascade_recommendation"] == "escalate_premium"


# ── PP-1 end-to-end through post_task ────────────────────────────

class TestPP1Integration:
    def test_predictive_layer_attached_and_gate_set(self):
        """Constructing a loop with both a gate and a predictive_layer
        must call set_gate() so PP-1 surprise routing fires.
        """
        from app.subia.prediction.layer import PredictiveLayer
        gate = CompetitiveGate(capacity=5)
        layer = PredictiveLayer(surprise_budget_per_cycle=2)
        loop = SubIALoop(
            kernel=SubjectivityKernel(),
            scene_gate=gate,
            predictive_layer=layer,
        )
        # set_gate stores the gate on the layer for auto-routing.
        assert layer._gate is gate

    def test_missing_predictive_layer_surfaces_gracefully(self):
        loop = _make_loop(predictive_layer=None)
        result = loop.post_task(
            agent_role="a", task_description="x",
            operation_type="task_execute",
            task_result={"summary": "s"},
        )
        compare = result.step("8_compare")
        assert compare is not None
        assert compare.ok
        assert compare.details == {"predictive_layer": "not_attached"}


# ── Performance / budget tracking ────────────────────────────────

class TestBudget:
    def test_within_budget_for_fast_loop(self):
        loop = _make_loop()
        result = loop.pre_task(
            agent_role="a", task_description="x",
            operation_type="task_execute",
        )
        assert result.within_budget
        assert result.total_elapsed_ms >= 0

    def test_out_of_budget_when_clock_exceeds(self):
        """Use an injected clock that jumps past the budget.

        Each step calls now() twice (t0 at start, t1 at end), plus
        the loop records t_start and t_end. With 7 steps that's
        (2×7)+1 = 15 calls. Use a counter that increments by 1s so
        the elapsed total easily exceeds the 1200ms full-loop budget.
        """
        counter = {"n": 0}

        def fake_now():
            counter["n"] += 1
            return float(counter["n"])  # each call advances 1 second

        loop = _make_loop(now=fake_now)
        result = loop.pre_task(
            agent_role="a", task_description="x",
            operation_type="task_execute",
        )
        # Elapsed in ms should be in the thousands → well past 1200 budget
        assert result.total_elapsed_ms > 1200
        assert not result.within_budget


# ── Serialization ────────────────────────────────────────────────

class TestResultSerialization:
    def test_result_to_dict_round_trips(self):
        loop = _make_loop()
        result = loop.pre_task(
            agent_role="a", task_description="x",
            operation_type="task_execute",
        )
        payload = result.to_dict()
        assert payload["loop_type"] == "full"
        assert payload["phase"] == "pre_task"
        assert isinstance(payload["steps"], list)
        assert all("step" in s and "ok" in s for s in payload["steps"])

    def test_step_outcome_fields(self):
        outcome = StepOutcome(step="x", ok=True, elapsed_ms=1.2)
        assert outcome.step == "x"
        assert outcome.ok
        assert outcome.elapsed_ms == 1.2
        assert outcome.error is None
