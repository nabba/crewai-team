"""
Phase 4: SubIA homeostasis engine regression tests.

Verifies:
  - ensure_variables fills missing vars with initial values + default
    set-points
  - New items nudge novelty_balance + overload
  - Conflicting items nudge contradiction_pressure
  - Success nudges progress up; failure nudges it down
  - Deviations and restoration_queue reflect the absolute delta from
    set-points, ordered correctly
  - Clamping to [0,1] applies to every variable
  - Update never raises
"""

from __future__ import annotations

import pytest

from app.subia.config import SUBIA_CONFIG
from app.subia.homeostasis.engine import (
    ensure_variables,
    update_homeostasis,
)
from app.subia.kernel import SceneItem, SubjectivityKernel


def _item(source="wiki", conflicts=None, summary="x") -> SceneItem:
    return SceneItem(
        id=f"i-{id(object())}", source=source,
        content_ref="c", summary=summary,
        salience=0.5, entered_at="",
        conflicts_with=list(conflicts or []),
    )


# ── ensure_variables ────────────────────────────────────────────

class TestEnsureVariables:
    def test_fills_missing_vars(self):
        kernel = SubjectivityKernel()
        ensure_variables(kernel)
        for var in SUBIA_CONFIG["HOMEOSTATIC_VARIABLES"]:
            assert var in kernel.homeostasis.variables
            assert var in kernel.homeostasis.set_points

    def test_preserves_existing_values(self):
        kernel = SubjectivityKernel()
        kernel.homeostasis.variables["coherence"] = 0.9
        kernel.homeostasis.set_points["coherence"] = 0.6
        ensure_variables(kernel)
        assert kernel.homeostasis.variables["coherence"] == 0.9
        assert kernel.homeostasis.set_points["coherence"] == 0.6


# ── Item-driven deltas (pre-task) ───────────────────────────────

class TestItemDrivenDeltas:
    def test_new_items_raise_novelty_balance(self):
        kernel = SubjectivityKernel()
        ensure_variables(kernel)
        before = kernel.homeostasis.variables["novelty_balance"]
        update_homeostasis(
            kernel,
            new_items=[_item(source="wiki"), _item(source="firecrawl")],
        )
        assert kernel.homeostasis.variables["novelty_balance"] > before

    def test_conflicts_raise_contradiction_pressure(self):
        kernel = SubjectivityKernel()
        ensure_variables(kernel)
        before = kernel.homeostasis.variables["contradiction_pressure"]
        update_homeostasis(
            kernel,
            new_items=[
                _item(conflicts=["i-A", "i-B"]),
                _item(conflicts=["i-C"]),
            ],
        )
        assert kernel.homeostasis.variables["contradiction_pressure"] > before

    def test_many_items_raise_overload(self):
        kernel = SubjectivityKernel()
        ensure_variables(kernel)
        before = kernel.homeostasis.variables["overload"]
        update_homeostasis(
            kernel,
            new_items=[_item() for _ in range(10)],
        )
        assert kernel.homeostasis.variables["overload"] > before

    def test_idle_tick_regulates_overload_down(self):
        kernel = SubjectivityKernel()
        ensure_variables(kernel)
        kernel.homeostasis.variables["overload"] = 0.8
        update_homeostasis(kernel, new_items=[])
        assert kernel.homeostasis.variables["overload"] < 0.8


# ── Outcome-driven deltas (post-task) ───────────────────────────

class TestOutcomeDrivenDeltas:
    def test_success_raises_progress_and_coherence(self):
        kernel = SubjectivityKernel()
        ensure_variables(kernel)
        p_before = kernel.homeostasis.variables["progress"]
        c_before = kernel.homeostasis.variables["coherence"]
        update_homeostasis(
            kernel,
            task_result={"success": True, "summary": "x"},
        )
        assert kernel.homeostasis.variables["progress"] > p_before
        assert kernel.homeostasis.variables["coherence"] > c_before

    def test_failure_lowers_progress_and_coherence(self):
        kernel = SubjectivityKernel()
        ensure_variables(kernel)
        p_before = kernel.homeostasis.variables["progress"]
        c_before = kernel.homeostasis.variables["coherence"]
        update_homeostasis(
            kernel,
            task_result={"success": False, "summary": "x"},
        )
        assert kernel.homeostasis.variables["progress"] < p_before
        assert kernel.homeostasis.variables["coherence"] < c_before

    def test_new_commitments_raise_commitment_load(self):
        kernel = SubjectivityKernel()
        ensure_variables(kernel)
        before = kernel.homeostasis.variables["commitment_load"]
        update_homeostasis(
            kernel,
            task_result={"success": True, "new_commitment_count": 3},
        )
        assert kernel.homeostasis.variables["commitment_load"] > before


# ── Deviations + restoration_queue ──────────────────────────────

class TestDeviations:
    def test_deviation_is_value_minus_setpoint(self):
        kernel = SubjectivityKernel()
        ensure_variables(kernel)
        # Default set-point is 0.5; force deviations
        kernel.homeostasis.variables["coherence"] = 0.8
        kernel.homeostasis.variables["progress"] = 0.2
        update_homeostasis(kernel, new_items=[])
        assert kernel.homeostasis.deviations["coherence"] == pytest.approx(0.3)
        assert kernel.homeostasis.deviations["progress"] == pytest.approx(-0.3, abs=1e-2)

    def test_restoration_queue_ordered_by_abs_deviation(self):
        kernel = SubjectivityKernel()
        ensure_variables(kernel)
        kernel.homeostasis.variables["coherence"] = 0.95  # +0.45
        kernel.homeostasis.variables["progress"] = 0.15   # -0.35
        kernel.homeostasis.variables["safety"] = 0.55     # +0.05 (below threshold)
        # Use a no-op outcome so outcome-driven deltas don't fire.
        # Idle tick still applies overload regulation but that's
        # currently 0.5 - 0.01 = 0.49, below threshold.
        update_homeostasis(kernel, new_items=[])
        queue = kernel.homeostasis.restoration_queue
        # coherence and progress must be the top two (in that order,
        # because 0.4 > 0.3 in absolute terms).
        assert queue[0] == "coherence"
        assert queue[1] == "progress"
        assert "safety" not in queue

    def test_threshold_filters_restoration_queue(self):
        kernel = SubjectivityKernel()
        ensure_variables(kernel)
        # All variables at set-point: queue should be empty.
        update_homeostasis(kernel, new_items=[])
        assert kernel.homeostasis.restoration_queue == []


# ── Clamping ────────────────────────────────────────────────────

class TestClamping:
    def test_values_clamp_to_unit_interval(self):
        kernel = SubjectivityKernel()
        ensure_variables(kernel)
        kernel.homeostasis.variables["novelty_balance"] = 0.95
        # Try to blow past 1.0 with many new items
        update_homeostasis(
            kernel,
            new_items=[_item(source="wiki") for _ in range(50)],
        )
        assert kernel.homeostasis.variables["novelty_balance"] <= 1.0

    def test_values_never_negative(self):
        kernel = SubjectivityKernel()
        ensure_variables(kernel)
        kernel.homeostasis.variables["progress"] = 0.02
        # Many failures
        for _ in range(20):
            update_homeostasis(
                kernel,
                task_result={"success": False, "summary": "x"},
            )
        assert kernel.homeostasis.variables["progress"] >= 0.0


# ── Safety: never raises ────────────────────────────────────────

class TestSafety:
    def test_malformed_kernel_does_not_crash(self):
        """Broken kernel homeostasis attrs should not propagate."""
        kernel = SubjectivityKernel()
        # Corrupt the homeostasis variables dict on purpose
        kernel.homeostasis.variables = {"not_a_real_var": "string-value"}
        result = update_homeostasis(kernel, new_items=[_item()])
        # Either it recovered OR returned an error dict; must not raise.
        assert isinstance(result, dict)

    def test_none_new_items_safe(self):
        kernel = SubjectivityKernel()
        ensure_variables(kernel)
        # Should treat None-ish iterables safely
        result = update_homeostasis(kernel)
        assert isinstance(result, dict)


# ── Loop integration ────────────────────────────────────────────

class TestLoopIntegration:
    def test_loop_pre_task_populates_homeostasis(self):
        """After a full pre_task run, homeostatic variables exist
        and at least one deviation is reported.
        """
        import sys
        from unittest.mock import MagicMock

        for _mod in ["chromadb", "chromadb.config", "chromadb.utils",
                     "app.memory.chromadb_manager"]:
            if _mod not in sys.modules:
                sys.modules[_mod] = MagicMock()

        from app.subia.kernel import Prediction
        from app.subia.loop import SubIALoop
        from app.subia.scene.buffer import CompetitiveGate, WorkspaceItem

        def predict(ctx):
            return Prediction(
                id="p", operation="o",
                predicted_outcome={}, predicted_self_change={},
                predicted_homeostatic_effect={},
                confidence=0.7, created_at="",
            )

        loop = SubIALoop(
            kernel=SubjectivityKernel(),
            scene_gate=CompetitiveGate(capacity=5),
            predict_fn=predict,
        )
        loop.pre_task(
            agent_role="researcher",
            task_description="ingest truepic news",
            operation_type="task_execute",
            input_items=[WorkspaceItem(item_id="i1", content="x",
                                        salience_score=0.5,
                                        source_channel="wiki")],
        )

        assert loop.kernel.homeostasis.variables
        # After a candidate item, novelty_balance should be above initial
        assert loop.kernel.homeostasis.variables["novelty_balance"] > 0.5

    def test_success_outcome_raises_progress(self):
        import sys
        from unittest.mock import MagicMock
        for _mod in ["chromadb", "chromadb.config", "chromadb.utils",
                     "app.memory.chromadb_manager"]:
            if _mod not in sys.modules:
                sys.modules[_mod] = MagicMock()

        from app.subia.loop import SubIALoop
        from app.subia.scene.buffer import CompetitiveGate

        loop = SubIALoop(
            kernel=SubjectivityKernel(),
            scene_gate=CompetitiveGate(capacity=5),
        )
        ensure_variables(loop.kernel)
        before = loop.kernel.homeostasis.variables["progress"]
        loop.post_task(
            agent_role="any", task_description="x",
            operation_type="task_execute",
            task_result={"success": True, "summary": "shipped"},
        )
        assert loop.kernel.homeostasis.variables["progress"] > before
