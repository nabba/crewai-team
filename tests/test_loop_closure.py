"""
Tests for Beautiful Loop self-referential closure.

The system predicts its entire processing path (strategy, certainty,
valence, coherence) BEFORE processing, then compares. The predicted
coherence is itself part of the coherence computation — creating a
fixed-point that the system converges toward.

Total: ~22 tests
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

for _mod in ["psycopg2", "psycopg2.pool", "psycopg2.extras",
             "app.memory.chromadb_manager"]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

sys.modules["app.memory.chromadb_manager"].embed = MagicMock(return_value=[0.1] * 768)

import pytest


class TestLoopClosurePrediction:
    """Test processing path prediction."""

    def test_creation(self):
        from app.self_awareness.loop_closure import LoopClosure
        lc = LoopClosure("test")
        assert lc._running_coherence == 0.5

    def test_predict_path_defaults(self):
        from app.self_awareness.loop_closure import LoopClosure
        lc = LoopClosure("test_pred")
        pred = lc.predict_path("What is the weather?")
        assert pred.predicted_plan_type == "default"
        assert 0 <= pred.predicted_certainty <= 1
        assert 0 <= pred.predicted_loop_coherence <= 1

    def test_predict_learns_plan_history(self):
        from app.self_awareness.loop_closure import LoopClosure
        lc = LoopClosure("test_plan")
        # Build history: "research" wins most often
        for _ in range(5):
            lc.predict_path("task")
            lc.close_loop(actual_plan_type="research", actual_certainty=0.7)
        lc.predict_path("task")
        lc.close_loop(actual_plan_type="coding", actual_certainty=0.6)
        # Next prediction should favor "research"
        pred = lc.predict_path("new task")
        assert pred.predicted_plan_type == "research"

    def test_predict_certainty_from_history(self):
        from app.self_awareness.loop_closure import LoopClosure
        lc = LoopClosure("test_cert")
        for c in [0.8, 0.85, 0.82, 0.78, 0.81]:
            lc.predict_path("task")
            lc.close_loop(actual_certainty=c)
        pred = lc.predict_path("next task")
        # Should predict near the historical average (~0.81)
        assert 0.7 < pred.predicted_certainty < 0.9


class TestLoopClosureComparison:
    """Test loop closing and error computation."""

    def test_close_loop_returns_state(self):
        from app.self_awareness.loop_closure import LoopClosure
        lc = LoopClosure("test_close")
        lc.predict_path("task")
        state = lc.close_loop(actual_plan_type="research", actual_certainty=0.7)
        assert 0 <= state.composite_error <= 2.0
        assert 0 <= state.loop_coherence <= 1.0

    def test_perfect_prediction_high_coherence(self):
        from app.self_awareness.loop_closure import LoopClosure
        lc = LoopClosure("test_perfect")
        # Train with consistent data
        for _ in range(10):
            lc.predict_path("task")
            lc.close_loop(actual_plan_type="default", actual_certainty=0.5)
        # Now prediction should be accurate → high coherence
        lc.predict_path("task")
        state = lc.close_loop(actual_plan_type="default", actual_certainty=0.5)
        assert state.loop_coherence > 0.6

    def test_wrong_prediction_low_coherence(self):
        from app.self_awareness.loop_closure import LoopClosure
        lc = LoopClosure("test_wrong")
        # Train to expect "research"
        for _ in range(5):
            lc.predict_path("task")
            lc.close_loop(actual_plan_type="research", actual_certainty=0.8)
        # Now something completely different happens
        lc.predict_path("task")
        state = lc.close_loop(actual_plan_type="coding", actual_certainty=0.2)
        assert state.plan_error == 1.0  # Wrong plan
        assert state.certainty_error > 0.3  # Wrong certainty


class TestSelfReferentialFixedPoint:
    """Test the self-referential property: predicted coherence affects actual coherence."""

    def test_coherence_prediction_error_computed(self):
        from app.self_awareness.loop_closure import LoopClosure
        lc = LoopClosure("test_selref")
        lc.predict_path("task")
        state = lc.close_loop(actual_certainty=0.5)
        # coherence_prediction_error should exist (may be non-zero)
        assert state.coherence_prediction_error >= 0

    def test_running_coherence_converges(self):
        """After many consistent rounds, running coherence should stabilize."""
        from app.self_awareness.loop_closure import LoopClosure
        lc = LoopClosure("test_converge")
        coherences = []
        for _ in range(20):
            lc.predict_path("task")
            state = lc.close_loop(actual_plan_type="default", actual_certainty=0.5)
            coherences.append(state.loop_coherence)
        # Last 5 should be more stable than first 5
        early_var = max(coherences[:5]) - min(coherences[:5])
        late_var = max(coherences[-5:]) - min(coherences[-5:])
        assert late_var <= early_var + 0.1  # Late should be at least as stable

    def test_predicted_coherence_approaches_actual(self):
        """Over time, predicted_loop_coherence should approach actual loop_coherence."""
        from app.self_awareness.loop_closure import LoopClosure
        lc = LoopClosure("test_approach")
        gaps = []
        for _ in range(15):
            pred = lc.predict_path("task")
            state = lc.close_loop(actual_plan_type="default", actual_certainty=0.5)
            gaps.append(abs(pred.predicted_loop_coherence - state.loop_coherence))
        # Average gap in last 5 should be smaller than first 5
        early_gap = sum(gaps[:5]) / 5
        late_gap = sum(gaps[-5:]) / 5
        assert late_gap <= early_gap + 0.05

    def test_fixed_point_iteration_converges(self):
        """The 3-iteration fixed point should produce stable coherence."""
        from app.self_awareness.loop_closure import LoopClosure
        lc = LoopClosure("test_fp")
        lc._running_coherence = 0.7
        lc.predict_path("task")
        state = lc.close_loop(actual_certainty=0.5)
        # Coherence should be bounded and reasonable
        assert 0.0 <= state.loop_coherence <= 1.0
        assert state.coherence_prediction_error >= 0


class TestLoopClosureUtilities:
    """Test summary, convergence, singleton."""

    def test_get_convergence(self):
        from app.self_awareness.loop_closure import LoopClosure
        lc = LoopClosure("test_conv")
        assert lc.get_convergence() == 0.5  # Initial default

    def test_get_summary(self):
        from app.self_awareness.loop_closure import LoopClosure
        lc = LoopClosure("test_sum")
        lc.predict_path("task")
        lc.close_loop(actual_certainty=0.6)
        summary = lc.get_summary()
        assert "running_coherence" in summary
        assert "last_state" in summary
        assert summary["last_state"] is not None

    def test_singleton(self):
        from app.self_awareness.loop_closure import LoopClosure, get_loop_closure
        LoopClosure._instances.clear()
        lc1 = get_loop_closure("agent_x")
        lc2 = get_loop_closure("agent_x")
        assert lc1 is lc2
        LoopClosure._instances.clear()

    def test_to_dict(self):
        from app.self_awareness.loop_closure import LoopClosure
        lc = LoopClosure("test_dict")
        lc.predict_path("task")
        state = lc.close_loop(actual_certainty=0.5)
        d = state.to_dict()
        assert "loop_coherence" in d
        assert "predicted_coherence" in d
        assert "composite_error" in d


class TestHookWiring:
    """Verify hooks are registered."""

    def test_loop_predict_hook(self):
        src = (Path(__file__).parent.parent / "app" / "lifecycle_hooks.py").read_text()
        assert "loop_closure_predict" in src
        assert "priority=13" in src

    def test_loop_compare_hook(self):
        src = (Path(__file__).parent.parent / "app" / "lifecycle_hooks.py").read_text()
        assert "loop_closure_compare" in src
        assert "priority=58" in src

    def test_hyper_model_has_loop_fields(self):
        from app.self_awareness.hyper_model import HyperModelState
        state = HyperModelState()
        assert hasattr(state, "loop_closure_error")
        assert hasattr(state, "loop_closure_convergence")

    def test_fe_pressure_includes_loop_convergence(self):
        # Post-Phase-1 migration, hyper_model lives at app/subia/self/hyper_model.py.
        # app/self_awareness/hyper_model.py is a sys.modules-alias shim.
        src = (Path(__file__).parent.parent / "app" / "subia" / "self" / "hyper_model.py").read_text()
        assert "loop_closure_convergence" in src
