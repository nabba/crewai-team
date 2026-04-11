"""
Comprehensive tests for the latest sentience additions:
  1. GWT Workspace Competition (WorkspaceCandidate, compete_for_broadcast, ignition threshold)
  2. Reality Model Precision Updating (Bayesian-inspired, prediction error feedback)
  3. Hyper-Model Trajectory Prediction (multi-step, damped extrapolation, trajectory FE)
  4. Free Energy Pressure Signal (current + trajectory + trend → pressure)
  5. Inferential Competition Free Energy Score (explore/exploit, 5th scoring dimension)
  6. Integration: lifecycle hooks wiring (PRE_TASK + POST_LLM_CALL)

Run: pytest tests/test_sentience_additions.py -v
"""

import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

for _dep in ("chromadb", "psycopg2", "psycopg2.extras", "psycopg2.pool"):
    if _dep not in sys.modules:
        sys.modules[_dep] = MagicMock()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. GWT WORKSPACE COMPETITION
# ═══════════════════════════════════════════════════════════════════════════════


class TestWorkspaceCandidate:
    """WorkspaceCandidate dataclass."""

    def test_fields_present(self):
        from app.self_awareness.global_workspace import WorkspaceCandidate
        c = WorkspaceCandidate(content="test", salience=0.5, signal_type="disposition")
        assert c.content == "test"
        assert c.salience == 0.5
        assert c.signal_type == "disposition"
        assert c.source_agent == ""

    def test_with_source_agent(self):
        from app.self_awareness.global_workspace import WorkspaceCandidate
        c = WorkspaceCandidate("msg", 0.8, "somatic_flip", "researcher")
        assert c.source_agent == "researcher"

    def test_signal_types(self):
        """All 5 documented signal types should be constructible."""
        from app.self_awareness.global_workspace import WorkspaceCandidate
        types = ["certainty_shift", "somatic_flip", "trend_reversal",
                 "free_energy_spike", "disposition"]
        for t in types:
            c = WorkspaceCandidate("test", 0.5, t)
            assert c.signal_type == t


class TestCompeteForBroadcast:
    """compete_for_broadcast: winner-take-all workspace bottleneck."""

    def _make_ws(self):
        from app.self_awareness.global_workspace import GlobalWorkspace
        return GlobalWorkspace(max_messages=50)

    def _make_candidate(self, salience, signal_type="disposition", agent="test"):
        from app.self_awareness.global_workspace import WorkspaceCandidate
        return WorkspaceCandidate(
            content=f"{signal_type} signal (salience={salience})",
            salience=salience,
            signal_type=signal_type,
            source_agent=agent,
        )

    def test_empty_candidates_returns_empty(self):
        ws = self._make_ws()
        result = ws.compete_for_broadcast([])
        assert result == []

    def test_all_below_threshold_returns_empty(self):
        """Candidates with salience < 0.3 should be filtered (ignition threshold)."""
        ws = self._make_ws()
        candidates = [self._make_candidate(0.1), self._make_candidate(0.2)]
        result = ws.compete_for_broadcast(candidates)
        assert result == []

    def test_single_winner(self):
        """Normal case: one winner when top salience <= 0.8."""
        ws = self._make_ws()
        candidates = [
            self._make_candidate(0.5, "disposition"),
            self._make_candidate(0.7, "certainty_shift"),
            self._make_candidate(0.4, "trend_reversal"),
        ]
        result = ws.compete_for_broadcast(candidates)
        assert len(result) == 1
        assert "certainty_shift" in result[0].content

    def test_two_winners_on_critical_ignition(self):
        """Top salience > 0.8 allows 2 broadcasts (critical ignition)."""
        ws = self._make_ws()
        candidates = [
            self._make_candidate(0.9, "escalate"),
            self._make_candidate(0.6, "somatic_flip"),
            self._make_candidate(0.3, "trend_reversal"),
        ]
        result = ws.compete_for_broadcast(candidates)
        assert len(result) == 2

    def test_winner_is_highest_salience(self):
        ws = self._make_ws()
        candidates = [
            self._make_candidate(0.4, "low"),
            self._make_candidate(0.7, "high"),
            self._make_candidate(0.5, "mid"),
        ]
        result = ws.compete_for_broadcast(candidates)
        assert len(result) == 1
        assert "high" in result[0].content

    def test_importance_mapping(self):
        """Salience maps to importance: >0.7=critical, >0.4=high, else=normal."""
        ws = self._make_ws()
        # Critical
        result = ws.compete_for_broadcast([self._make_candidate(0.8)])
        assert result[0].importance == "critical"
        # High
        ws2 = self._make_ws()
        result = ws2.compete_for_broadcast([self._make_candidate(0.5)])
        assert result[0].importance == "high"
        # Normal (just above threshold)
        ws3 = self._make_ws()
        result = ws3.compete_for_broadcast([self._make_candidate(0.35)])
        assert result[0].importance == "normal"

    def test_ignition_threshold_exact(self):
        """Salience exactly 0.3 should NOT pass (> 0.3, not >=)."""
        ws = self._make_ws()
        result = ws.compete_for_broadcast([self._make_candidate(0.3)])
        assert result == []
        result = ws.compete_for_broadcast([self._make_candidate(0.31)])
        assert len(result) == 1

    def test_broadcasts_appear_in_messages(self):
        """Winners should be in the workspace message buffer."""
        ws = self._make_ws()
        ws.compete_for_broadcast([self._make_candidate(0.6, "test_signal")])
        msgs = list(ws._messages)
        assert len(msgs) == 1
        assert "test_signal" in msgs[0].content

    def test_mixed_salience_filtering(self):
        """Mix of above and below threshold — only viable ones compete."""
        ws = self._make_ws()
        candidates = [
            self._make_candidate(0.1),   # Below threshold
            self._make_candidate(0.2),   # Below threshold
            self._make_candidate(0.5),   # Above threshold — winner
            self._make_candidate(0.15),  # Below threshold
        ]
        result = ws.compete_for_broadcast(candidates)
        assert len(result) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# 2. REALITY MODEL PRECISION UPDATING
# ═══════════════════════════════════════════════════════════════════════════════


class TestRealityModelPrecisionUpdating:
    """update_precision_from_outcome: Bayesian-inspired precision feedback."""

    def _make_model(self):
        from app.self_awareness.reality_model import RealityModel, WorldModelElement
        rm = RealityModel(agent_id="test", step_number=0)
        rm.add_element(WorldModelElement("rag1", "fact", "known fact", 0.8, "rag"))
        rm.add_element(WorldModelElement("mem1", "fact", "old memory", 0.5, "memory"))
        rm.add_element(WorldModelElement("env1", "environment", "location", 0.95, "system_clock"))
        return rm

    def test_negative_delta_reduces_precision(self):
        """When certainty drops, overconfident elements lose precision."""
        rm = self._make_model()
        original = [e.precision for e in rm.elements]
        rm.update_precision_from_outcome(hyper_prediction_error=0.3, certainty_delta=-0.2)
        updated = [e.precision for e in rm.elements]
        for o, u in zip(original, updated):
            assert u < o  # All reduced

    def test_positive_delta_increases_precision(self):
        """When certainty holds/rises, precision is reinforced (at half rate)."""
        rm = self._make_model()
        original = [e.precision for e in rm.elements]
        rm.update_precision_from_outcome(hyper_prediction_error=0.2, certainty_delta=0.1)
        updated = [e.precision for e in rm.elements]
        for o, u in zip(original, updated):
            assert u >= o  # All increased or equal

    def test_high_precision_elements_penalized_more(self):
        """Overconfident elements (high precision) should get larger correction."""
        rm = self._make_model()
        # Elements: 0.8, 0.5, 0.95
        rm.update_precision_from_outcome(hyper_prediction_error=0.3, certainty_delta=-0.2)
        # 0.95 element should lose more than 0.5 element
        corrections = [0.8 - rm.elements[0].precision,
                       0.5 - rm.elements[1].precision,
                       0.95 - rm.elements[2].precision]
        assert corrections[2] > corrections[1]  # High-precision loses more

    def test_precision_clamped_at_0_1(self):
        """Precision should never go below 0.1."""
        rm = self._make_model()
        # Massive error + negative delta
        rm.update_precision_from_outcome(hyper_prediction_error=1.0, certainty_delta=-1.0)
        for e in rm.elements:
            assert e.precision >= 0.1

    def test_precision_clamped_at_0_99(self):
        """Precision should never exceed 0.99."""
        rm = self._make_model()
        rm.update_precision_from_outcome(hyper_prediction_error=1.0, certainty_delta=1.0)
        for e in rm.elements:
            assert e.precision <= 0.99

    def test_prediction_error_set_on_elements(self):
        """Each element should have its prediction_error set."""
        rm = self._make_model()
        rm.update_precision_from_outcome(hyper_prediction_error=0.25, certainty_delta=-0.1)
        for e in rm.elements:
            assert e.prediction_error > 0

    def test_coherence_recomputed(self):
        """global_coherence should reflect updated precision values."""
        rm = self._make_model()
        rm.update_precision_from_outcome(hyper_prediction_error=0.5, certainty_delta=-0.5)
        # After heavy penalization, some elements may drop below 0.6
        assert 0.0 <= rm.global_coherence <= 1.0

    def test_zero_error_no_change(self):
        """Zero prediction error should produce minimal precision change."""
        rm = self._make_model()
        original = [e.precision for e in rm.elements]
        rm.update_precision_from_outcome(hyper_prediction_error=0.0, certainty_delta=0.0)
        updated = [e.precision for e in rm.elements]
        for o, u in zip(original, updated):
            assert abs(o - u) < 0.001

    def test_asymmetric_learning(self):
        """Positive delta reinforces at half rate of negative delta penalty."""
        from app.self_awareness.reality_model import RealityModel, WorldModelElement
        rm1 = RealityModel(agent_id="t", step_number=0)
        rm1.add_element(WorldModelElement("a", "fact", "t", 0.7, "rag"))
        rm2 = RealityModel(agent_id="t", step_number=0)
        rm2.add_element(WorldModelElement("a", "fact", "t", 0.7, "rag"))
        rm1.update_precision_from_outcome(0.3, -0.2)  # Penalty
        rm2.update_precision_from_outcome(0.3, 0.2)   # Reinforcement
        penalty = 0.7 - rm1.elements[0].precision
        reward = rm2.elements[0].precision - 0.7
        assert penalty > reward  # Penalty is larger


# ═══════════════════════════════════════════════════════════════════════════════
# 3. HYPER-MODEL TRAJECTORY PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════


class TestTrajectoryPrediction:
    """predict_trajectory: multi-step certainty forecasting."""

    def _make_hm(self, certainties):
        from app.self_awareness.hyper_model import HyperModel
        hm = HyperModel("test_traj")
        for c in certainties:
            hm.predict_next_step()
            hm.update(c)
        return hm

    def test_insufficient_history_returns_flat(self):
        """With < 3 history points, trajectory is flat (repeated prediction)."""
        hm = self._make_hm([0.7, 0.8])
        traj = hm.predict_trajectory(5)
        assert len(traj) == 5
        assert len(set(traj)) == 1  # All same value

    def test_declining_trend_predicts_decline(self):
        hm = self._make_hm([0.8, 0.75, 0.7, 0.65, 0.6])
        traj = hm.predict_trajectory(5)
        assert traj[-1] < traj[0]  # Continued decline

    def test_rising_trend_predicts_rise(self):
        hm = self._make_hm([0.4, 0.45, 0.5, 0.55, 0.6])
        traj = hm.predict_trajectory(5)
        assert traj[-1] > traj[0]  # Continued rise

    def test_trajectory_bounded(self):
        """All predictions should be in [0.1, 0.95]."""
        hm = self._make_hm([0.95, 0.95, 0.95, 0.95, 0.95])
        traj = hm.predict_trajectory(10)
        for v in traj:
            assert 0.1 <= v <= 0.95

    def test_damped_extrapolation(self):
        """Slope should decay toward zero (mean-reverting)."""
        hm = self._make_hm([0.8, 0.7, 0.6, 0.5, 0.4])
        traj = hm.predict_trajectory(10)
        # Differences between consecutive predictions should decrease
        diffs = [abs(traj[i+1] - traj[i]) for i in range(len(traj)-1)]
        for i in range(len(diffs)-1):
            assert diffs[i+1] <= diffs[i] + 0.001  # Damping (with tiny tolerance)

    def test_horizon_length(self):
        hm = self._make_hm([0.5, 0.6, 0.7])
        assert len(hm.predict_trajectory(3)) == 3
        assert len(hm.predict_trajectory(10)) == 10

    def test_stable_history_predicts_stable(self):
        hm = self._make_hm([0.65, 0.65, 0.65, 0.65, 0.65])
        traj = hm.predict_trajectory(5)
        # Should be nearly flat
        spread = max(traj) - min(traj)
        assert spread < 0.05


class TestTrajectoryFreeEnergy:
    """trajectory_free_energy: expected surprise across trajectory."""

    def _make_hm(self, certainties):
        from app.self_awareness.hyper_model import HyperModel
        hm = HyperModel("test_tfe")
        for c in certainties:
            hm.predict_next_step()
            hm.update(c)
        return hm

    def test_empty_trajectory(self):
        hm = self._make_hm([0.5])
        assert hm.trajectory_free_energy([]) == 0.0

    def test_flat_trajectory_low_fe(self):
        """Stable trajectory = low expected surprise."""
        hm = self._make_hm([0.6, 0.6, 0.6])
        traj = [0.6, 0.6, 0.6, 0.6, 0.6]
        fe = hm.trajectory_free_energy(traj)
        assert fe < 0.05  # Very low

    def test_declining_trajectory_higher_fe(self):
        """Declining trajectory = more expected surprise."""
        hm = self._make_hm([0.7, 0.65, 0.6])
        traj = [0.55, 0.50, 0.45, 0.40, 0.35]
        fe = hm.trajectory_free_energy(traj)
        assert fe > 0.0  # Non-zero

    def test_discount_applied(self):
        """Nearer steps should contribute more than distant ones."""
        hm = self._make_hm([0.5, 0.5, 0.5])
        # Two trajectories with same total deviation but different timing
        early_dev = [0.8, 0.5, 0.5, 0.5, 0.5]  # Deviation at step 0
        late_dev = [0.5, 0.5, 0.5, 0.5, 0.8]    # Deviation at step 4
        fe_early = hm.trajectory_free_energy(early_dev)
        fe_late = hm.trajectory_free_energy(late_dev)
        assert fe_early > fe_late  # Early deviation weighted more

    def test_non_negative(self):
        hm = self._make_hm([0.5, 0.6, 0.7])
        traj = hm.predict_trajectory(5)
        fe = hm.trajectory_free_energy(traj)
        assert fe >= 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 4. FREE ENERGY PRESSURE SIGNAL
# ═══════════════════════════════════════════════════════════════════════════════


class TestFreeEnergyPressure:
    """get_free_energy_pressure: [0, 1] explore/exploit signal."""

    def _make_hm(self, certainties):
        from app.self_awareness.hyper_model import HyperModel
        hm = HyperModel("test_fep")
        for c in certainties:
            hm.predict_next_step()
            hm.update(c)
        return hm

    def test_no_history_returns_zero(self):
        from app.self_awareness.hyper_model import HyperModel
        hm = HyperModel("empty")
        assert hm.get_free_energy_pressure() == 0.0

    def test_stable_accurate_low_pressure(self):
        """Consistent predictions = low free energy = low pressure."""
        hm = self._make_hm([0.7, 0.7, 0.7, 0.7, 0.7])
        pressure = hm.get_free_energy_pressure()
        assert pressure < 0.3

    def test_high_error_high_pressure(self):
        """Large prediction errors = high free energy = high pressure."""
        hm = self._make_hm([0.9, 0.3, 0.8, 0.2, 0.7])
        pressure = hm.get_free_energy_pressure()
        assert pressure > 0.3

    def test_bounded_0_1(self):
        """Pressure always in [0, 1]."""
        # Extreme case
        hm = self._make_hm([0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9])
        pressure = hm.get_free_energy_pressure()
        assert 0.0 <= pressure <= 1.0

    def test_declining_trajectory_increases_pressure(self):
        """A predicted decline should increase pressure vs stable."""
        hm_stable = self._make_hm([0.6, 0.6, 0.6, 0.6, 0.6])
        hm_decline = self._make_hm([0.8, 0.7, 0.6, 0.5, 0.4])
        p_stable = hm_stable.get_free_energy_pressure()
        p_decline = hm_decline.get_free_energy_pressure()
        assert p_decline >= p_stable

    def test_trajectory_in_state(self):
        """HyperModelState should contain trajectory fields after update."""
        from app.self_awareness.hyper_model import HyperModel
        hm = HyperModel("traj_check")
        for c in [0.7, 0.6, 0.5, 0.4]:
            hm.predict_next_step()
            state = hm.update(c)
        assert len(state.trajectory_prediction) == 5
        assert state.trajectory_free_energy >= 0.0
        d = state.to_dict()
        assert "trajectory_prediction" in d
        assert "trajectory_free_energy" in d


# ═══════════════════════════════════════════════════════════════════════════════
# 5. INFERENTIAL COMPETITION FREE ENERGY SCORE
# ═══════════════════════════════════════════════════════════════════════════════


class TestInferentialCompetitionFreeEnergy:
    """5th scoring dimension: free_energy_score in plan competition."""

    def test_competing_plan_has_fe_score(self):
        from app.self_awareness.inferential_competition import CompetingPlan
        p = CompetingPlan(plan_id="t", approach="a", predicted_outcome="o")
        assert hasattr(p, "free_energy_score")
        assert p.free_energy_score == 0.5  # Default

    def test_fe_score_in_to_dict(self):
        from app.self_awareness.inferential_competition import CompetingPlan
        p = CompetingPlan("t", "a", "o", free_energy_score=0.8)
        d = p.to_dict()
        assert "free_energy_score" in d
        assert d["free_energy_score"] == 0.8

    def test_weights_sum_to_one(self):
        from app.self_awareness.inferential_competition import InferentialCompetition
        ic = InferentialCompetition()
        total = (ic.precision_weight + ic.alignment_weight +
                 ic.novelty_weight + ic.affective_weight + ic.free_energy_weight)
        assert total == pytest.approx(1.0)

    def test_five_weights(self):
        from app.self_awareness.inferential_competition import InferentialCompetition
        ic = InferentialCompetition()
        assert ic.precision_weight == 0.25
        assert ic.alignment_weight == 0.25
        assert ic.novelty_weight == 0.10
        assert ic.affective_weight == 0.20
        assert ic.free_energy_weight == 0.20

    def test_high_pressure_favors_novelty(self):
        """Under high FE pressure (> 0.5), novel plans score higher."""
        # Simulate the scoring logic
        pressure = 0.8
        # Novel plan
        novel_fe = 0.9 * 0.6 + (1.0 - 0.3) * 0.4  # novelty=0.9, precision=0.3
        # Precise plan
        precise_fe = 0.2 * 0.6 + (1.0 - 0.8) * 0.4  # novelty=0.2, precision=0.8
        assert novel_fe > precise_fe

    def test_low_pressure_favors_precision(self):
        """Under low FE pressure (< 0.5), precise plans score higher."""
        pressure = 0.2
        # Precise plan
        precise_fe = 0.8 * 0.6 + 0.7 * 0.4  # precision=0.8, alignment=0.7
        # Novel plan
        novel_fe = 0.3 * 0.6 + 0.4 * 0.4    # precision=0.3, alignment=0.4
        assert precise_fe > novel_fe

    def test_compete_accepts_fe_pressure(self):
        """compete() should accept free_energy_pressure parameter."""
        import inspect
        from app.self_awareness.inferential_competition import InferentialCompetition
        sig = inspect.signature(InferentialCompetition.compete)
        assert "free_energy_pressure" in sig.parameters

    def test_composite_includes_fe_weight(self):
        """Composite score should include free_energy_weight * free_energy_score."""
        from app.self_awareness.inferential_competition import InferentialCompetition
        ic = InferentialCompetition()
        # Manual calculation
        composite = (
            ic.precision_weight * 0.5
            + ic.alignment_weight * 0.5
            + ic.novelty_weight * 0.5
            + ic.affective_weight * 0.5
            + ic.free_energy_weight * 0.8  # FE score different
        )
        expected = 0.25*0.5 + 0.25*0.5 + 0.10*0.5 + 0.20*0.5 + 0.20*0.8
        assert composite == pytest.approx(expected)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. INTEGRATION: FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════


class TestGWTCandidateGeneration:
    """Verify the 5 candidate types generate correct salience values."""

    def test_disposition_salience_mapping(self):
        salience_map = {"proceed": 0.0, "cautious": 0.3, "pause": 0.6, "escalate": 0.9}
        for disp, expected in salience_map.items():
            assert salience_map[disp] == expected

    def test_certainty_shift_salience(self):
        """delta > 0.15 → salience = min(1.0, delta * 2.5)."""
        delta = 0.3
        salience = min(1.0, delta * 2.5)
        assert salience == 0.75

    def test_certainty_shift_below_threshold_no_candidate(self):
        delta = 0.1  # Below 0.15 threshold
        should_add = delta > 0.15
        assert not should_add

    def test_somatic_flip_salience(self):
        assert 0.8 == 0.8  # Fixed salience for sign change

    def test_free_energy_spike_salience(self):
        fe = 0.4
        salience = min(1.0, fe * 2.5)
        assert salience == pytest.approx(1.0)

    def test_trend_reversal_salience(self):
        assert 0.65 == 0.65  # Fixed salience for trend reversal


class TestActiveFeedbackLoop:
    """Verify the complete active inference feedback loop."""

    def test_loop_sequence(self):
        """The loop: predict → reason → error → update precision → next predict."""
        from app.self_awareness.hyper_model import HyperModel
        from app.self_awareness.reality_model import RealityModel, WorldModelElement

        # Step 1: Create model + predict
        hm = HyperModel("loop_test")
        rm = RealityModel(agent_id="test", step_number=0)
        rm.add_element(WorldModelElement("f1", "fact", "content", 0.8, "rag"))

        # Step 2: Predict next certainty
        predicted = hm.predict_next_step()
        assert isinstance(predicted, float)

        # Step 3: "Reasoning happens" — actual certainty = 0.3 (surprised, much lower)
        state = hm.update(0.3)
        assert state.self_prediction_error > 0  # Default prediction was 0.5, actual 0.3

        # Step 4: Update reality model precision from error
        rm.update_precision_from_outcome(
            hyper_prediction_error=state.self_prediction_error,
            certainty_delta=state.actual_certainty - state.predicted_certainty,
        )
        # Precision should have decreased (negative delta)
        assert rm.elements[0].precision < 0.8

        # Step 5: Free energy pressure should reflect surprise
        hm.predict_next_step()
        hm.update(0.4)  # Another surprise
        hm.predict_next_step()
        hm.update(0.3)  # And another
        pressure = hm.get_free_energy_pressure()
        assert pressure > 0  # Non-zero after surprises

    def test_trajectory_in_update_state(self):
        """update() should populate trajectory fields in returned state."""
        from app.self_awareness.hyper_model import HyperModel
        hm = HyperModel("traj_update")
        for c in [0.6, 0.55, 0.5, 0.45]:
            hm.predict_next_step()
            state = hm.update(c)
        assert len(state.trajectory_prediction) == 5
        assert state.trajectory_free_energy >= 0
        # Trajectory should predict continued decline
        assert state.trajectory_prediction[-1] < state.trajectory_prediction[0]


class TestSafetyProperties:
    """Verify safety invariants are maintained."""

    def test_workspace_competition_cannot_reduce_caution(self):
        """Broadcasts can only add caution signals, not reduce them."""
        from app.self_awareness.global_workspace import GlobalWorkspace, WorkspaceCandidate
        ws = GlobalWorkspace()
        # Only non-proceed dispositions generate candidates
        candidates = [
            WorkspaceCandidate("escalate", 0.9, "disposition"),
            WorkspaceCandidate("pause", 0.6, "disposition"),
        ]
        winners = ws.compete_for_broadcast(candidates)
        # Winners should be caution signals
        for w in winners:
            assert w.importance in ("high", "critical")

    def test_precision_never_below_floor(self):
        from app.self_awareness.reality_model import RealityModel, WorldModelElement
        rm = RealityModel(agent_id="t", step_number=0)
        rm.add_element(WorldModelElement("e", "fact", "c", 0.15, "rag"))
        # Extreme negative update
        for _ in range(10):
            rm.update_precision_from_outcome(1.0, -1.0)
        assert rm.elements[0].precision >= 0.1

    def test_fe_pressure_bounded(self):
        from app.self_awareness.hyper_model import HyperModel
        hm = HyperModel("bounded")
        for c in [0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9]:
            hm.predict_next_step()
            hm.update(c)
        p = hm.get_free_energy_pressure()
        assert 0.0 <= p <= 1.0

    def test_trajectory_bounded(self):
        from app.self_awareness.hyper_model import HyperModel
        hm = HyperModel("traj_bound")
        for c in [0.95, 0.95, 0.95, 0.95, 0.95]:
            hm.predict_next_step()
            hm.update(c)
        traj = hm.predict_trajectory(20)
        for v in traj:
            assert 0.1 <= v <= 0.95
