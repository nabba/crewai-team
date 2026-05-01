"""
Tests for PDS-driven workspace capacity adaptation.

Personality traits control workspace parameters:
  focus_quality → capacity (narrow vs broad)
  solution_creativity → novelty floor (exploration vs exploitation)
  error_resilience → consumption decay (revisit vs move on)
  developmental stage → capacity bonus
  homeostasis → temporary override

Total: ~25 tests
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

for _mod in ["psycopg2", "psycopg2.pool", "psycopg2.extras",
             "app.control_plane", "app.control_plane.db",
             "app.memory.chromadb_manager"]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

sys.modules["app.memory.chromadb_manager"].embed = MagicMock(return_value=[0.1] * 768)
sys.modules["app.control_plane.db"].execute = MagicMock(return_value=[])

import pytest


def _mock_personality(focus=0.5, creativity=0.5, resilience=0.5,
                      stage="operational_independence"):
    """Create a mock PersonalityState."""
    return SimpleNamespace(
        temperament={"focus_quality": focus, "communication_initiative": 0.5,
                     "error_response_pattern": 0.5, "resource_discipline": 0.5,
                     "team_orientation": 0.5},
        personality_factors={"solution_creativity": creativity,
                            "error_resilience": resilience,
                            "communication_propensity": 0.5,
                            "cooperative_orientation": 0.5,
                            "task_discipline": 0.5},
        developmental_stage=stage,
        strengths={},
    )


class TestWorkspaceProfileMapping:
    """Test personality trait → workspace parameter mapping."""

    @patch("app.personality.state.get_personality")
    @patch("app.self_awareness.homeostasis.get_state", return_value={"frustration": 0.1, "cognitive_energy": 0.7})
    def test_default_personality_gives_default_capacity(self, mock_homeo, mock_pers):
        mock_pers.return_value = _mock_personality()
        from app.consciousness.personality_workspace import compute_workspace_profile
        profile = compute_workspace_profile("commander")
        assert profile.capacity == 5

    @patch("app.personality.state.get_personality")
    @patch("app.self_awareness.homeostasis.get_state", return_value={"frustration": 0.1, "cognitive_energy": 0.7})
    def test_high_focus_narrows_capacity(self, mock_homeo, mock_pers):
        mock_pers.return_value = _mock_personality(focus=0.85)
        from app.consciousness.personality_workspace import compute_workspace_profile
        profile = compute_workspace_profile("researcher")
        assert profile.capacity == 3

    @patch("app.personality.state.get_personality")
    @patch("app.self_awareness.homeostasis.get_state", return_value={"frustration": 0.1, "cognitive_energy": 0.7})
    def test_low_focus_broadens_capacity(self, mock_homeo, mock_pers):
        mock_pers.return_value = _mock_personality(focus=0.25)
        from app.consciousness.personality_workspace import compute_workspace_profile
        profile = compute_workspace_profile("writer")
        assert profile.capacity == 7

    @patch("app.personality.state.get_personality")
    @patch("app.self_awareness.homeostasis.get_state", return_value={"frustration": 0.1, "cognitive_energy": 0.7})
    def test_high_creativity_increases_novelty_floor(self, mock_homeo, mock_pers):
        mock_pers.return_value = _mock_personality(creativity=0.8)
        from app.consciousness.personality_workspace import compute_workspace_profile
        profile = compute_workspace_profile("researcher")
        assert profile.novelty_floor_pct == 0.30

    @patch("app.personality.state.get_personality")
    @patch("app.self_awareness.homeostasis.get_state", return_value={"frustration": 0.1, "cognitive_energy": 0.7})
    def test_low_creativity_decreases_novelty_floor(self, mock_homeo, mock_pers):
        mock_pers.return_value = _mock_personality(creativity=0.2)
        from app.consciousness.personality_workspace import compute_workspace_profile
        profile = compute_workspace_profile("coder")
        assert profile.novelty_floor_pct == 0.10

    @patch("app.personality.state.get_personality")
    @patch("app.self_awareness.homeostasis.get_state", return_value={"frustration": 0.1, "cognitive_energy": 0.7})
    def test_high_resilience_slows_consumption_decay(self, mock_homeo, mock_pers):
        mock_pers.return_value = _mock_personality(resilience=0.8)
        from app.consciousness.personality_workspace import compute_workspace_profile
        profile = compute_workspace_profile("commander")
        assert profile.consumption_decay == 0.30

    @patch("app.personality.state.get_personality")
    @patch("app.self_awareness.homeostasis.get_state", return_value={"frustration": 0.1, "cognitive_energy": 0.7})
    def test_low_resilience_speeds_consumption_decay(self, mock_homeo, mock_pers):
        mock_pers.return_value = _mock_personality(resilience=0.2)
        from app.consciousness.personality_workspace import compute_workspace_profile
        profile = compute_workspace_profile("commander")
        assert profile.consumption_decay == 0.70


class TestDevelopmentalStageBonus:
    """Test developmental stage → capacity adjustment."""

    @patch("app.personality.state.get_personality")
    @patch("app.self_awareness.homeostasis.get_state", return_value={"frustration": 0.1, "cognitive_energy": 0.7})
    def test_stage1_reduces_capacity(self, mock_homeo, mock_pers):
        mock_pers.return_value = _mock_personality(stage="system_trust")
        from app.consciousness.personality_workspace import compute_workspace_profile
        profile = compute_workspace_profile("commander")
        assert profile.capacity == 4  # 5 - 1

    @patch("app.personality.state.get_personality")
    @patch("app.self_awareness.homeostasis.get_state", return_value={"frustration": 0.1, "cognitive_energy": 0.7})
    def test_stage4_increases_capacity(self, mock_homeo, mock_pers):
        mock_pers.return_value = _mock_personality(stage="competence_confidence")
        from app.consciousness.personality_workspace import compute_workspace_profile
        profile = compute_workspace_profile("researcher")
        assert profile.capacity == 6  # 5 + 1

    @patch("app.personality.state.get_personality")
    @patch("app.self_awareness.homeostasis.get_state", return_value={"frustration": 0.1, "cognitive_energy": 0.7})
    def test_stage5_increases_capacity(self, mock_homeo, mock_pers):
        mock_pers.return_value = _mock_personality(stage="role_coherence")
        from app.consciousness.personality_workspace import compute_workspace_profile
        profile = compute_workspace_profile("commander")
        assert profile.capacity == 6  # 5 + 1


class TestHomeostasisOverride:
    """Test homeostasis temporary capacity override."""

    @patch("app.personality.state.get_personality")
    @patch("app.self_awareness.homeostasis.get_state")
    def test_high_frustration_broadens(self, mock_homeo, mock_pers):
        mock_pers.return_value = _mock_personality()
        mock_homeo.return_value = {"frustration": 0.8, "cognitive_energy": 0.7}
        from app.consciousness.personality_workspace import compute_workspace_profile
        profile = compute_workspace_profile("commander")
        assert profile.capacity == 7  # 5 + 2

    @patch("app.personality.state.get_personality")
    @patch("app.self_awareness.homeostasis.get_state")
    def test_low_energy_narrows(self, mock_homeo, mock_pers):
        mock_pers.return_value = _mock_personality()
        mock_homeo.return_value = {"frustration": 0.1, "cognitive_energy": 0.2}
        from app.consciousness.personality_workspace import compute_workspace_profile
        profile = compute_workspace_profile("commander")
        assert profile.capacity == 3  # 5 - 2

    @patch("app.personality.state.get_personality")
    @patch("app.self_awareness.homeostasis.get_state")
    def test_combined_frustration_and_low_energy(self, mock_homeo, mock_pers):
        """High frustration (+2) and low energy (-2) cancel out."""
        mock_pers.return_value = _mock_personality()
        mock_homeo.return_value = {"frustration": 0.8, "cognitive_energy": 0.2}
        from app.consciousness.personality_workspace import compute_workspace_profile
        profile = compute_workspace_profile("commander")
        assert profile.capacity == 5  # +2 -2 = 0


class TestSafetyBounds:
    """Test capacity clamping to [2, 9]."""

    @patch("app.personality.state.get_personality")
    @patch("app.self_awareness.homeostasis.get_state")
    def test_minimum_capacity_bound(self, mock_homeo, mock_pers):
        """Focused personality + early stage + low energy shouldn't go below 2."""
        mock_pers.return_value = _mock_personality(focus=0.9, stage="system_trust")
        mock_homeo.return_value = {"frustration": 0.1, "cognitive_energy": 0.1}
        from app.consciousness.personality_workspace import compute_workspace_profile
        profile = compute_workspace_profile("commander")
        # 3 (focus) - 1 (stage) - 2 (energy) = 0 → clamped to 2
        assert profile.capacity == 2

    @patch("app.personality.state.get_personality")
    @patch("app.self_awareness.homeostasis.get_state")
    def test_maximum_capacity_bound(self, mock_homeo, mock_pers):
        """Broad personality + mature stage + high frustration shouldn't exceed 9."""
        mock_pers.return_value = _mock_personality(focus=0.2, stage="role_coherence")
        mock_homeo.return_value = {"frustration": 0.9, "cognitive_energy": 0.7}
        from app.consciousness.personality_workspace import compute_workspace_profile
        profile = compute_workspace_profile("commander")
        # 7 (focus) + 1 (stage) + 2 (frustration) = 10 → clamped to 9
        assert profile.capacity == 9

    def test_fallback_on_personality_load_failure(self):
        """If personality loading fails, should return defaults."""
        from app.consciousness.personality_workspace import compute_workspace_profile
        # No mock — will fail to import personality module
        with patch("app.personality.state.get_personality", side_effect=ImportError):
            profile = compute_workspace_profile("nonexistent_agent")
            assert profile.capacity == 5
            assert profile.novelty_floor_pct == 0.20
            assert profile.consumption_decay == 0.50


class TestCompetitiveGateDynamicCapacity:
    """Test set_dynamic_capacity on CompetitiveGate."""

    def test_set_dynamic_capacity(self):
        from app.consciousness.workspace_buffer import CompetitiveGate
        gate = CompetitiveGate(capacity=5)
        gate.set_dynamic_capacity(7)
        assert gate.capacity == 7

    def test_dynamic_capacity_clamped_low(self):
        from app.consciousness.workspace_buffer import CompetitiveGate
        gate = CompetitiveGate(capacity=5)
        gate.set_dynamic_capacity(0)
        assert gate.capacity == 2

    def test_dynamic_capacity_clamped_high(self):
        from app.consciousness.workspace_buffer import CompetitiveGate
        gate = CompetitiveGate(capacity=5)
        gate.set_dynamic_capacity(20)
        assert gate.capacity == 9

    def test_dynamic_novelty_floor(self):
        from app.consciousness.workspace_buffer import CompetitiveGate
        gate = CompetitiveGate()
        gate.set_dynamic_capacity(5, novelty_floor_pct=0.30)
        assert gate.novelty_floor_pct == 0.30

    def test_dynamic_consumption_decay(self):
        from app.consciousness.workspace_buffer import CompetitiveGate
        gate = CompetitiveGate()
        gate.set_dynamic_capacity(5, consumption_decay=0.70)
        assert gate.consumption_decay == 0.70


class TestSourceTraitAudit:
    """Test that source traits are recorded for audit trail."""

    @patch("app.personality.state.get_personality")
    @patch("app.self_awareness.homeostasis.get_state", return_value={"frustration": 0.1, "cognitive_energy": 0.7})
    def test_source_traits_populated(self, mock_homeo, mock_pers):
        mock_pers.return_value = _mock_personality(focus=0.8, creativity=0.3)
        from app.consciousness.personality_workspace import compute_workspace_profile
        profile = compute_workspace_profile("researcher")
        assert "focus_quality" in profile.source_traits
        assert "solution_creativity" in profile.source_traits
        assert "developmental_stage" in profile.source_traits
        assert profile.source_traits["focus_quality"] == 0.8

    @patch("app.personality.state.get_personality")
    @patch("app.self_awareness.homeostasis.get_state")
    def test_homeostasis_adjustment_recorded(self, mock_homeo, mock_pers):
        mock_pers.return_value = _mock_personality()
        mock_homeo.return_value = {"frustration": 0.8, "cognitive_energy": 0.7}
        from app.consciousness.personality_workspace import compute_workspace_profile
        profile = compute_workspace_profile("commander")
        assert profile.source_traits.get("homeostasis_adjustment") == 2


class TestOrchestratorIntegration:
    """Verify PDS workspace is wired into orchestrator."""

    def test_orchestrator_calls_compute_workspace_profile(self):
        from pathlib import Path
        src = (Path(__file__).parent.parent / "app" / "agents" / "commander" / "orchestrator.py").read_text()
        assert "compute_workspace_profile" in src
        assert "set_dynamic_capacity" in src
