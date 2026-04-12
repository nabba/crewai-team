"""
Comprehensive tests for variational free energy and continuous attention schema.

Covers:
  1. Variational FE: KL divergence computation, surprise term, decomposition (F = KL + Surprise)
  2. Prior profiles: get_prior_profile() for each task type
  3. Attention schema: 5 continuous dimensions, correlation with discrete disposition
  4. Integration: VFE in HyperModelState, attention in InternalState, context string injection

Run: pytest tests/test_vfe_attention.py -v
"""

import math
import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

for _dep in ("chromadb", "psycopg2", "psycopg2.extras", "psycopg2.pool"):
    if _dep not in sys.modules:
        sys.modules[_dep] = MagicMock()

from app.self_awareness.internal_state import (
    CertaintyVector, SomaticMarker, MetaCognitiveState, InternalState,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. VARIATIONAL FREE ENERGY — KL DIVERGENCE
# ═══════════════════════════════════════════════════════════════════════════════


class TestVariationalFreeEnergy:
    """compute_variational_free_energy: F = KL(q||p) + Surprise."""

    def _make_hm(self):
        from app.self_awareness.hyper_model import HyperModel
        return HyperModel("vfe_test")

    def test_returns_dict_with_all_keys(self):
        hm = self._make_hm()
        cv = CertaintyVector()
        result = hm.compute_variational_free_energy(cv, "default", 0.1)
        assert "free_energy" in result
        assert "kl_divergence" in result
        assert "surprise" in result
        assert "complexity" in result
        assert "accuracy" in result

    def test_none_certainty_returns_zeros(self):
        hm = self._make_hm()
        result = hm.compute_variational_free_energy(None, "default", 0.1)
        assert result["free_energy"] == 0.0
        assert result["kl_divergence"] == 0.0

    def test_decomposition_f_equals_kl_plus_surprise(self):
        """F = KL(q||p) + Surprise must hold exactly."""
        hm = self._make_hm()
        cv = CertaintyVector(
            factual_grounding=0.8, tool_confidence=0.4,
            coherence=0.7, task_understanding=0.5,
            value_alignment=0.6, meta_certainty=0.5,
        )
        result = hm.compute_variational_free_energy(cv, "research", 0.2)
        assert result["free_energy"] == pytest.approx(
            result["kl_divergence"] + result["surprise"], abs=0.001
        )

    def test_complexity_equals_kl(self):
        hm = self._make_hm()
        cv = CertaintyVector()
        result = hm.compute_variational_free_energy(cv, "default", 0.1)
        assert result["complexity"] == result["kl_divergence"]

    def test_accuracy_equals_negative_surprise(self):
        hm = self._make_hm()
        cv = CertaintyVector()
        result = hm.compute_variational_free_energy(cv, "default", 0.1)
        assert result["accuracy"] == pytest.approx(-result["surprise"], abs=0.001)

    def test_kl_zero_when_beliefs_match_prior(self):
        """KL divergence should be near zero when q ≈ p."""
        hm = self._make_hm()
        # Default profile is all 0.7 — set certainty to match
        cv = CertaintyVector(
            factual_grounding=0.7, tool_confidence=0.7,
            coherence=0.7, task_understanding=0.7,
            value_alignment=0.7, meta_certainty=0.7,
        )
        result = hm.compute_variational_free_energy(cv, "default", 0.0)
        assert result["kl_divergence"] < 0.1  # Near zero

    def test_kl_increases_when_beliefs_diverge(self):
        """KL divergence should increase when certainty differs from prior."""
        hm = self._make_hm()
        # Research prior: factual=1.0, tool=0.6, etc.
        # Certainty that matches prior closely
        cv_close = CertaintyVector(
            factual_grounding=0.9, tool_confidence=0.6,
            coherence=0.8, task_understanding=0.7,
            value_alignment=0.4, meta_certainty=0.5,
        )
        # Certainty that diverges from prior
        cv_far = CertaintyVector(
            factual_grounding=0.1, tool_confidence=0.9,
            coherence=0.2, task_understanding=0.1,
            value_alignment=0.9, meta_certainty=0.9,
        )
        result_close = hm.compute_variational_free_energy(cv_close, "research", 0.1)
        result_far = hm.compute_variational_free_energy(cv_far, "research", 0.1)
        assert result_far["kl_divergence"] > result_close["kl_divergence"]

    def test_different_task_types_different_kl(self):
        """Same certainty vector should produce different KL for different task types."""
        hm = self._make_hm()
        cv = CertaintyVector(
            factual_grounding=0.9, tool_confidence=0.3,
            coherence=0.7, task_understanding=0.6,
            value_alignment=0.8, meta_certainty=0.5,
        )
        research = hm.compute_variational_free_energy(cv, "research", 0.1)
        coding = hm.compute_variational_free_energy(cv, "coding", 0.1)
        assert research["kl_divergence"] != coding["kl_divergence"]

    def test_surprise_increases_with_prediction_error(self):
        """Higher prediction error = more surprise."""
        hm = self._make_hm()
        cv = CertaintyVector()
        low = hm.compute_variational_free_energy(cv, "default", 0.05)
        high = hm.compute_variational_free_energy(cv, "default", 0.5)
        assert high["surprise"] > low["surprise"]

    def test_surprise_zero_at_zero_error(self):
        """Zero prediction error = minimal surprise."""
        hm = self._make_hm()
        cv = CertaintyVector()
        result = hm.compute_variational_free_energy(cv, "default", 0.0)
        assert result["surprise"] < 0.02  # -log(1.0) ≈ 0

    def test_surprise_bounded_at_max_error(self):
        """Near-1.0 prediction error should produce high but bounded surprise."""
        hm = self._make_hm()
        cv = CertaintyVector()
        result = hm.compute_variational_free_energy(cv, "default", 0.99)
        assert result["surprise"] > 2.0  # -log(0.01) ≈ 4.6
        assert result["surprise"] < 10.0

    def test_kl_non_negative(self):
        """KL divergence is always ≥ 0."""
        hm = self._make_hm()
        for fg in [0.1, 0.5, 0.9]:
            for tc in [0.1, 0.5, 0.9]:
                cv = CertaintyVector(factual_grounding=fg, tool_confidence=tc)
                result = hm.compute_variational_free_energy(cv, "default", 0.1)
                assert result["kl_divergence"] >= 0.0

    def test_free_energy_non_negative(self):
        """Total free energy is always ≥ 0 (KL ≥ 0, Surprise ≥ 0)."""
        hm = self._make_hm()
        cv = CertaintyVector()
        result = hm.compute_variational_free_energy(cv, "default", 0.3)
        assert result["free_energy"] >= 0.0


class TestVFEInHyperModelState:
    """Variational FE fields in HyperModelState after update()."""

    def test_vfe_in_state_after_update(self):
        from app.self_awareness.hyper_model import HyperModel
        hm = HyperModel("state_vfe")
        cv = CertaintyVector(factual_grounding=0.8)
        hm.predict_next_step()
        state = hm.update(0.6, certainty_vector=cv, task_type="research")
        assert state.variational_fe > 0
        assert state.kl_divergence >= 0
        assert state.surprise_term >= 0

    def test_vfe_in_to_dict(self):
        from app.self_awareness.hyper_model import HyperModel
        hm = HyperModel("dict_vfe")
        cv = CertaintyVector()
        hm.predict_next_step()
        state = hm.update(0.5, certainty_vector=cv)
        d = state.to_dict()
        assert "variational_fe" in d
        assert "kl_divergence" in d
        assert "surprise_term" in d

    def test_vfe_without_certainty_vector(self):
        """update() without certainty_vector should produce VFE=0 (graceful)."""
        from app.self_awareness.hyper_model import HyperModel
        hm = HyperModel("no_cv")
        hm.predict_next_step()
        state = hm.update(0.5)
        assert state.variational_fe == 0.0

    def test_pressure_uses_vfe_when_available(self):
        """get_free_energy_pressure should use variational_fe instead of proxy."""
        from app.self_awareness.hyper_model import HyperModel
        hm = HyperModel("pressure_vfe")
        cv = CertaintyVector(
            factual_grounding=0.1, tool_confidence=0.1,  # Very far from any prior
            coherence=0.1, task_understanding=0.1,
            value_alignment=0.1, meta_certainty=0.1,
        )
        hm.predict_next_step()
        hm.update(0.1, certainty_vector=cv, task_type="research")
        pressure = hm.get_free_energy_pressure()
        assert pressure > 0  # High VFE → non-zero pressure


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PRIOR PROFILES
# ═══════════════════════════════════════════════════════════════════════════════


class TestPriorProfiles:
    """get_prior_profile: task-type certainty priors for KL computation."""

    def test_returns_list_of_6(self):
        from app.self_awareness.precision_weighting import PrecisionWeighting
        pw = PrecisionWeighting()
        profile = pw.get_prior_profile("default")
        assert isinstance(profile, list)
        assert len(profile) == 6

    def test_all_values_in_range(self):
        from app.self_awareness.precision_weighting import PrecisionWeighting
        pw = PrecisionWeighting()
        for task_type in ["research", "coding", "writing", "media", "default"]:
            profile = pw.get_prior_profile(task_type)
            for v in profile:
                assert 0.0 < v <= 1.0

    def test_research_prioritizes_facts(self):
        from app.self_awareness.precision_weighting import PrecisionWeighting
        pw = PrecisionWeighting()
        profile = pw.get_prior_profile("research")
        assert profile[0] == 1.0  # factual_grounding is first, highest for research

    def test_coding_prioritizes_tools(self):
        from app.self_awareness.precision_weighting import PrecisionWeighting
        pw = PrecisionWeighting()
        profile = pw.get_prior_profile("coding")
        assert profile[1] == 1.0  # tool_confidence is second, highest for coding

    def test_writing_prioritizes_coherence(self):
        from app.self_awareness.precision_weighting import PrecisionWeighting
        pw = PrecisionWeighting()
        profile = pw.get_prior_profile("writing")
        assert profile[2] == 1.0  # coherence is third, highest for writing

    def test_default_is_uniform(self):
        from app.self_awareness.precision_weighting import PrecisionWeighting
        pw = PrecisionWeighting()
        profile = pw.get_prior_profile("default")
        assert all(v == 0.7 for v in profile)

    def test_unknown_task_type_uses_default(self):
        from app.self_awareness.precision_weighting import PrecisionWeighting
        pw = PrecisionWeighting()
        profile = pw.get_prior_profile("nonexistent_crew")
        assert profile == pw.get_prior_profile("default")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CONTINUOUS ATTENTION SCHEMA
# ═══════════════════════════════════════════════════════════════════════════════


class TestAttentionSchema:
    """AttentionSchema: 5D continuous attention model."""

    def test_dataclass_fields(self):
        from app.self_awareness.dual_channel import AttentionSchema
        a = AttentionSchema()
        assert hasattr(a, "focus_intensity")
        assert hasattr(a, "caution_level")
        assert hasattr(a, "exploration_drive")
        assert hasattr(a, "metacognitive_load")
        assert hasattr(a, "somatic_salience")

    def test_defaults(self):
        from app.self_awareness.dual_channel import AttentionSchema
        a = AttentionSchema()
        assert a.focus_intensity == 0.5
        assert a.caution_level == 0.25
        assert a.exploration_drive == 0.5
        assert a.metacognitive_load == 0.3
        assert a.somatic_salience == 0.0

    def test_to_dict(self):
        from app.self_awareness.dual_channel import AttentionSchema
        a = AttentionSchema(focus_intensity=0.8, caution_level=0.3)
        d = a.to_dict()
        assert d["focus_intensity"] == 0.8
        assert d["caution_level"] == 0.3
        assert len(d) == 5


class TestAttentionComputation:
    """_compute_attention_schema: continuous attention from internal state."""

    def _compose(self, certainty_kwargs, somatic_kwargs=None):
        from app.self_awareness.dual_channel import DualChannelComposer
        composer = DualChannelComposer()
        cv = CertaintyVector(**certainty_kwargs)
        sm = SomaticMarker(**(somatic_kwargs or {}))
        state = InternalState(certainty=cv, somatic=sm)
        return composer.compose(state)

    def test_high_certainty_high_focus(self):
        result = self._compose(
            {"factual_grounding": 0.9, "tool_confidence": 0.9, "coherence": 0.9,
             "task_understanding": 0.9, "value_alignment": 0.9, "meta_certainty": 0.9},
        )
        assert result.attention_schema["focus_intensity"] > 0.7

    def test_low_certainty_low_focus(self):
        result = self._compose(
            {"factual_grounding": 0.2, "tool_confidence": 0.2, "coherence": 0.2,
             "task_understanding": 0.2, "value_alignment": 0.2, "meta_certainty": 0.2},
        )
        assert result.attention_schema["focus_intensity"] < 0.3

    def test_negative_somatic_increases_caution(self):
        result_pos = self._compose(
            {"factual_grounding": 0.6, "tool_confidence": 0.6, "coherence": 0.6,
             "task_understanding": 0.6, "value_alignment": 0.6, "meta_certainty": 0.6},
            {"valence": 0.5, "intensity": 0.7},
        )
        result_neg = self._compose(
            {"factual_grounding": 0.6, "tool_confidence": 0.6, "coherence": 0.6,
             "task_understanding": 0.6, "value_alignment": 0.6, "meta_certainty": 0.6},
            {"valence": -0.7, "intensity": 0.7},
        )
        assert result_neg.attention_schema["caution_level"] > result_pos.attention_schema["caution_level"]

    def test_somatic_salience_from_intensity_and_valence(self):
        result = self._compose(
            {"factual_grounding": 0.5, "tool_confidence": 0.5, "coherence": 0.5,
             "task_understanding": 0.5, "value_alignment": 0.5, "meta_certainty": 0.5},
            {"valence": -0.8, "intensity": 0.9},
        )
        # somatic_salience = intensity * |valence| = 0.9 * 0.8 = 0.72
        assert result.attention_schema["somatic_salience"] == pytest.approx(0.72, abs=0.01)

    def test_zero_somatic_zero_salience(self):
        result = self._compose(
            {"factual_grounding": 0.5, "tool_confidence": 0.5, "coherence": 0.5,
             "task_understanding": 0.5, "value_alignment": 0.5, "meta_certainty": 0.5},
            {"valence": 0.0, "intensity": 0.0},
        )
        assert result.attention_schema["somatic_salience"] == 0.0

    def test_high_variance_increases_meta_load(self):
        # High variance = dims spread out
        result_spread = self._compose(
            {"factual_grounding": 0.1, "tool_confidence": 0.9, "coherence": 0.5,
             "task_understanding": 0.3, "value_alignment": 0.7, "meta_certainty": 0.5},
        )
        # Low variance = dims uniform
        result_uniform = self._compose(
            {"factual_grounding": 0.6, "tool_confidence": 0.6, "coherence": 0.6,
             "task_understanding": 0.6, "value_alignment": 0.6, "meta_certainty": 0.6},
        )
        assert result_spread.attention_schema["metacognitive_load"] > result_uniform.attention_schema["metacognitive_load"]

    def test_all_dimensions_bounded_0_1(self):
        """All attention dimensions should be in [0, 1]."""
        for fg in [0.1, 0.5, 0.9]:
            for val in [-0.9, 0.0, 0.9]:
                result = self._compose(
                    {"factual_grounding": fg, "tool_confidence": fg, "coherence": fg,
                     "task_understanding": fg, "value_alignment": fg, "meta_certainty": fg},
                    {"valence": val, "intensity": abs(val)},
                )
                a = result.attention_schema
                for key in ["focus_intensity", "caution_level", "exploration_drive",
                             "metacognitive_load", "somatic_salience"]:
                    assert 0.0 <= a[key] <= 1.0, f"{key}={a[key]} out of bounds"

    def test_backward_compatible_disposition_still_set(self):
        """Discrete disposition must still be computed alongside attention schema."""
        result = self._compose(
            {"factual_grounding": 0.9, "tool_confidence": 0.9, "coherence": 0.9,
             "task_understanding": 0.9, "value_alignment": 0.9, "meta_certainty": 0.9},
            {"valence": 0.5, "intensity": 0.5},
        )
        assert result.action_disposition == "proceed"
        assert result.risk_tier == 1
        assert result.attention_schema is not None


class TestCautionLevelCorrelation:
    """Continuous caution_level should correlate with discrete disposition."""

    def _compose(self, cert, val):
        from app.self_awareness.dual_channel import DualChannelComposer
        composer = DualChannelComposer()
        cv = CertaintyVector(
            factual_grounding=cert, tool_confidence=cert, coherence=cert,
            task_understanding=cert, value_alignment=cert, meta_certainty=1.0,
        )
        sm = SomaticMarker(valence=val, intensity=0.5)
        state = InternalState(certainty=cv, somatic=sm)
        return composer.compose(state)

    def test_proceed_has_low_caution(self):
        result = self._compose(cert=0.9, val=0.5)
        assert result.action_disposition == "proceed"
        assert result.attention_schema["caution_level"] < 0.3

    def test_cautious_has_mid_caution(self):
        result = self._compose(cert=0.9, val=-0.5)
        assert result.action_disposition == "cautious"
        assert 0.15 < result.attention_schema["caution_level"] < 0.6

    def test_escalate_has_high_caution(self):
        result = self._compose(cert=0.2, val=-0.5)
        assert result.action_disposition == "escalate"
        assert result.attention_schema["caution_level"] > 0.6

    def test_caution_monotonic_with_decreasing_certainty(self):
        """Lower certainty → higher continuous caution."""
        results = []
        for cert in [0.9, 0.6, 0.3]:
            result = self._compose(cert=cert, val=0.0)
            results.append(result.attention_schema["caution_level"])
        assert results[0] < results[1] < results[2]


# ═══════════════════════════════════════════════════════════════════════════════
# 4. INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════


class TestInternalStateIntegration:
    """attention_schema in InternalState: storage, context, JSON."""

    def test_attention_schema_field_exists(self):
        state = InternalState()
        assert hasattr(state, "attention_schema")
        assert state.attention_schema is None  # Default

    def test_attention_in_to_context_string(self):
        state = InternalState(
            attention_schema={"focus_intensity": 0.8, "caution_level": 0.2,
                             "exploration_drive": 0.4, "metacognitive_load": 0.3,
                             "somatic_salience": 0.1},
        )
        ctx = state.to_context_string()
        assert "Attention:" in ctx
        assert "focus=0.8" in ctx
        assert "caution=0.2" in ctx
        assert "explore=0.4" in ctx

    def test_no_attention_in_context_when_none(self):
        state = InternalState()
        ctx = state.to_context_string()
        assert "Attention:" not in ctx

    def test_attention_in_to_json(self):
        import json
        state = InternalState(
            attention_schema={"focus_intensity": 0.7, "caution_level": 0.3,
                             "exploration_drive": 0.5, "metacognitive_load": 0.2,
                             "somatic_salience": 0.0},
        )
        j = json.loads(state.to_json())
        assert "attention_schema" in j
        assert j["attention_schema"]["focus_intensity"] == 0.7

    def test_vfe_fields_in_hyper_model_state_json(self):
        """VFE fields should appear in HyperModelState.to_dict()."""
        from app.self_awareness.hyper_model import HyperModelState
        state = HyperModelState(variational_fe=1.5, kl_divergence=1.0, surprise_term=0.5)
        d = state.to_dict()
        assert d["variational_fe"] == 1.5
        assert d["kl_divergence"] == 1.0
        assert d["surprise_term"] == 0.5


class TestKLDivergenceMath:
    """Verify the KL divergence formula is mathematically correct."""

    def test_kl_symmetric_case(self):
        """KL(N(0.5, 0.25) || N(0.5, 0.25)) = 0 (same distribution)."""
        q_mu, p_mu = 0.5, 0.5
        q_var = max(0.01, q_mu * (1 - q_mu))  # 0.25
        p_var = max(0.01, p_mu * (1 - p_mu))  # 0.25
        kl = math.log(math.sqrt(p_var / q_var)) + (q_var + (q_mu - p_mu)**2) / (2 * p_var) - 0.5
        assert abs(kl) < 0.001  # Should be ~0

    def test_kl_different_means(self):
        """KL should be positive when means differ."""
        q_mu, p_mu = 0.8, 0.3
        q_var = max(0.01, q_mu * (1 - q_mu))
        p_var = max(0.01, p_mu * (1 - p_mu))
        kl = math.log(math.sqrt(p_var / q_var)) + (q_var + (q_mu - p_mu)**2) / (2 * p_var) - 0.5
        assert kl > 0

    def test_surprise_formula(self):
        """-log(1 - error) should be 0 at error=0, +inf as error→1."""
        assert -math.log(max(0.01, 1.0 - 0.0)) == pytest.approx(0.0, abs=0.01)
        assert -math.log(max(0.01, 1.0 - 0.5)) > 0.5  # ln(2) ≈ 0.693
        assert -math.log(max(0.01, 1.0 - 0.99)) > 4.0  # ln(100) ≈ 4.605
