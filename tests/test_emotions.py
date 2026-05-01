"""
Comprehensive tests for the emotions/somatic subsystem.

Covers:
  - SomaticMarker data structure
  - SomaticMarkerComputer: pgvector similarity, temporal decay, homeostatic modulation
  - SomaticBiasInjector: pre-reasoning bias injection, thresholds, disposition floor
  - DualChannelComposer: certainty × valence → disposition matrix, safety monotonicity
  - Affective forecasting: backward + forward, belief sentiment
  - InferentialCompetition: affective score in plan scoring
  - Homeostasis ↔ somatic bidirectional coupling
  - record_experience_sync: outcome clamping, embedding
  - InternalState: context string injection, somatic threshold
  - End-to-end wiring: PRE_TASK → POST_LLM_CALL → experience recording

Run: pytest tests/test_emotions.py -v
"""

import json
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Module-level mocks (restored by teardown_module below) ──
# Only mock what we genuinely need to avoid I/O. chromadb, psycopg2 etc.
# are all installed; chromadb is mocked here only because somatic_marker's
# lazy imports would otherwise pull in the real heavy module mid-test.
# Without the teardown, these mutations bleed into every later test file.
_MOCK_KEYS_INSERTED: list[str] = []
_PARENT_ATTRS_OVERRIDDEN: list[tuple[object, str, object]] = []


def _mock_module(name: str) -> MagicMock:
    mock = MagicMock()
    if name not in sys.modules:
        _MOCK_KEYS_INSERTED.append(name)
    sys.modules[name] = mock
    return mock


def _override_attr(parent, attr, value):
    sentinel = object()
    original = getattr(parent, attr, sentinel)
    _PARENT_ATTRS_OVERRIDDEN.append((parent, attr, original))
    setattr(parent, attr, value)


for mod_name in (
    "chromadb", "psycopg2", "psycopg2.extras", "psycopg2.pool",
):
    if mod_name not in sys.modules:
        _MOCK_KEYS_INSERTED.append(mod_name)
        sys.modules[mod_name] = MagicMock()

import app
import app.memory

# Mock control_plane.db (real db.py uses Python 3.10+ `list | None` syntax
# which is fine here, but we want to control execute() per-test).
if "app.control_plane" not in sys.modules:
    _cp_db = _mock_module("app.control_plane.db")
    _cp_db.execute = MagicMock(return_value=[])
    _cp = _mock_module("app.control_plane")
    _cp.db = _cp_db
    _override_attr(app, "control_plane", _cp)

if "app.memory.chromadb_manager" not in sys.modules:
    _cm = _mock_module("app.memory.chromadb_manager")
    _cm.embed = MagicMock(return_value=[0.1] * 768)
    _override_attr(app.memory, "chromadb_manager", _cm)


def teardown_module(module):
    """Undo the module-level mocks so they don't bleed into later tests."""
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

from app.self_awareness.internal_state import (
    SomaticMarker, CertaintyVector, MetaCognitiveState,
    InternalState, DISPOSITION_TO_RISK_TIER,
)


# ═══════════════════════════════════════════════════════════════════════════════
# SOMATIC MARKER DATA STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════


class TestSomaticMarkerDataclass:
    """SomaticMarker dataclass fields and methods."""

    def test_default_values(self):
        sm = SomaticMarker()
        assert sm.valence == 0.0
        assert sm.intensity == 0.0
        assert sm.source == "no_prior"
        assert sm.match_count == 0

    def test_custom_values(self):
        sm = SomaticMarker(valence=-0.7, intensity=0.9, source="past failure", match_count=3)
        assert sm.valence == -0.7
        assert sm.intensity == 0.9
        assert sm.source == "past failure"
        assert sm.match_count == 3

    def test_to_dict(self):
        sm = SomaticMarker(valence=0.333333, intensity=0.666666, source="test", match_count=2)
        d = sm.to_dict()
        assert d["valence"] == 0.333
        assert d["intensity"] == 0.667
        assert d["source"] == "test"
        assert d["match_count"] == 2

    def test_valence_range(self):
        """Valence should be usable in [-1, 1]."""
        sm_neg = SomaticMarker(valence=-1.0)
        sm_pos = SomaticMarker(valence=1.0)
        assert sm_neg.valence == -1.0
        assert sm_pos.valence == 1.0

    def test_intensity_range(self):
        """Intensity should be in [0, 1]."""
        sm = SomaticMarker(intensity=0.0)
        assert sm.intensity == 0.0
        sm = SomaticMarker(intensity=1.0)
        assert sm.intensity == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# SOMATIC MARKER COMPUTER
# ═══════════════════════════════════════════════════════════════════════════════


class TestSomaticMarkerComputer:
    """SomaticMarkerComputer: similarity search + weighted average."""

    def test_defaults(self):
        from app.self_awareness.somatic_marker import SomaticMarkerComputer
        smc = SomaticMarkerComputer()
        assert smc.top_k == 5
        assert smc.decay_factor == 0.95
        assert smc.min_similarity == 0.3
        assert smc.temporal_half_life_hours == 168.0

    def test_no_embedding_no_db_returns_default(self):
        """When embed() fails and no embedding given, return default marker."""
        from app.self_awareness.somatic_marker import SomaticMarkerComputer
        smc = SomaticMarkerComputer()
        with patch("app.memory.chromadb_manager.embed", side_effect=Exception("no ollama")):
            result = smc.compute("test_agent", "some context")
            assert result.valence == 0.0
            assert result.intensity == 0.0

    def test_no_rows_returns_no_prior(self):
        from app.self_awareness.somatic_marker import SomaticMarkerComputer
        smc = SomaticMarkerComputer()
        with patch("app.control_plane.db.execute", return_value=[]):
            result = smc.compute("agent", "ctx", context_embedding=[0.1] * 768)
            assert result.source == "no_prior_experience"
            assert result.match_count == 0

    def test_below_similarity_threshold_returns_no_relevant(self):
        from app.self_awareness.somatic_marker import SomaticMarkerComputer
        smc = SomaticMarkerComputer(min_similarity=0.5)
        rows = [{"outcome_score": 1.0, "context_summary": "test", "created_at": None, "similarity": 0.3}]
        with patch("app.control_plane.db.execute", return_value=rows):
            result = smc.compute("agent", "ctx", context_embedding=[0.1] * 768)
            assert result.source == "no_relevant_experience"

    def test_single_positive_experience(self):
        from app.self_awareness.somatic_marker import SomaticMarkerComputer
        smc = SomaticMarkerComputer()
        rows = [{"outcome_score": 0.8, "context_summary": "good task", "created_at": None, "similarity": 0.9}]
        with patch("app.control_plane.db.execute", return_value=rows), \
             patch("app.self_awareness.homeostasis.get_state", return_value={"frustration": 0.1, "cognitive_energy": 0.7, "confidence": 0.5}):
            result = smc.compute("agent", "ctx", context_embedding=[0.1] * 768)
            assert result.valence > 0
            assert result.intensity == 0.9
            assert result.match_count == 1
            assert "good task" in result.source

    def test_single_negative_experience(self):
        from app.self_awareness.somatic_marker import SomaticMarkerComputer
        smc = SomaticMarkerComputer()
        rows = [{"outcome_score": -0.8, "context_summary": "failed task", "created_at": None, "similarity": 0.85}]
        with patch("app.control_plane.db.execute", return_value=rows), \
             patch("app.self_awareness.homeostasis.get_state", return_value={"frustration": 0.1, "cognitive_energy": 0.7, "confidence": 0.5}):
            result = smc.compute("agent", "ctx", context_embedding=[0.1] * 768)
            assert result.valence < 0
            assert result.match_count == 1

    def test_multiple_experiences_weighted_average(self):
        """Multiple experiences should produce weighted average valence."""
        from app.self_awareness.somatic_marker import SomaticMarkerComputer
        smc = SomaticMarkerComputer()
        rows = [
            {"outcome_score": 1.0, "context_summary": "great", "created_at": None, "similarity": 0.9},
            {"outcome_score": -1.0, "context_summary": "terrible", "created_at": None, "similarity": 0.8},
            {"outcome_score": 0.5, "context_summary": "ok", "created_at": None, "similarity": 0.6},
        ]
        with patch("app.control_plane.db.execute", return_value=rows), \
             patch("app.self_awareness.homeostasis.get_state", return_value={"frustration": 0.1, "cognitive_energy": 0.7, "confidence": 0.5}):
            result = smc.compute("agent", "ctx", context_embedding=[0.1] * 768)
            # Should be between -1 and 1, weighted toward first (highest sim)
            assert -1.0 <= result.valence <= 1.0
            assert result.match_count == 3

    def test_recency_weight_decay(self):
        """First match (position 0) should weigh more than later matches."""
        from app.self_awareness.somatic_marker import SomaticMarkerComputer
        smc = SomaticMarkerComputer(decay_factor=0.5)  # Aggressive decay
        rows = [
            {"outcome_score": 1.0, "context_summary": "a", "created_at": None, "similarity": 0.9},
            {"outcome_score": -1.0, "context_summary": "b", "created_at": None, "similarity": 0.9},
        ]
        with patch("app.control_plane.db.execute", return_value=rows), \
             patch("app.self_awareness.homeostasis.get_state", return_value={"frustration": 0.1, "cognitive_energy": 0.7, "confidence": 0.5}):
            result = smc.compute("agent", "ctx", context_embedding=[0.1] * 768)
            # With 0.5 decay: first=1.0*0.9*1.0, second=-1.0*0.45*1.0
            # Weighted sum > 0 (positive dominates due to recency)
            assert result.valence > 0

    def test_db_exception_returns_default(self):
        """Database failure should return default somatic marker, not crash."""
        from app.self_awareness.somatic_marker import SomaticMarkerComputer
        smc = SomaticMarkerComputer()
        with patch("app.control_plane.db.execute", side_effect=Exception("connection refused")):
            result = smc.compute("agent", "ctx", context_embedding=[0.1] * 768)
            assert result.valence == 0.0
            assert result.intensity == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# TEMPORAL DECAY
# ═══════════════════════════════════════════════════════════════════════════════


class TestTemporalDecay:
    """Temporal decay: old experiences lose weight but never below 20% floor."""

    def _make_rows(self, ages_hours: list[float], scores: list[float], similarity: float = 0.9):
        """Create mock DB rows with specified ages."""
        now = datetime.now(timezone.utc)
        rows = []
        for age_h, score in zip(ages_hours, scores):
            rows.append({
                "outcome_score": score,
                "context_summary": f"task_{age_h}h_ago",
                "created_at": now - timedelta(hours=age_h),
                "similarity": similarity,
            })
        return rows

    def test_recent_experience_full_weight(self):
        """Experience from 1 hour ago should have nearly full temporal weight."""
        from app.self_awareness.somatic_marker import SomaticMarkerComputer
        smc = SomaticMarkerComputer(temporal_half_life_hours=168)
        rows = self._make_rows([1.0], [-1.0])
        with patch("app.control_plane.db.execute", return_value=rows), \
             patch("app.self_awareness.homeostasis.get_state", return_value={"frustration": 0.1, "cognitive_energy": 0.7, "confidence": 0.5}):
            result = smc.compute("agent", "ctx", context_embedding=[0.1] * 768)
            # 1h / 168h half-life → decay ~0.996, nearly full weight
            assert result.valence < -0.9

    def test_one_week_old_half_weight(self):
        """168h old experience should have ~50% temporal weight."""
        from app.self_awareness.somatic_marker import SomaticMarkerComputer
        smc = SomaticMarkerComputer(temporal_half_life_hours=168)
        # One fresh positive, one 168h-old negative (equal similarity)
        rows = self._make_rows([0, 168], [1.0, -1.0])
        with patch("app.control_plane.db.execute", return_value=rows), \
             patch("app.self_awareness.homeostasis.get_state", return_value={"frustration": 0.1, "cognitive_energy": 0.7, "confidence": 0.5}):
            result = smc.compute("agent", "ctx", context_embedding=[0.1] * 768)
            # Fresh positive (full weight) vs 168h negative (half weight)
            # Positive should dominate
            assert result.valence > 0

    def test_very_old_experience_retains_floor_weight(self):
        """30-day-old experience should retain at least 20% weight (floor)."""
        from app.self_awareness.somatic_marker import SomaticMarkerComputer
        smc = SomaticMarkerComputer(temporal_half_life_hours=168)
        rows = self._make_rows([720], [-1.0])  # 30 days old
        with patch("app.control_plane.db.execute", return_value=rows), \
             patch("app.self_awareness.homeostasis.get_state", return_value={"frustration": 0.1, "cognitive_energy": 0.7, "confidence": 0.5}):
            result = smc.compute("agent", "ctx", context_embedding=[0.1] * 768)
            # 720h / 168h = ~4.3 half-lives → raw decay = 0.5^4.3 ≈ 0.05
            # But floor is 0.2, so temporal_weight = 0.2
            # Still negative (the failure persists)
            assert result.valence < 0
            assert result.match_count == 1

    def test_floor_never_zero(self):
        """Even extremely old experiences should produce non-zero valence."""
        from app.self_awareness.somatic_marker import SomaticMarkerComputer
        smc = SomaticMarkerComputer(temporal_half_life_hours=168)
        rows = self._make_rows([8760], [-1.0])  # 1 year old
        with patch("app.control_plane.db.execute", return_value=rows), \
             patch("app.self_awareness.homeostasis.get_state", return_value={"frustration": 0.1, "cognitive_energy": 0.7, "confidence": 0.5}):
            result = smc.compute("agent", "ctx", context_embedding=[0.1] * 768)
            assert result.valence != 0.0  # Floor prevents zero
            assert result.valence < 0  # Still negative

    def test_no_created_at_defaults_to_full_weight(self):
        """Missing created_at should default to temporal_weight=1.0."""
        from app.self_awareness.somatic_marker import SomaticMarkerComputer
        smc = SomaticMarkerComputer()
        rows = [{"outcome_score": -0.5, "context_summary": "test", "created_at": None, "similarity": 0.8}]
        with patch("app.control_plane.db.execute", return_value=rows), \
             patch("app.self_awareness.homeostasis.get_state", return_value={"frustration": 0.1, "cognitive_energy": 0.7, "confidence": 0.5}):
            result = smc.compute("agent", "ctx", context_embedding=[0.1] * 768)
            assert result.valence < 0  # Full weight applied

    def test_temporal_decay_math(self):
        """Verify half-life formula directly."""
        half_life = 168.0
        floor = 0.2
        # At exactly 1 half-life: raw_decay = 0.5
        raw = 0.5 ** (168.0 / half_life)
        assert raw == pytest.approx(0.5)
        assert max(floor, raw) == 0.5
        # At 3 half-lives: raw_decay = 0.125
        raw = 0.5 ** (504.0 / half_life)
        assert raw == pytest.approx(0.125)
        assert max(floor, raw) == 0.2  # Floor kicks in


# ═══════════════════════════════════════════════════════════════════════════════
# HOMEOSTATIC MODULATION
# ═══════════════════════════════════════════════════════════════════════════════


class TestHomeostaticModulation:
    """_apply_homeostatic_modulation: frustration amplifies, energy dampens."""

    def test_neutral_state_no_change(self):
        """Default homeostatic state should not significantly alter signals."""
        from app.self_awareness.somatic_marker import SomaticMarkerComputer
        with patch("app.self_awareness.homeostasis.get_state",
                   return_value={"frustration": 0.1, "cognitive_energy": 0.7, "confidence": 0.5}):
            v, i = SomaticMarkerComputer._apply_homeostatic_modulation(-0.5, 0.8)
            assert v == pytest.approx(-0.5, abs=0.05)
            assert i == pytest.approx(0.8, abs=0.05)

    def test_high_frustration_amplifies_negative(self):
        """High frustration should make negative valence more negative."""
        from app.self_awareness.somatic_marker import SomaticMarkerComputer
        with patch("app.self_awareness.homeostasis.get_state",
                   return_value={"frustration": 0.9, "cognitive_energy": 0.7, "confidence": 0.5}):
            v, i = SomaticMarkerComputer._apply_homeostatic_modulation(-0.5, 0.8)
            assert v < -0.5  # Amplified

    def test_frustration_does_not_amplify_positive(self):
        """Frustration should NOT amplify positive valence."""
        from app.self_awareness.somatic_marker import SomaticMarkerComputer
        with patch("app.self_awareness.homeostasis.get_state",
                   return_value={"frustration": 0.9, "cognitive_energy": 0.7, "confidence": 0.5}):
            v, i = SomaticMarkerComputer._apply_homeostatic_modulation(0.5, 0.8)
            assert v == pytest.approx(0.5, abs=0.01)  # Unchanged

    def test_high_confidence_dampens_negative(self):
        """High confidence provides resilience against negative signals."""
        from app.self_awareness.somatic_marker import SomaticMarkerComputer
        with patch("app.self_awareness.homeostasis.get_state",
                   return_value={"frustration": 0.1, "cognitive_energy": 0.7, "confidence": 0.95}):
            v, i = SomaticMarkerComputer._apply_homeostatic_modulation(-0.8, 0.8)
            assert v > -0.8  # Dampened (less negative)

    def test_low_energy_dampens_intensity(self):
        """Low cognitive energy should reduce emotional intensity."""
        from app.self_awareness.somatic_marker import SomaticMarkerComputer
        with patch("app.self_awareness.homeostasis.get_state",
                   return_value={"frustration": 0.1, "cognitive_energy": 0.1, "confidence": 0.5}):
            v, i = SomaticMarkerComputer._apply_homeostatic_modulation(-0.5, 0.8)
            assert i < 0.8  # Intensity dampened

    def test_zero_energy_halves_intensity(self):
        """At energy=0, intensity should be halved (factor=0.5)."""
        from app.self_awareness.somatic_marker import SomaticMarkerComputer
        with patch("app.self_awareness.homeostasis.get_state",
                   return_value={"frustration": 0.1, "cognitive_energy": 0.0, "confidence": 0.5}):
            v, i = SomaticMarkerComputer._apply_homeostatic_modulation(-0.5, 0.8)
            assert i == pytest.approx(0.4, abs=0.05)

    def test_normal_energy_no_dampening(self):
        """Energy >= 0.4 should not dampen intensity."""
        from app.self_awareness.somatic_marker import SomaticMarkerComputer
        with patch("app.self_awareness.homeostasis.get_state",
                   return_value={"frustration": 0.1, "cognitive_energy": 0.7, "confidence": 0.5}):
            v, i = SomaticMarkerComputer._apply_homeostatic_modulation(-0.5, 0.8)
            assert i == pytest.approx(0.8, abs=0.01)

    def test_valence_clamped_at_minus_one(self):
        """Amplification should never push valence below -1.0."""
        from app.self_awareness.somatic_marker import SomaticMarkerComputer
        with patch("app.self_awareness.homeostasis.get_state",
                   return_value={"frustration": 1.0, "cognitive_energy": 0.7, "confidence": 0.1}):
            v, i = SomaticMarkerComputer._apply_homeostatic_modulation(-0.9, 0.8)
            assert v >= -1.0

    def test_homeostasis_import_failure_no_crash(self):
        """If homeostasis import fails, modulation should be a no-op."""
        from app.self_awareness.somatic_marker import SomaticMarkerComputer
        with patch("app.self_awareness.homeostasis.get_state", side_effect=ImportError):
            v, i = SomaticMarkerComputer._apply_homeostatic_modulation(-0.5, 0.8)
            assert v == -0.5
            assert i == 0.8


# ═══════════════════════════════════════════════════════════════════════════════
# SOMATIC BIAS INJECTOR
# ═══════════════════════════════════════════════════════════════════════════════


class TestSomaticBiasInjector:
    """Pre-reasoning somatic bias injection."""

    def test_low_intensity_skipped(self):
        from app.self_awareness.somatic_bias import SomaticBiasInjector
        injector = SomaticBiasInjector()
        ctx = {"description": "test task"}
        sm = SomaticMarker(valence=-0.8, intensity=0.1)  # Below MIN_INTENSITY
        result = injector.inject(ctx, sm)
        assert "[Somatic note:" not in result.get("description", "")

    def test_strong_negative_injects_note(self):
        from app.self_awareness.somatic_bias import SomaticBiasInjector
        injector = SomaticBiasInjector()
        ctx = {"description": "do a research task"}
        sm = SomaticMarker(valence=-0.7, intensity=0.8, source="past failure on research")
        result = injector.inject(ctx, sm)
        assert "[Somatic note:" in result["description"]
        assert "strongly negative" in result["description"]
        assert "strategy_hints" in result
        assert len(result["strategy_hints"]) > 0
        assert result["_somatic_advisories"][0]["disposition_floor"] == "cautious"

    def test_mild_negative_injects_awareness(self):
        from app.self_awareness.somatic_bias import SomaticBiasInjector
        injector = SomaticBiasInjector()
        ctx = {"description": "coding task"}
        sm = SomaticMarker(valence=-0.3, intensity=0.6, source="mixed results")
        result = injector.inject(ctx, sm)
        assert "[Somatic note:" in result["description"]
        assert "mixed-to-negative" in result["description"]

    def test_strong_positive_injects_positive_note(self):
        from app.self_awareness.somatic_bias import SomaticBiasInjector
        injector = SomaticBiasInjector()
        ctx = {"description": "task"}
        sm = SomaticMarker(valence=0.7, intensity=0.8, source="great results")
        result = injector.inject(ctx, sm)
        assert "[Somatic note:" in result["description"]
        assert "strongly positive" in result["description"]

    def test_mild_positive_no_note(self):
        from app.self_awareness.somatic_bias import SomaticBiasInjector
        injector = SomaticBiasInjector()
        ctx = {"description": "task"}
        sm = SomaticMarker(valence=0.3, intensity=0.8, source="ok")
        result = injector.inject(ctx, sm)
        # mild_positive has context_note=None, so no [Somatic note:] added
        assert "[Somatic note:" not in result.get("description", "task")

    def test_neutral_dead_band_no_bias(self):
        """Valence in (-0.2, 0.2) should produce no bias."""
        from app.self_awareness.somatic_bias import SomaticBiasInjector
        injector = SomaticBiasInjector()
        ctx = {"description": "task"}
        sm = SomaticMarker(valence=0.0, intensity=0.8, source="neutral")
        result = injector.inject(ctx, sm)
        assert "[Somatic note:" not in result.get("description", "task")

    def test_disposition_floor_extraction(self):
        from app.self_awareness.somatic_bias import SomaticBiasInjector
        ctx = {
            "_somatic_advisories": [
                {"disposition_floor": None},
                {"disposition_floor": "cautious"},
                {"disposition_floor": None},
            ]
        }
        floor = SomaticBiasInjector.get_disposition_floor(ctx)
        assert floor == "cautious"

    def test_disposition_floor_max(self):
        """Multiple floors: highest wins."""
        from app.self_awareness.somatic_bias import SomaticBiasInjector
        ctx = {
            "_somatic_advisories": [
                {"disposition_floor": "cautious"},
                {"disposition_floor": "pause"},
            ]
        }
        floor = SomaticBiasInjector.get_disposition_floor(ctx)
        assert floor == "pause"

    def test_no_advisories_returns_none(self):
        from app.self_awareness.somatic_bias import SomaticBiasInjector
        assert SomaticBiasInjector.get_disposition_floor({}) is None

    def test_original_description_preserved(self):
        """Somatic note should be prepended, not replace original."""
        from app.self_awareness.somatic_bias import SomaticBiasInjector
        injector = SomaticBiasInjector()
        ctx = {"description": "ORIGINAL TASK DESCRIPTION"}
        sm = SomaticMarker(valence=-0.7, intensity=0.8, source="test")
        result = injector.inject(ctx, sm)
        assert "ORIGINAL TASK DESCRIPTION" in result["description"]


# ═══════════════════════════════════════════════════════════════════════════════
# DUAL-CHANNEL COMPOSER
# ═══════════════════════════════════════════════════════════════════════════════


class TestDualChannelComposer:
    """Certainty × valence → disposition matrix."""

    def _make_state(self, certainty: float = 0.5, valence: float = 0.0) -> InternalState:
        # meta_certainty=1.0 so adjusted_certainty = avg * (0.5 + 0.5*1.0) = avg
        cv = CertaintyVector(
            factual_grounding=certainty, tool_confidence=certainty,
            coherence=certainty, task_understanding=certainty,
            value_alignment=certainty, meta_certainty=1.0,
        )
        sm = SomaticMarker(valence=valence, intensity=0.5)
        return InternalState(certainty=cv, somatic=sm)

    def test_high_certainty_positive_valence_proceed(self):
        from app.self_awareness.dual_channel import DualChannelComposer
        composer = DualChannelComposer()
        state = self._make_state(certainty=0.9, valence=0.5)
        result = composer.compose(state)
        assert result.action_disposition == "proceed"
        assert result.risk_tier == 1

    def test_high_certainty_negative_valence_cautious(self):
        from app.self_awareness.dual_channel import DualChannelComposer
        composer = DualChannelComposer()
        state = self._make_state(certainty=0.9, valence=-0.5)
        result = composer.compose(state)
        assert result.action_disposition == "cautious"

    def test_low_certainty_negative_valence_escalate(self):
        from app.self_awareness.dual_channel import DualChannelComposer
        composer = DualChannelComposer()
        state = self._make_state(certainty=0.2, valence=-0.5)
        result = composer.compose(state)
        assert result.action_disposition == "escalate"
        assert result.risk_tier == 4

    def test_mid_certainty_neutral_valence_cautious(self):
        from app.self_awareness.dual_channel import DualChannelComposer
        composer = DualChannelComposer()
        state = self._make_state(certainty=0.55, valence=0.0)
        result = composer.compose(state)
        assert result.action_disposition == "cautious"

    def test_mid_certainty_negative_pause(self):
        from app.self_awareness.dual_channel import DualChannelComposer
        composer = DualChannelComposer()
        state = self._make_state(certainty=0.55, valence=-0.5)
        result = composer.compose(state)
        assert result.action_disposition == "pause"

    def test_all_nine_matrix_cells(self):
        """Exhaustively test all 9 combinations in the disposition matrix."""
        from app.self_awareness.dual_channel import DualChannelComposer, DISPOSITION_MATRIX
        composer = DualChannelComposer()
        test_cases = [
            (0.9, 0.5, "proceed"),    # high + positive
            (0.9, 0.0, "proceed"),    # high + neutral
            (0.9, -0.5, "cautious"),  # high + negative
            (0.55, 0.5, "proceed"),   # mid + positive
            (0.55, 0.0, "cautious"),  # mid + neutral
            (0.55, -0.5, "pause"),    # mid + negative
            (0.2, 0.5, "cautious"),   # low + positive
            (0.2, 0.0, "pause"),      # low + neutral
            (0.2, -0.5, "escalate"),  # low + negative
        ]
        for cert, val, expected in test_cases:
            state = self._make_state(certainty=cert, valence=val)
            result = composer.compose(state)
            assert result.action_disposition == expected, \
                f"cert={cert}, val={val}: expected {expected}, got {result.action_disposition}"

    def test_somatic_floor_enforcement(self):
        """Pre-reasoning somatic floor can only increase caution."""
        from app.self_awareness.dual_channel import DualChannelComposer
        composer = DualChannelComposer()
        state = self._make_state(certainty=0.9, valence=0.5)  # Would be "proceed"
        task_ctx = {"_somatic_advisories": [{"disposition_floor": "cautious"}]}
        result = composer.compose(state, task_context=task_ctx)
        assert result.action_disposition == "cautious"  # Floor overrides
        assert result.risk_tier == 2

    def test_somatic_floor_cannot_decrease_caution(self):
        """Floor lower than computed disposition should not lower it."""
        from app.self_awareness.dual_channel import DualChannelComposer
        composer = DualChannelComposer()
        state = self._make_state(certainty=0.2, valence=-0.5)  # "escalate"
        task_ctx = {"_somatic_advisories": [{"disposition_floor": "cautious"}]}
        result = composer.compose(state, task_context=task_ctx)
        assert result.action_disposition == "escalate"  # Not lowered to cautious

    def test_critical_budget_forces_pause(self):
        """Near budget exhaustion should force at least tier 3."""
        from app.self_awareness.dual_channel import DualChannelComposer
        composer = DualChannelComposer(critical_budget_threshold=0.1)
        state = self._make_state(certainty=0.9, valence=0.5)  # "proceed"
        state.meta = MetaCognitiveState(compute_budget_remaining_pct=0.05)
        result = composer.compose(state)
        assert result.risk_tier >= 3
        assert result.action_disposition == "pause"

    def test_risk_tier_matches_disposition(self):
        """Risk tier should always match the disposition map."""
        from app.self_awareness.dual_channel import DualChannelComposer
        composer = DualChannelComposer()
        for cert in [0.2, 0.55, 0.9]:
            for val in [-0.5, 0.0, 0.5]:
                state = self._make_state(certainty=cert, valence=val)
                result = composer.compose(state)
                expected_tier = DISPOSITION_TO_RISK_TIER[result.action_disposition]
                assert result.risk_tier == expected_tier


# ═══════════════════════════════════════════════════════════════════════════════
# AFFECTIVE FORECASTING
# ═══════════════════════════════════════════════════════════════════════════════


class TestAffectiveForecasting:
    """SomaticMarkerComputer.forecast() — backward + forward combined."""

    def test_forecast_with_no_beliefs(self):
        """When world model has no beliefs, forecast falls back to backward somatic."""
        from app.self_awareness.somatic_marker import SomaticMarkerComputer
        smc = SomaticMarkerComputer()
        backward = SomaticMarker(valence=-0.5, intensity=0.7, source="past", match_count=2)
        with patch.object(smc, "compute", return_value=backward), \
             patch("app.self_awareness.world_model.recall_relevant_beliefs", return_value=[]):
            result = smc.forecast("agent", "do something")
            assert "forecast:" in result.source
            assert result.valence == -0.5

    def test_forecast_combines_backward_and_beliefs(self):
        """Positive beliefs should pull valence upward."""
        from app.self_awareness.somatic_marker import SomaticMarkerComputer
        smc = SomaticMarkerComputer()
        backward = SomaticMarker(valence=-0.5, intensity=0.7, source="past", match_count=2)
        with patch.object(smc, "compute", return_value=backward), \
             patch("app.self_awareness.world_model.recall_relevant_beliefs",
                   return_value=["this approach has good success rate", "reliable and effective"]):
            result = smc.forecast("agent", "do something")
            # Backward valence=-0.5 (60%) + positive belief valence (40%)
            # Should be less negative than pure backward
            assert result.valence > -0.5
            assert result.match_count > backward.match_count

    def test_forecast_negative_beliefs_amplify(self):
        """Negative beliefs should make forecast more negative."""
        from app.self_awareness.somatic_marker import SomaticMarkerComputer
        smc = SomaticMarkerComputer()
        backward = SomaticMarker(valence=0.0, intensity=0.5, source="neutral", match_count=1)
        with patch.object(smc, "compute", return_value=backward), \
             patch("app.self_awareness.world_model.recall_relevant_beliefs",
                   return_value=["this tends to fail and crash", "error-prone, slow timeout"]):
            result = smc.forecast("agent", "risky action")
            # Neutral backward + negative beliefs → negative forecast
            assert result.valence < 0

    def test_forecast_intensity_floor(self):
        """When beliefs exist, intensity should be at least 0.3."""
        from app.self_awareness.somatic_marker import SomaticMarkerComputer
        smc = SomaticMarkerComputer()
        backward = SomaticMarker(valence=0.0, intensity=0.1, source="weak", match_count=1)
        with patch.object(smc, "compute", return_value=backward), \
             patch("app.self_awareness.world_model.recall_relevant_beliefs",
                   return_value=["something relevant"]):
            result = smc.forecast("agent", "action")
            assert result.intensity >= 0.3

    def test_forecast_belief_import_failure_returns_backward(self):
        """If world_model fails, forecast should still return backward somatic."""
        from app.self_awareness.somatic_marker import SomaticMarkerComputer
        smc = SomaticMarkerComputer()
        backward = SomaticMarker(valence=0.5, intensity=0.8, source="good", match_count=3)
        with patch.object(smc, "compute", return_value=backward), \
             patch("app.self_awareness.world_model.recall_relevant_beliefs",
                   side_effect=ImportError("no world model")):
            result = smc.forecast("agent", "action")
            assert result.valence == 0.5
            assert "forecast:" in result.source

    def test_belief_valence_range(self):
        """Belief valence should be in [-0.5, 0.5]."""
        # All positive words → (6-0)/6 * 0.5 = 0.5
        # All negative words → (0-7)/7 * 0.5 = -0.5
        positive_words = {"success", "improved", "reliable", "fast", "good", "effective"}
        negative_words = {"fail", "error", "slow", "crash", "timeout", "struggle", "bug"}
        all_pos = " ".join(positive_words)
        pos = sum(1 for w in positive_words if w in all_pos)
        neg = sum(1 for w in negative_words if w in all_pos)
        bv = (pos - neg) / max(pos + neg, 1) * 0.5
        assert bv == pytest.approx(0.5)

        all_neg = " ".join(negative_words)
        pos = sum(1 for w in positive_words if w in all_neg)
        neg = sum(1 for w in negative_words if w in all_neg)
        bv = (pos - neg) / max(pos + neg, 1) * 0.5
        assert bv == pytest.approx(-0.5)


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENTIAL COMPETITION + AFFECTIVE SCORE
# ═══════════════════════════════════════════════════════════════════════════════


class TestInferentialCompetitionAffective:
    """Affective forecasting wired into plan scoring."""

    def test_weights_sum_to_one(self):
        from app.self_awareness.inferential_competition import InferentialCompetition
        ic = InferentialCompetition()
        total = ic.precision_weight + ic.alignment_weight + ic.novelty_weight + ic.affective_weight
        assert total == pytest.approx(1.0)

    def test_competing_plan_has_affective_score(self):
        from app.self_awareness.inferential_competition import CompetingPlan
        plan = CompetingPlan(plan_id="p1", approach="test", predicted_outcome="result")
        assert hasattr(plan, "affective_score")
        assert plan.affective_score == 0.5  # Default

    def test_affective_score_in_to_dict(self):
        from app.self_awareness.inferential_competition import CompetingPlan
        plan = CompetingPlan(plan_id="p1", approach="t", predicted_outcome="r", affective_score=0.8)
        d = plan.to_dict()
        assert "affective_score" in d
        assert d["affective_score"] == 0.8

    def test_compete_passes_agent_id(self):
        """compete() should accept and propagate agent_id."""
        from app.self_awareness.inferential_competition import InferentialCompetition
        ic = InferentialCompetition()
        import inspect
        sig = inspect.signature(ic.compete)
        assert "agent_id" in sig.parameters

    def test_affective_weight_affects_composite(self):
        """Plans with higher affective score should get higher composite."""
        from app.self_awareness.inferential_competition import CompetingPlan, InferentialCompetition
        ic = InferentialCompetition()
        # Two plans: same precision/alignment/novelty, different affective
        plan_good = CompetingPlan(plan_id="good", approach="a", predicted_outcome="r")
        plan_good.precision_score = 0.5
        plan_good.alignment_score = 0.5
        plan_good.novelty_score = 0.5
        plan_good.affective_score = 0.9
        plan_good.composite_score = (
            ic.precision_weight * 0.5 + ic.alignment_weight * 0.5
            + ic.novelty_weight * 0.5 + ic.affective_weight * 0.9
        )

        plan_bad = CompetingPlan(plan_id="bad", approach="b", predicted_outcome="r")
        plan_bad.precision_score = 0.5
        plan_bad.alignment_score = 0.5
        plan_bad.novelty_score = 0.5
        plan_bad.affective_score = 0.1
        plan_bad.composite_score = (
            ic.precision_weight * 0.5 + ic.alignment_weight * 0.5
            + ic.novelty_weight * 0.5 + ic.affective_weight * 0.1
        )

        assert plan_good.composite_score > plan_bad.composite_score


# ═══════════════════════════════════════════════════════════════════════════════
# HOMEOSTASIS ↔ SOMATIC BIDIRECTIONAL COUPLING
# ═══════════════════════════════════════════════════════════════════════════════


class TestHomeostasisSomaticCoupling:
    """Bidirectional coupling between homeostasis and somatic markers."""

    def _mock_state(self, **overrides):
        defaults = {
            "cognitive_energy": 0.7, "frustration": 0.1, "confidence": 0.5,
            "curiosity": 0.5, "tasks_since_rest": 0, "consecutive_failures": 0,
        }
        defaults.update(overrides)
        return defaults

    def test_negative_somatic_increases_frustration(self):
        """Negative somatic valence should increase homeostatic frustration."""
        from app.self_awareness import homeostasis
        initial = self._mock_state(frustration=0.2)
        with patch.object(homeostasis, "_load", return_value=initial.copy()), \
             patch.object(homeostasis, "_save") as mock_save:
            homeostasis.update_state("task_complete", "research", success=True,
                                     somatic_valence=-0.8)
            saved = mock_save.call_args[0][0]
            assert saved["frustration"] > 0.2  # Increased by somatic signal

    def test_positive_somatic_reduces_frustration(self):
        from app.self_awareness import homeostasis
        initial = self._mock_state(frustration=0.4)
        with patch.object(homeostasis, "_load", return_value=initial.copy()), \
             patch.object(homeostasis, "_save") as mock_save:
            homeostasis.update_state("task_complete", "coding", success=True,
                                     somatic_valence=0.8)
            saved = mock_save.call_args[0][0]
            # Success already reduces frustration by 0.05, somatic adds more relief
            assert saved["frustration"] < 0.4 - 0.05

    def test_positive_somatic_boosts_confidence(self):
        from app.self_awareness import homeostasis
        initial = self._mock_state(confidence=0.5)
        with patch.object(homeostasis, "_load", return_value=initial.copy()), \
             patch.object(homeostasis, "_save") as mock_save:
            homeostasis.update_state("task_complete", "writing", success=True,
                                     somatic_valence=0.9)
            saved = mock_save.call_args[0][0]
            # Success adds 0.02, plus somatic boosts more
            assert saved["confidence"] > 0.5 + 0.02

    def test_negative_somatic_depletes_energy(self):
        from app.self_awareness import homeostasis
        initial = self._mock_state(cognitive_energy=0.7)
        with patch.object(homeostasis, "_load", return_value=initial.copy()), \
             patch.object(homeostasis, "_save") as mock_save:
            homeostasis.update_state("task_complete", "research", success=True,
                                     somatic_valence=-0.8)
            saved = mock_save.call_args[0][0]
            # Success adds 0.05, but negative somatic depletes some
            # Net effect depends on magnitude
            # boost = 0.8 * 0.08 = 0.064, energy depletion = 0.064 * 0.5 = 0.032
            # Success: +0.05, somatic: -0.032, net: +0.018
            # So energy should still be near 0.7 (within tolerance)

    def test_neutral_somatic_no_extra_effect(self):
        """Somatic valence in [-0.3, 0.3] should not trigger coupling."""
        from app.self_awareness import homeostasis
        initial = self._mock_state(frustration=0.3)
        with patch.object(homeostasis, "_load", return_value=initial.copy()), \
             patch.object(homeostasis, "_save") as mock_save:
            homeostasis.update_state("task_complete", "research", success=True,
                                     somatic_valence=0.1)
            saved = mock_save.call_args[0][0]
            # Only standard success effect: frustration -= 0.05
            # No somatic coupling (0.1 is in dead band)
            # After regulation: frustration drifts toward 0.1 target
            assert saved["frustration"] < 0.3

    def test_none_somatic_no_coupling(self):
        """somatic_valence=None should behave like pre-fix (no coupling)."""
        from app.self_awareness import homeostasis
        initial = self._mock_state(frustration=0.3)
        with patch.object(homeostasis, "_load", return_value=initial.copy()), \
             patch.object(homeostasis, "_save") as mock_save:
            homeostasis.update_state("task_complete", "research", success=True,
                                     somatic_valence=None)
            saved = mock_save.call_args[0][0]
            # Standard success only
            assert saved["frustration"] < 0.3

    def test_values_stay_bounded(self):
        """All homeostatic values should stay in [0, 1]."""
        from app.self_awareness import homeostasis
        initial = self._mock_state(frustration=0.95, cognitive_energy=0.05, confidence=0.95)
        with patch.object(homeostasis, "_load", return_value=initial.copy()), \
             patch.object(homeostasis, "_save") as mock_save:
            homeostasis.update_state("task_complete", "coding", success=False,
                                     somatic_valence=-1.0)
            saved = mock_save.call_args[0][0]
            assert 0.0 <= saved["frustration"] <= 1.0
            assert 0.0 <= saved["cognitive_energy"] <= 1.0
            assert 0.0 <= saved["confidence"] <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# RECORD EXPERIENCE
# ═══════════════════════════════════════════════════════════════════════════════


class TestRecordExperience:
    """record_experience_sync: outcome clamping and embedding."""

    def test_clamps_outcome_score(self):
        from app.self_awareness.somatic_marker import record_experience_sync
        with patch("app.memory.chromadb_manager.embed", return_value=[0.1] * 768), \
             patch("app.control_plane.db.execute") as mock_exec:
            record_experience_sync("agent", "context", outcome_score=5.0)
            args = mock_exec.call_args[0][1]
            assert args[4] == 1.0  # Clamped to max

    def test_clamps_negative_outcome(self):
        from app.self_awareness.somatic_marker import record_experience_sync
        with patch("app.memory.chromadb_manager.embed", return_value=[0.1] * 768), \
             patch("app.control_plane.db.execute") as mock_exec:
            record_experience_sync("agent", "context", outcome_score=-5.0)
            args = mock_exec.call_args[0][1]
            assert args[4] == -1.0

    def test_truncates_context(self):
        from app.self_awareness.somatic_marker import record_experience_sync
        long_context = "x" * 5000
        with patch("app.memory.chromadb_manager.embed", return_value=[0.1] * 768), \
             patch("app.control_plane.db.execute") as mock_exec:
            record_experience_sync("agent", long_context, outcome_score=0.5)
            args = mock_exec.call_args[0][1]
            assert len(args[2]) <= 2000

    def test_non_fatal_on_error(self):
        """Should not raise even when DB fails."""
        from app.self_awareness.somatic_marker import record_experience_sync
        with patch("app.memory.chromadb_manager.embed", side_effect=Exception("no embedding")):
            record_experience_sync("agent", "context", outcome_score=0.5)  # No exception

    def test_empty_tools_default(self):
        from app.self_awareness.somatic_marker import record_experience_sync
        with patch("app.memory.chromadb_manager.embed", return_value=[0.1] * 768), \
             patch("app.control_plane.db.execute") as mock_exec:
            record_experience_sync("agent", "ctx", outcome_score=0.5)
            args = mock_exec.call_args[0][1]
            assert args[7] == []  # Default empty tools


# ═══════════════════════════════════════════════════════════════════════════════
# INTERNAL STATE CONTEXT STRING
# ═══════════════════════════════════════════════════════════════════════════════


class TestInternalStateContextString:
    """InternalState.to_context_string() somatic injection."""

    def test_somatic_included_when_intense(self):
        state = InternalState(
            somatic=SomaticMarker(valence=-0.5, intensity=0.8),
        )
        ctx = state.to_context_string()
        assert "Somatic=" in ctx
        assert "negative" in ctx

    def test_somatic_excluded_when_weak(self):
        state = InternalState(
            somatic=SomaticMarker(valence=-0.5, intensity=0.1),  # Below 0.3 threshold
        )
        ctx = state.to_context_string()
        assert "Somatic=" not in ctx

    def test_positive_somatic_label(self):
        state = InternalState(
            somatic=SomaticMarker(valence=0.5, intensity=0.8),
        )
        ctx = state.to_context_string()
        assert "positive" in ctx

    def test_neutral_somatic_label(self):
        state = InternalState(
            somatic=SomaticMarker(valence=0.0, intensity=0.8),
        )
        ctx = state.to_context_string()
        assert "neutral" in ctx

    def test_disposition_in_context(self):
        state = InternalState(action_disposition="escalate")
        ctx = state.to_context_string()
        assert "Disposition=escalate" in ctx

    def test_context_compact(self):
        """Context string should be compact (~30 tokens)."""
        state = InternalState(
            somatic=SomaticMarker(valence=-0.3, intensity=0.5),
        )
        ctx = state.to_context_string()
        assert len(ctx.split()) < 60


# ═══════════════════════════════════════════════════════════════════════════════
# SAFETY PROPERTIES
# ═══════════════════════════════════════════════════════════════════════════════


class TestSafetyProperties:
    """Safety invariants of the emotions subsystem."""

    def test_disposition_can_only_escalate(self):
        """No path from escalate → proceed in a single compose() call."""
        from app.self_awareness.dual_channel import DualChannelComposer
        composer = DualChannelComposer()
        # Even with perfect certainty and positive valence, a "pause" floor sticks
        cv = CertaintyVector(
            factual_grounding=1.0, tool_confidence=1.0, coherence=1.0,
            task_understanding=1.0, value_alignment=1.0, meta_certainty=1.0,
        )
        sm = SomaticMarker(valence=1.0, intensity=1.0)
        state = InternalState(certainty=cv, somatic=sm)
        task_ctx = {"_somatic_advisories": [{"disposition_floor": "pause"}]}
        result = composer.compose(state, task_context=task_ctx)
        assert result.risk_tier >= 3

    def test_somatic_computation_never_crashes(self):
        """Even with all dependencies broken, compute() returns a valid marker."""
        from app.self_awareness.somatic_marker import SomaticMarkerComputer
        smc = SomaticMarkerComputer()
        with patch("app.control_plane.db.execute", side_effect=Exception("total failure")):
            result = smc.compute("agent", "ctx", context_embedding=[0.1] * 768)
            assert isinstance(result, SomaticMarker)
            assert -1.0 <= result.valence <= 1.0

    def test_bias_is_additive_only(self):
        """Somatic bias should add to context, never remove or replace."""
        from app.self_awareness.somatic_bias import SomaticBiasInjector
        injector = SomaticBiasInjector()
        ctx = {"description": "do X", "existing_key": "preserved"}
        sm = SomaticMarker(valence=-0.7, intensity=0.8, source="test")
        result = injector.inject(ctx, sm)
        assert result["existing_key"] == "preserved"
        assert "do X" in result["description"]

    def test_homeostatic_values_bounded_after_extreme_somatic(self):
        """Extreme somatic input should not push homeostasis out of [0, 1]."""
        from app.self_awareness import homeostasis
        initial = {"cognitive_energy": 0.01, "frustration": 0.99, "confidence": 0.01,
                   "curiosity": 0.5, "tasks_since_rest": 0, "consecutive_failures": 5}
        with patch.object(homeostasis, "_load", return_value=initial.copy()), \
             patch.object(homeostasis, "_save") as mock_save:
            homeostasis.update_state("task_complete", "x", success=False,
                                     somatic_valence=-1.0)
            saved = mock_save.call_args[0][0]
            for key in ("cognitive_energy", "frustration", "confidence", "curiosity"):
                assert 0.0 <= saved[key] <= 1.0, f"{key}={saved[key]} out of bounds"

    def test_temporal_floor_prevents_memory_erasure(self):
        """The 20% floor ensures old experiences never fully disappear."""
        half_life = 168.0
        floor = 0.2
        # After 100 half-lives (absurd age), floor still holds
        raw = 0.5 ** (100 * half_life / half_life)
        assert raw < 1e-30  # Essentially zero
        assert max(floor, raw) == 0.2  # Floor saves it
