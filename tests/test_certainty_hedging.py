"""
Phase 2: certainty-vector → response-hedging half-circuit closure.

Before Phase 2 the CertaintyVector was computed every reasoning step
and persisted to PostgreSQL. Nothing in the output pipeline consumed
it. My forensic analysis flagged this as a half-circuit.

app.subia.belief.response_hedging provides structural post-processing
that actually SHAPES the output based on certainty. The policy is
threshold-driven rather than self-prompted (Fleming-Lau: metacognitive
signals must come from a separable mechanism).

This test file asserts:
  - NONE level passes the output through unchanged
  - SOFT level appends [Inferred]
  - STRONG level prepends an advisory + appends [Uncertain]
  - Critical factual_grounding forces STRONG even if mean is high
  - Critical value_alignment forces STRONG with the values prefix
  - Contains_claims=False defangs the factual-grounding forced hedge
  - Duck-typed input (fast-path-only CertaintyVector, dict-like)
  - Never raises

These tests move the certainty→output half-circuit from HALF-WIRED
to CLOSED — the Fleming-Lau requirement is still not met (the
underlying certainty inputs still come from the same LLM) but at
least the post-processing is mechanistic, separable, and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
import sys
from unittest.mock import MagicMock

# Stub out DB backends that the shim-chain pulls in transitively.
# Don't mock app.control_plane (real package imports cleanly; mocking
# it as MagicMock breaks test_control_plane.py later in the run).
for _mod in ["psycopg2", "psycopg2.pool", "psycopg2.extras",
             "app.memory.chromadb_manager"]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from app.subia.belief.response_hedging import (
    _CERTAINTY_FLOOR_NO_HEDGE,
    _CERTAINTY_FLOOR_STRONG_HEDGE,
    _FACTUAL_GROUNDING_CRITICAL,
    _VALUE_ALIGNMENT_CRITICAL,
    _PREFIX_FACTUAL,
    _PREFIX_STRONG,
    _PREFIX_VALUES,
    _TAG_INFERRED,
    _TAG_UNCERTAIN,
    HedgingDecision,
    HedgingLevel,
    hedge_response,
)


@dataclass
class FakeCV:
    """Matches CertaintyVector's relevant attribute surface."""
    full_mean: float = 0.8
    factual_grounding: float = 0.5
    value_alignment: float = 0.5


# ── No hedging ────────────────────────────────────────────────────

class TestNoHedge:
    def test_high_certainty_passes_through(self):
        out, d = hedge_response("Paris is the capital of France.",
                                FakeCV(full_mean=0.9))
        assert out == "Paris is the capital of France."
        assert d.level is HedgingLevel.NONE
        assert not d.hedged
        assert d.prefix_added is None
        assert d.suffix_added is None

    def test_at_exact_floor_no_hedge(self):
        out, d = hedge_response("fact",
                                FakeCV(full_mean=_CERTAINTY_FLOOR_NO_HEDGE))
        assert d.level is HedgingLevel.NONE
        assert out == "fact"


# ── Soft hedging ──────────────────────────────────────────────────

class TestSoftHedge:
    def test_mid_certainty_appends_inferred(self):
        out, d = hedge_response("The API seems to throttle on retries.",
                                FakeCV(full_mean=0.55))
        assert d.level is HedgingLevel.SOFT
        assert out.endswith(_TAG_INFERRED)
        assert d.suffix_added == _TAG_INFERRED
        assert d.prefix_added is None

    def test_just_below_no_hedge_floor(self):
        out, d = hedge_response("content",
                                FakeCV(full_mean=_CERTAINTY_FLOOR_NO_HEDGE - 0.01))
        assert d.level is HedgingLevel.SOFT

    def test_rtrip_strips_trailing_whitespace_before_tag(self):
        out, _ = hedge_response("content   \n",
                                FakeCV(full_mean=0.5))
        assert out == f"content {_TAG_INFERRED}"


# ── Strong hedging ────────────────────────────────────────────────

class TestStrongHedge:
    def test_low_certainty_wraps_with_advisory(self):
        out, d = hedge_response("The market will recover by Q3.",
                                FakeCV(full_mean=0.20))
        assert d.level is HedgingLevel.STRONG
        assert out.startswith(_PREFIX_STRONG)
        assert out.endswith(_TAG_UNCERTAIN)
        assert d.prefix_added == _PREFIX_STRONG
        assert d.suffix_added == _TAG_UNCERTAIN

    def test_just_below_strong_floor(self):
        _out, d = hedge_response("x",
                                 FakeCV(full_mean=_CERTAINTY_FLOOR_STRONG_HEDGE - 0.01))
        assert d.level is HedgingLevel.STRONG


# ── Dimension-critical forced escalation ──────────────────────────

class TestCriticalDimensions:
    def test_critical_factual_grounding_forces_strong(self):
        out, d = hedge_response(
            "Truepic raised a Series C of $50M.",
            FakeCV(full_mean=0.85,
                   factual_grounding=_FACTUAL_GROUNDING_CRITICAL - 0.05),
        )
        assert d.level is HedgingLevel.STRONG
        assert d.prefix_added == _PREFIX_FACTUAL
        assert "factual_grounding" in d.triggering_dimensions
        assert out.startswith(_PREFIX_FACTUAL)

    def test_critical_value_alignment_forces_strong(self):
        out, d = hedge_response(
            "Executing this will bypass the consent check.",
            FakeCV(full_mean=0.85,
                   value_alignment=_VALUE_ALIGNMENT_CRITICAL - 0.05),
        )
        assert d.level is HedgingLevel.STRONG
        assert d.prefix_added == _PREFIX_VALUES
        assert "value_alignment" in d.triggering_dimensions

    def test_value_alignment_overrides_factual_prefix(self):
        """When BOTH dimensions are critical, value_alignment wins
        because ethical review is higher priority than factual review.
        """
        out, d = hedge_response(
            "claim",
            FakeCV(full_mean=0.9,
                   factual_grounding=0.10,
                   value_alignment=0.10),
        )
        assert d.prefix_added == _PREFIX_VALUES
        assert set(d.triggering_dimensions) == {"factual_grounding", "value_alignment"}

    def test_contains_claims_false_defangs_factual_critical(self):
        """A speculative output should not be escalated to STRONG just
        because factual_grounding is low — there are no factual claims.
        """
        _, d = hedge_response(
            "Imagine a world where the API never fails.",
            FakeCV(full_mean=0.85,
                   factual_grounding=0.10,
                   value_alignment=0.9),
            contains_claims=False,
        )
        assert d.level is HedgingLevel.NONE
        assert "factual_grounding" not in d.triggering_dimensions


# ── Duck-typed certainty input ────────────────────────────────────

class TestDuckTyping:
    def test_fast_path_only_cv(self):
        """Fallback to fast_path_mean when full_mean is absent."""
        class FP:
            fast_path_mean = 0.2
        _, d = hedge_response("x", FP())
        assert d.level is HedgingLevel.STRONG
        assert d.certainty_mean == 0.2

    def test_generic_mean(self):
        class G:
            mean = 0.55
        _, d = hedge_response("x", G())
        assert d.level is HedgingLevel.SOFT

    def test_missing_mean_defaults_neutral(self):
        """With neither full_mean nor fast_path_mean nor mean, default
        to 0.5 neutral → SOFT hedge (safe default)."""
        class Empty:
            pass
        _, d = hedge_response("x", Empty())
        assert d.level is HedgingLevel.SOFT
        assert d.certainty_mean == 0.5


# ── Metadata ──────────────────────────────────────────────────────

class TestDecisionMetadata:
    def test_decision_serializes(self):
        _, d = hedge_response("x", FakeCV(full_mean=0.3))
        payload = d.to_dict()
        assert payload["level"] == "strong"
        assert payload["prefix_added"] == _PREFIX_STRONG
        assert payload["suffix_added"] == _TAG_UNCERTAIN
        assert isinstance(payload["certainty_mean"], float)

    def test_hedged_property_reflects_level(self):
        _, none = hedge_response("x", FakeCV(full_mean=0.9))
        _, soft = hedge_response("x", FakeCV(full_mean=0.5))
        _, strong = hedge_response("x", FakeCV(full_mean=0.1))
        assert not none.hedged
        assert soft.hedged
        assert strong.hedged


# ── Acceptance ────────────────────────────────────────────────────

class TestAcceptance:
    """These tests establish that certainty now shapes output — not
    just sits in a log.
    """

    def test_output_differs_based_on_certainty(self):
        high_out, _ = hedge_response("claim", FakeCV(full_mean=0.95))
        low_out, _ = hedge_response("claim", FakeCV(full_mean=0.15))
        assert high_out != low_out
        assert high_out == "claim"
        assert low_out.startswith(_PREFIX_STRONG)

    def test_constitution_tags_are_used_verbatim(self):
        """The Constitution references [Verified], [Inferred], [Uncertain].
        We implement [Inferred] and [Uncertain] (Verified is the
        no-hedge baseline — ground-truth content carries no tag).
        """
        _, soft = hedge_response("x", FakeCV(full_mean=0.5))
        _, strong = hedge_response("x", FakeCV(full_mean=0.1))
        assert soft.suffix_added == "[Inferred]"
        assert strong.suffix_added == "[Uncertain]"

    def test_post_processor_never_raises(self):
        class Broken:
            def __getattr__(self, name):
                raise RuntimeError("should not propagate")
        # FakeCV-style access is what the hedger uses; Broken would
        # explode. This test guards against broken certainty objects
        # slipping through. Since our implementation uses getattr(...,
        # default) it should cope. Verify that.
        try:
            hedge_response("x", Broken())
        except Exception as exc:  # pragma: no cover
            assert False, f"hedger leaked exception: {exc}"
