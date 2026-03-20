"""Tests for app/vetting.py — risk-based selective verification."""
import unittest
from unittest.mock import patch

from app.vetting import assess_risk_level, _verify_schema, _FAILURE_PATTERNS


class TestAssessRiskLevel(unittest.TestCase):
    """Test the risk matrix: crew_name × difficulty × model_tier → risk level."""

    # ── "none" tier ──────────────────────────────────────────────────────────

    def test_direct_crew_always_none(self):
        assert assess_risk_level("direct", 10, "local") == "none"
        assert assess_risk_level("direct", 1, "premium") == "none"

    def test_premium_easy_is_none(self):
        assert assess_risk_level("research", 2, "premium") == "none"
        assert assess_risk_level("writing", 3, "premium") == "none"

    # ── "full" tier ──────────────────────────────────────────────────────────

    def test_coding_always_full(self):
        assert assess_risk_level("coding", 1, "premium") == "full"
        assert assess_risk_level("coding", 5, "budget") == "full"
        assert assess_risk_level("coding", 3, "local") == "full"

    def test_local_tier_always_full(self):
        assert assess_risk_level("research", 3, "local") == "full"
        assert assess_risk_level("writing", 5, "local") == "full"

    def test_high_difficulty_always_full(self):
        assert assess_risk_level("research", 8, "premium") == "full"
        assert assess_risk_level("writing", 9, "mid") == "full"
        assert assess_risk_level("research", 10, "budget") == "full"

    # ── "schema" tier ────────────────────────────────────────────────────────

    def test_budget_easy_writing_is_schema(self):
        assert assess_risk_level("writing", 4, "budget") == "schema"
        assert assess_risk_level("research", 5, "mid") == "schema"

    def test_premium_moderate_is_schema(self):
        assert assess_risk_level("research", 5, "premium") == "schema"
        assert assess_risk_level("writing", 6, "premium") == "schema"

    # ── "cheap" tier ─────────────────────────────────────────────────────────

    def test_budget_moderate_non_writing_is_cheap(self):
        # difficulty 6, budget tier, not writing/research → cheap
        assert assess_risk_level("self_improvement", 6, "budget") == "cheap"
        assert assess_risk_level("self_improvement", 6, "mid") == "cheap"

    # ── Default fallthrough to full ──────────────────────────────────────────

    def test_unknown_tier_defaults_full(self):
        assert assess_risk_level("research", 7, "unknown") == "full"

    def test_high_difficulty_budget_is_full(self):
        assert assess_risk_level("research", 8, "budget") == "full"


class TestVerifySchema(unittest.TestCase):
    """Test the schema (no-LLM) verification step."""

    def test_passes_normal_response(self):
        passed, result = _verify_schema("This is a well-formed research report about AI.", "research")
        assert passed is True
        assert result == "This is a well-formed research report about AI."

    def test_fails_too_short(self):
        passed, _ = _verify_schema("Hi", "research")
        assert passed is False

    def test_fails_empty(self):
        passed, _ = _verify_schema("", "research")
        assert passed is False
        passed2, _ = _verify_schema("   ", "writing")
        assert passed2 is False

    def test_fails_refusal_patterns(self):
        refusals = [
            "I cannot help with that request.",
            "Sorry, I am unable to complete this task.",
            "As an AI language model, I cannot do that.",
            "I can't provide that information.",
        ]
        for refusal in refusals:
            passed, _ = _verify_schema(refusal, "research")
            assert passed is False, f"Should have failed: {refusal!r}"

    def test_passes_long_response_with_warning(self):
        """Long responses still pass (just log a warning)."""
        long_text = "A" * 5000
        passed, result = _verify_schema(long_text, "writing")
        assert passed is True

    def test_non_writing_crew_no_length_issue(self):
        long_text = "B" * 5000
        passed, _ = _verify_schema(long_text, "coding")
        assert passed is True


class TestFailurePatterns(unittest.TestCase):
    """Verify the regex patterns match expected refusal strings."""

    def test_patterns_exist(self):
        assert len(_FAILURE_PATTERNS) >= 4

    def test_does_not_false_positive_on_normal_text(self):
        normal = [
            "The research shows that...",
            "Here are the key findings:",
            "Python code to implement...",
            "According to the sources,",
        ]
        for text in normal:
            for pattern in _FAILURE_PATTERNS:
                assert not pattern.match(text), f"False positive on: {text!r}"


if __name__ == "__main__":
    unittest.main()
