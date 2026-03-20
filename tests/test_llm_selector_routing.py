"""Tests for app/llm_selector.py — difficulty_to_tier and detect_task_type."""
import unittest

from app.llm_selector import difficulty_to_tier, detect_task_type


class TestDifficultyToTier(unittest.TestCase):
    """Verify difficulty-based model tier routing."""

    def test_easy_hybrid_returns_local(self):
        assert difficulty_to_tier(1, "hybrid") == "local"
        assert difficulty_to_tier(2, "hybrid") == "local"
        assert difficulty_to_tier(3, "hybrid") == "local"

    def test_easy_local_returns_local(self):
        assert difficulty_to_tier(1, "local") == "local"

    def test_easy_cloud_returns_budget(self):
        assert difficulty_to_tier(1, "cloud") == "budget"
        assert difficulty_to_tier(3, "cloud") == "budget"

    def test_medium_returns_none(self):
        for d in range(4, 8):
            result = difficulty_to_tier(d, "hybrid")
            assert result is None, f"difficulty {d} should return None, got {result}"

    def test_hard_returns_premium(self):
        assert difficulty_to_tier(8, "hybrid") == "premium"
        assert difficulty_to_tier(9, "cloud") == "premium"
        assert difficulty_to_tier(10, "local") == "premium"

    def test_boundary_values(self):
        assert difficulty_to_tier(3, "hybrid") == "local"   # last easy
        assert difficulty_to_tier(4, "hybrid") is None       # first medium
        assert difficulty_to_tier(7, "hybrid") is None       # last medium
        assert difficulty_to_tier(8, "hybrid") == "premium"  # first hard


class TestDetectTaskType(unittest.TestCase):
    """Verify keyword-based task type detection."""

    def test_coding_keywords(self):
        assert detect_task_type("research", "implement a new function") == "coding"
        assert detect_task_type("research", "write a Python class") == "coding"

    def test_research_keywords(self):
        assert detect_task_type("coding", "research current best practices") == "research"
        assert detect_task_type("coding", "investigate the bug root cause") == "research"

    def test_debug_keywords(self):
        assert detect_task_type("research", "fix bug in traceback") == "debugging"

    def test_architecture_keywords(self):
        assert detect_task_type("research", "design a system architecture") == "architecture"

    def test_writing_keywords(self):
        assert detect_task_type("coding", "write a summary report") == "writing"

    def test_multimodal_keywords(self):
        assert detect_task_type("research", "look at this screenshot") == "multimodal"

    def test_no_hint_falls_back_to_role(self):
        assert detect_task_type("coding", "") == "coding"
        assert detect_task_type("research", "") == "research"
        assert detect_task_type("writing", "") == "writing"
        assert detect_task_type("critic", "") == "reasoning"

    def test_unknown_role_returns_general(self):
        assert detect_task_type("foobar", "") == "general"


if __name__ == "__main__":
    unittest.main()
