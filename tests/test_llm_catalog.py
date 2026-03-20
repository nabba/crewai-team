"""Tests for app/llm_catalog.py — model catalog structure and public API."""
import unittest

from app.llm_catalog import (
    CATALOG, ROLE_DEFAULTS, TASK_ALIASES,
    get_model, get_model_id, get_tier, get_provider,
    is_multimodal, get_default_for_role, get_candidates,
    get_candidates_by_tier, get_smallest_model, get_ram_requirement,
    estimate_task_cost, format_catalog, format_role_assignments,
)


class TestCatalogStructure(unittest.TestCase):
    """Verify every catalog entry has required fields."""

    REQUIRED_FIELDS = {
        "tier", "provider", "model_id", "context",
        "cost_input_per_m", "cost_output_per_m",
        "tool_use_reliability", "strengths",
    }

    def test_all_models_have_required_fields(self):
        for name, info in CATALOG.items():
            for field in self.REQUIRED_FIELDS:
                assert field in info, f"{name} missing field: {field}"

    def test_tiers_are_valid(self):
        valid_tiers = {"local", "budget", "mid", "premium"}
        for name, info in CATALOG.items():
            assert info["tier"] in valid_tiers, f"{name} has invalid tier: {info['tier']}"

    def test_providers_are_valid(self):
        valid_providers = {"ollama", "openrouter", "anthropic"}
        for name, info in CATALOG.items():
            assert info["provider"] in valid_providers, f"{name} has invalid provider"

    def test_costs_non_negative(self):
        for name, info in CATALOG.items():
            assert info["cost_input_per_m"] >= 0, f"{name} has negative input cost"
            assert info["cost_output_per_m"] >= 0, f"{name} has negative output cost"

    def test_local_models_are_free(self):
        for name, info in CATALOG.items():
            if info["tier"] == "local":
                assert info["cost_input_per_m"] == 0.0
                assert info["cost_output_per_m"] == 0.0

    def test_tool_use_reliability_in_range(self):
        for name, info in CATALOG.items():
            r = info["tool_use_reliability"]
            assert 0.0 <= r <= 1.0, f"{name} reliability out of range: {r}"

    def test_strengths_scores_in_range(self):
        for name, info in CATALOG.items():
            for task, score in info["strengths"].items():
                assert 0.0 <= score <= 1.0, f"{name} strength {task}={score} out of range"

    def test_minimum_model_count(self):
        """Should have at least one model per tier."""
        tiers_present = {info["tier"] for info in CATALOG.values()}
        assert "local" in tiers_present
        assert "budget" in tiers_present
        assert "premium" in tiers_present


class TestRoleDefaults(unittest.TestCase):
    """Verify role → model default assignments are valid."""

    def test_all_cost_modes_present(self):
        for mode in ("budget", "balanced", "quality"):
            assert mode in ROLE_DEFAULTS, f"Missing cost mode: {mode}"

    def test_all_default_models_exist_in_catalog(self):
        for mode, defaults in ROLE_DEFAULTS.items():
            for role, model in defaults.items():
                assert model in CATALOG, f"{mode}/{role} → {model} not in CATALOG"

    def test_commander_always_premium(self):
        for mode in ROLE_DEFAULTS:
            model = ROLE_DEFAULTS[mode]["commander"]
            assert CATALOG[model]["tier"] == "premium", (
                f"Commander in {mode} should be premium, got {CATALOG[model]['tier']}"
            )


class TestPublicAPI(unittest.TestCase):
    """Test catalog lookup functions."""

    def test_get_model_existing(self):
        entry = get_model("claude-sonnet-4.6")
        assert entry is not None
        assert entry["tier"] == "premium"

    def test_get_model_nonexistent(self):
        assert get_model("nonexistent-model") is None

    def test_get_model_id(self):
        mid = get_model_id("claude-sonnet-4.6")
        assert "claude" in mid

    def test_get_model_id_missing_raises(self):
        with self.assertRaises(KeyError):
            get_model_id("nonexistent")

    def test_get_tier(self):
        assert get_tier("qwen3:30b-a3b") == "local"
        assert get_tier("deepseek-v3.2") == "budget"
        assert get_tier("nonexistent") == "unknown"

    def test_get_provider(self):
        assert get_provider("claude-sonnet-4.6") == "anthropic"
        assert get_provider("deepseek-v3.2") == "openrouter"
        assert get_provider("qwen3:30b-a3b") == "ollama"

    def test_is_multimodal(self):
        assert is_multimodal("kimi-k2.5") is True
        assert is_multimodal("deepseek-v3.2") is False
        assert is_multimodal("nonexistent") is False

    def test_get_default_for_role(self):
        model = get_default_for_role("research", "balanced")
        assert model in CATALOG

    def test_get_candidates_returns_sorted(self):
        candidates = get_candidates("coding")
        assert len(candidates) > 0
        scores = [s for _, s in candidates]
        assert scores == sorted(scores, reverse=True)

    def test_get_candidates_by_tier(self):
        local_only = get_candidates_by_tier("coding", ["local"])
        for name, _ in local_only:
            assert CATALOG[name]["tier"] == "local"

    def test_get_smallest_model(self):
        smallest = get_smallest_model()
        assert smallest in CATALOG

    def test_get_ram_requirement(self):
        ram = get_ram_requirement("qwen3:30b-a3b")
        assert ram > 0
        assert get_ram_requirement("nonexistent") == 20.0

    def test_estimate_task_cost_local_is_free(self):
        cost = estimate_task_cost("qwen3:30b-a3b", 1000, 1000)
        assert cost == 0.0

    def test_estimate_task_cost_api_positive(self):
        cost = estimate_task_cost("claude-sonnet-4.6", 1_000_000, 1_000_000)
        assert cost > 0

    def test_format_catalog_contains_tiers(self):
        text = format_catalog()
        assert "[LOCAL]" in text
        assert "[PREMIUM]" in text

    def test_format_role_assignments(self):
        text = format_role_assignments("balanced")
        assert "commander" in text
        assert "research" in text


class TestTaskAliases(unittest.TestCase):
    """Verify task aliases map to valid task types."""

    def test_aliases_are_strings(self):
        for alias, task_type in TASK_ALIASES.items():
            assert isinstance(alias, str)
            assert isinstance(task_type, str)

    def test_common_aliases_present(self):
        assert "code" in TASK_ALIASES
        assert "debug" in TASK_ALIASES
        assert "write" in TASK_ALIASES


if __name__ == "__main__":
    unittest.main()
