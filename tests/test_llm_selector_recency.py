"""
Knowledge-cutoff filter tests
=============================

Covers ``min_recency`` plumbing across:
  - ``_below_min_recency`` / ``_find_recency_compliant`` helpers
  - ``select_model(..., min_recency=...)`` integration
  - ``create_specialist_llm`` role-keyed auto-default

Run:
    .venv/bin/python -m pytest tests/test_llm_selector_recency.py -v
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import patch

import pytest


# ── Helper-level tests ───────────────────────────────────────────────────

class TestBelowMinRecency:
    def test_old_cutoff_filtered(self):
        from app.llm_selector import _below_min_recency
        assert _below_min_recency({"knowledge_cutoff": "2024-01-01"}, date(2025, 1, 1)) is True

    def test_recent_cutoff_passes(self):
        from app.llm_selector import _below_min_recency
        assert _below_min_recency({"knowledge_cutoff": "2025-06-01"}, date(2025, 1, 1)) is False

    def test_exact_match_passes(self):
        """Boundary: knowledge_cutoff == min_recency is NOT below — passes."""
        from app.llm_selector import _below_min_recency
        assert _below_min_recency({"knowledge_cutoff": "2025-01-01"}, date(2025, 1, 1)) is False

    def test_missing_cutoff_passes(self):
        """Absence of evidence ≠ evidence of absence; missing field doesn't filter."""
        from app.llm_selector import _below_min_recency
        assert _below_min_recency({}, date(2025, 1, 1)) is False

    def test_malformed_cutoff_passes(self):
        """Garbage in catalog doesn't crash the selector — treated as unknown."""
        from app.llm_selector import _below_min_recency
        assert _below_min_recency({"knowledge_cutoff": "not-a-date"}, date(2025, 1, 1)) is False
        assert _below_min_recency({"knowledge_cutoff": ""}, date(2025, 1, 1)) is False
        assert _below_min_recency({"knowledge_cutoff": None}, date(2025, 1, 1)) is False


# ── Integration through select_model ─────────────────────────────────────

@pytest.fixture
def recency_catalog(monkeypatch):
    """Three-model fixture: stale default, fresh alternative, no-cutoff outlier."""
    import app.llm_catalog as lc
    catalog = {
        "stale-research-model": {
            "tier": "budget", "provider": "openrouter",
            "model_id": "openrouter/test/stale",
            "knowledge_cutoff": "2024-01-01",
            "context": 128_000, "multimodal": False,
            "cost_input_per_m": 0.10, "cost_output_per_m": 0.20,
            "supports_tools": True, "tool_use_reliability": 0.80,
            "strengths": {t: 0.8 for t in (
                "coding", "debugging", "architecture", "research", "writing",
                "reasoning", "multimodal", "vetting", "general",
            )},
        },
        "fresh-research-model": {
            "tier": "budget", "provider": "openrouter",
            "model_id": "openrouter/test/fresh",
            "knowledge_cutoff": "2026-01-01",
            "context": 128_000, "multimodal": False,
            "cost_input_per_m": 0.10, "cost_output_per_m": 0.20,
            "supports_tools": True, "tool_use_reliability": 0.80,
            "strengths": {t: 0.7 for t in (
                "coding", "debugging", "architecture", "research", "writing",
                "reasoning", "multimodal", "vetting", "general",
            )},
        },
        "no-cutoff-model": {
            "tier": "budget", "provider": "openrouter",
            "model_id": "openrouter/test/unknown",
            "context": 128_000, "multimodal": False,
            "cost_input_per_m": 0.10, "cost_output_per_m": 0.20,
            "supports_tools": True, "tool_use_reliability": 0.80,
            "strengths": {t: 0.6 for t in (
                "coding", "debugging", "architecture", "research", "writing",
                "reasoning", "multimodal", "vetting", "general",
            )},
        },
    }
    monkeypatch.setattr(lc, "CATALOG", catalog)
    yield catalog


class TestSelectModelRecency:
    def test_stale_default_demoted_to_fresh(self, recency_catalog):
        """When the default is too old, walk candidates for a fresh one."""
        from app.llm_selector import select_model

        with patch("app.llm_selector.get_default_for_role", return_value="stale-research-model"), \
             patch("app.llm_selector._model_available", return_value=True), \
             patch("app.llm_selector.get_candidates_by_tier",
                   return_value=[("fresh-research-model", 0.7),
                                 ("stale-research-model", 0.8),
                                 ("no-cutoff-model", 0.6)]):
            chosen = select_model("research", min_recency=date(2025, 6, 1))
        assert chosen == "fresh-research-model"

    def test_fresh_default_unchanged(self, recency_catalog):
        from app.llm_selector import select_model

        with patch("app.llm_selector.get_default_for_role", return_value="fresh-research-model"), \
             patch("app.llm_selector._model_available", return_value=True), \
             patch("app.llm_selector.get_candidates_by_tier", return_value=[]):
            chosen = select_model("research", min_recency=date(2025, 6, 1))
        assert chosen == "fresh-research-model"

    def test_no_min_recency_no_filter(self, recency_catalog):
        """Default behaviour preserved — no filtering when min_recency is None."""
        from app.llm_selector import select_model

        with patch("app.llm_selector.get_default_for_role", return_value="stale-research-model"), \
             patch("app.llm_selector._model_available", return_value=True), \
             patch("app.llm_selector.get_candidates_by_tier", return_value=[]):
            chosen = select_model("research")
        assert chosen == "stale-research-model"

    def test_no_compliant_candidate_keeps_default(self, recency_catalog, caplog):
        """Graceful degradation when no model meets the recency floor."""
        from app.llm_selector import select_model
        import logging

        with patch("app.llm_selector.get_default_for_role", return_value="stale-research-model"), \
             patch("app.llm_selector._model_available", return_value=True), \
             patch("app.llm_selector.get_candidates_by_tier",
                   return_value=[("stale-research-model", 0.8),
                                 ("no-cutoff-model", 0.6)]), \
             caplog.at_level(logging.WARNING, logger="app.llm_selector"):
            chosen = select_model("research", min_recency=date(2026, 6, 1))
        assert chosen == "stale-research-model"
        assert any("no candidate meets min_recency" in r.message for r in caplog.records)


# ── Factory-level auto-default ──────────────────────────────────────────

class TestFactoryRoleAutoDefault:
    def test_research_role_auto_applies_recency(self):
        """role=research → factory passes min_recency derived from today - 180d."""
        from app import llm_factory
        captured = {}

        def fake_select(role, task_hint, *, force_tier=None, min_recency=None, **_):
            captured["min_recency"] = min_recency
            return "fresh-research-model"

        with patch("app.llm_selector.select_model", side_effect=fake_select), \
             patch.object(llm_factory, "get_model", return_value={
                 "tier": "budget", "provider": "openrouter",
                 "model_id": "openrouter/test/fresh", "supports_tools": True,
             }), \
             patch.object(llm_factory, "_build_from_entry", return_value="LLM"), \
             patch("app.llm_mode.get_mode", return_value="balanced"):
            llm_factory.create_specialist_llm(role="research")

        cutoff = captured["min_recency"]
        assert cutoff is not None
        # Within 1 day of today - 180 days (allow for clock skew at midnight).
        expected = date.today() - timedelta(days=180)
        assert abs((cutoff - expected).days) <= 1

    def test_coding_role_no_auto_recency(self):
        """role=coding → no recency floor (knowledge cutoff doesn't matter)."""
        from app import llm_factory
        captured = {}

        def fake_select(role, task_hint, *, force_tier=None, min_recency=None, **_):
            captured["min_recency"] = min_recency
            return "any-model"

        with patch("app.llm_selector.select_model", side_effect=fake_select), \
             patch.object(llm_factory, "get_model", return_value={
                 "tier": "budget", "provider": "openrouter",
                 "model_id": "openrouter/test/any", "supports_tools": True,
             }), \
             patch.object(llm_factory, "_build_from_entry", return_value="LLM"), \
             patch("app.llm_mode.get_mode", return_value="balanced"):
            llm_factory.create_specialist_llm(role="coding")

        assert captured["min_recency"] is None

    def test_explicit_date_min_opts_out(self):
        """Sentinel date.min lets a research caller opt out of the auto-default."""
        from app import llm_factory
        captured = {}

        def fake_select(role, task_hint, *, force_tier=None, min_recency=None, **_):
            captured["min_recency"] = min_recency
            return "stale-research-model"

        with patch("app.llm_selector.select_model", side_effect=fake_select), \
             patch.object(llm_factory, "get_model", return_value={
                 "tier": "budget", "provider": "openrouter",
                 "model_id": "openrouter/test/stale", "supports_tools": True,
             }), \
             patch.object(llm_factory, "_build_from_entry", return_value="LLM"), \
             patch("app.llm_mode.get_mode", return_value="balanced"):
            llm_factory.create_specialist_llm(role="research", min_recency=date.min)

        assert captured["min_recency"] is None
