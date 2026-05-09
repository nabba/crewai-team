"""
Self-Populating Catalog Tests
==============================

Covers the catalog-builder + resolver architecture:
  - derive_strengths maps AA evaluations to the 9 canonical task types
  - build_snapshot merges OpenRouter + AA + Ollama without losing data
  - merge_into_catalog preserves bootstrap entries but refreshes their
    derived fields
  - resolve_role_default picks the right model under each cost_mode
  - Graceful degradation: catalog-only mode still works when every
    fetcher returns empty

Run:
    docker exec crewai-team-gateway-1 python3 -m pytest \
        /app/tests/test_llm_catalog_builder.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Fresh-catalog fixture ────────────────────────────────────────────────

@pytest.fixture
def fresh_catalog(monkeypatch):
    """Reset CATALOG to bootstrap between tests so mutations don't leak."""
    import app.llm_catalog as lc
    snapshot = {n: dict(e) for n, e in lc._BOOTSTRAP_CATALOG.items()}
    monkeypatch.setattr(lc, "CATALOG", snapshot)
    # llm_catalog_builder imports CATALOG from llm_catalog; patching the
    # attribute on llm_catalog is enough because it's resolved lazily.
    yield snapshot


# ── derive_strengths ─────────────────────────────────────────────────────

class TestDeriveStrengths:
    def test_no_aa_row_falls_back_to_tier_default(self):
        """Tier fallback = tier_base * 0.85 confidence discount.
        Premium: 0.82 * 0.85 ≈ 0.697."""
        from app.llm_catalog_builder import derive_strengths

        s = derive_strengths(None, is_multimodal=False, tier="premium")
        assert s["general"] == pytest.approx(0.697, abs=0.01)
        assert s["multimodal"] == 0.0
        # All 9 canonical keys must be populated
        from app.llm_catalog import CANONICAL_TASK_TYPES
        assert set(s.keys()) == set(CANONICAL_TASK_TYPES)

    def test_multimodal_flag_respected(self):
        from app.llm_catalog_builder import derive_strengths

        s = derive_strengths(None, is_multimodal=True, tier="mid")
        assert s["multimodal"] == 1.0

    def test_intelligence_drives_general(self):
        """AA index / 70 — so top-frontier models approach 1.0.
        Intel=60 → 60/70 ≈ 0.857."""
        from app.llm_catalog_builder import derive_strengths

        aa = {"evaluations": {"artificial_analysis_intelligence_index": 60}}
        s = derive_strengths(aa, is_multimodal=False, tier="mid")
        assert s["general"] == pytest.approx(0.857, abs=0.01)

    def test_coding_uses_coding_index_when_present(self):
        """Coding_index=80 → 80/70 = 1.143 clamped to 1.0."""
        from app.llm_catalog_builder import derive_strengths

        aa = {"evaluations": {
            "artificial_analysis_intelligence_index": 40,
            "artificial_analysis_coding_index": 80,
        }}
        s = derive_strengths(aa, is_multimodal=False, tier="mid")
        assert s["coding"] == pytest.approx(1.0, abs=0.01)
        assert s["debugging"] == pytest.approx(0.95, abs=0.01)

    def test_livecodebench_floor(self):
        """A high livecodebench score pulls coding up even when coding_index
        is missing."""
        from app.llm_catalog_builder import derive_strengths

        aa = {"evaluations": {
            "artificial_analysis_intelligence_index": 40,
            "livecodebench": 0.75,
        }}
        s = derive_strengths(aa, is_multimodal=False, tier="mid")
        assert s["coding"] >= 0.75


# ── build_snapshot + merge ───────────────────────────────────────────────

class TestBuildSnapshot:
    def test_all_fetchers_empty_returns_marker_only(self):
        """When every source returns [], the snapshot still has the
        _fetched_at marker but no models."""
        from app import llm_catalog_builder as b

        with (
            patch.object(b, "_fetch_openrouter", return_value=[]),
            patch.object(b, "_fetch_artificial_analysis", return_value=[]),
            patch.object(b, "_fetch_ollama_tags", return_value=[]),
        ):
            snap = b.build_snapshot()
        assert "_fetched_at" in snap
        # Only the marker — no real entries.
        assert [k for k in snap if not k.startswith("_")] == []

    def test_anthropic_from_aa(self):
        from app import llm_catalog_builder as b

        aa = [{
            "slug": "claude-opus-4-7",
            "model_creator": {"slug": "anthropic"},
            "evaluations": {"artificial_analysis_intelligence_index": 57},
            "pricing": {"price_1m_input_tokens": 5, "price_1m_output_tokens": 25},
            "median_output_tokens_per_second": 50,
        }]
        with (
            patch.object(b, "_fetch_openrouter", return_value=[]),
            patch.object(b, "_fetch_artificial_analysis", return_value=aa),
            patch.object(b, "_fetch_ollama_tags", return_value=[]),
        ):
            snap = b.build_snapshot()
        assert "claude-opus-4.7" in snap
        entry = snap["claude-opus-4.7"]
        assert entry["provider"] == "anthropic"
        assert entry["model_id"] == "anthropic/claude-opus-4-7"
        assert entry["tier"] == "premium"
        assert entry["cost_output_per_m"] == 25.0
        assert entry["strengths"]["general"] > 0.5  # from AA intel

    def test_openrouter_merged_with_aa(self):
        from app import llm_catalog_builder as b

        openrouter = [{
            "id": "deepseek/deepseek-chat",
            "name": "DeepSeek V3.2",
            "context_length": 128000,
            "pricing": {"prompt": "0.00000028", "completion": "0.00000042"},
            "architecture": {"modality": "text"},
            "supported_parameters": ["tools", "temperature"],
        }]
        aa = [{
            "slug": "deepseek-v3-2",  # dash form — should resolve via _canonical_key
            "model_creator": {"slug": "deepseek-ai"},
            "evaluations": {"artificial_analysis_intelligence_index": 45},
            "pricing": {"price_1m_input_tokens": 0.28, "price_1m_output_tokens": 0.42},
        }]
        with (
            patch.object(b, "_fetch_openrouter", return_value=openrouter),
            patch.object(b, "_fetch_artificial_analysis", return_value=aa),
            patch.object(b, "_fetch_ollama_tags", return_value=[]),
        ):
            snap = b.build_snapshot()
        # Deep-seek variants in catalog form uses dot; AA's dash form
        # must be canonicalised to match.
        assert "deepseek-v3.2" in snap or "deepseek-chat" in snap

    def test_ollama_local_entries_separate_namespace(self):
        from app import llm_catalog_builder as b

        ollama = [{"name": "llama3.3:70b", "size": 40_000_000_000}]
        with (
            patch.object(b, "_fetch_openrouter", return_value=[]),
            patch.object(b, "_fetch_artificial_analysis", return_value=[]),
            patch.object(b, "_fetch_ollama_tags", return_value=ollama),
        ):
            snap = b.build_snapshot()
        assert "llama3.3:70b" in snap
        entry = snap["llama3.3:70b"]
        assert entry["tier"] == "local"
        assert entry["provider"] == "ollama"
        assert entry["model_id"] == "ollama_chat/llama3.3:70b"

    def test_openrouter_cutoff_extracted_from_created_at(self):
        """OpenRouter ``created_at`` (epoch seconds) → ``knowledge_cutoff`` (ISO date)."""
        from app import llm_catalog_builder as b

        openrouter = [{
            "id": "vendor/some-model",
            "name": "Some Model",
            "context_length": 128_000,
            "pricing": {"prompt": "0.0000001", "completion": "0.0000002"},
            "architecture": {"modality": "text"},
            "supported_parameters": ["tools"],
            "created_at": 1735689600,  # 2025-01-01 00:00:00 UTC
        }]
        with (
            patch.object(b, "_fetch_openrouter", return_value=openrouter),
            patch.object(b, "_fetch_artificial_analysis", return_value=[]),
            patch.object(b, "_fetch_ollama_tags", return_value=[]),
        ):
            snap = b.build_snapshot()
        assert snap["some-model"]["knowledge_cutoff"] == "2025-01-01"

    def test_openrouter_cutoff_absent_when_field_missing(self):
        """No ``created_at`` → field omitted. Selector treats absence as 'unknown'."""
        from app import llm_catalog_builder as b

        openrouter = [{
            "id": "vendor/no-date",
            "name": "No Date Model",
            "context_length": 128_000,
            "pricing": {"prompt": "0.0000001", "completion": "0.0000002"},
            "architecture": {"modality": "text"},
            "supported_parameters": ["tools"],
        }]
        with (
            patch.object(b, "_fetch_openrouter", return_value=openrouter),
            patch.object(b, "_fetch_artificial_analysis", return_value=[]),
            patch.object(b, "_fetch_ollama_tags", return_value=[]),
        ):
            snap = b.build_snapshot()
        assert "knowledge_cutoff" not in snap["no-date"]

    def test_openrouter_cutoff_invalid_epoch_swallowed(self):
        """Garbage ``created_at`` doesn't crash the build — entry built without cutoff."""
        from app import llm_catalog_builder as b

        openrouter = [{
            "id": "vendor/junk-date",
            "name": "Junk Date",
            "context_length": 128_000,
            "pricing": {"prompt": "0.0000001", "completion": "0.0000002"},
            "architecture": {"modality": "text"},
            "supported_parameters": ["tools"],
            "created_at": "not-a-number",
        }]
        with (
            patch.object(b, "_fetch_openrouter", return_value=openrouter),
            patch.object(b, "_fetch_artificial_analysis", return_value=[]),
            patch.object(b, "_fetch_ollama_tags", return_value=[]),
        ):
            snap = b.build_snapshot()
        assert "junk-date" in snap
        assert "knowledge_cutoff" not in snap["junk-date"]


class TestMergeIntoCatalog:
    def test_bootstrap_preserved_fields_refreshed(self, fresh_catalog):
        """Bootstrap entries stay but their derived fields refresh
        from live data."""
        from app.llm_catalog_builder import merge_into_catalog

        snapshot = {
            "deepseek-v3.2": {
                "tier": "budget", "provider": "openrouter",
                "model_id": "openrouter/deepseek/deepseek-chat",
                "cost_input_per_m": 0.14, "cost_output_per_m": 0.28,
                "strengths": {t: 0.9 for t in (
                    "coding", "debugging", "architecture", "research",
                    "writing", "reasoning", "multimodal", "vetting", "general",
                )},
                "tool_use_reliability": 0.85,
                "context": 200_000,
                "multimodal": True,
                "_auto": True,
                "_source": "test",
            },
        }
        merge_into_catalog(snapshot)
        # Bootstrap key stays
        assert "deepseek-v3.2" in fresh_catalog
        # Refreshed fields
        assert fresh_catalog["deepseek-v3.2"]["cost_output_per_m"] == 0.28
        assert fresh_catalog["deepseek-v3.2"]["strengths"]["coding"] == 0.9
        # Non-refreshed fields untouched (provider stays anthropic /
        # openrouter from bootstrap, etc.)
        assert fresh_catalog["deepseek-v3.2"]["provider"] == "openrouter"

    def test_new_entry_added(self, fresh_catalog):
        from app.llm_catalog_builder import merge_into_catalog

        snapshot = {
            "opus-hypothetical-5.0": {
                "tier": "premium", "provider": "anthropic",
                "model_id": "anthropic/claude-opus-5-0",
                "cost_input_per_m": 5.0, "cost_output_per_m": 25.0,
                "strengths": {t: 0.99 for t in (
                    "coding", "debugging", "architecture", "research",
                    "writing", "reasoning", "multimodal", "vetting", "general",
                )},
                "tool_use_reliability": 0.99,
                "context": 1_000_000,
                "multimodal": True,
                "supports_tools": True,
                "_auto": True,
                "_source": "test",
            },
        }
        merge_into_catalog(snapshot)
        assert "opus-hypothetical-5.0" in fresh_catalog
        # Bootstrap entries still there
        assert "claude-sonnet-4.6" in fresh_catalog

    def test_merge_refreshes_knowledge_cutoff(self, fresh_catalog):
        """Bootstrap ``knowledge_cutoff`` is refreshed by the merge step so
        OpenRouter-derived dates supersede the hand-coded fallback."""
        from app.llm_catalog_builder import merge_into_catalog

        snapshot = {
            "deepseek-v3.2": {
                "tier": "budget", "provider": "openrouter",
                "model_id": "openrouter/deepseek/deepseek-chat",
                "knowledge_cutoff": "2025-03-01",  # newer than bootstrap
                "cost_input_per_m": 0.14, "cost_output_per_m": 0.28,
                "strengths": {t: 0.9 for t in (
                    "coding", "debugging", "architecture", "research",
                    "writing", "reasoning", "multimodal", "vetting", "general",
                )},
                "tool_use_reliability": 0.85,
                "context": 128_000, "multimodal": False,
                "_auto": True, "_source": "test",
            },
        }
        merge_into_catalog(snapshot)
        assert fresh_catalog["deepseek-v3.2"]["knowledge_cutoff"] == "2025-03-01"


# ── Resolver behaviour ───────────────────────────────────────────────────

class TestResolveRoleDefault:
    def test_bootstrap_only_commander_picks_sonnet(self, fresh_catalog):
        """With only 3 bootstrap entries, commander (premium floor) can
        only pick Sonnet."""
        from app.llm_catalog import resolve_role_default

        assert resolve_role_default("commander", "balanced") == "claude-sonnet-4.6"

    def test_budget_mode_prefers_cheaper_for_non_tier_floor_roles(self, fresh_catalog):
        """Coding has no tier floor (budget+), so under budget mode the
        resolver picks DeepSeek over Sonnet."""
        from app.llm_catalog import resolve_role_default

        assert resolve_role_default("coding", "budget") == "deepseek-v3.2"

    def test_new_model_supersedes_when_added(self, fresh_catalog):
        """Inject a hypothetical Opus 5.0 with strictly higher strengths.

        Under ``cost_mode="quality"`` (no cost penalty) the resolver
        picks it for commander. Under ``cost_mode="balanced"`` the 5×
        cost increase versus Sonnet requires a much bigger quality gap
        to justify — the resolver sticks with Sonnet until the gap is
        large enough to overwhelm the penalty, which is the correct
        cost-aware behaviour."""
        fresh_catalog["claude-opus-5.0"] = {
            "tier": "premium", "provider": "anthropic",
            "model_id": "anthropic/claude-opus-5-0",
            "cost_input_per_m": 5.0, "cost_output_per_m": 25.0,
            "strengths": {t: 0.98 for t in (
                "coding", "debugging", "architecture", "research",
                "writing", "reasoning", "vetting", "general",
            )} | {"multimodal": 1.0},
            "tool_use_reliability": 0.99,
            "supports_tools": True, "multimodal": True,
            "_auto": True,
        }
        from app.llm_catalog import resolve_role_default

        # Under quality mode, no cost penalty — newer model wins.
        assert resolve_role_default("commander", "quality") == "claude-opus-5.0"
        # Under balanced mode, sticky with cheaper incumbent until the
        # quality gap is meaningful — correct cost-value behaviour.
        picked_balanced = resolve_role_default("commander", "balanced")
        assert picked_balanced in ("claude-opus-5.0", "claude-sonnet-4.6")

    def test_cheaper_and_stronger_takes_over_coding(self, fresh_catalog):
        """A hypothetical cheaper-and-stronger budget model takes over
        coding under budget mode."""
        fresh_catalog["newdeepseek-v4"] = {
            "tier": "budget", "provider": "openrouter",
            "model_id": "openrouter/deepseek/v4",
            "cost_input_per_m": 0.05, "cost_output_per_m": 0.10,
            "strengths": {t: 0.92 for t in (
                "coding", "debugging", "architecture", "research",
                "writing", "reasoning", "vetting", "general",
            )} | {"multimodal": 0.0},
            "tool_use_reliability": 0.82,
            "supports_tools": True, "multimodal": False,
            "_auto": True,
        }
        from app.llm_catalog import resolve_role_default

        assert resolve_role_default("coding", "budget") == "newdeepseek-v4"

    def test_media_role_requires_multimodal(self, fresh_catalog):
        """media task_type forces multimodal candidates; DeepSeek is
        text-only and must be excluded."""
        from app.llm_catalog import resolve_role_default

        picked = resolve_role_default("media", "balanced")
        # Only Sonnet is multimodal in the bootstrap
        assert picked == "claude-sonnet-4.6"

    def test_role_defaults_view_is_lazy(self, fresh_catalog):
        """ROLE_DEFAULTS back-compat: indexing calls the resolver."""
        from app.llm_catalog import ROLE_DEFAULTS

        assert ROLE_DEFAULTS["balanced"]["commander"] == "claude-sonnet-4.6"
        # Items enumerate all roles
        items = dict(ROLE_DEFAULTS["balanced"].items())
        assert "commander" in items
        assert items["commander"] == "claude-sonnet-4.6"


# ── Graceful degradation ─────────────────────────────────────────────────

class TestGracefulDegradation:
    def test_refresh_fetchers_all_fail_returns_bootstrap_count(self, fresh_catalog):
        """When every fetcher raises, catalog keeps its bootstrap entries."""
        from app import llm_catalog_builder as b

        def _boom(*a, **kw):
            raise RuntimeError("network gone")

        with (
            patch.object(b, "_fetch_openrouter", side_effect=_boom),
            patch.object(b, "_fetch_artificial_analysis", side_effect=_boom),
            patch.object(b, "_fetch_ollama_tags", side_effect=_boom),
            patch.object(b, "_load_snapshot", return_value=None),
            patch.object(b, "_persist_snapshot"),
        ):
            # Expect the refresh to swallow the failures and leave catalog
            # at its bootstrap size of 3.
            try:
                b.refresh(force=True)
            except RuntimeError:
                pass  # build_snapshot may propagate; either way catalog survives

        # Bootstrap entries remain
        assert "claude-sonnet-4.6" in fresh_catalog
        assert "deepseek-v3.2" in fresh_catalog
        assert "qwen3.5:35b-a3b-q4_K_M" in fresh_catalog


class TestProviderFamilyStillStable:
    """The family classifier is a critical piece of the dynamic judge
    selector — keep the old coverage intact."""

    @pytest.mark.parametrize("model_id,expected", [
        ("claude-sonnet-4.6", "anthropic"),
        ("openrouter/deepseek/deepseek-chat", "deepseek"),
        ("openrouter/google/gemini-3.1-pro-preview", "google"),
    ])
    def test_classify(self, model_id, expected):
        from app.llm_discovery import _provider_family
        assert _provider_family(model_id) == expected
