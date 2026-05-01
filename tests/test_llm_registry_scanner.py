"""Tests for the Ollama registry scanner + host-capacity auto-detection.

Background — added 2026-04-25 to close the gap exposed by the qwen3.5
incident: ``llm_discovery.scan_ollama()`` only sees locally-pulled
models, so a strictly-better release like qwen3.5:35b-a3b-q4_K_M stayed
invisible for 3 weeks because nobody had pulled it yet.

The scanner crawls ollama.com/library/<family>/tags, filters by
quantization + size, and emits governance proposals. Size cap is
auto-detected from host RAM minus OS baseline minus Docker overhead so
the same code works on a 16 GB Mac, a 48 GB Mac, and a 256 GB workstation
without per-host constants.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from app import llm_registry_scanner as scanner
from app.llm_registry_scanner import (
    HostCapacity,
    RegistryCandidate,
    _family_base_and_version,
    _parse_model_id,
    _quant_rank,
    diff_against_local,
    filter_candidates,
    filter_dominated_by_installed,
    filter_quant_dominated,
    filter_recently_rejected,
    parse_tags_page,
    probe_host_capacity,
    scan_ollama_registry,
)


# ══════════════════════════════════════════════════════════════════════
# Tags-page HTML parsing
# ══════════════════════════════════════════════════════════════════════

# Realistic HTML fragment matching what ollama.com renders. Captures the
# essential row structure: name, digest12, sizeGB, contextK, modality.
_QWEN35_HTML = """
<html>
<body>
<div class="tags">
qwen3.5:35b 3460ffeede54 • 24GB • 256K context window • Text, Image input • 1 month ago
qwen3.5:35b-a3b a1b2c3d4e5f6 • 22GB • 256K context window • Text, Image input • 3 weeks ago
qwen3.5:35b-a3b-q4_K_M b2c3d4e5f6a7 • 20GB • 256K context window • Text, Image input • 3 weeks ago
qwen3.5:35b-a3b-q8_0 c3d4e5f6a7b8 • 37GB • 256K context window • Text, Image input • 3 weeks ago
qwen3.5:35b-a3b-fp16 d4e5f6a7b8c9 • 70GB • 256K context window • Text, Image input • 3 weeks ago
qwen3.5:0.8b e5f6a7b8c9d0 • 0.5GB • 256K context window • Text input • 1 month ago
qwen3.5:122b f6a7b8c9d0e1 • 81GB • 256K context window • Text, Image input • 1 month ago
qwen3.5:122b-a10b 0a1b2c3d4e5f • 70GB • 256K context window • Text, Image input • 1 month ago
qwen3.5:9b 1a2b3c4d5e6f • 5.5GB • 256K context window • Text input • 1 month ago
</div>
</body>
</html>
"""


class TestParseTagsPage:

    def test_parses_realistic_listing(self):
        cands = parse_tags_page(_QWEN35_HTML, "qwen3.5")
        names = [c.full_name for c in cands]
        assert "qwen3.5:35b-a3b-q4_K_M" in names
        assert "qwen3.5:35b" in names
        assert "qwen3.5:0.8b" in names

    def test_parses_size_and_context(self):
        cands = parse_tags_page(_QWEN35_HTML, "qwen3.5")
        by_name = {c.full_name: c for c in cands}
        assert by_name["qwen3.5:35b-a3b-q4_K_M"].size_gb == 20.0
        assert by_name["qwen3.5:35b-a3b-q4_K_M"].context_k == 256

    def test_parses_modality(self):
        cands = parse_tags_page(_QWEN35_HTML, "qwen3.5")
        by_name = {c.full_name: c for c in cands}
        assert "Image" in by_name["qwen3.5:35b-a3b-q4_K_M"].modality
        assert "Image" not in by_name["qwen3.5:0.8b"].modality

    def test_detects_feature_hints(self):
        cands = parse_tags_page(_QWEN35_HTML, "qwen3.5")
        by_name = {c.full_name: c for c in cands}
        # MoE markers detected from -a3b suffix
        assert "moe-3b-active" in by_name["qwen3.5:35b-a3b-q4_K_M"].features
        # fp16 suffix flagged so the filter can drop it
        assert "full-precision" in by_name["qwen3.5:35b-a3b-fp16"].features

    def test_strict_family_match(self):
        """Parser ignores rows from a different family (paranoia — current
        Ollama pages don't mix families, but the regex must be strict)."""
        mixed = _QWEN35_HTML + "gemma3:27b 999 • 17GB • 128K context window • Text input • 2 months ago"
        cands = parse_tags_page(mixed, "qwen3.5")
        for c in cands:
            assert c.family == "qwen3.5"

    def test_dedupe_repeated_rows(self):
        """The Ollama page renders each tag twice (detailed + compact);
        parser must dedupe by full_name."""
        doubled = _QWEN35_HTML.replace(
            "qwen3.5:35b-a3b-q4_K_M",
            "qwen3.5:35b-a3b-q4_K_M",  # appears verbatim 2x in fixture
        )
        cands = parse_tags_page(doubled + _QWEN35_HTML, "qwen3.5")
        seen = [c.full_name for c in cands]
        assert len(seen) == len(set(seen))

    def test_empty_html_returns_empty(self):
        assert parse_tags_page("", "qwen3.5") == []


# ══════════════════════════════════════════════════════════════════════
# Filtering — size cap + quant preference
# ══════════════════════════════════════════════════════════════════════

class TestFilterCandidates:

    def _mk(self, tag: str, size_gb: float) -> RegistryCandidate:
        return RegistryCandidate(
            family="qwen3.5", tag=tag,
            full_name=f"qwen3.5:{tag}", digest="abc",
            size_gb=size_gb, context_k=256, modality="Text",
            features=[],
        )

    def test_drops_oversized(self):
        cands = [self._mk("35b-a3b-q4_K_M", 20.0),
                 self._mk("35b-a3b-q8_0", 37.0)]
        kept = filter_candidates(cands, max_size_gb=24.0)
        names = [c.full_name for c in kept]
        assert "qwen3.5:35b-a3b-q4_K_M" in names
        assert "qwen3.5:35b-a3b-q8_0" not in names

    def test_drops_undersized(self):
        cands = [self._mk("0.8b", 0.5), self._mk("9b", 5.5)]
        kept = filter_candidates(cands, max_size_gb=24.0, min_size_gb=4.0)
        assert {c.full_name for c in kept} == {"qwen3.5:9b"}

    def test_drops_exotic_quants(self):
        cands = [self._mk("35b-a3b-q4_K_M", 20.0),
                 self._mk("35b-a3b-fp16", 22.0),    # over-precise — drop
                 self._mk("35b-a3b-bf16", 22.0),    # over-precise — drop
                 self._mk("35b-a3b-mxfp8", 22.0)]   # experimental — drop
        kept = filter_candidates(cands, max_size_gb=24.0)
        assert [c.tag for c in kept] == ["35b-a3b-q4_K_M"]

    def test_sort_prefers_features_then_smaller(self):
        cands = [
            RegistryCandidate(family="qwen3.5", tag="35b", full_name="qwen3.5:35b",
                              digest="x", size_gb=24.0, context_k=256,
                              modality="Text", features=[]),
            RegistryCandidate(family="qwen3.5", tag="9b", full_name="qwen3.5:9b",
                              digest="y", size_gb=5.5, context_k=256,
                              modality="Text", features=[]),
            RegistryCandidate(family="qwen3.5", tag="35b-a3b-q4_K_M",
                              full_name="qwen3.5:35b-a3b-q4_K_M",
                              digest="z", size_gb=20.0, context_k=256,
                              modality="Text",
                              features=["moe-3b-active"]),
        ]
        kept = filter_candidates(cands, max_size_gb=24.0)
        # MoE feature wins; among non-feature, smaller wins
        assert kept[0].tag == "35b-a3b-q4_K_M"


# ══════════════════════════════════════════════════════════════════════
# Local-tag dedupe
# ══════════════════════════════════════════════════════════════════════

class TestDiffAgainstLocal:

    def test_drops_already_pulled(self):
        cands = [
            RegistryCandidate(family="qwen3.5", tag="35b", full_name="qwen3.5:35b",
                              digest="x", size_gb=24, context_k=256,
                              modality="Text", features=[]),
            RegistryCandidate(family="qwen3.5", tag="35b-a3b-q4_K_M",
                              full_name="qwen3.5:35b-a3b-q4_K_M",
                              digest="y", size_gb=20, context_k=256,
                              modality="Text", features=[]),
        ]
        local = ["qwen3.5:35b", "llama3.1:8b"]
        new = diff_against_local(cands, local)
        assert {c.full_name for c in new} == {"qwen3.5:35b-a3b-q4_K_M"}


# ══════════════════════════════════════════════════════════════════════
# Host-capacity auto-detection — the headline feature
# ══════════════════════════════════════════════════════════════════════

class TestProbeHostCapacity:
    """Auto-detected size cap replaces the hardcoded 24 GB constant.

    The deepseek-r1:32b SIGKILL spiral happened because two ~38 GB models
    loaded into 48 GB unified memory simultaneously. A static cap
    couldn't have prevented that on a smaller machine; a static cap on
    a bigger machine wastes capability. Auto-detection adapts.
    """

    def test_returns_none_when_ram_undetectable(self, monkeypatch):
        monkeypatch.setattr(scanner, "_detect_total_ram_gb", lambda: 0.0)
        assert probe_host_capacity() is None

    def test_48gb_mac_with_1_loaded_model(self, monkeypatch):
        """Reproduce the actual host this code was built on:
        48 GB Mac, OLLAMA_MAX_LOADED_MODELS=1, ~14 GB Docker overhead.

        Uses HOST_TOTAL_RAM_GB so the probe takes the host-authoritative
        path (and DOES count container limits as overhead — see the
        Docker-on-Mac caveat in probe_host_capacity docstring)."""
        monkeypatch.setenv("HOST_TOTAL_RAM_GB", "48")
        monkeypatch.setattr(scanner, "_detect_os_baseline_gb", lambda: 10.0)
        monkeypatch.setattr(scanner, "_detect_docker_overhead_gb", lambda: 14.0)
        monkeypatch.setattr(scanner, "_detect_max_loaded_models", lambda: 1)
        cap = probe_host_capacity()
        assert cap is not None
        assert cap.total_ram_gb == 48.0
        assert cap.source == "env"
        assert cap.ollama_budget_gb == 24.0  # 48 - 10 - 14
        # 24 / 1 / 1.20 * 0.95 = 19.0 GB — fits qwen3.5:35b-a3b-q4_K_M (20 GB)
        # at marginal verdict, drops the q8_0 (37 GB) decisively.
        assert 18.5 <= cap.max_model_size_gb <= 19.5

    def test_16gb_mac_caps_aggressively(self, monkeypatch):
        """Verify a 16 GB Mac would NOT have proposed 20 GB models."""
        monkeypatch.setenv("HOST_TOTAL_RAM_GB", "16")
        monkeypatch.setattr(scanner, "_detect_os_baseline_gb", lambda: 8.0)
        monkeypatch.setattr(scanner, "_detect_docker_overhead_gb", lambda: 0.0)
        monkeypatch.setattr(scanner, "_detect_max_loaded_models", lambda: 1)
        cap = probe_host_capacity()
        assert cap is not None
        # 16 - 8 = 8 GB budget; 8 / 1 / 1.20 * 0.95 ≈ 6.3 GB
        assert cap.max_model_size_gb < 7.0
        # qwen3.5:35b-a3b-q4_K_M (20 GB) would be filtered out
        assert cap.max_model_size_gb < 20.0

    def test_max_loaded_models_2_halves_per_model_budget(self, monkeypatch):
        """OLLAMA_MAX_LOADED_MODELS=2 means each model gets half the budget.
        This is what we backed away from after the SIGKILL spiral."""
        monkeypatch.setenv("HOST_TOTAL_RAM_GB", "48")
        monkeypatch.setattr(scanner, "_detect_os_baseline_gb", lambda: 10.0)
        monkeypatch.setattr(scanner, "_detect_docker_overhead_gb", lambda: 14.0)
        monkeypatch.setattr(scanner, "_detect_max_loaded_models", lambda: 2)
        cap = probe_host_capacity()
        assert cap is not None
        # 24 / 2 / 1.20 * 0.95 ≈ 9.5 GB
        assert 9.0 <= cap.max_model_size_gb <= 10.0

    def test_docker_view_does_not_double_count(self, monkeypatch):
        """When total_ram comes from /proc/meminfo (Docker container view),
        the scanner must NOT also subtract container limits — that's the
        same memory measured twice. Bug fix from 2026-04-25 testing."""
        monkeypatch.delenv("HOST_TOTAL_RAM_GB", raising=False)
        # Simulate Docker-on-Mac: /proc/meminfo shows VM size 23.4 GB
        def _patched_total():
            scanner._LAST_PROBE_SOURCE["source"] = "proc_meminfo"
            return 23.4
        monkeypatch.setattr(scanner, "_detect_total_ram_gb", _patched_total)
        monkeypatch.setattr(scanner, "_detect_os_baseline_gb", lambda: 4.0)
        # Even if container limits sum to 22.8, we MUST NOT subtract again
        monkeypatch.setattr(scanner, "_detect_docker_overhead_gb", lambda: 22.8)
        monkeypatch.setattr(scanner, "_detect_max_loaded_models", lambda: 1)
        cap = probe_host_capacity()
        assert cap is not None
        # Should be 23.4 - 4.0 - 0 = 19.4 (overhead skipped)
        # NOT 23.4 - 4.0 - 22.8 = -3.4 (the regressed math)
        assert cap.docker_overhead_gb == 0.0
        assert cap.ollama_budget_gb == 19.4
        assert cap.max_model_size_gb > 0  # the regressed math gave 0.0


class TestSizeCapResolution:
    """Resolution priority: env override > auto-detect > fallback."""

    def test_env_override_wins(self, monkeypatch):
        monkeypatch.setenv("LLM_REGISTRY_MAX_SIZE_GB", "8.5")
        # Even if auto-detect would say 19, env wins.
        monkeypatch.setattr(
            scanner, "probe_host_capacity",
            lambda: HostCapacity(48, 10, 14, 24, 1, 1.2, 19.0, "test"),
        )
        assert scanner._max_size_from_env() == 8.5

    def test_auto_detect_used_when_no_env(self, monkeypatch):
        monkeypatch.delenv("LLM_REGISTRY_MAX_SIZE_GB", raising=False)
        monkeypatch.setattr(
            scanner, "probe_host_capacity",
            lambda: HostCapacity(48, 10, 14, 24, 1, 1.2, 19.0, "test"),
        )
        assert scanner._max_size_from_env() == 19.0

    def test_fallback_when_probe_fails(self, monkeypatch):
        monkeypatch.delenv("LLM_REGISTRY_MAX_SIZE_GB", raising=False)
        monkeypatch.setattr(scanner, "probe_host_capacity", lambda: None)
        assert scanner._max_size_from_env() == scanner._DEFAULT_MAX_SIZE_GB_FALLBACK

    def test_invalid_env_falls_through_to_auto(self, monkeypatch):
        monkeypatch.setenv("LLM_REGISTRY_MAX_SIZE_GB", "not-a-number")
        monkeypatch.setattr(
            scanner, "probe_host_capacity",
            lambda: HostCapacity(48, 10, 14, 24, 1, 1.2, 19.0, "test"),
        )
        assert scanner._max_size_from_env() == 19.0


# ══════════════════════════════════════════════════════════════════════
# End-to-end — scan_ollama_registry()
# ══════════════════════════════════════════════════════════════════════

class TestScanOllamaRegistry:

    def test_disabled_returns_empty(self, monkeypatch):
        monkeypatch.setenv("LLM_REGISTRY_SCAN_ENABLED", "false")
        out = scan_ollama_registry(fetch=lambda fam: _QWEN35_HTML)
        assert out == []

    def test_filters_by_capacity(self, monkeypatch):
        monkeypatch.setenv("LLM_REGISTRY_SCAN_ENABLED", "true")
        # Force capacity = 19 GB → q4_K_M (20 GB) just barely doesn't fit
        out = scan_ollama_registry(
            families=("qwen3.5",),
            max_size_gb=19.0,
            fetch=lambda fam: _QWEN35_HTML,
        )
        names = [c.full_name for c in out]
        # 20 GB > 19 GB cap → dropped
        assert "qwen3.5:35b-a3b-q4_K_M" not in names
        # 5.5 GB still passes if min_size_gb default (4) is satisfied
        assert "qwen3.5:9b" in names

    def test_caps_to_oversized_at_24gb(self, monkeypatch):
        monkeypatch.setenv("LLM_REGISTRY_SCAN_ENABLED", "true")
        out = scan_ollama_registry(
            families=("qwen3.5",),
            max_size_gb=24.0,
            fetch=lambda fam: _QWEN35_HTML,
        )
        names = [c.full_name for c in out]
        # The MoE q4_K_M now fits
        assert "qwen3.5:35b-a3b-q4_K_M" in names
        # The q8_0 (37 GB) and 122b (81 GB) variants always too big
        assert "qwen3.5:35b-a3b-q8_0" not in names
        assert "qwen3.5:122b" not in names


# ══════════════════════════════════════════════════════════════════════
# Tag-shape parser — _parse_model_id + _quant_rank
# ══════════════════════════════════════════════════════════════════════

class TestParseModelId:
    """Structural parser used by all three new filters. The filters only
    block when this parser successfully extracts the dimension they care
    about — bad parses degrade to "allow"."""

    def test_qwen35_moe_q4(self):
        p = _parse_model_id("qwen3.5:35b-a3b-q4_K_M")
        assert p.family == "qwen3.5"
        assert p.size_b == 35.0
        assert p.quant == "q4_k_m"
        assert "a3b" in p.variants

    def test_bare_size_no_quant(self):
        p = _parse_model_id("qwen3.5:35b")
        assert p.family == "qwen3.5"
        assert p.size_b == 35.0
        assert p.quant is None
        assert p.variants == ()

    def test_fractional_size(self):
        p = _parse_model_id("qwen3.5:0.8b")
        assert p.size_b == 0.8

    def test_no_size_token(self):
        # Some tags are pure :latest or :instruct — no numeric size.
        p = _parse_model_id("qwen3.5:latest")
        assert p.family == "qwen3.5"
        assert p.size_b is None

    def test_unparseable_returns_safe_default(self):
        p = _parse_model_id("garbage-no-colon")
        assert p.size_b is None
        assert p.quant is None

    def test_empty_string(self):
        p = _parse_model_id("")
        assert p.family == ""
        assert p.size_b is None

    def test_quant_rank_ordering(self):
        # q4_K_M (4) < q5_K_M (5) < q8_0 (7) < fp16 (9)
        assert _quant_rank("q4_K_M") < _quant_rank("q5_K_M")
        assert _quant_rank("q5_K_M") < _quant_rank("q8_0")
        assert _quant_rank("q8_0") < _quant_rank("fp16")

    def test_quant_rank_unknown_is_zero(self):
        assert _quant_rank("xyz_made_up") == 0
        assert _quant_rank(None) == 0


# ══════════════════════════════════════════════════════════════════════
# Filter 1 — same-family dominance
# ══════════════════════════════════════════════════════════════════════

class TestFilterDominatedByInstalled:
    """Skip same-family candidates that are strictly smaller than something
    already installed. The 9-of-9 rejection storm on 2026-04-30 was exactly
    this case: user had qwen3.5:35b-a3b-q4_K_M, scanner kept proposing
    qwen3.5:4b-* siblings."""

    def _mk(self, full: str, size_gb: float = 5.0) -> RegistryCandidate:
        family, tag = full.split(":", 1)
        return RegistryCandidate(
            family=family, tag=tag, full_name=full, digest="abc",
            size_gb=size_gb, context_k=128, modality="Text", features=[],
        )

    def test_skips_smaller_sibling(self):
        """Same family, smaller candidate → DROP."""
        cands = [self._mk("qwen3.5:4b-q8_0", 5.5)]
        local = ["qwen3.5:35b-a3b-q4_K_M"]
        kept = filter_dominated_by_installed(cands, local)
        assert kept == []

    def test_skips_multiple_smaller_siblings(self):
        """Reproduces the 9-of-9 governance rejection scenario."""
        cands = [
            self._mk("qwen3.5:4b-q8_0", 5.5),
            self._mk("qwen3.5:4b-instruct-q8_0", 5.5),
            self._mk("qwen3:4b-thinking-q8_0", 5.0),
        ]
        local = ["qwen3.5:35b-a3b-q4_K_M", "qwen3:32b"]
        kept = filter_dominated_by_installed(cands, local)
        # All three are smaller than their family's installed max → DROP
        assert kept == []

    def test_keeps_different_family(self):
        """llama3.1:8b is NOT dominated by qwen3.5:35b — different family."""
        cands = [self._mk("llama3.1:8b")]
        local = ["qwen3.5:35b-a3b-q4_K_M"]
        kept = filter_dominated_by_installed(cands, local)
        assert {c.full_name for c in kept} == {"llama3.1:8b"}

    def test_keeps_larger_sibling(self):
        """A 70B candidate is NOT dominated by a 35B installation."""
        cands = [self._mk("qwen3.5:122b-a10b", 70)]
        local = ["qwen3.5:35b-a3b-q4_K_M"]
        kept = filter_dominated_by_installed(cands, local)
        assert {c.full_name for c in kept} == {"qwen3.5:122b-a10b"}

    def test_keeps_when_family_not_installed(self):
        """Empty local list → no dominance possible."""
        cands = [self._mk("qwen3.5:4b-q8_0")]
        kept = filter_dominated_by_installed(cands, [])
        assert {c.full_name for c in kept} == {"qwen3.5:4b-q8_0"}

    def test_size_unknown_candidate_passes_when_family_not_installed(self):
        """Sizeless candidate from a family with no installed member
        passes through — we can't prove dominance against a vacuum."""
        cands = [self._mk("phi3:latest")]  # phi family not installed
        local = ["qwen3.5:35b-a3b-q4_K_M"]
        kept = filter_dominated_by_installed(cands, local)
        assert {c.full_name for c in kept} == {"phi3:latest"}

    # ── Rule 2: sizeless candidate of an already-installed family ────

    def test_skips_latest_alias_of_pinned_family(self):
        """Reproducer for the 2026-04-30 second-wave rejection: when
        the user has already pinned a specific qwen3.5 variant, a
        proposal for qwen3.5:latest must be suppressed — the alias is
        almost always the smallest default variant and not what was
        chosen."""
        cands = [self._mk("qwen3.5:latest", 6.6)]
        local = ["qwen3.5:35b-a3b-q4_K_M"]
        kept = filter_dominated_by_installed(cands, local)
        assert kept == []

    def test_skips_instruct_alias_of_pinned_family(self):
        cands = [self._mk("qwen3.5:instruct")]  # sizeless, same family
        local = ["qwen3.5:35b-a3b-q4_K_M"]
        kept = filter_dominated_by_installed(cands, local)
        assert kept == []

    # ── Rule 3: cross-version-within-base dominance ────────────────────

    def test_skips_older_lineage_qwen3_when_qwen35_installed(self):
        """Reproducer for the 2026-04-30 second-wave rejection: qwen3
        (parent lineage) dominated by qwen3.5 (newer minor)."""
        cands = [
            self._mk("qwen3:14b-q4_K_M", 9.3),
            self._mk("qwen3:8b-q4_K_M", 5.2),
        ]
        local = ["qwen3.5:35b-a3b-q4_K_M"]
        kept = filter_dominated_by_installed(cands, local)
        assert kept == []

    def test_skips_older_lineage_llama3_when_llama4_installed(self):
        cands = [self._mk("llama3.1:8b-instruct")]
        local = ["llama4:70b-q4_K_M"]
        kept = filter_dominated_by_installed(cands, local)
        assert kept == []

    def test_keeps_newer_lineage_than_installed(self):
        """A newer qwen4 should NOT be dominated by an installed qwen3.5."""
        cands = [self._mk("qwen4:14b-q4_K_M", 9.0)]
        local = ["qwen3.5:35b-a3b-q4_K_M"]
        kept = filter_dominated_by_installed(cands, local)
        assert {c.full_name for c in kept} == {"qwen4:14b-q4_K_M"}

    def test_keeps_unrelated_base_lineage(self):
        """Different base lineage → no cross-version effect."""
        cands = [self._mk("gemma3:27b-q4_K_M", 17.0)]
        local = ["qwen3.5:35b-a3b-q4_K_M"]
        kept = filter_dominated_by_installed(cands, local)
        assert {c.full_name for c in kept} == {"gemma3:27b-q4_K_M"}

    def test_keeps_same_lineage_higher_version(self):
        """If candidate version > installed version, NOT dominated."""
        cands = [self._mk("qwen3.5:14b-q4_K_M", 9.0)]
        local = ["qwen3:35b-a3b-q4_K_M"]
        kept = filter_dominated_by_installed(cands, local)
        # candidate is qwen3.5 (newer), installed is qwen3 (older)
        # → KEEP (cross-version is forward-only)
        assert {c.full_name for c in kept} == {"qwen3.5:14b-q4_K_M"}


class TestFamilyBaseAndVersion:
    """Direct tests for the lineage parser used by Rule 3."""

    def test_simple_versioned_families(self):
        assert _family_base_and_version("qwen3") == ("qwen", 3.0)
        assert _family_base_and_version("qwen3.5") == ("qwen", 3.5)
        assert _family_base_and_version("llama3.1") == ("llama", 3.1)
        assert _family_base_and_version("llama4") == ("llama", 4.0)
        assert _family_base_and_version("gemma4") == ("gemma", 4.0)
        assert _family_base_and_version("phi3.5") == ("phi", 3.5)

    def test_dashed_base_with_version(self):
        assert _family_base_and_version("deepseek-r1") == ("deepseek-r", 1.0)

    def test_no_version_digits(self):
        assert _family_base_and_version("codestral") == ("codestral", None)
        assert _family_base_and_version("glm-ocr") == ("glm-ocr", None)

    def test_empty_input(self):
        assert _family_base_and_version("") == ("", None)


# ══════════════════════════════════════════════════════════════════════
# Filter 2 — quantization preference
# ══════════════════════════════════════════════════════════════════════

class TestFilterQuantDominated:
    """Skip candidates that are higher-quant (bigger/more precise) versions
    of something already installed at a leaner quant."""

    def _mk(self, full: str, size_gb: float = 20.0) -> RegistryCandidate:
        family, tag = full.split(":", 1)
        return RegistryCandidate(
            family=family, tag=tag, full_name=full, digest="abc",
            size_gb=size_gb, context_k=128, modality="Text", features=[],
        )

    def test_skips_higher_quant(self):
        """Have q4_K_M, proposed q8_0 of same base → DROP (q8 is stricter)."""
        cands = [self._mk("qwen3.5:35b-a3b-q8_0", 37)]
        local = ["qwen3.5:35b-a3b-q4_K_M"]
        kept = filter_quant_dominated(cands, local)
        assert kept == []

    def test_skips_fp16_when_q4_installed(self):
        cands = [self._mk("qwen3.5:35b-a3b-fp16", 70)]
        local = ["qwen3.5:35b-a3b-q4_K_M"]
        kept = filter_quant_dominated(cands, local)
        assert kept == []

    def test_keeps_lower_quant(self):
        """Have q8_0, proposed q4_K_M of same base → KEEP (cheaper alt)."""
        cands = [self._mk("qwen3.5:35b-a3b-q4_K_M", 20)]
        local = ["qwen3.5:35b-a3b-q8_0"]
        kept = filter_quant_dominated(cands, local)
        assert {c.full_name for c in kept} == {"qwen3.5:35b-a3b-q4_K_M"}

    def test_keeps_intermediate_quant(self):
        """Have q8_0, proposed q5_K_M → KEEP (still leaner)."""
        cands = [self._mk("qwen3.5:35b-a3b-q5_K_M", 25)]
        local = ["qwen3.5:35b-a3b-q8_0"]
        kept = filter_quant_dominated(cands, local)
        assert len(kept) == 1

    def test_keeps_when_size_differs(self):
        """Same family but different size → quant filter doesn't apply."""
        cands = [self._mk("qwen3.5:122b-a10b-q8_0", 90)]
        local = ["qwen3.5:35b-a3b-q4_K_M"]
        kept = filter_quant_dominated(cands, local)
        assert len(kept) == 1

    def test_keeps_when_variants_differ(self):
        """Same family/size but different variants → distinct base, KEEP."""
        cands = [self._mk("qwen3.5:35b-instruct-q8_0", 37)]
        # MoE variant ('-a3b') vs vanilla — variants set differs, so the
        # filter treats them as different bases.
        local = ["qwen3.5:35b-a3b-q4_K_M"]
        kept = filter_quant_dominated(cands, local)
        assert len(kept) == 1


# ══════════════════════════════════════════════════════════════════════
# Filter 3 — rejection learning
# ══════════════════════════════════════════════════════════════════════

class TestFilterRecentlyRejected:
    """Skip candidates rejected via governance in the last 30 days. Reads
    from control_plane.governance_requests; injectable for tests."""

    def _mk(self, full: str) -> RegistryCandidate:
        family, tag = full.split(":", 1)
        return RegistryCandidate(
            family=family, tag=tag, full_name=full, digest="abc",
            size_gb=5.0, context_k=128, modality="Text", features=[],
        )

    def test_skips_rejected(self):
        cands = [self._mk("qwen3.5:4b-q8_0")]
        rejected = {"qwen3.5:4b-q8_0"}
        kept = filter_recently_rejected(cands, rejected_set=rejected)
        assert kept == []

    def test_skips_only_matching(self):
        """Only the rejected one is dropped; others pass."""
        cands = [
            self._mk("qwen3.5:4b-q8_0"),
            self._mk("qwen3.5:4b-instruct-q8_0"),
            self._mk("qwen3:4b-thinking-q8_0"),
        ]
        rejected = {"qwen3.5:4b-q8_0"}  # only one was rejected
        kept = filter_recently_rejected(cands, rejected_set=rejected)
        assert {c.full_name for c in kept} == {
            "qwen3.5:4b-instruct-q8_0",
            "qwen3:4b-thinking-q8_0",
        }

    def test_empty_rejected_set_is_passthrough(self):
        cands = [self._mk("qwen3.5:4b-q8_0")]
        kept = filter_recently_rejected(cands, rejected_set=set())
        assert len(kept) == 1

    def test_empty_candidates_is_passthrough(self):
        kept = filter_recently_rejected([], rejected_set={"x:y"})
        assert kept == []

    def test_db_failure_is_silent(self, monkeypatch):
        """If the DB query blows up, scanner should keep proposing — loud
        over silent. Verifies get_recently_rejected_models swallows.

        Inject a fake ``app.control_plane.db`` module via sys.modules so
        the lazy import inside ``get_recently_rejected_models`` resolves
        to a stub whose ``execute()`` raises — independent of whether
        psycopg2 is installed on the test host.
        """
        import sys
        import types
        cands = [self._mk("qwen3.5:4b-q8_0")]

        fake = types.ModuleType("app.control_plane.db")
        def _boom(*a, **kw):  # noqa: ANN001 — test stub
            raise RuntimeError("db down")
        fake.execute = _boom  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "app.control_plane.db", fake)

        # Don't pass rejected_set so the function fetches itself
        kept = filter_recently_rejected(cands)
        # On DB failure → empty rejected set → all candidates kept
        assert len(kept) == 1
