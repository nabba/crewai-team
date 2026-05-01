"""Tests for the dynamic evolution engine selector.

The audit found ShinkaEvolve had run zero times in production despite
being available — the selector's kept_ratio>0.60 gate locked the system
into AVO. The fix adds:

  1. days_since_engine_run() helper in evolution_roi
  2. Forced 7-day rotation rule placed BEFORE the kept_ratio gate
  3. ROI-aware recommendation when both engines have data

These tests verify the selector picks ShinkaEvolve under conditions where
it should, while still preferring AVO under safety-critical situations.
"""
from __future__ import annotations

import os
import sys
import time
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.test_metrics import _FakeSettings  # noqa: E402
import app.config as config_mod  # noqa: E402

config_mod.get_settings = lambda: _FakeSettings()
config_mod.get_anthropic_api_key = lambda: "fake-key"
config_mod.get_gateway_secret = lambda: "a" * 64


# ── Timestamp helper ────────────────────────────────────────────────────────

class TestEngineTimestampHelper:
    def test_no_runs_returns_zero(self, tmp_path, monkeypatch):
        import app.evolution_roi as roi
        monkeypatch.setattr(roi, "ROI_LEDGER_PATH", tmp_path / "roi.json")
        assert roi.get_last_run_timestamp("shinka") == 0.0
        assert roi.days_since_engine_run("shinka") == float("inf")

    def test_returns_most_recent(self, tmp_path, monkeypatch):
        import app.evolution_roi as roi
        monkeypatch.setattr(roi, "ROI_LEDGER_PATH", tmp_path / "roi.json")
        # Insert two runs of each engine
        roi.record_evolution_cost(
            experiment_id="avo_old", engine="avo", cost_usd=0.10,
            delta=0.01, status="keep",
        )
        # Force a tiny delay so timestamps differ
        time.sleep(0.01)
        roi.record_evolution_cost(
            experiment_id="shinka_recent", engine="shinka", cost_usd=1.00,
            delta=0.05, status="keep",
        )
        time.sleep(0.01)
        roi.record_evolution_cost(
            experiment_id="avo_recent", engine="avo", cost_usd=0.10,
            delta=0.02, status="keep",
        )
        assert roi.get_last_run_timestamp("avo") > roi.get_last_run_timestamp("shinka")

    def test_days_since_returns_positive_float(self, tmp_path, monkeypatch):
        import app.evolution_roi as roi
        monkeypatch.setattr(roi, "ROI_LEDGER_PATH", tmp_path / "roi.json")
        roi.record_evolution_cost(
            experiment_id="exp_1", engine="shinka", cost_usd=0.10,
            delta=0.05, status="keep",
        )
        days = roi.days_since_engine_run("shinka")
        assert 0.0 <= days < 1.0


# ── Selector rules ──────────────────────────────────────────────────────────

class TestSelectorManualOverride:
    def test_explicit_avo(self):
        """config.evolution_engine='avo' bypasses all logic."""
        with patch.object(_FakeSettings, "evolution_engine", "avo", create=True):
            from app.evolution import _select_evolution_engine
            assert _select_evolution_engine() == "avo"

    def test_explicit_shinka(self):
        """config.evolution_engine='shinka' bypasses all logic."""
        # Need to also mock availability or it'll fall back
        with patch.object(_FakeSettings, "evolution_engine", "shinka", create=True):
            from app.evolution import _select_evolution_engine
            # Manual override happens BEFORE availability check, so this returns "shinka"
            assert _select_evolution_engine() == "shinka"


class TestSelectorAvailability:
    def test_shinka_unavailable_returns_avo(self):
        with patch.object(_FakeSettings, "evolution_engine", "auto", create=True), \
             patch("app.evolution._is_shinka_available", return_value=False):
            from app.evolution import _select_evolution_engine
            assert _select_evolution_engine() == "avo"


class TestSelectorSafetyGate:
    def test_low_safety_returns_avo(self):
        """SUBIA safety < 0.70 forces conservative AVO."""
        with patch.object(_FakeSettings, "evolution_engine", "auto", create=True), \
             patch("app.evolution._is_shinka_available", return_value=True), \
             patch("app.evolution._get_subia_safety_value", return_value=0.50):
            from app.evolution import _select_evolution_engine
            assert _select_evolution_engine() == "avo"


class TestSelectorRotation:
    def test_forced_rotation_when_shinka_never_ran(self):
        """If shinka has never run, forced rotation picks it."""
        with patch.object(_FakeSettings, "evolution_engine", "auto", create=True), \
             patch("app.evolution._is_shinka_available", return_value=True), \
             patch("app.evolution._get_subia_safety_value", return_value=0.85), \
             patch("app.evolution.get_recent_results", return_value=[
                 {"status": "keep", "delta": 0.01} for _ in range(10)
             ]), \
             patch("app.evolution_roi.days_since_engine_run", return_value=float("inf")):
            from app.evolution import _select_evolution_engine
            assert _select_evolution_engine() == "shinka"

    def test_forced_rotation_when_shinka_run_long_ago(self):
        """If shinka ran > 7 days ago, forced rotation picks it again."""
        with patch.object(_FakeSettings, "evolution_engine", "auto", create=True), \
             patch("app.evolution._is_shinka_available", return_value=True), \
             patch("app.evolution._get_subia_safety_value", return_value=0.85), \
             patch("app.evolution.get_recent_results", return_value=[
                 {"status": "keep", "delta": 0.01} for _ in range(10)
             ]), \
             patch("app.evolution_roi.days_since_engine_run", return_value=10.0):
            from app.evolution import _select_evolution_engine
            assert _select_evolution_engine() == "shinka"

    def test_no_rotation_when_shinka_ran_recently(self):
        """If shinka ran within the last 7 days AND AVO healthy, prefer AVO."""
        recent_kept = [{"status": "keep", "delta": 0.05} for _ in range(10)]
        with patch.object(_FakeSettings, "evolution_engine", "auto", create=True), \
             patch("app.evolution._is_shinka_available", return_value=True), \
             patch("app.evolution._get_subia_safety_value", return_value=0.85), \
             patch("app.evolution.get_recent_results", return_value=recent_kept), \
             patch("app.evolution_roi.days_since_engine_run", return_value=2.0):
            from app.evolution import _select_evolution_engine
            assert _select_evolution_engine() == "avo"

    def test_stagnation_overrides_rotation(self):
        """Stagnation rule (4) should fire before rotation rule (5)."""
        all_failed = [{"status": "discard", "delta": 0.0} for _ in range(5)]
        with patch.object(_FakeSettings, "evolution_engine", "auto", create=True), \
             patch("app.evolution._is_shinka_available", return_value=True), \
             patch("app.evolution._get_subia_safety_value", return_value=0.85), \
             patch("app.evolution.get_recent_results", return_value=all_failed), \
             patch("app.evolution_roi.days_since_engine_run", return_value=0.5):
            from app.evolution import _select_evolution_engine
            # Both stagnation AND rotation point to shinka, but stagnation
            # should fire first per priority order
            assert _select_evolution_engine() == "shinka"


class TestSelectorPerformanceGates:
    def test_high_kept_ratio_returns_avo_when_shinka_recent(self):
        """When AVO is healthy AND shinka ran recently, stick with AVO."""
        kept = [{"status": "keep", "delta": 0.02} for _ in range(10)]
        with patch.object(_FakeSettings, "evolution_engine", "auto", create=True), \
             patch("app.evolution._is_shinka_available", return_value=True), \
             patch("app.evolution._get_subia_safety_value", return_value=0.85), \
             patch("app.evolution.get_recent_results", return_value=kept), \
             patch("app.evolution_roi.days_since_engine_run", return_value=2.0):
            from app.evolution import _select_evolution_engine
            assert _select_evolution_engine() == "avo"

    def test_low_kept_ratio_returns_shinka(self):
        """kept_ratio < 0.20 → ShinkaEvolve (AVO too ambitious)."""
        # 1 kept out of 10 → 10% kept ratio
        recent = [{"status": "discard", "delta": -0.01} for _ in range(9)]
        recent.insert(0, {"status": "keep", "delta": 0.01})
        with patch.object(_FakeSettings, "evolution_engine", "auto", create=True), \
             patch("app.evolution._is_shinka_available", return_value=True), \
             patch("app.evolution._get_subia_safety_value", return_value=0.85), \
             patch("app.evolution.get_recent_results", return_value=recent), \
             patch("app.evolution_roi.days_since_engine_run", return_value=2.0):
            from app.evolution import _select_evolution_engine
            assert _select_evolution_engine() == "shinka"


class TestSelectorROIRecommendation:
    def test_roi_recommendation_picks_better_engine(self):
        """When both engines have data, ROI recommendation breaks the tie."""
        # Mid-range kept_ratio (e.g. 40%) so rules 6-7 don't fire
        mixed = (
            [{"status": "keep", "delta": 0.02} for _ in range(4)]
            + [{"status": "discard", "delta": -0.01} for _ in range(6)]
        )

        # Mock the ROI snapshot to show shinka with better cost-per-improvement
        from app.evolution_roi import ROISnapshot
        snapshot = ROISnapshot(
            window_days=14,
            total_cost_usd=2.00,
            real_improvements=4,
            rollbacks=0,
            rollback_rate=0.0,
            cost_per_improvement=0.50,
            sample_size=20,
            by_engine={
                "avo": {"experiments": 18, "cost_usd": 1.80, "real_improvements": 2, "cost_per_improvement": 0.90},
                "shinka": {"experiments": 2, "cost_usd": 0.20, "real_improvements": 2, "cost_per_improvement": 0.10},
                "meta": {"experiments": 0, "cost_usd": 0.0, "real_improvements": 0, "cost_per_improvement": None},
            },
        )

        with patch.object(_FakeSettings, "evolution_engine", "auto", create=True), \
             patch("app.evolution._is_shinka_available", return_value=True), \
             patch("app.evolution._get_subia_safety_value", return_value=0.85), \
             patch("app.evolution.get_recent_results", return_value=mixed), \
             patch("app.evolution_roi.days_since_engine_run", return_value=2.0), \
             patch("app.evolution_roi.get_rolling_roi", return_value=snapshot), \
             patch("app.evolution_roi.get_engine_recommendation", return_value="shinka"):
            from app.evolution import _select_evolution_engine
            # Shinka has better cost-per-improvement → ROI rec picks shinka
            assert _select_evolution_engine() == "shinka"

    def test_roi_skipped_when_no_shinka_data(self):
        """Without shinka data, ROI rule abstains and falls through to default."""
        mixed = (
            [{"status": "keep", "delta": 0.02} for _ in range(4)]
            + [{"status": "discard", "delta": -0.01} for _ in range(6)]
        )

        from app.evolution_roi import ROISnapshot
        # No shinka data
        snapshot = ROISnapshot(
            window_days=14, total_cost_usd=1.80, real_improvements=2,
            rollbacks=0, rollback_rate=0.0, cost_per_improvement=0.90,
            sample_size=10,
            by_engine={
                "avo": {"experiments": 10, "cost_usd": 1.80, "real_improvements": 2, "cost_per_improvement": 0.90},
                "shinka": {"experiments": 0, "cost_usd": 0.0, "real_improvements": 0, "cost_per_improvement": None},
                "meta": {"experiments": 0, "cost_usd": 0.0, "real_improvements": 0, "cost_per_improvement": None},
            },
        )

        with patch.object(_FakeSettings, "evolution_engine", "auto", create=True), \
             patch("app.evolution._is_shinka_available", return_value=True), \
             patch("app.evolution._get_subia_safety_value", return_value=0.85), \
             patch("app.evolution.get_recent_results", return_value=mixed), \
             patch("app.evolution_roi.days_since_engine_run", return_value=2.0), \
             patch("app.evolution_roi.get_rolling_roi", return_value=snapshot), \
             patch("app.evolution_roi.get_engine_recommendation", return_value="avo"):
            from app.evolution import _select_evolution_engine
            # Without shinka data, ROI rule abstains; default returns avo
            assert _select_evolution_engine() == "avo"


# ── _is_shinka_available — deep-import verification ─────────────────────────
#
# Added 2026-04-30 after diagnosing that shinka had silently failed every
# session for weeks: shinka was installed --no-deps in the Dockerfile, so
# transitive deps (google-genai, psutil, seaborn, python-Levenshtein) were
# missing. The old _is_shinka_available only ran ``import shinka`` (the
# empty package namespace), passed, but the engine then crashed at
# session start with cryptic ImportError. Worse, no ledger record was
# written on those crashes, so days_since_engine_run("shinka") stayed at
# ``inf`` and forced rotation kept picking shinka — forever.

class TestIsShinkaAvailableDeepCheck:

    def test_returns_false_when_core_import_fails(self, monkeypatch):
        """Missing google-genai → shinka.core import fails → False."""
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("shinka.core"):
                raise ImportError("cannot import name 'genai' from 'google'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        from app.evolution import _is_shinka_available
        assert _is_shinka_available() is False

    def test_returns_false_when_launch_import_fails(self, monkeypatch):
        """Missing psutil → shinka.launch fails → False."""
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("shinka.launch"):
                raise ImportError("No module named 'psutil'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        from app.evolution import _is_shinka_available
        assert _is_shinka_available() is False

    def test_logs_warning_with_actionable_message(self, monkeypatch, caplog):
        """When a deep import fails, the operator should see a clear
        log line — not a silent loop. The warning is the single
        observable signal that shinka is misconfigured."""
        import builtins
        import logging
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("shinka.core"):
                raise ImportError("No module named 'google.genai'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with caplog.at_level(logging.WARNING, logger="app.evolution"):
            from app.evolution import _is_shinka_available
            assert _is_shinka_available() is False
        # The warning message should make it clear what to do
        msg = " ".join(r.getMessage() for r in caplog.records)
        assert "shinka unavailable" in msg.lower()
        assert "fall back to avo" in msg.lower() or "avo" in msg.lower()


# ── _map_llm_models — shinka-registry-compatible strings ────────────────────
#
# Added 2026-04-30 alongside the deep-availability fix. The legacy
# mapping returned ``us.anthropic.claude-sonnet-4-20250514-v1:0`` (a
# Bedrock ARN that needs AWS_* env we don't set) and
# ``openrouter/deepseek/deepseek-chat-v3-0324`` (not in shinka's
# OpenRouter allowlist). Result: shinka rejected every model and the
# session crashed at LLM-init.

class TestMapLlmModels:

    def test_anthropic_uses_direct_api_string_not_bedrock(self, monkeypatch):
        """When ANTHROPIC_API_KEY is set, use the shinka-registry name
        (``claude-sonnet-4-6``), NOT the Bedrock-style ARN."""
        from unittest.mock import MagicMock
        fake_settings = MagicMock()
        fake_settings.anthropic_api_key.get_secret_value.return_value = "sk-ant-x" * 5
        monkeypatch.setattr("app.config.get_settings", lambda: fake_settings)
        # Disable openrouter + ollama so we isolate the anthropic path
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setenv("OLLAMA_HOST", "http://invalid-host-no-such-thing.local:11434")

        from app.shinka_engine import _map_llm_models
        models = _map_llm_models()
        assert any("claude" in m and "anthropic" not in m for m in models), (
            f"expected non-Bedrock claude string, got {models}"
        )
        # Bedrock ARNs are the regression we're guarding against
        assert not any("us.anthropic" in m for m in models)

    def test_openrouter_uses_registry_compatible_string(self, monkeypatch):
        """OpenRouter mapping must use a model string that's in
        shinka's allowlist (qwen/qwen3-coder), not deepseek-chat-v3
        which isn't recognised."""
        from unittest.mock import MagicMock
        fake_settings = MagicMock()
        fake_settings.anthropic_api_key.get_secret_value.return_value = ""
        monkeypatch.setattr("app.config.get_settings", lambda: fake_settings)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fake")
        monkeypatch.setenv("OLLAMA_HOST", "http://invalid-host-no-such-thing.local:11434")

        from app.shinka_engine import _map_llm_models
        models = _map_llm_models()
        # qwen3-coder is in shinka's OpenRouter allowlist
        assert any("qwen" in m for m in models), (
            f"expected qwen-family openrouter model, got {models}"
        )
        # Regression guard: deepseek-chat-v3 is NOT in shinka's allowlist
        assert not any("deepseek-chat-v3" in m for m in models)

    def test_no_api_keys_still_returns_at_least_one_model(self, monkeypatch):
        """Function must always return a non-empty list — empty would
        crash shinka's LLM client init with a less-clear error."""
        from unittest.mock import MagicMock
        fake_settings = MagicMock()
        fake_settings.anthropic_api_key.get_secret_value.return_value = ""
        monkeypatch.setattr("app.config.get_settings", lambda: fake_settings)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setenv("OLLAMA_HOST", "http://invalid-host-no-such-thing.local:11434")

        from app.shinka_engine import _map_llm_models
        models = _map_llm_models()
        assert len(models) >= 1


# ── _record_result — dual-ledger writes ────────────────────────────────────
#
# Added 2026-04-30. The pre-fix shinka_engine wrote only to
# results_ledger; evolution_roi never saw shinka activity. Result:
# ``days_since_engine_run("shinka")`` always returned ``inf`` even
# after a successful shinka run, so forced-rotation kept firing every
# cycle.

class TestShinkaRecordResultDualLedger:

    def test_writes_to_both_results_and_roi_ledgers(self, monkeypatch):
        """A successful shinka run must populate both ledgers — not
        just results_ledger. Otherwise the rotation rule's input is
        permanently broken."""
        from unittest.mock import MagicMock
        results_calls: list[dict] = []
        roi_calls: list[dict] = []

        def _fake_results(**kwargs):
            results_calls.append(kwargs)

        def _fake_roi(**kwargs):
            roi_calls.append(kwargs)

        monkeypatch.setattr(
            "app.results_ledger.record_experiment", _fake_results,
        )
        monkeypatch.setattr(
            "app.evolution_roi.record_evolution_cost", _fake_roi,
        )

        from app.shinka_engine import _record_result
        _record_result(baseline=0.5, after=0.6, delta=0.1, status="keep")

        assert len(results_calls) == 1
        assert len(roi_calls) == 1
        # The ROI entry MUST be tagged engine="shinka" — the whole point
        assert roi_calls[0].get("engine") == "shinka"
        # Status threading
        assert roi_calls[0].get("status") == "keep"
        assert results_calls[0].get("status") == "keep"

    def test_results_ledger_failure_does_not_crash(self, monkeypatch):
        """If the results ledger write fails, log + bail (don't try
        ROI write either, since the experiment_id won't be valid)."""
        def _boom(**kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(
            "app.results_ledger.record_experiment", _boom,
        )
        roi_called = {"n": 0}

        def _fake_roi(**kwargs):
            roi_called["n"] += 1

        monkeypatch.setattr(
            "app.evolution_roi.record_evolution_cost", _fake_roi,
        )

        from app.shinka_engine import _record_result
        # Must not raise
        _record_result(baseline=0.5, after=0.6, delta=0.1, status="keep")
        assert roi_called["n"] == 0  # short-circuit after results failure
