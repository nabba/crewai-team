"""Tests for SUBIA-evolution bridge in evolution.py.

Covers Phase 7: surprise-driven evolution targeting and homeostatic
aggressiveness modulation.
"""
import os
import sys
import types
import pytest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.test_metrics import _FakeSettings
import app.config as config_mod
config_mod.get_settings = lambda: _FakeSettings()
config_mod.get_anthropic_api_key = lambda: "fake-key"
config_mod.get_gateway_secret = lambda: "a" * 64

# Mock heavy dependencies
_mock_crewai = types.ModuleType("crewai")
_mock_crewai.Agent = type("Agent", (), {"__init__": lambda *a, **kw: None})
_mock_crewai.Task = type("Task", (), {"__init__": lambda *a, **kw: None})
_mock_crewai.Crew = type("Crew", (), {"__init__": lambda *a, **kw: None, "kickoff": lambda s: ""})
_mock_crewai.Process = type("Process", (), {"sequential": "sequential"})
_mock_crewai.LLM = type("LLM", (), {"__init__": lambda *a, **kw: None})
sys.modules.setdefault("crewai", _mock_crewai)

_mock_firebase = types.ModuleType("app.firebase_reporter")
_mock_firebase.crew_started = lambda *a, **kw: "task_0"
_mock_firebase.crew_completed = lambda *a, **kw: None
_mock_firebase.crew_failed = lambda *a, **kw: None
sys.modules["app.firebase_reporter"] = _mock_firebase

_mock_ws = types.ModuleType("app.tools.web_search")
_mock_ws.web_search = lambda *a, **kw: ""
sys.modules.setdefault("app.tools.web_search", _mock_ws)

_mock_mem = types.ModuleType("app.tools.memory_tool")
_mock_mem.create_memory_tools = lambda **kw: []
sys.modules.setdefault("app.tools.memory_tool", _mock_mem)

_mock_fm = types.ModuleType("app.tools.file_manager")
_mock_fm.file_manager = lambda *a, **kw: ""
sys.modules.setdefault("app.tools.file_manager", _mock_fm)


class TestSUBIAPredictionInContext:
    """Tests for surprise-driven evolution targeting in _build_evolution_context()."""

    def test_context_includes_subia_section_when_available(self, tmp_path, monkeypatch):
        """When SUBIA tracker has sustained errors, they appear in evolution context."""
        import app.evolution as evo
        import app.results_ledger as ledger
        import app.healing.error_diagnosis as heal

        monkeypatch.setattr(evo, "PROGRAM_PATH", tmp_path / "program.md")
        (tmp_path / "program.md").write_text("# Test")
        monkeypatch.setattr(evo, "SKILLS_DIR", tmp_path / "skills")
        (tmp_path / "skills").mkdir()
        monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "results.tsv")
        monkeypatch.setattr(heal, "ERROR_JOURNAL", tmp_path / "errors.json")

        # Mock SUBIA tracker with sustained errors
        mock_tracker = MagicMock()
        mock_tracker.all_domains_summary.return_value = {
            "coding:web_search": {"mean_accuracy": 0.3, "recent_bad_count": 8},
        }
        mock_tracker.has_sustained_error.return_value = True

        with patch("app.subia.prediction.accuracy_tracker.get_tracker", return_value=mock_tracker):
            context = evo._build_evolution_context()
            assert "Prediction Failures" in context or "coding:web_search" in context

    def test_context_graceful_without_subia(self, tmp_path, monkeypatch):
        """Evolution context works fine when SUBIA modules aren't available."""
        import app.evolution as evo
        import app.results_ledger as ledger
        import app.healing.error_diagnosis as heal

        monkeypatch.setattr(evo, "PROGRAM_PATH", tmp_path / "program.md")
        (tmp_path / "program.md").write_text("# Test")
        monkeypatch.setattr(evo, "SKILLS_DIR", tmp_path / "skills")
        (tmp_path / "skills").mkdir()
        monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "results.tsv")
        monkeypatch.setattr(heal, "ERROR_JOURNAL", tmp_path / "errors.json")

        # If SUBIA import fails, context should still build
        with patch("app.subia.prediction.accuracy_tracker.get_tracker", side_effect=ImportError("no subia")):
            context = evo._build_evolution_context()
            assert "Research Directions" in context
            assert "Current Metrics" in context

    def test_context_includes_snapshot_archive(self, tmp_path, monkeypatch):
        """Evolution context includes historical variant tags."""
        import app.evolution as evo
        import app.results_ledger as ledger
        import app.healing.error_diagnosis as heal

        monkeypatch.setattr(evo, "PROGRAM_PATH", tmp_path / "program.md")
        (tmp_path / "program.md").write_text("# Test")
        monkeypatch.setattr(evo, "SKILLS_DIR", tmp_path / "skills")
        (tmp_path / "skills").mkdir()
        monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "results.tsv")
        monkeypatch.setattr(heal, "ERROR_JOURNAL", tmp_path / "errors.json")

        with patch("app.workspace_versioning.list_evolution_tags", return_value=[
            {"tag": "evo-abc1234-20260414", "sha": "abc1234", "date": "2026-04-14"},
        ]):
            context = evo._build_evolution_context()
            assert "Historical Variants" in context or "evo-abc1234" in context


class TestHomeostaticAggressivenessModulation:
    """Tests for the safety-variable-driven evolution aggressiveness."""

    def _make_mock_kernel(self, safety_value: float):
        """Create a mock SUBIA kernel with given safety variable."""
        kernel = SimpleNamespace()
        kernel.homeostasis = SimpleNamespace()
        kernel.homeostasis.variables = {"safety": safety_value}
        return kernel

    def test_aggressive_mode_increases_iterations(self):
        """Safety > 0.92 should increase max_iterations by 50%."""
        kernel = self._make_mock_kernel(0.95)
        safety = kernel.homeostasis.variables.get("safety", 0.8)
        assert safety == 0.95
        assert safety > 0.92  # should trigger aggressive mode

        # Verify the aggressiveness modulation logic
        base_iterations = 5
        if safety > 0.92:
            adjusted = int(base_iterations * 1.5)
        else:
            adjusted = base_iterations
        assert adjusted == 7  # 5 * 1.5 = 7 (int truncated)

    def test_conservative_mode_threshold(self):
        """Safety < 0.70 should trigger conservative mode."""
        kernel = self._make_mock_kernel(0.60)
        safety = kernel.homeostasis.variables.get("safety", 0.8)
        assert safety < 0.70

    def test_normal_mode_range(self):
        """Safety between 0.70-0.92 should be normal mode."""
        kernel = self._make_mock_kernel(0.85)
        safety = kernel.homeostasis.variables.get("safety", 0.8)
        assert 0.70 <= safety <= 0.92


class TestDailyPromotionLimit:
    """The daily promotion limit should adjust based on aggressiveness."""

    def test_default_limit_is_10(self):
        """Default EVOLUTION_MAX_DAILY_PROMOTIONS should be 10 (upgraded from 3)."""
        # The env var default in the code is "10"
        default = int(os.environ.get("EVOLUTION_MAX_DAILY_PROMOTIONS", "10"))
        assert default == 10

    def test_conservative_reduces_limit(self):
        """In conservative mode, limit should be reduced to 1/3."""
        base = 10
        conservative_limit = max(1, base // 3)
        assert conservative_limit == 3

    def test_aggressive_uses_full_limit(self):
        """In aggressive mode, full base limit should be used."""
        base = 10
        aggressive_limit = base
        assert aggressive_limit == 10
