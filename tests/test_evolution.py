"""Tests for app.evolution — autoresearch-style evolution loop."""
import os
import sys
import types
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Mock config
from tests.test_metrics import _FakeSettings
import app.config as config_mod
config_mod.get_settings = lambda: _FakeSettings()
config_mod.get_anthropic_api_key = lambda: "fake-key"
config_mod.get_gateway_secret = lambda: "a" * 64

# Mock crewai module since it's a heavy dependency not installed in test env
_mock_crewai = types.ModuleType("crewai")
_mock_crewai.Agent = type("Agent", (), {"__init__": lambda *a, **kw: None})
_mock_crewai.Task = type("Task", (), {"__init__": lambda *a, **kw: None})
_mock_crewai.Crew = type("Crew", (), {"__init__": lambda *a, **kw: None, "kickoff": lambda s: ""})
_mock_crewai.Process = type("Process", (), {"sequential": "sequential"})
_mock_crewai.LLM = type("LLM", (), {"__init__": lambda *a, **kw: None})
sys.modules.setdefault("crewai", _mock_crewai)

# Mock firebase_reporter
_mock_firebase = types.ModuleType("app.firebase_reporter")
_mock_firebase.crew_started = lambda *a, **kw: "task_0"
_mock_firebase.crew_completed = lambda *a, **kw: None
_mock_firebase.crew_failed = lambda *a, **kw: None
sys.modules["app.firebase_reporter"] = _mock_firebase

# Mock tools that require heavy deps
_mock_web_search = types.ModuleType("app.tools.web_search")
_mock_web_search.web_search = lambda *a, **kw: ""
sys.modules["app.tools.web_search"] = _mock_web_search

_mock_memory = types.ModuleType("app.tools.memory_tool")
_mock_memory.create_memory_tools = lambda **kw: []
sys.modules["app.tools.memory_tool"] = _mock_memory

_mock_file_mgr = types.ModuleType("app.tools.file_manager")
_mock_file_mgr.file_manager = lambda *a, **kw: ""
sys.modules["app.tools.file_manager"] = _mock_file_mgr


class TestLoadProgram:
    def test_load_existing(self, tmp_path, monkeypatch):
        import app.evolution as evo
        program_file = tmp_path / "program.md"
        program_file.write_text("# Test Program\n\nFocus on testing.")
        monkeypatch.setattr(evo, "PROGRAM_PATH", program_file)

        text = evo._load_program()
        assert "Test Program" in text
        assert "Focus on testing" in text

    def test_load_missing(self, tmp_path, monkeypatch):
        import app.evolution as evo
        monkeypatch.setattr(evo, "PROGRAM_PATH", tmp_path / "nonexistent.md")
        text = evo._load_program()
        assert "No program.md found" in text

    def test_truncation(self, tmp_path, monkeypatch):
        import app.evolution as evo
        program_file = tmp_path / "program.md"
        program_file.write_text("x" * 10000)
        monkeypatch.setattr(evo, "PROGRAM_PATH", program_file)

        text = evo._load_program()
        assert len(text) <= 4000


class TestHypothesisDedup:
    def test_same_hypothesis_same_hash(self):
        from app.evolution import _hypothesis_hash
        h1 = _hypothesis_hash("Add error handling for network timeouts")
        h2 = _hypothesis_hash("Add error handling for network timeouts")
        assert h1 == h2

    def test_different_hypothesis_different_hash(self):
        from app.evolution import _hypothesis_hash
        h1 = _hypothesis_hash("Add error handling")
        h2 = _hypothesis_hash("Improve web search")
        assert h1 != h2

    def test_case_insensitive(self):
        from app.evolution import _hypothesis_hash
        h1 = _hypothesis_hash("Add Error Handling")
        h2 = _hypothesis_hash("add error handling")
        assert h1 == h2

    def test_get_tried_hypotheses(self, tmp_path, monkeypatch):
        import app.results_ledger as ledger
        monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "results.tsv")

        ledger.record_experiment("e1", "hypothesis one", "skill", 0.5, 0.52, "keep")
        ledger.record_experiment("e2", "hypothesis two", "skill", 0.52, 0.50, "discard")

        from app.evolution import _get_tried_hypotheses
        tried = _get_tried_hypotheses()
        assert len(tried) == 2


class TestBuildContext:
    def test_context_includes_sections(self, tmp_path, monkeypatch):
        import app.evolution as evo
        import app.results_ledger as ledger
        import app.self_heal as heal

        monkeypatch.setattr(evo, "PROGRAM_PATH", tmp_path / "program.md")
        (tmp_path / "program.md").write_text("# Test Program")
        monkeypatch.setattr(evo, "SKILLS_DIR", tmp_path / "skills")
        (tmp_path / "skills").mkdir()
        monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "results.tsv")
        monkeypatch.setattr(heal, "ERROR_JOURNAL", tmp_path / "errors.json")

        context = evo._build_evolution_context()
        assert "Research Directions" in context
        assert "Current Metrics" in context
        assert "Recent Experiments" in context
        assert "Error Patterns" in context
        assert "Best Score" in context


class TestJournalSummaryCompat:
    def test_backward_compat(self, tmp_path, monkeypatch):
        """get_journal_summary should work (delegates to results_ledger)."""
        import app.results_ledger as ledger
        monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "results.tsv")

        from app.evolution import get_journal_summary
        summary = get_journal_summary(10)
        assert "No experiments" in summary

        ledger.record_experiment("e1", "test", "skill", 0.5, 0.52, "keep")
        summary = get_journal_summary(10)
        assert "keep" in summary
