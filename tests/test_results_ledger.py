"""Tests for app.results_ledger — TSV experiment tracking."""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Mock config
from tests.test_metrics import _FakeSettings
import app.config as config_mod
config_mod.get_settings = lambda: _FakeSettings()
config_mod.get_anthropic_api_key = lambda: "fake-key"
config_mod.get_gateway_secret = lambda: "a" * 64


class TestResultsLedger:
    def test_record_and_retrieve(self, tmp_path, monkeypatch):
        import app.results_ledger as ledger
        monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "results.tsv")

        # Should start empty
        results = ledger.get_recent_results(10)
        assert results == []

        # Record an experiment
        ledger.record_experiment(
            experiment_id="exp_test_001",
            hypothesis="Test hypothesis",
            change_type="skill",
            metric_before=0.5000,
            metric_after=0.5200,
            status="keep",
            files_changed=["skills/test.md"],
        )

        results = ledger.get_recent_results(10)
        assert len(results) == 1
        assert results[0]["experiment_id"] == "exp_test_001"
        assert results[0]["status"] == "keep"
        assert results[0]["metric_before"] == 0.5
        assert results[0]["metric_after"] == 0.52
        assert results[0]["delta"] == pytest.approx(0.02, abs=1e-4)

    def test_multiple_records(self, tmp_path, monkeypatch):
        import app.results_ledger as ledger
        monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "results.tsv")

        for i in range(5):
            ledger.record_experiment(
                experiment_id=f"exp_{i}",
                hypothesis=f"Hypothesis {i}",
                change_type="skill",
                metric_before=0.5,
                metric_after=0.5 + i * 0.01,
                status="keep" if i % 2 == 0 else "discard",
            )

        results = ledger.get_recent_results(10)
        assert len(results) == 5

    def test_get_best_score(self, tmp_path, monkeypatch):
        import app.results_ledger as ledger
        monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "results.tsv")

        ledger.record_experiment("e1", "h1", "skill", 0.5, 0.55, "keep")
        ledger.record_experiment("e2", "h2", "skill", 0.55, 0.50, "discard")
        ledger.record_experiment("e3", "h3", "skill", 0.55, 0.60, "keep")

        assert ledger.get_best_score() == 0.60

    def test_improvement_trend(self, tmp_path, monkeypatch):
        import app.results_ledger as ledger
        monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "results.tsv")

        ledger.record_experiment("e1", "h1", "skill", 0.5, 0.52, "keep")
        ledger.record_experiment("e2", "h2", "skill", 0.52, 0.50, "discard")
        ledger.record_experiment("e3", "h3", "skill", 0.52, 0.55, "keep")

        trend = ledger.get_improvement_trend(10)
        # Only "keep" results
        assert trend == [0.52, 0.55]

    def test_format_ledger(self, tmp_path, monkeypatch):
        import app.results_ledger as ledger
        monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "results.tsv")

        ledger.record_experiment("e1", "h1", "skill", 0.5, 0.52, "keep")
        text = ledger.format_ledger(10)
        assert "keep" in text
        assert "+0.0200" in text

    def test_tsv_sanitization(self, tmp_path, monkeypatch):
        import app.results_ledger as ledger
        monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "results.tsv")

        # Hypothesis with tabs and newlines should be sanitized
        ledger.record_experiment(
            "e1", "hypo\twith\ttabs\nand newlines", "skill", 0.5, 0.52, "keep"
        )
        content = (tmp_path / "results.tsv").read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 2  # header + 1 data row (no extra newlines)

    def test_empty_ledger_format(self, tmp_path, monkeypatch):
        import app.results_ledger as ledger
        monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "results.tsv")
        assert "No experiments" in ledger.format_ledger()
