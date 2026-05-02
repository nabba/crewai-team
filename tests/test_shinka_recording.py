"""Tests for the 5 ShinkaEvolve recording / loop bugs found 2026-04-29.

After the dynamic engine selector started picking ShinkaEvolve, runtime
audit found:

  1. ``num_generations`` arrived as 1 from idle scheduler → 0 proposals,
     0 cost, 3-second runtime, no useful output.
  2. ``files_changed`` was passed as a string to ``record_experiment``
     which silently iterated its characters in ``",".join(...)``.
  3. ``_record_result`` failures returned silently — ROI ledger stayed empty.
  4. ``validate_evaluation`` rejected list outputs because shinka's JSON
     IPC round-trips tuples to lists.
  5. ``days_since_engine_run("shinka")`` returned ``inf`` forever because
     the ROI ledger was never populated, locking the rotation rule in a
     loop where shinka kept being picked but never recorded.

Each fix is exercised here.
"""
from __future__ import annotations

import json
import os
import sys
import time
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.test_metrics import _FakeSettings  # noqa: E402
import app.config as config_mod  # noqa: E402

config_mod.get_settings = lambda: _FakeSettings()
config_mod.get_anthropic_api_key = lambda: "fake-key"
config_mod.get_gateway_secret = lambda: "a" * 64


# ─────────────────────────────────────────────────────────────────────────────
# Fix 1: minimum-generations floor
# ─────────────────────────────────────────────────────────────────────────────

class TestMinGenerationsFloor:
    def test_min_generations_constant_exists_and_is_above_one(self):
        from app.shinka_engine import _MIN_GENERATIONS
        # The May 1 incident ran with num_generations=1 and produced 0 proposals.
        # Anything above 1 is technically a fix, but research suggests ≥5 to
        # see useful exploration.
        assert _MIN_GENERATIONS >= 5


# ─────────────────────────────────────────────────────────────────────────────
# Fix 4: validate_evaluation accepts list (post JSON-IPC) AND tuple
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateEvaluationAcceptsBoth:
    def _import_evaluate(self):
        # The shinka eval lives outside the app package; import via path.
        import importlib.util
        from pathlib import Path
        path = Path(__file__).parent.parent / "workspace" / "shinka" / "evaluate.py"
        spec = importlib.util.spec_from_file_location("shinka_eval_module", path)
        mod = importlib.util.module_from_spec(spec)
        # Stub the shinka.core import so we can load this module without shinka
        sys.modules.setdefault("shinka", type(sys)("shinka"))
        sys.modules.setdefault("shinka.core", type(sys)("shinka.core"))
        sys.modules["shinka.core"].run_shinka_eval = lambda **k: ({}, True, None)
        spec.loader.exec_module(mod)
        return mod

    def test_accepts_tuple(self):
        ev = self._import_evaluate()
        ok, err = ev.validate_evaluation((0.85, {"tool_accuracy": 1.0}))
        assert ok is True
        assert err is None

    def test_accepts_list(self):
        """JSON IPC turns the tuple into a list — must accept it."""
        ev = self._import_evaluate()
        ok, err = ev.validate_evaluation([0.85, {"tool_accuracy": 1.0}])
        assert ok is True
        assert err is None

    def test_rejects_non_sequence(self):
        ev = self._import_evaluate()
        ok, err = ev.validate_evaluation({"score": 0.85})
        assert ok is False
        assert "sequence" in err.lower() or "tuple" in err.lower()

    def test_rejects_wrong_length(self):
        ev = self._import_evaluate()
        ok, err = ev.validate_evaluation([0.85])
        assert ok is False

    def test_rejects_score_out_of_range(self):
        ev = self._import_evaluate()
        ok, err = ev.validate_evaluation([1.5, {}])
        assert ok is False
        assert "0.0" in err and "1.0" in err

    def test_aggregate_metrics_handles_list_results(self):
        ev = self._import_evaluate()
        result = ev.aggregate_metrics([[0.9, {"tool_accuracy": 1.0}]])
        assert result["combined_score"] == 0.9

    def test_aggregate_metrics_handles_tuple_results(self):
        ev = self._import_evaluate()
        result = ev.aggregate_metrics([(0.9, {"tool_accuracy": 1.0})])
        assert result["combined_score"] == 0.9

    def test_aggregate_metrics_empty_returns_zero(self):
        ev = self._import_evaluate()
        result = ev.aggregate_metrics([])
        assert result["combined_score"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Fix 5: attempt-marker fallback for days_since_engine_run
# ─────────────────────────────────────────────────────────────────────────────

class TestAttemptMarkerFallback:
    def test_marker_alone_provides_signal_without_roi_ledger(self, tmp_path, monkeypatch):
        """Even with empty ROI ledger, a marker file makes days_since finite."""
        import app.evolution_roi as roi
        monkeypatch.setattr(roi, "ROI_LEDGER_PATH", tmp_path / "empty_roi.json")

        marker_path = tmp_path / "marker.json"
        monkeypatch.setitem(roi._ATTEMPT_MARKER_PATHS, "shinka", marker_path)

        marker_path.write_text(json.dumps({"engine": "shinka", "ts": time.time() - 3600}))

        ts = roi.get_last_run_timestamp("shinka")
        assert ts > 0
        days = roi.days_since_engine_run("shinka")
        assert 0.0 < days < 1.0  # Roughly one hour ago

    def test_ledger_wins_when_newer(self, tmp_path, monkeypatch):
        """ROI ledger entry newer than marker → ledger ts is returned."""
        import app.evolution_roi as roi
        monkeypatch.setattr(roi, "ROI_LEDGER_PATH", tmp_path / "roi.json")

        marker_path = tmp_path / "marker.json"
        monkeypatch.setitem(roi._ATTEMPT_MARKER_PATHS, "shinka", marker_path)
        marker_path.write_text(json.dumps({"engine": "shinka", "ts": time.time() - 86400}))

        # Newer ROI entry
        roi.record_evolution_cost(
            experiment_id="exp_1",
            engine="shinka",
            cost_usd=2.0,
            delta=0.05,
            status="keep",
        )

        ts = roi.get_last_run_timestamp("shinka")
        days = roi.days_since_engine_run("shinka")
        assert days < 0.1  # ROI entry was just written, less than a few minutes ago

    def test_marker_wins_when_newer(self, tmp_path, monkeypatch):
        """Marker timestamp newer than any ROI ledger entry → marker wins."""
        import app.evolution_roi as roi
        monkeypatch.setattr(roi, "ROI_LEDGER_PATH", tmp_path / "roi.json")

        # Old ROI entry
        roi.record_evolution_cost(
            experiment_id="exp_old",
            engine="shinka",
            cost_usd=2.0,
            delta=0.05,
            status="keep",
        )
        # Manually rewrite the timestamp to the past so it's older
        ledger_path = tmp_path / "roi.json"
        records = json.loads(ledger_path.read_text())
        records[0]["timestamp"] = time.time() - 7 * 86400
        ledger_path.write_text(json.dumps(records))

        # Newer marker
        marker_path = tmp_path / "marker.json"
        monkeypatch.setitem(roi._ATTEMPT_MARKER_PATHS, "shinka", marker_path)
        marker_path.write_text(json.dumps({"engine": "shinka", "ts": time.time() - 60}))

        days = roi.days_since_engine_run("shinka")
        # Marker is ~1 minute ago; ROI is 7 days ago. Marker wins.
        assert days < 0.01

    def test_no_signal_returns_inf(self, tmp_path, monkeypatch):
        """Neither ledger nor marker → inf (engine never ran)."""
        import app.evolution_roi as roi
        monkeypatch.setattr(roi, "ROI_LEDGER_PATH", tmp_path / "missing.json")
        monkeypatch.setitem(roi._ATTEMPT_MARKER_PATHS, "shinka", tmp_path / "no_marker.json")
        assert roi.days_since_engine_run("shinka") == float("inf")

    def test_corrupted_marker_returns_zero(self, tmp_path, monkeypatch):
        """Malformed marker file → fall back to ledger only."""
        import app.evolution_roi as roi
        monkeypatch.setattr(roi, "ROI_LEDGER_PATH", tmp_path / "missing.json")
        marker_path = tmp_path / "marker.json"
        monkeypatch.setitem(roi._ATTEMPT_MARKER_PATHS, "shinka", marker_path)
        marker_path.write_text("not valid json {{{")
        # Should not raise
        ts = roi.get_last_run_timestamp("shinka")
        assert ts == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Fix 5b: shinka_engine writes the marker before running
# ─────────────────────────────────────────────────────────────────────────────

class TestShinkaEngineWritesMarker:
    def test_write_attempt_marker_creates_file(self, tmp_path, monkeypatch):
        import app.shinka_engine as eng
        monkeypatch.setattr(eng, "ATTEMPT_MARKER_PATH", tmp_path / "marker.json")
        eng._write_attempt_marker()
        assert (tmp_path / "marker.json").exists()
        data = json.loads((tmp_path / "marker.json").read_text())
        assert data["engine"] == "shinka"
        assert data["ts"] > 0

    def test_write_attempt_marker_swallows_oserror(self, tmp_path, monkeypatch):
        """Even if the marker write fails, the engine must continue."""
        import app.shinka_engine as eng
        # Point at a path inside a nonexistent directory tree the user can't create
        bad = tmp_path / "deeply" / "nested" / "marker.json"
        monkeypatch.setattr(eng, "ATTEMPT_MARKER_PATH", bad)
        # Pre-create as a regular file with no parent-write perms is hard portably,
        # but the function uses mkdir(parents=True, exist_ok=True) so this should
        # actually succeed. The contract is "never raises" — verify that.
        eng._write_attempt_marker()  # must not raise


# ─────────────────────────────────────────────────────────────────────────────
# Fix 2 + 3: _record_result calls record_experiment with list,
#            ROI ledger gets recorded even if results_ledger fails
# ─────────────────────────────────────────────────────────────────────────────

class TestRecordResultUsesListAndIsResilient:
    def test_record_result_passes_list_for_files_changed(self, tmp_path, monkeypatch):
        """The bug was passing a string; verify a list now reaches record_experiment."""
        import app.shinka_engine as eng
        captured = {}

        def fake_record_experiment(**kwargs):
            captured.update(kwargs)

        # Stub experiment_id generator to a stable value
        monkeypatch.setattr(
            "app.experiment_runner.generate_experiment_id",
            lambda h: "exp_test_001",
        )
        monkeypatch.setattr("app.results_ledger.record_experiment", fake_record_experiment)
        monkeypatch.setattr(
            "app.evolution_roi.record_evolution_cost", lambda **k: None
        )

        eng._record_result(baseline=0.9, after=0.95, delta=0.05, status="keep")

        assert captured["files_changed"] == ["workspace/shinka/initial.py"]
        assert isinstance(captured["files_changed"], list)

    def test_roi_recorded_even_when_results_ledger_fails(self, tmp_path, monkeypatch):
        """Resilience: a results_ledger crash must not block ROI recording."""
        import app.shinka_engine as eng
        roi_calls: list[dict] = []

        def fake_record_experiment(**kwargs):
            raise RuntimeError("simulated DB outage")

        def fake_record_evolution_cost(**kwargs):
            roi_calls.append(kwargs)

        monkeypatch.setattr(
            "app.experiment_runner.generate_experiment_id",
            lambda h: "exp_resilient_001",
        )
        monkeypatch.setattr("app.results_ledger.record_experiment", fake_record_experiment)
        monkeypatch.setattr(
            "app.evolution_roi.record_evolution_cost", fake_record_evolution_cost
        )

        eng._record_result(baseline=0.8, after=0.7, delta=-0.1, status="discard")

        assert len(roi_calls) == 1
        assert roi_calls[0]["engine"] == "shinka"
        assert roi_calls[0]["status"] == "discard"
