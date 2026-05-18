"""Tests for app.healing.monitors.elegance_drift.

Verifies:
  * First pass over a fresh workspace fills baseline, alerts no-one.
  * Subsequent regression triggers ``_classify`` to return 'regressed'.
  * History stays capped per file.
  * Disabled state short-circuits.
  * Cadence gate honors ``last_run``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from app.healing.monitors import elegance_drift


@pytest.fixture
def isolated_workspace(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    # Force fresh workspace_root resolution in the module.
    monkeypatch.setattr(elegance_drift, "_workspace_root", lambda: tmp_path)
    return tmp_path


@pytest.fixture
def mini_app(tmp_path: Path, monkeypatch):
    """Make ``_app_root`` point at a tiny app tree we control."""
    app_root = tmp_path / "app"
    app_root.mkdir(parents=True)
    (app_root / "good.py").write_text(
        '''"""A tidy module."""

def add(a: int, b: int) -> int:
    """Add two ints."""
    return a + b
'''
    )
    monkeypatch.setattr(elegance_drift, "_app_root", lambda: app_root)
    return app_root


def test_first_pass_fills_baseline_no_regressors(isolated_workspace, mini_app):
    result = elegance_drift.run()
    assert result["checked"] is True
    assert result["regressors"] == 0
    # The single mini-app file should be in baseline-fill mode.
    assert result["baseline_filled"] == 1
    assert result["scanned"] == 1


def test_disabled_short_circuits(isolated_workspace, mini_app, monkeypatch):
    monkeypatch.setattr(elegance_drift, "_enabled", lambda: False)
    result = elegance_drift.run()
    assert result.get("disabled") is True
    assert result.get("checked") is False


def test_cadence_gate_blocks_rapid_reruns(isolated_workspace, mini_app):
    elegance_drift.run()  # first pass, sets last_run = now
    again = elegance_drift.run()  # immediate re-run
    assert again.get("skipped_cadence") is True


def test_classify_returns_regressed_when_below_median_minus_threshold():
    prior = [0.90, 0.92, 0.91, 0.93]  # median ~ 0.915
    verdict, median = elegance_drift._classify(prior, current=0.70)
    assert verdict == "regressed"
    assert median is not None and abs(median - 0.915) < 0.01


def test_classify_returns_ok_within_threshold():
    prior = [0.90, 0.92, 0.91, 0.93]
    verdict, _ = elegance_drift._classify(prior, current=0.85)
    assert verdict == "ok"


def test_classify_returns_baseline_with_too_little_history():
    verdict, median = elegance_drift._classify([0.9], current=0.50)
    assert verdict == "baseline"
    assert median is None


def test_history_capped_per_file(isolated_workspace, mini_app):
    history: dict[str, list[dict]] = {}
    cap = elegance_drift._HISTORY_CAP_PER_FILE
    for i in range(cap + 10):
        elegance_drift._append_sample(history, "app/x.py", {"ts": str(i), "composite": 0.9})
    assert len(history["app/x.py"]) == cap
    # The latest samples win.
    assert history["app/x.py"][-1]["ts"] == str(cap + 9)


def test_regression_detected_on_second_pass(isolated_workspace, mini_app, monkeypatch):
    """Seed history so the next run sees a regression — exercises the
    median-based regression path end-to-end."""
    history_path = isolated_workspace / "code_quality" / "elegance_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)

    # Seed 8 prior samples for the mini-app's file at composite 0.95.
    rel = "app/good.py"
    history_path.write_text(json.dumps({
        rel: [{"ts": f"2026-05-{d:02d}", "composite": 0.95} for d in range(1, 9)]
    }))

    # Bypass cadence (state file with old last_run).
    state_path = isolated_workspace / "healing" / "elegance_drift_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"last_run": time.time() - 999999.0}))

    # Make the file BAD: drop type hints + docstring.
    (mini_app / "good.py").write_text(
        "def bad():\n"
        "    if True:\n"
        "        if True:\n"
        "            if True:\n"
        "                return 1\n"
    )

    # Silence Signal notifications during the test.
    monkeypatch.setattr(elegance_drift, "_emit_alert", lambda *a, **k: None)

    result = elegance_drift.run()
    assert result["checked"] is True
    # One file regressed; appears in top_regressors.
    assert result["regressors"] == 1
    assert result["top_regressors"][0]["path"] == rel
