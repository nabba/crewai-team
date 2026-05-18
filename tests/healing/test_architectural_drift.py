"""Tests for app.healing.monitors.architectural_drift.

Covers:
  * Tarjan SCC finds simple cycles
  * Capability owners are deduped per file
  * First run never alerts (baseline-only)
  * Systemic SCCs (size > _MAX_ALERTABLE_CYCLE_SIZE) are tracked but
    excluded from the actionable "new cycle" list
  * New cycle on second run triggers regression
  * Disabled / cadence gates
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from app.healing.monitors import architectural_drift


@pytest.fixture
def isolated_workspace(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setattr(architectural_drift, "_workspace_root", lambda: tmp_path)
    return tmp_path


def test_tarjan_finds_simple_two_node_cycle():
    graph = {"a.py": ["b.py"], "b.py": ["a.py"], "c.py": []}
    sccs = architectural_drift._strongly_connected_components(graph)
    assert sccs == [["a.py", "b.py"]]


def test_tarjan_finds_three_node_cycle():
    graph = {
        "a.py": ["b.py"],
        "b.py": ["c.py"],
        "c.py": ["a.py"],
        "d.py": [],
    }
    sccs = architectural_drift._strongly_connected_components(graph)
    assert sccs == [["a.py", "b.py", "c.py"]]


def test_tarjan_ignores_trivial_singletons():
    """A node that imports nothing else is not a 'cycle'."""
    graph = {"a.py": ["b.py"], "b.py": []}
    sccs = architectural_drift._strongly_connected_components(graph)
    assert sccs == []


def test_capability_extraction_handles_list_kwarg():
    source = '''
@register_tool(name="x", capabilities=["does-a", "does-b"])
def factory():
    return None
'''
    assert sorted(architectural_drift._capabilities(source)) == ["does-a", "does-b"]


def test_capability_owners_deduped_when_one_file_has_multiple_decorators(tmp_path, monkeypatch):
    app_root = tmp_path / "app"
    app_root.mkdir()
    (app_root / "multi.py").write_text(
        '@register_tool(name="a", capabilities=["dup"])\n'
        'def f1(): pass\n'
        '@register_tool(name="b", capabilities=["dup"])\n'
        'def f2(): pass\n'
    )
    monkeypatch.setattr(architectural_drift, "_app_root", lambda: app_root)
    forward, owners = architectural_drift._build_graph(app_root)
    # Single file with two same-capability decorators → owner listed once.
    assert owners["dup"] == ["app/multi.py"]


def test_new_cycles_excludes_systemic_size():
    big = ["m{:03d}.py".format(i) for i in range(architectural_drift._MAX_ALERTABLE_CYCLE_SIZE + 1)]
    small = ["x.py", "y.py"]
    found = architectural_drift._new_cycles([big, small], baseline_cycles=[])
    assert small in found
    assert big not in found  # systemic size — excluded from actionable list


def test_first_run_no_alert(isolated_workspace, monkeypatch, tmp_path):
    """First pass over a tiny clean codebase fills baseline without alerting."""
    app_root = tmp_path / "app"
    app_root.mkdir()
    (app_root / "a.py").write_text("import app.b\n")
    (app_root / "b.py").write_text("def f(): pass\n")
    monkeypatch.setattr(architectural_drift, "_app_root", lambda: app_root)

    alert_calls: list = []
    monkeypatch.setattr(architectural_drift, "_emit_alert", lambda *a, **k: alert_calls.append(a))

    result = architectural_drift.run()
    assert result["checked"] is True
    assert result["first_run"] is True
    assert alert_calls == []  # no alert on first run
    # Baseline file written.
    assert (isolated_workspace / "code_quality" / "architectural_baseline.json").exists()


def test_new_cycle_on_second_run_triggers_alert(isolated_workspace, monkeypatch, tmp_path):
    app_root = tmp_path / "app"
    app_root.mkdir()
    (app_root / "a.py").write_text("def f(): pass\n")
    (app_root / "b.py").write_text("def g(): pass\n")
    monkeypatch.setattr(architectural_drift, "_app_root", lambda: app_root)
    alert_calls: list = []
    monkeypatch.setattr(architectural_drift, "_emit_alert", lambda *a, **k: alert_calls.append(a))

    # Pass 1 — clean baseline, no cycle.
    architectural_drift.run()
    assert alert_calls == []

    # Bypass cadence for pass 2.
    state_path = isolated_workspace / "healing" / "architectural_drift_state.json"
    state_path.write_text(json.dumps({"last_run": time.time() - 999999.0}))

    # Pass 2 — introduce a cycle.
    (app_root / "a.py").write_text("from app import b\n")
    (app_root / "b.py").write_text("from app import a\n")
    result = architectural_drift.run()

    assert result["checked"] is True
    assert result["first_run"] is False
    assert result["n_new_cycles"] >= 1
    assert len(alert_calls) == 1  # one alert fired


def test_disabled_short_circuits(isolated_workspace, monkeypatch):
    monkeypatch.setattr(architectural_drift, "_enabled", lambda: False)
    result = architectural_drift.run()
    assert result.get("disabled") is True
    assert result.get("checked") is False


def test_cadence_gate_blocks_rapid_reruns(isolated_workspace, monkeypatch, tmp_path):
    app_root = tmp_path / "app"
    app_root.mkdir()
    (app_root / "a.py").write_text("def f(): pass\n")
    monkeypatch.setattr(architectural_drift, "_app_root", lambda: app_root)
    monkeypatch.setattr(architectural_drift, "_emit_alert", lambda *a, **k: None)

    architectural_drift.run()
    again = architectural_drift.run()
    assert again.get("skipped_cadence") is True


def test_systemic_growth_detection():
    prior = [[f"m{i}.py" for i in range(50)]]  # one 50-file SCC
    current = [[f"m{i}.py" for i in range(70)]]  # grown to 70
    growth = architectural_drift._systemic_growth(current, prior)
    assert growth is not None
    assert growth["prior_size"] == 50
    assert growth["current_size"] == 70
    assert growth["delta"] == 20


def test_systemic_growth_silent_when_shrinking():
    prior = [[f"m{i}.py" for i in range(50)]]
    current = [[f"m{i}.py" for i in range(45)]]
    assert architectural_drift._systemic_growth(current, prior) is None
