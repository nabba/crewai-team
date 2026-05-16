"""Q14 (PROGRAM §49) — Year-2+ risk register.

Six items from §10 of the operator's audit framework: small-amendment
drift, hidden feedback loops, Goodhart on internal metrics, embedding-
space drift, interest-model ossification, live-process lock contention.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _isolated_module(rel_path: str, mod_name: str):
    spec = importlib.util.spec_from_file_location(
        mod_name, REPO_ROOT / rel_path,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ═════════════════════════════════════════════════════════════════════════
#   10.1 — Identity drift digest
# ═════════════════════════════════════════════════════════════════════════


def test_drift_digest_module_exists() -> None:
    p = REPO_ROOT / "app" / "identity" / "drift_digest.py"
    assert p.is_file()
    src = p.read_text()
    assert "def compute_digest" in src
    assert "def briefing_section" in src
    assert "def run" in src
    assert "summarise_drift" in src


def test_drift_digest_acceleration_below_threshold_silent(monkeypatch) -> None:
    """30d count consistent with annual rate → no alert."""
    mod = _isolated_module(
        "app/identity/drift_digest.py", "_q14_drift_digest_under_test",
    )

    class _FakeSummary:
        def __init__(self, n, by_kind):
            self.n_events = n
            self.by_kind = by_kind

    def _fake_summarise(window_days, now=None):
        # 1 event/month for 12 months → no acceleration.
        return _FakeSummary(1 if window_days == 30 else 12, {"x": 1})

    fake_cl = type(sys)("app.identity.continuity_ledger")
    fake_cl.summarise_drift = _fake_summarise
    fake_cl.record_event = lambda **kw: True
    monkeypatch.setitem(
        sys.modules, "app.identity.continuity_ledger", fake_cl,
    )
    digest = mod.compute_digest()
    # 1 / (12/12) = 1.0 acceleration; below 2.0 threshold.
    assert digest.aggregate_acceleration == 1.0
    assert digest.counts_30d == 1
    section = mod.briefing_section()
    assert section is None  # silent on routine


def test_drift_digest_acceleration_above_threshold_alerts(monkeypatch, tmp_path) -> None:
    mod = _isolated_module(
        "app/identity/drift_digest.py", "_q14_drift_digest_alert",
    )

    class _FakeSummary:
        def __init__(self, n, by_kind):
            self.n_events = n
            self.by_kind = by_kind

    def _fake_summarise(window_days, now=None):
        if window_days == 30:
            return _FakeSummary(30, {"tier3_amendment": 25, "soul_edit": 5})
        if window_days == 90:
            return _FakeSummary(45, {"tier3_amendment": 30, "soul_edit": 15})
        return _FakeSummary(60, {"tier3_amendment": 40, "soul_edit": 20})

    fake_cl = type(sys)("app.identity.continuity_ledger")
    fake_cl.summarise_drift = _fake_summarise
    captured_events: list = []
    fake_cl.record_event = lambda **kw: captured_events.append(kw) or True
    monkeypatch.setitem(
        sys.modules, "app.identity.continuity_ledger", fake_cl,
    )
    monkeypatch.setattr(
        mod, "_state_path", lambda: tmp_path / "state.json",
    )
    monkeypatch.setattr(
        mod, "_digest_path", lambda: tmp_path / "digest.json",
    )

    notify_capture: list = []
    fake_notify = type(sys)("app.notify")
    fake_notify.notify = lambda **kw: notify_capture.append(kw)
    monkeypatch.setitem(sys.modules, "app.notify", fake_notify)

    summary = mod.run()
    assert summary["ran"] is True
    # 30 / (60/12) = 6.0 → alert.
    assert summary["aggregate_acceleration"] == 6.0
    assert summary["alert_fired"] is True
    assert summary["landmark_emitted"] is True
    assert len(notify_capture) == 1
    assert "Identity drift acceleration" in notify_capture[0]["title"]
    assert captured_events[0]["kind"] == "identity_drift_acceleration"


def test_drift_digest_per_kind_acceleration_detected(monkeypatch, tmp_path) -> None:
    """Single kind running hot triggers the per-kind threshold even if
    aggregate is OK."""
    mod = _isolated_module(
        "app/identity/drift_digest.py", "_q14_drift_digest_per_kind",
    )

    class _FakeSummary:
        def __init__(self, n, by_kind):
            self.n_events = n
            self.by_kind = by_kind

    def _fake_summarise(window_days, now=None):
        if window_days == 30:
            # Aggregate count of 6 (annual avg / 12 * 12 = 12, so below
            # aggregate threshold). But "soul_edit" 5/month is 5× its
            # annual monthly average → per-kind alert fires.
            return _FakeSummary(6, {"tier3_amendment": 1, "soul_edit": 5})
        return _FakeSummary(72, {"tier3_amendment": 60, "soul_edit": 12})

    fake_cl = type(sys)("app.identity.continuity_ledger")
    fake_cl.summarise_drift = _fake_summarise
    fake_cl.record_event = lambda **kw: True
    monkeypatch.setitem(
        sys.modules, "app.identity.continuity_ledger", fake_cl,
    )

    digest = mod.compute_digest()
    assert "soul_edit" in digest.accelerated_kinds


# ═════════════════════════════════════════════════════════════════════════
#   10.2 — Influence graph + cycle detection + meta-agent loop probe
# ═════════════════════════════════════════════════════════════════════════


def test_influence_graph_edges_exist() -> None:
    """The curated edges list exists and is non-empty."""
    p = REPO_ROOT / "app" / "healing" / "influence_graph" / "edges.py"
    assert p.is_file()
    mod = _isolated_module(
        "app/healing/influence_graph/edges.py", "_q14_edges",
    )
    assert len(mod.EDGES) > 30, "graph should cover the ~30 idle jobs"
    # Every edge has a stable shape.
    for e in mod.EDGES:
        assert e.producer
        assert e.consumer
        assert e.signal


def test_cycle_detector_finds_meta_agent_loop() -> None:
    """The named loop (meta_agent.selector → factory → lifecycle → store
    → selector) IS detected as a cycle. This is the user's specific
    failure case in §10.2 — the cycle MUST be visible."""
    cycles_mod = _isolated_module(
        "app/healing/influence_graph/cycles.py", "_q14_cycles",
    )
    # Import the real edges via the cycles module's default path.
    edges_mod = _isolated_module(
        "app/healing/influence_graph/edges.py", "_q14_edges_for_cycles",
    )
    cycles = cycles_mod.find_cycles(edges_mod.EDGES)
    nodes_in_cycles = {n for c in cycles for n in c.nodes}
    assert "meta_agent.selector" in nodes_in_cycles, (
        "the meta-agent recipe-selection loop must be detected — "
        "this is the cycle the §10.2 drift probe watches"
    )


def test_cycle_detector_returns_stable_output(monkeypatch) -> None:
    """Same edge list → same cycles → output ordering is stable."""
    cycles_mod = _isolated_module(
        "app/healing/influence_graph/cycles.py", "_q14_cycles_stable",
    )
    # Build a synthetic edge list with a 3-cycle and a 2-cycle.
    EdgeKind = type("EK", (), {"DATA": "data"})

    class _E:
        def __init__(self, p, c):
            self.producer = p
            self.consumer = c
            self.signal = "x"
    edges = [
        _E("a", "b"), _E("b", "c"), _E("c", "a"),
        _E("x", "y"), _E("y", "x"),
        _E("z", "w"),  # not a cycle
    ]
    c1 = cycles_mod.find_cycles(edges)
    c2 = cycles_mod.find_cycles(edges)
    assert [c.nodes for c in c1] == [c.nodes for c in c2]
    # 2 cycles (3-node + 2-node), one non-cycle edge ignored.
    assert len(c1) == 2


def test_feedback_loop_drift_gini_monotonic_increase_alerts(monkeypatch, tmp_path) -> None:
    """The probe alerts when the meta-agent recipe-selection Gini
    increases monotonically over 4 weekly samples."""
    mod = _isolated_module(
        "app/healing/monitors/feedback_loop_drift.py",
        "_q14_feedback_drift_alert",
    )
    monkeypatch.setattr(mod, "_state_path", lambda: tmp_path / "state.json")
    monkeypatch.setattr(mod, "_history_path", lambda: tmp_path / "hist.jsonl")
    # Pre-populate the history with 3 monotonic-increasing samples
    # (will be augmented with the current sample to reach the
    # _MIN_SAMPLES_FOR_TREND threshold of 4).
    hist_path = tmp_path / "hist.jsonl"
    hist_path.write_text(
        "\n".join(json.dumps({"ts": i, "iso": "x", "gini": g, "n_recipes": 10})
                 for i, g in enumerate([0.30, 0.40, 0.50])) + "\n",
        encoding="utf-8",
    )

    # Stub meta_agent store to return increasingly concentrated uses.
    fake_recipes = []
    for i, uses in enumerate([100, 5, 5, 3, 2, 1, 1]):
        r = type("R", (), {"uses": uses})()
        fake_recipes.append(r)
    fake_store = type(sys)("app.self_improvement.meta_agent.store")
    fake_store.list_recipes = lambda limit=500, include_null=False: fake_recipes
    monkeypatch.setitem(
        sys.modules, "app.self_improvement.meta_agent.store", fake_store,
    )

    notify_capture: list = []
    fake_notify = type(sys)("app.notify")
    fake_notify.notify = lambda **kw: notify_capture.append(kw)
    monkeypatch.setitem(sys.modules, "app.notify", fake_notify)

    fake_cl = type(sys)("app.identity.continuity_ledger")
    captured_events: list = []
    fake_cl.record_event = lambda **kw: captured_events.append(kw) or True
    monkeypatch.setitem(
        sys.modules, "app.identity.continuity_ledger", fake_cl,
    )

    summary = mod.run()
    assert summary["ran"] is True
    assert summary["n_recipes"] == 7
    assert summary["gini"] > 0.6  # very concentrated
    assert summary["monotonic_increase"] is True
    assert summary["alert_fired"] is True
    assert summary["ledger_emitted"] is True
    assert len(notify_capture) == 1
    assert "Feedback-loop drift" in notify_capture[0]["title"]
    assert captured_events[0]["kind"] == "feedback_loop_drift"


def test_feedback_loop_drift_no_alert_on_stable_distribution(monkeypatch, tmp_path) -> None:
    """Stable uniform Gini → no alert."""
    mod = _isolated_module(
        "app/healing/monitors/feedback_loop_drift.py",
        "_q14_feedback_drift_stable",
    )
    monkeypatch.setattr(mod, "_state_path", lambda: tmp_path / "state.json")
    monkeypatch.setattr(mod, "_history_path", lambda: tmp_path / "hist.jsonl")

    # Pre-populate flat history.
    hist_path = tmp_path / "hist.jsonl"
    hist_path.write_text(
        "\n".join(json.dumps({"ts": i, "iso": "x", "gini": 0.20, "n_recipes": 10})
                 for i in range(5)) + "\n",
        encoding="utf-8",
    )
    # Uniform uses distribution.
    fake_recipes = [type("R", (), {"uses": 10})() for _ in range(10)]
    fake_store = type(sys)("app.self_improvement.meta_agent.store")
    fake_store.list_recipes = lambda limit=500, include_null=False: fake_recipes
    monkeypatch.setitem(
        sys.modules, "app.self_improvement.meta_agent.store", fake_store,
    )
    summary = mod.run()
    assert summary["monotonic_increase"] is False
    assert summary["alert_fired"] is False


def test_feedback_loop_drift_gini_formula() -> None:
    mod = _isolated_module(
        "app/healing/monitors/feedback_loop_drift.py",
        "_q14_feedback_drift_gini_pure",
    )
    # Perfect uniform → Gini close to 0.
    assert abs(mod.gini([5, 5, 5, 5, 5])) < 0.01
    # Perfect inequality → Gini close to (n-1)/n.
    n = 5
    expected = (n - 1) / n
    assert abs(mod.gini([0, 0, 0, 0, 100]) - expected) < 0.01


# ═════════════════════════════════════════════════════════════════════════
#   10.3 — Goodhart recipe-selection-divergence detector
# ═════════════════════════════════════════════════════════════════════════


def test_goodhart_guard_has_recipe_divergence_detector() -> None:
    """Source-level: goodhart_guard now defines the new detector."""
    p = REPO_ROOT / "app" / "goodhart_guard.py"
    src = p.read_text()
    assert "_detect_recipe_selection_divergence" in src
    assert "recipe_selection_divergence" in src
    assert "_DIVERGENCE_MIN_OUTCOMES" in src
    # Detector should be called from detect_gaming_signals.
    assert "signals.extend(_detect_recipe_selection_divergence" in src


# ═════════════════════════════════════════════════════════════════════════
#   10.4 — Embedding-model drift
# ═════════════════════════════════════════════════════════════════════════


def test_embedding_drift_first_run_initializes_baseline(monkeypatch, tmp_path) -> None:
    """First run with no baseline → persists baseline, no alerts."""
    mod = _isolated_module(
        "app/healing/monitors/embedding_drift.py",
        "_q14_embedding_drift_first",
    )
    monkeypatch.setattr(mod, "_state_path", lambda: tmp_path / "state.json")
    monkeypatch.setattr(mod, "_baseline_path", lambda: tmp_path / "baseline.json")

    fake_rs = type(sys)("app.runtime_settings")
    fake_rs.get_embedding_drift_monitor_enabled = lambda: True
    monkeypatch.setitem(sys.modules, "app.runtime_settings", fake_rs)

    # Deterministic embed: hash of text → fake vec.
    def _embed(text: str):
        h = abs(hash(text)) % 1000
        return [float(h % 7), float(h % 11), float(h % 13)]

    summary = mod.run(embed_fn=_embed, anchor_texts=("a", "b", "c"))
    assert summary["ran"] is True
    assert summary["baseline_initialized"] is True
    assert summary["n_baselined"] == 3
    assert summary["n_diverged"] == 0
    assert summary["alerts_fired"] == 0


def test_embedding_drift_second_run_no_change_silent(monkeypatch, tmp_path) -> None:
    """If embeddings match baseline → no alert."""
    mod = _isolated_module(
        "app/healing/monitors/embedding_drift.py",
        "_q14_embedding_drift_match",
    )
    monkeypatch.setattr(mod, "_state_path", lambda: tmp_path / "state.json")
    monkeypatch.setattr(mod, "_baseline_path", lambda: tmp_path / "baseline.json")
    fake_rs = type(sys)("app.runtime_settings")
    fake_rs.get_embedding_drift_monitor_enabled = lambda: True
    monkeypatch.setitem(sys.modules, "app.runtime_settings", fake_rs)

    def _embed(text: str):
        return [1.0, 0.5, 0.25]

    # First run with fixed `now` so we control the cadence gate.
    mod.run(embed_fn=_embed, anchor_texts=("a", "b", "c"), now=1.0)
    # Second run > CADENCE_SECONDS later.
    summary = mod.run(
        embed_fn=_embed, anchor_texts=("a", "b", "c"),
        now=1.0 + 14 * 86400,
    )
    assert summary["n_diverged"] == 0
    assert summary["alerts_fired"] == 0


def test_embedding_drift_silent_swap_triggers_alert(monkeypatch, tmp_path) -> None:
    """Vendor silently swaps model → cosine drops → alert fires."""
    mod = _isolated_module(
        "app/healing/monitors/embedding_drift.py",
        "_q14_embedding_drift_swap",
    )
    monkeypatch.setattr(mod, "_state_path", lambda: tmp_path / "state.json")
    monkeypatch.setattr(mod, "_baseline_path", lambda: tmp_path / "baseline.json")
    fake_rs = type(sys)("app.runtime_settings")
    fake_rs.get_embedding_drift_monitor_enabled = lambda: True
    monkeypatch.setitem(sys.modules, "app.runtime_settings", fake_rs)

    embed_versions = {"v1": True}

    def _embed(text: str):
        if embed_versions["v1"]:
            return [1.0, 0.0, 0.0]
        # Silent swap: completely different vector.
        return [0.0, 1.0, 0.0]

    # First run — baseline at v1. Use a fixed `now` so the cadence
    # gate doesn't lock us out of the second run.
    mod.run(embed_fn=_embed, anchor_texts=("a", "b", "c"), now=1.0)
    # Switch model.
    embed_versions["v1"] = False

    notify_capture: list = []
    fake_notify = type(sys)("app.notify")
    fake_notify.notify = lambda **kw: notify_capture.append(kw)
    monkeypatch.setitem(sys.modules, "app.notify", fake_notify)

    fake_cl = type(sys)("app.identity.continuity_ledger")
    captured_events: list = []
    fake_cl.record_event = lambda **kw: captured_events.append(kw) or True
    monkeypatch.setitem(
        sys.modules, "app.identity.continuity_ledger", fake_cl,
    )

    # Second run > CADENCE_SECONDS later so the cadence gate passes.
    summary = mod.run(
        embed_fn=_embed, anchor_texts=("a", "b", "c"),
        now=1.0 + 14 * 86400,
    )
    assert summary["n_diverged"] == 3  # all 3 anchors flipped
    assert summary["alerts_fired"] == 1
    assert "drift" in notify_capture[0]["title"].lower()
    assert captured_events[0]["kind"] == "embedding_model_swap"


# ═════════════════════════════════════════════════════════════════════════
#   10.5 — Interest-model ossification
# ═════════════════════════════════════════════════════════════════════════


def test_interest_ossification_entropy_pure() -> None:
    mod = _isolated_module(
        "app/healing/monitors/interest_ossification.py",
        "_q14_interest_ossification_entropy",
    )
    # Perfect uniform → entropy ratio = 1.0.
    assert abs(mod._shannon_entropy_ratio([1, 1, 1, 1, 1]) - 1.0) < 0.01
    # All mass on one → entropy ratio = 0.
    assert mod._shannon_entropy_ratio([100, 0, 0, 0, 0]) < 0.05


def test_interest_ossification_concentrated_alerts(monkeypatch, tmp_path) -> None:
    mod = _isolated_module(
        "app/healing/monitors/interest_ossification.py",
        "_q14_interest_concentrated",
    )
    monkeypatch.setattr(mod, "_state_path", lambda: tmp_path / "state.json")
    # Build a top-30 with one topic absolutely dominant.
    profile = {
        "topics": (
            [{"name": "dom", "score": 100.0}]
            + [{"name": f"t{i}", "score": 0.01} for i in range(29)]
        )
    }
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile))
    monkeypatch.setattr(mod, "_profile_path", lambda: profile_path)

    fake_rs = type(sys)("app.runtime_settings")
    fake_rs.get_interest_ossification_monitor_enabled = lambda: True
    monkeypatch.setitem(sys.modules, "app.runtime_settings", fake_rs)
    notify_capture: list = []
    fake_notify = type(sys)("app.notify")
    fake_notify.notify = lambda **kw: notify_capture.append(kw)
    monkeypatch.setitem(sys.modules, "app.notify", fake_notify)
    fake_cl = type(sys)("app.identity.continuity_ledger")
    captured_events: list = []
    fake_cl.record_event = lambda **kw: captured_events.append(kw) or True
    monkeypatch.setitem(
        sys.modules, "app.identity.continuity_ledger", fake_cl,
    )

    summary = mod.run()
    assert summary["ran"] is True
    assert summary["entropy_ratio"] < 0.30
    assert "interest_concentrated" in summary["alerts"]
    assert len(notify_capture) == 1
    assert captured_events[0]["kind"] == "interest_ossification"


def test_interest_ossification_diffuse_alerts(monkeypatch, tmp_path) -> None:
    mod = _isolated_module(
        "app/healing/monitors/interest_ossification.py",
        "_q14_interest_diffuse",
    )
    monkeypatch.setattr(mod, "_state_path", lambda: tmp_path / "state.json")
    profile = {"topics": [{"name": f"t{i}", "score": 1.0} for i in range(30)]}
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile))
    monkeypatch.setattr(mod, "_profile_path", lambda: profile_path)
    fake_rs = type(sys)("app.runtime_settings")
    fake_rs.get_interest_ossification_monitor_enabled = lambda: True
    monkeypatch.setitem(sys.modules, "app.runtime_settings", fake_rs)
    monkeypatch.setitem(sys.modules, "app.notify", type(sys)("app.notify"))
    sys.modules["app.notify"].notify = lambda **kw: None
    monkeypatch.setitem(
        sys.modules, "app.identity.continuity_ledger",
        type(sys)("app.identity.continuity_ledger"),
    )
    sys.modules["app.identity.continuity_ledger"].record_event = lambda **kw: True

    summary = mod.run()
    assert summary["entropy_ratio"] > 0.90
    assert "interest_diffuse" in summary["alerts"]


def test_interest_ossification_ossified_after_4_weeks(monkeypatch, tmp_path) -> None:
    """4 consecutive weeks of high Jaccard overlap → ossified alert."""
    mod = _isolated_module(
        "app/healing/monitors/interest_ossification.py",
        "_q14_interest_ossified",
    )
    state_path = tmp_path / "state.json"
    monkeypatch.setattr(mod, "_state_path", lambda: state_path)
    # Mid-entropy profile (avoids concentrated + diffuse alerts).
    profile = {"topics": [
        {"name": f"t{i}", "score": float(30 - i)} for i in range(30)
    ]}
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile))
    monkeypatch.setattr(mod, "_profile_path", lambda: profile_path)
    fake_rs = type(sys)("app.runtime_settings")
    fake_rs.get_interest_ossification_monitor_enabled = lambda: True
    monkeypatch.setitem(sys.modules, "app.runtime_settings", fake_rs)

    notify_capture: list = []
    fake_notify = type(sys)("app.notify")
    fake_notify.notify = lambda **kw: notify_capture.append(kw)
    monkeypatch.setitem(sys.modules, "app.notify", fake_notify)
    fake_cl = type(sys)("app.identity.continuity_ledger")
    fake_cl.record_event = lambda **kw: True
    monkeypatch.setitem(
        sys.modules, "app.identity.continuity_ledger", fake_cl,
    )

    # Seed state with 3 prior weeks of high overlap.
    state_path.write_text(json.dumps({
        "last_run_at": 0.0,
        "last_top30_names": [t["name"] for t in profile["topics"]],
        "consecutive_high_overlap_weeks": 3,
        "last_alert_at": {},
    }))
    # 4th week — same names → triggers ossified alert.
    summary = mod.run(now=10_000_000.0)
    assert summary["ran"] is True
    assert summary["consecutive_high_overlap_weeks"] == 4
    assert "interest_ossified" in summary["alerts"]


# ═════════════════════════════════════════════════════════════════════════
#   10.6 — Lock contention
# ═════════════════════════════════════════════════════════════════════════


def test_lock_metrics_module_exists() -> None:
    p = REPO_ROOT / "app" / "utils" / "lock_metrics.py"
    assert p.is_file()
    src = p.read_text()
    assert "def record_write_timing" in src
    assert "def time_write" in src
    assert "_RECORD_THRESHOLD_MS" in src


def test_lock_metrics_below_threshold_silent(monkeypatch, tmp_path) -> None:
    """Writes faster than 50ms are NOT recorded."""
    mod = _isolated_module(
        "app/utils/lock_metrics.py", "_q14_lock_metrics_below",
    )
    log_path = tmp_path / "lock_waits.jsonl"
    monkeypatch.setattr(mod, "_log_path", lambda: log_path)
    monkeypatch.setattr(mod, "_workspace", lambda: tmp_path)
    mod.record_write_timing("foo.json", 10.0)  # < 50ms
    assert not log_path.exists() or not log_path.read_text().strip()


def test_lock_metrics_above_threshold_records(monkeypatch, tmp_path) -> None:
    mod = _isolated_module(
        "app/utils/lock_metrics.py", "_q14_lock_metrics_above",
    )
    log_path = tmp_path / "lock_waits.jsonl"
    monkeypatch.setattr(mod, "_log_path", lambda: log_path)
    monkeypatch.setattr(mod, "_workspace", lambda: tmp_path)
    # Use a path inside the test workspace so _resource_label can
    # resolve it.
    target_path = tmp_path / "foo" / "bar.json"
    mod.record_write_timing(str(target_path), 150.0)
    assert log_path.exists()
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert "foo" in row["resource"]
    assert row["elapsed_ms"] == 150.0


def test_lock_metrics_context_manager_records_slow(monkeypatch, tmp_path) -> None:
    import time as _time
    mod = _isolated_module(
        "app/utils/lock_metrics.py", "_q14_lock_metrics_cm",
    )
    log_path = tmp_path / "lock_waits.jsonl"
    monkeypatch.setattr(mod, "_log_path", lambda: log_path)
    monkeypatch.setattr(mod, "_workspace", lambda: tmp_path)
    with mod.time_write(str(tmp_path / "slow.json")):
        _time.sleep(0.08)  # > 50ms
    assert log_path.exists()
    rows = log_path.read_text().strip().splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["elapsed_ms"] >= 50


def test_lock_contention_monitor_alerts_on_high_p99(monkeypatch, tmp_path) -> None:
    mod = _isolated_module(
        "app/healing/monitors/lock_contention.py",
        "_q14_lock_contention",
    )
    log_path = tmp_path / "lock_waits.jsonl"
    state_path = tmp_path / "state.json"
    monkeypatch.setattr(mod, "_log_path", lambda: log_path)
    monkeypatch.setattr(mod, "_state_path", lambda: state_path)
    fake_rs = type(sys)("app.runtime_settings")
    fake_rs.get_lock_contention_monitor_enabled = lambda: True
    monkeypatch.setitem(sys.modules, "app.runtime_settings", fake_rs)
    notify_capture: list = []
    fake_notify = type(sys)("app.notify")
    fake_notify.notify = lambda **kw: notify_capture.append(kw)
    monkeypatch.setitem(sys.modules, "app.notify", fake_notify)

    import time as _time
    now_ts = _time.time()
    # Write 15 high-latency rows for one resource (p99 ~ 800ms).
    lines: list[str] = []
    for elapsed in [100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 750, 800, 850, 900, 950]:
        lines.append(json.dumps({
            "ts": now_ts - 3600,
            "resource": "companion/interest_profile.json",
            "elapsed_ms": elapsed,
        }))
    log_path.write_text("\n".join(lines) + "\n")
    summary = mod.run(now=now_ts)
    assert summary["ran"] is True
    assert summary["n_rows"] == 15
    assert summary["n_resources"] == 1
    assert len(summary["high_p99_resources"]) == 1
    assert summary["alerts"] == 1
    assert "Lock contention" in notify_capture[0]["title"]


def test_safe_io_wrapped_with_timing(monkeypatch) -> None:
    """Source-level: safe_io.safe_write + safe_append wrap with lock_metrics.time_write."""
    p = REPO_ROOT / "app" / "safe_io.py"
    src = p.read_text()
    assert "from app.utils.lock_metrics import time_write" in src
    # Both safe_write + safe_append go through the wrapper.
    assert src.count("from app.utils.lock_metrics import time_write") >= 2


# ═════════════════════════════════════════════════════════════════════════
#   Wiring tests
# ═════════════════════════════════════════════════════════════════════════


def test_continuity_ledger_has_q14_kinds() -> None:
    p = REPO_ROOT / "app" / "identity" / "continuity_ledger.py"
    src = p.read_text()
    for kind in (
        "identity_drift_acceleration",
        "feedback_loop_drift",
        "embedding_model_swap",
        "interest_ossification",
    ):
        assert f'"{kind}"' in src, f"missing kind {kind}"


def test_runtime_settings_has_all_q14_switches() -> None:
    p = REPO_ROOT / "app" / "runtime_settings.py"
    src = p.read_text()
    for key in (
        "identity_drift_digest_enabled",
        "feedback_loop_drift_monitor_enabled",
        "embedding_drift_monitor_enabled",
        "interest_ossification_monitor_enabled",
        "lock_contention_monitor_enabled",
        "influence_graph_monitor_enabled",
    ):
        assert f'"{key}"' in src
        assert f"def get_{key}" in src
        assert f"def set_{key}" in src


def test_monitors_init_registers_all_q14_monitors() -> None:
    p = REPO_ROOT / "app" / "healing" / "monitors" / "__init__.py"
    src = p.read_text()
    for monitor in (
        "identity_drift_digest",
        "feedback_loop_drift",
        "embedding_drift",
        "interest_ossification",
        "lock_contention",
    ):
        assert f'"{monitor}"' in src, f"monitor {monitor} not registered"


def test_influence_graph_package_init_exports() -> None:
    p = REPO_ROOT / "app" / "healing" / "influence_graph" / "__init__.py"
    src = p.read_text()
    assert "find_cycles" in src
    assert "EDGES" in src
