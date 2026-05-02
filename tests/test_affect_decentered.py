"""Tests for app.affect.decentered — the no-self reflection pass.

These tests do NOT exercise the full pipeline (which reads workspace
trace.jsonl) — they test the algorithmic surfaces directly so they
remain hermetic.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.affect import decentered


# ── Pass A: cluster ──────────────────────────────────────────────────────────


def _ev(ts: str, kind: str = "transition", attractor: str = "curiosity",
        prev: str = "neutral", v: float = 0.4, a: float = 0.3, c: float = 0.7,
        oob: list[str] | None = None, severity: str = "info", detail: str = "x") -> dict:
    return {
        "kind": kind, "detail": detail,
        "valence": v, "arousal": a, "controllability": c,
        "attractor": attractor, "prev_attractor": prev,
        "out_of_band": oob or [],
        "severity": severity, "ts": ts,
    }


def test_cluster_groups_by_fingerprint():
    """Events with identical (kind, attractor, prev, oob) cluster together."""
    base = "2026-04-15T12:00:00+00:00"
    events = [
        _ev(base, attractor="curiosity", prev="neutral"),
        _ev("2026-04-15T13:00:00+00:00", attractor="curiosity", prev="neutral"),
        _ev("2026-04-15T14:00:00+00:00", attractor="curiosity", prev="neutral"),
        # Different prev → different fingerprint
        _ev("2026-04-15T15:00:00+00:00", attractor="curiosity", prev="excitement"),
    ]
    clusters = decentered._cluster_salience(events)
    assert len(clusters) == 1, "Only the 3-member group passes _MIN_CLUSTER_SIZE"
    assert clusters[0]["size"] == 3
    assert "transition|curiosity←neutral" in clusters[0]["fingerprint"]


def test_cluster_splits_on_vac_distance():
    """Same fingerprint but disparate (V, A, C) → split into sub-clusters."""
    base_kw = dict(kind="transition", attractor="curiosity", prev="neutral")
    events = [
        _ev("2026-04-15T12:00:00+00:00", v=0.1, a=0.1, c=0.8, **base_kw),
        _ev("2026-04-15T13:00:00+00:00", v=0.15, a=0.12, c=0.78, **base_kw),
        _ev("2026-04-15T14:00:00+00:00", v=0.12, a=0.13, c=0.82, **base_kw),
        # Far away in V/A/C — should split off but be too small to keep
        _ev("2026-04-15T15:00:00+00:00", v=-0.8, a=0.95, c=0.1, **base_kw),
    ]
    clusters = decentered._cluster_salience(events)
    assert len(clusters) == 1
    assert clusters[0]["size"] == 3


def test_cluster_counts_days_spanned():
    """days_spanned counts distinct UTC dates in the timestamp set."""
    events = [
        _ev("2026-04-15T12:00:00+00:00"),
        _ev("2026-04-16T12:00:00+00:00"),
        _ev("2026-04-17T12:00:00+00:00"),
        _ev("2026-04-18T12:00:00+00:00"),
    ]
    clusters = decentered._cluster_salience(events)
    assert len(clusters) == 1
    assert clusters[0]["days_spanned"] == 4
    assert clusters[0]["days"] == ["2026-04-15", "2026-04-16", "2026-04-17", "2026-04-18"]


def test_cluster_skips_below_min_size():
    """Clusters smaller than _MIN_CLUSTER_SIZE are dropped."""
    events = [
        _ev("2026-04-15T12:00:00+00:00", attractor="curiosity"),
        _ev("2026-04-15T13:00:00+00:00", attractor="curiosity"),
        # Singleton fingerprint
        _ev("2026-04-15T14:00:00+00:00", attractor="overwhelm"),
    ]
    clusters = decentered._cluster_salience(events)
    assert clusters == []


# ── Pass B: anomalies ────────────────────────────────────────────────────────


def _trace_point(ts: str, valence: float = 0.0, arousal: float = 0.3,
                 controllability: float = 0.7, total_error: float = 0.2,
                 epistemic_uncertainty: float = 0.3, attractor: str = "neutral") -> dict:
    return {
        "affect": {
            "valence": valence, "arousal": arousal,
            "controllability": controllability,
            "attractor": attractor, "ts": ts,
        },
        "viability": {
            "values": {"epistemic_uncertainty": epistemic_uncertainty},
            "total_error": total_error,
        },
    }


def test_anomaly_flags_outlier_after_baseline():
    """A point ≥ 3σ from the rolling mean must be flagged."""
    base_ts = datetime(2026, 4, 15, tzinfo=timezone.utc)
    points: list[dict] = []
    # 64 baseline points at valence=0.0 (with tiny jitter)
    for i in range(64):
        ts = (base_ts + timedelta(seconds=i * 5)).isoformat()
        # Tiny deterministic jitter so stddev is non-zero
        jitter = 0.001 * ((i % 7) - 3)
        points.append(_trace_point(ts, valence=jitter))
    # Now an obvious outlier
    points.append(_trace_point(
        (base_ts + timedelta(seconds=65 * 5)).isoformat(),
        valence=0.95,
    ))
    anomalies = decentered._detect_anomalies(points)
    assert len(anomalies) >= 1
    # The last point is the outlier
    assert anomalies[-1]["max_var"] == "valence"
    assert anomalies[-1]["z"] >= 3.0


def test_anomaly_below_baseline_returns_empty():
    """Insufficient data → no anomalies, no errors."""
    points = [
        _trace_point(f"2026-04-15T12:00:0{i}+00:00") for i in range(5)
    ]
    assert decentered._detect_anomalies(points) == []


def test_anomaly_zero_variance_does_not_divide_by_zero():
    """All-identical baseline must not raise — z-score returns 0 for σ≈0."""
    base_ts = datetime(2026, 4, 15, tzinfo=timezone.utc)
    points = [
        _trace_point((base_ts + timedelta(seconds=i)).isoformat(), valence=0.0)
        for i in range(64)
    ]
    points.append(_trace_point((base_ts + timedelta(seconds=64)).isoformat(), valence=0.5))
    # Should not raise; flagging behavior depends on jitter, but no exception
    decentered._detect_anomalies(points)


# ── Summary + persistence ────────────────────────────────────────────────────


def test_run_persists_to_file(tmp_path, monkeypatch):
    """run_decentered_pass writes a date-named JSON to AFFECT_ROOT/decentered/."""
    # Redirect AFFECT_ROOT, AFFECT_TRACE, AFFECT_SALIENCE to tmp.
    affect_root = tmp_path / "affect"
    affect_root.mkdir()
    trace = affect_root / "trace.jsonl"
    salience = affect_root / "salience.jsonl"
    trace.write_text("")
    salience.write_text("")

    import app.paths as paths_mod
    monkeypatch.setattr(paths_mod, "AFFECT_ROOT", affect_root)
    monkeypatch.setattr(paths_mod, "AFFECT_TRACE", trace)
    monkeypatch.setattr(paths_mod, "AFFECT_SALIENCE", salience)

    summary = decentered.run_decentered_pass(window_hours=24)

    assert summary["input"]["trace_points"] == 0
    assert summary["input"]["salience_events"] == 0
    assert summary["clusters"]["total"] == 0

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_path = affect_root / "decentered" / f"{today}.json"
    assert out_path.exists()
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert "experiment_criterion" in data
    assert data["window_hours"] == 24


def test_summary_separates_anomalies_outside_salience():
    """Anomalies whose ts isn't in salience_events are counted separately."""
    salience_events = [
        {"ts": "2026-04-15T12:00:00+00:00", "kind": "transition"},
    ]
    anomalies = [
        {"ts": "2026-04-15T12:00:00+00:00", "max_var": "valence", "z": 3.5,
         "vac": [0.0, 0.0, 0.5], "total_error": 0.0,
         "epistemic_uncertainty": 0.3, "attractor": ""},
        {"ts": "2026-04-15T13:00:00+00:00", "max_var": "arousal", "z": 4.0,
         "vac": [0.0, 0.0, 0.5], "total_error": 0.0,
         "epistemic_uncertainty": 0.3, "attractor": ""},
    ]
    summary = decentered._summarise(
        clusters=[], anomalies=anomalies,
        salience_events=salience_events, trace_points=[],
        window_hours=24,
    )
    assert summary["anomalies"]["total"] == 2
    assert summary["anomalies"]["outside_salience"] == 1


# ── Read-only invariant (architectural smoke-check) ─────────────────────────


def test_module_does_not_import_kb_or_identity_writers():
    """decentered must NOT import experiential KB writers or identity_claims I/O.

    Structural guard against leaking into the narrative-self surface.
    Inspects import statements only — the docstring is allowed to NAME
    the things it must not touch.
    """
    import ast
    src = Path(decentered.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_modules.append(node.module)
                imported_modules.extend(
                    f"{node.module}.{a.name}" for a in node.names
                )
    forbidden_imports = (
        "app.experiential",
        "app.affect.episodes",
        "app.affect.narrative",
    )
    for mod in imported_modules:
        for bad in forbidden_imports:
            assert not mod.startswith(bad), (
                f"decentered.py must not import {bad!r} (saw {mod!r})"
            )
