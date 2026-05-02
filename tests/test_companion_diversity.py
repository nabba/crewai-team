"""Tests for app.companion.diversity — workspace-scoped MAP-Elites hook."""

from unittest.mock import patch

import pytest

from app.companion import diversity as _div


# ── workspace_role ─────────────────────────────────────────────────────────

def test_workspace_role_prefix():
    assert _div.workspace_role("ws-1") == "companion:ws-1"


def test_workspace_role_sanitises_unsafe_chars():
    role = _div.workspace_role("../../etc/passwd")
    # Slashes + dots stripped → role still inside the companion namespace.
    assert role.startswith("companion:")
    assert "/" not in role
    assert "." not in role


def test_workspace_role_empty_falls_back():
    assert _div.workspace_role("") == "companion:default"
    assert _div.workspace_role("!!!") == "companion:default"


# ── record_cycle ───────────────────────────────────────────────────────────

def test_record_empty_text_returns_false():
    assert _div.record_cycle("ws-1", "i_x", "", 0.8) is False
    assert _div.record_cycle("ws-1", "i_x", "   ", 0.8) is False


def test_record_none_fitness_returns_false():
    assert _div.record_cycle("ws-1", "i_x", "body", None) is False


def test_record_calls_map_elites_with_workspace_role():
    captured: list = []

    def _capture(outcome):
        captured.append(outcome)
        return True

    with patch("app.companion.diversity._invoke_record", _capture):
        ok = _div.record_cycle("ws-1", "i_abc", "the polished idea body",
                                fitness=0.85, panel_score=0.9)

    assert ok is True
    assert len(captured) == 1
    outcome = captured[0]
    assert outcome.crew_name == "companion:ws-1"
    assert outcome.has_result is True
    assert outcome.confidence == "high"  # fitness 0.85 ≥ 0.7
    assert outcome.completeness == "complete"
    assert outcome.passed_quality_gate is True
    assert "i_abc" in outcome.task_description


def test_record_low_fitness_marks_failure_pattern():
    captured: list = []
    with patch("app.companion.diversity._invoke_record",
               lambda o: captured.append(o) or True):
        _div.record_cycle("ws-1", "i_x", "body", fitness=0.1)
    assert captured[0].is_failure_pattern is True
    assert captured[0].confidence == "low"
    assert captured[0].passed_quality_gate is False


def test_record_medium_fitness_partial_completeness():
    captured: list = []
    with patch("app.companion.diversity._invoke_record",
               lambda o: captured.append(o) or True):
        _div.record_cycle("ws-1", "i_x", "body", fitness=0.45)
    # 0.45 → medium / partial / no quality-gate
    assert captured[0].confidence == "medium"
    assert captured[0].completeness == "partial"


def test_record_clamps_fitness_above_one():
    captured: list = []
    with patch("app.companion.diversity._invoke_record",
               lambda o: captured.append(o) or True):
        _div.record_cycle("ws-1", "i_x", "body", fitness=99.0)
    assert captured[0].confidence == "high"


def test_record_absorbs_map_elites_failure():
    def _broken(outcome):
        raise RuntimeError("MAP-Elites schema missing")

    with patch("app.companion.diversity._invoke_record", _broken):
        ok = _div.record_cycle("ws-1", "i_x", "body", 0.7)
    assert ok is False


def test_record_returns_false_when_record_returns_falsy():
    with patch("app.companion.diversity._invoke_record", lambda o: False):
        assert _div.record_cycle("ws-1", "i_x", "body", 0.7) is False


# ── sparse_cell_hints ──────────────────────────────────────────────────────

def test_sparse_cell_hints_zero_limit_returns_empty():
    assert _div.sparse_cell_hints("ws-1", max_hints=0) == []


def test_sparse_cell_hints_returns_formatted_strings():
    fake_voids = [
        {"key": (1, 5, 9),
         "feature_target": {"complexity": 0.15, "cost_efficiency": 0.55,
                              "specialization": 0.95},
         "neighbor_count": 3,
         "mean_neighbor_fitness": 0.72},
        {"key": (8, 2, 1),
         "feature_target": {"complexity": 0.85, "cost_efficiency": 0.25,
                              "specialization": 0.15},
         "neighbor_count": 2,
         "mean_neighbor_fitness": 0.61},
    ]
    with patch("app.companion.diversity._invoke_get_voids",
               lambda ws: fake_voids):
        hints = _div.sparse_cell_hints("ws-1", max_hints=2)
    assert len(hints) == 2
    assert "complexity=0.15" in hints[0]
    assert "specialization=0.95" in hints[0]


def test_sparse_cell_hints_respects_max_hints():
    voids = [{"feature_target": {"complexity": v / 10}} for v in range(10)]
    with patch("app.companion.diversity._invoke_get_voids",
               lambda ws: voids):
        hints = _div.sparse_cell_hints("ws-1", max_hints=3)
    assert len(hints) == 3


def test_sparse_cell_hints_absorbs_lookup_failure():
    def _broken(ws):
        raise RuntimeError("schema missing")

    with patch("app.companion.diversity._invoke_get_voids", _broken):
        assert _div.sparse_cell_hints("ws-1") == []


def test_sparse_cell_hints_empty_void_returns_fallback():
    """Void with no feature_target falls back to a generic hint string."""
    with patch("app.companion.diversity._invoke_get_voids",
               lambda ws: [{}]):
        hints = _div.sparse_cell_hints("ws-1")
    assert hints == ["Explore an under-tried direction."]


# ── coverage ───────────────────────────────────────────────────────────────

def test_coverage_returns_report():
    fake_report = {"filled_cells": 42, "total_cells": 1000,
                    "coverage": 0.042, "best_fitness": 0.91}
    with patch("app.companion.diversity._invoke_coverage",
               lambda ws: fake_report):
        out = _div.coverage("ws-1")
    assert out == fake_report


def test_coverage_absorbs_failure():
    def _broken(ws):
        raise RuntimeError("DB down")

    with patch("app.companion.diversity._invoke_coverage", _broken):
        assert _div.coverage("ws-1") == {}
