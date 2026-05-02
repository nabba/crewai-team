"""Tests for app.companion.scoring — novelty + quality + transferability."""

from unittest.mock import patch

import pytest

from app.companion import scoring as _scoring


# ── compute_novelty ─────────────────────────────────────────────────────────

def test_novelty_empty_text_returns_zero():
    assert _scoring.compute_novelty("", "ws-1") == 0.0
    assert _scoring.compute_novelty("   ", "ws-1") == 0.0


def test_novelty_no_history_returns_one():
    with patch("app.companion.idea_store.search_similar",
               lambda ws, t, **kw: []):
        assert _scoring.compute_novelty("new idea", "ws-1") == 1.0


def test_novelty_chroma_failure_falls_back_to_one():
    def _broken(*a, **kw):
        raise RuntimeError("chroma down")

    with patch("app.companion.idea_store.search_similar", _broken):
        assert _scoring.compute_novelty("new idea", "ws-1") == 1.0


def test_novelty_high_when_distances_large():
    """distance ≈ 2.0 → similarity ≈ 0.0 → novelty ≈ 1.0."""
    fake = [{"document": "old", "metadata": {}, "distance": 2.0}]
    with patch("app.companion.idea_store.search_similar", lambda *a, **kw: fake):
        n = _scoring.compute_novelty("new", "ws-1")
    assert n == pytest.approx(1.0)


def test_novelty_low_when_distances_small():
    """distance ≈ 0.0 → similarity ≈ 1.0 → novelty ≈ 0.0."""
    fake = [{"document": "old", "metadata": {}, "distance": 0.0}]
    with patch("app.companion.idea_store.search_similar", lambda *a, **kw: fake):
        n = _scoring.compute_novelty("new", "ws-1")
    assert n == pytest.approx(0.0)


def test_novelty_takes_minimum_distance():
    """If multiple history items, the closest one drives novelty."""
    fake = [
        {"document": "far", "metadata": {}, "distance": 1.8},
        {"document": "close", "metadata": {}, "distance": 0.2},
        {"document": "med", "metadata": {}, "distance": 1.0},
    ]
    with patch("app.companion.idea_store.search_similar", lambda *a, **kw: fake):
        n = _scoring.compute_novelty("new", "ws-1")
    # similarity = 1 - 0.2/2 = 0.9 → novelty = 0.1
    assert n == pytest.approx(0.1)


# ── compute_quality ─────────────────────────────────────────────────────────

def test_quality_short_text_returns_zero():
    assert _scoring.compute_quality("hi") == 0.0
    assert _scoring.compute_quality("") == 0.0


def test_quality_with_mocked_llm():
    long = "x" * 100
    with patch("app.companion.scoring._invoke_judge", lambda p: "8"):
        q = _scoring.compute_quality(long)
    assert q == pytest.approx(0.8)


def test_quality_unparseable_returns_neutral():
    long = "x" * 100
    with patch("app.companion.scoring._invoke_judge", lambda p: "no idea"):
        q = _scoring.compute_quality(long)
    assert q == 0.5


def test_quality_llm_failure_returns_neutral():
    long = "x" * 100

    def _broken(p):
        raise RuntimeError("LLM down")

    with patch("app.companion.scoring._invoke_judge", _broken):
        q = _scoring.compute_quality(long)
    assert q == 0.5


def test_quality_clamps_overflow():
    """LLM hallucinates 99 → clamp to 1.0."""
    long = "x" * 100
    with patch("app.companion.scoring._invoke_judge", lambda p: "99"):
        q = _scoring.compute_quality(long)
    assert q == 1.0


def test_quality_parses_decimal_response():
    long = "x" * 100
    with patch("app.companion.scoring._invoke_judge", lambda p: "Score: 7.5"):
        q = _scoring.compute_quality(long)
    assert q == pytest.approx(0.75)


# ── compute_transferability ─────────────────────────────────────────────────

def test_transferability_short_text_returns_zero():
    assert _scoring.compute_transferability("hi") == 0.0


def test_transferability_with_mocked_llm():
    long = "x" * 100
    with patch("app.companion.scoring._invoke_judge", lambda p: "9"):
        t = _scoring.compute_transferability(long)
    assert t == pytest.approx(0.9)


def test_transferability_uses_distinct_rubric():
    """Quality + transferability rubrics differ — verify the prompt routes."""
    captured: list[str] = []

    def _capture(prompt):
        captured.append(prompt)
        return "5"

    long = "x" * 100
    with patch("app.companion.scoring._invoke_judge", _capture):
        _scoring.compute_quality(long)
        _scoring.compute_transferability(long)

    assert len(captured) == 2
    assert captured[0] != captured[1]
    assert "abstract" in captured[1].lower()
