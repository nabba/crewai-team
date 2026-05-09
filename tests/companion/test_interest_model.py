"""Tests for ``app.companion.interest_model`` (Phase B #1, 2026-05-09)."""
from __future__ import annotations

import json

import pytest


def test_tokenize_basic():
    from app.companion.interest_model import _tokenize
    out = _tokenize("Forest carbon sequestration in Estonia is a key topic.")
    # stopwords + short tokens dropped
    assert "forest" in out
    assert "carbon" in out
    assert "sequestration" in out
    assert "the" not in out
    assert "is" not in out


def test_bigrams():
    from app.companion.interest_model import _bigrams, _tokenize
    grams = _bigrams(_tokenize("forest carbon flux"))
    assert "forest carbon" in grams
    assert "carbon flux" in grams


def test_recency_weight_halflife():
    from app.companion.interest_model import _recency_weight
    fresh = _recency_weight(0.0)
    week_old = _recency_weight(7.0, halflife=7.0)
    two_weeks = _recency_weight(14.0, halflife=7.0)
    assert fresh == pytest.approx(1.0)
    assert week_old == pytest.approx(0.5)
    assert two_weeks == pytest.approx(0.25)


def test_score_terms_min_freq(monkeypatch):
    from app.companion import interest_model

    streams = {
        "convs": [
            ("forest carbon estonia", 0.0),
            ("forest carbon sequestration", 1.0),
        ],
        "emails": [],
        "events": [],
        "feedback": [],
        "affect": [],
    }
    scores = interest_model._score_terms(streams)
    # "forest carbon" appears in 2 docs → above MIN_FREQ
    assert "forest carbon" in scores
    assert "forest" in scores
    # Each term seen in only one doc — still scored, but counts == 1.
    assert scores["forest carbon"]["sources"]["convs"] == 2


def test_diversity_bonus():
    from app.companion.interest_model import _diversity_bonus
    assert _diversity_bonus({"convs": 1}) == 1.0
    assert _diversity_bonus({"convs": 1, "emails": 1}) == pytest.approx(1.1)
    assert _diversity_bonus({
        "convs": 1, "emails": 1, "events": 1, "affect": 1, "feedback": 1,
    }) == pytest.approx(1.4)


def test_compile_writes_profile(tmp_path, monkeypatch):
    from app.companion import interest_model
    monkeypatch.setattr(interest_model, "_PROFILE_PATH",
                        tmp_path / "interest_profile.json")

    monkeypatch.setattr(
        interest_model, "_conversations_text",
        lambda d: iter([
            ("forest carbon flux", 0.0),
            ("forest carbon flux estonia", 1.0),
            ("estonia winter forest data", 2.0),
        ]),
    )
    monkeypatch.setattr(
        interest_model, "_email_subject_text", lambda d: iter([]),
    )
    monkeypatch.setattr(
        interest_model, "_calendar_titles_text", lambda d: iter([]),
    )
    monkeypatch.setattr(
        interest_model, "_feedback_events_text", lambda d: iter([]),
    )
    monkeypatch.setattr(
        interest_model, "_affect_topics_text", lambda d: iter([]),
    )

    profile = interest_model.compile_interest_profile(lookback_days=14)
    assert (tmp_path / "interest_profile.json").exists()
    on_disk = json.loads((tmp_path / "interest_profile.json").read_text())
    assert on_disk["topics"]
    names = [t["name"] for t in on_disk["topics"]]
    assert any("forest" in n for n in names)


def test_current_profile_missing(tmp_path, monkeypatch):
    from app.companion import interest_model
    monkeypatch.setattr(interest_model, "_PROFILE_PATH",
                        tmp_path / "no-such-file.json")
    p = interest_model.current_profile()
    assert p["topics"] == []


def test_run_respects_cadence(tmp_path, monkeypatch):
    from app.companion import interest_model
    from app.life_companion import _common
    monkeypatch.setattr(_common, "_STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(interest_model, "background_enabled", lambda: True)
    monkeypatch.setattr(interest_model, "_PROFILE_PATH",
                        tmp_path / "interest_profile.json")

    calls = []
    def fake_compile(lookback_days=14):
        calls.append(1)
        return {"generated_at": "x", "lookback_days": 14, "topics": []}
    monkeypatch.setattr(interest_model, "compile_interest_profile", fake_compile)

    interest_model.run()
    interest_model.run()  # second within cadence — skipped
    assert len(calls) == 1
