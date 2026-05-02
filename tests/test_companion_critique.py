"""Tests for app.companion.critique — five-persona panel."""

from unittest.mock import patch

import pytest

from app.companion import critique as _crit


def test_run_panel_empty_idea_returns_zero():
    rep = _crit.run_panel("", "seed")
    assert rep.aggregate == 0.0
    assert rep.passed is False
    assert rep.scores == []


def test_run_panel_calls_each_persona_once():
    captured: list[str] = []

    def _fake_judge(prompt: str) -> str:
        # Capture which persona is being asked.
        for persona, _, _ in _crit.PERSONAS:
            if persona in prompt:
                captured.append(persona)
                break
        return "SCORE: 4\nRATIONALE: solid"

    with patch("app.companion.critique._invoke_judge", _fake_judge):
        rep = _crit.run_panel("idea body", "seed")

    assert len(captured) == 5
    assert sorted(captured) == sorted(p for p, _, _ in _crit.PERSONAS)
    assert len(rep.scores) == 5


def test_aggregate_is_average_normalised_to_unit():
    """5 personas all score 4/5 → aggregate 0.8."""
    with patch("app.companion.critique._invoke_judge",
               lambda p: "SCORE: 4\nRATIONALE: x"):
        rep = _crit.run_panel("idea body", "seed")
    assert rep.aggregate == pytest.approx(0.8)


def test_passes_above_threshold():
    with patch("app.companion.critique._invoke_judge",
               lambda p: "SCORE: 5\nRATIONALE: x"):
        rep = _crit.run_panel("idea", "seed", threshold=0.6)
    assert rep.passed is True


def test_fails_below_threshold():
    with patch("app.companion.critique._invoke_judge",
               lambda p: "SCORE: 2\nRATIONALE: weak"):
        rep = _crit.run_panel("idea", "seed", threshold=0.6)
    assert rep.passed is False
    assert rep.aggregate == pytest.approx(0.4)


def test_partial_persona_failure_uses_remainder():
    call_count = [0]

    def _flaky(p: str) -> str:
        call_count[0] += 1
        if call_count[0] in (2, 4):
            raise RuntimeError("LLM hiccup")
        return "SCORE: 5\nRATIONALE: x"

    with patch("app.companion.critique._invoke_judge", _flaky):
        rep = _crit.run_panel("idea", "seed")
    # 3 personas succeeded, 2 failed → aggregate based on 3 scores of 5.
    assert len(rep.scores) == 3
    assert rep.aggregate == pytest.approx(1.0)


def test_total_panel_failure_returns_neutral():
    def _broken(p):
        raise RuntimeError("LLM dead")

    with patch("app.companion.critique._invoke_judge", _broken):
        rep = _crit.run_panel("idea", "seed")
    assert rep.scores == []
    assert rep.aggregate == 0.5
    assert rep.passed is False


def test_score_clamped_to_1_5_range():
    """LLM hallucinates 99 → clamp to 5; 0 → clamp to 1."""
    with patch("app.companion.critique._invoke_judge",
               lambda p: "SCORE: 99\nRATIONALE: x"):
        rep = _crit.run_panel("idea", "seed")
    for s in rep.scores:
        assert 1.0 <= s.score <= 5.0


def test_parse_handles_missing_score_marker():
    """Falls back to first standalone digit when SCORE: is missing."""
    score, _ = _crit._parse_persona_response("4 because x")
    assert score == 4.0


def test_parse_neutral_fallback_when_no_number():
    """No number anywhere → neutral 3.0."""
    score, rationale = _crit._parse_persona_response("Sorry, can't help.")
    assert score == 3.0


def test_parse_extracts_rationale():
    raw = "SCORE: 4\nRATIONALE: this is the reason"
    score, rationale = _crit._parse_persona_response(raw)
    assert score == 4.0
    assert "this is the reason" in rationale


def test_parse_truncates_long_rationale():
    long = "x" * 1000
    raw = f"SCORE: 3\nRATIONALE: {long}"
    score, rationale = _crit._parse_persona_response(raw)
    assert len(rationale) <= 300


def test_to_dict_list_shape():
    with patch("app.companion.critique._invoke_judge",
               lambda p: "SCORE: 4\nRATIONALE: yes"):
        rep = _crit.run_panel("idea", "seed")
    out = rep.to_dict_list()
    assert len(out) == 5
    for item in out:
        assert set(item.keys()) == {"persona", "score", "rationale"}


def test_skeptic_persona_has_adversarial_prompt():
    """The Skeptic's dimension should reference adversarial framing —
    catches accidental rubric homogenisation if someone refactors."""
    skeptic = next(p for p in _crit.PERSONAS if p[0] == "Skeptic")
    persona, dimension, rubric = skeptic
    assert "adversarial" in dimension.lower() or "weakest" in dimension.lower()
