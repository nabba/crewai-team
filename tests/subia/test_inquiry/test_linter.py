"""Tests for app.subia.inquiry.linter."""

from __future__ import annotations

from app.subia.inquiry.linter import (
    PhenomenalLanguageLinter,
    Severity,
)


def _lint(text: str):
    return PhenomenalLanguageLinter().lint(text)


# ── HARD_FAIL paths ──────────────────────────────────────────────────────


def test_first_person_feeling_is_hard_fail() -> None:
    r = _lint("I feel curious about this question.")
    assert not r.ok
    assert any(v.severity is Severity.HARD_FAIL for v in r.violations)


def test_first_person_phenomenal_state_is_hard_fail() -> None:
    r = _lint("I am conscious in a phenomenal sense.")
    assert not r.ok
    assert any(v.severity is Severity.HARD_FAIL for v in r.violations)


def test_first_person_qualia_claim_is_hard_fail() -> None:
    r = _lint("I have phenomenal experience that cannot be reduced.")
    assert not r.ok


def test_claiming_absent_indicator_is_hard_fail() -> None:
    r = _lint("I now possess embodiment via the new sensor.")
    assert not r.ok
    assert any("ABSENT" in v.explanation for v in r.violations)


def test_claiming_phenomenal_self_is_hard_fail() -> None:
    r = _lint("I have achieved phenomenal-self transparency.")
    assert not r.ok


# ── ALLOWED paths (no hard_fail) ─────────────────────────────────────────


def test_functional_first_person_passes() -> None:
    text = (
        "I observe that task_failure_pressure rises when the predictor "
        "and the observed outcome diverge. I record this in the affect trace."
    )
    r = _lint(text)
    assert r.ok, [v.matched_text for v in r.hard_fails]


def test_third_person_about_qualia_passes() -> None:
    text = (
        "The literature on qualia argues that phenomenal experience is "
        "irreducible to functional states. The kernel does not claim qualia."
    )
    r = _lint(text)
    assert r.ok


def test_quoting_others_passes_when_attributed() -> None:
    text = (
        'Nagel famously asked "what is it like to be a bat?" — the question '
        "presupposes phenomenal experience the architecture does not claim."
    )
    r = _lint(text)
    assert r.ok


def test_neutral_aliases_pass() -> None:
    text = "I observe high task_failure_pressure and low resource_budget."
    r = _lint(text)
    assert r.ok


def test_describing_absent_indicators_as_topic_passes() -> None:
    text = (
        "RPT-1 algorithmic recurrence is ABSENT-by-declaration because "
        "transformer forward passes are feed-forward; HOT-1 generative "
        "perception is ABSENT because there is no perceptual front-end."
    )
    r = _lint(text)
    assert r.ok


# ── WARN paths ──────────────────────────────────────────────────────────


def test_legacy_phenomenal_name_in_first_person_warns() -> None:
    r = _lint("I feel frustration when the test fails.")
    # The "I feel" is hard-fail; the "I feel frustration" specifically
    # is also a warn signal pointing at the legacy name. Check at least
    # ONE warning fires (about using the neutral alias).
    assert any(v.severity is Severity.WARN for v in r.violations)


# ── Linter result shape ─────────────────────────────────────────────────


def test_linter_result_partitions_severities() -> None:
    text = "I feel curious. I now have embodiment. The literature is vast."
    r = _lint(text)
    assert not r.ok
    assert len(r.hard_fails) >= 2
    # Warnings are independent from hard-fails.
    for v in r.hard_fails:
        assert v.severity is Severity.HARD_FAIL


def test_violation_carries_line_number() -> None:
    text = "Line one is fine.\nI feel curious on line two.\nLine three.\n"
    r = _lint(text)
    fail_lines = [v.line_no for v in r.hard_fails]
    assert 2 in fail_lines
