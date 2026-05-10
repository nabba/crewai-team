"""Tests for app.creativity.novelty_wrap."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.creativity.novelty_wrap import (
    NoveltyVerdict,
    assess_brainstorm_idea,
)


@dataclass
class _FakeReport:
    decision: str
    nearest_distance: float
    nearest_collection: str
    nearest_text: str = ""


def _nr_returning(decision: str, distance: float = 0.5, collection: str = "skills"):
    def fake(_text: str) -> Any:
        return _FakeReport(
            decision=decision,
            nearest_distance=distance,
            nearest_collection=collection,
        )

    return fake


def _lc_returning(matches: list[dict]):
    def fake(_text: str, _k: int) -> list[dict]:
        return list(matches)

    return fake


# ── basic verdict projection ────────────────────────────────────────────


def test_novel_when_neither_check_matches() -> None:
    out = assess_brainstorm_idea(
        "a fresh idea no one has had",
        novelty_report_fn=_nr_returning("novel", distance=0.95),
        lessons_check_fn=_lc_returning([]),
    )
    assert out.verdict is NoveltyVerdict.NOVEL
    assert out.primary_decision == "novel"
    assert out.rejected_lesson_id is None


def test_restated_when_kb_covered() -> None:
    out = assess_brainstorm_idea(
        "rehash of a known thing",
        novelty_report_fn=_nr_returning("covered", distance=0.05),
        lessons_check_fn=_lc_returning([]),
    )
    assert out.verdict is NoveltyVerdict.RESTATED
    assert out.primary_decision == "covered"


def test_recombination_when_kb_overlap() -> None:
    out = assess_brainstorm_idea(
        "a remix of two known patterns",
        novelty_report_fn=_nr_returning("overlap", distance=0.40),
        lessons_check_fn=_lc_returning([]),
    )
    assert out.verdict is NoveltyVerdict.RECOMBINATION


def test_rejected_before_when_lessons_match_strongly() -> None:
    out = assess_brainstorm_idea(
        "a previously-rejected proposal coming around again",
        novelty_report_fn=_nr_returning("novel", distance=0.95),
        lessons_check_fn=_lc_returning([{"id": "lesson-42", "score": 0.80}]),
    )
    assert out.verdict is NoveltyVerdict.REJECTED_BEFORE
    assert out.rejected_lesson_id == "lesson-42"
    assert out.rejected_score == 0.80


def test_weak_lesson_match_does_not_trigger_rejected_verdict() -> None:
    """A lesson match below the rejected_match_threshold is ignored —
    a weak semantic overlap shouldn't gate brainstorm output."""
    out = assess_brainstorm_idea(
        "an idea with weak resemblance to a rejected one",
        novelty_report_fn=_nr_returning("novel", distance=0.85),
        lessons_check_fn=_lc_returning([{"id": "lesson-7", "score": 0.45}]),
    )
    assert out.verdict is NoveltyVerdict.NOVEL
    assert out.rejected_lesson_id is None


def test_threshold_override() -> None:
    """The rejected_match_threshold parameter actually controls the gate."""
    matches = [{"id": "lesson-1", "score": 0.50}]
    weak = assess_brainstorm_idea(
        "x",
        novelty_report_fn=_nr_returning("novel"),
        lessons_check_fn=_lc_returning(matches),
        rejected_match_threshold=0.55,
    )
    assert weak.verdict is NoveltyVerdict.NOVEL

    strict = assess_brainstorm_idea(
        "x",
        novelty_report_fn=_nr_returning("novel"),
        lessons_check_fn=_lc_returning(matches),
        rejected_match_threshold=0.45,
    )
    assert strict.verdict is NoveltyVerdict.REJECTED_BEFORE


# ── failure isolation ──────────────────────────────────────────────────


def test_novelty_report_exception_falls_back_to_novel() -> None:
    def boom(_):
        raise RuntimeError("ChromaDB unreachable")

    out = assess_brainstorm_idea(
        "x",
        novelty_report_fn=boom,
        lessons_check_fn=_lc_returning([]),
    )
    assert out.verdict is NoveltyVerdict.NOVEL
    assert any("novelty_report failed" in n for n in out.notes)


def test_lessons_check_exception_does_not_drop_kb_signal() -> None:
    """If lessons_learned is down, we still use the novelty_report's verdict."""
    def boom(_t, _k):
        raise RuntimeError("Mem0 down")

    out = assess_brainstorm_idea(
        "x",
        novelty_report_fn=_nr_returning("covered"),
        lessons_check_fn=boom,
    )
    assert out.verdict is NoveltyVerdict.RESTATED
    assert any("lessons_check failed" in n for n in out.notes)


def test_both_checks_failing_returns_novel_with_notes() -> None:
    def boom(*_a, **_k):
        raise RuntimeError("everything is on fire")

    out = assess_brainstorm_idea(
        "x",
        novelty_report_fn=boom,
        lessons_check_fn=boom,
    )
    assert out.verdict is NoveltyVerdict.NOVEL
    assert len(out.notes) == 2


# ── edge cases ────────────────────────────────────────────────────────


def test_empty_input_returns_novel_with_note() -> None:
    out = assess_brainstorm_idea(
        "  ",
        novelty_report_fn=_nr_returning("novel"),  # should never be called
        lessons_check_fn=_lc_returning([]),
    )
    assert out.verdict is NoveltyVerdict.NOVEL
    assert any("empty input" in n for n in out.notes)


def test_dict_shape_report_works_alongside_dataclass() -> None:
    """The wrapper should accept both NoveltyReport-shape and plain dict."""
    out = assess_brainstorm_idea(
        "x",
        novelty_report_fn=lambda _: {
            "decision": "overlap",
            "nearest_distance": 0.4,
            "nearest_collection": "skills",
        },
        lessons_check_fn=_lc_returning([]),
    )
    assert out.verdict is NoveltyVerdict.RECOMBINATION
    assert out.primary_distance == 0.4
    assert out.primary_collection == "skills"


def test_decision_with_value_attribute_unwrapped() -> None:
    """Real NoveltyReport returns a NoveltyDecision enum with .value."""

    class _FakeEnum:
        value = "covered"

    @dataclass
    class _ReportWithEnum:
        decision: Any
        nearest_distance: float = 0.1
        nearest_collection: str = "x"

    out = assess_brainstorm_idea(
        "x",
        novelty_report_fn=lambda _: _ReportWithEnum(decision=_FakeEnum()),
        lessons_check_fn=_lc_returning([]),
    )
    assert out.verdict is NoveltyVerdict.RESTATED
    assert out.primary_decision == "covered"
