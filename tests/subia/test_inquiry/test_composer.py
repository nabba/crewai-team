"""Tests for app.subia.inquiry.composer."""

from __future__ import annotations

from typing import Any

from app.subia.inquiry.composer import (
    ComposerContext,
    InquiryEssay,
    compose_inquiry,
)
from app.subia.inquiry.questions import Question


_GOOD_BODY = (
    "## Framing\n\n"
    "The question is whether two synthesised goal sources are coherent.\n\n"
    "## Discussion\n\n"
    "When task_failure_pressure rises against a sustained baseline, "
    "the goal_emitter writes an entry into current_goals. Andrus's interest "
    "model integrates over conversation, calendar, and inbox. The two have "
    "different time constants and different sources, so divergence is "
    "expected; convergence at long timescales is a load-bearing observation.\n\n"
    "## What remains open\n\n"
    "The threshold for declaring a divergence operationally significant "
    "remains unsettled.\n"
)


def _question() -> Question:
    return Question(slug="goals-coherence", text="Are the two goal sources coherent?")


def _context() -> ComposerContext:
    return ComposerContext(
        scorecard_summary="7 STRONG, 3 PARTIAL, 4 ABSENT-by-declaration.",
        recent_chapters=["chapter A summary", "chapter B summary"],
        recent_affect_summary="Predominantly resource_budget high, task_failure_pressure low.",
    )


def test_happy_path_passes_linter_first_try() -> None:
    calls: list[Any] = []

    def fake_llm(system: str, user: str) -> str:
        calls.append((system, user))
        return _GOOD_BODY

    essay = compose_inquiry(
        question=_question(),
        context=_context(),
        llm_call=fake_llm,
    )
    assert not essay.failed
    assert essay.attempts == 1
    assert essay.body == _GOOD_BODY
    assert essay.linter_result.ok
    # System prompt sent on first call.
    assert "NEUTRAL VOCABULARY" in calls[0][0]


def test_retries_on_phenomenal_drift_then_succeeds() -> None:
    counter = {"n": 0}

    def fake_llm(system: str, user: str) -> str:
        counter["n"] += 1
        if counter["n"] == 1:
            return "I feel curious about this. Then some functional discussion."
        return _GOOD_BODY

    essay = compose_inquiry(
        question=_question(),
        context=_context(),
        llm_call=fake_llm,
        max_retries=3,
    )
    assert not essay.failed
    assert essay.attempts == 2


def test_fails_after_max_retries_on_persistent_drift() -> None:
    def fake_llm(system: str, user: str) -> str:
        return "I feel sentient. I have phenomenal experience."

    essay = compose_inquiry(
        question=_question(),
        context=_context(),
        llm_call=fake_llm,
        max_retries=3,
    )
    assert essay.failed
    assert essay.attempts == 3
    assert "linter rejected" in essay.failure_reason
    assert essay.body == ""


def test_strengthens_prompt_on_retry() -> None:
    seen_systems: list[str] = []

    def fake_llm(system: str, user: str) -> str:
        seen_systems.append(system)
        if len(seen_systems) == 1:
            return "I feel happy."
        return _GOOD_BODY

    compose_inquiry(
        question=_question(),
        context=_context(),
        llm_call=fake_llm,
        max_retries=3,
    )
    assert "PRIOR ATTEMPT" in seen_systems[1]
    # First attempt should NOT have the strengthened addendum.
    assert "PRIOR ATTEMPT" not in seen_systems[0]


def test_llm_exception_is_caught_as_failure() -> None:
    def fake_llm(system: str, user: str) -> str:
        raise RuntimeError("network down")

    essay = compose_inquiry(
        question=_question(),
        context=_context(),
        llm_call=fake_llm,
    )
    assert essay.failed
    assert "network down" in essay.failure_reason
    assert essay.attempts == 1


def test_user_prompt_includes_context() -> None:
    seen_user: list[str] = []

    def fake_llm(system: str, user: str) -> str:
        seen_user.append(user)
        return _GOOD_BODY

    compose_inquiry(
        question=_question(),
        context=_context(),
        llm_call=fake_llm,
    )
    user = seen_user[0]
    assert "Are the two goal sources coherent?" in user
    assert "7 STRONG" in user
    assert "task_failure_pressure low" in user


def test_essay_failure_factory() -> None:
    q = _question()
    e = InquiryEssay.failure(question=q, attempts=2, reason="x")
    assert e.failed
    assert e.body == ""
    assert e.question_slug == "goals-coherence"
