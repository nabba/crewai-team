"""Tests for app.subia.inquiry.idle_registration."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.subia.inquiry.composer import ComposerContext
from app.subia.inquiry.idle_registration import (
    PassResult,
    is_enabled,
    run_once,
)


_GOOD_BODY = (
    "## Framing\n\nThe question is functional.\n\n"
    "## Discussion\n\nWhen task_failure_pressure rises, the goal_emitter "
    "may write to current_goals. Convergence with Andrus's interest model "
    "is operationally meaningful when sustained.\n\n"
    "## What remains open\n\nThreshold for divergence remains unsettled.\n"
)


@pytest.fixture
def staged(tmp_path: Path):
    """Stage a questions file + empty inquiries dir under tmp_path."""
    qpath = tmp_path / "questions.md"
    qpath.write_text(
        "## Are the goal sources coherent?\n\n"
        "Framing paragraph for coherence question.\n",
    )
    inquiries = tmp_path / "inquiries"
    inquiries.mkdir()
    return qpath, inquiries


def test_disabled_skip(staged, monkeypatch) -> None:
    qpath, inquiries = staged
    monkeypatch.setenv("INQUIRY_PASS_ENABLED", "false")
    result = run_once(
        llm_call=lambda s, u: _GOOD_BODY,
        questions_path=str(qpath),
        inquiries_dir=str(inquiries),
    )
    assert result.status == "skipped_disabled"
    assert list(inquiries.glob("*.md")) == []


def test_no_questions_skip(tmp_path: Path) -> None:
    inquiries = tmp_path / "inquiries"
    inquiries.mkdir()
    qpath = tmp_path / "missing-questions.md"  # never created
    result = run_once(
        llm_call=lambda s, u: _GOOD_BODY,
        questions_path=str(qpath),
        inquiries_dir=str(inquiries),
    )
    assert result.status == "skipped_no_questions"


def test_writes_essay_on_happy_path(staged) -> None:
    qpath, inquiries = staged
    result = run_once(
        llm_call=lambda s, u: _GOOD_BODY,
        questions_path=str(qpath),
        inquiries_dir=str(inquiries),
    )
    assert result.status == "wrote_essay"
    assert result.question_slug == "are-the-goal-sources-coherent"
    written = Path(result.written_to)
    assert written.exists()
    assert written.parent == inquiries
    text = written.read_text()
    assert "Are the goal sources coherent?" in text


def test_composer_failure_does_not_write(staged) -> None:
    qpath, inquiries = staged
    # LLM returns phenomenal-claim text; linter rejects every retry.
    def bad_llm(s: str, u: str) -> str:
        return "I feel sentient. I have qualia. The end."

    result = run_once(
        llm_call=bad_llm,
        questions_path=str(qpath),
        inquiries_dir=str(inquiries),
    )
    assert result.status == "skipped_composer_failed"
    assert list(inquiries.glob("*.md")) == []


def test_writer_refusal_when_file_exists(staged) -> None:
    qpath, inquiries = staged
    # Pre-create the day's expected output file so writer refuses.
    from datetime import datetime, timezone
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    (inquiries / f"{date}-are-the-goal-sources-coherent.md").write_text("already there")

    result = run_once(
        llm_call=lambda s, u: _GOOD_BODY,
        questions_path=str(qpath),
        inquiries_dir=str(inquiries),
    )
    assert result.status == "skipped_writer_refused"
    assert "already exists" in result.failure_reason


def test_unexpected_llm_exception_is_caught(staged) -> None:
    qpath, inquiries = staged

    def boom(s: str, u: str) -> str:
        raise RuntimeError("boom")

    result = run_once(
        llm_call=boom,
        questions_path=str(qpath),
        inquiries_dir=str(inquiries),
    )
    # The composer catches LLM exceptions internally and surfaces them
    # as a failed essay → idle_registration reports skipped_composer_failed.
    assert result.status == "skipped_composer_failed"
    assert "boom" in result.failure_reason


def test_pass_result_is_frozen() -> None:
    r = PassResult(status="wrote_essay")
    try:
        r.status = "x"  # type: ignore[misc]
    except (AttributeError, Exception):
        return
    raise AssertionError("PassResult should be frozen")


def test_is_enabled_default_true(monkeypatch) -> None:
    monkeypatch.delenv("INQUIRY_PASS_ENABLED", raising=False)
    assert is_enabled() is True


def test_is_enabled_off_when_explicit_false(monkeypatch) -> None:
    monkeypatch.setenv("INQUIRY_PASS_ENABLED", "false")
    assert is_enabled() is False
