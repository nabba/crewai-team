"""Tests for app.subia.inquiry.questions."""

from __future__ import annotations

from pathlib import Path

from app.subia.inquiry.questions import (
    Question,
    load_questions,
    parse_questions,
    slugify,
)


def test_slugify_strips_punctuation_and_lowercases() -> None:
    assert slugify("What is X?") == "what-is-x"
    assert slugify("How would my self-model differ?") == "how-would-my-self-model-differ"
    assert slugify("Memory & narrative — same thing?") == "memory-narrative-same-thing"


def test_slugify_handles_empty_and_unicode() -> None:
    assert slugify("") == "untitled"
    assert slugify("???") == "untitled"
    assert slugify("très bien") != ""  # should not raise


def test_parse_questions_extracts_h2_headings() -> None:
    src = """# Inquiry questions

## What is the meaning?

Some framing.

## Another question

More framing.
Multi-line.
"""
    qs = parse_questions(src)
    assert len(qs) == 2
    assert qs[0].text == "What is the meaning?"
    assert qs[0].slug == "what-is-the-meaning"
    assert "Some framing." in qs[0].framing
    assert qs[1].text == "Another question"
    assert "Multi-line." in qs[1].framing


def test_parse_questions_disambiguates_duplicate_slugs() -> None:
    src = """## Duplicate

framing A

## Duplicate

framing B
"""
    qs = parse_questions(src)
    assert len(qs) == 2
    assert qs[0].slug == "duplicate"
    assert qs[1].slug == "duplicate-2"


def test_load_questions_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_questions(tmp_path / "not-there.md") == []


def test_load_questions_round_trip(tmp_path: Path) -> None:
    f = tmp_path / "q.md"
    f.write_text("## A\n\nframing\n\n## B\n")
    qs = load_questions(f)
    assert [q.text for q in qs] == ["A", "B"]


def test_question_dataclass_is_frozen() -> None:
    q = Question(slug="x", text="X")
    try:
        q.text = "Y"  # type: ignore[misc]
    except (AttributeError, Exception):
        return
    raise AssertionError("Question should be frozen")
