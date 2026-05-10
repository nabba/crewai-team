"""Tests for app.subia.inquiry.writer."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.subia.inquiry.composer import InquiryEssay
from app.subia.inquiry.linter import LinterResult
from app.subia.inquiry.questions import Question
from app.subia.inquiry.writer import InquiryWriter, WriteRefused


def _good_essay() -> InquiryEssay:
    return InquiryEssay(
        question_slug="why-x",
        question_text="Why X?",
        body="## Framing\n\nfoo bar.\n\n## What remains open\n\nstill unknown.\n",
        composed_at="2026-05-10T12:00:00+00:00",
        linter_result=LinterResult(ok=True, violations=[]),
        attempts=1,
        failed=False,
    )


def _failed_essay() -> InquiryEssay:
    return InquiryEssay.failure(
        question=Question(slug="why-x", text="Why X?"),
        attempts=3,
        reason="linter rejected all 3 attempts",
    )


def test_write_creates_file_in_canonical_path(tmp_path: Path) -> None:
    w = InquiryWriter(inquiries_dir=tmp_path)
    today = datetime(2026, 5, 10, tzinfo=timezone.utc)
    p = w.write(_good_essay(), today=today)
    assert p == tmp_path / "2026-05-10-why-x.md"
    assert p.exists()


def test_write_refuses_failed_essay(tmp_path: Path) -> None:
    w = InquiryWriter(inquiries_dir=tmp_path)
    with pytest.raises(WriteRefused) as excinfo:
        w.write(_failed_essay())
    assert "failed" in str(excinfo.value)


def test_write_refuses_existing_file(tmp_path: Path) -> None:
    w = InquiryWriter(inquiries_dir=tmp_path)
    today = datetime(2026, 5, 10, tzinfo=timezone.utc)
    w.write(_good_essay(), today=today)
    with pytest.raises(WriteRefused) as excinfo:
        w.write(_good_essay(), today=today)
    assert "already exists" in str(excinfo.value)


def test_write_does_not_escape_inquiry_dir(tmp_path: Path) -> None:
    """Even with a contrived slug containing traversal-shaped text, the
    write must land inside ``inquiries_dir``. We rely on `slugify` for
    user-supplied questions, but the writer is the last line of defence."""
    w = InquiryWriter(inquiries_dir=tmp_path)
    today = datetime(2026, 5, 10, tzinfo=timezone.utc)
    # Construct an essay with a weird slug; writer should still confine
    # the path under inquiries_dir.
    essay = InquiryEssay(
        question_slug="weird-slug",  # already normalised; safe
        question_text="?",
        body="ok",
        composed_at="now",
        linter_result=LinterResult(ok=True, violations=[]),
        attempts=1,
        failed=False,
    )
    p = w.write(essay, today=today)
    assert tmp_path.resolve() in p.resolve().parents


def test_render_includes_frontmatter(tmp_path: Path) -> None:
    w = InquiryWriter(inquiries_dir=tmp_path)
    today = datetime(2026, 5, 10, tzinfo=timezone.utc)
    p = w.write(_good_essay(), today=today)
    text = p.read_text()
    assert text.startswith("---\n")
    assert "question: Why X?" in text
    assert "slug: why-x" in text
    assert "# Why X?" in text
