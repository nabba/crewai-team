"""Tests for app.subia.inquiry.selector."""

from __future__ import annotations

import os
import time
from pathlib import Path

from app.subia.inquiry.questions import Question
from app.subia.inquiry.selector import select_next_question


def test_returns_none_for_empty_list(tmp_path: Path) -> None:
    assert select_next_question([], inquiries_dir=tmp_path) is None


def test_picks_first_when_no_answers(tmp_path: Path) -> None:
    qs = [
        Question(slug="a", text="A"),
        Question(slug="b", text="B"),
    ]
    assert select_next_question(qs, inquiries_dir=tmp_path) is qs[0]


def test_skips_already_answered(tmp_path: Path) -> None:
    qs = [
        Question(slug="a", text="A"),
        Question(slug="b", text="B"),
        Question(slug="c", text="C"),
    ]
    (tmp_path / "2026-05-10-a.md").write_text("essay")
    next_q = select_next_question(qs, inquiries_dir=tmp_path)
    assert next_q is qs[1]


def test_picks_oldest_when_all_answered(tmp_path: Path) -> None:
    qs = [
        Question(slug="a", text="A"),
        Question(slug="b", text="B"),
    ]
    a_file = tmp_path / "2026-05-10-a.md"
    b_file = tmp_path / "2026-05-10-b.md"
    a_file.write_text("essay")
    b_file.write_text("essay")
    # Make A older than B by adjusting mtime explicitly.
    old_time = time.time() - 86400
    os.utime(a_file, (old_time, old_time))

    next_q = select_next_question(qs, inquiries_dir=tmp_path)
    assert next_q is qs[0]


def test_handles_missing_inquiries_dir(tmp_path: Path) -> None:
    qs = [Question(slug="a", text="A")]
    # Pass a path that doesn't exist; selector should treat it as empty.
    assert select_next_question(qs, inquiries_dir=tmp_path / "missing") is qs[0]


def test_question_filename_must_end_in_slug(tmp_path: Path) -> None:
    """Selector matches '<date>-<slug>.md' suffix; unrelated files don't count."""
    qs = [Question(slug="a", text="A")]
    (tmp_path / "2026-05-10-a-something-else.md").write_text("not the answer")
    next_q = select_next_question(qs, inquiries_dir=tmp_path)
    # The file ends in '-a-something-else', not '-a', so it doesn't match.
    assert next_q is qs[0]
