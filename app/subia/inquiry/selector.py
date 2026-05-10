"""Pick the next question for an inquiry pass.

Strategy: oldest unanswered first. We consider a question "answered"
when ``wiki/self/inquiries/`` contains a file whose stem ends in
``-<slug>``. If every question has been answered at least once, the
selector picks the question whose most recent answer is *oldest* —
essentially a long-cycle round-robin.

Tie-breaking is by the order questions appear in the curated list,
giving the operator predictable control: questions higher up rotate
sooner.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from app.subia.inquiry.questions import Question

logger = logging.getLogger(__name__)


_DEFAULT_INQUIRIES_DIR = Path("/app/wiki/self/inquiries")


@dataclass(frozen=True)
class _AnswerSummary:
    slug: str
    most_recent: datetime | None  # None means never answered


def _scan_answers(inquiries_dir: Path, questions: list[Question]) -> dict[str, _AnswerSummary]:
    """For each question slug, find the most recent answer file's mtime."""
    summary: dict[str, _AnswerSummary] = {
        q.slug: _AnswerSummary(slug=q.slug, most_recent=None) for q in questions
    }
    if not inquiries_dir.exists():
        return summary
    for f in inquiries_dir.glob("*.md"):
        for q in questions:
            # File naming convention: <date>-<slug>.md
            if f.stem.endswith(f"-{q.slug}"):
                try:
                    mtime = datetime.fromtimestamp(f.stat().st_mtime)
                except OSError:
                    continue
                prev = summary[q.slug].most_recent
                if prev is None or mtime > prev:
                    summary[q.slug] = _AnswerSummary(slug=q.slug, most_recent=mtime)
                break
    return summary


def select_next_question(
    questions: list[Question],
    inquiries_dir: Path | str | None = None,
) -> Question | None:
    """Return the next question to ask, or None if the list is empty.

    Selection: any never-answered question first (in list order); if
    all have been answered, the one whose most-recent answer is
    oldest (again, ties broken by list order).
    """
    if not questions:
        return None
    src = Path(inquiries_dir) if inquiries_dir else _DEFAULT_INQUIRIES_DIR
    summary = _scan_answers(src, questions)

    # First pass: any never-answered question.
    for q in questions:
        if summary[q.slug].most_recent is None:
            return q

    # Second pass: oldest answered.
    answered_in_order = [(q, summary[q.slug].most_recent) for q in questions]
    # All have a most_recent at this point; mypy-safe via filter
    answered_in_order = [(q, t) for q, t in answered_in_order if t is not None]
    if not answered_in_order:
        return questions[0]
    # Stable sort by mtime; lowest first (oldest).
    answered_in_order.sort(key=lambda pair: pair[1])
    return answered_in_order[0][0]
