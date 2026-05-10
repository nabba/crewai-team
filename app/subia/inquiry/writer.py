"""Path-confined writer for inquiry essays.

Writes go to ``wiki/self/inquiries/<date>-<slug>.md`` ONLY. Any
attempt to write outside that directory raises :class:`WriteRefused`.
Defence-in-depth against a (hypothetical) compromised composer
that returns a path string with traversal.

The writer never overwrites an existing file. If a file with the
target name already exists (same date + same slug — unlikely unless
two passes run within the same day), the existing file is kept and
the new write is refused with a clear reason.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from app.subia.inquiry.composer import InquiryEssay

logger = logging.getLogger(__name__)


class WriteRefused(RuntimeError):
    """Raised when a write would land outside the inquiry directory."""


_DEFAULT_INQUIRIES_DIR = Path("/app/wiki/self/inquiries")


class InquiryWriter:
    """Path-confined writer; instances target a single inquiry directory."""

    def __init__(self, inquiries_dir: Path | str | None = None) -> None:
        self.inquiries_dir = Path(inquiries_dir) if inquiries_dir else _DEFAULT_INQUIRIES_DIR
        self.inquiries_dir.mkdir(parents=True, exist_ok=True)

    def write(self, essay: InquiryEssay, *, today: datetime | None = None) -> Path:
        """Write a non-failed inquiry essay; return the path written.

        Raises :class:`WriteRefused` if the essay is failed, if the
        target file already exists, or if the resolved target path
        would land outside :attr:`inquiries_dir`.
        """
        if essay.failed:
            raise WriteRefused(
                f"refusing to write failed inquiry for {essay.question_slug!r}: "
                f"{essay.failure_reason}"
            )
        date_part = (today or datetime.now(timezone.utc)).strftime("%Y-%m-%d")
        filename = f"{date_part}-{essay.question_slug}.md"
        target = (self.inquiries_dir / filename).resolve()
        # Defence-in-depth: ensure target is inside the canonical dir.
        canonical = self.inquiries_dir.resolve()
        if canonical not in target.parents and target != canonical:
            raise WriteRefused(
                f"refusing write to {target!r}: outside inquiries dir {canonical!r}"
            )
        if target.exists():
            raise WriteRefused(
                f"refusing write to {target!r}: file already exists "
                f"(another pass already produced this question's essay today)"
            )
        target.write_text(_render_essay(essay), encoding="utf-8")
        logger.info("inquiry: wrote %s (%d chars)", target, len(essay.body))
        return target


def _render_essay(essay: InquiryEssay) -> str:
    """Final markdown form combining frontmatter + body."""
    return (
        f"---\n"
        f"question: {essay.question_text}\n"
        f"slug: {essay.question_slug}\n"
        f"composed_at: {essay.composed_at}\n"
        f"attempts: {essay.attempts}\n"
        f"linter_warnings: {len(essay.linter_result.warnings)}\n"
        f"---\n\n"
        f"# {essay.question_text}\n\n"
        f"{essay.body.lstrip()}\n"
    )
