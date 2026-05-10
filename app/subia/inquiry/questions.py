"""Operator-curated philosophical question list.

The list lives at ``wiki/self/inquiry_questions.md`` as a markdown
file with one question per H2 header. Each H2 line provides a slug
(extracted from the heading) and an optional one-paragraph framing
beneath. The system parses this file to enumerate questions; it
NEVER edits it. New questions are added by the operator (or by
agents through ``change_requests``, which routes through the same
human-gated review).

Format::

    # Inquiry questions
    *Operator-curated list. See app/subia/inquiry/.*

    ## What is the relationship between my goals and Andrus's?

    *Optional framing paragraph(s).*

    ## How would my self-model differ if my substrate changed?
    ...

The slug for each question is generated deterministically from the
heading text: lowercase, non-alphanumerics → hyphens, deduped.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


_DEFAULT_QUESTIONS_PATH = Path("/app/wiki/self/inquiry_questions.md")
_SLUG_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_SLUG_LEADING_TRAILING = re.compile(r"^-+|-+$")


@dataclass(frozen=True)
class Question:
    """One curated philosophical question."""

    slug: str
    text: str
    framing: str = ""

    def title(self) -> str:
        """Human-readable form for headings + filenames."""
        return self.text


def slugify(heading: str) -> str:
    """Generate the deterministic slug for a question heading."""
    s = heading.lower()
    s = _SLUG_NON_ALNUM.sub("-", s)
    s = _SLUG_LEADING_TRAILING.sub("", s)
    return s[:80] if s else "untitled"


def parse_questions(text: str) -> list[Question]:
    """Parse markdown text into a list of Question objects.

    Each ``## Heading`` starts a new question; the body until the
    next H2 (or EOF) is the optional framing paragraph(s). The H1
    header (``# Inquiry questions``) is ignored.
    """
    lines = text.splitlines()
    questions: list[Question] = []
    seen_slugs: set[str] = set()

    cur_heading: str | None = None
    cur_body: list[str] = []

    def flush() -> None:
        if cur_heading is None:
            return
        slug = slugify(cur_heading)
        # Disambiguate duplicate slugs.
        base = slug
        i = 2
        while slug in seen_slugs:
            slug = f"{base}-{i}"
            i += 1
        seen_slugs.add(slug)
        framing = "\n".join(cur_body).strip()
        questions.append(Question(slug=slug, text=cur_heading.strip(), framing=framing))

    for line in lines:
        if line.startswith("## "):
            flush()
            cur_heading = line[3:].rstrip()
            cur_body = []
        elif cur_heading is not None:
            cur_body.append(line)
    flush()
    return questions


def load_questions(path: Path | str | None = None) -> list[Question]:
    """Read + parse the operator-curated question file.

    Returns the empty list if the file is missing — the caller decides
    whether that's an error or a no-op.
    """
    src = Path(path) if path else _DEFAULT_QUESTIONS_PATH
    if not src.exists():
        logger.info("inquiry: questions file %s missing — returning empty list", src)
        return []
    try:
        return parse_questions(src.read_text(encoding="utf-8"))
    except OSError as exc:  # noqa: BLE001
        logger.warning("inquiry: cannot read questions file %s: %s", src, exc)
        return []
