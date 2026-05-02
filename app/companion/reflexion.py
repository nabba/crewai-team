"""Reflexion — fold past feedback into the next cycle's prompt.

Shinn et al. 2023: verbal reinforcement learning where past failures and
successes are summarised in the next prompt as guidance. Phase 5 ships
the simplest form: list of recent thumbs-down idea bodies as "directions
that didn't resonate" and recent thumbs-up bodies as "directions worth
building on".

Token-bounded: at most ``max_negative`` + ``max_positive`` ideas, each
truncated to ~200 chars. Empty string when no feedback exists yet.
Failures (missing idea record, etc.) are skipped silently — partial
context is better than no context.
"""

from __future__ import annotations

import logging

from app.companion import feedback as _feedback
from app.companion import idea_store as _idea_store

logger = logging.getLogger(__name__)

DEFAULT_MAX_NEGATIVE = 3
DEFAULT_MAX_POSITIVE = 2
SNIPPET_MAX_CHARS = 200


def build_block(workspace_id: str, *,
                max_negative: int = DEFAULT_MAX_NEGATIVE,
                max_positive: int = DEFAULT_MAX_POSITIVE) -> str:
    """Return a ready-to-inject Markdown block. Empty string when none apply."""
    try:
        summary = _feedback.summary(workspace_id)
    except Exception as exc:
        logger.debug("companion.reflexion: summary failed: %s", exc)
        return ""

    neg_texts = _texts_for_ids(workspace_id,
                                summary.get("recent_negative_idea_ids", []),
                                limit=max_negative)
    pos_texts = _texts_for_ids(workspace_id,
                                summary.get("recent_positive_idea_ids", []),
                                limit=max_positive)
    if not neg_texts and not pos_texts:
        return ""

    parts: list[str] = ["## Lessons from past feedback"]
    if neg_texts:
        parts.append("These directions did NOT resonate — avoid similar:")
        for t in neg_texts:
            parts.append(f"- {t}")
    if pos_texts:
        if neg_texts:
            parts.append("")
        parts.append("These directions DID resonate — build on these:")
        for t in pos_texts:
            parts.append(f"- {t}")
    return "\n".join(parts) + "\n"


def _texts_for_ids(workspace_id: str, idea_ids: list[str],
                    *, limit: int) -> list[str]:
    """Resolve up to ``limit`` idea_ids to truncated body text."""
    out: list[str] = []
    for iid in idea_ids[:limit]:
        try:
            rec = _idea_store.find_by_id(workspace_id, iid)
        except Exception as exc:
            logger.debug("companion.reflexion: find_by_id failed for %s: %s",
                         iid, exc)
            continue
        if rec is None or not rec.text:
            continue
        out.append(_truncate(rec.text))
    return out


def _truncate(text: str, max_chars: int = SNIPPET_MAX_CHARS) -> str:
    body = (text or "").strip().replace("\n", " ")
    if len(body) <= max_chars:
        return body
    return body[: max_chars - 3] + "..."
