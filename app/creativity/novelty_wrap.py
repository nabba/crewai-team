"""Brainstorm-idea novelty assessment combining KB novelty + rejection history.

The base :func:`app.self_improvement.novelty.novelty_report` answers
"is this text novel against the system's KBs?" across the four
research/experiential/aesthetic/tension stores plus skills. That
catches *coverage*: the idea is already in the literature, already
journaled, already a registered skill.

It does NOT catch *rejection history*: "this exact idea was proposed
before and the operator rejected it." That signal lives in
:mod:`app.companion.lessons_learned`, which clusters rejected change-
requests, 👎'd Signal feedback, and Goodhart-flagged proposals.

This wrapper combines the two checks and projects them to a four-way
verdict that brainstorm + reverie surfaces consume directly:

  NOVEL              — neither KBs nor rejection history match
  RECOMBINATION      — KB overlap (similar to known material)
  RESTATED           — KB covers it (essentially identical)
  REJECTED_BEFORE    — lessons-learned has a near-match — this is
                       a previously-rejected proposal coming around
                       again. Treat as a HARD signal not to surface.

The wrapper is *advisory*. Brainstorm consumers decide what to do
with each verdict — typically: NOVEL → keep, RECOMBINATION → keep
with credit, RESTATED → drop, REJECTED_BEFORE → drop + log.

Both underlying calls are injectable so this module is testable
without ChromaDB / live KBs.
"""
from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


class NoveltyVerdict(str, enum.Enum):
    NOVEL = "novel"
    RECOMBINATION = "recombination"
    RESTATED = "restated"
    REJECTED_BEFORE = "rejected_before"


@dataclass(frozen=True)
class NoveltyWrap:
    """Combined assessment from novelty_report + lessons_learned."""

    verdict: NoveltyVerdict
    primary_decision: str | None  # "covered" / "overlap" / "adjacent" / "novel" or None on failure
    primary_distance: float | None  # cosine distance to nearest KB neighbour
    primary_collection: str | None  # which KB the nearest neighbour was in
    rejected_lesson_id: str | None  # id of matching rejected lesson, if any
    rejected_score: float | None    # similarity score if a lesson matched
    notes: list[str] = field(default_factory=list)


# Type aliases for the injectable dependencies.
NoveltyReportFn = Callable[[str], Any]
"""Returns a ``NoveltyReport`` (from :mod:`app.self_improvement.types`)."""

LessonsCheckFn = Callable[[str, int], list[dict]]
"""Returns ``list[dict]`` from :func:`app.companion.lessons_learned.check_against`.

Each dict has at least ``id`` (str) and ``score`` (float in 0..1).
"""


def assess_brainstorm_idea(
    text: str,
    *,
    novelty_report_fn: NoveltyReportFn | None = None,
    lessons_check_fn: LessonsCheckFn | None = None,
    rejected_match_threshold: float = 0.55,
) -> NoveltyWrap:
    """Run both checks, project to a four-way verdict.

    Failure-isolated: if either underlying call raises, the wrapper
    returns NOVEL by default — i.e. the idea is *not* discarded on
    a tooling outage. The caller can inspect ``notes`` to see which
    check failed.

    ``rejected_match_threshold`` is the lessons-learned similarity
    score above which we report ``REJECTED_BEFORE``. The default 0.55
    is intentionally lower than ``check_against``'s own threshold
    (0.40) so that we only flag *strong* rejection matches — a weak
    semantic overlap is downgraded to ``RECOMBINATION``.
    """
    text = (text or "").strip()
    if not text:
        return NoveltyWrap(
            verdict=NoveltyVerdict.NOVEL,
            primary_decision=None,
            primary_distance=None,
            primary_collection=None,
            rejected_lesson_id=None,
            rejected_score=None,
            notes=["empty input → defaulting to NOVEL"],
        )

    primary_decision: str | None = None
    primary_distance: float | None = None
    primary_collection: str | None = None
    rejected_id: str | None = None
    rejected_score: float | None = None
    notes: list[str] = []

    # 1. KB novelty check — primary signal
    nr_fn = novelty_report_fn or _default_novelty_report
    try:
        report = nr_fn(text)
        # NoveltyReport has fields: decision, nearest_distance,
        # nearest_collection, nearest_text. Defensive access in case
        # the production type evolves.
        primary_decision = _extract(report, "decision")
        if hasattr(primary_decision, "value"):
            primary_decision = primary_decision.value
        primary_distance = _extract(report, "nearest_distance")
        primary_collection = _extract(report, "nearest_collection")
    except Exception as exc:  # noqa: BLE001
        notes.append(f"novelty_report failed: {exc}")
        logger.debug("novelty_wrap: novelty_report failed", exc_info=True)

    # 2. Rejection-history check — overlay signal
    lc_fn = lessons_check_fn or _default_lessons_check
    try:
        matches = lc_fn(text, 1) or []
        if matches:
            top = matches[0]
            score = float(top.get("score", 0.0))
            if score >= rejected_match_threshold:
                rejected_id = str(top.get("id") or "")
                rejected_score = score
    except Exception as exc:  # noqa: BLE001
        notes.append(f"lessons_check failed: {exc}")
        logger.debug("novelty_wrap: lessons_check failed", exc_info=True)

    # 3. Project to verdict.
    if rejected_id is not None:
        verdict = NoveltyVerdict.REJECTED_BEFORE
    elif primary_decision == "covered":
        verdict = NoveltyVerdict.RESTATED
    elif primary_decision == "overlap":
        verdict = NoveltyVerdict.RECOMBINATION
    else:
        verdict = NoveltyVerdict.NOVEL

    return NoveltyWrap(
        verdict=verdict,
        primary_decision=primary_decision if isinstance(primary_decision, str) else None,
        primary_distance=primary_distance if isinstance(primary_distance, (int, float)) else None,
        primary_collection=primary_collection if isinstance(primary_collection, str) else None,
        rejected_lesson_id=rejected_id,
        rejected_score=rejected_score,
        notes=notes,
    )


# ── Internals ──────────────────────────────────────────────────────────


def _extract(obj: Any, name: str) -> Any:
    """Best-effort attribute access — works for dataclass, dict, or namedtuple."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _default_novelty_report(text: str) -> Any:
    from app.self_improvement.novelty import novelty_report
    return novelty_report(text)


def _default_lessons_check(text: str, top_k: int) -> list[dict]:
    from app.companion.lessons_learned import check_against
    return check_against(text, top_k=top_k)
