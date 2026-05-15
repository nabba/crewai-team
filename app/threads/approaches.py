"""Q8.2 — thread-closure "approaches tried" distillation + bridge
into lessons_learned.

PROGRAM §46.2. When a thread closes (resolved or abandoned), this
module walks the thread's notes + blockers + sub-questions and
distills a brief "approaches tried" summary. The summary lands as
a single lesson row in ``app.companion.lessons_learned``'s KB
under the new ``source="thread_closure"`` so future thread creation
can consult it via ``check_against`` for adjacent topics.

Two synthesis paths:

  1. **Cheap-tier LLM call** (one ~$0.0001 call per closure) —
     summarises the approach trajectory in 2-3 sentences. Uses the
     same dispatcher pattern as ``app.companion.grand_task``.
  2. **Deterministic fallback** when LLM is unavailable — just
     concatenates the resolved sub-questions + notes + blockers
     into a structured plain-text body.

The closure synthesis is failure-isolated: if it throws, the thread
is still marked closed (the caller already saved the lifecycle
transition before invoking this).

The pre-creation consultation hook (``consult_before_create``) calls
``lessons_learned.check_against`` for the proposed title and returns
matching past closures. Wired into ``create_thread`` via a Signal
notification — never blocks creation, just surfaces signal.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from app.threads.models import Thread, ThreadStatus

logger = logging.getLogger(__name__)


_MIN_BODY_LEN_FOR_LLM = 60  # below this, deterministic is fine


def distill_on_closure(thread: Thread) -> str:
    """Produce the "approaches tried" summary for a closing thread.

    Stores the result on ``thread.approaches_summary`` AND emits a
    ``thread_closure`` source event into the lessons_learned KB.
    Returns the summary text (also accessible via the field).

    Failure-isolated — exceptions return an empty string and log;
    the thread's lifecycle state is NOT rolled back.
    """
    try:
        body_text = _build_deterministic_body(thread)
        # Decide synthesis path
        if _llm_enabled() and len(body_text) >= _MIN_BODY_LEN_FOR_LLM:
            summary = _llm_distill(thread, body_text) or body_text
        else:
            summary = body_text
        thread.approaches_summary = summary[:2000]
        try:
            from app.threads import store
            store.save(thread)
        except Exception:
            logger.debug("approaches: store.save failed", exc_info=True)

        _emit_lesson(thread, summary)
        return thread.approaches_summary
    except Exception:
        logger.debug("approaches: distill_on_closure raised", exc_info=True)
        return ""


def consult_before_create(title: str, description: str = "") -> list[dict[str, Any]]:
    """Return matching past thread-closure lessons for a proposed
    new thread.

    Empty list when:
      * lessons_learned KB is empty
      * no past closure clusters cross the match threshold
      * the lessons_learned module is unavailable

    Failure-isolated. Callers MUST treat the return as advisory —
    new threads are never blocked.
    """
    if not title.strip():
        return []
    try:
        from app.companion.lessons_learned import check_against
    except Exception:
        return []
    try:
        return check_against(f"{title} — {description}", top_k=3)
    except Exception:
        logger.debug("approaches: check_against raised", exc_info=True)
        return []


# ─────────────────────────────────────────────────────────────────────
#   Internals
# ─────────────────────────────────────────────────────────────────────


def _llm_enabled() -> bool:
    return os.getenv("APPROACHES_LLM_ENABLED", "true").lower() in (
        "true", "1", "yes", "on",
    )


def _build_deterministic_body(thread: Thread) -> str:
    """Build the structured plain-text "approaches tried" body.

    Used as both the LLM input and the fallback summary. Format::

        Question: <title>
        Closed as: <status>
        Sub-questions resolved: N/M
          - <text> → <resolution>
          ...
        Blockers encountered:
          - <text>
        Hints generated:
          - <text>
        Resolution note: <last note>
    """
    lines: list[str] = []
    lines.append(f"Question: {thread.title}")
    closed_as = thread.status.value if isinstance(thread.status, ThreadStatus) else str(thread.status)
    lines.append(f"Closed as: {closed_as}")
    if thread.abandon_reason:
        lines.append(f"Abandon reason: {thread.abandon_reason}")

    resolved = thread.resolved_subquestions
    if thread.sub_questions:
        lines.append(
            f"Sub-questions resolved: {len(resolved)}/{len(thread.sub_questions)}"
        )
        for sq in resolved[:5]:
            res = (sq.resolution or "").strip()
            if res:
                lines.append(f"  - {sq.text} → {res}")
            else:
                lines.append(f"  - {sq.text}")
    if thread.blockers:
        lines.append("Blockers encountered:")
        for b in thread.blockers[:5]:
            lines.append(f"  - {b}")
    if thread.unblock_hints:
        lines.append("Hints generated:")
        for h in thread.unblock_hints[:5]:
            lines.append(f"  - {h}")
    if thread.notes:
        # Take the last note (typically the resolution note) verbatim.
        lines.append(f"Last note: {thread.notes[-1]}")
    return "\n".join(lines)


def _llm_distill(thread: Thread, body_text: str) -> str:
    """Cheap-tier LLM call: summarise the approach trajectory.

    Uses Anthropic Haiku 4.5 with a small budget. Returns empty
    string on any failure (caller falls back to body_text).
    """
    try:
        import anthropic
    except ImportError:
        return ""
    try:
        client = anthropic.Anthropic()
        prompt = (
            "Below is a record of a long-horizon question that just closed. "
            "Distill a 2-3 sentence summary of the APPROACHES that were "
            "tried, in order, and which one ultimately resolved it (or "
            "what made it unresolvable). Be terse. No preamble. No "
            "first-person voice. Future operators reading this will use "
            "it to avoid repeating dead ends.\n\n"
            f"{body_text}\n\n"
            "Approaches-tried summary:"
        )
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=240,
            messages=[{"role": "user", "content": prompt}],
        )
        # Extract text blocks
        text_parts = [
            getattr(b, "text", "")
            for b in (msg.content or [])
            if getattr(b, "type", "") == "text"
        ]
        out = "".join(text_parts).strip()
        # Guard against the model returning JSON or other garbage.
        if out and len(out) <= 1500:
            return out
        return ""
    except Exception:
        logger.debug("approaches: LLM distill failed", exc_info=True)
        return ""


def _emit_lesson(thread: Thread, summary: str) -> None:
    """Append a ``thread_closure`` event to the lessons_learned KB
    via a one-shot synthetic call.

    Re-uses the public clustering helper rather than reaching into
    private internals — keeps lessons_learned the single source of
    truth for KB shape.
    """
    if not summary or len(summary) < 10:
        return
    try:
        from app.companion import lessons_learned as ll
    except Exception:
        return
    try:
        existing = ll._read_kb()
        # The "proposal_text" surface for clustering is the title +
        # the approaches summary; the "decision_reason" is how the
        # thread closed (resolved / abandoned).
        closed_as = thread.status.value if isinstance(thread.status, ThreadStatus) else str(thread.status)
        event = {
            "source": "thread_closure",
            "ts": thread.resolved_at or thread.abandoned_at or thread.last_touched_at or "",
            "proposal_text": f"{thread.title}\n{summary}"[:1000],
            "decision_reason": (
                f"thread closed as {closed_as}"
                + (f": {thread.abandon_reason}" if thread.abandon_reason else "")
            )[:500],
        }
        updated = ll._cluster_into_kb([event], existing)
        # Keep singleton clusters too — for thread closures, every
        # thread is itself a one-off lesson worth preserving (unlike
        # mass-CR-rejection where singletons are noise).
        ll._write_kb(updated)
    except Exception:
        logger.debug("approaches: lesson emit failed", exc_info=True)
