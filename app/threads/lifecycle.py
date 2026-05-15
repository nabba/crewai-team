"""Thread lifecycle — function-per-transition entry points.

Mirrors the change_requests + architecture_requests pattern. Every
transition saves the thread; illegal transitions raise
:class:`InvalidThreadTransition`.

State machine::

    OPEN ──→ IN_PROGRESS ──┬──→ RESOLVED
        │            │      │
        │            ├──→ BLOCKED ──→ IN_PROGRESS
        │            │
        ↓            ↓
     ABANDONED   ABANDONED

OPEN is the just-created state. Any non-trivial work (subquestion
added, blocker added, note recorded) implicitly advances OPEN →
IN_PROGRESS. The operator can also explicitly transition.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from app.threads import store
from app.threads.models import (
    InvalidThreadTransition,
    SubQuestion,
    Thread,
    ThreadStatus,
)

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _touch(thread: Thread) -> None:
    thread.last_touched_at = _now_iso()


def _get_or_raise(thread_id: str) -> Thread:
    t = store.get(thread_id)
    if t is None:
        raise KeyError(f"unknown thread {thread_id!r}")
    return t


def _advance_to_in_progress_if_open(thread: Thread) -> None:
    if thread.status is ThreadStatus.OPEN:
        thread.status = ThreadStatus.IN_PROGRESS


def _require_status(
    thread: Thread,
    expected: set[ThreadStatus],
    transition: str,
) -> None:
    if thread.status not in expected:
        raise InvalidThreadTransition(
            f"cannot {transition}: thread {thread.id} is in status "
            f"{thread.status.value}; expected one of "
            f"{sorted(s.value for s in expected)}"
        )


# ── Public API ──────────────────────────────────────────────────────


def create_thread(*, title: str, description: str = "") -> Thread:
    if not title.strip():
        raise ValueError("title must be non-empty")
    now = _now_iso()
    thread = Thread(
        id=str(uuid.uuid4()),
        created_at=now,
        title=title.strip(),
        description=description.strip(),
        last_touched_at=now,
    )
    store.save(thread)
    # Q8.2 (PROGRAM §46.2) — consult lessons_learned KB for adjacent
    # past closures. Failure-isolated: a broken consultation never
    # blocks thread creation.
    try:
        from app.threads.approaches import consult_before_create
        matches = consult_before_create(thread.title, thread.description)
    except Exception:
        matches = []
    if matches:
        try:
            _notify_relevant_closures(thread, matches)
        except Exception:
            logger.debug(
                "thread create: notify failed", exc_info=True,
            )
    return thread


def _notify_relevant_closures(thread: Thread, matches: list[dict]) -> None:
    """Surface the top thread-closure lesson via Signal arbiter.

    Quiet by default — only fires when ≥1 match has similarity ≥ 0.6
    (a tighter threshold than the KB's general MATCH_THRESHOLD,
    because we want to avoid noise during thread creation).
    """
    high_sim = [m for m in matches if float(m.get("similarity", 0)) >= 0.6]
    if not high_sim:
        return
    try:
        from app.notify import notify
    except Exception:
        return
    top = high_sim[0]
    body_lines = [
        f"📚 Found {len(high_sim)} similar past thread(s) for "
        f"'{thread.title[:80]}':",
    ]
    for m in high_sim[:3]:
        sim = float(m.get("similarity", 0))
        reason = (m.get("sample_reason") or "").strip()
        if reason:
            body_lines.append(f"  • ({sim:.2f}) {reason[:140]}")
    body_lines.append(
        f"Review at /cp/threads/{thread.id[:8]}. Q8.2 (PROGRAM §46.2).",
    )
    try:
        notify(
            title="Similar past thread(s) found",
            body="\n".join(body_lines),
            url=f"/cp/threads",
            topic=f"thread_consult:{thread.id}",
            arbitrate=True,
        )
    except Exception:
        # Try the simpler shape if topic/arbitrate aren't supported.
        try:
            notify(
                title="Similar past thread(s) found",
                body="\n".join(body_lines),
                url="/cp/threads",
            )
        except Exception:
            logger.debug("thread notify failed", exc_info=True)


def add_subquestion(thread_id: str, text: str) -> Thread:
    if not text.strip():
        raise ValueError("subquestion text must be non-empty")
    t = _get_or_raise(thread_id)
    if t.is_terminal:
        raise InvalidThreadTransition(
            f"cannot add subquestion: thread is {t.status.value}",
        )
    sq = SubQuestion(id=str(uuid.uuid4()), text=text.strip())
    t.sub_questions.append(sq)
    _advance_to_in_progress_if_open(t)
    _touch(t)
    store.save(t)
    return t


def resolve_subquestion(
    thread_id: str, subquestion_id: str, resolution: str = "",
) -> Thread:
    t = _get_or_raise(thread_id)
    if t.is_terminal:
        raise InvalidThreadTransition(
            f"cannot resolve subquestion: thread is {t.status.value}",
        )
    for sq in t.sub_questions:
        if sq.id == subquestion_id:
            sq.resolved = True
            sq.resolution = resolution.strip()
            sq.resolved_at = _now_iso()
            _touch(t)
            store.save(t)
            return t
    raise KeyError(f"subquestion {subquestion_id!r} not found in {thread_id!r}")


def add_blocker(thread_id: str, text: str) -> Thread:
    if not text.strip():
        raise ValueError("blocker text must be non-empty")
    t = _get_or_raise(thread_id)
    if t.is_terminal:
        raise InvalidThreadTransition(
            f"cannot add blocker: thread is {t.status.value}",
        )
    t.blockers.append(text.strip())
    if t.status is not ThreadStatus.BLOCKED:
        _advance_to_in_progress_if_open(t)
    _touch(t)
    store.save(t)
    return t


def clear_blockers(thread_id: str) -> Thread:
    t = _get_or_raise(thread_id)
    t.blockers.clear()
    if t.status is ThreadStatus.BLOCKED:
        t.status = ThreadStatus.IN_PROGRESS
    _touch(t)
    store.save(t)
    return t


# ── Q8.1 (PROGRAM §46.1) — unblock_hints ──────────────────────────────


def add_unblock_hint(thread_id: str, text: str) -> Thread:
    """Append a "what might unblock this" hypothesis to the thread.

    Symmetric to :func:`add_blocker`: blockers say what IS blocking;
    unblock hints say what MIGHT resolve the block. The recovery loop
    consults open-thread unblock_hints via
    :mod:`app.recovery.thread_consultation` to surface relevant prior
    context during strategy selection.
    """
    if not text.strip():
        raise ValueError("unblock-hint text must be non-empty")
    t = _get_or_raise(thread_id)
    if t.is_terminal:
        raise InvalidThreadTransition(
            f"cannot add unblock-hint: thread is {t.status.value}",
        )
    t.unblock_hints.append(text.strip())
    _advance_to_in_progress_if_open(t)
    _touch(t)
    store.save(t)
    return t


def clear_unblock_hints(thread_id: str) -> Thread:
    """Wipe unblock-hint list (e.g. operator decided the hypotheses
    are stale). Does not change thread status — the blockers themselves
    still apply."""
    t = _get_or_raise(thread_id)
    t.unblock_hints.clear()
    _touch(t)
    store.save(t)
    return t


def mark_blocked(thread_id: str, blocker: str | None = None) -> Thread:
    t = _get_or_raise(thread_id)
    _require_status(
        t,
        {ThreadStatus.OPEN, ThreadStatus.IN_PROGRESS, ThreadStatus.BLOCKED},
        "mark_blocked",
    )
    if blocker and blocker.strip():
        t.blockers.append(blocker.strip())
    t.status = ThreadStatus.BLOCKED
    _touch(t)
    store.save(t)
    return t


def mark_in_progress(thread_id: str) -> Thread:
    t = _get_or_raise(thread_id)
    _require_status(
        t,
        {ThreadStatus.OPEN, ThreadStatus.BLOCKED, ThreadStatus.IN_PROGRESS},
        "mark_in_progress",
    )
    t.status = ThreadStatus.IN_PROGRESS
    _touch(t)
    store.save(t)
    return t


def resolve_thread(thread_id: str, *, summary: str = "") -> Thread:
    t = _get_or_raise(thread_id)
    _require_status(
        t,
        {ThreadStatus.OPEN, ThreadStatus.IN_PROGRESS, ThreadStatus.BLOCKED},
        "resolve_thread",
    )
    t.status = ThreadStatus.RESOLVED
    t.resolved_at = _now_iso()
    if summary.strip():
        t.notes.append(f"[resolution] {summary.strip()}")
    _touch(t)
    store.save(t)
    _distill_on_closure_safely(t)
    return t


def abandon_thread(thread_id: str, *, reason: str) -> Thread:
    if not reason.strip():
        raise ValueError("abandon reason must be non-empty")
    t = _get_or_raise(thread_id)
    _require_status(
        t,
        {ThreadStatus.OPEN, ThreadStatus.IN_PROGRESS, ThreadStatus.BLOCKED},
        "abandon_thread",
    )
    t.status = ThreadStatus.ABANDONED
    t.abandoned_at = _now_iso()
    t.abandon_reason = reason.strip()
    _touch(t)
    store.save(t)
    _distill_on_closure_safely(t)
    return t


def _distill_on_closure_safely(thread: Thread) -> None:
    """Q8.2 hook — call the approaches-tried distiller after the
    closure transition has been persisted. Failure-isolated: the
    thread is already saved as RESOLVED/ABANDONED before we get here,
    so a broken distiller never rolls back the lifecycle change."""
    try:
        from app.threads.approaches import distill_on_closure
        distill_on_closure(thread)
    except Exception:
        logger.debug("thread closure distill failed", exc_info=True)


def link_crew_task(thread_id: str, crew_task_id: str) -> Thread:
    t = _get_or_raise(thread_id)
    if crew_task_id not in t.related_crew_task_ids:
        t.related_crew_task_ids.append(crew_task_id)
    _advance_to_in_progress_if_open(t)
    _touch(t)
    store.save(t)
    return t


def link_inquiry(thread_id: str, inquiry_slug: str) -> Thread:
    t = _get_or_raise(thread_id)
    if inquiry_slug not in t.related_inquiry_slugs:
        t.related_inquiry_slugs.append(inquiry_slug)
    _touch(t)
    store.save(t)
    return t


def record_note(thread_id: str, text: str) -> Thread:
    if not text.strip():
        raise ValueError("note text must be non-empty")
    t = _get_or_raise(thread_id)
    if t.is_terminal:
        raise InvalidThreadTransition(
            f"cannot add note: thread is {t.status.value}",
        )
    t.notes.append(text.strip())
    _advance_to_in_progress_if_open(t)
    _touch(t)
    store.save(t)
    return t
