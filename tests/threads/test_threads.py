"""Tests for app.threads.* — long-horizon thread primitive."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.threads import (
    InvalidThreadTransition,
    Thread,
    ThreadStatus,
    abandon_thread,
    add_blocker,
    add_subquestion,
    clear_blockers,
    create_thread,
    get,
    link_crew_task,
    link_inquiry,
    list_all,
    list_open,
    mark_blocked,
    mark_in_progress,
    record_note,
    reset_for_tests,
    resolve_subquestion,
    resolve_thread,
)
from app.threads.models import SubQuestion


@pytest.fixture(autouse=True)
def isolate(tmp_path: Path):
    reset_for_tests(tmp_path)
    yield
    reset_for_tests(None)


# ── models ───────────────────────────────────────────────────────────────


def test_thread_round_trip_serialization() -> None:
    t = create_thread(title="Investigate forest data gap")
    add_subquestion(t.id, "Which provinces have aerial coverage?")
    add_subquestion(t.id, "What's the missing-data threshold?")
    add_blocker(t.id, "ESA dataset access not yet granted")
    record_note(t.id, "operator pinged ESA contact 2026-05-08")
    link_crew_task(t.id, "task-abc-123")
    link_inquiry(t.id, "are-the-goal-sources-coherent")

    loaded = get(t.id)
    assert loaded is not None
    raw = loaded.to_dict()
    rt = Thread.from_dict(raw)
    assert rt.id == t.id
    assert len(rt.sub_questions) == 2
    assert rt.blockers == ["ESA dataset access not yet granted"]
    assert "task-abc-123" in rt.related_crew_task_ids
    assert "are-the-goal-sources-coherent" in rt.related_inquiry_slugs


def test_subquestion_open_resolved_partition() -> None:
    t = create_thread(title="x")
    add_subquestion(t.id, "Q1")
    t = add_subquestion(t.id, "Q2")
    sq1 = t.sub_questions[0]
    t = resolve_subquestion(t.id, sq1.id, resolution="answered")
    assert len(t.open_subquestions) == 1
    assert len(t.resolved_subquestions) == 1
    assert t.resolved_subquestions[0].resolution == "answered"


def test_is_terminal_property() -> None:
    t = create_thread(title="x")
    assert not t.is_terminal
    t = resolve_thread(t.id, summary="done")
    assert t.is_terminal


# ── lifecycle happy paths ───────────────────────────────────────────────


def test_create_then_resolve_full_flow() -> None:
    t = create_thread(title="Refactor X subsystem", description="why")
    assert t.status is ThreadStatus.OPEN

    t = add_subquestion(t.id, "What does X currently do?")
    assert t.status is ThreadStatus.IN_PROGRESS  # auto-advance from OPEN

    t = add_subquestion(t.id, "What invariants must survive?")
    sq_id = t.sub_questions[0].id

    t = resolve_subquestion(t.id, sq_id, resolution="X is the foo bar")
    assert len(t.open_subquestions) == 1

    t = resolve_thread(t.id, summary="refactor landed")
    assert t.status is ThreadStatus.RESOLVED
    assert any("[resolution]" in n for n in t.notes)


def test_blocked_then_unblocked() -> None:
    t = create_thread(title="x")
    t = add_blocker(t.id, "waiting on credentials")
    assert t.blockers == ["waiting on credentials"]
    assert t.status is ThreadStatus.IN_PROGRESS  # blocker doesn't auto-mark BLOCKED

    t = mark_blocked(t.id)
    assert t.status is ThreadStatus.BLOCKED

    t = clear_blockers(t.id)
    assert t.blockers == []
    assert t.status is ThreadStatus.IN_PROGRESS


def test_mark_blocked_with_inline_blocker() -> None:
    t = create_thread(title="x")
    t = mark_blocked(t.id, blocker="api access pending")
    assert t.status is ThreadStatus.BLOCKED
    assert "api access pending" in t.blockers


def test_abandon_with_reason() -> None:
    t = create_thread(title="x")
    t = abandon_thread(t.id, reason="duplicates a parallel investigation")
    assert t.status is ThreadStatus.ABANDONED
    assert t.abandon_reason == "duplicates a parallel investigation"
    assert t.is_terminal


# ── illegal transitions ─────────────────────────────────────────────────


def test_cannot_add_subquestion_after_resolve() -> None:
    t = create_thread(title="x")
    resolve_thread(t.id)
    with pytest.raises(InvalidThreadTransition):
        add_subquestion(t.id, "late question")


def test_cannot_add_blocker_after_abandon() -> None:
    t = create_thread(title="x")
    abandon_thread(t.id, reason="r")
    with pytest.raises(InvalidThreadTransition):
        add_blocker(t.id, "late blocker")


def test_cannot_resolve_after_abandon() -> None:
    t = create_thread(title="x")
    abandon_thread(t.id, reason="r")
    with pytest.raises(InvalidThreadTransition):
        resolve_thread(t.id)


def test_unknown_thread_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        add_subquestion("nonexistent", "x")


def test_resolve_subquestion_unknown_raises() -> None:
    t = create_thread(title="x")
    with pytest.raises(KeyError):
        resolve_subquestion(t.id, "no-such-subq", resolution="x")


def test_create_with_empty_title_raises() -> None:
    with pytest.raises(ValueError):
        create_thread(title="   ")


def test_add_subquestion_empty_text_raises() -> None:
    t = create_thread(title="x")
    with pytest.raises(ValueError):
        add_subquestion(t.id, "")


# ── list operations ────────────────────────────────────────────────────


def test_list_open_excludes_terminal() -> None:
    a = create_thread(title="A")
    b = create_thread(title="B")
    c = create_thread(title="C")
    resolve_thread(b.id)
    abandon_thread(c.id, reason="x")

    open_ids = {t.id for t in list_open()}
    assert open_ids == {a.id}

    all_ids = {t.id for t in list_all()}
    assert all_ids == {a.id, b.id, c.id}


def test_list_orders_newest_activity_first() -> None:
    a = create_thread(title="A")
    b = create_thread(title="B")
    # b was just touched (post-creation). Make a fresher activity on a.
    add_subquestion(a.id, "x")
    out = list_open()
    assert out[0].id == a.id
    assert out[1].id == b.id


# ── linking ─────────────────────────────────────────────────────────────


def test_link_crew_task_dedups() -> None:
    t = create_thread(title="x")
    link_crew_task(t.id, "task-1")
    link_crew_task(t.id, "task-1")  # dup
    link_crew_task(t.id, "task-2")
    assert get(t.id).related_crew_task_ids == ["task-1", "task-2"]


def test_link_inquiry_dedups() -> None:
    t = create_thread(title="x")
    link_inquiry(t.id, "slug-a")
    link_inquiry(t.id, "slug-a")
    link_inquiry(t.id, "slug-b")
    assert get(t.id).related_inquiry_slugs == ["slug-a", "slug-b"]


# ── persistence ────────────────────────────────────────────────────────


def test_persistence_survives_index_reset(tmp_path: Path) -> None:
    """After reset_for_tests with the same dir, the in-memory index is
    rebuilt from disk and the thread is recovered intact."""
    t = create_thread(title="persistence test")
    add_subquestion(t.id, "Q1")
    add_blocker(t.id, "b")

    # Simulate a process restart: re-init store with same dir.
    reset_for_tests(tmp_path)
    reloaded = get(t.id)
    assert reloaded is not None
    assert reloaded.title == "persistence test"
    assert len(reloaded.sub_questions) == 1
    assert reloaded.blockers == ["b"]


# ── notes ──────────────────────────────────────────────────────────────


def test_record_note_appends() -> None:
    t = create_thread(title="x")
    record_note(t.id, "first observation")
    t = record_note(t.id, "second observation")
    assert t.notes == ["first observation", "second observation"]


def test_record_note_advances_open_to_in_progress() -> None:
    t = create_thread(title="x")
    assert t.status is ThreadStatus.OPEN
    t = record_note(t.id, "started thinking about it")
    assert t.status is ThreadStatus.IN_PROGRESS
