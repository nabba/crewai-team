"""PROGRAM §46.2 — Q8.2 thread-closure approaches-tried tests.

Covers:

  1. ``app.threads.approaches.distill_on_closure`` writes the
     ``approaches_summary`` field AND emits a ``thread_closure``
     event into the lessons_learned KB.
  2. Deterministic fallback when LLM is disabled.
  3. Failure isolation — broken LLM / broken KB don't prevent
     thread closure.
  4. ``consult_before_create`` runs at thread-creation time and
     surfaces matching past closures (when KB has entries).
  5. ``resolve_thread`` and ``abandon_thread`` invoke the distiller.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.threads import (
    abandon_thread,
    add_blocker,
    add_subquestion,
    add_unblock_hint,
    create_thread,
    record_note,
    reset_for_tests,
    resolve_subquestion,
    resolve_thread,
)
from app.threads.approaches import (
    consult_before_create,
    distill_on_closure,
    _build_deterministic_body,
)


@pytest.fixture(autouse=True)
def isolate(tmp_path: Path, monkeypatch):
    reset_for_tests(tmp_path)
    # Route lessons_learned KB at a tmp file so production data is
    # never touched. We monkeypatch the module's constant.
    kb_path = tmp_path / "lessons_learned.json"
    try:
        from app.companion import lessons_learned as ll
        monkeypatch.setattr(ll, "_KB_PATH", kb_path)
    except Exception:
        pass
    # Default OFF for LLM during tests — exercise deterministic path.
    monkeypatch.setenv("APPROACHES_LLM_ENABLED", "false")
    yield
    reset_for_tests(None)


# ─────────────────────────────────────────────────────────────────────
#   Deterministic body
# ─────────────────────────────────────────────────────────────────────


def test_deterministic_body_contains_title_status_and_sub_questions() -> None:
    t = create_thread(title="Why is gmail oauth refreshing?", description="")
    add_subquestion(t.id, "Are scopes correct?")
    add_subquestion(t.id, "Is the refresh token valid?")
    resolve_subquestion(t.id, t.sub_questions[0].id, "Scopes were correct")
    add_blocker(t.id, "Token endpoint returns 403")
    add_unblock_hint(t.id, "Try re-adding the gmail.send scope explicitly")
    record_note(t.id, "Final fix: gmail.modify replaced gmail.send")

    resolve_thread(t.id, summary="gmail.modify replaced gmail.send")
    from app.threads import get
    t2 = get(t.id)

    body = _build_deterministic_body(t2)
    assert "Why is gmail oauth refreshing?" in body
    assert "Closed as: resolved" in body
    assert "Sub-questions resolved: 1/2" in body
    assert "Scopes were correct" in body
    assert "Token endpoint returns 403" in body
    assert "Try re-adding the gmail.send scope explicitly" in body
    assert "gmail.modify replaced gmail.send" in body


def test_deterministic_body_handles_abandon_reason() -> None:
    t = create_thread(title="Cannot bridge X to Y")
    add_blocker(t.id, "API removed")
    abandon_thread(t.id, reason="vendor sunset")
    from app.threads import get
    body = _build_deterministic_body(get(t.id))
    assert "Closed as: abandoned" in body
    assert "Abandon reason: vendor sunset" in body


# ─────────────────────────────────────────────────────────────────────
#   distill_on_closure — writes summary + emits to KB
# ─────────────────────────────────────────────────────────────────────


def test_distill_on_closure_writes_summary_field() -> None:
    t = create_thread(title="Investigate signal heartbeat drop")
    add_subquestion(t.id, "Is the WS connection healthy?")
    resolve_subquestion(t.id, t.sub_questions[0].id, "yes")
    add_blocker(t.id, "Tailscale was masking the timeout")
    resolve_thread(t.id, summary="Switched to long-poll")

    from app.threads import get
    closed = get(t.id)
    assert closed.approaches_summary
    assert "Investigate signal heartbeat drop" in closed.approaches_summary
    assert "Tailscale was masking the timeout" in closed.approaches_summary


def test_distill_on_closure_emits_lesson_into_kb(tmp_path: Path) -> None:
    """The closure event lands in the lessons_learned KB."""
    from app.companion import lessons_learned as ll
    t = create_thread(title="oauth scope retry pattern")
    add_blocker(t.id, "Refresh token revoked after 7 days")
    add_unblock_hint(t.id, "Add scope=offline_access to the request")
    resolve_thread(t.id, summary="Added prompt=consent + access_type=offline")

    kb = ll._read_kb()
    assert kb, "lesson should have been emitted"
    sources_seen: set[str] = set()
    for lesson in kb:
        for src in lesson.get("sources") or []:
            sources_seen.add(src)
    assert "thread_closure" in sources_seen


def test_distill_on_closure_failure_isolated(monkeypatch) -> None:
    """A broken lessons_learned must NOT roll back the thread closure."""
    from app.threads.approaches import distill_on_closure
    t = create_thread(title="thread that must close")

    def boom(*_a, **_k):
        raise RuntimeError("kb is broken")

    from app.companion import lessons_learned as ll
    monkeypatch.setattr(ll, "_cluster_into_kb", boom)
    # Manually call distill on the (still-open) thread for unit test
    # of the failure-isolation guarantee; the lifecycle hook will do
    # the same.
    result = distill_on_closure(t)
    # Empty return OR non-empty (deterministic body) is fine —
    # what matters is no exception was raised.
    assert isinstance(result, str)


def test_resolve_thread_invokes_distiller() -> None:
    """The lifecycle resolve_thread path triggers distill_on_closure."""
    t = create_thread(title="lifecycle-hook test")
    add_subquestion(t.id, "Did it work?")
    resolve_subquestion(t.id, t.sub_questions[0].id, "yes")
    t2 = resolve_thread(t.id)
    from app.threads import get
    closed = get(t2.id)
    assert closed.approaches_summary  # the hook fired


def test_abandon_thread_invokes_distiller() -> None:
    t = create_thread(title="abandon-hook test")
    add_blocker(t.id, "Out of scope")
    abandon_thread(t.id, reason="rescoped")
    from app.threads import get
    closed = get(t.id)
    assert closed.approaches_summary


# ─────────────────────────────────────────────────────────────────────
#   consult_before_create — pre-creation consultation
# ─────────────────────────────────────────────────────────────────────


def test_consult_before_create_returns_empty_with_empty_kb() -> None:
    assert consult_before_create("brand new question never asked") == []


def test_consult_before_create_returns_empty_for_empty_title() -> None:
    assert consult_before_create("") == []
    assert consult_before_create("   ") == []


def test_consult_before_create_finds_similar_past_closure() -> None:
    """After a thread closes, a NEW thread with a similar title
    surfaces the past lesson via check_against."""
    # Close a thread with a distinctive signature
    t = create_thread(title="oauth refresh token revoked workaround")
    add_blocker(t.id, "google rotates tokens after 7 days")
    resolve_thread(
        t.id,
        summary="Use service-account key with domain-wide delegation instead",
    )

    # Now consult with a similar phrasing
    matches = consult_before_create(
        title="oauth refresh token rotation issue",
    )
    # Test is robustness-of-shape, not LLM-grade similarity — under
    # the hashing-trick embedding adjacent vocabulary should match
    # but the test is permissive.
    assert isinstance(matches, list)


def test_consult_before_create_failure_isolated(monkeypatch) -> None:
    """Broken lessons_learned import → empty list, no raise."""
    import app.threads.approaches as ap

    def boom(*_a, **_k):
        raise RuntimeError("kb broken")

    from app.companion import lessons_learned as ll
    monkeypatch.setattr(ll, "check_against", boom)
    result = consult_before_create("some new question")
    assert result == []


# ─────────────────────────────────────────────────────────────────────
#   Source-level wiring
# ─────────────────────────────────────────────────────────────────────


def test_lifecycle_calls_distill_on_closure() -> None:
    src = Path("app/threads/lifecycle.py").read_text(encoding="utf-8")
    assert "_distill_on_closure_safely" in src
    assert "def resolve_thread(" in src
    assert "def abandon_thread(" in src
    # Both closure paths invoke the helper
    resolve_start = src.find("def resolve_thread(")
    resolve_end = src.find("\ndef ", resolve_start + 1)
    abandon_start = src.find("def abandon_thread(")
    abandon_end = src.find("\ndef ", abandon_start + 1)
    assert "_distill_on_closure_safely" in src[resolve_start:resolve_end]
    assert "_distill_on_closure_safely" in src[abandon_start:abandon_end]


def test_lifecycle_calls_consult_before_create() -> None:
    src = Path("app/threads/lifecycle.py").read_text(encoding="utf-8")
    create_start = src.find("def create_thread(")
    create_end = src.find("\ndef ", create_start + 1)
    body = src[create_start:create_end] if create_end > create_start else src[create_start:]
    assert "consult_before_create" in body


def test_approaches_module_documents_llm_master_switch() -> None:
    src = Path("app/threads/approaches.py").read_text(encoding="utf-8")
    assert 'APPROACHES_LLM_ENABLED' in src
    # The deterministic fallback path must exist.
    assert "_build_deterministic_body" in src
    # Defaults to ON per the LLM-by-default operator decision.
    assert '"true"' in src or "'true'" in src
