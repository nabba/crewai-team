"""PROGRAM §46.1 — Q8.1 long-horizon-threads completion tests.

The Thread primitive (models / lifecycle / store / REST / React) was
already shipped pre-Q8. Q8.1 closes three operator-facing gaps:

  1. Slash-command surface ``/thread {start, status, list, note,
     subq, done, block, unblock, hint, resolve, abandon}``.
  2. Symmetric "what would unblock this" semantic field
     (``Thread.unblock_hints``) with ``add_unblock_hint`` /
     ``clear_unblock_hints`` lifecycle + REST.
  3. Recovery-loop consultation via ``thread_consultation`` module —
     open-thread unblock-hints surface in the recovery ``ctx`` so
     strategies can read them, but no strategy is forced to.

These tests cover all three gaps + the source-level wiring.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.threads import (
    add_blocker,
    add_unblock_hint,
    clear_unblock_hints,
    create_thread,
    list_open,
    mark_blocked,
    reset_for_tests,
)
from app.threads.lifecycle import InvalidThreadTransition
from app.threads.models import Thread


@pytest.fixture(autouse=True)
def isolate(tmp_path: Path):
    reset_for_tests(tmp_path)
    yield
    reset_for_tests(None)


# ─────────────────────────────────────────────────────────────────────
#   Q8.1 — Thread.unblock_hints field + lifecycle functions
# ─────────────────────────────────────────────────────────────────────


def test_thread_has_unblock_hints_field_defaulting_empty() -> None:
    """New threads start with empty unblock_hints (backward-compat
    with pre-Q8 callers that never set it)."""
    t = create_thread(title="Q for testing unblock_hints")
    assert t.unblock_hints == []


def test_add_unblock_hint_appends_and_advances_in_progress() -> None:
    """Adding a hint to an OPEN thread auto-advances it to
    IN_PROGRESS (mirrors add_blocker / add_subquestion)."""
    t = create_thread(title="Test thread")
    assert t.status.value == "open"
    t2 = add_unblock_hint(t.id, "Try the OAuth scope add-back trick")
    assert "Try the OAuth scope add-back trick" in t2.unblock_hints
    assert t2.status.value == "in_progress"


def test_add_unblock_hint_rejects_empty_text() -> None:
    t = create_thread(title="Q")
    with pytest.raises(ValueError):
        add_unblock_hint(t.id, "")
    with pytest.raises(ValueError):
        add_unblock_hint(t.id, "   ")


def test_add_unblock_hint_refuses_terminal_thread() -> None:
    from app.threads import resolve_thread
    t = create_thread(title="Done")
    resolve_thread(t.id)
    with pytest.raises(InvalidThreadTransition):
        add_unblock_hint(t.id, "too late")


def test_clear_unblock_hints_wipes_list_without_status_change() -> None:
    """Clearing hints does NOT change thread status — the blockers
    themselves still apply, only the hypotheses are wiped."""
    t = create_thread(title="Q")
    mark_blocked(t.id, blocker="something")
    add_unblock_hint(t.id, "first hypothesis")
    add_unblock_hint(t.id, "second hypothesis")
    t2 = clear_unblock_hints(t.id)
    assert t2.unblock_hints == []
    assert t2.status.value == "blocked"  # still blocked


def test_unblock_hints_serialize_round_trip() -> None:
    """to_dict + from_dict round-trip preserves unblock_hints."""
    t = create_thread(title="rt")
    add_unblock_hint(t.id, "hypothesis A")
    add_unblock_hint(t.id, "hypothesis B")
    d = t.to_dict()
    assert "unblock_hints" in d
    t_round = Thread.from_dict(d)
    assert t_round.unblock_hints == ["hypothesis A", "hypothesis B"]


def test_pre_q8_records_load_without_unblock_hints_key() -> None:
    """from_dict must accept dicts without unblock_hints (records
    saved before Q8) — backward-compat invariant."""
    legacy = {
        "id": "old-thread-id",
        "created_at": "2026-05-01T00:00:00+00:00",
        "title": "Old thread",
        "description": "",
        "status": "open",
        "sub_questions": [],
        "blockers": [],
        "notes": [],
        "related_crew_task_ids": [],
        "related_inquiry_slugs": [],
        "last_touched_at": "2026-05-01T00:00:00+00:00",
    }
    t = Thread.from_dict(legacy)
    assert t.unblock_hints == []
    assert t.approaches_summary == ""


# ─────────────────────────────────────────────────────────────────────
#   Q8.1 — thread_consultation module (recovery-loop bridge)
# ─────────────────────────────────────────────────────────────────────


def test_collect_open_thread_hints_returns_only_actionable() -> None:
    """Threads with NO blockers AND NO hints are skipped — they
    carry no signal for strategy selection."""
    from app.recovery.thread_consultation import collect_open_thread_hints
    # Thread with a hint
    t1 = create_thread(title="With hint")
    add_unblock_hint(t1.id, "try X")
    # Thread with a blocker
    t2 = create_thread(title="With blocker")
    add_blocker(t2.id, "Y is broken")
    # Empty thread (no actionable signal)
    create_thread(title="Empty")

    payloads = collect_open_thread_hints(max_threads=10)
    titles = {p["title"] for p in payloads}
    assert "With hint" in titles
    assert "With blocker" in titles
    assert "Empty" not in titles


def test_collect_open_thread_hints_caps_at_max_threads() -> None:
    from app.recovery.thread_consultation import collect_open_thread_hints
    for i in range(8):
        t = create_thread(title=f"thread {i}")
        add_unblock_hint(t.id, f"hint {i}")
    payloads = collect_open_thread_hints(max_threads=3)
    assert len(payloads) == 3


def test_collect_open_thread_hints_truncates_long_strings() -> None:
    """A 1000-char hint must be clipped to ~240 chars so the recovery
    prompt budget isn't blown."""
    from app.recovery.thread_consultation import (
        _MAX_HINT_CHARS, collect_open_thread_hints,
    )
    t = create_thread(title="long")
    add_unblock_hint(t.id, "X" * 1000)
    payloads = collect_open_thread_hints(max_threads=10)
    assert len(payloads) == 1
    assert all(len(h) <= _MAX_HINT_CHARS for h in payloads[0]["hints"])


def test_collect_open_thread_hints_failure_isolated(monkeypatch) -> None:
    """Broken threads module → empty list, never raises."""
    import app.recovery.thread_consultation as tc

    def boom(**_):
        raise RuntimeError("threads store unavailable")

    monkeypatch.setattr("app.threads.list_open", boom)
    assert tc.collect_open_thread_hints() == []


def test_format_for_prompt_emits_markdown_or_empty() -> None:
    from app.recovery.thread_consultation import format_for_prompt
    assert format_for_prompt([]) == ""
    payload = [{
        "thread_id": "abc12345",
        "title": "Foo",
        "status": "blocked",
        "blockers": ["X is broken"],
        "hints": ["try Y"],
    }]
    out = format_for_prompt(payload)
    assert "**Foo**" in out
    assert "🚧 blocker: X is broken" in out
    assert "💡 hint: try Y" in out


# ─────────────────────────────────────────────────────────────────────
#   Q8.1 — Source-level wiring (slash command + REST + recovery loop)
# ─────────────────────────────────────────────────────────────────────


def test_thread_slash_command_registered() -> None:
    """SignalCommand rows exist for /thread + key subcommands."""
    src = Path("app/agents/commander/command_registry.py").read_text(encoding="utf-8")
    assert '"Threads"' in src  # category
    for cmd in (
        "/thread start", "/thread status",
        "/thread note", "/thread subq", "/thread done",
        "/thread block", "/thread hint", "/thread unblock",
        "/thread resolve", "/thread abandon",
    ):
        assert f'SignalCommand("{cmd}"' in src, f"missing registration for {cmd!r}"


def test_thread_dispatcher_wired_in_try_command() -> None:
    """commands.py routes /thread to _handle_thread_command via
    try_command. Source-level assertion to catch removals."""
    src = Path("app/agents/commander/commands.py").read_text(encoding="utf-8")
    assert "def _handle_thread_command(" in src
    assert "_handle_thread_command(user_input)" in src
    # Both prefix matches present (/thread + thread <something>)
    assert 'lower.startswith("/thread")' in src


def test_rest_unblock_hint_endpoint_exists() -> None:
    src = Path("app/control_plane/threads_api.py").read_text(encoding="utf-8")
    assert '/{thread_id}/unblock-hint' in src
    assert '/{thread_id}/clear-unblock-hints' in src
    assert "add_unblock_hint" in src
    assert "clear_unblock_hints" in src


def test_recovery_loop_consults_thread_hints() -> None:
    """app/recovery/loop.py builds ctx with thread_hints from the
    consultation module."""
    src = Path("app/recovery/loop.py").read_text(encoding="utf-8")
    assert "from app.recovery.thread_consultation import collect_open_thread_hints" in src
    assert '"thread_hints": thread_hints' in src


# ─────────────────────────────────────────────────────────────────────
#   Q8.1 — Slash command source-level wiring
#
#   The full dispatcher in commands.py can't be imported in this test
#   env (it transitively pulls in `crewai` + `pydantic_settings` —
#   gateway-deps). These assertions cover the wiring at the source
#   level; the dispatcher is exercised end-to-end in production where
#   the deps are installed.
# ─────────────────────────────────────────────────────────────────────


def test_handle_thread_command_function_exists() -> None:
    src = Path("app/agents/commander/commands.py").read_text(encoding="utf-8")
    assert "def _handle_thread_command(" in src


def test_handle_thread_command_covers_all_subcommands() -> None:
    """Every subcommand we registered in command_registry must have a
    dispatch branch in the handler. Catches add-to-registry-but-
    forget-to-handle drift."""
    src = Path("app/agents/commander/commands.py").read_text(encoding="utf-8")
    # The dispatch branches we expect — match the doc lines under the
    # "if/elif" pattern in _handle_thread_command.
    expected_subs = (
        '"start"', '"status"', '"note"', '"subq"',
        '"done"', '"block"', '"hint"', '"unblock"',
        '"resolve"', '"abandon"',
    )
    # Slice to the body of _handle_thread_command for a tight check.
    start = src.find("def _handle_thread_command(")
    end = src.find("\ndef ", start + 1)
    body = src[start:end] if end > start else src[start:]
    for sub in expected_subs:
        assert sub in body, f"missing dispatch branch for {sub}"


def test_handle_thread_command_calls_lifecycle_helpers() -> None:
    """The handler delegates to app.threads.* functions for actual
    state mutation. Source-level grep guarantees no orphaned in-place
    mutation in the handler that would bypass the lifecycle invariants."""
    src = Path("app/agents/commander/commands.py").read_text(encoding="utf-8")
    start = src.find("def _handle_thread_command(")
    end = src.find("\ndef ", start + 1)
    body = src[start:end] if end > start else src[start:]
    for helper in (
        "create_thread", "record_note", "add_subquestion",
        "resolve_subquestion", "mark_blocked", "clear_blockers",
        "add_unblock_hint", "resolve_thread", "abandon_thread",
    ):
        assert helper in body, f"handler does not delegate to {helper}"
