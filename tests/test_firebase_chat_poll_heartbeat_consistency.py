"""Regression: firebase-chat-poll must heartbeat consistently with the
other 8 firebase pollers, regardless of FIREBASE_ENABLED.

Pre-fix shape (the operator-reported bug, 2026-05-10):

  Self-heal: listener `firebase-chat-poll` has produced NO heartbeat
  — known listener never started, or crashed before its first loop
  iteration. Other listeners are healthy (heartbeat subsystem on).

  ``start_chat_inbox_poller`` had a function-level early-return:

      if not _firebase_enabled():
          logger.debug("...skipped (FIREBASE_ENABLED=0)")
          return

  …so on a deployment with FIREBASE_ENABLED unset (laptop dev, CI),
  the thread never started and the heartbeat file never appeared.
  The other 8 firebase pollers (mode / kb / phil / fiction /
  episteme / experiential / aesthetics / tensions) start their
  thread unconditionally and heartbeat regardless — only the chat
  poller had the gate.

Post-fix:

  • Function starts the thread unconditionally.
  • Heartbeat fires once before the first wait() (so the monitor
    sees liveness within ~1 s of startup) and at the top of every
    iteration.
  • If _get_db() returns None (Firebase disabled / unreachable),
    the loop continues — does NOT exit the thread.
  • This matches the kb / mode / phil / fiction / episteme /
    experiential / aesthetics / tensions pattern.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
_LISTENERS = _REPO_ROOT / "app" / "firebase" / "listeners.py"


@pytest.fixture(scope="module")
def listeners_src() -> str:
    return _LISTENERS.read_text(encoding="utf-8")


# ── KNOWN_LISTENERS contract ────────────────────────────────────────


class TestChatPollIsKnown:
    """firebase-chat-poll must remain in KNOWN_LISTENERS — the fix is
    to make the poller behave consistently, NOT to remove it from
    the expected set."""

    def test_chat_poll_in_known_listeners(self) -> None:
        from app.healing.listener_heartbeats import KNOWN_LISTENERS
        assert "firebase-chat-poll" in KNOWN_LISTENERS, (
            "firebase-chat-poll must stay in KNOWN_LISTENERS — the "
            "fix makes the poller honor the contract, not remove it"
        )


# ── Source-grep contracts ──────────────────────────────────────────


class TestNoFunctionLevelFirebaseGate:
    """The pre-fix shape was a function-level early-return gating on
    _firebase_enabled(). It must not return."""

    def test_no_early_return_on_firebase_enabled(
        self, listeners_src: str,
    ) -> None:
        # Slice the chat poller function body — from its def through
        # the inner threading.Thread(...) line.
        idx = listeners_src.index("def start_chat_inbox_poller(")
        end = listeners_src.index(
            'name="firebase-chat-poll")', idx,
        )
        body = listeners_src[idx:end]

        # The pre-fix pattern was:
        #   if not _firebase_enabled():
        #       logger.debug(...)
        #       return
        # Lock that pattern out — at function-top-level (8 spaces or
        # less of leading whitespace).
        forbidden = re.search(
            r"\n\s{0,8}if\s+not\s+_firebase_enabled\(\)\s*:"
            r"(?:\s*\n\s*[^\n]+)*?\s*\n\s*return\b",
            body,
        )
        assert forbidden is None, (
            "start_chat_inbox_poller must NOT early-return on "
            "_firebase_enabled() — the thread must always start so the "
            "heartbeat fires"
        )


class TestThreadStartsUnconditionally:
    """The threading.Thread(name='firebase-chat-poll') must always be
    reached after start_chat_inbox_poller() returns — no precondition
    that could prevent thread creation."""

    def test_thread_creation_at_top_level(
        self, listeners_src: str,
    ) -> None:
        # Slice from the def to the end of the function (next def at
        # column 0, or EOF).
        m = re.search(
            r"def start_chat_inbox_poller\(.*?\n(.*?)(?=\n(?:def |class )|\Z)",
            listeners_src, re.DOTALL,
        )
        assert m is not None
        body = m.group(0)

        # threading.Thread(...) must appear.
        thread_call = (
            'threading.Thread(target=_poll, daemon=True, '
            'name="firebase-chat-poll")'
        )
        thread_line_idx = body.find(thread_call)
        assert thread_line_idx >= 0, (
            "thread creation must remain in the function"
        )

        # The line must be at function-body indentation (4 spaces).
        # Read from the preceding newline and inspect the leading
        # whitespace before any non-whitespace chars.
        line_start = body.rfind("\n", 0, thread_line_idx) + 1
        line = body[line_start:thread_line_idx + len(thread_call)]
        # Indentation = leading whitespace before first non-WS char.
        m2 = re.match(r"^(\s*)", line)
        indent = m2.group(1) if m2 else ""
        normalized = indent.replace("\t", "    ")
        assert normalized == "    ", (
            f"thread creation must be at function-body indentation "
            f"(4 spaces); got indent={indent!r}, line={line!r}"
        )


class TestHeartbeatBeforeFirstWait:
    """The heartbeat must fire BEFORE the first ``_chat_poll_stop.wait``
    so the monitor sees liveness within ~1 s of startup, not 3 s in."""

    def test_first_touch_precedes_first_wait(
        self, listeners_src: str,
    ) -> None:
        # Slice the _poll inner function body.
        m = re.search(
            r"def _poll\(\):\s*\n(.*?)(?=\n    t = threading\.Thread)",
            listeners_src, re.DOTALL,
        )
        assert m is not None, "could not slice _poll body"
        body = m.group(1)

        first_touch = body.find('touch("firebase-chat-poll")')
        first_wait = body.find("_chat_poll_stop.wait")
        assert first_touch >= 0, "heartbeat call must remain"
        assert first_wait >= 0, "wait() must remain"
        assert first_touch < first_wait, (
            "heartbeat must fire BEFORE the first wait so the monitor "
            "sees liveness immediately — pre-fix it fired only after "
            "the first 3-s wait, leaving a 3-s window where the "
            "listener looked dead"
        )


class TestNoDbContinuesNotReturns:
    """When _get_db() returns None, the loop must `continue`, not
    `return` — pre-fix the no-db path killed the thread."""

    def test_no_db_path_uses_continue(
        self, listeners_src: str,
    ) -> None:
        # Slice the chat poller function.
        m = re.search(
            r"def start_chat_inbox_poller\(.*?\n(.*?)(?=\n(?:def |class )|\Z)",
            listeners_src, re.DOTALL,
        )
        assert m is not None
        body = m.group(0)

        # Find the "if not db:" line in the chat poller body.
        m2 = re.search(
            r"db = _get_db\(\)\s*\n\s*if not db:\s*\n((?:\s*#[^\n]*\n)*)\s*(\w+)",
            body,
        )
        assert m2 is not None, (
            "chat poller must have a `db = _get_db(); if not db:` "
            "guard to handle Firebase-down gracefully"
        )
        action = m2.group(2)
        assert action == "continue", (
            f"no-db path must `continue` (loop survives); got {action!r}"
        )


# ── Functional simulation ──────────────────────────────────────────


class TestPollerHeartbeatsWhenFirebaseDisabled:
    """When _firebase_enabled() returns False (or _get_db returns None),
    the poller's first iteration must still produce a heartbeat file."""

    def test_heartbeat_appears_within_first_loop(
        self, monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        import time

        # Patch the heartbeat dir to a tmp path so we don't pollute
        # the real workspace.
        monkeypatch.setattr(
            "app.healing.listener_heartbeats._HEARTBEAT_DIR",
            tmp_path / "heartbeats",
        )

        # Patch _get_db to None so the poller hits the no-db continue.
        from app.firebase import listeners
        monkeypatch.setattr(listeners, "_get_db", lambda: None)

        # Reset the stop event so we can stop it after one iteration.
        listeners._chat_poll_stop = __import__("threading").Event()

        # No-op handler.
        def _handler(text: str) -> str:
            return ""

        listeners.start_chat_inbox_poller(_handler)

        # The pre-wait heartbeat fires immediately on _poll start.
        # Allow a brief moment for the thread to schedule.
        for _ in range(20):
            heartbeat = tmp_path / "heartbeats" / "firebase-chat-poll.heartbeat"
            if heartbeat.exists():
                break
            time.sleep(0.1)

        # Stop the poller.
        listeners._chat_poll_stop.set()

        assert heartbeat.exists(), (
            "heartbeat file must appear within ~2 s of starting the "
            "poller, even when _get_db returns None"
        )
