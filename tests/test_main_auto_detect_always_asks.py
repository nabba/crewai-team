"""Locks down the workspace auto-detect routing rule.

Before 2026-05-09 the routing had three modes:
  1. No explicit user pick → AUTO-SWITCH silently.
  2. Explicit pick + detection differs → propose via Signal.
  3. Detection matches current → no-op.

Operator feedback after PR #71 ("expand keyword coverage for KaiCart /
Archibal / PLG") was explicit: "no automatic switching even if
workspace is not picked. always ask as in bullet 1."

Mode 1 was therefore collapsed into Mode 2 — the system **always
proposes** when detection differs from current, regardless of whether
the user has explicitly picked a workspace before. Two-mode logic
now lives in app/main.py:1600+.

These tests freeze the behaviour at the source level so a future
refactor can't silently revert. The auto-detect block is inline
inside main.py's request handler (not extractable without a much
bigger change), so we assert on its source rather than calling it.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_MAIN_PY = Path(__file__).resolve().parent.parent / "app" / "main.py"
_AUTO_DETECT_BLOCK_HEADER = (
    "Two-mode workspace auto-detection (revised 2026-05-09)"
)


@pytest.fixture(scope="module")
def main_src() -> str:
    """Read main.py once per test module."""
    if not _MAIN_PY.is_file():
        pytest.skip(f"main.py not at expected path: {_MAIN_PY}")
    return _MAIN_PY.read_text(encoding="utf-8")


# ── Comment-block contract ──────────────────────────────────────────


class TestRoutingCommentContract:
    """The header comment is the source of truth for the operator's
    intent. If it drifts, the runtime behaviour is suspect."""

    def test_two_mode_header_present(self, main_src: str) -> None:
        assert _AUTO_DETECT_BLOCK_HEADER in main_src, (
            "main.py auto-detect block must declare 'Two-mode' rule "
            "in its header comment; the operator's 2026-05-09 "
            "explicit ask was 'always ask, never auto-switch'."
        )

    def test_three_mode_header_absent(self, main_src: str) -> None:
        """Pre-fix wording must NOT be present — would indicate a
        refactor reintroduced the old behaviour."""
        assert (
            "Three-mode workspace auto-detection" not in main_src
        ), (
            "main.py still has the pre-2026-05-09 'Three-mode' header. "
            "The operator wants the auto-switch path removed."
        )


# ── Behavioural contract ────────────────────────────────────────────


class TestNoAutoSwitchPath:
    """Source-level checks that the auto-switch branch is gone."""

    def _auto_detect_block(self, src: str) -> str:
        """Slice out the auto-detect block via the header comment.

        Slightly fragile — depends on the comment header staying
        roughly where it is. If the slice fails the block-content
        tests below skip rather than fail spuriously.
        """
        m = re.search(
            r"# Two-mode workspace auto-detection.*?Mirror to dashboard",
            src,
            re.DOTALL,
        )
        if not m:
            pytest.skip("could not slice auto-detect block from main.py")
        return m.group(0)

    def test_no_user_chose_branch(self, main_src: str) -> None:
        """The pre-fix code had an `elif user_chose:` branch that
        gated whether to propose. After the 2026-05-09 fix, propose
        runs unconditionally on mismatch — no `user_chose` flag
        anywhere in the dispatch."""
        block = self._auto_detect_block(main_src)
        assert "user_chose" not in block, (
            "auto-detect block still references `user_chose`; the "
            "2026-05-09 fix removed that branch — the system always "
            "proposes regardless of whether the user has explicitly "
            "picked a workspace yet."
        )

    def test_no_silent_auto_switch(self, main_src: str) -> None:
        """The pre-fix `else:` branch called `cp.switch(detected,
        source="auto")` directly. Post-fix, the only switch path
        through this block goes via `propose()` (which fires only
        after the user 👍s the Signal proposal)."""
        block = self._auto_detect_block(main_src)
        # The string `cp.switch(detected, source="auto")` is what
        # used to silently switch. It must not appear inside the
        # auto-detect block.
        assert 'cp.switch(detected, source="auto")' not in block, (
            "auto-detect block silently switches via cp.switch(...);"
            " operator wants explicit user confirmation only."
        )

    def test_proposes_on_mismatch(self, main_src: str) -> None:
        """The single switch path that remains is `propose(...)` —
        guarded by `has_recent_decision` to avoid asking twice in a
        short window."""
        block = self._auto_detect_block(main_src)
        assert "propose(" in block, (
            "auto-detect block must still call propose() on detection "
            "mismatch — that's the whole point of the 'always ask' rule."
        )
        assert "has_recent_decision" in block, (
            "auto-detect block must guard propose() with "
            "has_recent_decision() to avoid pestering the user."
        )


# ── propose() side: unchanged behaviour ─────────────────────────────


class TestProposeFunctionUnchanged:
    """Sanity that the propose() function and its persistence still
    behave the way the rest of the system expects. The 2026-05-09
    fix only changes the *call site* in main.py, not the proposal
    queue."""

    def test_propose_still_persists_entry(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If the propose() function regresses, the auto-detect path
        is silent again. Lock the existing contract."""
        from app import workspace_switch_proposals as wsp

        monkeypatch.setattr(
            wsp, "_QUEUE_PATH", tmp_path / "queue.json",
        )
        # The call we now make from main.py for ALL mismatches
        proposal_id = wsp.propose(
            detected_name="eesti mets",
            current_name="default",
            sender="+1user",
            notifier=lambda *a, **kw: 1700000000,  # injected; no Signal needed
        )
        assert proposal_id is not None
        entries = wsp._load()
        assert any(e["proposal_id"] == proposal_id for e in entries)
        # Specifically: the entry knows what was proposed, what the
        # current was, and who to ask. These fields back the eventual
        # 👍/👎 reaction handler.
        entry = [e for e in entries if e["proposal_id"] == proposal_id][0]
        assert entry["detected_name"] == "eesti mets"
        assert entry["current_name"] == "default"
        assert entry["sender"] == "+1user"
