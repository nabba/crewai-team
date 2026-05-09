"""Regression: the Signal ack message for change-request approvals
must reflect whether the apply step actually succeeded.

Pre-fix shape (the operator-reported bug):

    ✅ Change request eb677b22… approved + applied.
      ok: False
      branch: ?
      PR: (failed to open)
      module reload: None
      ERROR: host bridge unreachable; cannot write file or run git

The headline ("✅ approved + applied") contradicts the body
("ok: False / ERROR: …"). Operator can't tell at a glance whether
the change is on disk.

Post-fix shape:

  Success path:
    ✅ Change request eb677b22… approved + applied.
      branch: auto/change_eb677b22…
      PR: https://github.com/…/pull/123
      module reload: reloaded app.foo

  Failure path:
    ⚠️ Change request eb677b22… approved, but apply FAILED.
      ERROR: host bridge unreachable; …
      Status is now APPLY_FAILED — use the 'Retry apply' button …
      branch: (none)
      PR: (not opened)

These tests assert source-grep contracts (the ack-formatter is
inline inside main.py's request handler — extracting it cleanly
is a larger refactor; source-level tests catch the regression
without that scope).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_MAIN_PY = Path(__file__).resolve().parent.parent / "app" / "main.py"


@pytest.fixture(scope="module")
def main_src() -> str:
    if not _MAIN_PY.is_file():
        pytest.skip(f"main.py not found at {_MAIN_PY}")
    return _MAIN_PY.read_text(encoding="utf-8")


# ── Honest-ack contract ─────────────────────────────────────────────


class TestAckHeadlineReflectsApplyOutcome:

    def _ack_block(self, src: str) -> str:
        """Slice the Signal-ack section from the change-request
        reaction handler.  Anchored on a marker comment from the
        post-fix code so a regression that drops the conditional
        renders the slice empty (and the tests skip rather than
        false-pass)."""
        m = re.search(
            r"# Honest ack:.*?(?=\n\s+else:\s*\n\s+await loop\.run_in_executor)",
            src,
            re.DOTALL,
        )
        if not m:
            pytest.skip("could not slice ack block; marker comment moved")
        return m.group(0)

    def test_success_branch_uses_check_emoji(self, main_src: str) -> None:
        block = self._ack_block(main_src)
        # Success line
        assert "✅" in block, "success branch must use ✅"
        assert "approved + applied" in block, (
            "success branch must say 'approved + applied'"
        )

    def test_failure_branch_uses_warning_emoji(self, main_src: str) -> None:
        block = self._ack_block(main_src)
        assert "⚠️" in block, "failure branch must use ⚠️ (not ✅)"
        assert "apply FAILED" in block, (
            "failure branch must say 'apply FAILED' (not 'applied')"
        )

    def test_failure_branch_mentions_retry(self, main_src: str) -> None:
        """Failure should point the user at the retry path so they
        know what to do next."""
        block = self._ack_block(main_src)
        assert "Retry apply" in block, (
            "failure ack should mention the Retry button at /cp/changes"
        )
        assert "/cp/changes" in block

    def test_failure_branch_includes_error_text(self, main_src: str) -> None:
        block = self._ack_block(main_src)
        # The error text must appear in the failure ack so the
        # operator can see why the apply failed.
        assert "apply_result.error" in block

    def test_no_unconditional_applied_headline(self, main_src: str) -> None:
        """Pre-fix bug: a single hard-coded f-string said '✅ approved +
        applied' regardless of apply_result.ok. After the fix, the
        f-string with that headline must only appear under the
        `if apply_result.ok:` branch.

        We count non-comment lines containing 'approved + applied' —
        explanatory comments are fine, only the live code matters.
        """
        block = self._ack_block(main_src)

        def _is_code(line: str) -> bool:
            stripped = line.strip()
            return bool(stripped) and not stripped.startswith("#")

        applied_code_lines = [
            ln for ln in block.splitlines()
            if "approved + applied" in ln and _is_code(ln)
        ]
        assert len(applied_code_lines) == 1, (
            f"'approved + applied' headline should appear in exactly one "
            f"non-comment line (inside the apply_result.ok branch), "
            f"found {len(applied_code_lines)}: {applied_code_lines}"
        )
        assert "if apply_result.ok:" in block

        failed_code_lines = [
            ln for ln in block.splitlines()
            if "apply FAILED" in ln and _is_code(ln)
        ]
        assert len(failed_code_lines) == 1


# ── Reject branch unchanged ─────────────────────────────────────────


class TestRejectAckUnchanged:
    """The 👎-reject path's ack message wasn't part of the bug — it
    correctly used ❌. Lock that in so a future refactor doesn't
    accidentally swap the emojis."""

    def test_reject_uses_x_emoji(self, main_src: str) -> None:
        # Find the line that builds the reject-ack
        m = re.search(
            r'ack_msg\s*=\s*f"❌ Change request \{cr_id\} rejected\.',
            main_src,
        )
        assert m is not None, (
            "reject ack should still use '❌ rejected.'"
        )
