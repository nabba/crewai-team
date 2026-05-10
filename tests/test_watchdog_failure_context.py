"""Regression: watchdog apology must surface the LAST-KNOWN failure
reason instead of generic "narrow your question" advice (Cure C,
2026-05-10).

Pre-fix shape (the operator-reported forest-age failure):

  Coding crew hit vetting rejection (truncated mid-function +
  hallucinated GEE asset ID + no graphic produced). The retry path
  ran, hit similar issues, eventually output-progress went stale
  for 5 minutes → watchdog fired the ``output-stall`` apology:

    "Sorry — the task stopped producing partial results (no new
    rows / findings for several minutes). I'll deliver what's been
    streamed so far; please re-send a narrower question to fill
    the gaps."

  The actual cause was vetting rejections with three specific
  issues, NONE of which is fixable by "narrowing the question".
  The misleading apology hid the real problem.

Post-fix:
  • Failure points (vetting reject, artifact missing, completion
    truncation, dispatch exceptions) call
    ``record_failure_context(kind, detail)`` to stash the cause
    on the task's progress tracker.
  • The watchdog reads the stashed context via
    ``get_failure_context`` and weaves a per-kind explanation
    into the apology — the user sees the actual reason and
    actionable next step.
"""
from __future__ import annotations

import pytest

from app.observability.task_progress import (
    current_task_id,
    get_failure_context,
    record_failure_context,
    reset_task,
)


# ── Tracker API ────────────────────────────────────────────────────


@pytest.fixture
def task_tid():
    """Provide a unique task id; reset state at start AND end so
    tests don't trample each other through the module-global dict."""
    tid = "test-task-cure-c-unique"
    reset_task(tid)
    token = current_task_id.set(tid)
    yield tid
    current_task_id.reset(token)
    reset_task(tid)


class TestFailureContextRecord:

    def test_record_with_explicit_tid(self, task_tid: str) -> None:
        record_failure_context(
            "vetting_fail",
            "missing source citations; truncated mid-function",
            task_id=task_tid,
        )
        ctx = get_failure_context(task_tid)
        assert ctx is not None
        assert ctx["kind"] == "vetting_fail"
        assert "missing source citations" in ctx["detail"]
        assert "age_s" in ctx
        assert ctx["age_s"] >= 0

    def test_record_uses_contextvar_when_tid_omitted(self, task_tid: str) -> None:
        record_failure_context("artifact_missing", "expected .png; got code-only")
        ctx = get_failure_context(task_tid)
        assert ctx is not None
        assert ctx["kind"] == "artifact_missing"

    def test_latest_overwrites(self, task_tid: str) -> None:
        record_failure_context("vetting_fail", "first failure")
        record_failure_context("artifact_missing", "second failure")
        ctx = get_failure_context(task_tid)
        assert ctx["kind"] == "artifact_missing"
        assert ctx["detail"] == "second failure"

    def test_returns_none_for_clean_task(self) -> None:
        assert get_failure_context("never-set-task") is None

    def test_returns_none_for_empty_tid(self) -> None:
        assert get_failure_context("") is None

    def test_reset_clears_failure_context(self, task_tid: str) -> None:
        record_failure_context("vetting_fail", "something")
        assert get_failure_context(task_tid) is not None
        reset_task(task_tid)
        assert get_failure_context(task_tid) is None

    def test_kind_truncated_to_64_chars(self, task_tid: str) -> None:
        record_failure_context("x" * 200, "y")
        ctx = get_failure_context(task_tid)
        assert len(ctx["kind"]) == 64

    def test_detail_truncated_to_500_chars(self, task_tid: str) -> None:
        record_failure_context("vetting_fail", "x" * 1000)
        ctx = get_failure_context(task_tid)
        assert len(ctx["detail"]) == 500


# ── Apology suffix formatting ──────────────────────────────────────


class TestFormatFailureContextSuffix:
    """Per-failure-kind templated apology suffixes."""

    def _format(self, ctx):
        from app.main import _format_failure_context_suffix
        return _format_failure_context_suffix(ctx)

    def test_none_returns_empty(self) -> None:
        assert self._format(None) == ""

    def test_empty_dict_returns_empty(self) -> None:
        assert self._format({}) == ""

    def test_vetting_fail_template(self) -> None:
        suffix = self._format({
            "kind": "vetting_fail",
            "detail": "missing citations; truncated mid-function",
            "age_s": 12,
        })
        assert "rejected by quality review" in suffix
        assert "missing citations" in suffix
        # The template prompts the user toward an actionable fix.
        assert "address" in suffix.lower() or "gaps" in suffix.lower()

    def test_artifact_missing_template(self) -> None:
        suffix = self._format({
            "kind": "artifact_missing",
            "detail": "expected .png; got code-only",
            "age_s": 5,
        })
        assert "artifact-producing" in suffix
        assert ".png" in suffix
        assert "execute" in suffix.lower(), (
            "should hint at executing the script"
        )

    def test_completion_truncated_template(self) -> None:
        suffix = self._format({
            "kind": "completion_truncated",
            "detail": "model=claude-sonnet-4-6 max_tokens=4096 partial_chars=8000",
            "age_s": 3,
        })
        assert "cut off" in suffix.lower() or "truncated" in suffix.lower()
        assert "max_tokens" in suffix or "token" in suffix.lower()

    def test_unknown_kind_renders_generically(self) -> None:
        """Future failure kinds must surface SOMETHING actionable —
        not silently drop."""
        suffix = self._format({
            "kind": "some_future_kind",
            "detail": "details about it",
            "age_s": 1,
        })
        assert "some_future_kind" in suffix
        assert "details about it" in suffix

    def test_long_detail_truncated_in_suffix(self) -> None:
        suffix = self._format({
            "kind": "vetting_fail",
            "detail": "x" * 500,
            "age_s": 1,
        })
        # 300-char cap + ellipsis
        assert "…" in suffix
        assert "x" * 300 in suffix

    def test_malformed_ctx_does_not_raise(self) -> None:
        """The watchdog path is already handling a timeout — a bug
        here must NOT swallow the underlying failure."""
        # Missing 'detail' key.
        suffix = self._format({"kind": "vetting_fail"})
        assert isinstance(suffix, str)


# ── Wiring contracts ──────────────────────────────────────────────


class TestFailurePointsRecordContext:
    """The context helper is useless if the failure points don't
    record into it. Source-grep that each one wires in."""

    def test_vetting_records_on_fail(self) -> None:
        from pathlib import Path
        src = (
            Path(__file__).resolve().parent.parent / "app" / "vetting.py"
        ).read_text(encoding="utf-8")
        # The FAIL branch must call record_failure_context with
        # kind="vetting_fail".
        idx = src.index('elif verdict == "FAIL":')
        end = src.index("else:", idx)
        block = src[idx:end]
        assert "record_failure_context" in block
        assert '"vetting_fail"' in block

    def test_artifact_verifier_records_on_failure(self) -> None:
        from pathlib import Path
        src = (
            Path(__file__).resolve().parent.parent
            / "app" / "agents" / "commander" / "orchestrator.py"
        ).read_text(encoding="utf-8")
        idx = src.index("except ArtifactNotProduced as _exc:")
        # Slice the except block.
        end = idx + 2000
        block = src[idx:end]
        assert "record_failure_context" in block
        assert '"artifact_missing"' in block

    def test_completion_guard_records_on_truncation(self) -> None:
        from pathlib import Path
        src = (
            Path(__file__).resolve().parent.parent
            / "app" / "llm_completion_guard.py"
        ).read_text(encoding="utf-8")
        assert "record_failure_context" in src
        assert '"completion_truncated"' in src


class TestWatchdogReadsFailureContext:
    """The watchdog's apology message in app/main.py must read
    failure context AND format a suffix."""

    def test_main_pulls_failure_context_in_timeout_path(self) -> None:
        from pathlib import Path
        src = (
            Path(__file__).resolve().parent.parent / "app" / "main.py"
        ).read_text(encoding="utf-8")
        # The TimeoutError handler must reference get_failure_context.
        idx = src.index("except asyncio.TimeoutError as _to_exc:")
        end = src.index("except Exception as _handle_exc:", idx)
        block = src[idx:end]
        assert "get_failure_context" in block
        assert "_format_failure_context_suffix" in block
