"""Cure C follow-up regressions surfaced by the operator's
13:39 forest-age task that stalled with the legacy generic
"narrower question" message.

Two bugs in the original Cure C (PR #101):

  Bug 1 — ContextVar propagation
  ───────────────────────────────
  ``record_failure_context`` defaults its task_id from the
  ``current_task_id`` ContextVar.  ``_handle_locked`` sets it
  in the commander thread.  But the orchestrator submits
  vetting to ``_ctx_pool`` via ``ThreadPoolExecutor.submit`` —
  which does NOT propagate ContextVars by default.  So when
  ``vet_response_detailed`` calls ``record_failure_context``
  in a worker thread, ``current_task_id.get()`` returns the
  default empty string, and the function silently no-ops.

  Cure: wrap the submit in ``contextvars.copy_context().run``.

  Bug 2 — Race between vetting + watchdog
  ────────────────────────────────────────
  When vetting takes >90s, the future-level timeout fires.
  The vetting thread KEEPS RUNNING and eventually calls
  ``record_failure_context`` — but by then the watchdog
  already fired its apology with no failure context to
  weave in.  Result: operator gets generic "narrow your
  question" even though vetting was about to record three
  specific issues.

  Cure: when the orchestrator catches the vetting timeout,
  record an explicit ``vetting_timeout`` failure context
  synchronously, in the orchestrator thread (where
  ContextVar IS set), so the watchdog has SOMETHING to
  surface.
"""
from __future__ import annotations

from pathlib import Path

import pytest


_ORCHESTRATOR = (
    Path(__file__).resolve().parent.parent
    / "app" / "agents" / "commander" / "orchestrator.py"
)


# ── Bug 1 — vetting submit copies context ───────────────────────────


class TestVettingSubmitCopiesContext:

    def test_orchestrator_wraps_vetting_submit_in_copy_context(self) -> None:
        """The vetting submission must use copy_context().run so
        ContextVars (current_task_id) propagate to the worker thread."""
        src = _ORCHESTRATOR.read_text(encoding="utf-8")
        # Find the vetting submission block.
        idx = src.find("_vet_future = _ctx_pool.submit(")
        assert idx >= 0, "could not locate vetting submission"
        # Read a 400-char window after.
        block = src[idx:idx + 600]
        assert "copy_context()" in block or "_vet_ctx" in block, (
            "vetting submission must wrap with copy_context() so "
            "current_task_id ContextVar propagates into the worker "
            "thread; otherwise record_failure_context inside vetting "
            "silently no-ops"
        )
        assert ".run" in block, (
            "must call ctx.run(fn, *args) to invoke vetting in the "
            "captured context"
        )


# ── Bug 2 — vetting timeout records a failure context ──────────────


class TestVettingTimeoutRecordsContext:

    def test_orchestrator_records_vetting_timeout(self) -> None:
        """When vetting times out (90s), the orchestrator must
        record an explicit ``vetting_timeout`` failure context
        synchronously so the watchdog's apology has something to
        surface — instead of falling back to generic "narrow your
        question" advice."""
        src = _ORCHESTRATOR.read_text(encoding="utf-8")
        # Find the except handler for vetting timeout.
        idx = src.find('"vetting did not complete in time')
        assert idx >= 0, "could not locate vetting-timeout handler"
        # Read 1500 chars from there to capture the except body.
        block = src[idx:idx + 1500]
        assert "record_failure_context" in block, (
            "vetting-timeout handler must call record_failure_context"
        )
        assert '"vetting_timeout"' in block, (
            "must record kind='vetting_timeout' so the watchdog's "
            "templated suffix renders"
        )

    def test_main_apology_has_vetting_timeout_template(self) -> None:
        """The watchdog formatter must have a template for the
        'vetting_timeout' kind so the suffix actually renders
        (vs falling through to the generic 'unknown kind' branch)."""
        main_src = (
            Path(__file__).resolve().parent.parent / "app" / "main.py"
        ).read_text(encoding="utf-8")
        # Find the templates dict.
        idx = main_src.find("_FAILURE_CONTEXT_TEMPLATES")
        end = main_src.find("\ndef _format_failure_context_suffix", idx)
        assert idx >= 0 and end > idx
        templates = main_src[idx:end]
        assert '"vetting_timeout"' in templates, (
            "main._FAILURE_CONTEXT_TEMPLATES must have a "
            "'vetting_timeout' entry for the suffix formatter"
        )


# ── End-to-end functional ───────────────────────────────────────────


class TestRecordFailureContextThreadPoolPropagation:
    """The actual contract: when a function inside _ctx_pool.submit
    calls record_failure_context (no explicit task_id), and the
    submit was wrapped with copy_context, the per-task tracker
    receives the right tid."""

    def test_record_propagates_through_copy_context(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from concurrent.futures import ThreadPoolExecutor
        import contextvars

        from app.observability.task_progress import (
            current_task_id,
            get_failure_context,
            record_failure_context,
            reset_task,
        )

        tid = "test-cure-c-propagation"
        reset_task(tid)
        token = current_task_id.set(tid)
        try:
            # Simulate the orchestrator's pattern: copy current ctx,
            # submit a function that calls record_failure_context.
            def _worker():
                # Inside the worker, current_task_id should still be
                # the value the orchestrator set.
                record_failure_context(
                    "vetting_fail",
                    "test detail; ctx propagated correctly",
                )

            ctx = contextvars.copy_context()
            with ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(ctx.run, _worker)
                fut.result(timeout=5)

            # Verify the recording landed under the right tid.
            stored = get_failure_context(tid)
            assert stored is not None, (
                "record_failure_context didn't propagate ContextVar "
                "into worker thread; copy_context.run did not preserve "
                "current_task_id"
            )
            assert stored["kind"] == "vetting_fail"
            assert "test detail" in stored["detail"]
        finally:
            current_task_id.reset(token)
            reset_task(tid)

    def test_record_silently_noops_without_copy_context(
        self,
    ) -> None:
        """Negative test: the same code path WITHOUT copy_context
        results in silent no-op — proves the bug exists when
        propagation is missing."""
        from concurrent.futures import ThreadPoolExecutor

        from app.observability.task_progress import (
            current_task_id,
            get_failure_context,
            record_failure_context,
            reset_task,
        )

        tid = "test-cure-c-no-propagation"
        reset_task(tid)
        token = current_task_id.set(tid)
        try:
            def _worker_no_ctx():
                # No ctx propagation — current_task_id.get() returns ""
                record_failure_context(
                    "vetting_fail", "should be lost",
                )

            with ThreadPoolExecutor(max_workers=1) as pool:
                # Direct submit, no copy_context.run
                fut = pool.submit(_worker_no_ctx)
                fut.result(timeout=5)

            # Should NOT have been recorded under our tid.
            assert get_failure_context(tid) is None, (
                "negative-test: without copy_context.run, ContextVar "
                "should NOT propagate — record_failure_context "
                "silently no-ops"
            )
        finally:
            current_task_id.reset(token)
            reset_task(tid)


class TestVettingTimeoutSuffixRenders:
    """The whole point of bug-2's fix is that the watchdog's
    apology now contains the vetting-timeout reason instead of
    generic advice."""

    def test_suffix_template_renders_vetting_timeout(self) -> None:
        from app.main import _format_failure_context_suffix

        suffix = _format_failure_context_suffix({
            "kind": "vetting_timeout",
            "detail": "crew=coding (TimeoutError)",
            "age_s": 5.0,
        })
        assert "quality-review" in suffix or "vetting" in suffix.lower()
        assert "90s budget" in suffix
        # And should mention the actionable next step.
        assert "narrow" in suffix.lower() or "narrowing" in suffix.lower()
