"""Regression: never silently accept a ``finish_reason == "length"``
LLM completion (Cure A, 2026-05-10).

Pre-fix shape (the operator-reported "30-min stall + please re-send"):

  Coding crew was asked to make a forest-age graphic for Estonia.
  The LLM produced a Python script, but the response was truncated
  mid-function ("ending at 'subtitle' with no closing"). The system
  proceeded as if the response were complete; only the downstream
  LLM-vetter happened to notice the syntactic incompleteness, by
  which point 15+ minutes had elapsed. Eventually the watchdog
  fired with a generic "narrow your question" message.

  Root cause: ``grep -rn finish_reason app/`` returned ZERO matches.
  The SDK's signal that the response was cut by ``max_tokens`` was
  available on every completion but never read.

Post-fix:
  ``app/llm_completion_guard.check_completion_truncation`` runs at
  the single point all LLM calls funnel through
  (``rate_throttle._throttled_completion``). On
  ``finish_reason == "length"`` it raises a typed
  ``CompletionTruncated`` exception with the partial text, model id,
  and max_tokens budget. Callers can no longer silently accept
  truncated output.
"""
from __future__ import annotations

import logging

import pytest

from app.llm_completion_guard import (
    CompletionTruncated,
    _extract_finish_reason,
    _extract_partial_text,
    check_completion_truncation,
    was_last_completion_truncated,
)


# ── Extractors are robust against shape variation ──────────────────


class TestExtractFinishReason:
    """``_extract_finish_reason`` must handle dict-shaped, object-shaped,
    and degenerate inputs. Returning None on missing data is correct —
    we never raise on shape variation; that would convert SDK changes
    into outages."""

    def test_dict_shape(self) -> None:
        resp = {"choices": [{"finish_reason": "length"}]}
        assert _extract_finish_reason(resp) == "length"

    def test_object_shape(self) -> None:
        class _Choice:
            finish_reason = "stop"

        class _Resp:
            choices = [_Choice()]

        assert _extract_finish_reason(_Resp()) == "stop"

    def test_none_response(self) -> None:
        assert _extract_finish_reason(None) is None

    def test_empty_choices(self) -> None:
        assert _extract_finish_reason({"choices": []}) is None

    def test_missing_finish_reason(self) -> None:
        assert _extract_finish_reason({"choices": [{}]}) is None

    def test_unparseable(self) -> None:
        assert _extract_finish_reason("not a response") is None


class TestExtractPartialText:
    """``_extract_partial_text`` must NOT raise on shape variation.
    Best-effort extraction; degrade to empty string on failure."""

    def test_modern_dict_shape(self) -> None:
        resp = {"choices": [{"message": {"content": "hello"}}]}
        assert _extract_partial_text(resp) == "hello"

    def test_modern_object_shape(self) -> None:
        class _Msg:
            content = "world"

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        assert _extract_partial_text(_Resp()) == "world"

    def test_legacy_text_field(self) -> None:
        resp = {"choices": [{"text": "legacy completion text"}]}
        assert _extract_partial_text(resp) == "legacy completion text"

    def test_missing_content(self) -> None:
        assert _extract_partial_text({"choices": [{}]}) == ""

    def test_none_response(self) -> None:
        assert _extract_partial_text(None) == ""


# ── Core contract ──────────────────────────────────────────────────


class TestCheckCompletionTruncation:

    def setup_method(self) -> None:
        # Reset the per-task flag so tests don't leak state.
        from app.llm_completion_guard import _last_truncated
        _last_truncated.set(False)

    def test_stop_does_not_raise(self) -> None:
        """Natural end of generation must pass through unchanged."""
        resp = {"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]}
        # No exception.
        check_completion_truncation(resp, {})
        assert was_last_completion_truncated() is False

    def test_tool_calls_does_not_raise(self) -> None:
        """Function/tool-call paths are not truncations."""
        resp = {"choices": [{"finish_reason": "tool_calls"}]}
        check_completion_truncation(resp, {})
        assert was_last_completion_truncated() is False

    def test_content_filter_does_not_raise(self) -> None:
        """Moderation trips are a separate concern — we don't conflate."""
        resp = {"choices": [{"finish_reason": "content_filter"}]}
        check_completion_truncation(resp, {})
        assert was_last_completion_truncated() is False

    def test_length_raises(self) -> None:
        resp = {
            "choices": [{
                "finish_reason": "length",
                "message": {"content": "def foo(\n    bar: int,\n    baz: str,"},
            }],
        }
        with pytest.raises(CompletionTruncated) as excinfo:
            check_completion_truncation(resp, {"model": "gpt-4o", "max_tokens": 4096})
        # Exception must carry diagnostic data.
        exc = excinfo.value
        assert exc.model == "gpt-4o"
        assert exc.max_tokens == 4096
        assert exc.finish_reason == "length"
        assert "def foo(" in exc.partial_text
        # Flag set as side effect.
        assert was_last_completion_truncated() is True

    def test_max_tokens_alias_raises(self) -> None:
        """Some providers report ``max_tokens`` instead of ``length``.
        Both must trigger the guard."""
        resp = {"choices": [{"finish_reason": "max_tokens"}]}
        with pytest.raises(CompletionTruncated):
            check_completion_truncation(resp, {})

    def test_no_signal_does_not_raise(self) -> None:
        """When the SDK changes shape and finish_reason is missing,
        we must NOT convert that into an outage."""
        resp = {"choices": [{}]}  # no finish_reason
        check_completion_truncation(resp, {})
        assert was_last_completion_truncated() is False

    def test_raise_on_truncation_false_just_sets_flag(self) -> None:
        """Read-only inspection mode for observability paths."""
        resp = {"choices": [{"finish_reason": "length"}]}
        # No exception when raise=False
        check_completion_truncation(resp, {}, raise_on_truncation=False)
        # But the flag IS set so observers can read it.
        assert was_last_completion_truncated() is True

    def test_flag_resets_each_call(self) -> None:
        """Stale "last call was truncated" state must NOT leak."""
        # First call: truncated.
        check_completion_truncation(
            {"choices": [{"finish_reason": "length"}]},
            {}, raise_on_truncation=False,
        )
        assert was_last_completion_truncated() is True
        # Second call: clean stop. Flag must reset.
        check_completion_truncation(
            {"choices": [{"finish_reason": "stop"}]}, {},
        )
        assert was_last_completion_truncated() is False


# ── Wired into rate_throttle ──────────────────────────────────────


class TestWiredIntoThrottledCompletion:
    """The guard is useless if the wrapper doesn't call it.
    Source-grep contract: both sync and async paths must invoke
    ``check_completion_truncation`` after a successful response."""

    def test_sync_path_calls_check(self) -> None:
        from pathlib import Path
        src = (
            Path(__file__).resolve().parent.parent
            / "app" / "rate_throttle.py"
        ).read_text(encoding="utf-8")
        # Find the sync wrapper body.
        idx_sync = src.find("def _throttled_completion(")
        idx_acomp = src.find("def _throttled_acompletion(")
        assert 0 <= idx_sync < idx_acomp, "wrappers should be in this order"
        sync_body = src[idx_sync:idx_acomp]
        assert "check_completion_truncation" in sync_body, (
            "_throttled_completion must invoke check_completion_truncation "
            "after a successful response"
        )

    def test_async_path_calls_check(self) -> None:
        from pathlib import Path
        src = (
            Path(__file__).resolve().parent.parent
            / "app" / "rate_throttle.py"
        ).read_text(encoding="utf-8")
        idx_acomp = src.find("def _throttled_acompletion(")
        assert idx_acomp > 0
        # Slice from the async wrapper to the next top-level statement
        # (the ``litellm.acompletion = ...`` assignment).
        end = src.find("litellm.acompletion =", idx_acomp)
        async_body = src[idx_acomp:end]
        assert "check_completion_truncation" in async_body, (
            "_throttled_acompletion must invoke check_completion_truncation"
        )


# ── Logging behavior ──────────────────────────────────────────────


class TestLogsAtWarn:
    """The guard logs at WARNING so the truncation lands in
    errors.jsonl (the structured-error handler is at WARN+).
    This is the audit trail for "did this task ever truncate?"."""

    def test_truncation_logs_at_warning(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="app.llm_completion_guard"):
            with pytest.raises(CompletionTruncated):
                check_completion_truncation(
                    {"choices": [{"finish_reason": "length"}]},
                    {"model": "claude-sonnet-4-6", "max_tokens": 8192},
                )
        warn_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warn_records) == 1
        assert "truncated by max_tokens" in warn_records[0].message
        assert "claude-sonnet-4-6" in warn_records[0].message
        assert "8192" in warn_records[0].message
