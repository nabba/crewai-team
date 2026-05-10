"""LLM completion-quality guard — translate SDK signals into our domain.

The litellm / OpenAI / Anthropic SDKs surface a ``finish_reason`` on
every completion response that tells us *why* the generation stopped:

  ``stop``           — the model finished naturally
  ``tool_calls`` / ``function_call`` — the model emitted a tool/function call
  ``content_filter`` — moderation system intervened
  ``length``         — ``max_tokens`` budget hit MID-OUTPUT (truncation)

Pre-2026-05-10 the codebase ignored ``finish_reason`` entirely
(``grep -rn finish_reason app/`` returned zero matches). A
``length`` truncation produced a partial-but-syntactically-broken
response that the orchestrator treated as success and only the
downstream LLM-vetter happened to catch. That's the failure mode
behind the operator-reported "30-min stall + please re-send a
narrower question" symptom — a script truncated mid-function had
no signal, no retry, no recovery.

This module gives the system one job: **never silently accept a
``finish_reason == "length"`` completion**. The check runs at the
single point all LLM calls funnel through (``rate_throttle._throttled_completion``)
and raises a typed ``CompletionTruncated`` so callers can distinguish
truncation from network errors / provider failures / moderation
trips.

What this module does NOT do:
  * **No auto-continuation.** Continuation across token-budget
    boundaries is brittle for code generation — models re-emit
    preamble or change indentation. The right cure is "raise a
    typed signal, let the caller bump max_tokens and re-run with
    clean state". Auto-continuation can be layered on top later
    as an explicit caller policy.
  * **No retries.** The orchestrator's existing retry path
    (``_build_retry_task`` + ``vet_response_detailed``) handles
    failure surfaces uniformly. Adding a second retry layer
    here would produce competing recovery loops.

Usage from inside ``rate_throttle._throttled_completion``::

    response = _original_completion(*args, **kwargs)
    check_completion_truncation(response, kwargs)  # raises on length

Inspection from elsewhere (read-only thread-local flag — set by
``check_completion_truncation`` even when not raising)::

    from app.llm_completion_guard import was_last_completion_truncated
    if was_last_completion_truncated():
        ...
"""
from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Any

logger = logging.getLogger(__name__)


# Per-task / per-thread flag — set by ``check_completion_truncation``
# when it sees a truncation signal, even when the caller has opted
# out of raising. Useful for read-only inspection from logging /
# metrics layers without changing control flow.
_last_truncated: ContextVar[bool] = ContextVar(
    "llm_last_completion_truncated", default=False,
)


# Finish reasons that mean "incomplete output" — the model wanted to
# say more but couldn't. Distinct from natural-end ("stop") and
# tool-call paths.
_BAD_FINISH_REASONS = frozenset(["length", "max_tokens"])


class CompletionTruncated(Exception):
    """The LLM stopped mid-output because ``max_tokens`` was hit.

    Carries the partial text + the original kwargs so callers can
    decide whether to retry with a higher budget, switch model,
    surface the failure, or attempt a continuation.

    Attributes:
        partial_text: the text the model managed to emit before the
            cutoff. May be empty.
        model: the model id reported by the response (or kwargs).
        max_tokens: the budget that was hit.
        finish_reason: the literal ``finish_reason`` value, in case
            we ever expand the trigger set (see ``_BAD_FINISH_REASONS``).
    """

    def __init__(
        self,
        partial_text: str = "",
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        finish_reason: str = "length",
    ) -> None:
        self.partial_text = partial_text
        self.model = model
        self.max_tokens = max_tokens
        self.finish_reason = finish_reason
        msg = (
            f"LLM completion truncated by max_tokens budget "
            f"(model={model or '?'}, max_tokens={max_tokens or '?'}, "
            f"finish_reason={finish_reason!r}, partial_chars={len(partial_text)})"
        )
        super().__init__(msg)


def _extract_finish_reason(response: Any) -> str | None:
    """Robust ``finish_reason`` extractor.

    Handles dict-shaped responses (older SDK versions, JSON-decoded),
    object-shaped responses (modern litellm), and degenerate inputs.
    Returns None if no finish_reason is found — callers treat None as
    "no signal, assume normal completion" (we never raise on missing
    data; that would convert SDK changes into outages).
    """
    if response is None:
        return None
    try:
        choices = (
            response.get("choices") if isinstance(response, dict)
            else getattr(response, "choices", None)
        )
    except Exception:
        return None
    if not choices:
        return None
    first = choices[0]
    try:
        if isinstance(first, dict):
            return first.get("finish_reason")
        return getattr(first, "finish_reason", None)
    except Exception:
        return None


def _extract_partial_text(response: Any) -> str:
    """Best-effort extractor for the partial assistant text. Never raises."""
    if response is None:
        return ""
    try:
        choices = (
            response.get("choices") if isinstance(response, dict)
            else getattr(response, "choices", None)
        )
    except Exception:
        return ""
    if not choices:
        return ""
    first = choices[0]
    try:
        # Modern shape: choices[0].message.content
        message = (
            first.get("message") if isinstance(first, dict)
            else getattr(first, "message", None)
        )
        if message is not None:
            content = (
                message.get("content") if isinstance(message, dict)
                else getattr(message, "content", None)
            )
            if isinstance(content, str):
                return content
        # Fallback: choices[0].text (older completion shape)
        text = (
            first.get("text") if isinstance(first, dict)
            else getattr(first, "text", None)
        )
        if isinstance(text, str):
            return text
    except Exception:
        return ""
    return ""


def was_last_completion_truncated() -> bool:
    """Read-only inspection of the per-task truncation flag.

    Set to True by ``check_completion_truncation`` whenever it detects
    a ``length`` finish_reason — including when the caller opts out
    of raising. Logging / metrics / audit layers can read this without
    affecting control flow. The flag is per-ContextVar (thread- and
    asyncio-safe) so concurrent requests don't trample each other.
    """
    return bool(_last_truncated.get())


def check_completion_truncation(
    response: Any,
    request_kwargs: dict | None = None,
    *,
    raise_on_truncation: bool = True,
) -> None:
    """Inspect ``response.choices[0].finish_reason``; raise on ``length``.

    Args:
        response: a litellm / OpenAI-shaped completion response.
        request_kwargs: the kwargs the call was made with — used to
            populate ``model`` and ``max_tokens`` on the exception
            for diagnostic clarity.
        raise_on_truncation: when False, just sets the per-task flag
            (visible via ``was_last_completion_truncated()``) and logs
            at WARN. Useful for read-only observability paths that
            shouldn't change control flow.

    Raises:
        CompletionTruncated: when ``finish_reason`` indicates mid-
            output cutoff AND ``raise_on_truncation`` is True.

    Reset semantics: this also clears the per-task flag at the start
    of every call, so ``was_last_completion_truncated()`` always
    reflects the most recent completion.
    """
    # Reset the per-task flag every call so stale "last call was
    # truncated" state from a prior request can't leak.
    _last_truncated.set(False)

    finish_reason = _extract_finish_reason(response)
    if finish_reason is None:
        return  # no signal — treat as normal
    if finish_reason not in _BAD_FINISH_REASONS:
        return  # natural end / tool call / moderation — not our concern

    # Truncation detected.
    _last_truncated.set(True)
    request_kwargs = request_kwargs or {}
    partial = _extract_partial_text(response)
    model = request_kwargs.get("model")
    max_tokens = request_kwargs.get("max_tokens")

    logger.warning(
        "llm_completion_guard: truncated by max_tokens "
        "(model=%s, max_tokens=%s, finish_reason=%s, partial_chars=%d)",
        model, max_tokens, finish_reason, len(partial),
    )

    # Cure C (2026-05-10) — stash on the task tracker so the
    # watchdog's apology can name the specific failure cause
    # ("response was truncated mid-output by max_tokens=4096")
    # instead of generic "narrow your question". Best-effort —
    # never let the tracker import break the LLM-call boundary.
    try:
        from app.observability.task_progress import record_failure_context
        record_failure_context(
            "completion_truncated",
            f"model={model or '?'} max_tokens={max_tokens or '?'} "
            f"partial_chars={len(partial)}",
        )
    except Exception:
        pass

    if raise_on_truncation:
        raise CompletionTruncated(
            partial_text=partial,
            model=model,
            max_tokens=max_tokens,
            finish_reason=finish_reason,
        )


__all__ = [
    "CompletionTruncated",
    "check_completion_truncation",
    "was_last_completion_truncated",
]
