"""
credit_aware_anthropic.py — a subclass of CrewAI's native Anthropic LLM
provider that handles "Your credit balance is too low" failures cleanly.

Motivation
----------
When the Anthropic API returns::

    400 invalid_request_error
        "Your credit balance is too low to access the Anthropic API.
         Please go to Plans & Billing to upgrade or purchase credits."

the generic `circuit_breaker["anthropic"]` breaker (tuned for transient
glitches: threshold 8, 45s cooldown) is wrong — credit exhaustion is
authoritative on the first occurrence and takes the operator's action
to resolve.  A dedicated `circuit_breaker["anthropic_credits"]` breaker
(threshold 1, 3600s cooldown) represents that semantic precisely.

Design
------
`CreditAwareAnthropicCompletion` is a *proper* subclass of
`crewai.llms.providers.anthropic.completion.AnthropicCompletion` — it
passes every interface-level check Pydantic runs against an
``Agent.llm: str | BaseLLM`` field (which a wrapper class doesn't — that
was the previous attempt's bug: a wrapper class fails Pydantic
validation with "Input should be a valid string").

Behaviour is a single, standard override of ``call()``:

  1. If an OpenRouter-backed fallback has already been built on this
     instance, every subsequent call goes straight to it.
  2. Otherwise: delegate to the parent ``AnthropicCompletion.call()``,
     wrapped in a wall-clock timeout (``CREDIT_AWARE_CALL_TIMEOUT_SECS``,
     default 180s).
  3. On a 400 whose message matches the credit-exhausted signature OR
     a timeout firing:
       - Trip ``circuit_breaker["anthropic_credits"]`` (process-wide
         authoritative state; all other LLM factories read this).
       - Build the fallback LLM via the injected factory and retry the
         same call through it transparently.
  4. All other exceptions propagate unchanged — we don't catch generic
     400s, rate-limits, or network errors, those belong to the existing
     circuit breaker and retry layers.

The timeout (added 2026-04-30 after a 28-min PIM stall during a
credit-exhausted window) is the second-most-important guardrail: even
if the credit-exhausted detection misses (e.g. Anthropic returns a
different shape one day), an unresponsive direct call will fail fast
and route through OpenRouter rather than block the orchestrator until
its 15-min soft-timeout fires with zero output.

Thread safety
-------------
``_fallback_build_lock`` serialises the one-shot failover build so two
concurrent calls that both see the 400 don't race to construct the
fallback LLM twice.

No monkey-patching.  No global mutable flags.  All state is
instance-scoped (the fallback LLM) or circuit-breaker-scoped (the
authoritative "credits exhausted" boolean).
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import threading
from typing import Any, Callable, Optional

from pydantic import PrivateAttr

from crewai.llms.providers.anthropic.completion import AnthropicCompletion

logger = logging.getLogger(__name__)


# Phrases whose presence in an Anthropic 400 body proves the account is
# out of credits rather than any other class of BadRequest.  We match
# case-insensitively and require both substrings AND-together to avoid
# false positives on unrelated "too low" copy (e.g. temperature error).
_CREDIT_EXHAUSTED_MARKERS = ("credit balance", "too low")


def is_credit_exhausted_error(exc: BaseException) -> bool:
    """Return True iff the exception represents the Anthropic 400
    'Your credit balance is too low' response.

    Exposed as a module-level helper so the factory layer can reuse
    the same detection when deciding whether to trip the breaker from
    outside a CreditAware LLM (e.g. if some other code path catches
    the exception first).
    """
    msg = str(exc).lower()
    return all(marker in msg for marker in _CREDIT_EXHAUSTED_MARKERS)


# ── Per-call wall-clock guard ────────────────────────────────────────
#
# An Anthropic completion that hangs without ever emitting a response
# is the worst outage shape: the task pinned a thread, no telemetry
# fires, the orchestrator's soft-timeout silently kills it 15 minutes
# later. This timeout caps ``super().call()`` so the credit-aware
# failover path can fire even when Anthropic returns no response at
# all (vs. a clean 400).
#
# Default 180s gives ~80% margin over the longest legitimate completion
# (max-output Sonnet ≈ 100s) while staying well under the 15-min
# orchestrator soft cap. Operators can tune via env if they routinely
# hit longer completions. Set 0 to disable (disabling reverts to the
# pre-2026-04-30 behaviour and is NOT recommended).

_DEFAULT_CALL_TIMEOUT_SECS = 180.0
_TIMEOUT_MARKER = "anthropic-call-timeout"


def _resolve_call_timeout() -> float:
    """Read ``CREDIT_AWARE_CALL_TIMEOUT_SECS`` from env, falling back
    to the safe default. Re-resolved per call so live env mutations
    take effect without restart.
    """
    raw = os.getenv("CREDIT_AWARE_CALL_TIMEOUT_SECS", "").strip()
    if not raw:
        return _DEFAULT_CALL_TIMEOUT_SECS
    try:
        val = float(raw)
        # 0 or negative disables the timeout entirely (escape hatch);
        # otherwise floor at 5s so a misconfiguration can't make every
        # call fail instantly.
        if val <= 0:
            return 0.0
        return max(5.0, val)
    except ValueError:
        return _DEFAULT_CALL_TIMEOUT_SECS


def is_anthropic_timeout(exc: BaseException) -> bool:
    """Return True iff the exception represents the per-call timeout
    we synthesise above (sync or async path)."""
    return _TIMEOUT_MARKER in str(exc)


# Single shared executor for sync-path timeouts.  Daemon-threaded so
# Python shutdown isn't blocked by a stuck Anthropic call. max_workers
# is generous (32) because individual Anthropic completions can run
# for a minute+ and we don't want unrelated calls to queue.
_TIMEOUT_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=32,
    thread_name_prefix="anth-timeout",
)


class _AnthropicCallTimeout(TimeoutError):
    """Raised when a direct Anthropic call exceeds the wall-clock cap.

    Subclasses :class:`TimeoutError` so any external retry logic that
    treats timeouts specially still works; the marker substring lets
    :func:`is_anthropic_timeout` recognise it for failover routing.
    """

    def __init__(self, secs: float):
        super().__init__(
            f"{_TIMEOUT_MARKER}: anthropic direct call exceeded {secs:.0f}s "
            "wall-clock — failing over to OpenRouter"
        )


class CreditAwareAnthropicCompletion(AnthropicCompletion):
    """Anthropic-direct LLM that fails over to a caller-supplied
    OpenRouter equivalent on credit-exhausted 400.

    The fallback factory must be injected *after* construction via
    :meth:`set_fallback_factory`.  (We avoid declaring it as a
    constructor argument because CrewAI's ``AnthropicCompletion.__init__``
    has a rigid signature and injecting non-standard kwargs there is
    fragile across library upgrades.)

    The fallback is built lazily on first credit-exhausted error.  Once
    built, all subsequent ``call()`` invocations on this instance skip
    the direct Anthropic path entirely.
    """

    # Pydantic private attrs — not part of the public schema, not
    # validated, but supported by `model_config.arbitrary_types_allowed`
    # inherited from the parent.
    _fallback_factory: Optional[Callable[[], Any]] = PrivateAttr(default=None)
    _fallback_llm: Optional[Any] = PrivateAttr(default=None)
    _fallback_build_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def set_fallback_factory(
        self, factory: Callable[[], Any]
    ) -> "CreditAwareAnthropicCompletion":
        """Attach the factory that builds an OpenRouter-backed
        equivalent on credit exhaustion.  Returns self for chaining so
        the factory call site stays a single expression.
        """
        self._fallback_factory = factory
        return self

    # ── Overrides ───────────────────────────────────────────────────
    #
    # Routing rule on every call (not just "first miss"):
    #   * anthropic_credits breaker OPEN  → skip direct Anthropic, use
    #     the already-built fallback LLM.  The breaker transitions
    #     OPEN→HALF_OPEN after its cooldown, at which point the next
    #     call probes direct Anthropic again.
    #   * breaker CLOSED or HALF_OPEN     → try direct Anthropic.  On a
    #     credit-exhausted 400, trip the breaker, lazily build the
    #     fallback (one-shot), and retry through it.
    #
    # Checking the breaker per-call — rather than caching a sticky
    # "failed over" flag on the instance — is what lets
    # ``_cached_llm`` cache instances of this class safely: a cached
    # instance always observes current breaker state, so auto-recovery
    # after credits are restored works without invalidating caches.

    def call(self, *args, **kwargs):
        from app import circuit_breaker
        if not circuit_breaker.is_available("anthropic_credits"):
            # Breaker already open — use fallback directly, no direct probe.
            fallback = self._ensure_fallback()
            if fallback is not None:
                return fallback.call(*args, **kwargs)
            # No fallback configured: try direct and let the error speak.

        try:
            return self._call_with_timeout(*args, **kwargs)
        except Exception as exc:
            # Both credit-exhausted 400s AND wall-clock timeouts route
            # through the same failover path. The timeout case was added
            # 2026-04-30 after a 28-min PIM stall where a hung Anthropic
            # call blocked the agent until the orchestrator gave up.
            if not (is_credit_exhausted_error(exc) or is_anthropic_timeout(exc)):
                raise
            circuit_breaker.record_failure("anthropic_credits")
            fallback = self._ensure_fallback()
            if fallback is None:
                raise
            cause = "credit-exhausted 400" if is_credit_exhausted_error(exc) else "wall-clock timeout"
            # INFO not WARN: failover IS the by-design behavior of
            # this layer.  The original credit-exhausted 400 already
            # tripped the breaker (which logs once at OPEN); the
            # downstream call goes to OpenRouter and recovers.  No
            # operator action is needed beyond the breaker alert.
            logger.info(
                "CreditAwareAnthropicCompletion: %s from Anthropic — "
                "failing over mid-call to OpenRouter Claude.", cause,
            )
            return fallback.call(*args, **kwargs)

    async def acall(self, *args, **kwargs):
        from app import circuit_breaker
        if not circuit_breaker.is_available("anthropic_credits"):
            fallback = self._ensure_fallback()
            if fallback is not None:
                return await fallback.acall(*args, **kwargs)

        try:
            return await self._acall_with_timeout(*args, **kwargs)
        except Exception as exc:
            if not (is_credit_exhausted_error(exc) or is_anthropic_timeout(exc)):
                raise
            circuit_breaker.record_failure("anthropic_credits")
            fallback = self._ensure_fallback()
            if fallback is None:
                raise
            cause = "credit-exhausted 400" if is_credit_exhausted_error(exc) else "wall-clock timeout"
            # INFO not WARN: see sync-path note above — failover is the
            # designed behavior; the breaker logs once at OPEN.
            logger.info(
                "CreditAwareAnthropicCompletion: %s from Anthropic (async) — "
                "failing over mid-call to OpenRouter Claude.", cause,
            )
            return await fallback.acall(*args, **kwargs)

    # ── Timeout wrappers ────────────────────────────────────────────
    #
    # Sync path: submit ``super().call`` to a daemon-thread executor
    # and wait with a wall-clock timeout. If the future doesn't return
    # in time, we surface :class:`_AnthropicCallTimeout` so the failover
    # path triggers. The hung thread keeps running in the background
    # (Python can't kill threads) but the caller gets control back —
    # which is what allows the orchestrator to deliver an answer via
    # OpenRouter instead of pinning a thread for 15+ minutes with zero
    # progress.
    #
    # Async path: ``asyncio.wait_for`` cancels the underlying coroutine
    # cleanly, which is the right behaviour because async cancellation
    # actually propagates.

    def _call_with_timeout(self, *args, **kwargs):
        timeout = _resolve_call_timeout()
        if timeout <= 0:
            return super().call(*args, **kwargs)
        future = _TIMEOUT_EXECUTOR.submit(super().call, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()  # best-effort; the thread keeps running
            raise _AnthropicCallTimeout(timeout)

    async def _acall_with_timeout(self, *args, **kwargs):
        timeout = _resolve_call_timeout()
        if timeout <= 0:
            return await super().acall(*args, **kwargs)
        try:
            return await asyncio.wait_for(
                super().acall(*args, **kwargs), timeout=timeout,
            )
        except asyncio.TimeoutError:
            raise _AnthropicCallTimeout(timeout)

    # ── Internals ───────────────────────────────────────────────────

    def _ensure_fallback(self):
        """Build the OR fallback under a lock (one-shot, thread-safe).

        Breaker tripping is the caller's responsibility — this method
        only builds the LLM.  Keeping breaker manipulation at the call
        site prevents the cooldown timer from being reset when an
        already-open breaker is encountered by a fresh cached instance.
        """
        if self._fallback_llm is not None:
            return self._fallback_llm
        with self._fallback_build_lock:
            if self._fallback_llm is not None:
                return self._fallback_llm
            if self._fallback_factory is None:
                return None
            self._fallback_llm = self._fallback_factory()
            return self._fallback_llm
