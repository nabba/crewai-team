"""
rate_throttle.py — Per-provider rate limiter for outgoing LLM API calls.

Each LLM provider gets its own rate bucket:
  - Anthropic:   conservative (default 10 RPM, configurable)
  - OpenRouter:  generous (default 60 RPM)
  - Ollama:      unlimited (local, no API limit)

User-facing requests get priority over background tasks.  Background
callers should call `set_background_caller(True)` before making LLM
calls; they will yield to any waiting user-facing request.

Also configures litellm's built-in retry with exponential backoff
so transient 429s are retried automatically.
"""

import contextvars
import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
# Per-provider RPM limits (overridable via .env)
_ANTHROPIC_RPM = int(os.environ.get("ANTHROPIC_MAX_RPM", "10"))
_OPENROUTER_RPM = int(os.environ.get("OPENROUTER_MAX_RPM", "60"))
_RETRY_COUNT = int(os.environ.get("LITELLM_NUM_RETRIES", "5"))
_RETRY_BACKOFF = float(os.environ.get("LITELLM_RETRY_BACKOFF", "3"))  # seconds (was 15)

# ── litellm retry config (set before any litellm import) ──────────────────────
os.environ.setdefault("LITELLM_NUM_RETRIES", str(_RETRY_COUNT))

# ── Token bucket rate limiter ─────────────────────────────────────────────────

class _TokenBucket:
    """Thread-safe token bucket allowing at most `rate` calls per 60 seconds.

    Uses threading.Event for non-blocking wait that can be interrupted.
    """

    def __init__(self, rate: int, name: str = ""):
        self.rate = max(1, rate)
        self.interval = 60.0 / self.rate  # seconds between tokens
        self.name = name
        self._lock = threading.Lock()
        self._last = 0.0
        self._wake = threading.Event()

    def acquire(self) -> None:
        """Block until a token is available.

        Releases the lock while waiting so other threads can compute their
        own wait time.  Re-acquires the lock to stamp _last on completion.
        """
        while True:
            with self._lock:
                now = time.monotonic()
                wait = self._last + self.interval - now
                if wait <= 0:
                    # Token available — claim it immediately
                    self._last = now
                    return
                if wait > 1.0:
                    logger.debug(f"rate_throttle[{self.name}]: waiting {wait:.1f}s")
            # Sleep outside the lock so other threads aren't blocked
            self._wake.clear()
            self._wake.wait(timeout=wait)
            # Loop back to re-check under lock (another thread may have consumed the slot)


# Per-provider buckets
_buckets: dict[str, _TokenBucket] = {
    "anthropic": _TokenBucket(_ANTHROPIC_RPM, "anthropic"),
    "openrouter": _TokenBucket(_OPENROUTER_RPM, "openrouter"),
    # Ollama: no bucket — unlimited local calls
}

# ── Background caller tracking ────────────────────────────────────────────────
_is_background = threading.local()


def set_background_caller(is_bg: bool) -> None:
    """Mark the current thread as a background caller (lower priority)."""
    _is_background.value = is_bg


def _is_bg() -> bool:
    return getattr(_is_background, "value", False)


def _detect_provider(model: str = "", base_url: str = "", **kwargs) -> str:
    """Detect LLM provider from model string or base_url."""
    model_lower = (model or "").lower()
    base_lower = (base_url or "").lower()

    if "ollama" in model_lower or "ollama" in base_lower or "11434" in base_lower:
        return "ollama"
    if "openrouter" in base_lower:
        return "openrouter"
    if "anthropic" in model_lower or "claude" in model_lower:
        return "anthropic"
    # OpenRouter model IDs contain slashes like "deepseek/deepseek-chat"
    if "/" in model_lower and "anthropic" not in model_lower:
        return "openrouter"
    return "anthropic"  # default to most restrictive


def throttle_for_provider(provider: str) -> None:
    """Apply rate limit for a specific provider. No-op for Ollama."""
    if provider == "ollama":
        return
    bucket = _buckets.get(provider)
    if bucket:
        # Background callers yield briefly to let user-facing requests go first
        if _is_bg():
            time.sleep(0.1)
        bucket.acquire()


def throttle() -> None:
    """Legacy: throttle using Anthropic bucket (for unknown callers)."""
    throttle_for_provider("anthropic")


# ── Monkey-patch litellm completion to inject throttle ────────────────────────

_patched = False
_patch_lock = threading.Lock()


def install_throttle() -> None:
    """
    Patch litellm.completion to call per-provider throttle before each request.
    Safe to call multiple times (idempotent).
    """
    global _patched
    if _patched:
        return
    with _patch_lock:
        if _patched:
            return
        try:
            import litellm
            _original_completion = litellm.completion

            def _throttled_completion(*args, **kwargs):
                model = kwargs.get("model", args[0] if args else "")
                base_url = kwargs.get("base_url") or kwargs.get("api_base") or ""
                provider = _detect_provider(model, base_url)
                throttle_for_provider(provider)
                # Inject retry params if not already set
                kwargs.setdefault("num_retries", _RETRY_COUNT)
                # Fresh guard per call — success_callback, explicit inline
                # record, or failure branch share the first-writer-wins rule.
                _benchmark_recorded.set(False)
                t_start = time.monotonic()
                try:
                    response = _original_completion(*args, **kwargs)
                except Exception as exc:
                    latency_ms = int((time.monotonic() - t_start) * 1000)
                    _record_benchmark_failure(model, latency_ms)
                    _check_credit_error(exc, provider)
                    _capture_upstream_error_body(exc, model, provider, kwargs)
                    _record_model_reliability(model, exc, success=False)
                    # Credit-failover: on 402 / "insufficient credits",
                    # retry ONCE against a local Ollama model before
                    # propagating the error. Keeps the system usable
                    # when OpenRouter runs out of balance.
                    fallback = _try_credit_failover_sync(
                        exc, model, kwargs, _original_completion,
                    )
                    if fallback is not None:
                        fb_latency_ms = int((time.monotonic() - t_start) * 1000)
                        _record_token_usage(fallback, kwargs, latency_ms=fb_latency_ms)
                        return fallback
                    raise
                latency_ms = int((time.monotonic() - t_start) * 1000)
                # Successful call — resolve any prior credit alert for this provider
                _resolve_credit_if_needed(provider)
                _record_model_reliability(model, None, success=True)
                _record_token_usage(response, kwargs, latency_ms=latency_ms)
                # Cure A (2026-05-10): translate the SDK's
                # ``finish_reason == "length"`` signal into a typed
                # ``CompletionTruncated`` exception. Pre-fix the
                # codebase ignored ``finish_reason`` entirely and a
                # mid-output truncation looked identical to a clean
                # completion to every layer except the eventual LLM
                # vetter — that's the failure mode behind the
                # operator-reported "30-min stall + please re-send"
                # symptom for the forest-age task.
                from app.llm_completion_guard import check_completion_truncation
                check_completion_truncation(response, kwargs)
                return response

            litellm.completion = _throttled_completion

            # Also patch acompletion for async paths
            if hasattr(litellm, "acompletion"):
                _original_acompletion = litellm.acompletion

                async def _throttled_acompletion(*args, **kwargs):
                    model = kwargs.get("model", args[0] if args else "")
                    base_url = kwargs.get("base_url") or kwargs.get("api_base") or ""
                    provider = _detect_provider(model, base_url)
                    throttle_for_provider(provider)
                    kwargs.setdefault("num_retries", _RETRY_COUNT)
                    _benchmark_recorded.set(False)
                    t_start = time.monotonic()
                    try:
                        response = await _original_acompletion(*args, **kwargs)
                    except Exception as exc:
                        latency_ms = int((time.monotonic() - t_start) * 1000)
                        _record_benchmark_failure(model, latency_ms)
                        _check_credit_error(exc, provider)
                        _capture_upstream_error_body(exc, model, provider, kwargs)
                        _record_model_reliability(model, exc, success=False)
                        fallback = await _try_credit_failover_async(
                            exc, model, kwargs, _original_acompletion,
                        )
                        if fallback is not None:
                            fb_latency_ms = int((time.monotonic() - t_start) * 1000)
                            _record_token_usage(
                                fallback, kwargs, latency_ms=fb_latency_ms,
                            )
                            return fallback
                        raise
                    latency_ms = int((time.monotonic() - t_start) * 1000)
                    _record_model_reliability(model, None, success=True)
                    _record_token_usage(response, kwargs, latency_ms=latency_ms)
                    # Cure A — same truncation guard as the sync path.
                    from app.llm_completion_guard import check_completion_truncation
                    check_completion_truncation(response, kwargs)
                    return response

                litellm.acompletion = _throttled_acompletion

            _patched = True
            logger.info(
                f"rate_throttle: installed (anthropic={_ANTHROPIC_RPM}RPM, "
                f"openrouter={_OPENROUTER_RPM}RPM, ollama=unlimited, "
                f"{_RETRY_COUNT} retries, {_RETRY_BACKOFF}s backoff)"
            )
        except ImportError:
            logger.warning("rate_throttle: litellm not found, throttle not installed")

        # NOTE: we used to patch ``BaseLLM._track_token_usage_internal``
        # here to cover CrewAI-native providers (Anthropic, Gemini,
        # Azure, Bedrock) that bypass litellm's callback machinery.
        # That responsibility moved to the CrewAI event-bus subscriber
        # in ``app.observability.llm_events`` — one hook on
        # ``LLMCallCompletedEvent`` covers every provider uniformly,
        # including any CrewAI ships in the future.

        # ── OpenAI SDK credit-failover patch (2026-04-28) ────────────
        # CrewAI 1.14.x ships a "providers" system whose openai branch
        # (crewai/llms/providers/openai/completion.py) calls
        # openai.OpenAI directly — bypassing litellm.completion entirely
        # and therefore bypassing the credit-failover patch above.
        # Result: when OpenRouter returns 402 "insufficient credits",
        # the openai SDK raises APIStatusError, the orchestrator
        # propagates it up, and the user sees "Crew pim failed: Error
        # code: 402".
        #
        # Mirror of the litellm patch: wrap chat.completions.create
        # (sync + async) so 402s trigger _try_credit_failover_sync
        # the same way. Idempotent — guarded by an attribute flag on
        # the bound method's class.
        try:
            _install_openai_credit_failover()
        except Exception as exc:
            logger.warning(
                "rate_throttle: failed to install openai credit-failover patch "
                "(%s) — credit failover only covers litellm calls",
                exc,
            )

    # Litellm success_callback is still useful for the observability
    # concerns that the event payload can't represent: measured
    # ``latency_ms`` for benchmark scoring and the raw response shape
    # for training-data capture.  See ``_record_token_usage``.
    try:
        import litellm
        litellm.success_callback = [_record_token_usage]
        logger.info("rate_throttle: litellm success_callback registered (scoped to latency-aware benchmark scoring + training capture)")
    except Exception:
        logger.debug("rate_throttle: could not register litellm callback", exc_info=True)


_cost_lookup: dict[str, tuple[float, float]] | None = None


def _get_cost_lookup() -> dict[str, tuple[float, float]]:
    """Lazily build model→(cost_input, cost_output) dict from CATALOG. O(1) per lookup.

    Maps multiple key variants for each model so we match regardless of how
    litellm reports the model name in the response:
      - catalog name:     "deepseek-v3.2"
      - model_id:         "openrouter/deepseek/deepseek-chat"
      - stripped of prefix: "deepseek/deepseek-chat"   (litellm often strips "openrouter/")
      - bare model:       "deepseek-chat"              (sometimes just the last segment)
      - anthropic:        "claude-opus-4-6"            (litellm strips "anthropic/")
    """
    global _cost_lookup
    if _cost_lookup is not None:
        return _cost_lookup
    try:
        from app.llm_catalog import CATALOG
        lookup: dict[str, tuple[float, float]] = {}
        for name, info in CATALOG.items():
            costs = (info.get("cost_input_per_m", 0), info.get("cost_output_per_m", 0))
            lookup[name] = costs
            model_id = info.get("model_id", "")
            if model_id and model_id != name:
                lookup[model_id] = costs
                # Strip provider prefix: "openrouter/deepseek/deepseek-chat" → "deepseek/deepseek-chat"
                if model_id.startswith("openrouter/"):
                    stripped = model_id[len("openrouter/"):]
                    lookup[stripped] = costs
                # Strip "anthropic/" prefix: "anthropic/claude-opus-4-6" → "claude-opus-4-6"
                if model_id.startswith("anthropic/"):
                    stripped = model_id[len("anthropic/"):]
                    lookup[stripped] = costs
                # Strip "ollama_chat/" prefix
                if model_id.startswith("ollama_chat/"):
                    stripped = model_id[len("ollama_chat/"):]
                    lookup[stripped] = costs
        _cost_lookup = lookup
    except Exception:
        _cost_lookup = {}
    return _cost_lookup


def _find_cost(model: str) -> tuple[float, float] | None:
    """Look up cost for a model, trying exact match then prefix match."""
    lookup = _get_cost_lookup()
    # Exact match
    hit = lookup.get(model)
    if hit:
        return hit
    # Try stripping version suffixes: "deepseek/deepseek-chat-v3" → "deepseek/deepseek-chat"
    # Common pattern: litellm appends version info not in our catalog
    import re
    base = re.sub(r"-v\d+(\.\d+)*$", "", model)
    if base != model:
        hit = lookup.get(base)
        if hit:
            return hit
    # Prefix match: find any key that starts with the model name or vice versa
    for key, costs in lookup.items():
        if model.startswith(key) or key.startswith(model):
            return costs
    return None


def _record_token_usage(response, kwargs: dict, latency_ms: int = 0) -> None:
    """Scoped observability hooks that need the raw LiteLLM response
    shape (richer than what the CrewAI event payload carries).

    Token + cost accounting and the activity heartbeat are handled by
    the event-bus subscriber in ``app.observability.llm_events`` — that
    path covers every provider uniformly and doesn't need this richer
    payload.

    What remains here:
      - **Benchmark scoring** (success row with ``latency_ms``) — the
        event payload doesn't carry latency, and measuring it here
        gives us an accurate number for the scoring model.
      - **Training-data capture** — needs the full LiteLLM response
        (prompt+completion text, tool_use blocks), not the normalised
        event shape.

    Called both as a litellm ``success_callback`` and inline from the
    throttled completion wrapper.  A ContextVar guard
    (:data:`_benchmark_recorded`) keeps the benchmark row idempotent
    when both callers fire for the same call.
    """
    try:
        usage = getattr(response, "usage", None)
        if not usage:
            return
        model = getattr(response, "model", "") or kwargs.get("model", "unknown")
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total = prompt_tokens + completion_tokens
        if total <= 0:
            return

        # Training pipeline capture (fire-and-forget) — stays here
        # because it consumes the raw response object.
        try:
            _capture_training_data(response, kwargs, model)
        except Exception:
            pass

        # Benchmark scoring — guarded so _throttled_completion's direct
        # invocation and the success_callback don't both write a row for
        # the same call.  First writer wins; guard is cleared when the
        # llm_context scope ends.
        if not _benchmark_recorded.get(False):
            try:
                from app.llm_benchmarks import record
                from app.llm_context import current as _current_ctx
                ctx = _current_ctx()
                task_type = ctx.task_type if ctx else "general"
                record(model, task_type, True,
                       latency_ms=latency_ms, tokens=total)
                _benchmark_recorded.set(True)
            except Exception:
                pass
    except Exception:
        pass  # never fail the actual LLM call


def _record_benchmark_failure(model: str, latency_ms: int) -> None:
    """Record a failed LLM call into the benchmarks table.

    Called from the throttled completion wrapper when the underlying
    provider call raises. Reads the active :class:`~app.llm_context.CallContext`
    for the canonical task type so failures land in the same task-type
    partition that successes use.
    """
    try:
        from app.llm_benchmarks import record
        from app.llm_context import current as _current_ctx
        ctx = _current_ctx()
        task_type = ctx.task_type if ctx else "general"
        record(model, task_type, False, latency_ms=latency_ms, tokens=0)
        _benchmark_recorded.set(True)
    except Exception:
        pass
    # Heartbeat is emitted by the CrewAI event-bus subscriber
    # (LLMCallFailedEvent).  No explicit call needed here.


# ── Credit alert integration ─────────────────────────────────────────────────

_resolved_providers: set[str] = set()  # avoid repeated resolve calls


def _capture_training_data(response, kwargs: dict, model: str) -> None:
    """Extract prompt-completion pair from litellm response for self-training.

    Called from _record_token_usage on every successful LLM call.
    Filters out short/internal responses and deduplicates by content hash.
    Writes to JSONL + PostgreSQL via training_collector.
    """
    import threading

    # Extract completion text
    try:
        choices = getattr(response, "choices", [])
        if not choices:
            return
        completion = getattr(choices[0], "message", None)
        if not completion:
            return
        response_text = getattr(completion, "content", "") or ""
    except Exception:
        return

    # Filter: skip short/empty responses (internal routing, vetting, etc.)
    if len(response_text) < 50:
        return

    # Extract input messages
    messages = kwargs.get("messages", [])
    if not messages:
        return

    # Filter: skip system-only messages (no user content)
    has_user = any(m.get("role") == "user" for m in messages if isinstance(m, dict))
    if not has_user:
        return

    # Build training record
    def _store():
        try:
            from app.training_collector import _content_hash, _classify_model, _store_record
            from app.training_collector import MAX_RESPONSE_LENGTH
            from datetime import datetime, timezone

            stored_messages = [
                {"role": m.get("role", "user"), "content": str(m.get("content", ""))[:2000]}
                for m in messages[-5:] if isinstance(m, dict)
            ]
            stored_response = response_text[:MAX_RESPONSE_LENGTH]
            source_tier, provenance = _classify_model(model)

            record = {
                "id": _content_hash(stored_messages, stored_response),
                "agent_role": kwargs.get("metadata", {}).get("agent_role", "unknown")
                    if isinstance(kwargs.get("metadata"), dict) else "unknown",
                "task_description": str(stored_messages[-1].get("content", ""))[:500]
                    if stored_messages else "",
                "messages": stored_messages,
                "response": stored_response,
                "source_model": model,
                "source_tier": source_tier,
                "provenance": provenance,
                "quality_score": None,
                "training_eligible": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            _store_record(record)
        except Exception:
            pass

    threading.Thread(target=_store, daemon=True, name="training-capture").start()


def _check_credit_error(exc: Exception, provider: str) -> None:
    """If the error looks like a credit/billing issue, report it."""
    try:
        from app.firebase_reporter import detect_credit_error, report_credit_alert
        detected = detect_credit_error(exc)
        if detected:
            report_credit_alert(detected, str(exc)[:300])
            _resolved_providers.discard(detected)
    except Exception:
        pass  # never interfere with the real error


def _resolve_credit_if_needed(provider: str) -> None:
    """On success, resolve any active credit alert for this provider."""
    if provider in _resolved_providers:
        return  # already resolved, skip
    try:
        from app.firebase_reporter import _active_alerts, resolve_credit_alert
        if provider in _active_alerts:
            resolve_credit_alert(provider)
            _resolved_providers.add(provider)
    except Exception:
        pass


# ── Request-level cost tracking ──────────────────────────────────────────────

_request_cost: contextvars.ContextVar["RequestCostTracker | None"] = contextvars.ContextVar(
    "request_cost", default=None,
)

# Per-call guard ensuring exactly one benchmarks row per LLM invocation.
# Set by the first writer (either ``_throttled_completion`` directly, its
# failure branch, or the ``success_callback`` via ``_record_token_usage``)
# and implicitly cleared when the request-level context unwinds — a fresh
# ContextVar read inside the next call returns False.
_benchmark_recorded: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "llm_benchmark_recorded", default=False,
)


# ── LLM activity heartbeat ─────────────────────────────────────────────
#
# Process-wide "last time any LLM call completed (success OR failure)"
# timestamp.  Used by the progressive timeout in ``handle_task`` to
# distinguish a genuinely long-running task (LLM calls still completing,
# tokens still accruing) from a stalled one (no LLM activity for N
# minutes, usually means the orchestrator is stuck in a loop or the
# provider is unreachable).
#
# Thread-safe.  The monotonic clock is used so wall-clock jumps
# (NTP sync, suspend-resume) don't confuse the stall detector.

_llm_activity_lock = threading.Lock()
_last_llm_activity_ts: float = 0.0
_llm_activity_count: int = 0


def record_llm_activity() -> None:
    """Mark that an LLM call just completed (success or failure).

    Called from ``_record_token_usage`` (success path) and
    ``_record_benchmark_failure`` (failure path).  Both paths count as
    "activity" because both prove the orchestrator is cycling through
    LLM interactions — a stalled task is one where NO LLM call has
    returned for an extended period.
    """
    global _last_llm_activity_ts, _llm_activity_count
    with _llm_activity_lock:
        _last_llm_activity_ts = time.monotonic()
        _llm_activity_count += 1


def seconds_since_last_llm_activity() -> float | None:
    """Return seconds since the last LLM call completed, or None if no
    call has completed yet this process.

    Callers use this to implement progress-gated timeouts:
      * None              → never seen activity (bootstrap / cold start)
      * small number      → task alive, LLM calls happening
      * large number (>N) → stalled, safe to abandon
    """
    with _llm_activity_lock:
        if _last_llm_activity_ts == 0.0:
            return None
        return time.monotonic() - _last_llm_activity_ts


def llm_activity_count() -> int:
    """Process-wide count of completed LLM calls.  Useful as an
    alternative to the timestamp when callers want to detect progress
    strictly by "new calls happened since I last looked"."""
    with _llm_activity_lock:
        return _llm_activity_count


class RequestCostTracker:
    """Accumulates token usage across all LLM calls in a single user request."""

    def __init__(self, request_id: str = ""):
        self.request_id = request_id
        self.crew_name = ""  # set by commander before dispatch
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost_usd = 0.0
        self.call_count = 0
        self.models_used: set[str] = set()
        self._lock = threading.Lock()

    def record(self, model: str, prompt_tokens: int, completion_tokens: int, cost_usd: float) -> None:
        with self._lock:
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_cost_usd += cost_usd
            self.call_count += 1
            self.models_used.add(model)

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    def summary(self) -> str:
        models = ", ".join(sorted(self.models_used)) if self.models_used else "none"
        return (
            f"{self.call_count} LLM calls, "
            f"{self.total_tokens:,} tokens, "
            f"${self.total_cost_usd:.4f}, "
            f"models: {models}"
        )


def start_request_tracking(request_id: str = "") -> RequestCostTracker:
    """Begin accumulating costs for a user request. Returns the tracker.

    Nesting-aware: if a tracker is already active in this context (a parent
    caller is already tracking), return the EXISTING tracker instead of
    creating a new one.  This prevents nested crews (media, critic,
    retrospective, etc.) from clobbering the outer Commander-level tracker
    and making the ticket report $0 cost.  The tracker keeps accumulating
    across the nested call so the Commander sees the full cost at the end.
    """
    existing = _request_cost.get(None)
    if existing is not None:
        return existing
    tracker = RequestCostTracker(request_id)
    _request_cost.set(tracker)
    return tracker


def stop_request_tracking() -> RequestCostTracker | None:
    """Return the accumulated tracker.  Does NOT clear the context-var
    unless called by the top of the request stack (via finalize_request_tracking()).

    Nested crews read this to get their contribution totals but should not
    strand the outer Commander's tracker.
    """
    return _request_cost.get(None)


def finalize_request_tracking() -> RequestCostTracker | None:
    """Clear the context-var and return the tracker.

    Use at the TOP of the request stack (Commander.handle) where the
    request lifecycle actually ends.  Nested callers use
    stop_request_tracking() instead.
    """
    tracker = _request_cost.get(None)
    _request_cost.set(None)
    return tracker


def get_active_tracker() -> "RequestCostTracker | None":
    """Get the active request tracker (for propagating to threads)."""
    return _request_cost.get(None)


def set_active_tracker(tracker: "RequestCostTracker | None") -> None:
    """Set the request tracker (for thread propagation)."""
    _request_cost.set(tracker)


# ── Upstream-error diagnostics ─────────────────────────────────────────
#
# LiteLLM wraps the raw provider error in its own Exception subclass
# and the default ``str(exc)`` truncates the JSON body.  When Claude
# Opus 4.7 via OpenRouter rejects a request (the 2026-04-22 PSP trace
# showed "Error code: 400 - {'error': {'message': 'Provider returned
# error', ...'raw': ...}}" with the actual reason buried inside
# ``raw``), we need the full body to diagnose whether it was:
#
#   * context-window overflow (huge conversation history)
#   * malformed tool schema (strict validator rejected a pydantic
#     Field shape)
#   * content-filter hit
#   * upstream 5xx passed through as 400
#
# This helper pulls every field it can reach on the exception and
# dumps them as a structured WARNING log line so the next 400 is
# debuggable from the logs alone.

# ── Credit-exhaustion failover to local Ollama ───────────────────────
#
# When OpenRouter / Anthropic / OpenAI returns a 402 / 429 / insufficient-
# credits error, re-issue the same request against a local Ollama model
# once.  Rationale: a hosted-LLM balance hitting zero should degrade the
# system to "slower but still working" rather than "completely broken" —
# the UX failure we saw with the 2026-04-23 PSP task was that every LLM
# call failed, so even the commander couldn't route, and the user got
# "Sorry, I had trouble understanding".
#
# Design
# ------
# * **One-shot** — we never failover a failover.  A process-wide
#   ContextVar guards against recursion.
# * **Provider-agnostic** — we use :func:`select_model` with
#   ``force_tier='local'`` to pick a role-appropriate local model,
#   falling back to a hard-coded default if the selector can't resolve.
# * **Context-window-aware** — caps ``max_tokens`` at 4096 since local
#   models typically have smaller context budgets than Opus/Sonnet.
# * **No re-throttle** — we call ``_original_completion`` directly
#   (bypassing the wrapper) so the local call doesn't run through the
#   OpenRouter throttle.
# * **Token accounting still works** — on success, the caller records
#   the local-model tokens via ``_record_token_usage`` as if it were a
#   direct call.

# ContextVar is process-wide but unwinds automatically with the asyncio
# task / thread that set it.  True = we're inside a failover retry and
# must not start another one.
_failover_in_progress: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "credit_failover_in_progress", default=False,
)

# Last-resort local model when select_model can't pick one.  Updated
# manually if the local Ollama inventory changes.
_FAILOVER_DEFAULT_LOCAL_MODEL = "ollama/llama3.1:8b"

# Maximum tokens we'll request from a local model.  Most Ollama models
# we ship are 8K-context; 4096 output leaves enough for the prompt.
_FAILOVER_MAX_TOKENS = 4096


def _select_local_failover_model(original_model: str) -> str | None:
    """Pick a local Ollama model to use as the failover target.

    Uses :func:`app.llm_selector.select_model` with ``force_tier='local'``
    when the caller's role can be inferred; otherwise falls back to a
    hard-coded default.  Returns ``None`` when no local model is
    available — caller will propagate the original error.
    """
    # Try the selector first.  Role inference is best-effort: the model
    # string doesn't carry its role back, so we default to 'research'
    # (the most common credit-hungry path).  The selector respects
    # local-tier affinities in role_assignments.
    try:
        from app.llm_selector import select_model
        local_model = select_model(role="research", force_tier="local")
        if local_model and local_model.startswith("ollama/"):
            return local_model
    except Exception as exc:
        logger.debug("failover: select_model(force_tier='local') raised: %s", exc)
    # Fallback to a known-good default.
    return _FAILOVER_DEFAULT_LOCAL_MODEL


def _prepare_failover_kwargs(
    original_model: str, kwargs: dict, local_model: str,
) -> dict:
    """Clone kwargs, swap the model, cap max_tokens, clear retry count.

    Retries are cleared because LiteLLM's internal retry loop can
    interact badly with the failover path — we want the local call to
    fail fast if Ollama is unreachable, not hang on retry backoff.
    """
    new_kwargs = dict(kwargs)
    new_kwargs["model"] = local_model
    new_kwargs.pop("api_base", None)   # let litellm use OLLAMA_HOST
    new_kwargs.pop("base_url", None)
    new_kwargs["num_retries"] = 0
    mt = new_kwargs.get("max_tokens") or 0
    if mt > _FAILOVER_MAX_TOKENS:
        new_kwargs["max_tokens"] = _FAILOVER_MAX_TOKENS
    return new_kwargs


def _try_credit_failover_sync(
    exc: Exception, model: str, kwargs: dict, original_completion,
):
    """Sync-path failover.  Returns the local response on success, or
    ``None`` if failover isn't applicable (non-credit error, already in
    failover, no local model available, local call also failed)."""
    if _failover_in_progress.get():
        # Already retrying once — don't loop.
        return None
    try:
        from app.firebase.publish import detect_credit_error
    except Exception:
        return None
    if detect_credit_error(exc) is None:
        return None

    local_model = _select_local_failover_model(model)
    if not local_model:
        logger.warning(
            "failover: credit error on %r but no local model available — "
            "propagating original error", model,
        )
        return None

    new_kwargs = _prepare_failover_kwargs(model, kwargs, local_model)
    logger.warning(
        "failover: credit error on %r → retrying once with %r "
        "(max_tokens=%s)", model, local_model, new_kwargs.get("max_tokens"),
    )
    token = _failover_in_progress.set(True)
    try:
        return original_completion(**new_kwargs)
    except Exception as fallback_exc:
        logger.warning(
            "failover: local retry on %r also failed (%s: %s) — "
            "propagating original credit error",
            local_model, type(fallback_exc).__name__,
            str(fallback_exc)[:200],
        )
        return None
    finally:
        _failover_in_progress.reset(token)


async def _try_credit_failover_async(
    exc: Exception, model: str, kwargs: dict, original_acompletion,
):
    """Async-path failover — same logic as ``_try_credit_failover_sync``."""
    if _failover_in_progress.get():
        return None
    try:
        from app.firebase.publish import detect_credit_error
    except Exception:
        return None
    if detect_credit_error(exc) is None:
        return None

    local_model = _select_local_failover_model(model)
    if not local_model:
        logger.warning(
            "failover: credit error on %r but no local model available — "
            "propagating", model,
        )
        return None

    new_kwargs = _prepare_failover_kwargs(model, kwargs, local_model)
    logger.warning(
        "failover: credit error on %r → retrying once with %r async "
        "(max_tokens=%s)", model, local_model, new_kwargs.get("max_tokens"),
    )
    token = _failover_in_progress.set(True)
    try:
        return await original_acompletion(**new_kwargs)
    except Exception as fallback_exc:
        logger.warning(
            "failover: async local retry on %r also failed (%s: %s) — "
            "propagating", local_model, type(fallback_exc).__name__,
            str(fallback_exc)[:200],
        )
        return None
    finally:
        _failover_in_progress.reset(token)


# ── OpenAI SDK credit-failover (2026-04-28) ────────────────────────────
#
# CrewAI's openai-provider branch calls openai.OpenAI directly. When
# OpenRouter returns 402 the SDK raises APIStatusError without ever
# touching litellm.completion, so the litellm-level failover above
# never runs. This patch wraps the SDK's chat.completions.create
# (sync + async) symmetrically.
#
# Strategy:
#   * Patch the BOUND METHOD on the Completions class, not individual
#     instances. CrewAI creates a fresh OpenAI client per agent, so
#     instance-level patching wouldn't cover them.
#   * On 402 / "insufficient credits" / "afford" → call into the
#     existing _try_credit_failover_sync/async machinery, which already
#     knows how to pick a local Ollama model and retry once.
#   * The local retry goes through litellm so it's still subject to
#     the rate_throttle wrapper above (no double-failover, the
#     ContextVar guard prevents recursion).

_openai_patched = False


def _install_openai_credit_failover() -> None:
    """Patch ``openai.resources.chat.completions.Completions.create`` and
    its async counterpart so 402 errors trigger the same credit
    failover path as the litellm wrapper above. Idempotent."""
    global _openai_patched
    if _openai_patched:
        return
    try:
        from openai.resources.chat.completions import (
            Completions, AsyncCompletions,
        )
    except Exception as exc:
        logger.debug("openai SDK not importable, skipping patch: %s", exc)
        return

    _orig_create = Completions.create
    _orig_acreate = AsyncCompletions.create

    def _patched_create(self, *args, **kwargs):
        try:
            return _orig_create(self, *args, **kwargs)
        except Exception as exc:
            # Only hijack credit-shaped errors. Everything else propagates.
            try:
                from app.firebase.publish import detect_credit_error
                provider = detect_credit_error(exc)
                if provider is None:
                    raise
            except Exception:
                raise
            # Surface the alert so the dashboard's CreditAlertsPanel
            # can show the user a "top up" button. Same call the litellm
            # wrapper makes via _check_credit_error.
            try:
                _check_credit_error(exc, provider)
            except Exception:
                logger.debug("openai patch: credit alert reporting failed",
                             exc_info=True)
            # Build a litellm-compatible kwargs dict and call into the
            # existing failover helper. The completion request from the
            # openai SDK looks like
            # {model, messages, temperature, max_tokens, tools, ...} —
            # which is mostly compatible with litellm.completion.
            model = kwargs.get("model", "")
            try:
                import litellm
                fallback = _try_credit_failover_sync(
                    exc, model, kwargs, litellm.completion,
                )
            except Exception as fallback_exc:
                logger.debug(
                    "openai credit failover plumbing failed (%s); "
                    "propagating original 402", fallback_exc,
                )
                raise exc from None
            if fallback is None:
                # No local model available, or local also failed.
                raise
            logger.warning(
                "openai credit failover: 402 on %r → served by local Ollama",
                model,
            )
            return fallback

    async def _patched_acreate(self, *args, **kwargs):
        try:
            return await _orig_acreate(self, *args, **kwargs)
        except Exception as exc:
            try:
                from app.firebase.publish import detect_credit_error
                provider = detect_credit_error(exc)
                if provider is None:
                    raise
            except Exception:
                raise
            try:
                _check_credit_error(exc, provider)
            except Exception:
                logger.debug("openai async patch: credit alert reporting failed",
                             exc_info=True)
            model = kwargs.get("model", "")
            try:
                import litellm
                fallback = await _try_credit_failover_async(
                    exc, model, kwargs, litellm.acompletion,
                )
            except Exception as fallback_exc:
                logger.debug(
                    "openai async credit failover plumbing failed (%s); "
                    "propagating", fallback_exc,
                )
                raise exc from None
            if fallback is None:
                raise
            logger.warning(
                "openai async credit failover: 402 on %r → served by local Ollama",
                model,
            )
            return fallback

    Completions.create = _patched_create
    AsyncCompletions.create = _patched_acreate
    _openai_patched = True
    logger.info(
        "rate_throttle: openai SDK credit-failover patch installed "
        "(covers crewai's openai-provider path)"
    )


# ── Per-model reliability circuit breaker ─────────────────────────────
#
# The provider-level breakers (``openrouter``, ``anthropic``) are too
# coarse — a single flaky model behind OpenRouter (e.g. stepfun/
# step-3.5-flash going silent for 4+ min on 2026-04-25) shouldn't
# blackhole every other OpenRouter model.  This feeds a finer
# per-model breaker so the selector's pareto-demotion path can skip
# a model that keeps network-failing in-session.
#
# Trips on CONNECTION-shape failures only: TCP connect errors, read
# timeouts, "Connection error" strings from litellm's retry layer,
# service-unavailable responses.  Does NOT trip on 400 (malformed
# request) or 402 (credit-exhausted) — those are already handled by
# their own specific paths and don't reflect model reliability.

_CONNECTION_ERROR_MARKERS = (
    "connection error",
    "connection refused",
    "connection reset",
    "failed to connect",
    "timeout",
    "timed out",
    "read timed out",
    "service unavailable",
    "temporarily unavailable",
    "gateway timeout",
    "apiconnectionerror",
    "serviceunavailableerror",
    "readtimeouterror",
    "remote end closed connection",
)


def _is_connection_shaped_error(exc: Exception) -> bool:
    """Return True when the exception looks like a transient network /
    upstream-unavailable failure worth tripping the breaker on.

    Conservative: prefers false-negative (don't trip) over false-positive
    (don't blackhole a working model) since a single missed signal just
    means one more bad call; a wrong trip blacklists a good model for
    the session."""
    msg = str(exc).lower()
    if any(m in msg for m in _CONNECTION_ERROR_MARKERS):
        return True
    # HTTP status codes that indicate upstream trouble (not client error).
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if isinstance(status, int) and status in (502, 503, 504):
        return True
    return False


def _normalize_model_name(model: str) -> str:
    """Strip common litellm provider prefixes so the breaker key is
    consistent across provider routes (``openrouter/stepfun/X`` and
    ``stepfun/X`` both map to ``stepfun/X``)."""
    if not model:
        return ""
    m = model.strip()
    for prefix in ("openrouter/", "openai/", "anthropic/", "litellm/"):
        if m.lower().startswith(prefix):
            m = m[len(prefix):]
            break
    return m


def _record_model_reliability(
    model: str, exc: Exception | None, *, success: bool,
) -> None:
    """Feed the per-model circuit breaker.  Silent no-op on any
    failure of the breaker module itself — reliability telemetry
    must never break the actual LLM call path."""
    try:
        from app.circuit_breaker import get_breaker
    except Exception:
        return
    key = f"model:{_normalize_model_name(model)}"
    if not key.endswith(":"):  # i.e. we had a non-empty model name
        try:
            breaker = get_breaker(key)
            # Use a shorter threshold + cooldown than provider-level
            # breakers.  A model that fails 3 times in a row has lost
            # its chance for this session; 5 min is long enough that
            # stepfun-style multi-minute outages don't pull a good
            # model back too eagerly.
            breaker.failure_threshold = 3
            breaker.cooldown_seconds = 300
            if success:
                breaker.record_success()
            elif exc is not None and _is_connection_shaped_error(exc):
                breaker.record_failure()
            # Non-connection errors (400s, 402s, malformed tool schema)
            # don't count toward the model's reliability score.
        except Exception:
            pass


def _capture_upstream_error_body(
    exc: Exception, model: str, provider: str, kwargs: dict,
) -> None:
    """Log the full upstream response body for a failed LLM call.

    Fail-soft: wrapped in try/except — a logging bug must never hide
    the original error.  The caller still ``raise``s afterward.
    """
    try:
        # LiteLLM's OpenAI-compat errors expose these attributes; the
        # field availability varies by SDK version so we probe.
        parts: list[str] = []
        status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        if status_code:
            parts.append(f"status={status_code}")
        parts.append(f"model={model!r}")
        parts.append(f"provider={provider!r}")
        # Request shape diagnostics — most useful for context-overflow +
        # tool-schema-reject cases.
        messages = kwargs.get("messages") or []
        if messages:
            # Rough prompt-size estimate without importing tokenizers.
            # A character / 4 heuristic is within ~30 % of real token
            # counts for English — good enough to flag "200 K ballpark".
            total_chars = sum(len(str(m.get("content", ""))) for m in messages)
            parts.append(f"n_messages={len(messages)}")
            parts.append(f"~chars={total_chars}")
        tools = kwargs.get("tools") or []
        if tools:
            parts.append(f"n_tools={len(tools)}")
        # Response body (the goldmine).
        body = getattr(exc, "body", None) or getattr(exc, "response", None)
        body_str: str | None = None
        if body is not None:
            # Response object → try .text, .json(), .content, or str().
            for attr in ("text", "content"):
                val = getattr(body, attr, None)
                if val:
                    body_str = str(val)[:2000]
                    break
            if body_str is None:
                try:
                    body_str = str(body)[:2000]
                except Exception:
                    body_str = "<unprintable>"
        # Some SDKs stash the raw JSON error on the exception itself.
        raw = getattr(exc, "raw", None)
        if raw and not body_str:
            body_str = str(raw)[:2000]
        if body_str:
            parts.append(f"body={body_str!r}")
        # Exception type + message as fallback for anything we missed.
        parts.append(f"exc_type={type(exc).__name__}")
        parts.append(f"exc_msg={str(exc)[:1000]!r}")

        logger.warning(
            "llm_error_captured: %s", " | ".join(parts),
        )
    except Exception as log_exc:
        logger.debug(
            "llm_error_capture_failed: %s", log_exc,
        )
