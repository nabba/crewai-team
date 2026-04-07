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
                try:
                    response = _original_completion(*args, **kwargs)
                except Exception as exc:
                    _check_credit_error(exc, provider)
                    raise
                # Successful call — resolve any prior credit alert for this provider
                _resolve_credit_if_needed(provider)
                _record_token_usage(response, kwargs)
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
                    response = await _original_acompletion(*args, **kwargs)
                    _record_token_usage(response, kwargs)
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

        # Also patch CrewAI's BaseLLM._track_token_usage_internal to record
        # token usage for ALL providers (Anthropic, Gemini, etc.) that CrewAI
        # handles natively without going through litellm.
        try:
            from crewai.llms.base_llm import BaseLLM
            _original_track = BaseLLM._track_token_usage_internal

            def _patched_track(self, usage_data: dict):
                _original_track(self, usage_data)
                try:
                    model = getattr(self, "model", "unknown")
                    prompt = (
                        usage_data.get("prompt_tokens")
                        or usage_data.get("input_tokens")
                        or usage_data.get("prompt_token_count")
                        or 0
                    )
                    completion = (
                        usage_data.get("completion_tokens")
                        or usage_data.get("output_tokens")
                        or usage_data.get("candidates_token_count")
                        or 0
                    )
                    total = prompt + completion
                    if total > 0:
                        cost_usd = 0.0
                        cost_info = _find_cost(model) or _find_cost(f"anthropic/{model}")
                        if cost_info:
                            cost_usd = (
                                (prompt / 1_000_000) * cost_info[0]
                                + (completion / 1_000_000) * cost_info[1]
                            )
                        from app.llm_benchmarks import record_tokens, record
                        record_tokens(model, prompt, completion, cost_usd)
                        # Also record in benchmarks table for model scoring
                        record(model, "general", True, latency_ms=0, tokens=prompt+completion)
                        tracker = _request_cost.get(None)
                        if tracker is not None:
                            tracker.record(model, prompt, completion, cost_usd)
                except Exception:
                    pass

            BaseLLM._track_token_usage_internal = _patched_track
            logger.info("rate_throttle: CrewAI BaseLLM token tracking patched")
        except Exception:
            logger.debug("rate_throttle: could not patch BaseLLM", exc_info=True)


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


def _record_token_usage(response, kwargs: dict) -> None:
    """Extract token usage from litellm response and record it."""
    try:
        usage = getattr(response, "usage", None)
        if not usage:
            return
        model = getattr(response, "model", "") or kwargs.get("model", "unknown")
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total = prompt_tokens + completion_tokens
        if total > 0:
            # Estimate cost from catalog
            cost_usd = 0.0
            try:
                cost_info = _find_cost(model)
                if cost_info:
                    cost_usd = (
                        (prompt_tokens / 1_000_000) * cost_info[0]
                        + (completion_tokens / 1_000_000) * cost_info[1]
                    )
            except Exception:
                pass
            from app.llm_benchmarks import record_tokens
            record_tokens(model, prompt_tokens, completion_tokens, cost_usd)

            # Also accumulate into the active request tracker if present
            tracker = _request_cost.get(None)
            if tracker is not None:
                tracker.record(model, prompt_tokens, completion_tokens, cost_usd)
    except Exception:
        pass  # never fail the actual LLM call


# ── Credit alert integration ─────────────────────────────────────────────────

_resolved_providers: set[str] = set()  # avoid repeated resolve calls


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
    """Begin accumulating costs for a user request. Returns the tracker."""
    tracker = RequestCostTracker(request_id)
    _request_cost.set(tracker)
    return tracker


def stop_request_tracking() -> RequestCostTracker | None:
    """Stop tracking and return the accumulated tracker."""
    tracker = _request_cost.get(None)
    _request_cost.set(None)
    return tracker


def get_active_tracker() -> "RequestCostTracker | None":
    """Get the active request tracker (for propagating to threads)."""
    return _request_cost.get(None)


def set_active_tracker(tracker: "RequestCostTracker | None") -> None:
    """Set the request tracker (for thread propagation)."""
    _request_cost.set(tracker)
