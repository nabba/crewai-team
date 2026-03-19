"""
rate_throttle.py — Global rate limiter for outgoing LLM API calls.

Anthropic free/low-tier orgs have a 5 requests/minute limit.
This module provides a token-bucket throttle that all CrewAI LLM
instances share, preventing rate-limit errors by waiting instead
of hammering the API.

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
# These can be overridden via .env
# Default 3 RPM: conservative for free tier (5 RPM + 10K input tokens/min).
# Each crew call uses ~2-4K input tokens, so 3 RPM ≈ 9K tokens/min — just under limit.
_MAX_RPM = int(os.environ.get("ANTHROPIC_MAX_RPM", "3"))
_RETRY_COUNT = int(os.environ.get("LITELLM_NUM_RETRIES", "5"))
_RETRY_BACKOFF = float(os.environ.get("LITELLM_RETRY_BACKOFF", "15"))  # seconds

# ── litellm retry config (set before any litellm import) ──────────────────────
os.environ.setdefault("LITELLM_NUM_RETRIES", str(_RETRY_COUNT))

# ── Token bucket rate limiter ─────────────────────────────────────────────────

class _TokenBucket:
    """Thread-safe token bucket allowing at most `rate` calls per 60 seconds."""

    def __init__(self, rate: int):
        self.rate = max(1, rate)
        self.interval = 60.0 / self.rate  # seconds between tokens
        self._lock = threading.Lock()
        self._last = 0.0

    def acquire(self) -> None:
        """Block until a token is available."""
        with self._lock:
            now = time.monotonic()
            wait = self._last + self.interval - now
            if wait > 0:
                logger.debug(f"rate_throttle: waiting {wait:.1f}s before next API call")
                time.sleep(wait)
            self._last = time.monotonic()


_bucket = _TokenBucket(_MAX_RPM)


def throttle() -> None:
    """Call before every LLM API request to respect the rate limit."""
    _bucket.acquire()


# ── Monkey-patch litellm completion to inject throttle ────────────────────────

_patched = False
_patch_lock = threading.Lock()


def install_throttle() -> None:
    """
    Patch litellm.completion to call throttle() before each request.
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
                throttle()
                # Inject retry params if not already set
                kwargs.setdefault("num_retries", _RETRY_COUNT)
                response = _original_completion(*args, **kwargs)
                _record_token_usage(response, kwargs)
                return response

            litellm.completion = _throttled_completion

            # Also patch acompletion for async paths
            if hasattr(litellm, "acompletion"):
                _original_acompletion = litellm.acompletion

                async def _throttled_acompletion(*args, **kwargs):
                    throttle()  # blocking is fine — runs in thread anyway
                    kwargs.setdefault("num_retries", _RETRY_COUNT)
                    response = await _original_acompletion(*args, **kwargs)
                    _record_token_usage(response, kwargs)
                    return response

                litellm.acompletion = _throttled_acompletion

            _patched = True
            logger.info(f"rate_throttle: installed ({_MAX_RPM} RPM, {_RETRY_COUNT} retries, token tracking)")
        except ImportError:
            logger.warning("rate_throttle: litellm not found, throttle not installed")


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
                from app.llm_catalog import CATALOG
                for _name, info in CATALOG.items():
                    if info.get("model_id", "") == model or _name == model:
                        cost_usd = (
                            (prompt_tokens / 1_000_000) * info.get("cost_input_per_m", 0)
                            + (completion_tokens / 1_000_000) * info.get("cost_output_per_m", 0)
                        )
                        break
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


# ── Request-level cost tracking ──────────────────────────────────────────────

_request_cost: contextvars.ContextVar["RequestCostTracker | None"] = contextvars.ContextVar(
    "request_cost", default=None,
)


class RequestCostTracker:
    """Accumulates token usage across all LLM calls in a single user request."""

    def __init__(self, request_id: str = ""):
        self.request_id = request_id
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
