"""
rate_throttle.py — Global rate limiter for outgoing LLM API calls.

Anthropic free/low-tier orgs have a 5 requests/minute limit.
This module provides a token-bucket throttle that all CrewAI LLM
instances share, preventing rate-limit errors by waiting instead
of hammering the API.

Also configures litellm's built-in retry with exponential backoff
so transient 429s are retried automatically.
"""

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
                return _original_completion(*args, **kwargs)

            litellm.completion = _throttled_completion

            # Also patch acompletion for async paths
            if hasattr(litellm, "acompletion"):
                _original_acompletion = litellm.acompletion

                async def _throttled_acompletion(*args, **kwargs):
                    throttle()  # blocking is fine — runs in thread anyway
                    kwargs.setdefault("num_retries", _RETRY_COUNT)
                    return await _original_acompletion(*args, **kwargs)

                litellm.acompletion = _throttled_acompletion

            _patched = True
            logger.info(f"rate_throttle: installed ({_MAX_RPM} RPM, {_RETRY_COUNT} retries)")
        except ImportError:
            logger.warning("rate_throttle: litellm not found, throttle not installed")
