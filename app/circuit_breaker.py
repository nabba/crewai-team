"""
circuit_breaker.py — Circuit breaker pattern for LLM provider resilience.

When a provider (Ollama, OpenRouter) fails repeatedly, the circuit opens
and subsequent requests skip that provider instantly instead of waiting
for a slow timeout on every call.

State machine:
  CLOSED → (failure_threshold consecutive failures) → OPEN
  OPEN   → (cooldown_seconds elapsed) → HALF_OPEN
  HALF_OPEN → (one success) → CLOSED
  HALF_OPEN → (one failure) → OPEN
"""

import logging
import threading
import time

logger = logging.getLogger(__name__)

CLOSED = "closed"
OPEN = "open"
HALF_OPEN = "half_open"


class CircuitBreaker:
    """Thread-safe circuit breaker for a single provider."""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        cooldown_seconds: int = 60,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._lock = threading.Lock()
        self._state = CLOSED
        self._failure_count = 0
        self._opened_at: float = 0.0

    @property
    def state(self) -> str:
        with self._lock:
            if self._state == OPEN:
                if time.monotonic() - self._opened_at >= self.cooldown_seconds:
                    self._state = HALF_OPEN
                    logger.info(
                        f"circuit_breaker[{self.name}]: OPEN → HALF_OPEN "
                        f"(cooldown {self.cooldown_seconds}s elapsed)"
                    )
            return self._state

    @property
    def failure_count(self) -> int:
        with self._lock:
            return self._failure_count

    def is_open(self) -> bool:
        """Returns True if the circuit is OPEN (should skip this provider)."""
        return self.state == OPEN

    def record_success(self) -> None:
        """Reset the breaker on a successful call."""
        with self._lock:
            if self._state != CLOSED:
                logger.info(
                    f"circuit_breaker[{self.name}]: {self._state} → CLOSED (success)"
                )
            self._state = CLOSED
            self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failure. Opens the circuit after threshold consecutive failures."""
        with self._lock:
            self._failure_count += 1
            if self._state == HALF_OPEN:
                # One failure in half-open → back to open
                self._state = OPEN
                self._opened_at = time.monotonic()
                logger.warning(
                    f"circuit_breaker[{self.name}]: HALF_OPEN → OPEN "
                    f"(probe failed, cooldown {self.cooldown_seconds}s)"
                )
            elif self._failure_count >= self.failure_threshold:
                prev = self._state
                self._state = OPEN
                self._opened_at = time.monotonic()
                logger.warning(
                    f"circuit_breaker[{self.name}]: {prev} → OPEN "
                    f"({self._failure_count} consecutive failures, "
                    f"cooldown {self.cooldown_seconds}s)"
                )


# ── Module-level registry ────────────────────────────────────────────────────

_breakers: dict[str, CircuitBreaker] = {
    "ollama": CircuitBreaker("ollama", failure_threshold=3, cooldown_seconds=60),
    "openrouter": CircuitBreaker("openrouter", failure_threshold=3, cooldown_seconds=60),
    "anthropic": CircuitBreaker("anthropic", failure_threshold=5, cooldown_seconds=120),
}


def get_breaker(provider: str) -> CircuitBreaker:
    """Get or create a circuit breaker for a provider."""
    if provider not in _breakers:
        _breakers[provider] = CircuitBreaker(provider)
    return _breakers[provider]


def is_available(provider: str) -> bool:
    """Check if a provider's circuit is NOT open (i.e., safe to try)."""
    return not get_breaker(provider).is_open()


def record_success(provider: str) -> None:
    get_breaker(provider).record_success()


def record_failure(provider: str) -> None:
    get_breaker(provider).record_failure()


def get_all_states() -> dict[str, dict]:
    """Return all breaker states for observability (dashboard/heartbeat)."""
    return {
        name: {"state": b.state, "failures": b.failure_count}
        for name, b in _breakers.items()
    }
