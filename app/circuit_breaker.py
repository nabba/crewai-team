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
    """Thread-safe circuit breaker for a single provider.

    Supports ``force_allow()`` for user-facing requests that should
    bypass the OPEN state and probe the provider directly — this
    prevents background-task failures from blocking real users.
    """

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

    def seconds_until_half_open(self) -> float:
        """Seconds remaining before this breaker transitions to HALF_OPEN."""
        with self._lock:
            if self._state != OPEN:
                return 0.0
            elapsed = time.monotonic() - self._opened_at
            return max(0.0, self.cooldown_seconds - elapsed)

    def is_open(self) -> bool:
        """Returns True if the circuit is OPEN (should skip this provider)."""
        return self.state == OPEN

    def force_allow(self) -> bool:
        """Force-transition to HALF_OPEN for a user-facing probe request.

        Returns True if the caller should proceed with a probe call.
        Only works when OPEN — if CLOSED or HALF_OPEN already, returns True.
        """
        with self._lock:
            if self._state == CLOSED:
                return True
            if self._state == HALF_OPEN:
                return True
            # OPEN → force to HALF_OPEN so exactly one probe goes through
            self._state = HALF_OPEN
            logger.info(
                f"circuit_breaker[{self.name}]: OPEN → HALF_OPEN "
                f"(forced by user-facing request)"
            )
            return True

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
                # One failure in half-open → back to open.  This is the
                # "still broken" transition — we tried to probe recovery
                # and the underlying issue is still there.  INFO not
                # WARN: the operator already saw the original CLOSED→
                # OPEN warning when the issue first appeared, and the
                # breaker is doing its job by re-opening.  Logging this
                # as WARN every cooldown cycle (e.g. once/hour for
                # anthropic_credits) just spams errors.jsonl with the
                # same root cause.
                self._state = OPEN
                self._opened_at = time.monotonic()
                logger.info(
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
    "ollama": CircuitBreaker("ollama", failure_threshold=5, cooldown_seconds=30),
    "openrouter": CircuitBreaker("openrouter", failure_threshold=5, cooldown_seconds=30),
    "anthropic": CircuitBreaker("anthropic", failure_threshold=8, cooldown_seconds=45),
    # `anthropic_credits` is a semantic-specific breaker: it trips the
    # instant we see a 400 "credit balance too low" response.  Unlike the
    # generic `anthropic` transient-failure breaker (threshold 8, 45s),
    # credit exhaustion is not a transient glitch — a single occurrence
    # is authoritative.  The long cooldown gives the operator enough
    # time to top up the account before we probe direct Anthropic again.
    "anthropic_credits": CircuitBreaker(
        "anthropic_credits", failure_threshold=1, cooldown_seconds=3600,
    ),
    # Operator-action breakers.  Same pattern as anthropic_credits —
    # the underlying issue (auth failure / wrong token) cannot be
    # fixed by code; only the operator can rotate the credential.
    # We trip on first failure with a long cooldown so the
    # reconciler / client doesn't hammer the upstream once/minute.
    # First trip logs WARN (operator alert); subsequent re-trips
    # within the cooldown log INFO via the HALF_OPEN→OPEN path.
    "neo4j_auth": CircuitBreaker(
        "neo4j_auth", failure_threshold=1, cooldown_seconds=3600,
    ),
    "mcp_auth": CircuitBreaker(
        "mcp_auth", failure_threshold=1, cooldown_seconds=3600,
    ),
    "self_healer": CircuitBreaker("self_healer", failure_threshold=3, cooldown_seconds=600),
}


def get_breaker(provider: str) -> CircuitBreaker:
    """Get or create a circuit breaker for a provider."""
    if provider not in _breakers:
        _breakers[provider] = CircuitBreaker(provider)
    return _breakers[provider]


def ensure_breaker(
    provider: str,
    *,
    failure_threshold: int = 5,
    cooldown_seconds: int = 30,
) -> CircuitBreaker:
    """Get or create a breaker WITH explicit config.

    Differs from ``get_breaker``: dynamically-created breakers (e.g.
    one per MCP server, one per Neo4j cluster shard) need the right
    threshold/cooldown shape from the start, not the generic 5/30
    defaults.  Returns the existing breaker unchanged if one is
    already registered under ``provider`` (existing config wins —
    we never silently re-config).
    """
    if provider not in _breakers:
        _breakers[provider] = CircuitBreaker(
            provider,
            failure_threshold=failure_threshold,
            cooldown_seconds=cooldown_seconds,
        )
    return _breakers[provider]


def is_available(provider: str) -> bool:
    """Check if a provider's circuit is NOT open (i.e., safe to try)."""
    return not get_breaker(provider).is_open()


def record_success(provider: str) -> None:
    get_breaker(provider).record_success()


def record_failure(provider: str) -> None:
    get_breaker(provider).record_failure()


def force_all_half_open() -> None:
    """Force all LLM breakers to HALF_OPEN for a user-facing request.

    This ensures at least one probe attempt per provider, so a user
    request isn't rejected just because background tasks tripped the
    breakers simultaneously.
    """
    for name in ("anthropic", "openrouter", "ollama"):
        if name in _breakers:
            _breakers[name].force_allow()


def get_all_states() -> dict[str, dict]:
    """Return all breaker states for observability (dashboard/heartbeat)."""
    return {
        name: {"state": b.state, "failures": b.failure_count}
        for name, b in _breakers.items()
    }
