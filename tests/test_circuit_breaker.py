"""Tests for app/circuit_breaker.py — provider resilience state machine."""
import time
import threading
import unittest

from app.circuit_breaker import (
    CircuitBreaker, CLOSED, OPEN, HALF_OPEN,
    is_available, record_success, record_failure, get_all_states, get_breaker,
)


class TestCircuitBreakerStateMachine(unittest.TestCase):
    """Test the CLOSED → OPEN → HALF_OPEN → CLOSED state transitions."""

    def setUp(self):
        self.cb = CircuitBreaker("test", failure_threshold=3, cooldown_seconds=1)

    def test_starts_closed(self):
        assert self.cb.state == CLOSED
        assert self.cb.failure_count == 0
        assert not self.cb.is_open()

    def test_stays_closed_under_threshold(self):
        self.cb.record_failure()
        self.cb.record_failure()
        assert self.cb.state == CLOSED
        assert self.cb.failure_count == 2
        assert not self.cb.is_open()

    def test_opens_at_threshold(self):
        for _ in range(3):
            self.cb.record_failure()
        assert self.cb.state == OPEN
        assert self.cb.is_open()
        assert self.cb.failure_count == 3

    def test_success_resets_from_closed(self):
        self.cb.record_failure()
        self.cb.record_failure()
        self.cb.record_success()
        assert self.cb.state == CLOSED
        assert self.cb.failure_count == 0

    def test_half_open_after_cooldown(self):
        for _ in range(3):
            self.cb.record_failure()
        assert self.cb.state == OPEN
        time.sleep(1.1)  # cooldown_seconds=1
        assert self.cb.state == HALF_OPEN

    def test_half_open_success_closes(self):
        for _ in range(3):
            self.cb.record_failure()
        time.sleep(1.1)
        assert self.cb.state == HALF_OPEN
        self.cb.record_success()
        assert self.cb.state == CLOSED
        assert self.cb.failure_count == 0

    def test_half_open_failure_reopens(self):
        for _ in range(3):
            self.cb.record_failure()
        time.sleep(1.1)
        assert self.cb.state == HALF_OPEN
        self.cb.record_failure()
        assert self.cb.state == OPEN

    def test_stays_open_during_cooldown(self):
        for _ in range(3):
            self.cb.record_failure()
        assert self.cb.state == OPEN
        # No sleep — should still be open
        assert self.cb.state == OPEN


class TestCircuitBreakerThreadSafety(unittest.TestCase):
    """Verify concurrent access doesn't corrupt state."""

    def test_concurrent_failures(self):
        cb = CircuitBreaker("thread_test", failure_threshold=100, cooldown_seconds=60)
        threads = []
        for _ in range(100):
            t = threading.Thread(target=cb.record_failure)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        assert cb.failure_count == 100
        assert cb.state == OPEN


class TestModuleLevelAPI(unittest.TestCase):
    """Test the module-level convenience functions."""

    def test_registry_has_default_providers(self):
        states = get_all_states()
        assert "ollama" in states
        assert "openrouter" in states
        assert "anthropic" in states

    def test_is_available_initially_true(self):
        # Reset by recording success
        record_success("ollama")
        assert is_available("ollama") is True

    def test_get_breaker_creates_new(self):
        b = get_breaker("new_provider_xyz")
        assert b.name == "new_provider_xyz"
        assert b.state == CLOSED

    def test_get_all_states_format(self):
        record_success("ollama")
        states = get_all_states()
        assert "state" in states["ollama"]
        assert "failures" in states["ollama"]
        assert states["ollama"]["state"] == "closed"
        assert states["ollama"]["failures"] == 0


if __name__ == "__main__":
    unittest.main()
