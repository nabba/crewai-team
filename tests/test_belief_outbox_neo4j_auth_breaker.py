"""Regression: belief_outbox must circuit-break Neo4j auth failures.

Pre-fix shape (the operator-reported bug):

  pattern_learner reported 589 occurrences/week of:
    "belief_outbox: neo4j read failed: {neo4j_code:
     Neo.ClientError.Security.Unauthorized} {message: The client is
     unauthorized due to authentication failure.}"

  The reconciler ran every MEDIUM idle slot (≈few-minutes cadence),
  hit the auth failure each time, logged WARN each time. The
  underlying issue (wrong password / rotated credential) cannot be
  fixed by retry — only the operator can rotate. So we were
  hammering Neo4j 589×/week with no chance of self-recovery.

Post-fix:
  • First auth failure trips the `neo4j_auth` circuit breaker
    (failure_threshold=1, cooldown=1 h)
  • Subsequent calls inside the cooldown short-circuit silently
  • The breaker's first-trip CLOSED→OPEN transition logs WARN
    once (operator alert); HALF_OPEN→OPEN re-trips log INFO
    (per T1.3, no spam)
  • Transient (non-auth) errors still log WARN — we don't lose
    visibility into legitimately-broken connectivity
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def reset_breakers():
    """Each test starts with a clean breaker state to avoid order
    dependence."""
    from app import circuit_breaker
    # Force-close the neo4j_auth breaker so prior tests don't leak.
    breaker = circuit_breaker.get_breaker("neo4j_auth")
    with breaker._lock:
        breaker._state = "CLOSED"
        breaker._failure_count = 0
        breaker._opened_at = None
    yield
    with breaker._lock:
        breaker._state = "CLOSED"
        breaker._failure_count = 0
        breaker._opened_at = None


# ── Auth-error detection ───────────────────────────────────────────


class TestIsNeo4jAuthError:

    def test_unauthorized_code_matches(self) -> None:
        from app.memory.belief_outbox import _is_neo4j_auth_error

        exc = Exception(
            "{neo4j_code: Neo.ClientError.Security.Unauthorized} "
            "{message: The client is unauthorized due to authentication failure.}"
        )
        assert _is_neo4j_auth_error(exc) is True

    def test_auth_rate_limit_matches(self) -> None:
        from app.memory.belief_outbox import _is_neo4j_auth_error

        exc = Exception(
            "{neo4j_code: Neo.ClientError.Security.AuthenticationRateLimit}"
        )
        assert _is_neo4j_auth_error(exc) is True

    def test_lower_case_message_matches(self) -> None:
        from app.memory.belief_outbox import _is_neo4j_auth_error

        # The driver also surfaces "client is unauthorized" without
        # the formal code; substring still hits.
        exc = Exception("server says: client is unauthorized; aborting")
        assert _is_neo4j_auth_error(exc) is True

    def test_transient_error_does_not_match(self) -> None:
        """A connection timeout / server-down should NOT trip the
        operator-action breaker — those are transient and benefit
        from retry."""
        from app.memory.belief_outbox import _is_neo4j_auth_error

        exc = Exception("ConnectionRefusedError: [Errno 111] Connection refused")
        assert _is_neo4j_auth_error(exc) is False

        exc2 = Exception("ServiceUnavailable: Failed to read from defunct connection")
        assert _is_neo4j_auth_error(exc2) is False


# ── Breaker registration ───────────────────────────────────────────


class TestNeo4jAuthBreakerRegistered:

    def test_breaker_exists(self) -> None:
        from app import circuit_breaker
        breaker = circuit_breaker.get_breaker("neo4j_auth")
        assert breaker is not None
        assert breaker.name == "neo4j_auth"

    def test_breaker_threshold_and_cooldown(self) -> None:
        """Operator-action breaker must trip on first failure with
        a long cooldown — anything else is just retry-storm theater."""
        from app import circuit_breaker
        breaker = circuit_breaker.get_breaker("neo4j_auth")
        assert breaker.failure_threshold == 1, (
            "neo4j_auth must trip on FIRST failure (operator-action)"
        )
        assert breaker.cooldown_seconds >= 600, (
            f"neo4j_auth cooldown must be ≥10 min, got {breaker.cooldown_seconds}s"
        )


# ── Reconciler integration ─────────────────────────────────────────


class TestReconcilerShortCircuitsWhenBreakerOpen:

    def test_breaker_open_returns_none_without_neo4j_call(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When the auth breaker is OPEN, the function must return
        None WITHOUT touching neo4j_mirror (saves the round trip and
        avoids re-logging)."""
        from app import circuit_breaker
        from app.memory import belief_outbox

        # Trip the breaker.
        circuit_breaker.record_failure("neo4j_auth")
        assert not circuit_breaker.is_available("neo4j_auth")

        # Sentinel that gets flipped if neo4j_mirror import is reached.
        called = []

        class _FakeMirror:
            @staticmethod
            def is_available() -> bool:
                called.append("is_available")
                return True

        # If we get past the breaker check, the import will run; we
        # patch the symbol pre-import to detect that.
        import sys
        monkeypatch.setitem(
            sys.modules, "app.subia.belief.neo4j_mirror", _FakeMirror,
        )

        result = belief_outbox._fetch_existing_neo4j_belief_ids()
        assert result is None
        assert called == [], (
            "breaker-open path must NOT touch neo4j_mirror; "
            f"got calls: {called}"
        )


class TestAuthErrorTripsTheBreaker:

    def test_auth_error_trips_breaker_and_logs_info(
        self, monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """When the Neo4j read raises an auth error, the breaker must
        trip AND the belief_outbox-side log must be INFO (the breaker
        already logs WARN at first OPEN — no need to double up)."""
        import logging
        import sys
        from app import circuit_breaker
        from app.memory import belief_outbox

        # Build a fake neo4j_mirror that raises an auth-error string.
        class _FakeDriver:
            def session(self):
                return self

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def run(self, *_a, **_kw):
                raise Exception(
                    "{neo4j_code: Neo.ClientError.Security.Unauthorized}"
                )

        class _FakeMirror:
            @staticmethod
            def is_available() -> bool:
                return True

            @staticmethod
            def _get_driver():
                return _FakeDriver()

        monkeypatch.setitem(
            sys.modules, "app.subia.belief.neo4j_mirror", _FakeMirror,
        )
        # Also stub the parent package so the `from .. import neo4j_mirror`
        # resolves correctly.
        if "app.subia.belief" not in sys.modules:
            import types
            mod = types.ModuleType("app.subia.belief")
            mod.neo4j_mirror = _FakeMirror
            monkeypatch.setitem(sys.modules, "app.subia.belief", mod)
        else:
            monkeypatch.setattr(
                sys.modules["app.subia.belief"], "neo4j_mirror", _FakeMirror,
                raising=False,
            )

        with caplog.at_level(logging.INFO, logger="app.memory.belief_outbox"):
            result = belief_outbox._fetch_existing_neo4j_belief_ids()

        assert result is None
        # Breaker should be OPEN now.
        assert not circuit_breaker.is_available("neo4j_auth")

        # belief_outbox-side log should be INFO (not WARN) — the
        # breaker provided the operator-visible WARN.
        ours = [
            r for r in caplog.records
            if r.name == "app.memory.belief_outbox"
            and "neo4j auth failed" in r.message
        ]
        assert len(ours) == 1, (
            f"expected 1 INFO from belief_outbox; got {len(ours)}: "
            f"{[r.message for r in ours]}"
        )
        assert ours[0].levelno == logging.INFO


class TestTransientErrorStillWarnsAndDoesNotTripBreaker:

    def test_transient_error_warns_and_does_not_trip(
        self, monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Connection timeouts must STILL log WARN (real connectivity
        is broken) and must NOT trip the operator-action breaker."""
        import logging
        import sys
        import types

        from app import circuit_breaker
        from app.memory import belief_outbox

        class _FakeDriver:
            def session(self):
                return self

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def run(self, *_a, **_kw):
                raise Exception(
                    "ConnectionRefusedError: [Errno 111] Connection refused"
                )

        class _FakeMirror:
            @staticmethod
            def is_available() -> bool:
                return True

            @staticmethod
            def _get_driver():
                return _FakeDriver()

        if "app.subia.belief" not in sys.modules:
            mod = types.ModuleType("app.subia.belief")
            mod.neo4j_mirror = _FakeMirror
            monkeypatch.setitem(sys.modules, "app.subia.belief", mod)
        monkeypatch.setitem(
            sys.modules, "app.subia.belief.neo4j_mirror", _FakeMirror,
        )

        # Pre-condition: breaker closed.
        assert circuit_breaker.is_available("neo4j_auth")

        with caplog.at_level(logging.WARNING, logger="app.memory.belief_outbox"):
            result = belief_outbox._fetch_existing_neo4j_belief_ids()

        assert result is None
        # Breaker must remain CLOSED — transient errors are not
        # operator-action.
        assert circuit_breaker.is_available("neo4j_auth"), (
            "transient connection error must NOT trip the auth breaker"
        )

        warns = [
            r for r in caplog.records
            if r.name == "app.memory.belief_outbox"
            and "neo4j read failed" in r.message
            and r.levelno == logging.WARNING
        ]
        assert len(warns) == 1, (
            f"expected 1 WARN for transient error; got {len(warns)}"
        )
