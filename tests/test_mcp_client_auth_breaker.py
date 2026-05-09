"""Regression: mcp_client must circuit-break per-server on 401/403.

Pre-fix shape (the operator-reported bug):

  pattern_learner reported recurring:
    "mcp_client: 'STUzhy/py_execute_mcp' init failed:
     HTTP 401: {'error':'invalid_token','error_description':...}"

  The MCP registry calls connect_all() at startup AND on each
  reconnect-after-restart cycle. With a stale/rotated token, every
  connect attempt logged WARN. Since the operator must rotate the
  token, retrying is pointless until they do.

Post-fix:
  • _is_mcp_auth_error() detects 401/403/invalid_token/Unauthorized
  • Per-server breaker key (`mcp_auth:<server-name>`) — auth failure
    on one server does NOT block connections to other servers
  • First trip logs WARN once via the breaker's CLOSED→OPEN path;
    belief_outbox-style INFO on the client side (no double-WARN)
  • Subsequent connect()s on the same server skip silently while
    the breaker is OPEN (1 h cooldown)
  • Non-auth failures still WARN at the client level
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def reset_breakers():
    """Each test starts with a clean breaker state."""
    from app import circuit_breaker
    yield
    # Wipe any test-created mcp_auth:* breakers so prior tests
    # don't leak.
    keys_to_clear = [
        k for k in circuit_breaker._breakers
        if k.startswith("mcp_auth:")
    ]
    for k in keys_to_clear:
        breaker = circuit_breaker._breakers[k]
        with breaker._lock:
            breaker._state = "CLOSED"
            breaker._failure_count = 0
            breaker._opened_at = None


# ── Auth-error detection ───────────────────────────────────────────


class TestIsMCPAuthError:

    def test_http_401_matches(self) -> None:
        from app.mcp.client import _is_mcp_auth_error
        assert _is_mcp_auth_error('HTTP 401: {"error":"invalid_token"}')
        assert _is_mcp_auth_error("RuntimeError: HTTP 401 from upstream")

    def test_http_403_matches(self) -> None:
        from app.mcp.client import _is_mcp_auth_error
        assert _is_mcp_auth_error('HTTP 403: {"error":"forbidden"}')

    def test_invalid_token_matches(self) -> None:
        from app.mcp.client import _is_mcp_auth_error
        assert _is_mcp_auth_error(
            '{"error":"invalid_token","error_description":"expired"}'
        )

    def test_unauthorized_matches(self) -> None:
        from app.mcp.client import _is_mcp_auth_error
        assert _is_mcp_auth_error("server returned: Unauthorized")

    def test_transient_error_does_not_match(self) -> None:
        """Connection refused / timeout must NOT trip auth breaker —
        those are transient and benefit from retry."""
        from app.mcp.client import _is_mcp_auth_error
        assert _is_mcp_auth_error(
            "ConnectionRefusedError: [Errno 111]"
        ) is False
        assert _is_mcp_auth_error(
            "TimeoutError: read timed out after 30s"
        ) is False


# ── Breaker name namespacing ───────────────────────────────────────


class TestPerServerBreakerName:

    def test_breaker_name_is_per_server(self) -> None:
        from app.mcp.client import _mcp_breaker_name

        assert _mcp_breaker_name("STUzhy/py_execute_mcp") == (
            "mcp_auth:STUzhy/py_execute_mcp"
        )
        # Different server → different breaker
        assert _mcp_breaker_name("memory") != _mcp_breaker_name("filesystem")


# ── ensure_breaker integration ─────────────────────────────────────


class TestEnsureBreakerHelper:

    def test_creates_with_explicit_threshold_and_cooldown(self) -> None:
        from app import circuit_breaker

        name = "test_ensure_breaker_unique_xyz_123"
        # Pre-condition: no such breaker.
        assert name not in circuit_breaker._breakers

        b = circuit_breaker.ensure_breaker(
            name, failure_threshold=1, cooldown_seconds=3600,
        )
        assert b.failure_threshold == 1
        assert b.cooldown_seconds == 3600

        # Cleanup so we don't pollute other tests.
        del circuit_breaker._breakers[name]

    def test_returns_existing_unchanged(self) -> None:
        """Existing breaker config wins — we never silently re-config
        a breaker mid-process."""
        from app import circuit_breaker

        name = "test_ensure_existing_unique_xyz_456"
        # Create with one config.
        circuit_breaker.ensure_breaker(
            name, failure_threshold=1, cooldown_seconds=3600,
        )
        # Try to recreate with different config.
        b = circuit_breaker.ensure_breaker(
            name, failure_threshold=99, cooldown_seconds=99,
        )
        # First wins.
        assert b.failure_threshold == 1
        assert b.cooldown_seconds == 3600

        del circuit_breaker._breakers[name]


# ── Client integration ─────────────────────────────────────────────


def _make_client(server_name: str = "test-server"):
    """Build an MCPClient bound to a stdio transport that's never
    actually used (we patch the transport methods per-test)."""
    from app.mcp.client import MCPClient, MCPServerConfig

    return MCPClient(MCPServerConfig(
        name=server_name,
        transport="stdio",
        command="echo",
        args=[],
    ))


class TestClientShortCircuitsWhenBreakerOpen:

    def test_open_breaker_skips_transport_start(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When the per-server breaker is OPEN, connect() must return
        False without starting the transport."""
        from app import circuit_breaker
        from app.mcp.client import _mcp_breaker_name

        client = _make_client("test-srv-open")

        # Track whether transport.start gets called.
        started = []
        original_start = client._transport.start
        def _track_start(*a, **kw):
            started.append(True)
            return original_start(*a, **kw)
        monkeypatch.setattr(
            client._transport, "start", _track_start,
        )

        # Trip the per-server breaker.
        circuit_breaker.ensure_breaker(
            _mcp_breaker_name("test-srv-open"),
            failure_threshold=1, cooldown_seconds=3600,
        )
        circuit_breaker.record_failure(_mcp_breaker_name("test-srv-open"))
        assert not circuit_breaker.is_available(
            _mcp_breaker_name("test-srv-open")
        )

        result = client.connect()
        assert result is False
        assert started == [], (
            "OPEN breaker must skip transport.start; got "
            f"{len(started)} call(s)"
        )


class TestAuthErrorTripsBreakerAndLogsInfo:

    def test_auth_error_during_start_trips_breaker(
        self, monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import logging
        from app import circuit_breaker
        from app.mcp.client import _mcp_breaker_name

        client = _make_client("test-srv-auth")

        # Mock transport.start to raise an auth-shaped error.
        def _start():
            raise RuntimeError('HTTP 401: {"error":"invalid_token"}')

        monkeypatch.setattr(client._transport, "start", _start)

        with caplog.at_level(logging.INFO, logger="app.mcp.client"):
            result = client.connect()

        assert result is False
        # Breaker should be OPEN now.
        assert not circuit_breaker.is_available(
            _mcp_breaker_name("test-srv-auth")
        )

        # client-side log should be INFO (the breaker provides
        # the operator-visible WARN once).
        ours = [
            r for r in caplog.records
            if r.name == "app.mcp.client"
            and "auth-" in r.message
            and "breaker tripped" in r.message
        ]
        assert len(ours) == 1, (
            f"expected 1 INFO from mcp_client; got {len(ours)}: "
            f"{[(r.levelname, r.message) for r in ours]}"
        )
        assert ours[0].levelno == logging.INFO


class TestTransientErrorStillWarns:

    def test_transient_error_warns_and_does_not_trip(
        self, monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import logging
        from app import circuit_breaker
        from app.mcp.client import _mcp_breaker_name

        client = _make_client("test-srv-transient")

        def _start():
            raise ConnectionRefusedError("[Errno 111] Connection refused")

        monkeypatch.setattr(client._transport, "start", _start)

        # Pre-condition: per-server breaker not yet created/opened.
        # Use ensure_breaker to set proper config.
        circuit_breaker.ensure_breaker(
            _mcp_breaker_name("test-srv-transient"),
            failure_threshold=1, cooldown_seconds=3600,
        )
        assert circuit_breaker.is_available(
            _mcp_breaker_name("test-srv-transient")
        )

        with caplog.at_level(
            logging.WARNING, logger="app.mcp.client",
        ):
            result = client.connect()

        assert result is False
        # Breaker must remain CLOSED — transient errors are not
        # auth, so we want retry.
        assert circuit_breaker.is_available(
            _mcp_breaker_name("test-srv-transient")
        ), "transient error must NOT trip auth breaker"

        warns = [
            r for r in caplog.records
            if r.name == "app.mcp.client"
            and "start failed" in r.message
            and r.levelno == logging.WARNING
        ]
        assert len(warns) == 1, (
            f"expected 1 WARN for transient error; got {len(warns)}"
        )
