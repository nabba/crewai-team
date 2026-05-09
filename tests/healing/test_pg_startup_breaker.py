"""Tests for control_plane PG startup circuit breaker (Phase D #1)."""
from __future__ import annotations

import time

import pytest


@pytest.fixture(autouse=True)
def reset_module_state(monkeypatch):
    """Reset the module-level pool + breaker state between tests."""
    from app.control_plane import db
    monkeypatch.setattr(db, "_pool", None, raising=False)
    monkeypatch.setattr(db, "_breaker_open_until", 0.0, raising=False)
    monkeypatch.setattr(db, "_breaker_alert_sent", False, raising=False)
    yield


def test_dsn_with_timeout_appends_when_missing():
    from app.control_plane.db import _dsn_with_timeout
    out = _dsn_with_timeout("postgresql://u:p@host/db")
    assert "connect_timeout=" in out


def test_dsn_with_timeout_idempotent():
    from app.control_plane.db import _dsn_with_timeout
    in_ = "postgresql://u:p@host/db?connect_timeout=42"
    out = _dsn_with_timeout(in_)
    assert out == in_


def test_breaker_opens_after_consecutive_failures(monkeypatch):
    from app.control_plane import db
    import psycopg2

    class _S:
        mem0_postgres_url = "postgresql://u:p@nonexistent/x"
    monkeypatch.setattr("app.config.get_settings", lambda: _S())

    # Make the pool constructor raise immediately, no real network.
    raised = []
    def boom(*a, **kw):
        raised.append(1)
        raise psycopg2.OperationalError("no such host")
    monkeypatch.setattr(db.pg_pool, "ThreadedConnectionPool", boom)
    # Skip the backoff sleeps so the test runs in <1 ms.
    monkeypatch.setattr(time, "sleep", lambda *_: None)

    # First call: 3 attempts → all fail → breaker opens.
    assert db.get_pool() is None
    assert len(raised) == 3
    assert db._circuit_open() is True
    # Second call: breaker open → no further attempts.
    assert db.get_pool() is None
    assert len(raised) == 3


def test_breaker_signal_alert_fires_once(monkeypatch):
    from app.control_plane import db
    import psycopg2

    class _S:
        mem0_postgres_url = "postgresql://u:p@nonexistent/x"
    monkeypatch.setattr("app.config.get_settings", lambda: _S())
    monkeypatch.setattr(
        db.pg_pool, "ThreadedConnectionPool",
        lambda *a, **kw: (_ for _ in ()).throw(psycopg2.OperationalError("x"))
    )
    monkeypatch.setattr(time, "sleep", lambda *_: None)

    sent: list[str] = []
    monkeypatch.setattr(
        "app.healing.handlers._common.send_signal_alert",
        lambda body, **kw: sent.append(body) or True,
    )
    db.get_pool()
    db.get_pool()
    db.get_pool()
    # Only one alert per breaker-open episode.
    assert len(sent) == 1
    assert "breaker open" in sent[0].lower()


def test_breaker_resets_on_success(monkeypatch):
    from app.control_plane import db

    # Pre-OPEN the breaker.
    db._open_breaker("forced for test")
    assert db._circuit_open() is True

    # Now allow the pool to be created. Use a stub that returns a
    # marker object so we can verify it's stored.
    class _FakePool:
        pass
    monkeypatch.setattr(db.pg_pool, "ThreadedConnectionPool",
                        lambda *a, **kw: _FakePool())
    class _S:
        mem0_postgres_url = "postgresql://u:p@host/x"
    monkeypatch.setattr("app.config.get_settings", lambda: _S())
    # Force the breaker timer to "expired" so retry kicks in.
    db._breaker_open_until = 0.0

    pool = db.get_pool()
    assert pool is not None
    assert db._circuit_open() is False


def test_no_postgres_url_returns_none(monkeypatch):
    from app.control_plane import db
    class _S:
        mem0_postgres_url = ""
    monkeypatch.setattr("app.config.get_settings", lambda: _S())
    assert db.get_pool() is None
