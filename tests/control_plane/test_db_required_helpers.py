"""Tests for the required/optional DB helper split + pool diagnostics.

PR 1 (2026-05-16). Before this PR, every public helper in
``app.control_plane.db`` swallowed all failures and returned ``None``,
so a missing table looked exactly like "query returned no rows". The
required family (``execute_required`` / ``execute_one_required`` /
``execute_scalar_required``) now raises on failure.

These tests pin three properties:

1. The required and optional families are TRULY split — required
   raises where optional returns None, for every failure kind
   (pool unavailable, pool exhausted, SQL error).
2. Pool diagnostics counters increment on the right events
   (acquire/release/exhaust) and are observational only.
3. Both families share the same private inner loop so they cannot
   drift apart on connection-state quirks (autocommit retry, etc.).

Import-isolation note: psycopg2 is imported at module-top so the
real classes are captured at collection time. Other test modules
in the suite use ``patch.dict("sys.modules", ...)`` which can
replace psycopg2 with a MagicMock if pytest collects this file
after them — capturing references here defends against that.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import psycopg2
import pytest
from psycopg2 import pool as pg_pool


@pytest.fixture(autouse=True)
def _reset_db_state(monkeypatch):
    """Reset module-level singletons so tests don't bleed into each other."""
    from app.control_plane import db
    monkeypatch.setattr(db, "_pool", None, raising=False)
    monkeypatch.setattr(db, "_breaker_open_until", 0.0, raising=False)
    monkeypatch.setattr(db, "_breaker_alert_sent", False, raising=False)
    db.reset_pool_diagnostics()
    yield
    db.reset_pool_diagnostics()


def test_dbunavailable_raised_when_pool_is_none(monkeypatch):
    from app.control_plane import db
    monkeypatch.setattr(db, "get_pool", lambda: None)
    with pytest.raises(db.DBUnavailable):
        db.execute_required("SELECT 1")
    diag = db.get_pool_diagnostics()
    assert diag["failures_pool_unavailable"] == 1


def test_execute_returns_none_when_pool_is_none(monkeypatch):
    from app.control_plane import db
    monkeypatch.setattr(db, "get_pool", lambda: None)
    assert db.execute("SELECT 1") is None
    assert db.execute_one("SELECT 1") is None
    assert db.execute_scalar("SELECT 1") is None


def _fake_pool(rows: list | None = None, raise_at: str | None = None):
    """Build a MagicMock pool that mimics psycopg2.ThreadedConnectionPool.

    ``raise_at`` controls where a psycopg2 error is raised:
      - "getconn" → pool.getconn() raises pg_pool.PoolError("exhausted")
      - "sql"     → cur.execute() raises psycopg2.Error
      - None      → success path
    """
    pool = MagicMock()
    if raise_at == "getconn":
        pool.getconn.side_effect = pg_pool.PoolError("connection pool exhausted")
        return pool

    # Build a successful connection + cursor
    cur = MagicMock()
    if raise_at == "sql":
        cur.execute.side_effect = psycopg2.Error("relation \"x\" does not exist")
    else:
        cur.description = [("a",), ("b",)] if rows else []
        cur.fetchall.return_value = [
            tuple(r.values()) if isinstance(r, dict) else r for r in (rows or [])
        ]

    cur_ctx = MagicMock()
    cur_ctx.__enter__.return_value = cur
    cur_ctx.__exit__.return_value = False
    conn = MagicMock()
    conn.cursor.return_value = cur_ctx
    pool.getconn.return_value = conn
    return pool


def test_execute_required_raises_on_sql_error(monkeypatch):
    from app.control_plane import db
    pool = _fake_pool(raise_at="sql")
    monkeypatch.setattr(db, "get_pool", lambda: pool)
    with pytest.raises(psycopg2.Error):
        db.execute_required("SELECT * FROM bogus")


def test_execute_returns_none_on_sql_error(monkeypatch):
    from app.control_plane import db
    pool = _fake_pool(raise_at="sql")
    monkeypatch.setattr(db, "get_pool", lambda: pool)
    assert db.execute("SELECT * FROM bogus") is None


def test_execute_required_raises_on_pool_exhausted(monkeypatch):
    from app.control_plane import db
    pool = _fake_pool(raise_at="getconn")
    monkeypatch.setattr(db, "get_pool", lambda: pool)
    with pytest.raises(pg_pool.PoolError):
        db.execute_required("SELECT 1")
    diag = db.get_pool_diagnostics()
    assert diag["failures_pool_exhausted"] == 1
    # last_exhaust_ts must be populated when this fires
    assert diag["last_exhaust_ts"] > 0


def test_execute_returns_none_on_pool_exhausted(monkeypatch):
    from app.control_plane import db
    pool = _fake_pool(raise_at="getconn")
    monkeypatch.setattr(db, "get_pool", lambda: pool)
    assert db.execute("SELECT 1") is None
    diag = db.get_pool_diagnostics()
    assert diag["failures_pool_exhausted"] == 1


def test_diagnostics_track_borrows(monkeypatch):
    """current_borrows increments at acquire and decrements at release."""
    from app.control_plane import db
    pool = _fake_pool(rows=[])
    monkeypatch.setattr(db, "get_pool", lambda: pool)
    db.execute_required("SELECT 1")
    diag = db.get_pool_diagnostics()
    assert diag["acquires_total"] == 1
    # Borrow accounting is balanced — no leak per call
    assert diag["current_borrows"] == 0
    assert diag["peak_borrows"] == 1


def test_diagnostics_snapshot_is_immutable(monkeypatch):
    """get_pool_diagnostics returns a copy, not the live dict."""
    from app.control_plane import db
    snap = db.get_pool_diagnostics()
    snap["acquires_total"] = 99
    snap2 = db.get_pool_diagnostics()
    assert snap2["acquires_total"] == 0


def test_execute_required_fetch_true_returns_dict_rows(monkeypatch):
    from app.control_plane import db
    pool = _fake_pool(rows=[{"a": 1, "b": 2}, {"a": 3, "b": 4}])
    monkeypatch.setattr(db, "get_pool", lambda: pool)
    rows = db.execute_required("SELECT a, b FROM t", fetch=True)
    assert rows == [{"a": 1, "b": 2}, {"a": 3, "b": 4}]


def test_execute_required_fetch_false_returns_empty_list(monkeypatch):
    from app.control_plane import db
    pool = _fake_pool(rows=[])
    monkeypatch.setattr(db, "get_pool", lambda: pool)
    result = db.execute_required("INSERT INTO t VALUES (1)")
    assert result == []


def test_execute_one_required_returns_first_row(monkeypatch):
    from app.control_plane import db
    pool = _fake_pool(rows=[{"a": 1, "b": 2}, {"a": 3, "b": 4}])
    monkeypatch.setattr(db, "get_pool", lambda: pool)
    row = db.execute_one_required("SELECT a, b FROM t LIMIT 1")
    assert row == {"a": 1, "b": 2}


def test_execute_one_required_returns_none_on_empty(monkeypatch):
    from app.control_plane import db
    pool = _fake_pool(rows=[])
    monkeypatch.setattr(db, "get_pool", lambda: pool)
    assert db.execute_one_required("SELECT * FROM empty") is None


def test_execute_scalar_required_returns_first_column(monkeypatch):
    from app.control_plane import db
    pool = _fake_pool(rows=[{"count": 42}])
    monkeypatch.setattr(db, "get_pool", lambda: pool)
    assert db.execute_scalar_required("SELECT count(*) FROM t") == 42


def test_execute_scalar_required_raises_dbunavailable(monkeypatch):
    from app.control_plane import db
    monkeypatch.setattr(db, "get_pool", lambda: None)
    with pytest.raises(db.DBUnavailable):
        db.execute_scalar_required("SELECT 1")


def test_required_and_optional_share_same_inner_path(monkeypatch):
    """Both families call the same private helpers, so a per-row schema
    quirk (autocommit retry, dict cursor format) can't drift between
    them. We assert this structurally by verifying both call _checkout
    and _run_query."""
    from app.control_plane import db
    pool = _fake_pool(rows=[{"a": 1}])
    monkeypatch.setattr(db, "get_pool", lambda: pool)

    calls = {"checkout": 0, "run_query": 0}
    orig_checkout = db._checkout
    orig_run_query = db._run_query

    def spy_checkout(p):
        calls["checkout"] += 1
        return orig_checkout(p)

    def spy_run_query(conn, q, p, f):
        calls["run_query"] += 1
        return orig_run_query(conn, q, p, f)

    monkeypatch.setattr(db, "_checkout", spy_checkout)
    monkeypatch.setattr(db, "_run_query", spy_run_query)

    db.execute_required("SELECT a FROM t", fetch=True)
    db.execute("SELECT a FROM t", fetch=True)
    assert calls["checkout"] == 2
    assert calls["run_query"] == 2
