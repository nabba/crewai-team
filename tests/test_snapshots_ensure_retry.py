"""Tests for the snapshots._ensure_table retry behavior.

PR 1 (2026-05-16). Before this PR, ``PostgresSnapshotStore._ensure_table``
called ``execute()`` (optional semantics — swallows exceptions and
returns None). When the CREATE TABLE silently failed (pool unavailable,
permission denied, SQL error against half-migrated schema), the call
still reached the ``_ensured = True`` line because no exception was
raised. Subsequent put/latest/recent calls all silently failed against
a never-created table.

The fix switches ``_ensure_table`` to ``execute_required``, which RAISES
on failure. The except branch catches it without setting ``_ensured``,
so the next ``put()`` retries.

These tests pin the retry behavior: a failed create leaves
``_ensured=False``, a subsequent successful create flips it to True.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def reset_ensured():
    """Reset the class-level _ensured flag between tests."""
    from app.observability.snapshots import PostgresSnapshotStore
    PostgresSnapshotStore._ensured = False
    yield
    PostgresSnapshotStore._ensured = False


def test_ensure_table_failure_does_not_set_ensured(monkeypatch, reset_ensured):
    """A raised exception from execute_required must leave _ensured=False."""
    from app.observability.snapshots import PostgresSnapshotStore
    from app.control_plane import db

    def boom(*a, **kw):
        raise db.DBUnavailable("pool not configured")

    monkeypatch.setattr(db, "execute_required", boom)

    store = PostgresSnapshotStore()
    store._ensure_table()

    assert PostgresSnapshotStore._ensured is False


def test_ensure_table_retries_after_failure(monkeypatch, reset_ensured):
    """If the first attempt fails, the next call MUST retry (not short-circuit)."""
    from app.observability.snapshots import PostgresSnapshotStore
    from app.control_plane import db

    attempts = {"n": 0}

    def maybe_raise(*a, **kw):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise db.DBUnavailable("pool not configured")
        return []  # success

    monkeypatch.setattr(db, "execute_required", maybe_raise)

    store = PostgresSnapshotStore()
    store._ensure_table()
    assert PostgresSnapshotStore._ensured is False
    assert attempts["n"] == 1

    store._ensure_table()
    assert PostgresSnapshotStore._ensured is True
    assert attempts["n"] == 2


def test_ensure_table_success_path_sets_ensured(monkeypatch, reset_ensured):
    """Happy path: execute_required returns normally → _ensured=True."""
    from app.observability.snapshots import PostgresSnapshotStore
    from app.control_plane import db

    calls = {"n": 0}

    def ok(*a, **kw):
        calls["n"] += 1
        return []

    monkeypatch.setattr(db, "execute_required", ok)

    store = PostgresSnapshotStore()
    store._ensure_table()
    assert PostgresSnapshotStore._ensured is True
    assert calls["n"] == 1

    # Second call should be a no-op (early return on _ensured=True)
    store._ensure_table()
    assert calls["n"] == 1


def test_ensure_table_treats_sql_error_as_retryable(monkeypatch, reset_ensured):
    """A real SQL error during CREATE TABLE should also leave _ensured=False."""
    import psycopg2
    from app.observability.snapshots import PostgresSnapshotStore
    from app.control_plane import db

    def boom(*a, **kw):
        raise psycopg2.Error("permission denied for schema public")

    monkeypatch.setattr(db, "execute_required", boom)

    store = PostgresSnapshotStore()
    store._ensure_table()
    assert PostgresSnapshotStore._ensured is False
