"""Tests for ``app.healing.monitors.db_vacuum`` + the underlying
``app.conversation_store.vacuum`` function (Wave 0/1 #A6)."""
from __future__ import annotations

import sqlite3
import time

import pytest


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    """Repoint conversation_store at a fresh DB inside tmp_path."""
    import app.conversation_store as cs
    db = tmp_path / "conversations.db"
    monkeypatch.setattr(cs, "DB_PATH", db)
    # Reset the cached connection so _get_conn uses the new path.
    if hasattr(cs, "_conn"):
        monkeypatch.setattr(cs, "_conn", None, raising=False)
    yield tmp_path, db


def test_vacuum_runs_on_empty_db(isolated_store):
    tmp_path, db_path = isolated_store
    from app import conversation_store

    summary = conversation_store.vacuum()
    assert summary["ok"] is True
    assert summary["bytes_after"] >= 0
    assert summary["duration_s"] >= 0


def test_vacuum_returns_size_summary(isolated_store):
    tmp_path, db_path = isolated_store
    from app import conversation_store

    # Force schema init by calling _get_conn().
    conversation_store._get_conn()
    # Insert some data.
    for i in range(50):
        conversation_store.add_message(f"sender{i}", "user", f"hello {i}")

    summary = conversation_store.vacuum()
    assert summary["ok"] is True
    # Must report freed_bytes >= 0; can't depend on exact byte values
    # since SQLite layout has overhead.
    assert summary["freed_bytes"] >= 0


def test_vacuum_swallows_errors(isolated_store, monkeypatch):
    tmp_path, db_path = isolated_store
    from app import conversation_store

    def raise_oserror(*a, **kw):
        raise OSError("disk full")
    monkeypatch.setattr(conversation_store, "_get_conn", raise_oserror)
    summary = conversation_store.vacuum()
    # Returns the failed-default summary instead of raising.
    assert summary["ok"] is False


# ── Monitor cadence-guard tests ───────────────────────────────────────────


@pytest.fixture
def isolated_monitor(tmp_path, monkeypatch):
    from app.healing.monitors import db_vacuum
    from app.life_companion import _common
    import app.conversation_store as cs

    monkeypatch.setattr(_common, "_STATE_DIR", tmp_path / "lc")
    monkeypatch.setattr(db_vacuum, "background_enabled", lambda: True)

    sent: list[str] = []
    monkeypatch.setattr(db_vacuum, "send_signal_alert",
                        lambda body, **kw: sent.append(body) or True)
    monkeypatch.setattr(db_vacuum, "audit_event", lambda *a, **k: None)

    runs: list[dict] = []
    def stub_vacuum():
        result = {"ok": True, "bytes_before": 200_000_000,
                  "bytes_after": 50_000_000, "freed_bytes": 150_000_000,
                  "duration_s": 0.1}
        runs.append(result)
        return result
    # The monitor calls `from app.conversation_store import vacuum` inside
    # run(), so patch the symbol on that module.
    monkeypatch.setattr(cs, "vacuum", stub_vacuum)

    yield tmp_path, sent, runs


def test_monitor_first_run_does_vacuum(isolated_monitor):
    tmp_path, sent, runs = isolated_monitor
    from app.healing.monitors import db_vacuum
    db_vacuum.run()
    assert len(runs) == 1


def test_monitor_second_run_within_cadence_skipped(isolated_monitor):
    tmp_path, sent, runs = isolated_monitor
    from app.healing.monitors import db_vacuum
    db_vacuum.run()
    db_vacuum.run()  # second call within cadence
    assert len(runs) == 1


def test_alerts_when_freed_above_threshold(isolated_monitor):
    """Monitor alerts when VACUUM frees more than 50 MB — surface to operator."""
    tmp_path, sent, runs = isolated_monitor
    from app.healing.monitors import db_vacuum
    db_vacuum.run()
    assert any("freed" in s.lower() for s in sent)
