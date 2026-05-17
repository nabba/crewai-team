"""Tests for inbound-queue durable-state transitions on load shedding.

Closes the historical gap where handle_task() marked the durable row
'processing' but the load-shed branch returned without ever marking it
done/failed/deferred — the row stalled in 'processing' forever (audit
2026-05-12; productization plan WP A2).
"""
import os
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.test_metrics import _FakeSettings  # noqa: E402
import app.config as config_mod  # noqa: E402

config_mod.get_settings = lambda: _FakeSettings()
config_mod.get_anthropic_api_key = lambda: "fake-key"
config_mod.get_gateway_secret = lambda: "a" * 64


# ── Conversation-store layer ────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Point the conversation store at an isolated sqlite file.

    The store uses a thread-local connection that runs migrations on
    creation. We swap DB_PATH and clear the thread-local so the next
    _get_conn() call builds a fresh schema in the tmp file.
    """
    from app import conversation_store
    import threading

    db_path = tmp_path / "conv.db"
    monkeypatch.setattr(conversation_store, "DB_PATH", db_path)
    # Replace the thread-local store so the next _get_conn builds fresh
    monkeypatch.setattr(conversation_store, "_local", threading.local())
    # Touch a connection to trigger migrations
    conversation_store._get_conn()
    return conversation_store


class TestStoreTransitions:
    """Unit-test the state machine directly."""

    def test_processing_then_done(self, store):
        qid = store.enqueue_inbound("alice", "hi", 12345, [])
        assert qid is not None
        store.mark_inbound_processing(qid)
        assert _status_of(store, qid) == "processing"
        store.mark_inbound_done(qid)
        assert _status_of(store, qid) == "done"

    def test_processing_then_failed(self, store):
        qid = store.enqueue_inbound("bob", "x", 12346, [])
        store.mark_inbound_processing(qid)
        store.mark_inbound_failed(qid, "oops")
        assert _status_of(store, qid) == "failed"

    def test_processing_then_deferred(self, store):
        qid = store.enqueue_inbound("carol", "y", 12347, [])
        store.mark_inbound_processing(qid)
        store.mark_inbound_deferred(qid, "load_shed: buffered")
        assert _status_of(store, qid) == "deferred"
        # Sanity: last_error captures the note
        row = _row(store, qid)
        assert "load_shed" in (row["last_error"] or "")


class TestStartupReplayIgnoresDeferred:
    """get_pending_inbound() must NOT replay deferred rows — the DLQ owns them."""

    def test_get_pending_inbound_skips_deferred(self, store):
        qid_q = store.enqueue_inbound("alice", "queued msg", 1, [])
        qid_d = store.enqueue_inbound("bob", "deferred msg", 2, [])
        store.mark_inbound_processing(qid_d)
        store.mark_inbound_deferred(qid_d, "load_shed: buffered")

        pending = store.get_pending_inbound()
        ids = {p["id"] for p in pending}
        assert qid_q in ids
        assert qid_d not in ids, "deferred rows must not be replayed at startup"


# ── handle_task() load-shed branch ──────────────────────────────────────────


@pytest.fixture
def handle_task_sandbox(tmp_path, monkeypatch):
    """Make handle_task() driveable without Signal / Firebase / crews."""
    from app import conversation_store
    from app import main as main_mod
    import threading

    # Isolate conversation store
    db_path = tmp_path / "ht.db"
    monkeypatch.setattr(conversation_store, "DB_PATH", db_path)
    monkeypatch.setattr(conversation_store, "_local", threading.local())
    conversation_store._get_conn()

    # Bolt the missing fields onto _FakeSettings for this test only
    fs = _FakeSettings()
    fs.load_shed_threshold = 4

    monkeypatch.setattr(
        "app.config.get_settings", lambda: fs, raising=False
    )
    # main.py also has a module-local get_settings reference
    monkeypatch.setattr(main_mod, "get_settings", lambda: fs, raising=False)

    # Silence Signal sends
    class _FakeSignal:
        async def send(self, *a, **kw):
            return None

    monkeypatch.setattr(main_mod, "signal_client", _FakeSignal(), raising=False)

    # Reset inflight counter so load-shed path is reachable
    monkeypatch.setattr(main_mod, "_inflight_tasks", 0, raising=False)

    # Bypass message dedup
    if hasattr(main_mod, "_msg_dedup"):
        monkeypatch.setattr(
            main_mod._msg_dedup, "is_dup", lambda *a, **kw: False, raising=False
        )

    return main_mod


class TestHandleTaskLoadShed:
    """The load-shed branch must mark the durable row 'deferred' or 'failed'."""

    def test_load_shed_buffered_marks_deferred(self, handle_task_sandbox, monkeypatch):
        from app import main as main_mod
        from app import conversation_store
        import asyncio

        # Force load-shed by pushing inflight above threshold
        # _FakeSettings has max_parallel_crews=3 → default shed_threshold=4
        monkeypatch.setattr(main_mod, "_inflight_tasks", 99, raising=False)

        # Make DLQ enqueue succeed
        monkeypatch.setattr(
            "app.dead_letter_inbound.enqueue",
            lambda *a, **kw: True,
        )
        monkeypatch.setattr(
            "app.dead_letter_inbound.queue_depth",
            lambda: 1,
        )

        qid = conversation_store.enqueue_inbound("alice", "drop me", 555, [])

        asyncio.run(
            main_mod.handle_task("alice", "drop me", [], 555, qid)
        )

        assert _status_of(conversation_store, qid) == "deferred"
        row = _row(conversation_store, qid)
        assert "load_shed" in (row["last_error"] or "")

    def test_load_shed_unbuffered_marks_failed(self, handle_task_sandbox, monkeypatch):
        from app import main as main_mod
        from app import conversation_store
        import asyncio

        monkeypatch.setattr(main_mod, "_inflight_tasks", 99, raising=False)

        # DLQ also full — buffered=False
        monkeypatch.setattr(
            "app.dead_letter_inbound.enqueue",
            lambda *a, **kw: False,
        )
        monkeypatch.setattr(
            "app.dead_letter_inbound.queue_depth",
            lambda: 100,
        )

        qid = conversation_store.enqueue_inbound("alice", "x", 556, [])

        asyncio.run(
            main_mod.handle_task("alice", "x", [], 556, qid)
        )

        assert _status_of(conversation_store, qid) == "failed"
        row = _row(conversation_store, qid)
        assert "queue full" in (row["last_error"] or "").lower()


class TestDLQDrainDoesNotReplayDeferred:
    """The DLQ drain owns the work; startup replay must not duplicate it."""

    def test_deferred_row_does_not_double_process(self, store):
        qid = store.enqueue_inbound("alice", "msg", 123, [])
        store.mark_inbound_processing(qid)
        store.mark_inbound_deferred(qid, "load_shed: buffered to dlq")

        pending = store.get_pending_inbound()
        assert all(p["id"] != qid for p in pending), (
            "deferred rows must be invisible to startup replay so the DLQ "
            "drain doesn't double-process them"
        )


# ── helpers ─────────────────────────────────────────────────────────────────


def _status_of(store_mod, queue_id: int) -> str:
    conn = store_mod._get_conn()
    cur = conn.execute(
        "SELECT status FROM inbound_queue WHERE id=?", (queue_id,)
    )
    row = cur.fetchone()
    assert row is not None, f"queue row {queue_id} not found"
    return row[0]


def _row(store_mod, queue_id: int) -> dict:
    conn = store_mod._get_conn()
    conn.row_factory = sqlite3.Row
    cur = conn.execute(
        "SELECT * FROM inbound_queue WHERE id=?", (queue_id,)
    )
    row = cur.fetchone()
    assert row is not None
    return dict(row)
