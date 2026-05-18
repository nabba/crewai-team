"""Tests for the signal forwarder durable outbox.

Pins the terminal-cap behavior shipped 2026-05-18 (A3 audit fix): rows
that exceed ``_MAX_ATTEMPTS`` retries are moved to the ``outbox_dead``
dead-letter table instead of being rescheduled forever. A permanently
broken endpoint would otherwise grow the outbox without bound.

The forwarder lives at top-level ``signal/forwarder.py`` (it's an
out-of-container host process), which collides with stdlib ``signal``.
We load it via importlib from path to bypass the name collision.
"""
from __future__ import annotations

import importlib.util
import sys
import types
import sqlite3
from pathlib import Path

import pytest


def _load_forwarder():
    """Load signal/forwarder.py without colliding with stdlib signal.

    The forwarder is a host-side daemon (not gateway-container) and
    imports ``requests``. In a CI / sandboxed env where ``requests``
    isn't installed we stub it: this test only exercises the SQLite
    outbox math, never makes an HTTP call.
    """
    if "requests" not in sys.modules:
        fake = types.ModuleType("requests")
        class _Session:  # minimal stub — never invoked here
            headers: dict = {}
            def post(self, *a, **kw):
                raise RuntimeError("network unavailable in test")
        class _RequestException(Exception):
            pass
        exceptions = types.ModuleType("requests.exceptions")
        exceptions.RequestException = _RequestException
        fake.Session = _Session
        fake.exceptions = exceptions
        sys.modules["requests"] = fake
        sys.modules["requests.exceptions"] = exceptions
    path = Path(__file__).parent.parent / "signal" / "forwarder.py"
    spec = importlib.util.spec_from_file_location("_test_forwarder", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_test_forwarder"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_reschedule_moves_row_to_outbox_dead_at_max_attempts(tmp_path):
    """After _MAX_ATTEMPTS, the row leaves the live outbox and lands in
    outbox_dead. Pins the A3 ship-blocker fix from 2026-05-18."""
    fwd = _load_forwarder()
    fwd.OUTBOX_DB = str(tmp_path / "outbox.sqlite")

    conn = fwd._outbox_conn()
    try:
        conn.execute(
            "INSERT INTO outbox (kind, payload, attempts, next_attempt_at, "
            "created_at, last_error) VALUES (?, ?, ?, ?, ?, ?)",
            ("inbound", '{"hello":"world"}', fwd._MAX_ATTEMPTS - 1,
             0.0, 0.0, "prior error"),
        )
        row_id = conn.execute("SELECT id FROM outbox").fetchone()[0]
        # Next failure crosses the threshold (attempts + 1 == _MAX_ATTEMPTS).
        fwd._reschedule(
            conn, row_id, fwd._MAX_ATTEMPTS - 1, "still broken",
        )
        live = conn.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
        dead = conn.execute("SELECT COUNT(*) FROM outbox_dead").fetchone()[0]
        assert live == 0
        assert dead == 1
        # The dead row preserves payload + attempts + error for inspection.
        dead_row = conn.execute(
            "SELECT kind, payload, attempts, last_error FROM outbox_dead"
        ).fetchone()
        assert dead_row[0] == "inbound"
        assert dead_row[1] == '{"hello":"world"}'
        assert dead_row[2] == fwd._MAX_ATTEMPTS
        assert dead_row[3] == "still broken"
    finally:
        conn.close()


def test_reschedule_below_threshold_stays_in_outbox(tmp_path):
    """Routine retries (well below _MAX_ATTEMPTS) stay in the live outbox
    with their attempts counter bumped. Guards against regressing the
    happy-path retry behavior alongside the new terminal cap."""
    fwd = _load_forwarder()
    fwd.OUTBOX_DB = str(tmp_path / "outbox.sqlite")

    conn = fwd._outbox_conn()
    try:
        conn.execute(
            "INSERT INTO outbox (kind, payload, attempts, next_attempt_at, "
            "created_at, last_error) VALUES (?, ?, ?, ?, ?, ?)",
            ("inbound", '{"hi":"there"}', 0, 0.0, 0.0, None),
        )
        row_id = conn.execute("SELECT id FROM outbox").fetchone()[0]
        fwd._reschedule(conn, row_id, 0, "transient")
        live = conn.execute(
            "SELECT attempts, last_error FROM outbox"
        ).fetchone()
        dead = conn.execute("SELECT COUNT(*) FROM outbox_dead").fetchone()[0]
        assert live[0] == 1
        assert live[1] == "transient"
        assert dead == 0
    finally:
        conn.close()
