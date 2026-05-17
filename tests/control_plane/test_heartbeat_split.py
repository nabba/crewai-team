"""Tests for the passive vs active heartbeat split + autonomous_mode gate.

PR 2 (2026-05-16). Before this PR, ``HeartbeatScheduler.run_heartbeat``
unconditionally pulled assigned tickets and dispatched them through
Commander, regardless of the ``autonomous_mode`` setting (which was
defined in ``config.py`` but never read).

The fix:
- ``run_heartbeat()`` is now a dispatch wrapper that consults
  ``settings.autonomous_mode`` at call time.
- ``run_passive_heartbeat()`` records a telemetry beat and returns
  ``status="passive"`` without touching the ticket queue.
- ``run_active_heartbeat()`` retains the pre-PR behavior (pull queue,
  budget gate, dispatch, log).
- Reading the flag at call time means an operator can flip the
  setting without a gateway restart.

These tests pin those properties.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def heartbeat():
    from app.control_plane.heartbeats import HeartbeatScheduler
    return HeartbeatScheduler()


def _stub_settings(autonomous: bool):
    """Build a settings-like object exposing autonomous_mode."""
    s = MagicMock()
    s.autonomous_mode = autonomous
    return s


def test_run_heartbeat_routes_to_passive_when_autonomous_disabled(
    heartbeat, monkeypatch,
):
    monkeypatch.setattr(
        "app.config.get_settings", lambda: _stub_settings(False),
    )
    passive_called = {"n": 0}
    active_called = {"n": 0}
    monkeypatch.setattr(
        heartbeat, "run_passive_heartbeat",
        lambda *a, **kw: passive_called.__setitem__("n", passive_called["n"] + 1) or {"status": "passive"},
    )
    monkeypatch.setattr(
        heartbeat, "run_active_heartbeat",
        lambda *a, **kw: active_called.__setitem__("n", active_called["n"] + 1) or {"status": "active"},
    )

    result = heartbeat.run_heartbeat("researcher", "p1")

    assert passive_called["n"] == 1
    assert active_called["n"] == 0
    assert result["status"] == "passive"


def test_run_heartbeat_routes_to_active_when_autonomous_enabled(
    heartbeat, monkeypatch,
):
    monkeypatch.setattr(
        "app.config.get_settings", lambda: _stub_settings(True),
    )
    passive_called = {"n": 0}
    active_called = {"n": 0}
    monkeypatch.setattr(
        heartbeat, "run_passive_heartbeat",
        lambda *a, **kw: passive_called.__setitem__("n", passive_called["n"] + 1) or {"status": "passive"},
    )
    monkeypatch.setattr(
        heartbeat, "run_active_heartbeat",
        lambda *a, **kw: active_called.__setitem__("n", active_called["n"] + 1) or {"status": "active"},
    )

    result = heartbeat.run_heartbeat("researcher", "p1")

    assert active_called["n"] == 1
    assert passive_called["n"] == 0
    assert result["status"] == "active"


def test_run_passive_heartbeat_records_beat_and_returns_status_passive(
    heartbeat, monkeypatch,
):
    recorded = {"n": 0}

    def fake_record(role, project, trigger, **_):
        recorded["n"] += 1
        recorded["trigger"] = trigger
        recorded["role"] = role
        recorded["project"] = project

    monkeypatch.setattr(heartbeat, "record_beat", fake_record)

    result = heartbeat.run_passive_heartbeat("coder", "p7")

    assert recorded["n"] == 1
    assert recorded["trigger"] == "passive"
    assert recorded["role"] == "coder"
    assert recorded["project"] == "p7"
    assert result["status"] == "passive"
    assert result["tickets_processed"] == 0
    assert "autonomous_mode disabled" in result["reason"]


def test_run_passive_heartbeat_does_not_touch_ticket_queue(
    heartbeat, monkeypatch,
):
    """Passive heartbeat must NOT pull tickets or dispatch to Commander."""
    monkeypatch.setattr(heartbeat, "record_beat", lambda *a, **kw: None)

    db_called = {"n": 0}

    def fake_execute(*a, **kw):
        db_called["n"] += 1
        return []

    monkeypatch.setattr(
        "app.control_plane.heartbeats.execute", fake_execute,
    )

    heartbeat.run_passive_heartbeat("writer", "p1")
    assert db_called["n"] == 0


def test_autonomous_mode_read_at_call_time(heartbeat, monkeypatch):
    """Flipping autonomous_mode between calls must change routing.

    The flag is read every call (not cached) so an operator can toggle
    it via ``/cp/settings`` and the next heartbeat picks it up.
    """
    calls = {"passive": 0, "active": 0}

    monkeypatch.setattr(
        heartbeat, "run_passive_heartbeat",
        lambda *a, **kw: calls.__setitem__("passive", calls["passive"] + 1) or {"status": "passive"},
    )
    monkeypatch.setattr(
        heartbeat, "run_active_heartbeat",
        lambda *a, **kw: calls.__setitem__("active", calls["active"] + 1) or {"status": "active"},
    )

    # Start disabled
    flag = {"on": False}
    monkeypatch.setattr(
        "app.config.get_settings",
        lambda: _stub_settings(flag["on"]),
    )

    heartbeat.run_heartbeat("researcher", "p1")
    assert calls == {"passive": 1, "active": 0}

    # Operator flips the flag
    flag["on"] = True

    heartbeat.run_heartbeat("researcher", "p1")
    assert calls == {"passive": 1, "active": 1}

    # And back off again
    flag["on"] = False

    heartbeat.run_heartbeat("researcher", "p1")
    assert calls == {"passive": 2, "active": 1}


def test_autonomous_mode_falls_closed_on_settings_failure(
    heartbeat, monkeypatch,
):
    """If settings can't load, default to passive (safe-by-default)."""
    def boom():
        raise RuntimeError("settings unavailable")

    monkeypatch.setattr("app.config.get_settings", boom)

    calls = {"passive": 0, "active": 0}
    monkeypatch.setattr(
        heartbeat, "run_passive_heartbeat",
        lambda *a, **kw: calls.__setitem__("passive", calls["passive"] + 1) or {"status": "passive"},
    )
    monkeypatch.setattr(
        heartbeat, "run_active_heartbeat",
        lambda *a, **kw: calls.__setitem__("active", calls["active"] + 1) or {"status": "active"},
    )

    heartbeat.run_heartbeat("researcher", "p1")
    assert calls == {"passive": 1, "active": 0}


def test_run_passive_heartbeat_preserves_wake_events(heartbeat, monkeypatch):
    """Wakes queued via trigger_wake must NOT be consumed by passive beat.

    When autonomous_mode is later re-enabled, the active heartbeat
    should still see those wakes pending — passive mode is not allowed
    to silently drop them.
    """
    monkeypatch.setattr(heartbeat, "record_beat", lambda *a, **kw: None)

    heartbeat.trigger_wake("researcher", "ticket_assigned", ticket_id="t1")
    heartbeat.run_passive_heartbeat("researcher", "p1")

    # Wakes still pending after passive beat
    pending = heartbeat.get_pending_wakes("researcher")
    assert len(pending) == 1
    assert pending[0]["ticket_id"] == "t1"
