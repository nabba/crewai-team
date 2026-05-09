"""Tests for ``app.healing.monitors.signal_heartbeat``."""
from __future__ import annotations

import time

import pytest


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    from app.life_companion import _common
    from app.healing.monitors import signal_heartbeat

    monkeypatch.setattr(_common, "_STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(signal_heartbeat, "background_enabled", lambda: True)

    sent_signal: list[str] = []
    sent_pwa: list[str] = []
    sent_email: list[str] = []
    monkeypatch.setattr(signal_heartbeat, "send_signal_alert",
                         lambda body, **kw: sent_signal.append(body) or True)
    monkeypatch.setattr(signal_heartbeat, "_send_pwa_alert",
                         lambda body: sent_pwa.append(body) or True)
    monkeypatch.setattr(signal_heartbeat, "_send_email_alert",
                         lambda body: sent_email.append(body) or True)
    monkeypatch.setattr(signal_heartbeat, "audit_event", lambda *a, **k: None)

    yield tmp_path, sent_signal, sent_pwa, sent_email


def test_healthy_no_alert(isolated, monkeypatch):
    tmp, signal_, pwa, email = isolated
    from app.healing.monitors import signal_heartbeat

    monkeypatch.setattr(signal_heartbeat, "_probe_signal_health", lambda: {
        "now": time.time(),
        "convo_age_s": 60,           # recent inbound
        "outbound_age_s": 60,        # recent outbound
        "convo_db_exists": True,
        "outbound_state_exists": True,
    })

    signal_heartbeat.run()
    assert signal_ == []
    assert pwa == []
    assert email == []


def test_asymmetric_traffic_triggers_signal_alert(isolated, monkeypatch):
    tmp, signal_, pwa, email = isolated
    from app.healing.monitors import signal_heartbeat

    monkeypatch.setattr(signal_heartbeat, "_probe_signal_health", lambda: {
        "now": time.time(),
        "convo_age_s": 60,                       # got recent inbound
        "outbound_age_s": 10 * 24 * 3600,        # outbound 10d stale
        "convo_db_exists": True,
        "outbound_state_exists": True,
    })

    signal_heartbeat.run()
    assert signal_  # at least one Signal alert
    assert pwa == []  # below PWA threshold (1 fail)


def test_pwa_fallback_at_three_fails(isolated, monkeypatch):
    tmp, signal_, pwa, email = isolated
    from app.healing.monitors import signal_heartbeat
    from app.life_companion._common import write_state_json

    monkeypatch.setattr(signal_heartbeat, "_probe_signal_health", lambda: {
        "now": time.time(),
        "convo_age_s": 60,
        "outbound_age_s": 10 * 24 * 3600,
        "convo_db_exists": True,
        "outbound_state_exists": True,
    })

    # Pre-seed 2 prior fails; 3rd fail triggers PWA escalation.
    write_state_json(signal_heartbeat._STATE_FILE, {
        "last_run_at": 0.0,
        "consecutive_fails": 2,
        "last_alert_level": "",
    })

    signal_heartbeat.run()
    assert pwa  # PWA fired


def test_email_fallback_at_seven_fails(isolated, monkeypatch):
    tmp, signal_, pwa, email = isolated
    from app.healing.monitors import signal_heartbeat
    from app.life_companion._common import write_state_json

    monkeypatch.setattr(signal_heartbeat, "_probe_signal_health", lambda: {
        "now": time.time(),
        "convo_age_s": 60,
        "outbound_age_s": 10 * 24 * 3600,
        "convo_db_exists": True,
        "outbound_state_exists": True,
    })
    write_state_json(signal_heartbeat._STATE_FILE, {
        "last_run_at": 0.0,
        "consecutive_fails": 6,
        "last_alert_level": "pwa",
    })

    signal_heartbeat.run()
    assert email  # email fired


def test_recovery_resets_streak(isolated, monkeypatch):
    tmp, signal_, pwa, email = isolated
    from app.healing.monitors import signal_heartbeat
    from app.life_companion._common import (
        write_state_json, read_state_json,
    )

    monkeypatch.setattr(signal_heartbeat, "_probe_signal_health", lambda: {
        "now": time.time(),
        "convo_age_s": 60,
        "outbound_age_s": 60,        # healthy
        "convo_db_exists": True,
        "outbound_state_exists": True,
    })
    write_state_json(signal_heartbeat._STATE_FILE, {
        "last_run_at": 0.0,
        "consecutive_fails": 5,
        "last_alert_level": "pwa",
    })

    signal_heartbeat.run()
    state = read_state_json(signal_heartbeat._STATE_FILE)
    assert state["consecutive_fails"] == 0
    assert state["last_alert_level"] == ""


def test_cadence_skips_under_window(isolated, monkeypatch):
    tmp, signal_, pwa, email = isolated
    from app.healing.monitors import signal_heartbeat
    from app.life_companion._common import write_state_json

    write_state_json(signal_heartbeat._STATE_FILE, {"last_run_at": time.time()})

    # If we ran, _probe_signal_health would be called — patching it to
    # raise lets us verify we DIDN'T run.
    def _explode():
        raise RuntimeError("should not be called")

    monkeypatch.setattr(signal_heartbeat, "_probe_signal_health", _explode)
    signal_heartbeat.run()  # cadence guard prevents the explode
