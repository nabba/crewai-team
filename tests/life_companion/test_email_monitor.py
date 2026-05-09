"""Tests for app.life_companion.email_monitor."""
from __future__ import annotations

import time
from typing import Any

import pytest


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Redirect state files + stub Signal so tests don't talk to anything."""
    from app.life_companion import _common
    monkeypatch.setattr(_common, "_STATE_DIR", tmp_path)

    sent: list[str] = []
    monkeypatch.setattr(_common, "send_signal_alert",
                        lambda body, **kw: sent.append(body) or True)
    monkeypatch.setattr(_common, "audit_event", lambda *a, **k: None)
    # Default: master and feature flags ON; idle scheduler ON.
    monkeypatch.setenv("LIFE_COMPANION_ENABLED", "true")
    monkeypatch.setenv("LIFE_COMPANION_EMAIL_ENABLED", "true")
    monkeypatch.setattr(_common, "background_enabled", lambda: True)
    yield tmp_path, sent


def test_email_monitor_alerts_on_first_pass(isolated, monkeypatch):
    tmp_path, sent = isolated
    from app.life_companion import email_monitor
    from app.life_companion import _common

    # Re-bind the alert/audit indirection that lives at the module level
    # of email_monitor (it imports them eagerly).
    monkeypatch.setattr(email_monitor, "send_signal_alert",
                        lambda body, **kw: sent.append(body) or True)
    monkeypatch.setattr(email_monitor, "audit_event", lambda *a, **k: None)

    # Three unread stubs with varying urgency: only the high-score one
    # passes the threshold.
    stubs = [
        {"id": "m1", "from": "Boss <boss@example.com>",
         "subject": "URGENT: server down — please respond",
         "date": "Fri, 09 May 2026 07:00:00 +0000",
         "snippet": "we're losing customers",
         "label_ids": ["UNREAD", "INBOX"]},
        {"id": "m2", "from": "Newsletter <noreply@news.example>",
         "subject": "Sale this weekend",
         "date": "Fri, 09 May 2026 06:30:00 +0000",
         "snippet": "buy stuff",
         "label_ids": ["UNREAD", "INBOX"]},
        {"id": "m3", "from": "Andrus <andrus.raudsalu@plgmoments.com>",
         "subject": "self-test",
         "date": "Fri, 09 May 2026 06:00:00 +0000",
         "snippet": "test",
         "label_ids": ["UNREAD", "INBOX"]},
    ]
    monkeypatch.setattr(email_monitor, "_fetch_unread", lambda: stubs)

    email_monitor.run()
    # First pass: at least one alert.
    assert any("URGENT" in body for body in sent)


def test_email_monitor_dedups_same_id(isolated, monkeypatch):
    tmp_path, sent = isolated
    from app.life_companion import email_monitor

    monkeypatch.setattr(email_monitor, "send_signal_alert",
                        lambda body, **kw: sent.append(body) or True)
    monkeypatch.setattr(email_monitor, "audit_event", lambda *a, **k: None)

    stubs = [
        {"id": "m1", "from": "Important <imp@example.com>",
         "subject": "deadline today URGENT",
         "date": "Fri, 09 May 2026 07:00:00 +0000",
         "snippet": "deadline",
         "label_ids": ["UNREAD", "INBOX"]},
    ]
    monkeypatch.setattr(email_monitor, "_fetch_unread", lambda: stubs)

    # Bypass the cadence guard for the second call by resetting last_run_at.
    email_monitor.run()
    n1 = len(sent)

    from app.life_companion._common import read_state_json, write_state_json
    state = read_state_json("email_monitor.json", {})
    state["last_run_at"] = 0
    write_state_json("email_monitor.json", state)

    email_monitor.run()
    n2 = len(sent)
    # Second pass: same message id is in alerted_ids, so no new alert.
    assert n2 == n1


def test_email_monitor_respects_master_switch(isolated, monkeypatch):
    tmp_path, sent = isolated
    from app.life_companion import email_monitor

    monkeypatch.setenv("LIFE_COMPANION_ENABLED", "false")

    stubs = [
        {"id": "m1", "from": "Anyone <a@x.com>", "subject": "URGENT",
         "date": "", "snippet": "", "label_ids": ["UNREAD"]},
    ]
    monkeypatch.setattr(email_monitor, "_fetch_unread", lambda: stubs)

    email_monitor.run()
    assert sent == []


def test_email_monitor_respects_kill_switch(isolated, monkeypatch):
    tmp_path, sent = isolated
    from app.life_companion import email_monitor
    from app.life_companion import _common

    monkeypatch.setattr(email_monitor, "background_enabled", lambda: False)

    stubs = [{"id": "m1", "from": "Imp <i@x.com>",
              "subject": "URGENT now please", "date": "",
              "snippet": "", "label_ids": ["UNREAD"]}]
    monkeypatch.setattr(email_monitor, "_fetch_unread", lambda: stubs)

    email_monitor.run()
    assert sent == []


def test_email_monitor_cadence_skips_under_interval(isolated, monkeypatch):
    """If last_run_at is recent, run() should be a no-op."""
    tmp_path, sent = isolated
    from app.life_companion import email_monitor

    fetch_calls = []

    def fake_fetch():
        fetch_calls.append(1)
        return []

    monkeypatch.setattr(email_monitor, "_fetch_unread", fake_fetch)

    from app.life_companion._common import write_state_json
    write_state_json("email_monitor.json", {"alerted_ids": [], "last_run_at": time.time()})

    email_monitor.run()
    assert fetch_calls == []  # cadence guard short-circuited
