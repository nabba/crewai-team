"""Tests for ``app.life_companion.calendar_prep`` (Phase B #2, 2026-05-09)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    from app.life_companion import calendar_prep
    from app.life_companion import _common

    monkeypatch.setattr(_common, "_STATE_DIR", tmp_path / "lc")
    monkeypatch.setattr(calendar_prep, "background_enabled", lambda: True)
    monkeypatch.setattr(calendar_prep, "audit_event", lambda *a, **k: None)

    sent: list[tuple[str, str]] = []
    def fake_alert(body, tag=None, **kw):
        sent.append((body, tag or ""))
        return True
    monkeypatch.setattr(calendar_prep, "send_signal_alert", fake_alert)

    # Always disable enrichment in tests — those reach to mem0/inbox
    # which we don't want network in unit tests.
    monkeypatch.setattr(calendar_prep, "_recent_inbox_from",
                        lambda *a, **k: [])
    monkeypatch.setattr(calendar_prep, "_mem0_facts_about",
                        lambda *a, **k: [])
    yield tmp_path, sent


def test_run_sends_one_per_event_in_window(isolated, monkeypatch):
    tmp_path, sent = isolated
    from app.life_companion import calendar_prep

    now = datetime.now(timezone.utc)
    events = [
        {
            "id": "ev1",
            "summary": "Team standup",
            "start_dt": now + timedelta(minutes=30),
            "location": "https://meet.google.com/xyz",
            "attendees": ["alice@example.com"],
            "description": "Weekly sync.",
            "html_link": "https://calendar.google.com/event?eid=ev1",
        },
    ]
    monkeypatch.setattr(
        calendar_prep, "_list_upcoming_events_with_description",
        lambda lo, hi: events,
    )

    summary = calendar_prep.run()
    assert summary["ran"] is True
    assert summary["events_in_window"] == 1
    assert summary["sent"] == 1
    body, tag = sent[0]
    assert "Team standup" in body
    assert "alice@example.com" in body
    assert tag == "calendar_prep:ev1"


def test_dedup_does_not_re_send(isolated, monkeypatch):
    tmp_path, sent = isolated
    from app.life_companion import calendar_prep

    now = datetime.now(timezone.utc)
    event = {
        "id": "ev2", "summary": "x",
        "start_dt": now + timedelta(minutes=30),
        "location": "", "attendees": [], "description": "", "html_link": "",
    }
    monkeypatch.setattr(
        calendar_prep, "_list_upcoming_events_with_description",
        lambda lo, hi: [event],
    )

    calendar_prep.run()
    # Reset cadence so the second run actually executes.
    state_path = tmp_path / "lc" / "calendar_prep.json"
    import json
    state = json.loads(state_path.read_text())
    state["last_run_at"] = 0.0
    state_path.write_text(json.dumps(state))

    calendar_prep.run()
    assert len(sent) == 1  # not re-sent


def test_format_includes_agenda_and_attendees():
    from app.life_companion import calendar_prep
    now = datetime(2026, 5, 9, 12, tzinfo=timezone.utc)
    event = {
        "id": "ev3",
        "summary": "Forest carbon review",
        "start_dt": now + timedelta(minutes=30),
        "location": "Helsinki office",
        "attendees": ["alice@plg.com", "bob@plg.com"],
        "description": "Review Q2 carbon flux estimates.",
        "html_link": "https://calendar.google.com/event?eid=ev3",
    }
    msg = calendar_prep._format_prep(event, now)
    assert "Forest carbon review" in msg
    assert "30 min" in msg
    assert "alice@plg.com" in msg
    assert "Review Q2" in msg
    assert "Helsinki office" in msg


def test_run_respects_cadence(isolated, monkeypatch):
    tmp_path, sent = isolated
    from app.life_companion import calendar_prep

    monkeypatch.setattr(
        calendar_prep, "_list_upcoming_events_with_description",
        lambda lo, hi: [],
    )
    calendar_prep.run()
    calendar_prep.run()  # within cadence → no second call observed
    assert sent == []
