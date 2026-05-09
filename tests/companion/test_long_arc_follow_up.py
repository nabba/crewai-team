"""Tests for ``app.life_companion.long_arc_follow_up`` (Phase B #4, 2026-05-09)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    from app.life_companion import long_arc_follow_up
    from app.life_companion import _common

    monkeypatch.setattr(_common, "_STATE_DIR", tmp_path / "lc")
    monkeypatch.setattr(long_arc_follow_up, "background_enabled", lambda: True)
    monkeypatch.setattr(long_arc_follow_up, "audit_event", lambda *a, **k: None)

    sent: list[tuple[str, str]] = []
    def fake_alert(body, tag=None, **kw):
        sent.append((body, tag or ""))
        return True
    monkeypatch.setattr(long_arc_follow_up, "send_signal_alert", fake_alert)

    yield tmp_path, sent


def test_due_first_check_in_after_one_week(isolated):
    from app.life_companion import long_arc_follow_up
    now = datetime(2026, 5, 9, 12, tzinfo=timezone.utc)
    commitment = {
        "id": "c1",
        "description": "Ship Phase B",
        "venture": "self",
        "created_at": (now - timedelta(days=8)).isoformat(),
    }
    state = {"last_check_in_at": "", "deadline_nudges_sent": 0, "muted": False}
    due, reason = long_arc_follow_up._due(commitment, state, now)
    assert due is True
    assert reason == "first_check_in"


def test_not_due_first_week(isolated):
    from app.life_companion import long_arc_follow_up
    now = datetime(2026, 5, 9, 12, tzinfo=timezone.utc)
    commitment = {
        "id": "c1",
        "description": "x",
        "venture": "self",
        "created_at": (now - timedelta(days=3)).isoformat(),
    }
    state = {"last_check_in_at": "", "deadline_nudges_sent": 0, "muted": False}
    due, _ = long_arc_follow_up._due(commitment, state, now)
    assert due is False


def test_deadline_imminent(isolated):
    from app.life_companion import long_arc_follow_up
    now = datetime(2026, 5, 9, 12, tzinfo=timezone.utc)
    commitment = {
        "id": "c1",
        "description": "x",
        "venture": "self",
        "created_at": (now - timedelta(days=20)).isoformat(),
        "deadline": (now + timedelta(days=3)).isoformat(),
    }
    state = {"last_check_in_at": "", "deadline_nudges_sent": 0, "muted": False}
    due, reason = long_arc_follow_up._due(commitment, state, now)
    assert due is True
    assert reason == "deadline_imminent"


def test_post_deadline_one_nudge(isolated):
    from app.life_companion import long_arc_follow_up
    now = datetime(2026, 5, 9, 12, tzinfo=timezone.utc)
    commitment = {
        "id": "c1",
        "description": "x",
        "venture": "self",
        "created_at": (now - timedelta(days=40)).isoformat(),
        "deadline": (now - timedelta(days=2)).isoformat(),
    }
    state = {"last_check_in_at": "", "deadline_nudges_sent": 0, "muted": False}
    due, reason = long_arc_follow_up._due(commitment, state, now)
    assert due is True
    assert reason == "post_deadline"

    state2 = {"last_check_in_at": "x", "deadline_nudges_sent": 1, "muted": False}
    due2, _ = long_arc_follow_up._due(commitment, state2, now)
    assert due2 is False


def test_muted_skipped(isolated):
    from app.life_companion import long_arc_follow_up
    now = datetime(2026, 5, 9, 12, tzinfo=timezone.utc)
    commitment = {
        "id": "c1",
        "description": "x",
        "venture": "self",
        "created_at": (now - timedelta(days=60)).isoformat(),
    }
    state = {"last_check_in_at": "", "deadline_nudges_sent": 0, "muted": True}
    due, reason = long_arc_follow_up._due(commitment, state, now)
    assert due is False
    assert reason == "muted"


def test_run_sends_one_message_per_due(isolated, monkeypatch):
    tmp_path, sent = isolated
    from app.life_companion import long_arc_follow_up

    now = datetime.now(timezone.utc)
    fake_commitments = [
        {
            "id": "ship_phase_b",
            "description": "Ship Phase B",
            "venture": "self",
            "created_at": (now - timedelta(days=12)).isoformat(),
            "status": "active",
        },
        {
            "id": "fresh",
            "description": "Just started",
            "venture": "kaicart",
            "created_at": (now - timedelta(days=2)).isoformat(),
            "status": "active",
        },
    ]
    monkeypatch.setattr(long_arc_follow_up, "_load_active_commitments",
                        lambda: fake_commitments)

    summary = long_arc_follow_up.run()
    assert summary["ran"] is True
    assert summary["sent"] == 1  # only the 12-day-old commitment is due
    body, tag = sent[0]
    assert "Ship Phase B" in body
    assert tag == "long_arc:ship_phase_b"


def test_format_message_uses_correct_header(isolated):
    from app.life_companion import long_arc_follow_up
    now = datetime(2026, 5, 9, tzinfo=timezone.utc)
    commitment = {
        "description": "Plant 100 trees",
        "venture": "plg",
        "created_at": (now - timedelta(days=30)).isoformat(),
        "deadline": (now + timedelta(days=2)).isoformat(),
    }
    msg = long_arc_follow_up._format_message(commitment, "deadline_imminent", now)
    assert "deadline approaching" in msg.lower()
    assert "Plant 100 trees" in msg
    assert "plg" in msg
