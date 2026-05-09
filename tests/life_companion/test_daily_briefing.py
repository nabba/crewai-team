"""Tests for app.life_companion.daily_briefing."""
from __future__ import annotations

from datetime import datetime

import pytest


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    from app.life_companion import _common
    monkeypatch.setattr(_common, "_STATE_DIR", tmp_path)

    sent: list[str] = []
    monkeypatch.setattr(_common, "send_signal_alert",
                        lambda body, **kw: sent.append(body) or True)
    monkeypatch.setattr(_common, "audit_event", lambda *a, **k: None)
    monkeypatch.setenv("LIFE_COMPANION_ENABLED", "true")
    monkeypatch.setenv("LIFE_COMPANION_BRIEFING_ENABLED", "true")
    monkeypatch.setattr(_common, "background_enabled", lambda: True)
    yield tmp_path, sent


def test_briefing_morning_window_fires(isolated, monkeypatch):
    tmp_path, sent = isolated
    from app.life_companion import daily_briefing

    monkeypatch.setattr(daily_briefing, "send_signal_alert",
                        lambda body, **kw: sent.append(body) or True)
    monkeypatch.setattr(daily_briefing, "audit_event", lambda *a, **k: None)

    # Stub the data collectors so the test doesn't need Gmail / Calendar / DB.
    monkeypatch.setattr(daily_briefing, "_gather_calendar_24h",
                        lambda: ["  • 09:00 — standup"])
    monkeypatch.setattr(daily_briefing, "_gather_top_emails",
                        lambda n=3: ["  • [3.5] Boss: PRs to review"])
    monkeypatch.setattr(daily_briefing, "_gather_open_tickets",
                        lambda n=5: ["  • [in_progress] migrate auth (PLG)"])

    # Force the wall clock to land in the morning window.
    fake_now = datetime(2026, 5, 9, 7, 5)  # 07:05 — within 07:00 window
    monkeypatch.setattr(daily_briefing, "_now_local", lambda: fake_now)

    daily_briefing.run()
    assert any("Morning briefing" in body for body in sent)
    assert any("standup" in body for body in sent)


def test_briefing_idempotent_within_window(isolated, monkeypatch):
    tmp_path, sent = isolated
    from app.life_companion import daily_briefing

    monkeypatch.setattr(daily_briefing, "send_signal_alert",
                        lambda body, **kw: sent.append(body) or True)
    monkeypatch.setattr(daily_briefing, "audit_event", lambda *a, **k: None)
    monkeypatch.setattr(daily_briefing, "_gather_calendar_24h", lambda: [])
    monkeypatch.setattr(daily_briefing, "_gather_top_emails", lambda n=3: [])
    monkeypatch.setattr(daily_briefing, "_gather_open_tickets", lambda n=5: [])

    fake_now = datetime(2026, 5, 9, 7, 0)
    monkeypatch.setattr(daily_briefing, "_now_local", lambda: fake_now)

    daily_briefing.run()
    n1 = len(sent)
    daily_briefing.run()  # second call same window
    n2 = len(sent)
    assert n2 == n1  # idempotent — already sent today's morning


def test_briefing_skips_outside_windows(isolated, monkeypatch):
    tmp_path, sent = isolated
    from app.life_companion import daily_briefing

    monkeypatch.setattr(daily_briefing, "send_signal_alert",
                        lambda body, **kw: sent.append(body) or True)
    fake_now = datetime(2026, 5, 9, 13, 30)  # midday — no window
    monkeypatch.setattr(daily_briefing, "_now_local", lambda: fake_now)

    daily_briefing.run()
    assert sent == []


def test_briefing_weekly_takes_priority_on_dow(isolated, monkeypatch):
    """Monday at the weekly time should fire weekly, not morning."""
    tmp_path, sent = isolated
    from app.life_companion import daily_briefing

    monkeypatch.setattr(daily_briefing, "send_signal_alert",
                        lambda body, **kw: sent.append(body) or True)
    monkeypatch.setattr(daily_briefing, "audit_event", lambda *a, **k: None)
    monkeypatch.setattr(daily_briefing, "_gather_calendar_24h", lambda: [])
    monkeypatch.setattr(daily_briefing, "_gather_top_emails", lambda n=3: [])
    monkeypatch.setattr(daily_briefing, "_gather_open_tickets", lambda n=8: [])
    monkeypatch.setattr(daily_briefing, "_gather_companion_surfaced", lambda: [])

    # 2026-05-04 is a Monday. Weekly default is 09:00 Monday, morning is 07:00.
    monkeypatch.setenv("LIFE_COMPANION_BRIEFING_WEEKLY_TIME", "09:00")
    fake_now = datetime(2026, 5, 4, 9, 5)
    monkeypatch.setattr(daily_briefing, "_now_local", lambda: fake_now)

    daily_briefing.run()
    assert any("Weekly review" in body for body in sent)
