"""Tests for ``app.healing.silent_regression_detector`` (Phase C #2)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    from app.healing import silent_regression_detector
    from app.healing.handlers import _common as _h_common

    monkeypatch.setattr(_h_common, "_STATE_DIR", tmp_path / "self_heal")
    monkeypatch.setattr(silent_regression_detector,
                        "_AUDIT_JOURNAL_PATH", tmp_path / "audit_journal.json")
    monkeypatch.setattr(silent_regression_detector,
                        "_recent_git_commits", lambda *a, **kw: [])
    monkeypatch.setattr(silent_regression_detector,
                        "_recent_change_requests", lambda *a, **kw: [])

    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(silent_regression_detector, "send_signal_alert",
                        lambda body, tag=None, **kw: sent.append((body, tag or "")))
    monkeypatch.setattr(silent_regression_detector, "audit_event",
                        lambda *a, **k: None)

    yield tmp_path, sent


def _seed_journal(path, events: list[tuple[datetime, str]]) -> None:
    rows = [{"ts": ts.isoformat(), "event": event, "detail": "x", "files_changed": []}
            for ts, event in events]
    path.write_text(json.dumps(rows))


def test_no_journal_no_alerts(isolated):
    tmp_path, sent = isolated
    from app.healing import silent_regression_detector
    summary = silent_regression_detector.run()
    assert summary["ran"] is True
    assert sent == []


def test_baseline_too_thin_no_alert(isolated):
    tmp_path, sent = isolated
    from app.healing import silent_regression_detector
    now = datetime.now(timezone.utc)
    # Only 3 baseline events (under MIN_BASELINE_SAMPLES=8) — should not alert.
    events = [(now - timedelta(days=2 + i), "error_resolution") for i in range(3)]
    _seed_journal(tmp_path / "audit_journal.json", events)
    silent_regression_detector.run()
    assert sent == []


def test_alert_when_recent_drops_below_threshold(isolated):
    tmp_path, sent = isolated
    from app.healing import silent_regression_detector
    now = datetime.now(timezone.utc)
    # 26 baseline events evenly spread over 13 days (2/day) + only 0 recent.
    events = []
    for d in range(2, 15):  # days 2-14 in baseline window
        events.append((now - timedelta(days=d, hours=0), "error_resolution"))
        events.append((now - timedelta(days=d, hours=12), "error_resolution"))
    _seed_journal(tmp_path / "audit_journal.json", events)
    silent_regression_detector.run()
    assert any("error_resolution" in body for body, _ in sent)
    assert any("Silent regression" in body for body, _ in sent)


def test_no_alert_when_recent_matches_baseline(isolated):
    tmp_path, sent = isolated
    from app.healing import silent_regression_detector
    now = datetime.now(timezone.utc)
    events = []
    # Strong baseline — 26 events over 13 days (2/day).
    for d in range(2, 15):
        events.append((now - timedelta(days=d, hours=0), "error_resolution"))
        events.append((now - timedelta(days=d, hours=12), "error_resolution"))
    # And 2 recent events — matching baseline rate.
    events.append((now - timedelta(hours=4), "error_resolution"))
    events.append((now - timedelta(hours=18), "error_resolution"))
    _seed_journal(tmp_path / "audit_journal.json", events)
    silent_regression_detector.run()
    assert sent == []


def test_dedup_within_window(isolated):
    tmp_path, sent = isolated
    from app.healing import silent_regression_detector
    now = datetime.now(timezone.utc)
    events = []
    for d in range(2, 15):
        events.append((now - timedelta(days=d, hours=0), "self_improve"))
        events.append((now - timedelta(days=d, hours=12), "self_improve"))
    _seed_journal(tmp_path / "audit_journal.json", events)
    silent_regression_detector.run()

    # Reset cadence guard; rerun — alert should NOT fire (24h dedup).
    state_path = tmp_path / "self_heal" / "silent_regression_alerts.json"
    state = json.loads(state_path.read_text())
    state["last_run_at"] = 0.0
    state_path.write_text(json.dumps(state))
    silent_regression_detector.run()

    assert sum("self_improve" in body for body, _ in sent) == 1


def test_disabled_skips(monkeypatch, isolated):
    tmp_path, sent = isolated
    monkeypatch.setenv("HEALING_SILENT_REGRESSION_ENABLED", "0")
    from app.healing import silent_regression_detector
    summary = silent_regression_detector.run()
    assert summary["ran"] is False
