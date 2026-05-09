"""Phase G targeted tests — pin behavior for the 4 audit-gap fixes."""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone

import pytest


# ── G1: 72h calendar horizon scan + conflicts ────────────────────────────


@pytest.fixture
def horizon(tmp_path, monkeypatch):
    from app.life_companion import calendar_horizon as ch
    from app.life_companion import _common

    monkeypatch.setattr(_common, "_STATE_DIR", tmp_path / "lc")
    monkeypatch.setattr(ch, "background_enabled", lambda: True)
    monkeypatch.setattr(ch, "audit_event", lambda *a, **k: None)
    monkeypatch.setattr(ch, "_target_hour", lambda: 0)  # always within window
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(
        ch, "send_signal_alert",
        lambda body, tag=None, **kw: sent.append((body, tag or "")),
    )
    yield tmp_path, sent


def test_calendar_horizon_no_conflicts_no_alert(horizon, monkeypatch):
    tmp_path, sent = horizon
    from app.life_companion import calendar_horizon as ch
    monkeypatch.setattr(ch, "_list_events_72h", lambda: [])
    summary = ch.run()
    assert summary["ran"] is True
    assert summary["conflicts"] == 0
    assert sent == []


def test_calendar_horizon_detects_overlap(horizon, monkeypatch):
    tmp_path, sent = horizon
    from app.life_companion import calendar_horizon as ch
    base = datetime.now(timezone.utc).replace(microsecond=0) + timedelta(hours=10)
    events = [
        {"id": "a", "summary": "Standup",
         "start": base, "end": base + timedelta(minutes=30),
         "all_day": False},
        {"id": "b", "summary": "Gov briefing",
         "start": base + timedelta(minutes=15),
         "end": base + timedelta(minutes=45),
         "all_day": False},
    ]
    monkeypatch.setattr(ch, "_list_events_72h", lambda: events)
    summary = ch.run()
    assert summary["conflicts"] == 1
    assert summary["sent"] is True
    body, _ = sent[0]
    assert "Standup" in body
    assert "Gov briefing" in body


def test_calendar_horizon_detects_density_cluster(horizon, monkeypatch):
    tmp_path, sent = horizon
    from app.life_companion import calendar_horizon as ch
    base = datetime.now(timezone.utc).replace(microsecond=0) + timedelta(hours=2)
    events = []
    for i in range(4):
        start = base + timedelta(minutes=i * 35)
        events.append({
            "id": f"e{i}", "summary": f"Mtg {i}",
            "start": start, "end": start + timedelta(minutes=30),
            "all_day": False,
        })
    monkeypatch.setattr(ch, "_list_events_72h", lambda: events)
    summary = ch.run()
    assert summary["dense_clusters"] == 1
    assert summary["sent"] is True
    assert any("dense cluster" in b.lower() for b, _ in sent)


def test_calendar_horizon_ignores_all_day(horizon, monkeypatch):
    tmp_path, sent = horizon
    from app.life_companion import calendar_horizon as ch
    today = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0,
    )
    events = [
        {"id": "a", "summary": "All-day",
         "start": today, "end": today + timedelta(days=1),
         "all_day": True},
        {"id": "b", "summary": "Real meeting",
         "start": today + timedelta(hours=10),
         "end": today + timedelta(hours=11),
         "all_day": False},
    ]
    monkeypatch.setattr(ch, "_list_events_72h", lambda: events)
    summary = ch.run()
    assert summary["conflicts"] == 0  # all-day excluded


def test_calendar_horizon_dedup_same_day(horizon, monkeypatch):
    tmp_path, sent = horizon
    from app.life_companion import calendar_horizon as ch
    base = datetime.now(timezone.utc) + timedelta(hours=10)
    events = [
        {"id": "a", "summary": "x",
         "start": base, "end": base + timedelta(minutes=30),
         "all_day": False},
        {"id": "b", "summary": "y",
         "start": base + timedelta(minutes=10),
         "end": base + timedelta(minutes=40),
         "all_day": False},
    ]
    monkeypatch.setattr(ch, "_list_events_72h", lambda: events)
    ch.run()
    initial = len(sent)
    # Reset cadence; rerun same day → already-sent, no second alert.
    state_path = tmp_path / "lc" / "calendar_horizon.json"
    state = json.loads(state_path.read_text())
    state["last_run_at"] = 0.0
    state_path.write_text(json.dumps(state))
    ch.run()
    assert len(sent) == initial


# ── G2: topic-level feedback weights ─────────────────────────────────────


@pytest.fixture
def topic_weights_isolated(tmp_path, monkeypatch):
    from app.companion import topic_weights
    monkeypatch.setattr(topic_weights, "_STATE_PATH",
                        tmp_path / "topic_weights.json")
    yield tmp_path / "topic_weights.json"


def test_topic_weights_default_one(topic_weights_isolated):
    from app.companion.topic_weights import current_multiplier
    assert current_multiplier("forest carbon") == 1.0


def test_topic_weights_one_negative(topic_weights_isolated):
    from app.companion.topic_weights import current_multiplier, record_negative
    record_negative("forest carbon")
    m = current_multiplier("forest carbon")
    assert 0.79 <= m <= 0.81


def test_topic_weights_distinct_topics_isolated(topic_weights_isolated):
    from app.companion.topic_weights import current_multiplier, record_negative
    record_negative("forest carbon")
    assert current_multiplier("forest carbon") < 1.0
    assert current_multiplier("kaicart") == 1.0


def test_topic_weights_normalize_case_insensitive(topic_weights_isolated):
    from app.companion.topic_weights import current_multiplier, record_negative
    record_negative("Forest Carbon")
    assert current_multiplier("forest carbon") < 1.0
    assert current_multiplier("FOREST CARBON") < 1.0


def test_record_negative_from_comment(topic_weights_isolated, monkeypatch):
    """Comment text → topic mentions matched against live profile → downweighted."""
    from app.companion import topic_weights
    monkeypatch.setattr(
        "app.companion.interest_model.current_profile",
        lambda: {"topics": [
            {"name": "forest carbon", "score": 1.0},
            {"name": "kaicart", "score": 0.5},
        ]},
    )
    matched = topic_weights.record_negative_from_comment(
        "I really dislike the forest carbon angle"
    )
    assert "forest carbon" in matched
    assert "kaicart" not in matched
    assert topic_weights.current_multiplier("forest carbon") < 1.0


def test_topic_weights_decays_over_time(topic_weights_isolated):
    from app.companion import topic_weights
    topic_weights.record_negative("forest carbon")
    state = json.loads(topic_weights_isolated.read_text())
    # Pretend the anchor was set 14 days ago (~2 halflives).
    state["forest carbon"]["first_observed_at"] = time.time() - 14 * 86400
    topic_weights_isolated.write_text(json.dumps(state))
    m = topic_weights.current_multiplier("forest carbon")
    assert m > 0.93  # near 1.0


# ── G3: topic-dormancy detection ─────────────────────────────────────────


@pytest.fixture
def dormancy(tmp_path, monkeypatch):
    from app.life_companion import topic_dormancy as td
    from app.life_companion import _common

    monkeypatch.setattr(_common, "_STATE_DIR", tmp_path / "lc")
    monkeypatch.setattr(td, "background_enabled", lambda: True)
    monkeypatch.setattr(td, "audit_event", lambda *a, **k: None)
    monkeypatch.setattr(td, "_HISTORY_PATH", tmp_path / "interest_history.jsonl")
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(
        td, "send_signal_alert",
        lambda body, tag=None, **kw: sent.append((body, tag or "")),
    )
    yield tmp_path, sent


def _seed_history(tmp_path, name: str, points: list[tuple[float, float]]) -> None:
    p = tmp_path / "interest_history.jsonl"
    with p.open("a") as f:
        for ts, score in points:
            row = {
                "ts": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                "name": name,
                "score": score,
            }
            f.write(json.dumps(row) + "\n")


def test_dormancy_no_history_no_alert(dormancy):
    tmp_path, sent = dormancy
    from app.life_companion import topic_dormancy as td
    summary = td.run()
    assert summary["ran"] is True
    assert summary["dormant"] == 0


def test_dormancy_detects_old_peak_recent_silence(dormancy):
    tmp_path, sent = dormancy
    from app.life_companion import topic_dormancy as td
    now = time.time()
    # 5 high old observations + 1 low recent observation.
    _seed_history(tmp_path, "forest carbon", [
        (now - 200 * 86400, 2.0),
        (now - 180 * 86400, 1.8),
        (now - 150 * 86400, 1.5),
        (now - 120 * 86400, 1.7),
        (now - 100 * 86400, 1.9),
        (now - 5 * 86400, 0.1),
    ])
    summary = td.run()
    assert summary["dormant"] >= 1
    assert summary["sent"] is True
    body, tag = sent[0]
    assert "forest carbon" in body
    assert tag == "topic_dormancy"


def test_dormancy_skipped_when_recent_active(dormancy):
    tmp_path, sent = dormancy
    from app.life_companion import topic_dormancy as td
    now = time.time()
    _seed_history(tmp_path, "kaicart", [
        (now - 200 * 86400, 2.0),
        (now - 150 * 86400, 1.8),
        (now - 100 * 86400, 1.5),
        (now - 80 * 86400, 1.7),
        (now - 5 * 86400, 1.5),  # recent activity → not dormant
    ])
    summary = td.run()
    assert summary["dormant"] == 0


def test_dormancy_mute_silences_topic(dormancy):
    tmp_path, sent = dormancy
    from app.life_companion import topic_dormancy as td
    now = time.time()
    _seed_history(tmp_path, "forest carbon", [
        (now - 200 * 86400, 2.0),
        (now - 180 * 86400, 1.8),
        (now - 150 * 86400, 1.5),
        (now - 100 * 86400, 1.9),
        (now - 5 * 86400, 0.1),
    ])
    td.mute("forest carbon")
    summary = td.run()
    assert summary["sent"] is False
    assert "forest carbon" not in summary["alerted"]


def test_dormancy_dedup_within_30_days(dormancy):
    tmp_path, sent = dormancy
    from app.life_companion import topic_dormancy as td
    now = time.time()
    _seed_history(tmp_path, "forest carbon", [
        (now - 200 * 86400, 2.0),
        (now - 180 * 86400, 1.8),
        (now - 150 * 86400, 1.5),
        (now - 100 * 86400, 1.9),
        (now - 5 * 86400, 0.1),
    ])
    td.run()
    initial = len(sent)
    state_path = tmp_path / "lc" / "topic_dormancy.json"
    state = json.loads(state_path.read_text())
    state["last_run_at"] = 0.0
    state_path.write_text(json.dumps(state))
    td.run()
    # Same alert dedup'd within 30-day window.
    assert len(sent) == initial


# ── G4: Finland-seasonal nudges ──────────────────────────────────────────


@pytest.fixture
def seasonal(tmp_path, monkeypatch):
    from app.life_companion import seasonal_nudges as sn
    from app.life_companion import _common

    monkeypatch.setattr(_common, "_STATE_DIR", tmp_path / "lc")
    monkeypatch.setattr(sn, "background_enabled", lambda: True)
    monkeypatch.setattr(sn, "audit_event", lambda *a, **k: None)
    monkeypatch.setattr(sn, "_operator_in_finland", lambda: True)
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(
        sn, "send_signal_alert",
        lambda body, tag=None, **kw: sent.append((body, tag or "")),
    )
    yield tmp_path, sent


def test_seasonal_first_frost(seasonal, monkeypatch):
    from datetime import date
    tmp_path, sent = seasonal
    from app.life_companion import seasonal_nudges as sn
    monkeypatch.setattr(sn, "_today", lambda: date(2026, 10, 20))
    summary = sn.run()
    assert summary["sent"] is True
    assert summary["trigger_key"].startswith("first_frost_")
    assert any("frost" in b.lower() for b, _ in sent)


def test_seasonal_kaamos(seasonal, monkeypatch):
    from datetime import date
    tmp_path, sent = seasonal
    from app.life_companion import seasonal_nudges as sn
    monkeypatch.setattr(sn, "_today", lambda: date(2026, 11, 22))
    summary = sn.run()
    assert summary["sent"] is True
    assert "kaamos" in summary["trigger_key"]


def test_seasonal_dedup_per_year(seasonal, monkeypatch):
    from datetime import date
    tmp_path, sent = seasonal
    from app.life_companion import seasonal_nudges as sn
    monkeypatch.setattr(sn, "_today", lambda: date(2026, 12, 21))
    sn.run()
    initial = len(sent)
    state_path = tmp_path / "lc" / "seasonal_nudges.json"
    state = json.loads(state_path.read_text())
    state["last_run_at"] = 0.0
    state_path.write_text(json.dumps(state))
    sn.run()
    assert len(sent) == initial  # already fired this year


def test_seasonal_skips_when_outside_finland(seasonal, monkeypatch):
    from datetime import date
    tmp_path, sent = seasonal
    from app.life_companion import seasonal_nudges as sn
    monkeypatch.setattr(sn, "_operator_in_finland", lambda: False)
    monkeypatch.setattr(sn, "_today", lambda: date(2026, 12, 21))
    summary = sn.run()
    assert summary["ran"] is False
    assert sent == []


def test_seasonal_no_trigger_quiet_day(seasonal, monkeypatch):
    from datetime import date
    tmp_path, sent = seasonal
    from app.life_companion import seasonal_nudges as sn
    # March 5 is not in any trigger window.
    monkeypatch.setattr(sn, "_today", lambda: date(2026, 3, 5))
    summary = sn.run()
    assert summary["ran"] is True
    assert summary["sent"] is False


def test_seasonal_midsummer_warn_10_days_early(seasonal, monkeypatch):
    from datetime import date
    tmp_path, sent = seasonal
    from app.life_companion import seasonal_nudges as sn
    # 2026 Juhannus is Saturday June 20. Warn fires 10-8 days before.
    monkeypatch.setattr(sn, "_today", lambda: date(2026, 6, 11))
    summary = sn.run()
    assert summary["sent"] is True
    assert "juhannus_warn" in summary["trigger_key"]


# ── G2 wiring: interest_model multiplies by topic_weights ────────────────


def test_interest_model_applies_topic_multiplier(monkeypatch, tmp_path):
    from app.companion import interest_model
    from app.companion import topic_weights
    monkeypatch.setattr(topic_weights, "_STATE_PATH",
                        tmp_path / "topic_weights.json")
    monkeypatch.setattr(interest_model, "_PROFILE_PATH",
                        tmp_path / "interest_profile.json")
    monkeypatch.setattr(interest_model, "_HISTORY_PATH",
                        tmp_path / "interest_history.jsonl")

    monkeypatch.setattr(
        interest_model, "_conversations_text",
        lambda d: iter([
            ("forest carbon estonia", 0.0),
            ("forest carbon flux", 1.0),
            ("forest carbon estonia winter", 2.0),
        ]),
    )
    monkeypatch.setattr(
        interest_model, "_email_subject_text", lambda d: iter([]),
    )
    monkeypatch.setattr(
        interest_model, "_calendar_titles_text", lambda d: iter([]),
    )
    monkeypatch.setattr(
        interest_model, "_feedback_events_text", lambda d: iter([]),
    )
    monkeypatch.setattr(
        interest_model, "_affect_topics_text", lambda d: iter([]),
    )

    # Baseline run.
    p1 = interest_model.compile_interest_profile(lookback_days=14)
    base_score = next(
        t["score"] for t in p1["topics"] if t["name"] == "forest carbon"
    )
    # Now downweight + recompile.
    topic_weights.record_negative("forest carbon")
    topic_weights.record_negative("forest carbon")
    p2 = interest_model.compile_interest_profile(lookback_days=14)
    after = next(
        t["score"] for t in p2["topics"] if t["name"] == "forest carbon"
    )
    assert after < base_score


# ── G3 wiring: interest_model writes history ─────────────────────────────


def test_interest_model_appends_history(monkeypatch, tmp_path):
    from app.companion import interest_model
    monkeypatch.setattr(interest_model, "_PROFILE_PATH",
                        tmp_path / "profile.json")
    monkeypatch.setattr(interest_model, "_HISTORY_PATH",
                        tmp_path / "history.jsonl")
    monkeypatch.setattr(
        interest_model, "_conversations_text",
        lambda d: iter([
            ("forest carbon estonia", 0.0),
            ("forest carbon flux estonia", 1.0),
        ]),
    )
    monkeypatch.setattr(
        interest_model, "_email_subject_text", lambda d: iter([]),
    )
    monkeypatch.setattr(
        interest_model, "_calendar_titles_text", lambda d: iter([]),
    )
    monkeypatch.setattr(
        interest_model, "_feedback_events_text", lambda d: iter([]),
    )
    monkeypatch.setattr(
        interest_model, "_affect_topics_text", lambda d: iter([]),
    )
    interest_model.compile_interest_profile(lookback_days=14)
    assert (tmp_path / "history.jsonl").exists()
    lines = (tmp_path / "history.jsonl").read_text().strip().splitlines()
    assert len(lines) >= 1
    row = json.loads(lines[0])
    assert "name" in row
    assert "score" in row
    assert "ts" in row


# ── /topic mute slash command ────────────────────────────────────────────


def test_topic_command_help():
    from app.agents.commander.commands import _handle_topic_command
    out = _handle_topic_command("/topic")
    assert "mute" in out
    assert "unmute" in out


def test_topic_mute_returns_acknowledgement(monkeypatch):
    from app.agents.commander import commands
    captured = []
    monkeypatch.setattr(
        "app.life_companion.topic_dormancy.mute",
        lambda topic: captured.append(topic) or True,
    )
    out = commands._handle_topic_command("/topic mute forest carbon")
    assert "forest carbon" in out
    assert captured == ["forest carbon"]
