"""Tests for app.identity.continuity_ledger."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.identity import continuity_ledger as cl


@pytest.fixture
def ledger_path(tmp_path: Path) -> Path:
    return tmp_path / "ledger.jsonl"


def test_record_event_round_trip(ledger_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IDENTITY_LEDGER_ENABLED", "true")
    ok = cl.record_event(
        kind="tier3_amendment",
        actor="operator",
        summary="raised SAFETY_MINIMUM",
        detail={"old": 0.7, "new": 0.75},
        path=ledger_path,
    )
    assert ok is True
    events = cl.list_events(path=ledger_path)
    assert len(events) == 1
    assert events[0].kind == "tier3_amendment"
    assert events[0].actor == "operator"
    assert events[0].summary == "raised SAFETY_MINIMUM"
    assert events[0].detail == {"old": 0.7, "new": 0.75}


def test_record_event_disabled_short_circuits(
    ledger_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IDENTITY_LEDGER_ENABLED", "false")
    ok = cl.record_event(
        kind="tier3_amendment",
        actor="operator",
        summary="should not be written",
        path=ledger_path,
    )
    assert ok is False
    assert not ledger_path.exists()


def test_record_event_rejects_unknown_kind(
    ledger_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IDENTITY_LEDGER_ENABLED", "true")
    ok = cl.record_event(
        kind="bogus_kind",
        actor="operator",
        summary="should not be recorded",
        path=ledger_path,
    )
    assert ok is False
    assert not ledger_path.exists()


def test_record_event_rejects_blank_summary(
    ledger_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IDENTITY_LEDGER_ENABLED", "true")
    ok = cl.record_event(
        kind="soul_edit",
        actor="operator",
        summary="   ",
        path=ledger_path,
    )
    assert ok is False


def test_record_event_default_actor_when_blank(
    ledger_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IDENTITY_LEDGER_ENABLED", "true")
    cl.record_event(
        kind="soul_edit",
        actor="",
        summary="something happened",
        path=ledger_path,
    )
    events = cl.list_events(path=ledger_path)
    assert len(events) == 1
    assert events[0].actor == "unknown"


def test_list_events_returns_chronological(
    ledger_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IDENTITY_LEDGER_ENABLED", "true")
    # Append out of order; reader should sort.
    cl.record_event(
        kind="soul_edit",
        actor="operator",
        summary="second",
        path=ledger_path,
        now=datetime(2026, 5, 2, tzinfo=timezone.utc),
    )
    cl.record_event(
        kind="soul_edit",
        actor="operator",
        summary="first",
        path=ledger_path,
        now=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )
    cl.record_event(
        kind="soul_edit",
        actor="operator",
        summary="third",
        path=ledger_path,
        now=datetime(2026, 5, 3, tzinfo=timezone.utc),
    )
    events = cl.list_events(path=ledger_path)
    assert [e.summary for e in events] == ["first", "second", "third"]


def test_list_events_filter_by_kind(
    ledger_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IDENTITY_LEDGER_ENABLED", "true")
    cl.record_event(
        kind="soul_edit", actor="operator", summary="a", path=ledger_path,
    )
    cl.record_event(
        kind="tier3_amendment", actor="operator", summary="b", path=ledger_path,
    )
    cl.record_event(
        kind="governance_ratchet", actor="operator", summary="c", path=ledger_path,
    )
    events = cl.list_events(
        path=ledger_path,
        kinds={"tier3_amendment", "governance_ratchet"},
    )
    assert {e.summary for e in events} == {"b", "c"}


def test_list_events_filter_by_since_iso(
    ledger_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IDENTITY_LEDGER_ENABLED", "true")
    cl.record_event(
        kind="soul_edit", actor="operator", summary="old",
        path=ledger_path,
        now=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    cl.record_event(
        kind="soul_edit", actor="operator", summary="new",
        path=ledger_path,
        now=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )
    events = cl.list_events(
        path=ledger_path,
        since_iso="2026-03-01T00:00:00+00:00",
    )
    assert [e.summary for e in events] == ["new"]


def test_list_events_missing_file_returns_empty(tmp_path: Path) -> None:
    events = cl.list_events(path=tmp_path / "does_not_exist.jsonl")
    assert events == []


def test_list_events_skips_malformed_lines(
    ledger_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IDENTITY_LEDGER_ENABLED", "true")
    cl.record_event(
        kind="soul_edit", actor="operator", summary="real",
        path=ledger_path,
    )
    # Append junk lines that should be skipped silently.
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write("not json\n")
        f.write("\n")
        f.write(json.dumps({"missing": "fields"}) + "\n")
    events = cl.list_events(path=ledger_path)
    assert len(events) == 1
    assert events[0].summary == "real"


def test_summarise_drift_aggregates_by_kind_and_actor(
    ledger_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IDENTITY_LEDGER_ENABLED", "true")
    fixed_now = datetime(2026, 6, 1, tzinfo=timezone.utc)
    cl.record_event(
        kind="soul_edit", actor="operator", summary="a",
        path=ledger_path, now=fixed_now,
    )
    cl.record_event(
        kind="soul_edit", actor="self_improver", summary="b",
        path=ledger_path, now=fixed_now,
    )
    cl.record_event(
        kind="tier3_amendment", actor="operator", summary="c",
        path=ledger_path, now=fixed_now,
    )
    drift = cl.summarise_drift(window_days=365, path=ledger_path, now=fixed_now)
    assert drift.n_events == 3
    assert drift.by_kind == {"soul_edit": 2, "tier3_amendment": 1}
    assert drift.by_actor == {"operator": 2, "self_improver": 1}
    assert drift.first_seen is not None
    assert drift.last_seen is not None


def test_summarise_drift_window_excludes_old(
    ledger_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IDENTITY_LEDGER_ENABLED", "true")
    cl.record_event(
        kind="soul_edit", actor="operator", summary="ancient",
        path=ledger_path,
        now=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    cl.record_event(
        kind="soul_edit", actor="operator", summary="recent",
        path=ledger_path,
        now=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )
    drift = cl.summarise_drift(
        window_days=365,
        path=ledger_path,
        now=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )
    assert drift.n_events == 1
    assert drift.by_kind == {"soul_edit": 1}


def test_summarise_drift_empty(ledger_path: Path) -> None:
    drift = cl.summarise_drift(window_days=365, path=ledger_path)
    assert drift.n_events == 0
    assert drift.by_kind == {}
    assert drift.by_actor == {}
    assert drift.first_seen is None
    assert drift.last_seen is None


def test_identity_event_dict_round_trip() -> None:
    event = cl.IdentityEvent(
        ts="2026-05-01T00:00:00+00:00",
        kind="soul_edit",
        actor="operator",
        summary="test",
        detail={"key": "value"},
    )
    rebuilt = cl.IdentityEvent.from_dict(event.to_dict())
    assert rebuilt == event


def test_known_event_kinds_set() -> None:
    expected = {
        "tier3_amendment",
        "governance_ratchet",
        "soul_edit",
        "integrity_regen",
        "scorecard_change",
        "self_quarantine_change",
        "substrate_migration",          # PROGRAM §40 Item 12 — Q3.1
        "person_correlation_policy",    # PROGRAM §42 — Q4.2 (Q4.2.2#1)
        "sentience_observation",        # PROGRAM §43 — Q5.4.2
        "resilience_drill",             # PROGRAM §44 — Q6.1
    }
    assert cl.IDENTITY_EVENT_KINDS == frozenset(expected)
