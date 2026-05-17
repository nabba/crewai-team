"""Tests for app.tools.travel_tools — PIM-side TripIt surface."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def tmp_workspace(tmp_path, monkeypatch):
    """Point WORKSPACE_ROOT at a fresh dir so the travel module's
    snapshot lookup hits empty data by default."""
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    yield tmp_path


def _seed_snapshot(workspace: Path, segments: list[dict]) -> None:
    """Write a tripit_trips.json the production reader will load."""
    d = workspace / "life_companion" / "travel"
    d.mkdir(parents=True, exist_ok=True)
    (d / "tripit_trips.json").write_text(json.dumps(segments), encoding="utf-8")


def test_factory_returns_two_tools_when_crewai_present(tmp_workspace):
    from app.tools.travel_tools import create_travel_tools
    tools = create_travel_tools("pim")
    # CrewAI present in the dev/test env, so we expect both tools.
    # If CrewAI is genuinely missing we'd get [] — both states are valid;
    # we assert the shape.
    assert isinstance(tools, list)
    if tools:
        names = {t.name for t in tools}
        assert names == {"list_upcoming_flights", "list_upcoming_trips"}


def test_empty_snapshot_returns_no_data_with_hint(tmp_workspace, monkeypatch):
    # Stub _get_tripit_url to "" so the hint is the unconfigured one
    from app.life_companion import travel
    monkeypatch.setattr(travel, "_get_tripit_url", lambda: "")

    from app.tools.travel_tools import create_travel_tools
    tools = create_travel_tools("pim")
    if not tools:
        pytest.skip("CrewAI tooling not installed in this env")
    flights = next(t for t in tools if t.name == "list_upcoming_flights")
    payload = json.loads(flights.func(window_days=14))
    assert payload["status"] == "no_data"
    assert "TripIt iCal URL is not configured" in payload["hint"]


def test_configured_but_no_trips_in_window(tmp_workspace, monkeypatch):
    from app.life_companion import travel
    monkeypatch.setattr(
        travel, "_get_tripit_url",
        lambda: "https://www.tripit.com/feed/ical/private/x.ics",
    )
    # Snapshot exists but is empty
    _seed_snapshot(tmp_workspace, [])

    from app.tools.travel_tools import create_travel_tools
    tools = create_travel_tools("pim")
    if not tools:
        pytest.skip("CrewAI tooling not installed in this env")
    flights = next(t for t in tools if t.name == "list_upcoming_flights")
    payload = json.loads(flights.func(window_days=14))
    assert payload["status"] == "no_data"
    assert "idle job hasn't refreshed yet" in payload["hint"]


def test_flights_tool_filters_to_flights(tmp_workspace):
    from datetime import datetime, timedelta, timezone
    future = (datetime.now(timezone.utc) + timedelta(days=2)).isoformat()
    later = (datetime.now(timezone.utc) + timedelta(days=2, hours=3)).isoformat()
    _seed_snapshot(tmp_workspace, [
        {"summary": "LH881 TLL to FRA", "location": "Tallinn",
         "starts_at": future, "ends_at": later,
         "uid": "f1", "kind": "flight", "flight_number": "LH881"},
        {"summary": "Check-in: Radisson", "location": "Bucharest",
         "starts_at": future, "ends_at": later,
         "uid": "h1", "kind": "hotel", "flight_number": ""},
    ])

    from app.tools.travel_tools import create_travel_tools
    tools = create_travel_tools("pim")
    if not tools:
        pytest.skip("CrewAI tooling not installed in this env")

    flights = next(t for t in tools if t.name == "list_upcoming_flights")
    payload = json.loads(flights.func(window_days=14))
    assert payload["status"] == "ok"
    assert payload["count"] == 1
    assert payload["flights"][0]["flight_number"] == "LH881"

    trips = next(t for t in tools if t.name == "list_upcoming_trips")
    payload = json.loads(trips.func(window_days=14))
    assert payload["status"] == "ok"
    assert payload["count"] == 2
    kinds = {s["kind"] for s in payload["segments"]}
    assert kinds == {"flight", "hotel"}

    # kind filter
    payload = json.loads(trips.func(window_days=14, kind="hotel"))
    assert payload["count"] == 1
    assert payload["segments"][0]["kind"] == "hotel"


def test_window_days_clamped(tmp_workspace):
    from app.tools.travel_tools import create_travel_tools
    tools = create_travel_tools("pim")
    if not tools:
        pytest.skip("CrewAI tooling not installed in this env")
    flights = next(t for t in tools if t.name == "list_upcoming_flights")
    # 9999 → 60, -5 → 1, "bogus" → 14
    for given, expected in [(9999, 60), (-5, 1), ("bogus", 14)]:
        payload = json.loads(flights.func(window_days=given))
        assert payload["window_days"] == expected


# ── Helper-level tests (run even without CrewAI installed) ────────────


def test_hint_when_unconfigured(tmp_workspace, monkeypatch):
    """Helper-level test — exercises _hint_when_empty without needing
    CrewAI installed."""
    from app.life_companion import travel
    monkeypatch.setattr(travel, "_get_tripit_url", lambda: "")
    from app.tools.travel_tools import _hint_when_empty
    assert "TripIt iCal URL is not configured" in _hint_when_empty()


def test_hint_when_configured_but_idle_job_unrun(tmp_workspace, monkeypatch):
    from app.life_companion import travel
    monkeypatch.setattr(
        travel, "_get_tripit_url",
        lambda: "https://www.tripit.com/feed/ical/private/x.ics",
    )
    from app.tools.travel_tools import _hint_when_empty
    assert "idle job hasn't refreshed yet" in _hint_when_empty()


def test_segments_in_window_reads_snapshot(tmp_workspace):
    """End-to-end through the snapshot reader, no CrewAI needed."""
    from datetime import datetime, timedelta, timezone
    future = (datetime.now(timezone.utc) + timedelta(days=2)).isoformat()
    later = (datetime.now(timezone.utc) + timedelta(days=2, hours=3)).isoformat()
    _seed_snapshot(tmp_workspace, [
        {"summary": "LH881 TLL to FRA", "location": "TLL",
         "starts_at": future, "ends_at": later,
         "uid": "x", "kind": "flight", "flight_number": "LH881"},
    ])

    from app.tools.travel_tools import _segments_in_window
    segs = _segments_in_window(14)
    assert len(segs) == 1
    assert segs[0]["flight_number"] == "LH881"
    assert segs[0]["kind"] == "flight"
