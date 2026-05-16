"""Tests for app.browse.store — append, read, forget paths, cursors."""
from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from app.browse import store
from app.browse.models import BrowseEvent


def _ev(domain: str = "example.com", *, ts: str = "2026-05-16T10:00:00+00:00",
        path: str = "/", browser: str = "chrome") -> BrowseEvent:
    return BrowseEvent(
        visit_ts=ts, domain=domain, path=path, title=None,
        browser=browser, profile=None,
    )


def test_append_round_trip(_reset_browse_state: Path) -> None:
    n = store.append_events([_ev(domain="github.com")])
    assert n == 1
    out = store.list_events_for_day(date(2026, 5, 16))
    assert len(out) == 1
    assert out[0].domain == "github.com"


def test_append_disabled_short_circuits(
    _reset_browse_state: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PRIVACY PIN: when BROWSE_INGESTION_ENABLED is off, nothing
    hits disk."""
    monkeypatch.setenv("BROWSE_INGESTION_ENABLED", "false")
    n = store.append_events([_ev()])
    assert n == 0
    assert not (_reset_browse_state / "events").exists()


def test_append_groups_by_day(_reset_browse_state: Path) -> None:
    events = [
        _ev(ts="2026-05-15T23:59:00+00:00", domain="a.com"),
        _ev(ts="2026-05-16T00:01:00+00:00", domain="b.com"),
        _ev(ts="2026-05-16T15:00:00+00:00", domain="c.com"),
    ]
    n = store.append_events(events)
    assert n == 3
    day15 = store.list_events_for_day(date(2026, 5, 15))
    day16 = store.list_events_for_day(date(2026, 5, 16))
    assert {e.domain for e in day15} == {"a.com"}
    assert {e.domain for e in day16} == {"b.com", "c.com"}


def test_list_events_window(_reset_browse_state: Path) -> None:
    events = [
        _ev(ts="2026-05-10T12:00:00+00:00", domain="old.com"),
        _ev(ts="2026-05-14T12:00:00+00:00", domain="mid.com"),
        _ev(ts="2026-05-16T12:00:00+00:00", domain="today.com"),
    ]
    store.append_events(events)
    out = store.list_events_window(
        days=3,
        now=datetime(2026, 5, 16, 23, 59, tzinfo=timezone.utc),
    )
    domains = {e.domain for e in out}
    assert "today.com" in domains
    assert "mid.com" in domains
    assert "old.com" not in domains


def test_cursor_round_trip(_reset_browse_state: Path) -> None:
    store.save_cursor("chrome", "Default", 13000000000000000)
    assert store.get_cursor("chrome", "Default") == 13000000000000000
    store.save_cursor("safari", None, 700000000)
    cursors = store.load_cursors()
    assert cursors["chrome:Default"] == 13000000000000000
    assert cursors["safari"] == 700000000


def test_forget_all_clears_events_and_cursors(_reset_browse_state: Path) -> None:
    store.append_events([_ev()])
    store.save_cursor("chrome", "Default", 12345)
    assert store.event_counts()["total"] == 1
    removed = store.forget_all()
    assert removed >= 2
    assert store.event_counts()["total"] == 0
    assert store.load_cursors() == {}


def test_forget_all_preserves_blocklist(_reset_browse_state: Path) -> None:
    bl_path = _reset_browse_state / "blocklist.txt"
    bl_path.write_text("custom.test\n", encoding="utf-8")
    store.forget_all()
    assert bl_path.exists()
    assert "custom.test" in bl_path.read_text(encoding="utf-8")


def test_forget_day(_reset_browse_state: Path) -> None:
    events = [
        _ev(ts="2026-05-15T12:00:00+00:00", domain="a.com"),
        _ev(ts="2026-05-16T12:00:00+00:00", domain="b.com"),
    ]
    store.append_events(events)
    assert store.forget_day(date(2026, 5, 15)) is True
    assert store.list_events_for_day(date(2026, 5, 15)) == []
    assert len(store.list_events_for_day(date(2026, 5, 16))) == 1


def test_forget_day_missing_file(_reset_browse_state: Path) -> None:
    assert store.forget_day(date(2099, 1, 1)) is False


def test_forget_domain_removes_matching_rows(_reset_browse_state: Path) -> None:
    events = [
        _ev(ts="2026-05-16T08:00:00+00:00", domain="github.com"),
        _ev(ts="2026-05-16T09:00:00+00:00", domain="api.github.com"),
        _ev(ts="2026-05-16T10:00:00+00:00", domain="gitlab.com"),
    ]
    store.append_events(events)
    removed = store.forget_domain("github.com")
    assert removed == 2
    remaining = store.list_events_for_day(date(2026, 5, 16))
    assert [e.domain for e in remaining] == ["gitlab.com"]


def test_event_counts_reports_top_domains(_reset_browse_state: Path) -> None:
    events = (
        [_ev(domain="github.com")] * 3
        + [_ev(domain="wikipedia.org")] * 5
        + [_ev(domain="news.ycombinator.com")] * 1
    )
    store.append_events(events)
    counts = store.event_counts(days=2)
    assert counts["total"] == 9
    top_domains = [row["domain"] for row in counts["by_domain_top"]]
    assert top_domains[0] == "wikipedia.org"
