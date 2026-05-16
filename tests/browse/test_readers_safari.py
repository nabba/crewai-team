"""Tests for the Safari history reader."""
from __future__ import annotations

from pathlib import Path

from app.browse.readers import safari
from tests.browse.conftest import build_safari_db


# Mac-epoch (CFAbsoluteTime) seconds for 2026-05-16 12:00 UTC.
# = unix-seconds 1778932800 − Mac-epoch-offset 978307200
_T_2026_05_16 = 800_625_600.0


def _build(tmp_path: Path, rows: list[tuple]) -> Path:
    db = tmp_path / "History.db"
    build_safari_db(db, rows)
    return db


def test_reads_canonicalised_event(tmp_path: Path) -> None:
    db = _build(tmp_path, [
        (_T_2026_05_16, "https://en.wikipedia.org/wiki/Helsinki?lang=en",
         "Helsinki - Wikipedia"),
    ])
    r = safari.read_new(cursor_s=0.0, override_path=db)
    assert r.error is None
    assert len(r.events) == 1
    ev = r.events[0]
    assert ev.domain == "en.wikipedia.org"
    assert ev.path == "/wiki/Helsinki"
    assert "lang=en" not in ev.path
    assert ev.title == "Helsinki - Wikipedia"
    assert ev.browser == "safari"
    assert ev.profile is None
    # Safari doesn't expose dwell or transition.
    assert ev.visit_duration_ms is None
    assert ev.transition is None


def test_blocklisted_domain_skipped(tmp_path: Path) -> None:
    db = _build(tmp_path, [
        (_T_2026_05_16, "https://kanta.fi/omakanta", "OmaKanta"),
        (_T_2026_05_16 + 1, "https://wikipedia.org/", "Wikipedia"),
    ])
    r = safari.read_new(cursor_s=0.0, override_path=db)
    assert [e.domain for e in r.events] == ["wikipedia.org"]
    assert r.skipped_blocklisted == 1


def test_cursor_filters_old_visits(tmp_path: Path) -> None:
    db = _build(tmp_path, [
        (_T_2026_05_16, "https://a.com/", "A"),
        (_T_2026_05_16 + 100, "https://b.com/", "B"),
    ])
    r = safari.read_new(cursor_s=_T_2026_05_16 + 50, override_path=db)
    assert [e.domain for e in r.events] == ["b.com"]


def test_cursor_returned_in_microseconds(tmp_path: Path) -> None:
    """Safari uses seconds natively but we normalise to microseconds
    in the cursor so the state file format is uniform across readers."""
    db = _build(tmp_path, [
        (_T_2026_05_16, "https://a.com/", "A"),
    ])
    r = safari.read_new(cursor_s=0.0, override_path=db)
    expected = int(round(_T_2026_05_16 * 1_000_000))
    assert r.last_cursor == expected


def test_missing_db_returns_empty_result(tmp_path: Path) -> None:
    r = safari.read_new(cursor_s=0.0, override_path=tmp_path / "absent")
    assert r.error is None
    assert r.events == []
