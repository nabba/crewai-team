"""Tests for the Firefox history reader."""
from __future__ import annotations

from pathlib import Path

from app.browse.readers import firefox
from tests.browse.conftest import build_firefox_db


# Unix-epoch microseconds for 2026-05-16 12:00 UTC.
_T_2026_05_16 = 1_778_932_800_000_000


def _build(tmp_path: Path, rows: list[tuple]) -> Path:
    db = tmp_path / "places.sqlite"
    build_firefox_db(db, rows)
    return db


def test_reads_canonicalised_event(tmp_path: Path) -> None:
    db = _build(tmp_path, [
        (_T_2026_05_16, "https://en.wikipedia.org/wiki/Tallinn?lang=en",
         "Tallinn - Wikipedia", 2),  # 2 = typed
    ])
    results = firefox.read_new(
        cursors={},
        overrides=[("default-release", db)],
    )
    assert len(results) == 1
    r = results[0]
    assert r.error is None
    assert len(r.events) == 1
    ev = r.events[0]
    assert ev.domain == "en.wikipedia.org"
    assert ev.path == "/wiki/Tallinn"
    assert "lang=en" not in ev.path
    assert ev.title == "Tallinn - Wikipedia"
    assert ev.browser == "firefox"
    assert ev.profile == "default-release"
    assert ev.transition == "typed"
    # Firefox doesn't expose dwell.
    assert ev.visit_duration_ms is None


def test_blocklisted_domain_skipped(tmp_path: Path) -> None:
    db = _build(tmp_path, [
        (_T_2026_05_16, "https://nordea.fi/account", "Nordea", 1),
        (_T_2026_05_16 + 1, "https://wikipedia.org/", "Wiki", 1),
    ])
    results = firefox.read_new(
        cursors={},
        overrides=[("default-release", db)],
    )
    r = results[0]
    assert [e.domain for e in r.events] == ["wikipedia.org"]
    assert r.skipped_blocklisted == 1


def test_cursor_filters_old_visits(tmp_path: Path) -> None:
    db = _build(tmp_path, [
        (_T_2026_05_16, "https://a.com/", "A", 1),
        (_T_2026_05_16 + 100_000, "https://b.com/", "B", 1),
    ])
    results = firefox.read_new(
        cursors={"firefox:default-release": _T_2026_05_16 + 1},
        overrides=[("default-release", db)],
    )
    assert [e.domain for e in results[0].events] == ["b.com"]


def test_discover_profiles_walks_random_directory_names(tmp_path: Path) -> None:
    home = tmp_path / "home"
    profiles = home / "Library/Application Support/Firefox/Profiles"
    for name in ("aaaaaaaa.default-release", "bbbbbbbb.dev-edition-default"):
        d = profiles / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "places.sqlite").touch()
    out = firefox.discover_profiles(home=home)
    names = [name for name, _ in out]
    assert sorted(names) == [
        "aaaaaaaa.default-release", "bbbbbbbb.dev-edition-default",
    ]
