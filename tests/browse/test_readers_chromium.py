"""Tests for the Chromium-family history reader."""
from __future__ import annotations

from pathlib import Path

from app.browse.readers import chromium
from tests.browse.conftest import build_chromium_db


# Chrome-epoch microseconds for 2026-05-16 12:00 UTC.
# = (1778932800 unix-seconds + 11644473600 epoch-offset) * 1e6
_T_2026_05_16 = 13_423_406_400_000_000


def _build(tmp_path: Path, rows: list[tuple]) -> Path:
    db = tmp_path / "History"
    build_chromium_db(db, rows)
    return db


def test_reads_canonicalised_event(tmp_path: Path) -> None:
    db = _build(tmp_path, [
        (_T_2026_05_16, "https://github.com/anthropics/claude-code?ref=hn",
         "Claude Code", 0, 5_000_000),
    ])
    results = chromium.read_new(
        cursors={},
        overrides=[("chrome", "Default", db)],
    )
    assert len(results) == 1
    r = results[0]
    assert r.error is None
    assert len(r.events) == 1
    ev = r.events[0]
    assert ev.domain == "github.com"
    assert ev.path == "/anthropics/claude-code"
    # PRIVACY: ref=hn must be stripped
    assert "ref=hn" not in ev.path
    assert ev.title == "Claude Code"
    assert ev.visit_duration_ms == 5_000
    assert ev.transition == "link"
    assert ev.browser == "chrome"
    assert ev.profile == "Default"


def test_skips_non_http_schemes(tmp_path: Path) -> None:
    db = _build(tmp_path, [
        (_T_2026_05_16, "chrome://settings/", "Settings", 0, 0),
        (_T_2026_05_16 + 1, "https://github.com/", "GitHub", 0, 0),
    ])
    results = chromium.read_new(
        cursors={},
        overrides=[("chrome", "Default", db)],
    )
    domains = [e.domain for e in results[0].events]
    assert domains == ["github.com"]


def test_blocklisted_domain_skipped(tmp_path: Path) -> None:
    """PRIVACY PIN: a banking-domain row never produces an event."""
    db = _build(tmp_path, [
        (_T_2026_05_16, "https://paypal.com/account", "Account", 0, 0),
        (_T_2026_05_16 + 1, "https://github.com/", "GitHub", 0, 0),
    ])
    results = chromium.read_new(
        cursors={},
        overrides=[("chrome", "Default", db)],
    )
    r = results[0]
    domains = [e.domain for e in r.events]
    assert domains == ["github.com"]
    assert r.skipped_blocklisted == 1


def test_cursor_advances_past_max_visit_time(tmp_path: Path) -> None:
    db = _build(tmp_path, [
        (_T_2026_05_16, "https://a.com/", "A", 0, 0),
        (_T_2026_05_16 + 1000, "https://b.com/", "B", 0, 0),
    ])
    results = chromium.read_new(
        cursors={},
        overrides=[("chrome", "Default", db)],
    )
    assert results[0].last_cursor == _T_2026_05_16 + 1000


def test_cursor_filters_old_visits(tmp_path: Path) -> None:
    """Past-cursor visits must not reappear in the next pass."""
    db = _build(tmp_path, [
        (_T_2026_05_16, "https://a.com/", "A", 0, 0),
        (_T_2026_05_16 + 500, "https://b.com/", "B", 0, 0),
    ])
    results = chromium.read_new(
        cursors={"chrome:Default": _T_2026_05_16 + 100},
        overrides=[("chrome", "Default", db)],
    )
    domains = [e.domain for e in results[0].events]
    assert domains == ["b.com"]


def test_open_failure_returns_error_not_exception(tmp_path: Path) -> None:
    missing = tmp_path / "no-such-file"
    results = chromium.read_new(
        cursors={},
        overrides=[("chrome", "Default", missing)],
    )
    assert len(results) == 1
    assert results[0].error is not None
    assert results[0].events == []


def test_discover_profiles_finds_default_and_numbered(tmp_path: Path) -> None:
    home = tmp_path / "home"
    chrome_root = home / "Library/Application Support/Google/Chrome"
    for prof in ("Default", "Profile 1", "Profile 2", "System Profile"):
        (chrome_root / prof).mkdir(parents=True, exist_ok=True)
        (chrome_root / prof / "History").touch()
    out = chromium.discover_profiles("chrome", home=home)
    names = [name for name, _ in out]
    assert "Default" in names
    assert "Profile 1" in names
    assert "Profile 2" in names
    # System Profile is filtered out — no user history accrues there.
    assert "System Profile" not in names


def test_arc_uses_separate_root(tmp_path: Path) -> None:
    home = tmp_path / "home"
    arc_root = home / "Library/Application Support/Arc/User Data/Default"
    arc_root.mkdir(parents=True, exist_ok=True)
    (arc_root / "History").touch()
    out = chromium.discover_profiles("arc", home=home)
    assert [name for name, _ in out] == ["Default"]
