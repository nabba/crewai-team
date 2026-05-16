"""End-to-end aggregator tests with all three reader families."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from app.browse import aggregator, store
from tests.browse.conftest import (
    build_chromium_db,
    build_firefox_db,
    build_safari_db,
)


# Timestamps for 2026-05-16T12:00:00Z in each browser's epoch.
_CHROME_T = 13_423_406_400_000_000
_SAFARI_T = 800_625_600.0
_FIREFOX_T = 1_778_932_800_000_000


def _fake_home_with_dbs(tmp_path: Path) -> Path:
    home = tmp_path / "home"

    chrome_dir = home / "Library/Application Support/Google/Chrome/Default"
    chrome_dir.mkdir(parents=True, exist_ok=True)
    build_chromium_db(chrome_dir / "History", [
        (_CHROME_T, "https://github.com/x/y?ref=hn", "GH", 0, 1_000_000),
        (_CHROME_T + 1, "https://paypal.com/x", "PayPal", 0, 0),  # blocklisted
    ])

    safari_dir = home / "Library/Safari"
    safari_dir.mkdir(parents=True, exist_ok=True)
    build_safari_db(safari_dir / "History.db", [
        (_SAFARI_T, "https://en.wikipedia.org/wiki/Helsinki", "Helsinki"),
    ])

    ff_profile = home / "Library/Application Support/Firefox/Profiles/aa.default"
    ff_profile.mkdir(parents=True, exist_ok=True)
    build_firefox_db(ff_profile / "places.sqlite", [
        (_FIREFOX_T, "https://news.ycombinator.com/", "HN", 1),
    ])
    return home


def test_full_pass_persists_events(_reset_browse_state: Path, tmp_path: Path) -> None:
    home = _fake_home_with_dbs(tmp_path)
    result = aggregator.run_one_pass(home=home)
    assert result.status == "ok"
    assert result.total_events == 3  # paypal dropped at the blocklist
    assert result.total_skipped_blocklisted == 1
    assert result.written == 3

    events = store.list_events_for_day(date(2026, 5, 16))
    domains = sorted(e.domain for e in events)
    assert domains == [
        "en.wikipedia.org", "github.com", "news.ycombinator.com",
    ]


def test_full_pass_advances_cursors(_reset_browse_state: Path, tmp_path: Path) -> None:
    home = _fake_home_with_dbs(tmp_path)
    aggregator.run_one_pass(home=home)
    cursors = store.load_cursors()
    assert cursors["chrome:Default"] >= _CHROME_T
    # Safari cursor is stored in microseconds.
    assert cursors["safari"] >= int(_SAFARI_T * 1_000_000)
    assert cursors["firefox:aa.default"] >= _FIREFOX_T


def test_second_pass_no_new_events(_reset_browse_state: Path, tmp_path: Path) -> None:
    """After one pass advances the cursor, a re-run over the same DB
    must produce zero new events."""
    home = _fake_home_with_dbs(tmp_path)
    aggregator.run_one_pass(home=home)
    r2 = aggregator.run_one_pass(home=home)
    assert r2.status == "ok"
    assert r2.total_events == 0


def test_disabled_short_circuits(
    _reset_browse_state: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PRIVACY PIN: master switch off → no disk activity, status disabled."""
    monkeypatch.setenv("BROWSE_INGESTION_ENABLED", "false")
    home = _fake_home_with_dbs(tmp_path)
    result = aggregator.run_one_pass(home=home)
    assert result.status == "disabled"
    assert (_reset_browse_state / "events").exists() is False


def test_pass_isolates_broken_reader(
    _reset_browse_state: Path, tmp_path: Path,
) -> None:
    """A reader whose DB is missing should not block other readers."""
    home = tmp_path / "home"

    # Chrome OK
    chrome_dir = home / "Library/Application Support/Google/Chrome/Default"
    chrome_dir.mkdir(parents=True, exist_ok=True)
    build_chromium_db(chrome_dir / "History", [
        (_CHROME_T, "https://github.com/", "GH", 0, 0),
    ])

    # Safari: directory exists but DB is corrupt (write garbage bytes).
    safari_dir = home / "Library/Safari"
    safari_dir.mkdir(parents=True, exist_ok=True)
    (safari_dir / "History.db").write_bytes(b"not-a-sqlite-file")

    # Firefox absent entirely.

    result = aggregator.run_one_pass(home=home)
    assert result.status == "ok"
    # Chrome event landed even though Safari is broken.
    events = store.list_events_for_day(date(2026, 5, 16))
    assert [e.domain for e in events] == ["github.com"]
    # Safari error surfaces in the result for operator visibility.
    assert any(b == "safari" and err for b, _, err in result.errors)
