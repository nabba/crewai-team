"""Shared fixtures + synthetic SQLite builders for browse tests."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.browse import blocklist, store


@pytest.fixture(autouse=True)
def _reset_browse_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Per-test isolation: point the store at a fresh tmp dir + reset
    the blocklist cache so each test sees only the seeded defaults."""
    base = tmp_path / "browse"
    base.mkdir(parents=True, exist_ok=True)
    store._reset_for_tests(base)
    blocklist.reset_cache()
    monkeypatch.setenv("BROWSE_INGESTION_ENABLED", "true")
    # Identity ledger writes default-on; route it into the tmp tree.
    monkeypatch.setenv("IDENTITY_LEDGER_ENABLED", "false")
    yield base
    store._reset_for_tests(None)
    blocklist.reset_cache()


# ── Synthetic SQLite builders ─────────────────────────────────────────


def build_chromium_db(path: Path, rows: list[tuple]) -> None:
    """Build a minimal Chromium-shaped History DB.

    ``rows`` are ``(visit_time_us, url, title, transition, visit_duration_us)``.
    Use a visit_time_us > 11644473600000000 to land in the post-1970
    era (Chrome's epoch is 1601-01-01).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE urls (
                id INTEGER PRIMARY KEY,
                url TEXT,
                title TEXT
            );
            CREATE TABLE visits (
                id INTEGER PRIMARY KEY,
                url INTEGER,
                visit_time INTEGER,
                transition INTEGER,
                visit_duration INTEGER
            );
            """
        )
        for i, (visit_time, url, title, transition, duration) in enumerate(rows, start=1):
            conn.execute(
                "INSERT INTO urls (id, url, title) VALUES (?, ?, ?)",
                (i, url, title),
            )
            conn.execute(
                "INSERT INTO visits (id, url, visit_time, transition, visit_duration) "
                "VALUES (?, ?, ?, ?, ?)",
                (i, i, visit_time, transition, duration),
            )
        conn.commit()
    finally:
        conn.close()


def build_safari_db(path: Path, rows: list[tuple]) -> None:
    """Build a minimal Safari-shaped History.db.

    ``rows`` are ``(visit_time_s, url, title)``. ``visit_time_s`` is
    Mac CFAbsoluteTime (seconds since 2001-01-01).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE history_items (
                id INTEGER PRIMARY KEY,
                url TEXT
            );
            CREATE TABLE history_visits (
                id INTEGER PRIMARY KEY,
                history_item INTEGER,
                visit_time REAL,
                title TEXT
            );
            """
        )
        for i, (visit_time, url, title) in enumerate(rows, start=1):
            conn.execute(
                "INSERT INTO history_items (id, url) VALUES (?, ?)",
                (i, url),
            )
            conn.execute(
                "INSERT INTO history_visits (id, history_item, visit_time, title) "
                "VALUES (?, ?, ?, ?)",
                (i, i, visit_time, title),
            )
        conn.commit()
    finally:
        conn.close()


def build_firefox_db(path: Path, rows: list[tuple]) -> None:
    """Build a minimal Firefox-shaped places.sqlite.

    ``rows`` are ``(visit_date_us, url, title, visit_type)``.
    ``visit_date_us`` is microseconds since Unix epoch.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE moz_places (
                id INTEGER PRIMARY KEY,
                url TEXT,
                title TEXT
            );
            CREATE TABLE moz_historyvisits (
                id INTEGER PRIMARY KEY,
                place_id INTEGER,
                visit_date INTEGER,
                visit_type INTEGER
            );
            """
        )
        for i, (visit_date, url, title, visit_type) in enumerate(rows, start=1):
            conn.execute(
                "INSERT INTO moz_places (id, url, title) VALUES (?, ?, ?)",
                (i, url, title),
            )
            conn.execute(
                "INSERT INTO moz_historyvisits (id, place_id, visit_date, visit_type) "
                "VALUES (?, ?, ?, ?)",
                (i, i, visit_date, visit_type),
            )
        conn.commit()
    finally:
        conn.close()
