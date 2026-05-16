"""Chromium-family history reader (Chrome, Arc, Brave, Edge).

All four browsers share the same SQLite schema for ``History``::

    urls(id, url, title, visit_count, last_visit_time, …)
    visits(id, url, visit_time, transition, visit_duration, …)

``visit_time`` is microseconds since the Windows-FILETIME epoch
(1601-01-01 UTC), Chrome's idiosyncratic convention. We store the
cursor in that native unit to keep round-trip simple.

The reader opens the DB read-only with ``mode=ro&immutable=1`` URI
flags, which lets us read even while the browser holds the WAL —
the right behaviour for an idle-tick scanner that doesn't want to
wait for the user to close their browser.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Iterable

from app.browse import blocklist
from app.browse.models import BrowseEvent, ReaderResult
from app.browse.url_canon import canonicalize, truncate_title

logger = logging.getLogger(__name__)


# ── Path discovery ────────────────────────────────────────────────────


# (browser_id, root_dir_relative_to_~/Library/Application Support)
_FAMILY: tuple[tuple[str, str], ...] = (
    ("chrome", "Google/Chrome"),
    ("arc", "Arc/User Data"),
    ("brave", "BraveSoftware/Brave-Browser"),
    ("edge", "Microsoft Edge"),
)


def _root_for(browser: str, *, home: Path | None = None) -> Path:
    h = home if home is not None else Path.home()
    rel = next((r for b, r in _FAMILY if b == browser), None)
    if rel is None:
        raise ValueError(f"unknown chromium browser id: {browser}")
    return h / "Library" / "Application Support" / rel


def discover_profiles(
    browser: str, *, home: Path | None = None,
) -> list[tuple[str, Path]]:
    """Return ``[(profile_dir_name, history_db_path)]`` for every
    ``History`` file under the browser's install root.

    Multi-profile Chromium installs name profile dirs ``Default``,
    ``Profile 1``, ``Profile 2``, … and put a ``History`` SQLite file
    in each. A glob is enough — we don't need to parse ``Local State``.
    """
    root = _root_for(browser, home=home)
    if not root.exists():
        return []
    out: list[tuple[str, Path]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        if name != "Default" and not name.startswith("Profile "):
            # Some Chromium variants have additional system dirs
            # (``System Profile``, ``Guest Profile``); skip them — they
            # don't accrue user history.
            continue
        hist = child / "History"
        if hist.is_file():
            out.append((name, hist))
    return out


# ── Epoch conversion ─────────────────────────────────────────────────


# Microseconds between 1601-01-01 and 1970-01-01 (Chrome's epoch
# offset). 11644473600 seconds × 1e6.
_CHROME_EPOCH_OFFSET_US = 11644473600 * 1_000_000


def _chrome_us_to_iso(us: int) -> str:
    """Convert Chrome's ``visit_time`` to an ISO-8601 UTC string."""
    from datetime import datetime, timezone
    if us <= _CHROME_EPOCH_OFFSET_US:
        # Pre-1970 visit; should never happen in practice. Clamp to
        # epoch so we don't write malformed timestamps.
        return datetime(1970, 1, 1, tzinfo=timezone.utc).isoformat()
    sec = (us - _CHROME_EPOCH_OFFSET_US) / 1_000_000
    return datetime.fromtimestamp(sec, tz=timezone.utc).isoformat()


# Lower-byte mapping from ``visits.transition`` to a human label.
# See chromium/components/sessions/core/serialized_navigation_entry.h
_TRANSITION_TYPE: dict[int, str] = {
    0: "link",
    1: "typed",
    2: "auto_bookmark",
    3: "auto_subframe",
    4: "manual_subframe",
    5: "generated",
    6: "auto_toplevel",
    7: "form_submit",
    8: "reload",
    9: "keyword",
    10: "keyword_generated",
}


def _transition_label(raw: int | None) -> str | None:
    if raw is None:
        return None
    return _TRANSITION_TYPE.get(raw & 0xFF)


# ── Query ─────────────────────────────────────────────────────────────


_QUERY = """
    SELECT v.visit_time, u.url, u.title, v.transition, v.visit_duration
    FROM visits v
    JOIN urls u ON u.id = v.url
    WHERE v.visit_time > ?
    ORDER BY v.visit_time ASC
    LIMIT ?
"""


def _open_readonly(path: Path) -> sqlite3.Connection:
    """Open the SQLite DB read-only with ``immutable=1`` so the WAL
    doesn't block us when the browser is running.

    ``immutable=1`` tells SQLite to assume nothing else will write
    while we read — true enough during one snapshot pass."""
    uri = f"file:{path}?mode=ro&immutable=1"
    return sqlite3.connect(uri, uri=True, timeout=2.0)


def _read_one_db(
    browser: str,
    profile: str,
    db_path: Path,
    *,
    cursor_us: int,
    limit: int,
) -> ReaderResult:
    """Read new rows from one (browser, profile) history DB."""
    try:
        conn = _open_readonly(db_path)
    except sqlite3.Error as exc:
        return ReaderResult(
            browser=browser, profile=profile,
            error=f"open failed: {exc}",
        )
    events: list[BrowseEvent] = []
    skipped = 0
    last_cursor = cursor_us
    err: str | None = None
    try:
        for row in conn.execute(_QUERY, (cursor_us, limit)):
            visit_time, url, title, transition, duration = row
            try:
                visit_time = int(visit_time)
            except (TypeError, ValueError):
                continue
            if visit_time > last_cursor:
                last_cursor = visit_time
            canon = canonicalize(url or "")
            if canon is None:
                continue
            if blocklist.is_blocked(canon.domain):
                skipped += 1
                continue
            dur_ms = None
            try:
                if duration is not None:
                    dur_ms = int(int(duration) / 1000)
            except (TypeError, ValueError):
                dur_ms = None
            events.append(BrowseEvent(
                visit_ts=_chrome_us_to_iso(visit_time),
                domain=canon.domain,
                path=canon.path,
                title=truncate_title(title),
                browser=browser,
                profile=profile,
                visit_duration_ms=dur_ms,
                transition=_transition_label(transition),
            ))
    except sqlite3.Error as exc:
        err = f"query failed: {exc}"
    finally:
        try:
            conn.close()
        except sqlite3.Error:
            pass
    return ReaderResult(
        browser=browser,
        profile=profile,
        events=events,
        error=err,
        skipped_blocklisted=skipped,
        last_cursor=last_cursor,
    )


# ── Public API ────────────────────────────────────────────────────────


def read_new(
    *,
    cursors: dict[str, int],
    home: Path | None = None,
    limit_per_profile: int = 5000,
    overrides: Iterable[tuple[str, str, Path]] | None = None,
) -> list[ReaderResult]:
    """Read all Chromium-family history files past their cursors.

    ``cursors`` is the per-browser-profile last-seen ``visit_time`` in
    Chrome microsecond units. Keys formatted as ``"<browser>:<profile>"``.

    ``overrides`` lets tests inject ``(browser, profile, path)`` tuples
    directly without populating ``~/Library/Application Support/...``.
    """
    if overrides is not None:
        sources = list(overrides)
    else:
        sources = []
        for browser, _ in _FAMILY:
            for prof, db in discover_profiles(browser, home=home):
                sources.append((browser, prof, db))

    results: list[ReaderResult] = []
    for browser, prof, db in sources:
        key = f"{browser}:{prof}"
        cursor_us = int(cursors.get(key, 0))
        results.append(
            _read_one_db(
                browser, prof, db,
                cursor_us=cursor_us, limit=limit_per_profile,
            )
        )
    return results
