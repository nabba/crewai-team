"""Safari history reader.

Safari stores history in ``~/Library/Safari/History.db``. Reading
requires Full Disk Access for the gateway process.

Schema (Safari 17, stable since ~2018)::

    history_items(id, url, domain_expansion, visit_count, …)
    history_visits(id, history_item, visit_time, title,
                   load_successful, …, redirect_source,
                   redirect_destination, …)

``visit_time`` is seconds since 2001-01-01 UTC (Mac CFAbsoluteTime).
We store the cursor in those native units to keep round-trip simple.

There is no per-visit transition kind exposed (Safari's table has
``origin`` but it's coarse), and no dwell-time column. The
``visit_duration_ms`` + ``transition`` fields are ``None`` for
Safari events.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from app.browse import blocklist
from app.browse.models import BrowseEvent, ReaderResult
from app.browse.url_canon import canonicalize, truncate_title

logger = logging.getLogger(__name__)


# Mac CFAbsoluteTime to Unix epoch offset in seconds.
# 2001-01-01 UTC - 1970-01-01 UTC = 978307200 s.
_MAC_EPOCH_OFFSET_S = 978_307_200


def _safari_s_to_iso(s: float) -> str:
    if s <= 0:
        return datetime(1970, 1, 1, tzinfo=timezone.utc).isoformat()
    unix_s = s + _MAC_EPOCH_OFFSET_S
    return datetime.fromtimestamp(unix_s, tz=timezone.utc).isoformat()


def history_path(home: Path | None = None) -> Path:
    h = home if home is not None else Path.home()
    return h / "Library" / "Safari" / "History.db"


_QUERY = """
    SELECT v.visit_time, i.url, v.title
    FROM history_visits v
    JOIN history_items i ON i.id = v.history_item
    WHERE v.visit_time > ?
    ORDER BY v.visit_time ASC
    LIMIT ?
"""


def _open_readonly(path: Path) -> sqlite3.Connection:
    uri = f"file:{path}?mode=ro&immutable=1"
    return sqlite3.connect(uri, uri=True, timeout=2.0)


def read_new(
    *,
    cursor_s: float = 0.0,
    home: Path | None = None,
    limit: int = 5000,
    override_path: Path | None = None,
) -> ReaderResult:
    """Read all new history rows past ``cursor_s`` (Mac-epoch seconds).

    Returns one :class:`ReaderResult`. Safari is single-profile so
    ``profile`` is always ``None``.
    """
    path = override_path if override_path is not None else history_path(home)
    if not path.exists():
        return ReaderResult(browser="safari", profile=None)
    try:
        conn = _open_readonly(path)
    except sqlite3.Error as exc:
        return ReaderResult(
            browser="safari", profile=None,
            error=f"open failed: {exc}",
        )
    events: list[BrowseEvent] = []
    skipped = 0
    last_cursor = cursor_s
    err: str | None = None
    try:
        for row in conn.execute(_QUERY, (cursor_s, limit)):
            visit_time, url, title = row
            try:
                visit_time_f = float(visit_time)
            except (TypeError, ValueError):
                continue
            if visit_time_f > last_cursor:
                last_cursor = visit_time_f
            canon = canonicalize(url or "")
            if canon is None:
                continue
            if blocklist.is_blocked(canon.domain):
                skipped += 1
                continue
            events.append(BrowseEvent(
                visit_ts=_safari_s_to_iso(visit_time_f),
                domain=canon.domain,
                path=canon.path,
                title=truncate_title(title),
                browser="safari",
                profile=None,
                visit_duration_ms=None,
                transition=None,
            ))
    except sqlite3.Error as exc:
        err = f"query failed: {exc}"
    finally:
        try:
            conn.close()
        except sqlite3.Error:
            pass
    # Cursor is stored as int microseconds in state.json for uniformity
    # across readers — Safari uses seconds natively, scale up.
    cursor_int = int(round(last_cursor * 1_000_000))
    return ReaderResult(
        browser="safari",
        profile=None,
        events=events,
        error=err,
        skipped_blocklisted=skipped,
        last_cursor=cursor_int,
    )
