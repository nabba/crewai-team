"""Firefox history reader.

Firefox stores history in
``~/Library/Application Support/Firefox/Profiles/<random>.<profile>/places.sqlite``.
Profile discovery enumerates the ``Profiles/`` directory; ``profiles.ini``
isn't required (every subdir with a ``places.sqlite`` is a candidate).

Schema (Firefox 100+ stable)::

    moz_places(id, url, title, …, last_visit_date, …)
    moz_historyvisits(id, place_id, visit_date, visit_type, …)

``visit_date`` is microseconds since 1970-01-01 UTC (Unix epoch).
That's the cleanest of the three — no offset arithmetic needed.

``visit_type`` is a small int (1=link, 2=typed, …); we map the
common values to friendly labels.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from app.browse import blocklist
from app.browse.models import BrowseEvent, ReaderResult
from app.browse.url_canon import canonicalize, truncate_title

logger = logging.getLogger(__name__)


# Firefox transition codes (nsINavHistoryService::TransitionType)
_TRANSITION_TYPE: dict[int, str] = {
    1: "link",
    2: "typed",
    3: "bookmark",
    4: "embed",
    5: "redirect_permanent",
    6: "redirect_temporary",
    7: "download",
    8: "framed_link",
    9: "reload",
}


def _transition_label(raw: int | None) -> str | None:
    if raw is None:
        return None
    try:
        return _TRANSITION_TYPE.get(int(raw))
    except (TypeError, ValueError):
        return None


def _firefox_root(home: Path | None = None) -> Path:
    h = home if home is not None else Path.home()
    return h / "Library" / "Application Support" / "Firefox" / "Profiles"


def discover_profiles(
    *, home: Path | None = None,
) -> list[tuple[str, Path]]:
    """Return ``[(profile_dir_name, places.sqlite path)]`` for each
    Firefox profile."""
    root = _firefox_root(home=home)
    if not root.exists():
        return []
    out: list[tuple[str, Path]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        places = child / "places.sqlite"
        if places.is_file():
            out.append((child.name, places))
    return out


_QUERY = """
    SELECT v.visit_date, p.url, p.title, v.visit_type
    FROM moz_historyvisits v
    JOIN moz_places p ON p.id = v.place_id
    WHERE v.visit_date > ?
    ORDER BY v.visit_date ASC
    LIMIT ?
"""


def _open_readonly(path: Path) -> sqlite3.Connection:
    uri = f"file:{path}?mode=ro&immutable=1"
    return sqlite3.connect(uri, uri=True, timeout=2.0)


def _unix_us_to_iso(us: int) -> str:
    if us <= 0:
        return datetime(1970, 1, 1, tzinfo=timezone.utc).isoformat()
    return datetime.fromtimestamp(us / 1_000_000, tz=timezone.utc).isoformat()


def _read_one_db(
    profile: str,
    db_path: Path,
    *,
    cursor_us: int,
    limit: int,
) -> ReaderResult:
    try:
        conn = _open_readonly(db_path)
    except sqlite3.Error as exc:
        return ReaderResult(
            browser="firefox", profile=profile,
            error=f"open failed: {exc}",
        )
    events: list[BrowseEvent] = []
    skipped = 0
    last_cursor = cursor_us
    err: str | None = None
    try:
        for row in conn.execute(_QUERY, (cursor_us, limit)):
            visit_date, url, title, visit_type = row
            try:
                visit_date_i = int(visit_date)
            except (TypeError, ValueError):
                continue
            if visit_date_i > last_cursor:
                last_cursor = visit_date_i
            canon = canonicalize(url or "")
            if canon is None:
                continue
            if blocklist.is_blocked(canon.domain):
                skipped += 1
                continue
            events.append(BrowseEvent(
                visit_ts=_unix_us_to_iso(visit_date_i),
                domain=canon.domain,
                path=canon.path,
                title=truncate_title(title),
                browser="firefox",
                profile=profile,
                visit_duration_ms=None,
                transition=_transition_label(visit_type),
            ))
    except sqlite3.Error as exc:
        err = f"query failed: {exc}"
    finally:
        try:
            conn.close()
        except sqlite3.Error:
            pass
    return ReaderResult(
        browser="firefox",
        profile=profile,
        events=events,
        error=err,
        skipped_blocklisted=skipped,
        last_cursor=last_cursor,
    )


def read_new(
    *,
    cursors: dict[str, int],
    home: Path | None = None,
    limit_per_profile: int = 5000,
    overrides: Iterable[tuple[str, Path]] | None = None,
) -> list[ReaderResult]:
    """Read all Firefox profiles past their cursors.

    ``overrides`` lets tests inject ``(profile, path)`` pairs without
    populating ``~/Library/Application Support/Firefox/Profiles/...``.
    """
    if overrides is not None:
        sources = list(overrides)
    else:
        sources = list(discover_profiles(home=home))
    results: list[ReaderResult] = []
    for prof, db in sources:
        key = f"firefox:{prof}"
        cursor_us = int(cursors.get(key, 0))
        results.append(
            _read_one_db(
                prof, db, cursor_us=cursor_us, limit=limit_per_profile,
            )
        )
    return results
