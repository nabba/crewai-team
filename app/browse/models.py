"""Typed records for browser-history ingestion.

A :class:`BrowseEvent` is one canonicalised, blocklist-cleared visit.
URLs are already truncated to ``scheme://domain/path`` (no query, no
fragment) by the time we instantiate one — see :mod:`app.browse.url_canon`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BrowseEvent:
    """One canonical browser visit."""

    visit_ts: str
    """ISO-8601 timestamp of the visit (UTC)."""

    domain: str
    """Lowercased eTLD-stripped host (``www.`` prefix dropped)."""

    path: str
    """URL path component without query string or fragment. May be ``/``."""

    title: str | None
    """Page title as the browser captured it, truncated to 200 chars.
    May be ``None`` when the browser didn't record one."""

    browser: str
    """Source browser identifier (``safari``, ``chrome``, ``arc``,
    ``brave``, ``edge``, ``firefox``)."""

    profile: str | None = None
    """For multi-profile Chromium browsers, the profile dir name
    (``Default``, ``Profile 1``…). ``None`` for single-profile browsers."""

    visit_duration_ms: int | None = None
    """Best-effort dwell estimate. Chromium gives us this; Safari and
    Firefox don't (set to ``None``)."""

    transition: str | None = None
    """How the user arrived (``link``, ``typed``, ``reload``, …).
    Best-effort; ``None`` when the source browser doesn't expose it."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "visit_ts": self.visit_ts,
            "domain": self.domain,
            "path": self.path,
            "title": self.title,
            "browser": self.browser,
            "profile": self.profile,
            "visit_duration_ms": self.visit_duration_ms,
            "transition": self.transition,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BrowseEvent":
        return cls(
            visit_ts=str(data["visit_ts"]),
            domain=str(data["domain"]),
            path=str(data.get("path", "")),
            title=data.get("title"),
            browser=str(data["browser"]),
            profile=data.get("profile"),
            visit_duration_ms=data.get("visit_duration_ms"),
            transition=data.get("transition"),
        )


@dataclass(frozen=True)
class ReaderResult:
    """What one reader pass returned for a single browser."""

    browser: str
    profile: str | None
    events: list[BrowseEvent] = field(default_factory=list)
    error: str | None = None
    """Non-fatal error description; events list may still be partial."""
    skipped_blocklisted: int = 0
    """How many rows were dropped by the blocklist (operator visibility)."""
    last_cursor: int | None = None
    """The newest visit timestamp (microseconds-since-epoch, Chromium
    convention) advanced past in this pass. Persist into the state file
    so next pass picks up here."""
