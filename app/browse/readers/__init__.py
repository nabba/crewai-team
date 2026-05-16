"""Browser-history readers.

Each reader has the same shape::

    read_new(*, base: Path | None = None) -> list[ReaderResult]

Returning one :class:`ReaderResult` per (browser, profile) discovered.
A reader that finds no DBs returns ``[]`` silently — it's normal for
the operator to only use a subset of the four supported browsers.

Discovery is path-based today (the canonical macOS install locations).
A reader that can't open its DB (locked, schema-mismatch, missing)
returns a result with ``error`` set; events list may still be partial.
"""
from app.browse.readers import chromium, firefox, safari

__all__ = ["chromium", "firefox", "safari"]
