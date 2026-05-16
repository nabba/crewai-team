"""Aggregator: orchestrate readers → blocklist (already inside reader)
→ store → cursor advance.

The single public entry point :func:`run_one_pass` is what the idle
job calls. It's failure-isolated per reader — a broken Firefox read
doesn't take down the Safari + Chrome reads in the same pass.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from app.browse import store
from app.browse.models import BrowseEvent, ReaderResult
from app.browse.readers import chromium, firefox, safari

logger = logging.getLogger(__name__)


@dataclass
class PassResult:
    """Summary of one aggregator pass for telemetry / tests."""

    status: str  # "ok" | "disabled"
    reader_results: list[ReaderResult] = field(default_factory=list)
    written: int = 0

    @property
    def total_events(self) -> int:
        return sum(len(r.events) for r in self.reader_results)

    @property
    def total_skipped_blocklisted(self) -> int:
        return sum(r.skipped_blocklisted for r in self.reader_results)

    @property
    def errors(self) -> list[tuple[str, str | None, str]]:
        return [
            (r.browser, r.profile, r.error)
            for r in self.reader_results if r.error
        ]


def _collect_results(
    cursors: dict[str, int],
    *,
    home: Path | None = None,
) -> list[ReaderResult]:
    """Run all three readers; never raises. Each reader gets a
    try/except shield because a broken family shouldn't kill the pass."""
    out: list[ReaderResult] = []
    try:
        out.extend(chromium.read_new(cursors=cursors, home=home))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("browse.aggregator: chromium reader raised: %s", exc)
    try:
        safari_cursor_us = int(cursors.get("safari", 0))
        safari_cursor_s = safari_cursor_us / 1_000_000
        out.append(safari.read_new(cursor_s=safari_cursor_s, home=home))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("browse.aggregator: safari reader raised: %s", exc)
    try:
        out.extend(firefox.read_new(cursors=cursors, home=home))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("browse.aggregator: firefox reader raised: %s", exc)
    return out


def run_one_pass(
    *,
    home: Path | None = None,
    base: Path | None = None,
) -> PassResult:
    """Walk every browser past its cursor, persist new events, advance
    cursors. Honors the master switch — disabled returns immediately.

    Returns a :class:`PassResult` for telemetry. The aggregator never
    raises — it logs and reports via ``PassResult.errors``.
    """
    if not store.enabled():
        return PassResult(status="disabled")

    cursors = store.load_cursors(base=base)
    results = _collect_results(cursors, home=home)

    # Flatten events for persistence; record per-(browser,profile)
    # last-cursor for state advance.
    all_events: list[BrowseEvent] = []
    for r in results:
        all_events.extend(r.events)
    written = store.append_events(all_events, base=base)

    # Advance cursors only for readers that ran without an error.
    # Errors leave the previous cursor in place so the next pass retries
    # from the same point rather than skipping over rows the reader
    # might not have actually emitted.
    for r in results:
        if r.error is not None or r.last_cursor is None:
            continue
        store.save_cursor(r.browser, r.profile, r.last_cursor, base=base)

    return PassResult(status="ok", reader_results=results, written=written)
