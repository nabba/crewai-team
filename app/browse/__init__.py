"""Browser-history ingestion (PROGRAM §50 — Q15.1).

Reads Safari + Chromium-family (Chrome, Arc, Brave, Edge) + Firefox
history SQLite databases at idle, canonicalises URLs (domain+path only
— no query strings, no fragments), filters through a per-domain
blocklist, and appends to per-day JSONL under ``workspace/browse/``.

Phase A scope
-------------

  * Three readers (Safari / Chromium-family / Firefox) reading their
    native SQLite stores read-only with ``mode=ro&immutable=1``.
  * URL canonicalisation that **structurally strips** query strings
    and fragments — privacy contract enforced at the type boundary.
  * Seeded blocklist (banking, health portals, auth) + operator-
    editable file.
  * Per-day JSONL store with explicit ``forget_all`` / ``forget_day``
    / ``forget_domain`` paths.
  * LIGHT idle job at 30-min cadence.
  * Identity continuity ledger emission on policy changes.

Phase A does NOT consume the events — no LLM batch, no interest-model
wiring, no daily-briefing section, no React surface. Those land in
Phase B+.

Privacy posture
---------------

  * ``BROWSE_INGESTION_ENABLED=false`` by default.
  * Private/incognito browsing is automatically excluded (browsers
    don't persist it to disk).
  * Query strings + fragments stripped at canonicalisation time; never
    stored.
  * Blocked domains never reach disk.
  * Titles preserved on-host but capped at 200 chars. Phase B's LLM
    batch (when enabled) is the first point where titles leave the host.
  * ``forget_all`` removes every event + cursor state without touching
    the operator's blocklist (mute history is durable independent of
    event history).

Public API
----------

  * :func:`app.browse.aggregator.run_one_pass` — one orchestrated pass.
  * :func:`app.browse.idle_job.run_browse_tick` — LIGHT idle entry point.
  * :func:`app.browse.idle_job.get_idle_jobs` — register with
    ``app.companion.loop.get_idle_jobs``.
  * :func:`app.browse.store.event_counts` — telemetry for React.
  * :func:`app.browse.store.forget_all` / ``forget_day`` / ``forget_domain``.
  * :func:`app.browse.blocklist.mute_domain` / ``is_blocked``.
"""
from app.browse.idle_job import get_idle_jobs as _ingest_jobs
from app.browse.idle_job import run_browse_tick
from app.browse.models import BrowseEvent, ReaderResult
from app.browse.topic_extraction import (
    get_idle_jobs as _topic_jobs,
)
from app.browse.topic_extraction import run_topic_extraction_tick


def get_idle_jobs():
    """Combined idle-job registration for both Phase A ingestion and
    Phase B daily topic extraction."""
    return [*_ingest_jobs(), *_topic_jobs()]


__all__ = [
    "BrowseEvent",
    "ReaderResult",
    "get_idle_jobs",
    "run_browse_tick",
    "run_topic_extraction_tick",
]
