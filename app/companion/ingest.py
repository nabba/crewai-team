"""Source ingestion — fetch fresh material into the companion_sources KB.

Registered with ``app.idle_scheduler`` as a LIGHT-weight job. Each tick
walks every active workspace's enabled sources, skips any source ingested
within the last ``MIN_REINGEST_S``, fetches new items via the source's
connector, and indexes them in the ChromaDB ``companion_sources``
collection (one row per item, ``workspace_id`` + ``source_id`` in
metadata so workspace-scoped retrieval is cheap).

Connectors per source.type:
  - ``web_search`` → ``app.tools.web_search.search_brave`` cascade
  - others (rss, url_poll) — Phase 6.5+

All failures are logged and absorbed: a broken backend, missing chromadb,
or quota-exhausted source must never crash the idle thread.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Callable

from app.companion import sources as _sources

logger = logging.getLogger(__name__)

COMPANION_SOURCES_COLLECTION = "companion_sources"
MIN_REINGEST_S = 12 * 3600
WEB_SEARCH_COUNT = 8


def run_ingest() -> None:
    """Ingest for every active companion-enabled workspace."""
    try:
        rows = _list_projects()
    except Exception as exc:
        logger.debug("companion.ingest: list_projects failed: %s", exc)
        return
    now = time.time()
    for row in rows:
        pid = row.get("id")
        if not pid:
            continue
        cfg_raw = (row.get("config_json") or {}).get("companion") or {}
        if cfg_raw.get("enabled") is False:
            continue
        try:
            ingest_for_workspace(pid, now=now)
        except Exception as exc:
            logger.warning("companion.ingest: failed for %s: %s", pid, exc)


def ingest_for_workspace(workspace_id: str, *,
                          now: float | None = None) -> int:
    """Run all due sources for one workspace. Returns total items indexed."""
    if now is None:
        now = time.time()
    items_indexed = 0
    for src in _sources.list_sources(workspace_id):
        if not src.enabled:
            continue
        if (now - src.last_ingested_at) < MIN_REINGEST_S:
            continue
        try:
            items = _fetch(src)
        except Exception as exc:
            logger.debug("companion.ingest: fetch failed for %s/%s: %s",
                         workspace_id, src.source_id, exc)
            _sources.update_ingest_status(
                workspace_id, src.source_id,
                ts=now, status=f"error: {type(exc).__name__}",
            )
            continue
        for item in items:
            try:
                if _index_item(workspace_id, src.source_id, item):
                    items_indexed += 1
            except Exception as exc:
                logger.debug("companion.ingest: index failed: %s", exc)
        _sources.update_ingest_status(
            workspace_id, src.source_id, ts=now, status="ok",
        )
    return items_indexed


def get_idle_jobs() -> list[tuple[str, Callable[[], None], str]]:
    """Idle-scheduler tuple — appended in ``_default_jobs()``."""
    from app.idle_scheduler import JobWeight
    return [("companion-ingest", run_ingest, JobWeight.LIGHT)]


def _fetch(source: _sources.Source) -> list[dict]:
    """Dispatch to a connector based on ``source.type``."""
    if source.type == "web_search":
        return _fetch_web_search(source)
    logger.debug("companion.ingest: unknown source type %r", source.type)
    return []


def _fetch_web_search(source: _sources.Source) -> list[dict]:
    query = (source.config or {}).get("query", "").strip()
    if not query:
        return []
    return _invoke_search(query, WEB_SEARCH_COUNT)


def _invoke_search(query: str, count: int) -> list[dict]:
    """Indirection over ``search_brave`` (cascade) for testability."""
    from app.tools.web_search import search_brave
    return search_brave(query, count=count)


def _index_item(workspace_id: str, source_id: str, item: dict) -> bool:
    """Upsert one item into companion_sources. Returns True when written."""
    title = (item.get("title") or "").strip()
    body = (item.get("description") or item.get("snippet") or "").strip()
    url = (item.get("url") or "").strip()
    text = f"{title}\n{body}".strip()
    if not text:
        return False
    item_id = _stable_id(workspace_id, source_id, url or text)
    return _index_chromadb(item_id, text, {
        "workspace_id": workspace_id,
        "source_id": source_id,
        "url": url,
        "title": title[:300],
        "fetched_at": time.time(),
    })


def _index_chromadb(item_id: str, text: str, metadata: dict) -> bool:
    """Indirection over the ChromaDB upsert, for testability."""
    try:
        from app.memory.chromadb_manager import _get_col, embed
        col = _get_col(COMPANION_SOURCES_COLLECTION)
        col.upsert(
            ids=[item_id],
            documents=[text],
            embeddings=[embed(text)],
            metadatas=[metadata],
        )
        return True
    except Exception as exc:
        logger.debug("companion.ingest: chromadb upsert failed: %s", exc)
        return False


def _stable_id(workspace_id: str, source_id: str, key: str) -> str:
    h = hashlib.sha1(
        f"{workspace_id}|{source_id}|{key}".encode("utf-8"),
    ).hexdigest()[:16]
    return f"src_{h}"


def _list_projects() -> list[dict]:
    """Indirection over the CP project listing — same seam as scheduler."""
    from app.control_plane.projects import get_projects
    return get_projects().list_all() or []
