"""Source registry — per-workspace external research sources.

Phase 6 ships the data model + JSON sidecar persistence. Connectors
themselves live in ``app.companion.ingest``. The Source ``type`` field
selects which connector handles fetching:

  - ``web_search`` (Phase 6)  — query goes through the existing
    ``app.tools.web_search.search_brave`` cascade
  - ``rss``        (Phase 6.5+) — feedparser, deferred
  - ``url_poll``   (Phase 6.5+) — periodic GET, deferred

Storage is a sidecar JSON at
``workspace/companion/sources/<workspace_id>.json`` rather than CP
``config_json`` because sources mutate frequently and the runtime/durable
split keeps writes isolated. Reads/writes are atomic via temp + rename.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)

_SOURCES_DIR = Path(os.environ.get(
    "COMPANION_SOURCES_DIR", "workspace/companion/sources"))
_LOCK = Lock()

ALLOWED_TYPES = ("web_search",)


@dataclass
class Source:
    """One external research source bound to a workspace."""
    source_id: str = field(
        default_factory=lambda: f"src_{uuid.uuid4().hex[:12]}")
    type: str = "web_search"
    config: dict = field(default_factory=dict)
    enabled: bool = True
    added_at: float = field(default_factory=time.time)
    last_ingested_at: float = 0.0
    last_ingest_status: str = ""


def list_sources(workspace_id: str) -> list[Source]:
    """Read all sources for a workspace, oldest first. Empty if none."""
    p = _path_for(workspace_id)
    if not p.exists():
        return []
    try:
        raw = json.loads(p.read_text())
    except Exception as exc:
        logger.warning("companion.sources: load failed for %s: %s",
                       workspace_id, exc)
        return []
    out: list[Source] = []
    for item in raw or []:
        try:
            kwargs = {k: item[k] for k in Source.__dataclass_fields__
                      if k in item}
            out.append(Source(**kwargs))
        except Exception:
            continue
    return out


def add_source(workspace_id: str, type: str, config: dict | None = None,
               *, enabled: bool = True) -> Source | None:
    """Append a new source. Returns the persisted Source, or None on bad type."""
    if type not in ALLOWED_TYPES:
        logger.warning("companion.sources: rejected unknown type %r", type)
        return None
    src = Source(type=type, config=dict(config or {}), enabled=enabled)
    with _LOCK:
        sources = list_sources(workspace_id)
        sources.append(src)
        _save_all(workspace_id, sources)
    return src


def remove_source(workspace_id: str, source_id: str) -> bool:
    """Remove a source by id. Returns True if a row was deleted."""
    with _LOCK:
        sources = list_sources(workspace_id)
        before = len(sources)
        sources = [s for s in sources if s.source_id != source_id]
        if len(sources) == before:
            return False
        _save_all(workspace_id, sources)
    return True


def set_enabled(workspace_id: str, source_id: str, enabled: bool) -> bool:
    """Toggle ``enabled``. Returns True if the source was found."""
    with _LOCK:
        sources = list_sources(workspace_id)
        for s in sources:
            if s.source_id == source_id:
                s.enabled = bool(enabled)
                _save_all(workspace_id, sources)
                return True
    return False


def update_ingest_status(workspace_id: str, source_id: str, *,
                          ts: float, status: str = "ok") -> bool:
    """Stamp last_ingested_at + last_ingest_status. Returns True if found."""
    with _LOCK:
        sources = list_sources(workspace_id)
        for s in sources:
            if s.source_id == source_id:
                s.last_ingested_at = float(ts)
                s.last_ingest_status = (status or "")[:120]
                _save_all(workspace_id, sources)
                return True
    return False


def _path_for(workspace_id: str) -> Path:
    safe = "".join(c for c in workspace_id if c.isalnum() or c in "-_") \
        or "default"
    return _SOURCES_DIR / f"{safe}.json"


def _save_all(workspace_id: str, sources: list[Source]) -> None:
    p = _path_for(workspace_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps([asdict(s) for s in sources], indent=2, sort_keys=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=p.parent, delete=False,
        prefix=".tmp.", suffix=".json",
    ) as tmp:
        tmp.write(body)
        tmp_path = tmp.name
    os.replace(tmp_path, p)
