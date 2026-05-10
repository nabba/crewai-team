"""Thread persistence — per-record JSON + lazy in-memory index.

Mirrors the change_requests + architecture_requests pattern:

    workspace/threads/<id>.json       — full Thread serialised

Atomic writes via tempfile + rename. The on-disk JSON is the source
of truth; the in-memory ``_INDEX`` mirrors it for fast list queries.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from app.threads.models import Thread, ThreadStatus

logger = logging.getLogger(__name__)


_DEFAULT_BASE_DIR = Path("/app/workspace/threads")
_base_dir_override: Path | None = None
_LOCK = threading.RLock()
_INDEX: dict[str, Thread] | None = None


def _base_dir() -> Path:
    return _base_dir_override or _DEFAULT_BASE_DIR


def get_base_dir() -> Path:
    return _base_dir()


def _ensure_dir() -> None:
    _base_dir().mkdir(parents=True, exist_ok=True)


def _record_path(thread_id: str) -> Path:
    return _base_dir() / f"{thread_id}.json"


def _index() -> dict[str, Thread]:
    global _INDEX
    if _INDEX is not None:
        return _INDEX
    with _LOCK:
        if _INDEX is not None:
            return _INDEX
        _ensure_dir()
        loaded: dict[str, Thread] = {}
        for f in _base_dir().glob("*.json"):
            try:
                t = Thread.from_dict(json.loads(f.read_text()))
                loaded[t.id] = t
            except Exception as exc:  # noqa: BLE001
                logger.warning("threads: cannot load %s: %s", f, exc)
        _INDEX = loaded
        return _INDEX


def _persist(thread: Thread) -> None:
    _ensure_dir()
    path = _record_path(thread.id)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(thread.to_dict(), indent=2, default=str))
    tmp.replace(path)


def save(thread: Thread) -> None:
    with _LOCK:
        idx = _index()
        idx[thread.id] = thread
        _persist(thread)


def get(thread_id: str) -> Thread | None:
    return _index().get(thread_id)


def list_all(*, limit: int = 100) -> list[Thread]:
    items = list(_index().values())
    items.sort(key=lambda t: t.last_touched_at or t.created_at, reverse=True)
    return items[:limit]


def list_open(*, limit: int = 100) -> list[Thread]:
    """Open + IN_PROGRESS + BLOCKED, newest activity first."""
    items = [t for t in _index().values() if not t.is_terminal]
    items.sort(key=lambda t: t.last_touched_at or t.created_at, reverse=True)
    return items[:limit]


def reset_for_tests(base_dir: Path | None = None) -> None:
    global _INDEX, _base_dir_override
    with _LOCK:
        _INDEX = None
        _base_dir_override = base_dir
