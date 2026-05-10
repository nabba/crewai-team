"""ActionRequest persistence — per-record JSON + lazy in-memory index.

Mirrors the change_requests + architecture_requests pattern:

    workspace/action_requests/<id>.json    — full ActionRequest serialised

Atomic writes via tempfile + rename. The on-disk JSON is the source
of truth; the in-memory ``_INDEX`` mirrors it for fast list queries.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from app.action_requests.models import ActionRequest, ActionStatus

logger = logging.getLogger(__name__)


_DEFAULT_BASE_DIR = Path("/app/workspace/action_requests")
_base_dir_override: Path | None = None
_LOCK = threading.RLock()
_INDEX: dict[str, ActionRequest] | None = None


def _base_dir() -> Path:
    return _base_dir_override or _DEFAULT_BASE_DIR


def get_base_dir() -> Path:
    return _base_dir()


def _ensure_dir() -> None:
    _base_dir().mkdir(parents=True, exist_ok=True)


def _record_path(request_id: str) -> Path:
    return _base_dir() / f"{request_id}.json"


def _index() -> dict[str, ActionRequest]:
    global _INDEX
    if _INDEX is not None:
        return _INDEX
    with _LOCK:
        if _INDEX is not None:
            return _INDEX
        _ensure_dir()
        loaded: dict[str, ActionRequest] = {}
        for f in _base_dir().glob("*.json"):
            try:
                req = ActionRequest.from_dict(json.loads(f.read_text()))
                loaded[req.id] = req
            except Exception as exc:  # noqa: BLE001
                logger.warning("action_requests: cannot load %s: %s", f, exc)
        _INDEX = loaded
        return _INDEX


def _persist(req: ActionRequest) -> None:
    _ensure_dir()
    path = _record_path(req.id)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(req.to_dict(), indent=2, default=str))
    tmp.replace(path)


def save(req: ActionRequest) -> None:
    with _LOCK:
        idx = _index()
        idx[req.id] = req
        _persist(req)


def get(request_id: str) -> ActionRequest | None:
    return _index().get(request_id)


def list_all(
    *,
    status: ActionStatus | None = None,
    limit: int = 100,
) -> list[ActionRequest]:
    items = list(_index().values())
    if status is not None:
        items = [r for r in items if r.status is status]
    items.sort(key=lambda r: r.created_at, reverse=True)
    return items[:limit]


def find_by_signal_ts(signal_ts: int) -> str | None:
    if not signal_ts:
        return None
    for req in _index().values():
        if req.signal_message_ts == signal_ts:
            return req.id
    return None


def reset_for_tests(base_dir: Path | None = None) -> None:
    global _INDEX, _base_dir_override
    with _LOCK:
        _INDEX = None
        _base_dir_override = base_dir
