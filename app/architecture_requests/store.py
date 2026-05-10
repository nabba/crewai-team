"""ArchitectureRequest persistence.

Two stores under ``workspace/architecture_requests/``:

* ``<id>.json`` — one file per request, full ``ArchitectureRequest``
  serialised. Atomic writes (tempfile + rename); the on-disk JSON
  is the source of truth, the in-memory ``_INDEX`` mirrors it.
* ``audit/`` (a directory) — rolled-segment hash-chained audit log
  via :class:`app.audit.rolled_log.RolledLogStore`. Every state
  transition appends one entry. The chain is verifiable across
  segment rotations; tampering is detectable.

The audit storage uses the same primitive as the auditor's journal
(C1+C2). New segments roll automatically when ``current.jsonl``
crosses the rotation threshold.

Read API (used by the lifecycle module + future API + Signal handler):
  * :func:`get` — single record by id
  * :func:`list_all` — filtered by status, newest-first
  * :func:`find_by_signal_ts` — Signal reaction handler hook
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.architecture_requests.models import ArchitectureRequest, ArchStatus
from app.audit.rolled_log import RolledLogReader, RolledLogStore

logger = logging.getLogger(__name__)


_DEFAULT_BASE_DIR = Path("/app/workspace/architecture_requests")
_AUDIT_LOG_NAME = "audit"

_base_dir_override: Path | None = None
_LOCK = threading.RLock()
_INDEX: dict[str, ArchitectureRequest] | None = None


def _base_dir() -> Path:
    return _base_dir_override or _DEFAULT_BASE_DIR


def get_base_dir() -> Path:
    """Public accessor for the active base dir.

    The scaffolder reads this so it stages files in the same root as
    the store's per-record JSON files. Tests inject a tmp_path via
    :func:`reset_for_tests`; production uses ``_DEFAULT_BASE_DIR``.
    """
    return _base_dir()


def _ensure_dir() -> None:
    _base_dir().mkdir(parents=True, exist_ok=True)


def _record_path(request_id: str) -> Path:
    return _base_dir() / f"{request_id}.json"


def _audit_store() -> RolledLogStore:
    return RolledLogStore(_base_dir(), _AUDIT_LOG_NAME)


def _audit_reader() -> RolledLogReader:
    return RolledLogReader(_base_dir(), _AUDIT_LOG_NAME)


def _index() -> dict[str, ArchitectureRequest]:
    global _INDEX
    if _INDEX is not None:
        return _INDEX
    with _LOCK:
        if _INDEX is not None:
            return _INDEX
        _ensure_dir()
        loaded: dict[str, ArchitectureRequest] = {}
        for f in _base_dir().glob("*.json"):
            try:
                data = json.loads(f.read_text())
                req = ArchitectureRequest.from_dict(data)
                loaded[req.id] = req
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "architecture_requests: cannot load %s: %s", f, exc,
                )
        _INDEX = loaded
        return _INDEX


def _persist(req: ArchitectureRequest) -> None:
    _ensure_dir()
    path = _record_path(req.id)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(req.to_dict(), indent=2, default=str))
    tmp.replace(path)


def _append_audit(payload: dict[str, Any]) -> None:
    try:
        _audit_store().append(payload)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "architecture_requests: audit append failed: %s", exc,
        )


def save(req: ArchitectureRequest, *, audit_event: str | None = None) -> None:
    """Persist a request and optionally append an audit entry.

    Args:
        req: the request to save (overwrites any prior version with the same id).
        audit_event: short label for the transition motivating the save
            (e.g. ``"created"``, ``"approved"``, ``"scaffolded"``).
            None means save without audit (for tests / non-transition writes).
    """
    with _LOCK:
        idx = _index()
        idx[req.id] = req
        _persist(req)
    if audit_event:
        _append_audit({
            "event": audit_event,
            "request_id": req.id,
            "status": req.status.value,
            "package_path": req.package_path,
            "requestor": req.requestor,
            "decided_by": req.decided_by.value if req.decided_by else None,
            "ts": datetime.now(timezone.utc).isoformat(),
        })


def get(request_id: str) -> ArchitectureRequest | None:
    return _index().get(request_id)


def list_all(
    *,
    status: ArchStatus | None = None,
    limit: int = 100,
) -> list[ArchitectureRequest]:
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


def iter_audit_entries():
    """Yield audit-log payloads in chronological order."""
    return (env["payload"] for env in _audit_reader().iter_entries())


def reset_for_tests(base_dir: Path | None = None) -> None:
    """Clear in-memory state. Tests pass a fresh tmp_path."""
    global _INDEX, _base_dir_override
    with _LOCK:
        _INDEX = None
        _base_dir_override = base_dir
