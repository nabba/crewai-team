"""ChangeRequest persistence layer.

Two stores:
  * ``workspace/change_requests/<id>.json`` — one file per request,
    full ChangeRequest serialized.
  * ``workspace/change_requests/audit.jsonl`` — append-only,
    hash-chained log of every state transition. Mirrors the Forge
    audit-log discipline (see ``app/forge/registry.py``).

Concurrency: the in-memory ``_PENDING`` index is locked on writes;
filesystem writes are atomic (write to a tempfile, then rename).
The audit log is append-with-flush — race-safe enough for the
expected single-writer (gateway) workload.

Read API (used by both the API endpoints and the Signal handler):
  * ``get(request_id)`` — single
  * ``list_all(status=None, limit=100)`` — filtered
  * ``find_by_signal_ts(ts)`` — Signal reaction handler hook
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.change_requests.models import ChangeRequest, Status

logger = logging.getLogger(__name__)


_STORE_DIR = Path("/app/workspace/change_requests")
_AUDIT_LOG = _STORE_DIR / "audit.jsonl"

_LOCK = threading.RLock()  # reentrant — save() holds it while calling _index()
# In-memory index: id → ChangeRequest. Repopulated lazily on first
# access. Filesystem is the source of truth.
_INDEX: dict[str, ChangeRequest] | None = None


def _ensure_dir() -> None:
    _STORE_DIR.mkdir(parents=True, exist_ok=True)


def _index() -> dict[str, ChangeRequest]:
    """Lazy-load the index from disk on first access."""
    global _INDEX
    if _INDEX is not None:
        return _INDEX
    with _LOCK:
        if _INDEX is not None:
            return _INDEX
        _ensure_dir()
        loaded: dict[str, ChangeRequest] = {}
        for f in _STORE_DIR.glob("*.json"):
            if f.name == "audit.jsonl":
                continue
            try:
                data = json.loads(f.read_text())
                cr = ChangeRequest.from_dict(data)
                loaded[cr.id] = cr
            except Exception as exc:
                logger.warning("change_requests: cannot load %s: %s", f, exc)
        _INDEX = loaded
        return _INDEX


# ── Audit log (hash-chained) ────────────────────────────────────────


def _last_audit_hash() -> str:
    """Read the last audit entry's hash. Empty string if no entries."""
    if not _AUDIT_LOG.exists():
        return ""
    try:
        # Read just the last line — efficient for large logs
        with _AUDIT_LOG.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size == 0:
                return ""
            # Tail the last ~4KB — should contain the last line for any
            # reasonable entry.
            tail_size = min(size, 4096)
            f.seek(size - tail_size)
            tail = f.read().decode("utf-8", errors="replace")
        last_line = [ln for ln in tail.split("\n") if ln.strip()][-1]
        entry = json.loads(last_line)
        return entry.get("entry_hash", "")
    except Exception as exc:
        logger.debug("audit log: cannot read last hash: %s", exc)
        return ""


def _append_audit(payload: dict[str, Any]) -> None:
    """Append a hash-chained entry to the audit log."""
    _ensure_dir()
    prev_hash = _last_audit_hash()
    body = json.dumps(payload, sort_keys=True, default=str)
    entry_hash = hashlib.sha256(
        (prev_hash + body).encode("utf-8"),
    ).hexdigest()[:16]
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "prev_hash": prev_hash,
        "entry_hash": entry_hash,
        "payload": payload,
    }
    try:
        with _AUDIT_LOG.open("a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception as exc:
        logger.warning("audit log: append failed: %s", exc)


# ── Per-record persistence ──────────────────────────────────────────


def _record_path(request_id: str) -> Path:
    return _STORE_DIR / f"{request_id}.json"


def _persist(cr: ChangeRequest) -> None:
    """Atomically write the ChangeRequest to disk. The on-disk JSON
    is the source of truth; the in-memory index mirrors it."""
    _ensure_dir()
    path = _record_path(cr.id)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cr.to_dict(), indent=2, default=str))
    tmp.replace(path)


# ── Public API ──────────────────────────────────────────────────────


def save(cr: ChangeRequest, *, audit_event: str | None = None) -> None:
    """Persist a ChangeRequest + optionally append an audit entry.

    Args:
        cr: The request to save (overwrites any prior version with
            the same id).
        audit_event: Short label for the transition that motivated
            the save, e.g. ``"created"``, ``"approved"``,
            ``"applied"``, ``"rolled_back"``. None means save without
            audit (e.g. for tests).
    """
    with _LOCK:
        idx = _index()
        idx[cr.id] = cr
        _persist(cr)
    if audit_event:
        _append_audit({
            "event": audit_event,
            "request_id": cr.id,
            "status": cr.status.value,
            "path": cr.path,
            "requestor": cr.requestor,
            "decided_by": cr.decided_by.value if cr.decided_by else None,
        })


def get(request_id: str) -> ChangeRequest | None:
    return _index().get(request_id)


def list_all(
    *,
    status: Status | None = None,
    limit: int = 100,
) -> list[ChangeRequest]:
    """List requests filtered by status. Newest first."""
    items = list(_index().values())
    if status is not None:
        items = [c for c in items if c.status is status]
    items.sort(key=lambda c: c.created_at, reverse=True)
    return items[:limit]


def find_by_signal_ts(signal_ts: int) -> str | None:
    """Look up a request by the Signal message timestamp it was sent
    on. Used by the reaction handler to dispatch 👍/👎 to the right
    request.

    Returns the request id, or None if no match.
    """
    if not signal_ts:
        return None
    for cr in _index().values():
        if cr.signal_message_ts == signal_ts:
            return cr.id
    return None


def reset_for_tests() -> None:
    """Clear the in-memory index. Tests use this with a tmp_path
    monkeypatch on _STORE_DIR to start fresh."""
    global _INDEX
    with _LOCK:
        _INDEX = None
