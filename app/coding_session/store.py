"""CodingSession persistence layer.

Two stores, mirroring the change-requests pattern (and the Forge
audit-log discipline):

  * ``workspace/coding_sessions/<id>.json`` — one file per session,
    full ``CodingSession`` serialized.
  * ``workspace/coding_sessions/audit.jsonl`` — append-only,
    hash-chained log of every state transition.

Concurrency: ``RLock`` (not plain ``Lock``) because ``save()`` holds
the lock while calling ``_index()`` for the first-time lazy-load. The
change-requests store hit the same deadlock; we use the same fix here.

Filesystem writes are atomic (write to a tempfile, then rename). The
audit log is append-with-flush — race-safe enough for the expected
single-writer (gateway) workload.

Read API used by the manager + reconciler + control-plane API:

  * ``get(session_id)`` — single
  * ``list_all(*, status=None, agent_id=None, limit=200)`` — filtered
  * ``count_active(*, agent_id=None)`` — quota check helper
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

from app.coding_session.models import CodingSession, Status

logger = logging.getLogger(__name__)


_STORE_DIR = Path("/app/workspace/coding_sessions")
_AUDIT_LOG = _STORE_DIR / "audit.jsonl"

_LOCK = threading.RLock()  # reentrant — save() calls _index() under lock
# In-memory index: id → CodingSession. Lazy-loaded on first access.
_INDEX: dict[str, CodingSession] | None = None


def _ensure_dir() -> None:
    _STORE_DIR.mkdir(parents=True, exist_ok=True)


def _index() -> dict[str, CodingSession]:
    """Lazy-load the index from disk on first access."""
    global _INDEX
    if _INDEX is not None:
        return _INDEX
    with _LOCK:
        if _INDEX is not None:
            return _INDEX
        _ensure_dir()
        loaded: dict[str, CodingSession] = {}
        for f in _STORE_DIR.glob("*.json"):
            if f.name == "audit.jsonl":
                continue
            try:
                data = json.loads(f.read_text())
                cs = CodingSession.from_dict(data)
                loaded[cs.id] = cs
            except Exception as exc:
                logger.warning("coding_session: cannot load %s: %s", f, exc)
        _INDEX = loaded
        return _INDEX


# ── Audit log (hash-chained) ────────────────────────────────────────


def _last_audit_hash() -> str:
    """Read the last audit entry's hash. Empty string if no entries."""
    if not _AUDIT_LOG.exists():
        return ""
    try:
        with _AUDIT_LOG.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size == 0:
                return ""
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


def _record_path(session_id: str) -> Path:
    return _STORE_DIR / f"{session_id}.json"


def _persist(cs: CodingSession) -> None:
    """Atomically write the CodingSession to disk."""
    _ensure_dir()
    path = _record_path(cs.id)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cs.to_dict(), indent=2, default=str))
    tmp.replace(path)


# ── Public API ──────────────────────────────────────────────────────


def save(cs: CodingSession, *, audit_event: str | None = None) -> None:
    """Persist the CodingSession + optionally append an audit entry.

    Args:
        cs: The session to save (overwrites prior version with same id).
        audit_event: Short label for the transition that motivated the
            save (``"started"``, ``"submitted"``, ``"discarded"``,
            ``"expired"``, ``"touched"``). None means save without audit.
    """
    with _LOCK:
        idx = _index()
        idx[cs.id] = cs
        _persist(cs)
    if audit_event:
        _append_audit({
            "event": audit_event,
            "session_id": cs.id,
            "agent_id": cs.agent_id,
            "status": cs.status.value,
            "base": cs.base,
            "base_sha": cs.base_sha,
            "worktree_path": cs.worktree_path,
            "files_touched_count": len(cs.files_touched),
            "bytes_written": cs.bytes_written,
            "terminated_reason": cs.terminated_reason,
        })


def get(session_id: str) -> CodingSession | None:
    return _index().get(session_id)


def list_all(
    *,
    status: Status | None = None,
    agent_id: str | None = None,
    limit: int = 200,
) -> list[CodingSession]:
    """List sessions filtered by status and/or agent. Newest first."""
    items = list(_index().values())
    if status is not None:
        items = [s for s in items if s.status is status]
    if agent_id is not None:
        items = [s for s in items if s.agent_id == agent_id]
    items.sort(key=lambda s: s.created_at, reverse=True)
    return items[:limit]


def count_active(*, agent_id: str | None = None) -> int:
    """Count ACTIVE sessions, optionally filtered by agent.

    Used by the quota module — fast scan in O(N) over the index, but
    N is bounded (system cap = 20 active sessions).
    """
    items = _index().values()
    n = 0
    for s in items:
        if s.status is not Status.ACTIVE:
            continue
        if agent_id is not None and s.agent_id != agent_id:
            continue
        n += 1
    return n


def reset_for_tests() -> None:
    """Clear the in-memory index. Tests use this with monkeypatch on
    ``_STORE_DIR`` + ``_AUDIT_LOG`` to start fresh."""
    global _INDEX
    with _LOCK:
        _INDEX = None
