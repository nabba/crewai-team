"""The system audit journal — a thin facade over rolled_log.

Single source of truth for the legacy ``audit_journal`` log instance.
Writers call :func:`append`, readers call :func:`read_recent` or
:func:`read_since`. Internal storage is the hash-chained rolled log
at ``<workspace>/audit_journal/``; the legacy single-file
``<workspace>/audit_journal.json`` is migrated lazily on first use.

Returned entries are *payloads* (the original event dicts the
auditor writes — ``{ts, event, detail, files_changed}``), not
rolled-log envelopes. Callers see exactly the shape they used to
read out of the legacy JSON file.

Probe path. The active segment of the rolled log is at
``<workspace>/audit_journal/current.jsonl``; that file's mtime is
the canonical heartbeat probe replacing
``<workspace>/audit_journal.json``'s mtime.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

from app.audit.migration import migrate_json_list
from app.audit.rolled_log import RolledLogReader, RolledLogStore

logger = logging.getLogger(__name__)

_LOG_NAME = "audit_journal"
_DEFAULT_RECENT_N = 200

_workspace_override: Path | None = None
_init_lock = threading.Lock()
_initialized = False


def _resolve_workspace() -> Path:
    if _workspace_override is not None:
        return _workspace_override
    from app.paths import WORKSPACE_ROOT
    return Path(WORKSPACE_ROOT)


def legacy_path() -> Path:
    """Path of the pre-migration single-file journal (becomes ``.preserved``)."""
    return _resolve_workspace() / "audit_journal.json"


def active_segment_path() -> Path:
    """Path whose mtime is the canonical liveness heartbeat for the auditor."""
    return _resolve_workspace() / _LOG_NAME / "current.jsonl"


def _ensure_initialized() -> None:
    global _initialized
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return
        ensure_migrated()
        _initialized = True


def ensure_migrated() -> dict:
    """If the legacy single file exists and no rolled log yet does, migrate.

    Idempotent: safe to call concurrently from multiple threads
    (process-local lock); cross-process safety holds because
    ``migrate_json_list`` is itself a single atomic rename + index
    write, and post-migration the legacy file no longer exists.
    """
    workspace = _resolve_workspace()
    log_dir = workspace / _LOG_NAME
    if (log_dir / "INDEX.json").exists():
        return {"status": "already_migrated", "n_entries": 0}
    legacy = legacy_path()
    if not legacy.exists():
        return {"status": "fresh_genesis", "n_entries": 0}
    try:
        return migrate_json_list(legacy, workspace, _LOG_NAME)
    except Exception:
        logger.warning(
            "audit.journal: migration failed; new entries go to a fresh log",
            exc_info=True,
        )
        return {"status": "migration_failed", "n_entries": 0}


def append(
    event: str,
    detail: str = "",
    files_changed: list[str] | None = None,
) -> int:
    """Append an audit entry. Returns the assigned sequence number."""
    _ensure_initialized()
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "detail": (detail or "")[:500],
        "files_changed": list(files_changed or []),
    }
    store = RolledLogStore(_resolve_workspace(), _LOG_NAME)
    return store.append(payload)


def read_recent(n: int = _DEFAULT_RECENT_N) -> list[dict]:
    """Last n audit-entry payloads in chronological order (newest last)."""
    if n <= 0:
        return []
    _ensure_initialized()
    reader = RolledLogReader(_resolve_workspace(), _LOG_NAME)
    return [env["payload"] for env in reader.recent(n)]


def read_since(cutoff_ts: datetime) -> list[dict]:
    """Return audit-entry payloads with payload.ts >= cutoff_ts.

    Chronological order. Skips entries whose ``ts`` field is missing
    or unparseable rather than failing the whole read.
    """
    _ensure_initialized()
    reader = RolledLogReader(_resolve_workspace(), _LOG_NAME)
    out: list[dict] = []
    for env in reader.iter_entries():
        payload = env.get("payload") or {}
        ts_raw = payload.get("ts") or ""
        ts_iso = ts_raw.replace("Z", "+00:00") if isinstance(ts_raw, str) else ""
        try:
            t = datetime.fromisoformat(ts_iso)
        except (ValueError, TypeError):
            continue
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        if t >= cutoff_ts:
            out.append(payload)
    return out


def stats() -> dict:
    """Underlying rolled-log stats (segment count, entry count, etc.)."""
    _ensure_initialized()
    return RolledLogStore(_resolve_workspace(), _LOG_NAME).stats()


def _reset_for_tests(workspace: Path | None = None) -> None:
    """Reset module state. Tests call this with a fresh tmp_path."""
    global _workspace_override, _initialized
    _workspace_override = workspace
    _initialized = False
