"""Retention monitors — three independent cleanup jobs.

Closes resilience gap #8 by capping the unbounded growth of three
state stores. Each is registered as a separate monitor with its own
cadence; each cadence-guards internally so the daemon driver can
poll them at a uniform tick.

  * ``chromadb_retention``  — per-collection record cap.
  * ``worktree_retention``  — terminal coding-session worktrees > 7 d old.
  * ``attachment_retention`` — Signal attachments by age + total size.

All three are conservative by design — dry-run mode (``RETENTION_DRY_RUN=true``)
logs what would be deleted without actually deleting, recommended for
the first month after enabling. Every action is audited.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

from app.life_companion._common import (
    audit_event,
    background_enabled,
    read_state_json,
    send_signal_alert,
    write_state_json,
)

logger = logging.getLogger(__name__)


def _dry_run() -> bool:
    return os.getenv("RETENTION_DRY_RUN", "false").lower() in ("true", "1", "yes")


# ════════════════════════════════════════════════════════════════════════
# (a) ChromaDB record-cap retention
# ════════════════════════════════════════════════════════════════════════


_CHROMA_CADENCE_S = 7 * 24 * 3600   # weekly
_CHROMA_STATE_FILE = "chromadb_retention.json"
_CHROMA_DEFAULT_CAP = 100_000   # records per collection


def _chroma_cap_for(collection_name: str, defaults: dict) -> int:
    """Per-collection cap, or the global default."""
    raw = defaults.get(collection_name) or defaults.get("__default__")
    try:
        return int(raw) if raw is not None else _CHROMA_DEFAULT_CAP
    except (TypeError, ValueError):
        return _CHROMA_DEFAULT_CAP


def run_chromadb() -> None:
    """One ChromaDB retention pass — cadence-guarded internally."""
    if not background_enabled():
        return

    state = read_state_json(_CHROMA_STATE_FILE, {
        "last_run_at": 0.0,
        "caps": {"__default__": _CHROMA_DEFAULT_CAP},
    })
    now = time.time()
    if now - float(state.get("last_run_at", 0)) < _CHROMA_CADENCE_S:
        return
    state["last_run_at"] = now

    try:
        from app.memory.chromadb_manager import get_client
        client = get_client()
    except Exception:
        logger.debug("retention[chromadb]: client unavailable", exc_info=True)
        write_state_json(_CHROMA_STATE_FILE, state)
        return

    caps = state.get("caps") or {"__default__": _CHROMA_DEFAULT_CAP}
    summary = {"collections": [], "total_deleted": 0}

    try:
        collections = client.list_collections()
    except Exception:
        logger.debug("retention[chromadb]: list_collections failed", exc_info=True)
        write_state_json(_CHROMA_STATE_FILE, state)
        return

    for col in collections:
        try:
            name = col.name
            count = col.count()
        except Exception:
            continue
        cap = _chroma_cap_for(name, caps)
        if count <= cap:
            continue
        excess = count - cap

        # Find the oldest records by metadata.timestamp if available.
        # ChromaDB doesn't have direct "delete oldest" so we fetch all
        # ids + sort by timestamp metadata. Fail-soft if metadata
        # missing or schema unexpected.
        try:
            ids_to_delete = _oldest_ids(col, excess)
        except Exception:
            logger.debug(
                "retention[chromadb]: oldest_ids failed for %s", name,
                exc_info=True,
            )
            continue
        if not ids_to_delete:
            continue
        if _dry_run():
            summary["collections"].append({
                "name": name, "count": count, "cap": cap,
                "would_delete": len(ids_to_delete), "dry_run": True,
            })
            continue
        try:
            col.delete(ids=ids_to_delete)
            summary["total_deleted"] += len(ids_to_delete)
            summary["collections"].append({
                "name": name, "count": count, "cap": cap,
                "deleted": len(ids_to_delete),
            })
        except Exception:
            logger.debug(
                "retention[chromadb]: delete failed for %s", name,
                exc_info=True,
            )

    audit_event("chromadb_retention_pass", **{
        k: v for k, v in summary.items() if k != "collections"
    }, n_collections=len(summary["collections"]))
    state["last_summary"] = summary
    write_state_json(_CHROMA_STATE_FILE, state)


def _oldest_ids(collection, n: int) -> list[str]:
    """Return up to ``n`` IDs sorted oldest-first by metadata.timestamp.

    ChromaDB API: ``collection.get(include=["metadatas"])`` returns all
    records (cap on large collections; here we accept the O(N) cost
    because retention runs weekly).
    """
    try:
        rows = collection.get(include=["metadatas"])
    except Exception:
        return []
    ids = rows.get("ids") or []
    metas = rows.get("metadatas") or []
    if not ids:
        return []
    # Pair (id, ts) using metadata.timestamp if present, else 0 (oldest).
    pairs = []
    for i, m in zip(ids, metas):
        ts = 0.0
        if isinstance(m, dict):
            v = m.get("timestamp") or m.get("ts") or m.get("created_at")
            if isinstance(v, (int, float)):
                ts = float(v)
            elif isinstance(v, str):
                try:
                    from datetime import datetime
                    ts = datetime.fromisoformat(
                        v.replace("Z", "+00:00")
                    ).timestamp()
                except Exception:
                    ts = 0.0
        pairs.append((i, ts))
    pairs.sort(key=lambda p: p[1])  # oldest first
    return [p[0] for p in pairs[:n]]


# ════════════════════════════════════════════════════════════════════════
# (b) Coding-session worktree retention
# ════════════════════════════════════════════════════════════════════════


_WT_CADENCE_S = 24 * 3600   # daily
_WT_STATE_FILE = "worktree_retention.json"
_WT_AGE_S = 7 * 24 * 3600
_WT_TERMINAL_STATES = {"submitted", "discarded", "expired", "failed"}
_WT_LEAK_THRESHOLD = 50


def run_worktrees() -> None:
    """One coding-session worktree-retention pass — cadence-guarded."""
    if not background_enabled():
        return

    state = read_state_json(_WT_STATE_FILE, {"last_run_at": 0.0})
    now = time.time()
    if now - float(state.get("last_run_at", 0)) < _WT_CADENCE_S:
        return
    state["last_run_at"] = now

    store_dir = Path("/app/workspace/coding_sessions")
    if not store_dir.exists():
        write_state_json(_WT_STATE_FILE, state)
        return

    summary: dict[str, Any] = {
        "examined": 0,
        "removed": 0,
        "spared_active": 0,
        "spared_too_young": 0,
        "leak_alert": False,
    }

    for f in store_dir.glob("*.json"):
        if f.name == "audit.jsonl":
            continue
        summary["examined"] += 1
        try:
            data = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        status = (data.get("status") or "").lower()
        if status not in _WT_TERMINAL_STATES:
            summary["spared_active"] += 1
            continue
        # Use the file's own mtime as the "last_modified" proxy.
        try:
            mtime = f.stat().st_mtime
        except OSError:
            continue
        if now - mtime < _WT_AGE_S:
            summary["spared_too_young"] += 1
            continue

        # Found a stale terminal session — remove its worktree dir
        # (read from session record) AND its JSON record.
        worktree_path = data.get("worktree_path") or ""
        wt = Path(worktree_path) if worktree_path else None
        if _dry_run():
            summary["removed"] += 1  # accounting only
            continue
        try:
            if wt and wt.exists():
                shutil.rmtree(str(wt), ignore_errors=True)
        except Exception:
            logger.debug(
                "retention[worktrees]: rmtree failed for %s", wt,
                exc_info=True,
            )
        try:
            f.unlink()
            summary["removed"] += 1
        except OSError:
            logger.debug(
                "retention[worktrees]: unlink failed for %s", f,
                exc_info=True,
            )

    if summary["examined"] >= _WT_LEAK_THRESHOLD:
        summary["leak_alert"] = True
        send_signal_alert(
            f"🧪 Self-heal: {summary['examined']} coding-session records "
            f"under `workspace/coding_sessions/` — strong signal of a "
            f"leak (terminal sessions accumulating). Cleaned up "
            f"{summary['removed']} this pass; "
            f"{summary['spared_active']} still active.",
            tag="worktree_retention",
        )

    audit_event("worktree_retention_pass", **summary)
    state["last_summary"] = summary
    write_state_json(_WT_STATE_FILE, state)


# ════════════════════════════════════════════════════════════════════════
# (c) Signal-attachment retention
# ════════════════════════════════════════════════════════════════════════


_ATT_CADENCE_S = 24 * 3600   # daily
_ATT_STATE_FILE = "attachment_retention.json"
_ATT_AGE_S = 30 * 24 * 3600
_ATT_TOTAL_BYTES_CAP = 1024 ** 3   # 1 GB


def _attachments_dir() -> Path:
    """Location of Signal/inbound attachments. Env-tunable."""
    raw = os.getenv("SIGNAL_ATTACHMENTS_DIR", "/app/attachments")
    return Path(raw)


def run_attachments() -> None:
    """One Signal-attachment retention pass — cadence-guarded."""
    if not background_enabled():
        return

    state = read_state_json(_ATT_STATE_FILE, {"last_run_at": 0.0})
    now = time.time()
    if now - float(state.get("last_run_at", 0)) < _ATT_CADENCE_S:
        return
    state["last_run_at"] = now

    root = _attachments_dir()
    if not root.exists() or not root.is_dir():
        write_state_json(_ATT_STATE_FILE, state)
        return

    files: list[tuple[Path, float, int]] = []
    try:
        for p in root.iterdir():
            if not p.is_file():
                continue
            try:
                stat = p.stat()
            except OSError:
                continue
            files.append((p, stat.st_mtime, stat.st_size))
    except OSError:
        write_state_json(_ATT_STATE_FILE, state)
        return

    summary: dict[str, Any] = {
        "examined": len(files),
        "deleted_age": 0,
        "deleted_size": 0,
        "total_bytes_before": sum(s for _, _, s in files),
        "total_bytes_after": 0,
    }

    # ── Pass 1: age-based deletion ───────────────────────────────────
    surviving: list[tuple[Path, float, int]] = []
    for p, mtime, size in files:
        if now - mtime > _ATT_AGE_S:
            if not _dry_run():
                try:
                    p.unlink()
                except OSError:
                    surviving.append((p, mtime, size))
                    continue
            summary["deleted_age"] += 1
        else:
            surviving.append((p, mtime, size))

    # ── Pass 2: size-cap deletion (oldest first) ─────────────────────
    surviving.sort(key=lambda t: t[1])  # oldest first
    total = sum(s for _, _, s in surviving)
    while total > _ATT_TOTAL_BYTES_CAP and surviving:
        p, _, size = surviving.pop(0)
        if not _dry_run():
            try:
                p.unlink()
            except OSError:
                continue
        total -= size
        summary["deleted_size"] += 1

    summary["total_bytes_after"] = sum(s for _, _, s in surviving)
    audit_event("attachment_retention_pass", **summary)
    state["last_summary"] = summary
    write_state_json(_ATT_STATE_FILE, state)
