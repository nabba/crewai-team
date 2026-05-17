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
        # ids + sort by timestamp metadata. Discipline (post-2026-05-16):
        # records WITHOUT a parseable timestamp are EXCLUDED, not
        # treated as "oldest via zero-fallback" — see _oldest_ids.
        try:
            ids_to_delete, ts_stats = _oldest_ids(col, excess)
        except Exception:
            logger.debug(
                "retention[chromadb]: oldest_ids failed for %s", name,
                exc_info=True,
            )
            continue

        # If most records lack timestamps, retention can't safely
        # enforce the cap. Surface this so the operator can decide
        # (raise the cap, fix the writer, or accept the accumulation).
        if ts_stats["total"] > 0 and ts_stats["without_ts"] > ts_stats["with_ts"]:
            logger.warning(
                "retention[chromadb]: %s has %d records without timestamp "
                "metadata vs %d with — cap cannot be safely enforced; "
                "writer should populate `timestamp` / `ts` / `created_at`",
                name, ts_stats["without_ts"], ts_stats["with_ts"],
            )

        if not ids_to_delete:
            summary["collections"].append({
                "name": name, "count": count, "cap": cap,
                "deleted": 0, "ts_stats": ts_stats,
                "reason": "no_candidates_with_timestamp"
                    if ts_stats["without_ts"] > 0 else "below_cap_after_filter",
            })
            continue

        if _dry_run():
            summary["collections"].append({
                "name": name, "count": count, "cap": cap,
                "would_delete": len(ids_to_delete), "dry_run": True,
                "ts_stats": ts_stats,
            })
            continue
        try:
            col.delete(ids=ids_to_delete)
            # PROGRAM §56 iter-2 — ledger tombstone. Retention deletes
            # are large bulk operations; we hook them so replay doesn't
            # resurrect the rows the retention monitor pruned.
            try:
                from app.memory.source_ledger import hook_collection_delete
                # KB inference: this retention monitor scans every KB,
                # not just memory. The current pass's KB is the parent
                # of the chroma client's path. Best-effort.
                kb_name_inferred = "memory"
                try:
                    from pathlib import Path as _P
                    cpath = getattr(getattr(col, "_client", None), "_persist_directory", "") or ""
                    if cpath:
                        kb_name_inferred = _P(str(cpath)).name
                except Exception:
                    pass
                hook_collection_delete(kb_name_inferred, name, list(ids_to_delete))
            except Exception:
                pass
            summary["total_deleted"] += len(ids_to_delete)
            summary["collections"].append({
                "name": name, "count": count, "cap": cap,
                "deleted": len(ids_to_delete),
                "ts_stats": ts_stats,
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


def _oldest_ids(
    collection, n: int,
) -> tuple[list[str], dict]:
    """Return up to ``n`` IDs sorted oldest-first by metadata.timestamp,
    PLUS a stats dict describing what was filtered.

    ChromaDB API: ``collection.get(include=["metadatas"])`` returns all
    records (cap on large collections; here we accept the O(N) cost
    because retention runs weekly).

    Discipline (post-2026-05-16): records lacking a parseable timestamp
    metadata are EXCLUDED from deletion candidates, not classified as
    "oldest" via a silent zero-fallback. The earlier behaviour treated
    timestamp-less records as preferentially deletable, which is the
    same shape as the chromadb_hygiene incident (silent classification
    fallback → automatic destructive action). If most records lack
    timestamps, retention does nothing this pass and the audit row
    surfaces it.
    """
    stats = {"total": 0, "with_ts": 0, "without_ts": 0, "selected": 0}
    try:
        rows = collection.get(include=["metadatas"])
    except Exception:
        return [], stats
    ids = rows.get("ids") or []
    metas = rows.get("metadatas") or []
    stats["total"] = len(ids)
    if not ids:
        return [], stats

    pairs: list[tuple[str, float]] = []
    for i, m in zip(ids, metas):
        ts = _parse_ts_from_metadata(m)
        if ts is None:
            stats["without_ts"] += 1
            continue
        stats["with_ts"] += 1
        pairs.append((i, ts))

    pairs.sort(key=lambda p: p[1])  # oldest first
    selected = [p[0] for p in pairs[:n]]
    stats["selected"] = len(selected)
    return selected, stats


def _parse_ts_from_metadata(m) -> float | None:
    """Extract a parseable timestamp from a chromadb metadata dict.

    Returns ``None`` if no recognised timestamp field is present or if
    the value can't be parsed. The None-vs-zero distinction is
    load-bearing: callers MUST NOT treat absent-timestamp records as
    "oldest" — see ``_oldest_ids`` docstring."""
    if not isinstance(m, dict):
        return None
    v = m.get("timestamp")
    if v is None:
        v = m.get("ts")
    if v is None:
        v = m.get("created_at")
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            from datetime import datetime
            return datetime.fromisoformat(
                v.replace("Z", "+00:00")
            ).timestamp()
        except Exception:
            return None
    return None


# ════════════════════════════════════════════════════════════════════════
# (b) Coding-session worktree retention
# ════════════════════════════════════════════════════════════════════════


_WT_CADENCE_S = 24 * 3600   # daily
_WT_STATE_FILE = "worktree_retention.json"
_WT_AGE_S = 7 * 24 * 3600
_WT_TERMINAL_STATES = {"submitted", "discarded", "expired", "failed"}
_WT_LEAK_THRESHOLD = 50


def _worktree_root() -> str:
    """Where new worktrees are created (matches app.coding_session.runtime)."""
    return os.environ.get("CODING_SESSION_WORKTREE_ROOT") or "/tmp/agent-sessions"


def _validate_worktree_path(
    wt_path: str, *, expected_session_id: str,
) -> tuple[bool, str]:
    """Validate that ``wt_path`` is safe to ``shutil.rmtree``.

    Returns ``(is_valid, reason)``. The caller MUST NOT rmtree unless
    ``is_valid`` is True. This is the structural guard against the
    2026-05-16 class of bug (silent classification fallback → automatic
    destructive action against unintended targets).

    Refuses if:
      * path is empty or not a string
      * path is not absolute (relative paths resolve against cwd → could
        hit anywhere)
      * path is not under ``worktree_root()`` (with separator; defends
        against suffix attacks like ``/tmp/agent-sessions-evil``)
      * path's basename doesn't match ``expected_session_id`` (defends
        against session JSONs that point at someone else's worktree)
      * path contains symlink components that escape worktree_root
        (defends against ``/tmp/agent-sessions/abc/../../../etc``)
    """
    if not wt_path or not isinstance(wt_path, str):
        return False, "empty_or_non_string_path"
    p = Path(wt_path)
    if not p.is_absolute():
        return False, f"relative_path: {wt_path!r}"
    root = _worktree_root().rstrip("/")
    # Resolve symlinks to defend against escapes via symlinks INSIDE
    # the worktree. If the worktree dir itself is a symlink (unusual
    # but possible) we resolve through it; what matters is the target.
    try:
        resolved = p.resolve(strict=False)
    except (OSError, RuntimeError):
        return False, "path_resolve_failed"
    resolved_str = str(resolved)
    root_with_sep = root + "/"
    if not resolved_str.startswith(root_with_sep):
        return False, (
            f"path_outside_worktree_root: resolved={resolved_str!r} "
            f"not under {root_with_sep!r}"
        )
    # Basename must match the session ID — defends against a corrupted
    # session JSON pointing at someone else's worktree dir.
    if resolved.name != expected_session_id:
        return False, (
            f"basename_mismatch: {resolved.name!r} vs session_id "
            f"{expected_session_id!r}"
        )
    # Final structural check: exactly one level under worktree_root
    # (no nested paths like /tmp/agent-sessions/abc/def).
    try:
        rel = resolved.relative_to(root)
        if len(rel.parts) != 1:
            return False, (
                f"path_depth_unexpected: {len(rel.parts)} segments under "
                f"worktree_root, expected 1"
            )
    except ValueError:
        return False, "relative_to_root_failed"
    return True, "ok"


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
        #
        # Discipline (post-2026-05-16): validate worktree_path BEFORE
        # rmtree. Earlier behaviour was `shutil.rmtree(data.get('worktree_path'))`
        # with only an `if wt and wt.exists()` guard — a corrupted session
        # JSON with a stale or out-of-tree path would have rmtree'd the
        # wrong target. Now we require absolute, inside worktree_root,
        # basename matching the session ID, and exactly one level deep.
        worktree_path = data.get("worktree_path") or ""
        session_id = data.get("id") or f.stem
        if _dry_run():
            summary["removed"] += 1  # accounting only
            continue
        is_valid, reason = _validate_worktree_path(
            worktree_path, expected_session_id=session_id,
        )
        if not is_valid:
            summary.setdefault("refused_validation", 0)
            summary["refused_validation"] += 1
            logger.warning(
                "retention[worktrees]: refusing rmtree for session %s — %s "
                "(see _validate_worktree_path docstring)",
                session_id, reason,
            )
            # Leave the session JSON in place so the operator can inspect.
            continue
        try:
            wt = Path(worktree_path)
            if wt.exists():
                shutil.rmtree(str(wt), ignore_errors=True)
        except Exception:
            logger.debug(
                "retention[worktrees]: rmtree failed for %s", worktree_path,
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

# Discipline (post-2026-05-16): refuse to operate on env-set
# SIGNAL_ATTACHMENTS_DIR unless it resolves to a known-safe prefix.
# The retention sweep deletes files; an env mis-set could nuke
# anything writable from inside the container.
_ATT_SAFE_PREFIXES: tuple[str, ...] = (
    "/app/attachments",
    "/app/workspace/",
    "/tmp/",
)


def _attachments_dir() -> Path:
    """Location of Signal/inbound attachments. Env-tunable."""
    raw = os.getenv("SIGNAL_ATTACHMENTS_DIR", "/app/attachments")
    return Path(raw)


def _is_attachments_dir_safe(p: Path) -> tuple[bool, str]:
    """Return ``(is_safe, reason)``. Caller MUST NOT delete files
    inside ``p`` unless ``is_safe`` is True.

    The check refuses any path that isn't a sub-tree of one of
    ``_ATT_SAFE_PREFIXES``. Refusal is the safe default — an operator
    who really wants a custom attachments dir can either:

      * move it inside ``/app/workspace/attachments/`` (the standard
        location), or
      * extend ``_ATT_SAFE_PREFIXES`` here with a code change.
    """
    if not p.is_absolute():
        return False, f"relative_path: {p!s}"
    try:
        resolved = p.resolve(strict=False)
    except (OSError, RuntimeError):
        return False, "path_resolve_failed"
    resolved_str = str(resolved)
    for prefix in _ATT_SAFE_PREFIXES:
        if resolved_str == prefix.rstrip("/"):
            return True, "ok"
        if resolved_str.startswith(prefix if prefix.endswith("/") else prefix + "/"):
            return True, "ok"
    return False, (
        f"attachments_dir_not_in_safe_prefix: {resolved_str!r}; "
        f"allowed prefixes: {_ATT_SAFE_PREFIXES}"
    )


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
    # Discipline (post-2026-05-16): structural guard against env mis-set.
    is_safe, reason = _is_attachments_dir_safe(root)
    if not is_safe:
        logger.warning(
            "retention[attachments]: refusing to sweep %s — %s",
            root, reason,
        )
        state["last_summary"] = {"refused": True, "reason": reason}
        write_state_json(_ATT_STATE_FILE, state)
        return
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
