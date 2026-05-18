"""
chromadb_integrity.py — Protection layer for ChromaDB SQLite KBs.

PROGRAM §55 (2026-05-17) — built in response to the 2026-04-25 and
2026-05-17 ``memory.corrupt_*`` quarantine events. Root cause was a
dual-writer SQLite race between the standalone ``chromadb`` HTTP
server container (`chromadb 0.5.23`) and the gateway's embedded
``chromadb.PersistentClient`` (`chromadb 1.1.1`), both bind-mounted
to the same ``workspace/memory/chroma.sqlite3``. The standalone
container has been removed from docker-compose.yml (see the inline
comment block where the service used to live).

This module is the defense-in-depth layer that catches the next class
of corruption even after the dual-writer is gone — unclean gateway
restarts, journal recovery anomalies, and silent btree damage.

Five primitives, each independently useful:

  enforce_wal_mode(db_path)
      Sets ``journal_mode=WAL; synchronous=FULL`` idempotently. WAL
      survives crashes much better than the default rollback-journal
      mode every chromadb KB ships with.

  integrity_check(db_path)
      Runs ``PRAGMA integrity_check``. Returns ``"ok"`` or the first
      error string from sqlite. Quick — uses LIMIT 1 to bail early
      when a database is intact.

  quarantine_kb(kb_dir, reason)
      Renames ``workspace/<kb>/`` to ``workspace/<kb>.corrupt_<ts>/``
      atomically, emits an identity-continuity ledger event, then
      creates a fresh empty replacement directory. Mirror of what the
      operator did manually on 2026-05-17.

  daily_snapshot(db_path)
      SQLite's online ``.backup`` API atomically writes a consistent
      copy to ``workspace/<kb>/.sqlite_snapshots/<ts>.db``. Keeps the
      last 7 days. Cheap (~ a few MB per snapshot for `memory/`).

  replay_from_postgres(kb_name)
      For the ``memory`` KB only: re-embeds rows from the Postgres
      ``beliefs`` and ``crewai_memories`` source-of-truth tables back
      into the freshly-created chromadb collections. Idempotent —
      checks existing collection counts and only adds missing rows.

  boot_integrity_scan()
      Orchestrator: walks every KB at gateway start, enforces WAL,
      runs integrity_check, quarantines + (for memory KB) auto-replays
      anything that fails. Called from ``app/main.py`` lifespan.

Every public function is failure-isolated — exceptions are logged but
never re-raised. The whole module is observational: turning all four
runtime_settings switches OFF makes it a no-op.

Master switches (all default ON, read at call time from
``app.runtime_settings`` so they can be flipped without restart):

  chromadb_wal_enforcement_enabled        ── enforce_wal_mode
  chromadb_boot_integrity_check_enabled   ── boot_integrity_scan
  chromadb_daily_snapshot_enabled         ── daily_snapshot
  chromadb_auto_replay_enabled            ── replay_from_postgres
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


# ── Paths ────────────────────────────────────────────────────────────────


def _workspace_root() -> Path:
    """Resolve workspace root.

    Try ``app.paths.WORKSPACE_ROOT`` first; fall back to the container
    bind-mount path. This matches the pattern in
    ``app/healing/monitors/chromadb_hygiene.py`` so test harnesses with
    a custom workspace get picked up automatically.
    """
    try:
        from app.paths import WORKSPACE_ROOT  # type: ignore
        return Path(WORKSPACE_ROOT)
    except Exception:
        return Path("/app/workspace")


def chromadb_kbs(root: Optional[Path] = None) -> list[Path]:
    """Return every live ``chroma.sqlite3`` path under workspace.

    Skips quarantined snapshots (``*.corrupt_*``, ``*.bak_*``,
    ``*_backup``, ``*.backup``) — same filter as chromadb_hygiene so
    we never run integrity check against a known-bad file.
    """
    if root is None:
        root = _workspace_root()
    if not root.exists():
        return []
    found: list[Path] = []
    for p in root.glob("*/chroma.sqlite3"):
        if any(
            seg in p.parent.name
            for seg in ("corrupt_", "bak_", "_backup", ".backup")
        ):
            continue
        found.append(p)
    return found


# ── Runtime-settings gates (failure-isolated) ────────────────────────────


def _gate(name: str, default: bool = True) -> bool:
    """Read a runtime_settings boolean by name, defaulting ON.

    Failure-isolated: if runtime_settings is unimportable (tests,
    stripped builds) or the key is missing, return the default so the
    integrity layer fails OPEN — corruption protection should not
    silently disable itself because a settings file is missing.
    """
    try:
        from app import runtime_settings  # type: ignore
        getter = getattr(runtime_settings, f"get_{name}", None)
        if getter is not None:
            return bool(getter())
        # Generic fallback — _ensure_initialized returns the full dict
        return bool(runtime_settings._ensure_initialized().get(name, default))
    except Exception:
        return default


# ── WAL mode enforcement ─────────────────────────────────────────────────


def enforce_wal_mode(db_path: Path) -> dict:
    """Idempotently set ``journal_mode=WAL`` + ``synchronous=FULL`` on a
    chromadb SQLite file.

    Returns a status dict ``{ok, mode_before, mode_after, sync_before,
    sync_after, error}``. The two pragmas are persistent across
    connections — running this once per boot is sufficient.

    WAL mode is significantly more crash-tolerant than the default
    ``delete`` (rollback journal) mode:
      * Writers don't block readers and vice versa
      * Atomicity is via the WAL file; a crash leaves the main DB
        intact and the WAL replays on next open
      * No journal-file-truncation race that can lose committed pages

    ``synchronous=FULL`` retains the fsync-after-commit guarantee
    (synchronous=NORMAL is faster but loses durability under power loss
    — not a tradeoff worth making for identity-critical data).
    """
    info: dict[str, Any] = {
        "ok": False, "path": str(db_path),
        "mode_before": None, "mode_after": None,
        "sync_before": None, "sync_after": None,
        "error": None,
    }
    try:
        conn = sqlite3.connect(str(db_path), timeout=30.0)
        try:
            cur = conn.execute("PRAGMA journal_mode")
            info["mode_before"] = (cur.fetchone() or [None])[0]
            cur = conn.execute("PRAGMA synchronous")
            info["sync_before"] = (cur.fetchone() or [None])[0]
            # journal_mode=WAL takes effect immediately and is sticky;
            # synchronous needs to be set after.
            cur = conn.execute("PRAGMA journal_mode=WAL")
            info["mode_after"] = (cur.fetchone() or [None])[0]
            # FULL=2 is the strictest non-EXTRA setting; safe across
            # all SQLite versions and macOS Docker fs quirks.
            conn.execute("PRAGMA synchronous=FULL")
            cur = conn.execute("PRAGMA synchronous")
            info["sync_after"] = (cur.fetchone() or [None])[0]
            conn.commit()
            info["ok"] = (
                str(info["mode_after"]).lower() == "wal"
                and int(info["sync_after"] or 0) == 2
            )
        finally:
            conn.close()
    except sqlite3.OperationalError as exc:
        info["error"] = f"sqlite_op_error: {exc}"
    except Exception as exc:
        info["error"] = f"unexpected: {type(exc).__name__}: {exc}"
    return info


# ── Integrity check ──────────────────────────────────────────────────────


def kb_integrity_check(kb_name: str, root: Optional[Path] = None) -> str:
    """Convenience wrapper for ``integrity_check`` by KB name.

    Returns the same shape as ``integrity_check``: ``"ok"`` when clean,
    ``"missing"`` when the sqlite file isn't there, otherwise the first
    sqlite error line. Used by the source-ledger daemon before triggering
    a drift-replay: hammering a corrupt SQLite with re-embed writes only
    makes things worse, so when integrity fails the daemon defers to
    the ``chromadb_integrity_monitor`` quarantine path (PROGRAM §55).
    """
    if root is None:
        root = _workspace_root()
    return integrity_check(root / kb_name / "chroma.sqlite3")


def integrity_check(db_path: Path) -> str:
    """Run SQLite's built-in B-tree integrity check.

    Returns ``"ok"`` when the database is clean, otherwise the first
    error line (sqlite produces many lines on a damaged file; the
    first is the most informative).

    Uses ``PRAGMA integrity_check(1)`` — the optional argument caps
    the result to 1 row, which makes the check effectively early-exit
    on the first damaged page and avoids spending minutes enumerating
    every page on a heavily corrupted file.

    The check is read-only and lock-light. It does NOT detect HNSW
    segment damage (those are separate files chromadb owns) — only
    SQLite-layer damage. The dual-writer corruption we saw is
    SQLite-layer, so this catches the exact failure mode of concern.
    """
    if not db_path.exists():
        return "missing"
    try:
        # read-only mode: don't take a writer lock just to probe
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=30.0)
        try:
            rows = conn.execute("PRAGMA integrity_check(1)").fetchall()
            if not rows:
                return "empty_result"
            first = rows[0]
            text = first[0] if first else ""
            return "ok" if text == "ok" else str(text)
        finally:
            conn.close()
    except sqlite3.DatabaseError as exc:
        # ``database disk image is malformed`` shows up here when
        # corruption is severe enough to fail open. That's still a
        # diagnostic worth surfacing — caller treats anything other
        # than ``"ok"`` as a fail.
        return f"open_failed: {exc}"
    except Exception as exc:
        return f"unexpected: {type(exc).__name__}: {exc}"


# ── Snapshot / restore ───────────────────────────────────────────────────


_SNAPSHOT_DIRNAME = ".sqlite_snapshots"
_SNAPSHOT_KEEP_DAYS = 7


def _now_iso_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def daily_snapshot(db_path: Path, *, retention_days: int = _SNAPSHOT_KEEP_DAYS) -> dict:
    """Atomic SQLite backup to ``<kb>/.sqlite_snapshots/<ts>.db``.

    Uses ``sqlite3.Connection.backup()`` which is the official
    online-backup API — atomic, consistent under concurrent writes,
    and skips the WAL file entirely. The destination is a normal
    SQLite file you can open without any chromadb dependencies.

    Old snapshots beyond ``retention_days`` are pruned. The first
    snapshot of a fresh KB has no priors to prune — still emits a
    snapshot row so the operator sees the daemon is alive.

    Returns ``{ok, snapshot_path, bytes, removed, error}``.
    """
    info: dict[str, Any] = {
        "ok": False, "path": str(db_path), "snapshot_path": None,
        "bytes": 0, "removed": [], "error": None,
    }
    if not db_path.exists():
        info["error"] = "source_missing"
        return info
    snap_dir = db_path.parent / _SNAPSHOT_DIRNAME
    try:
        snap_dir.mkdir(parents=True, exist_ok=True)
        target = snap_dir / f"{_now_iso_compact()}.db"
        # Open source read-only; let sqlite handle the locking.
        src = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30.0)
        try:
            dst = sqlite3.connect(str(target), timeout=30.0)
            try:
                src.backup(dst)
            finally:
                dst.close()
        finally:
            src.close()
        info["snapshot_path"] = str(target)
        info["bytes"] = target.stat().st_size

        # Prune old snapshots. Use file mtime, not name parsing —
        # mtime survives clock-skew + manual file copies.
        cutoff = time.time() - retention_days * 86400
        for p in snap_dir.glob("*.db"):
            if p == target:
                continue
            try:
                if p.stat().st_mtime < cutoff:
                    p.unlink()
                    info["removed"].append(str(p))
            except OSError:
                continue
        info["ok"] = True
    except sqlite3.OperationalError as exc:
        info["error"] = f"sqlite_op_error: {exc}"
    except Exception as exc:
        info["error"] = f"unexpected: {type(exc).__name__}: {exc}"
    return info


def list_snapshots(kb_dir: Path) -> list[Path]:
    """Return existing snapshots for a KB dir, newest first."""
    snap_dir = kb_dir / _SNAPSHOT_DIRNAME
    if not snap_dir.exists():
        return []
    snaps = sorted(snap_dir.glob("*.db"), key=lambda p: p.stat().st_mtime, reverse=True)
    return snaps


def latest_snapshot(kb_dir: Path) -> Optional[Path]:
    snaps = list_snapshots(kb_dir)
    return snaps[0] if snaps else None


# ── Quarantine ───────────────────────────────────────────────────────────


def _emit_ledger_event(kind: str, summary: str, detail: dict[str, Any], actor: str) -> None:
    """Best-effort identity-continuity ledger emission.

    The ``chromadb_corruption`` event kind is new in PROGRAM §55 and
    automatically surfaces via the annual-reflection drift Counter
    because ``summarise_drift.by_kind`` is dynamic.
    """
    try:
        from app.identity.continuity_ledger import append_event  # type: ignore
        append_event(
            kind=kind, actor=actor, summary=summary,
            detail=detail, ts=datetime.now(timezone.utc).isoformat(),
        )
    except Exception:
        logger.debug("chromadb_integrity: ledger emission failed", exc_info=True)


def quarantine_kb(kb_dir: Path, *, reason: str) -> dict:
    """Rename ``workspace/<kb>/`` to ``workspace/<kb>.corrupt_<ts>/``.

    Same shape as the operator-authored rename from 2026-05-17. The
    timestamp is local-time YYYYMMDD_HHMMSS so it sorts naturally and
    matches the existing ``memory.corrupt_*`` precedent visible on
    disk. Atomic on the filesystem (single ``rename``).

    Creates a fresh empty replacement directory at the original path
    so chromadb's next ``PersistentClient(path=...)`` call finds an
    empty slate and starts a clean database — the same path
    ``quarantine + auto-replay`` followed manually.

    Emits a ``chromadb_corruption`` identity-continuity event with
    ``reason`` + ``quarantine_path`` so the operator can correlate
    later.

    Returns ``{ok, quarantine_path, error}``.
    """
    info: dict[str, Any] = {
        "ok": False, "kb_dir": str(kb_dir), "reason": reason,
        "quarantine_path": None, "error": None,
    }
    try:
        if not kb_dir.is_dir():
            info["error"] = "kb_dir_missing"
            return info
        # Local-time timestamp; matches existing memory.corrupt_<ts>
        # filename pattern on disk (YYYYMMDD_HHMMSS).
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = kb_dir.parent / f"{kb_dir.name}.corrupt_{ts}"
        if target.exists():
            # vanishingly unlikely (same-second collision), but guard
            target = kb_dir.parent / f"{kb_dir.name}.corrupt_{ts}_{int(time.monotonic()*1000)%1000:03d}"
        kb_dir.rename(target)
        info["quarantine_path"] = str(target)
        # Re-create the empty original directory so chromadb can
        # immediately start a fresh database on next open.
        kb_dir.mkdir(parents=True, exist_ok=True)
        info["ok"] = True

        _emit_ledger_event(
            kind="chromadb_corruption",
            summary=f"quarantined {kb_dir.name} ({reason})",
            detail={
                "kb_name": kb_dir.name,
                "reason": reason,
                "quarantine_path": str(target),
            },
            actor="chromadb_integrity",
        )
    except OSError as exc:
        info["error"] = f"os_error: {exc}"
    except Exception as exc:
        info["error"] = f"unexpected: {type(exc).__name__}: {exc}"
    return info


# ── Postgres replay (memory KB only for now) ─────────────────────────────


def _postgres_connect():
    """Open a connection to the Mem0 postgres. Returns ``None`` on any
    failure so callers can no-op gracefully (Postgres down, missing
    DSN, etc.). Connection is the caller's responsibility to close.

    Resolution mirrors the rest of the gateway (training_pipeline,
    governance, version_manifest): import ``app.config.get_settings``,
    read ``s.mem0_postgres_url``, connect via psycopg2.
    """
    try:
        from app.config import get_settings  # type: ignore
        import psycopg2  # type: ignore
        s = get_settings()
        dsn = getattr(s, "mem0_postgres_url", "") or ""
        if not dsn:
            dsn = os.getenv("MEM0_POSTGRES_DSN", "") or os.getenv("DATABASE_URL", "")
        if not dsn:
            return None
        return psycopg2.connect(dsn, connect_timeout=10)
    except Exception:
        logger.debug("chromadb_integrity: postgres connect failed", exc_info=True)
        return None


def replay_from_postgres(kb_name: str = "memory") -> dict:
    """Re-embed rows from postgres source-of-truth into a fresh chromadb.

    For ``memory`` KB only. The other KBs (episteme/knowledge/etc.)
    don't have a postgres mirror — their canonical data is in
    markdown files or other application-level sources, which their
    own pipelines repopulate on schedule.

    Idempotent on row-id: each row from postgres carries a stable
    primary key; we pass that as the chromadb document id so re-runs
    don't duplicate. Already-present ids are silently skipped.

    Failure-isolated per-row: a bad row (missing field, embed error)
    is logged and skipped — never aborts the replay.

    Returns ``{ok, kb_name, sources, total_added, total_skipped,
    error}``.
    """
    info: dict[str, Any] = {
        "ok": False, "kb_name": kb_name, "sources": {},
        "total_added": 0, "total_skipped": 0, "error": None,
    }
    if kb_name != "memory":
        info["error"] = "unsupported_kb"
        return info
    if not _gate("chromadb_auto_replay_enabled"):
        info["error"] = "disabled_by_runtime_settings"
        return info

    # Late import — chromadb_manager has heavyweight side-effects we
    # don't want at module-load time.
    try:
        from app.memory import chromadb_manager  # type: ignore
    except Exception as exc:
        info["error"] = f"chromadb_manager_import: {exc}"
        return info

    conn = _postgres_connect()
    if conn is None:
        info["error"] = "postgres_unavailable"
        return info

    try:
        # 1) beliefs → "beliefs" collection
        try:
            beliefs_stats = _replay_beliefs(conn, chromadb_manager)
            info["sources"]["beliefs"] = beliefs_stats
            info["total_added"] += beliefs_stats.get("added", 0)
            info["total_skipped"] += beliefs_stats.get("skipped", 0)
        except Exception as exc:
            info["sources"]["beliefs"] = {"error": str(exc)}

        # 2) crewai_memories → "team_shared" + per-agent collections
        try:
            crewai_stats = _replay_crewai_memories(conn, chromadb_manager)
            info["sources"]["crewai_memories"] = crewai_stats
            info["total_added"] += crewai_stats.get("added", 0)
            info["total_skipped"] += crewai_stats.get("skipped", 0)
        except Exception as exc:
            info["sources"]["crewai_memories"] = {"error": str(exc)}

        info["ok"] = True
    finally:
        try:
            conn.close()
        except Exception:
            pass

    return info


def _replay_beliefs(conn, chromadb_manager) -> dict:
    """Replay postgres ``beliefs`` rows into chromadb ``beliefs``
    collection. Idempotent.

    Real schema: ``belief_id`` (uuid PK), ``content``, ``domain``,
    ``confidence``, plus ``evidence_sources`` / ``metacognitive_flags``
    / ``update_history`` as jsonb. We use the structured columns as
    metadata so an operator querying the rebuilt KB sees the same
    facets as the live one.
    """
    added = 0
    skipped = 0
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT belief_id, content, domain, confidence, "
            "COALESCE(evidence_sources, '[]'::jsonb)::text, "
            "COALESCE(belief_status, 'ACTIVE') "
            "FROM beliefs ORDER BY belief_id"
        )
    except Exception:
        return {"error": "beliefs_table_missing_or_schema_changed"}
    rows = cur.fetchall() or []
    for row in rows:
        try:
            belief_id, content, domain, confidence, evidence_json, status = row
            if not content or not str(content).strip():
                skipped += 1
                continue
            # ChromaDB metadata only accepts scalar values (str / int /
            # float / bool). Evidence is a list of dicts or strings —
            # serialise to JSON so the round-trip is lossless.
            meta = {
                "postgres_id": str(belief_id),
                "domain": str(domain) if domain else "",
                "confidence": float(confidence) if confidence is not None else 0.0,
                "belief_status": str(status) if status else "ACTIVE",
                "evidence_sources_json": str(evidence_json) if evidence_json else "[]",
                "origin": "beliefs_replay",
            }
            doc_id = f"beliefs-pg-{belief_id}"
            _store_idempotent(chromadb_manager, "beliefs", str(content), meta, doc_id)
            added += 1
        except Exception:
            skipped += 1
            continue
    cur.close()
    return {"added": added, "skipped": skipped, "rows_seen": len(rows)}


def _replay_crewai_memories(conn, chromadb_manager) -> dict:
    """Replay postgres ``crewai_memories`` rows.

    Real schema (Mem0): ``id`` (uuid PK), ``vector`` (embedding,
    discarded — we re-embed with the current model), ``payload``
    (jsonb containing ``data`` = the text, ``user_id``, ``entities``,
    ``created_at``, ``attributed_to``, etc.).

    Each row carries Mem0-extracted long-term memory text — exactly
    the operator-facing data that gets lost when the memory KB is
    quarantined. Routed into ``team_shared`` with origin metadata so
    the operator can tell which rows were rehydrated vs newly written.
    """
    added = 0
    skipped = 0
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT id, payload::text FROM crewai_memories ORDER BY id"
        )
    except Exception:
        return {"error": "crewai_memories_table_missing_or_schema_changed"}
    rows = cur.fetchall() or []
    for row in rows:
        try:
            row_id, payload_json = row[0], row[1]
            try:
                payload = json.loads(payload_json) if payload_json else {}
            except Exception:
                payload = {}
            # The text Mem0 indexes is the "data" key. Older rows may
            # use "memory" or "content"; fall back to those.
            text = (
                payload.get("data")
                or payload.get("memory")
                or payload.get("content")
                or ""
            )
            if not text or not str(text).strip():
                skipped += 1
                continue
            meta = {
                "postgres_id": str(row_id),
                "user_id": payload.get("user_id", ""),
                "attributed_to": payload.get("attributed_to", ""),
                "created_at": payload.get("created_at", ""),
                "origin": "crewai_memories_replay",
            }
            # Entities is a list of [TYPE, NAME] pairs; flatten for
            # chromadb metadata (must be scalar). Comma-join the names.
            ents = payload.get("entities") or []
            if isinstance(ents, list):
                names = []
                for e in ents:
                    if isinstance(e, list) and len(e) >= 2:
                        names.append(str(e[1]))
                    elif isinstance(e, str):
                        names.append(e)
                if names:
                    meta["entities"] = ", ".join(names[:20])
            doc_id = f"crewai_memories-pg-{row_id}"
            _store_idempotent(chromadb_manager, "team_shared", str(text), meta, doc_id)
            added += 1
        except Exception:
            skipped += 1
            continue
    cur.close()
    return {"added": added, "skipped": skipped, "rows_seen": len(rows)}


def _store_idempotent(chromadb_manager, collection_name: str, text: str,
                      metadata: dict, doc_id: str) -> None:
    """Add a row to a chromadb collection only if the doc_id isn't
    already present. Mirrors ``chromadb_manager.store`` (curated embed
    + dimension-mismatch handling) but allows a caller-supplied
    ``doc_id`` for idempotency. Used by replay paths.
    """
    try:
        col = chromadb_manager._get_col(collection_name)
        # get(ids=...) returns an empty list when missing; cheap probe.
        existing = col.get(ids=[doc_id])
        if existing and existing.get("ids"):
            return
        # Use the gateway's curated embed function (Metal-accelerated
        # Ollama nomic-embed-text — matches the live KB's vector
        # dimension). Going through col.add() without explicit
        # embeddings would silently fall back to chromadb's bundled
        # ONNX MiniLM-L6-v2 (384-dim) which doesn't match the live
        # store's 768-dim — the add would fail or corrupt.
        embedding = chromadb_manager.embed(text)
        col.add(
            ids=[doc_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata or {}],
        )
        # Bust the manager's count cache so subsequent retrieve calls
        # see the new row.
        try:
            chromadb_manager._count_cache.pop(collection_name, None)
        except Exception:
            pass
    except Exception as exc:
        # WARN level — replay swallowing failures silently is what
        # made the original incident take a full day to diagnose.
        logger.warning(
            "chromadb_integrity: idempotent store failed for %s/%s: %s",
            collection_name, doc_id, exc,
        )


# ── Boot orchestrator ────────────────────────────────────────────────────


def boot_integrity_scan() -> dict:
    """Walk every chromadb KB at gateway start.

    For each KB:
      1. ``enforce_wal_mode`` (gated by ``chromadb_wal_enforcement_enabled``)
      2. ``integrity_check``  (gated by ``chromadb_boot_integrity_check_enabled``)
      3. On failure: quarantine + Signal alert + (memory KB only)
         auto-replay from postgres if ``chromadb_auto_replay_enabled``.

    Failure-isolated end-to-end: any per-KB failure logs + continues.
    The whole function is wrapped in a final try/except so a
    catastrophic error never blocks gateway startup.

    Returns ``{kbs, wal_results, integrity_results, quarantines,
    replays}`` for the operator-visible startup summary.
    """
    summary: dict[str, Any] = {
        "kbs": [], "wal_results": {}, "integrity_results": {},
        "quarantines": {}, "replays": {},
    }
    if not _gate("chromadb_boot_integrity_check_enabled"):
        summary["skipped"] = "boot_integrity_check_disabled"
        return summary
    try:
        root = _workspace_root()
        sqlites = chromadb_kbs(root)
        summary["kbs"] = [str(p) for p in sqlites]

        for db_path in sqlites:
            kb_dir = db_path.parent
            kb_name = kb_dir.name

            # 1. WAL enforcement (safe to run on a corrupt DB — fails
            # gracefully).
            if _gate("chromadb_wal_enforcement_enabled"):
                wal = enforce_wal_mode(db_path)
                summary["wal_results"][kb_name] = wal

            # 2. Integrity check
            result = integrity_check(db_path)
            summary["integrity_results"][kb_name] = result
            if result == "ok":
                continue

            # 3. Quarantine on failure
            logger.warning(
                "chromadb_integrity: %s failed integrity check: %s",
                kb_name, result,
            )
            quar = quarantine_kb(kb_dir, reason=f"integrity_check_failed: {result}")
            summary["quarantines"][kb_name] = quar

            # 4. Signal alert (best-effort)
            _send_corruption_alert(kb_name, result, quar)

            # 5. Auto-replay — PROGRAM §56 makes this universal.
            # First try replay-from-ledger (covers every KB); fall
            # back to replay-from-postgres for the legacy memory path
            # if the ledger replay returned no rows (likely the
            # ledger is still empty for that KB at first boot).
            if quar.get("ok"):
                replay = _try_universal_replay(kb_name)
                summary["replays"][kb_name] = replay

        # 6. PROGRAM §56 — drift detection is intentionally NOT run
        # here at boot. The source_ledger_daemon runs it every 24h
        # after a 5-min warm-up; doing it at boot also opens
        # chromadb.PersistentClient against every KB (including
        # large ones — philosophy was 87 MB on 2026-05-17), which
        # adds minutes to gateway startup as chromadb loads HNSW
        # segments. Lifespan startup must be FAST so the gateway
        # binds its port and serves /api/cp routes promptly; the
        # daemon picks up any drift on its next pass.
        #
        # If you need drift detection at boot (e.g. after a known
        # KB rebuild), invoke the daemon's _run_one_pass directly
        # from a one-shot script — don't put it back in lifespan.
    except Exception:
        logger.exception("chromadb_integrity.boot_integrity_scan: top-level failure")
        summary["error"] = "top_level_exception"

    return summary


def _try_universal_replay(kb_name: str) -> dict:
    """Prefer replay-from-ledger (PROGRAM §56); fall back to
    replay-from-postgres (PROGRAM §55) for the legacy memory path
    when the ledger is empty.

    The ledger covers every KB uniformly. Postgres replay only
    handles the ``memory`` KB but is the path that worked before §56
    shipped — keeping it as a fallback gives a smooth migration.
    """
    try:
        from app.memory.source_ledger import replay_kb
        ledger_result = replay_kb(kb_name)
        # If the ledger had rows, the replay did something useful.
        if ledger_result.rows_seen > 0:
            return {"source": "ledger", **ledger_result.to_dict()}
    except Exception:
        logger.debug("chromadb_integrity: ledger replay failed", exc_info=True)
    # Fallback: postgres replay (memory KB only).
    if kb_name == "memory":
        try:
            pg_result = replay_from_postgres(kb_name)
            return {"source": "postgres", **pg_result}
        except Exception:
            logger.debug("chromadb_integrity: postgres replay failed", exc_info=True)
    return {"source": "none", "ok": False, "error": "no_replay_path_available"}


def _send_corruption_alert(kb_name: str, integrity_msg: str, quarantine_info: dict) -> None:
    """Best-effort Signal alert on quarantine. Tag-keyed so dedup
    works if the same KB fires twice in a short window."""
    try:
        from app.life_companion._common import send_signal_alert  # type: ignore
        quar_path = quarantine_info.get("quarantine_path") or "(quarantine failed)"
        text = (
            f"🚨 ChromaDB integrity failure on `{kb_name}` — "
            f"`{integrity_msg[:200]}`. Quarantined to `{quar_path}`. "
            f"{'Auto-replay from postgres triggered.' if kb_name == 'memory' else 'Manual recovery needed.'}"
        )
        send_signal_alert(text, tag=f"chromadb_integrity:{kb_name}")
    except Exception:
        logger.debug(
            "chromadb_integrity: signal alert failed for %s",
            kb_name, exc_info=True,
        )


# ── Operator-facing summary helpers ──────────────────────────────────────


def integrity_summary() -> dict:
    """One-shot read-only summary suitable for /api/cp surfaces.

    Reports per-KB integrity status + WAL mode + snapshot count +
    latest-snapshot age. Never mutates anything.
    """
    out: dict[str, Any] = {"kbs": []}
    root = _workspace_root()
    for db_path in chromadb_kbs(root):
        kb_dir = db_path.parent
        snaps = list_snapshots(kb_dir)
        latest = snaps[0] if snaps else None
        latest_age_s = None
        if latest is not None:
            try:
                latest_age_s = int(time.time() - latest.stat().st_mtime)
            except OSError:
                latest_age_s = None
        # Read journal_mode without taking a writer lock.
        mode = None
        try:
            c = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
            try:
                row = c.execute("PRAGMA journal_mode").fetchone()
                mode = row[0] if row else None
            finally:
                c.close()
        except Exception:
            mode = "unknown"
        out["kbs"].append({
            "name": kb_dir.name,
            "path": str(db_path),
            "bytes": db_path.stat().st_size if db_path.exists() else 0,
            "journal_mode": mode,
            "integrity": integrity_check(db_path),
            "snapshot_count": len(snaps),
            "latest_snapshot_age_s": latest_age_s,
        })
    return out
