"""Quarterly ``VACUUM`` on every ChromaDB SQLite metadata file.

PROGRAM §40 (2026-05-10) — Q3 Item 10.

ChromaDB persists in two layers under ``workspace/<kb>/``:

  * ``chroma.sqlite3``         — collection metadata + embedding records
                                  (rowstore + small auxiliary indexes).
  * ``<collection-uuid>/``     — HNSW segment files (immutable file
                                  format owned by chromadb-internal code).

ChromaDB has **no public ``compact()`` API**. The closest hygiene we
can do without taking the whole vector store offline is plain SQLite
``VACUUM`` on the metadata file: that recovers space from rows that
were soft-deleted by ``Collection.delete(...)`` calls but whose pages
were never released back to the filesystem.

Reclamation from HNSW segments is only achievable via a full
collection rebuild (``app.memory.chromadb_rebuild``) — that is an
operator-initiated action because it briefly takes a collection
offline. This monitor never invokes the rebuild on its own.

Cadence: ~90 days. ChromaDB workloads are write-light + read-heavy;
the SQLite freelist grows slowly and a quarterly VACUUM is cheap
relative to the win. We run on a daily probe with an internal
cadence guard so the operator can observe each pass on its own
timeline without restart drift accumulating.

Master switch: shares ``HEALING_MONITORS_ENABLED`` with the rest of
the daemon.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path

from app.life_companion._common import (
    audit_event,
    background_enabled,
    read_state_json,
    send_signal_alert,
    write_state_json,
)

logger = logging.getLogger(__name__)

_STATE_FILE = "chromadb_hygiene.json"
_RUN_CADENCE_S = 90 * 24 * 3600     # quarterly
_BIG_FREE_THRESHOLD_BYTES = 50 * 1024 * 1024   # alert at >50 MB recovered
_PER_FILE_TIMEOUT_S = 120.0         # bail out a single VACUUM if it stalls

# We scan WORKSPACE_ROOT for ``chroma.sqlite3`` so the monitor follows
# wherever the operator pointed `WORKSPACE_DIR`. Includes every KB that
# uses chromadb (memory, philosophy, episteme, knowledge, experiential,
# tensions, aesthetics, …).


def _workspace_root() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT)
    except Exception:
        return Path("/app/workspace")


def _find_chroma_sqlites(root: Path) -> list[Path]:
    """Return every ``chroma.sqlite3`` directly under a workspace KB
    directory. Skips quarantined snapshots like ``memory.corrupt_*``."""
    if not root.exists():
        return []
    found: list[Path] = []
    # Two-level glob covers the KB layout (workspace/<kb>/chroma.sqlite3).
    for p in root.glob("*/chroma.sqlite3"):
        # Skip recovery snapshots (``*.corrupt_*`` / ``*.bak_*``).
        if any(
            seg in p.parent.name
            for seg in ("corrupt_", "bak_", "_backup", ".backup")
        ):
            continue
        found.append(p)
    return found


def _vacuum_one(db_path: Path) -> dict:
    """Run ``VACUUM`` on one chroma.sqlite3. Returns
    ``{ok, bytes_before, bytes_after, freed_bytes, duration_s, error}``.
    """
    try:
        bytes_before = db_path.stat().st_size
    except OSError:
        return {
            "ok": False, "bytes_before": 0, "bytes_after": 0,
            "freed_bytes": 0, "duration_s": 0.0, "error": "stat_failed",
        }
    started = time.monotonic()
    err: str | None = None
    try:
        # ``timeout`` arg = how long to wait for the lock if another
        # process holds it. Generous so we don't fight a transient
        # ChromaDB write, but short enough to fail loud if something
        # is genuinely wedged.
        conn = sqlite3.connect(str(db_path), timeout=30.0)
        try:
            # PRAGMA optimize is cheap & safe; refreshes stats so the
            # query planner stays sharp. Run before VACUUM so VACUUM
            # has up-to-date stats.
            conn.execute("PRAGMA optimize")
            conn.execute("VACUUM")
            conn.commit()
        finally:
            conn.close()
    except sqlite3.OperationalError as exc:
        # Most common: "database is locked" if a chroma write is in
        # flight. Best-effort — try again next quarter.
        err = f"sqlite_op_error: {exc}"
    except Exception as exc:
        err = f"unexpected: {type(exc).__name__}: {exc}"
    duration_s = time.monotonic() - started
    try:
        bytes_after = db_path.stat().st_size
    except OSError:
        bytes_after = bytes_before
    return {
        "ok": err is None,
        "bytes_before": bytes_before,
        "bytes_after": bytes_after,
        "freed_bytes": max(0, bytes_before - bytes_after),
        "duration_s": round(duration_s, 3),
        "error": err,
    }


def run() -> None:
    """One pass — cadence-guarded internally."""
    if not background_enabled():
        return

    state = read_state_json(_STATE_FILE, {"last_run_at": 0.0})
    now = time.time()
    if now - float(state.get("last_run_at", 0)) < _RUN_CADENCE_S:
        return
    state["last_run_at"] = now

    sqlites = _find_chroma_sqlites(_workspace_root())
    if not sqlites:
        # No KBs yet — record the run and move on. (Don't tell the
        # operator: empty workspace is a normal early state.)
        state["last_summary"] = {"files": 0, "freed_bytes_total": 0}
        write_state_json(_STATE_FILE, state)
        return

    per_file: list[dict] = []
    freed_total = 0
    for p in sqlites:
        info = _vacuum_one(p)
        info["path"] = str(p)
        per_file.append(info)
        if info.get("ok"):
            freed_total += int(info.get("freed_bytes", 0))

    audit_event(
        "chromadb_hygiene_pass",
        files=len(per_file),
        freed_bytes_total=freed_total,
        per_file=per_file,
    )

    if freed_total >= _BIG_FREE_THRESHOLD_BYTES:
        # One alert summarising all the files; never a fan-out per file.
        send_signal_alert(
            f"🗜 Self-heal: quarterly ChromaDB VACUUM freed "
            f"{freed_total / 1024 / 1024:.1f} MB across {len(per_file)} "
            f"files. Operator can run "
            f"`python -m app.memory.chromadb_rebuild --collection <name>` "
            f"for further reclaim from HNSW segments.",
            tag="chromadb_hygiene",
        )

    state["last_summary"] = {
        "files": len(per_file),
        "freed_bytes_total": freed_total,
        "per_file": per_file,
    }
    write_state_json(_STATE_FILE, state)
