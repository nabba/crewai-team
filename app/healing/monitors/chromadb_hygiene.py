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

    # Q3.1 (2026-05-11) — publish to SubIA Global Workspace so the
    # consciousness substrate sees its own hygiene pass. Salience scales
    # with the amount reclaimed; sub-MB passes are below the noise floor
    # and skipped. Failure-isolated.
    _publish_hygiene_outcome_to_gw(len(per_file), freed_total)

    # Q3.2 Item 5 — HNSW orphan segment detection. Each chromadb
    # collection's HNSW data lives in a per-collection UUID directory
    # next to chroma.sqlite3. Orphan dirs (segment dirs present on
    # disk but not referenced by the collections table) accumulate
    # from crashes mid-write. We don't auto-delete — that's destructive
    # and chromadb may legitimately know about dirs we don't expect.
    # We only ALERT so the operator can run the rebuild manually.
    orphan_findings = _scan_for_orphan_segments(sqlites)
    if orphan_findings:
        _alert_orphan_segments(orphan_findings)

    # Q3.2 Item 6 — reclaim trend. If we keep freeing more bytes each
    # pass, the SQLite freelist isn't the bottleneck — HNSW segments
    # need a full rebuild. Compare current freed_total against the
    # last 4 passes; alert when reclaim grows ≥2× the rolling median.
    #
    # Q3.3 (PROGRAM §40.3 Item 6) — alert dedup. The trend alert
    # exists to nudge the operator "run a rebuild." Repeating that
    # nudge every quarterly pass without acknowledgment becomes
    # noise. We suppress repeat alerts within a 4-pass window
    # (~1 year of quarterly cadence) tracked by alert count in
    # `state["trend_alerts"]`. The first alert in a series fires;
    # subsequent ones increment a counter but stay silent until
    # the trend recovers (alert clears) — at which point the
    # counter resets and a future spike alerts again.
    trend = _update_and_check_reclaim_trend(state, freed_total)
    if trend.get("alert"):
        # Q3.3 dedup logic.
        repeats = int(state.get("trend_alert_repeats") or 0)
        if repeats == 0:
            _alert_growing_reclaim(trend)
            state["trend_alert_repeats"] = 1
        elif repeats >= 4:
            # Resurface the alert after 4 silent passes — operator
            # may have missed the first one.
            _alert_growing_reclaim(trend, repeat_n=repeats + 1)
            state["trend_alert_repeats"] = 1
        else:
            state["trend_alert_repeats"] = repeats + 1
    else:
        # Trend cleared — reset dedup so future spikes alert fresh.
        state["trend_alert_repeats"] = 0

    state["last_summary"] = {
        "files": len(per_file),
        "freed_bytes_total": freed_total,
        "per_file": per_file,
        "orphan_findings": orphan_findings,
        "reclaim_trend": trend,
    }
    write_state_json(_STATE_FILE, state)


def _scan_for_orphan_segments(sqlites: list[Path]) -> list[dict]:
    """For each chroma.sqlite3, find UUID-named subdirectories that
    aren't referenced by the ``collections`` table. Returns one
    summary dict per chroma file with non-empty findings.

    Best-effort: SQLite read errors / unexpected schema variants
    silently skip that file."""
    findings: list[dict] = []
    for db_path in sqlites:
        kb_dir = db_path.parent
        try:
            # 1. Collect on-disk UUID dirs.
            disk_uuids: set[str] = set()
            for sub in kb_dir.iterdir():
                if not sub.is_dir():
                    continue
                name = sub.name
                # ChromaDB collection dirs are UUID4. Cheap filter:
                # 36 chars, 4 dashes at canonical positions.
                if (
                    len(name) == 36
                    and name[8] == "-" and name[13] == "-"
                    and name[18] == "-" and name[23] == "-"
                ):
                    disk_uuids.add(name)
            if not disk_uuids:
                continue
            # 2. Collect known collection IDs from SQLite.
            known: set[str] = set()
            try:
                conn = sqlite3.connect(str(db_path), timeout=10.0)
                try:
                    cur = conn.execute("SELECT id FROM collections")
                    for (cid,) in cur.fetchall():
                        if isinstance(cid, str):
                            known.add(cid)
                finally:
                    conn.close()
            except sqlite3.OperationalError:
                # Unknown schema variant — skip silently.
                continue
            orphans = sorted(disk_uuids - known)
            if not orphans:
                continue
            # 3. Compute orphan disk usage for the alert weight.
            orphan_bytes = 0
            for u in orphans:
                p = kb_dir / u
                if not p.is_dir():
                    continue
                for f in p.rglob("*"):
                    try:
                        if f.is_file():
                            orphan_bytes += f.stat().st_size
                    except OSError:
                        continue
            findings.append({
                "kb": kb_dir.name,
                "orphan_count": len(orphans),
                "orphan_uuids": orphans[:10],   # truncate for the report
                "orphan_bytes": orphan_bytes,
            })
        except Exception:
            logger.debug(
                "chromadb_hygiene: orphan scan failed for %s",
                db_path, exc_info=True,
            )
    return findings


def _alert_orphan_segments(findings: list[dict]) -> None:
    """Single consolidated alert for all KBs with orphans."""
    total_orphans = sum(f["orphan_count"] for f in findings)
    total_bytes = sum(f["orphan_bytes"] for f in findings)
    if total_orphans == 0:
        return
    # Suppress noise: only alert if orphan disk usage is meaningful.
    if total_bytes < 10 * 1024 * 1024:    # <10 MB → ignore
        return
    per_kb = ", ".join(
        f"{f['kb']}: {f['orphan_count']} orphans / "
        f"{f['orphan_bytes'] / 1024 / 1024:.1f} MB"
        for f in findings
    )
    try:
        send_signal_alert(
            f"🗂 ChromaDB orphan segments detected — "
            f"{total_orphans} orphans, "
            f"{total_bytes / 1024 / 1024:.1f} MB total. {per_kb}. "
            f"Run `python -m app.memory.chromadb_rebuild --kb <kb> "
            f"--collection <name>` per affected collection to reclaim.",
            tag="chromadb_hygiene_orphans",
        )
    except Exception:
        logger.debug(
            "chromadb_hygiene: orphan alert failed", exc_info=True,
        )


def _update_and_check_reclaim_trend(
    state: dict, current_freed: int,
) -> dict:
    """Maintain a rolling 4-pass history of reclaim sizes. Alert if
    the current pass freed ≥2× the median of the prior passes — a
    signal that SQLite VACUUM isn't enough and a full collection
    rebuild would reclaim much more.

    Returns ``{history: [...], median_prior, alert: bool, reason}``.
    """
    history = list(state.get("reclaim_history") or [])
    history.append(int(current_freed))
    history = history[-5:]   # keep last 5 (4 prior + current)
    state["reclaim_history"] = history
    if len(history) < 4:
        return {
            "history": history,
            "median_prior": None,
            "alert": False,
            "reason": "not enough history yet",
        }
    prior = sorted(history[:-1])
    median_prior = prior[len(prior) // 2]
    if median_prior < 10 * 1024 * 1024:   # ignore tiny baselines
        return {
            "history": history,
            "median_prior": median_prior,
            "alert": False,
            "reason": "baseline reclaim too small to be meaningful",
        }
    alert = current_freed >= 2 * median_prior
    return {
        "history": history,
        "median_prior": median_prior,
        "current": current_freed,
        "alert": alert,
        "reason": (
            f"current {current_freed / 1024 / 1024:.1f} MB ≥ 2× "
            f"prior-median {median_prior / 1024 / 1024:.1f} MB"
            if alert else "within normal range"
        ),
    }


def _alert_growing_reclaim(trend: dict, repeat_n: int = 1) -> None:
    """Send the growing-reclaim Signal alert. ``repeat_n=1`` is the
    first alert in a series (fresh trend). Higher values are the
    resurfacing alert after the dedup window — formatted differently
    so the operator can see we're nagging."""
    try:
        prefix = "📈" if repeat_n == 1 else f"📈🔁 (repeat #{repeat_n})"
        send_signal_alert(
            f"{prefix} ChromaDB hygiene reclaim growing — current pass "
            f"freed {trend['current'] / 1024 / 1024:.1f} MB vs prior-"
            f"median {trend['median_prior'] / 1024 / 1024:.1f} MB. "
            f"SQLite VACUUM alone isn't keeping up; consider a full "
            f"collection rebuild via `python -m app.memory.chromadb_rebuild`.",
            tag="chromadb_hygiene_growing_reclaim",
        )
    except Exception:
        logger.debug(
            "chromadb_hygiene: trend alert failed", exc_info=True,
        )


def _publish_hygiene_outcome_to_gw(files: int, freed_bytes: int) -> None:
    """Best-effort GW publish. Never raises."""
    if freed_bytes < 1_000_000:   # <1 MB — below the SubIA noise floor
        return
    try:
        from app.workspace_publish import publish_to_workspace
        freed_mb = freed_bytes / 1024 / 1024
        # Salience: small-to-medium — substrate hygiene is housekeeping,
        # not a dispositional event. 50 MB → 0.30; 500 MB → 0.55 (capped).
        salience = min(0.55, 0.20 + freed_mb / 200.0)
        publish_to_workspace(
            source="chromadb_hygiene",
            content=(
                f"Quarterly ChromaDB hygiene: VACUUM reclaimed "
                f"{freed_mb:.1f} MB across {files} files."
            ),
            salience=salience,
            signal_type="disposition",
        )
    except Exception:
        logger.debug(
            "chromadb_hygiene: GW publish failed", exc_info=True,
        )
