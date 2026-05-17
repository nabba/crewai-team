"""Daily ChromaDB integrity check + atomic snapshot.

PROGRAM §55 (2026-05-17) — 35th healing monitor. Composes with the
boot-time integrity scan in ``app/main.py`` lifespan: boot catches
damage present at startup; this monitor catches damage that appears
during a long-running session.

Two distinct passes per probe, gated independently:

  integrity branch — gated by ``chromadb_integrity_monitor_enabled``
    Runs ``PRAGMA integrity_check`` on every chromadb KB. On failure,
    quarantines + Signal alert + (memory KB only) auto-replay from
    Postgres. Same routine the boot scan uses — single source of
    truth for "what to do on a damaged file."

  snapshot branch — gated by ``chromadb_daily_snapshot_enabled``
    Atomic SQLite ``.backup`` to ``<kb>/.sqlite_snapshots/<ts>.db``.
    Retains 7 days. Cheap: ``memory/`` snapshot is ~2 MB and growing
    slowly.

Cadence: daily probe with no internal stretching. The integrity
check is fast (read-only PRAGMA, early-exit at 1 row) and snapshot
size is bounded by retention. Compared to ``chromadb_hygiene`` (90d
internal cadence for VACUUM), this is meant to fire frequently — a
silent corruption that goes 90 days unnoticed would defeat the
whole point.

Master gate: ``HEALING_MONITORS_ENABLED`` like every other monitor.
Disabling that umbrella switch disables this monitor entirely.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from app.life_companion._common import (
    audit_event,
    background_enabled,
    read_state_json,
    send_signal_alert,
    write_state_json,
)

logger = logging.getLogger(__name__)

_STATE_FILE = "chromadb_integrity.json"
_RUN_CADENCE_S = 23 * 3600   # daily — give a little slack so probe doesn't drift
_INTEGRITY_FAILURE_ALERT_DEDUP_DAYS = 1  # same-KB re-alert window


def _runtime_gate(name: str, default: bool = True) -> bool:
    """Read a runtime_settings gate by full key name; failure-isolated.

    Used for the two per-branch switches (integrity, snapshot). The
    monitor itself shares the global ``HEALING_MONITORS_ENABLED`` via
    ``background_enabled`` higher up the call chain.
    """
    try:
        from app import runtime_settings  # type: ignore
        getter = getattr(runtime_settings, f"get_{name}", None)
        if getter is not None:
            return bool(getter())
        return bool(runtime_settings._ensure_initialized().get(name, default))
    except Exception:
        return default


def run() -> None:
    """One probe — cadence-guarded internally.

    Failure-isolated: per-KB exceptions logged + continue. The
    aggregate per-pass result is written to ``state_path`` so the
    /cp/settings card can display the last result.
    """
    if not background_enabled():
        return

    state = read_state_json(_STATE_FILE, {"last_run_at": 0.0})
    now = time.time()
    if now - float(state.get("last_run_at", 0)) < _RUN_CADENCE_S:
        return
    state["last_run_at"] = now

    integrity_on = _runtime_gate("chromadb_integrity_monitor_enabled")
    snapshot_on = _runtime_gate("chromadb_daily_snapshot_enabled")
    if not integrity_on and not snapshot_on:
        state["last_summary"] = {"skipped": "all_branches_disabled"}
        write_state_json(_STATE_FILE, state)
        return

    # Import the core module lazily — keeps the monitor loadable even
    # if chromadb_integrity has a transient import error.
    try:
        from app.memory.chromadb_integrity import (
            chromadb_kbs,
            integrity_check,
            quarantine_kb,
            daily_snapshot,
            replay_from_postgres,
        )
    except Exception:
        logger.exception("chromadb_integrity_monitor: core module import failed")
        state["last_summary"] = {"error": "core_module_import_failed"}
        write_state_json(_STATE_FILE, state)
        return

    kbs = chromadb_kbs()
    per_kb: list[dict[str, Any]] = []
    new_quarantines: list[str] = []
    snapshot_failures: list[dict] = []
    total_bytes_snapshotted = 0

    # Per-KB alert dedup state (so we don't spam Signal every probe
    # when a corruption is persistent and not yet manually recovered).
    alerted = state.get("alerted_corruption") or {}
    if not isinstance(alerted, dict):
        alerted = {}
    cutoff_alert = now - _INTEGRITY_FAILURE_ALERT_DEDUP_DAYS * 86400

    for db_path in kbs:
        kb_dir = db_path.parent
        kb_name = kb_dir.name
        row: dict[str, Any] = {"name": kb_name, "path": str(db_path)}

        # ── Integrity ────────────────────────────────────────────
        if integrity_on:
            verdict = integrity_check(db_path)
            row["integrity"] = verdict
            if verdict != "ok":
                logger.warning(
                    "chromadb_integrity_monitor: %s failed integrity: %s",
                    kb_name, verdict,
                )
                # Quarantine + replay path mirrors boot_integrity_scan.
                quar = quarantine_kb(
                    kb_dir,
                    reason=f"integrity_check_failed: {verdict}",
                )
                row["quarantine"] = quar
                if quar.get("ok"):
                    new_quarantines.append(kb_name)
                if kb_name == "memory" and quar.get("ok"):
                    replay = replay_from_postgres("memory")
                    row["replay"] = replay
                # Per-KB dedup: alert once per day even if probe
                # keeps catching the same KB.
                last_alerted_at = float(alerted.get(kb_name, 0.0))
                if last_alerted_at < cutoff_alert:
                    quar_path = quar.get("quarantine_path") or "(quarantine failed)"
                    auto_replay_blurb = ""
                    if kb_name == "memory":
                        replay_info = row.get("replay") or {}
                        if replay_info.get("ok"):
                            added = replay_info.get("total_added", 0)
                            auto_replay_blurb = (
                                f" Auto-replayed {added} rows from postgres."
                            )
                        elif replay_info.get("error"):
                            auto_replay_blurb = (
                                f" Auto-replay failed: {replay_info.get('error')}."
                            )
                    send_signal_alert(
                        f"🚨 ChromaDB integrity failure on `{kb_name}` — "
                        f"`{verdict[:200]}`. Quarantined to `{quar_path}`."
                        f"{auto_replay_blurb} "
                        f"See /cp/settings → ChromaDB integrity.",
                        tag=f"chromadb_integrity:{kb_name}",
                    )
                    alerted[kb_name] = now
        # ── Snapshot ─────────────────────────────────────────────
        # Only snapshot if integrity is OK (or skipped) — never
        # snapshot a known-bad file.
        if snapshot_on and row.get("integrity", "ok") == "ok":
            snap = daily_snapshot(db_path)
            row["snapshot"] = snap
            if snap.get("ok"):
                total_bytes_snapshotted += int(snap.get("bytes") or 0)
            else:
                snapshot_failures.append({
                    "kb": kb_name,
                    "error": snap.get("error"),
                })

        per_kb.append(row)

    state["alerted_corruption"] = alerted

    # Aggregate snapshot-failure alert: don't fan out per file; one
    # daily summary alert is enough.
    if snapshot_failures:
        prev_alert_at = float(state.get("snapshot_failure_alerted_at", 0.0))
        if prev_alert_at < cutoff_alert:
            paths = ", ".join(f"{x['kb']}: {x['error']}" for x in snapshot_failures)
            send_signal_alert(
                f"⚠️ ChromaDB daily snapshot failed for {len(snapshot_failures)} "
                f"KB(s): {paths[:400]}. Last successful snapshots remain on "
                f"disk; integrity protection still active.",
                tag="chromadb_integrity_snapshot_fail",
            )
            state["snapshot_failure_alerted_at"] = now

    audit_event(
        "chromadb_integrity_pass",
        kbs=len(per_kb),
        integrity_failures=[
            x["name"] for x in per_kb if x.get("integrity", "ok") != "ok"
        ],
        new_quarantines=new_quarantines,
        snapshot_failures_count=len(snapshot_failures),
        bytes_snapshotted=total_bytes_snapshotted,
    )

    state["last_summary"] = {
        "kbs": len(per_kb),
        "per_kb": per_kb,
        "new_quarantines": new_quarantines,
        "snapshot_failures_count": len(snapshot_failures),
        "bytes_snapshotted": total_bytes_snapshotted,
    }
    write_state_json(_STATE_FILE, state)
