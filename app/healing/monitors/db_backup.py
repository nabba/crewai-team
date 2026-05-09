"""DB backup monitor — runs weekly and alerts on staleness or failure.

Wave 0/1 closure (#A1, 2026-05-09). Off by default. Operators set
``HEALING_DB_BACKUP_ENABLED=1`` (or flip ``db_backup_enabled`` in
``runtime_settings``) to schedule weekly Postgres + Neo4j + ChromaDB
backups via ``app/healing/db_backup.py``.

Two responsibilities:

  1. **Run the backup** on internal weekly cadence. The monitor driver
     ticks this monitor daily (cheap), but the run itself is gated by
     ``DB_BACKUP_RUN_INTERVAL_S`` (default 7 days, min 1 day).

  2. **Alert on staleness/failure**. If the most recent successful
     backup is older than ``DB_BACKUP_STALE_DAYS`` (default 14 days),
     escalate via Signal. Per-monitor cooldown so a chronic failure
     doesn't spam.

Default OFF because:
  * Laptop dev rarely needs automated DB backups (operator can run the
    deploy/scripts/backup.sh script manually before destructive work).
  * Production K8s should use a dedicated CronJob with proper alerting,
    not the gateway monitor (which has gateway uptime as a precondition).

Operators turn it on once they've decided "the gateway is the backup
runner." The freshness alert still fires either way — even with
backups disabled, if a stale manifest indicates someone forgot to run
the operator script, the alert nudges them.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from app.healing.handlers._common import (
    audit_event,
    read_state_json,
    send_signal_alert,
    write_state_json,
)

logger = logging.getLogger(__name__)


_STATE_FILE = "db_backup_monitor.json"
_DEFAULT_RUN_INTERVAL_S = 7 * 24 * 3600
_DEFAULT_STALE_DAYS = 14
_FAIL_ALERT_COOLDOWN_S = 24 * 3600
_STALE_ALERT_COOLDOWN_S = 24 * 3600

_MANIFEST_PATH = Path("/app/workspace/backups/manifest.json")


def _enabled() -> bool:
    # runtime_settings beats env when present.
    try:
        from app.runtime_settings import get_runtime_settings
        rs = get_runtime_settings()
        if hasattr(rs, "db_backup_enabled"):
            return bool(getattr(rs, "db_backup_enabled"))
    except Exception:
        pass
    raw = os.getenv("HEALING_DB_BACKUP_ENABLED", "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _run_interval_s() -> int:
    raw = os.getenv("DB_BACKUP_RUN_INTERVAL_S", str(_DEFAULT_RUN_INTERVAL_S)).strip()
    try:
        return max(86400, int(raw))  # never run more often than daily
    except ValueError:
        return _DEFAULT_RUN_INTERVAL_S


def _stale_days() -> int:
    raw = os.getenv("DB_BACKUP_STALE_DAYS", str(_DEFAULT_STALE_DAYS)).strip()
    try:
        return max(2, int(raw))
    except ValueError:
        return _DEFAULT_STALE_DAYS


def _last_successful_run_age_s(now: float) -> float | None:
    """Walk the manifest in reverse for the freshest fully-successful run.

    Returns age in seconds, or None if no manifest / no successful run.
    """
    if not _MANIFEST_PATH.exists():
        return None
    try:
        import json
        manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
        runs = manifest.get("runs", [])
        for entry in reversed(runs):
            if entry.get("all_ok"):
                from datetime import datetime
                completed_at = entry.get("completed_at", "")
                if not completed_at:
                    continue
                # Python's fromisoformat handles +00:00 but not Z; normalize.
                normalized = completed_at.replace("Z", "+00:00")
                ts = datetime.fromisoformat(normalized).timestamp()
                return now - ts
    except Exception:
        logger.debug("db_backup_monitor: manifest read failed", exc_info=True)
    return None


def run() -> None:
    if not _enabled():
        return

    now = time.time()
    state = read_state_json(_STATE_FILE, {
        "last_run_at": 0.0,
        "last_fail_alert_at": 0.0,
        "last_stale_alert_at": 0.0,
    })

    # --- Phase 1: maybe run a backup -------------------------------------
    interval = _run_interval_s()
    if now - float(state.get("last_run_at", 0)) >= interval:
        try:
            from app.healing.db_backup import run_backup
            entry = run_backup()
            state["last_run_at"] = now
            state["last_run_summary"] = {
                "all_ok": entry["all_ok"],
                "duration_s": entry["duration_s"],
                "postgres_ok": entry["postgres"]["ok"],
                "neo4j_ok": entry["neo4j"]["ok"],
                "chromadb_ok": entry["chromadb"]["ok"],
            }
            audit_event(
                "db_backup_monitor_run",
                all_ok=entry["all_ok"],
                postgres_ok=entry["postgres"]["ok"],
                neo4j_ok=entry["neo4j"]["ok"],
                chromadb_ok=entry["chromadb"]["ok"],
                duration_s=entry["duration_s"],
            )
            if not entry["all_ok"] and now - float(state.get("last_fail_alert_at", 0)) >= _FAIL_ALERT_COOLDOWN_S:
                state["last_fail_alert_at"] = now
                failed_parts = [
                    name for name in ("postgres", "neo4j", "chromadb")
                    if not entry[name]["ok"]
                ]
                errors = " | ".join(
                    f"{name}: {entry[name].get('error') or '?'}"
                    for name in failed_parts
                )
                send_signal_alert(
                    f"💾 Self-heal: DB backup failed for "
                    f"{', '.join(failed_parts)}. Errors: {errors}. "
                    f"Manifest: workspace/backups/manifest.json. "
                    f"Run `bash deploy/scripts/backup.sh` from the host as "
                    f"a fallback.",
                    tag="db_backup:run_failed",
                )
        except Exception as exc:
            logger.debug("db_backup_monitor: run_backup raised", exc_info=True)
            state["last_run_at"] = now  # back off; don't tight-loop on a broken engine
            audit_event(
                "db_backup_monitor_run",
                all_ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )

    # --- Phase 2: alert on staleness -------------------------------------
    age_s = _last_successful_run_age_s(now)
    threshold_s = _stale_days() * 24 * 3600
    if (age_s is None or age_s > threshold_s) and now - float(state.get("last_stale_alert_at", 0)) >= _STALE_ALERT_COOLDOWN_S:
        state["last_stale_alert_at"] = now
        if age_s is None:
            human_age = "never"
        else:
            human_age = f"{int(age_s // 86400)}d"
        send_signal_alert(
            f"💾 Self-heal: most-recent successful DB backup is "
            f"{human_age} old (threshold {_stale_days()}d). Postgres + "
            f"Neo4j + ChromaDB. Check workspace/backups/manifest.json "
            f"or run `bash deploy/scripts/backup.sh` from the host.",
            tag="db_backup:stale",
        )
        audit_event(
            "db_backup_stale_alert",
            age_s=int(age_s) if age_s is not None else None,
            threshold_s=threshold_s,
        )

    write_state_json(_STATE_FILE, state)
