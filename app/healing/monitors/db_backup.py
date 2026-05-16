"""DB backup monitor — runs weekly and alerts on per-component staleness.

Wave 0/1 closure (#A1, 2026-05-09). Off by default. Operators set
``HEALING_DB_BACKUP_ENABLED=1`` (or flip ``db_backup_enabled`` in
``runtime_settings``) to schedule weekly ChromaDB backups via the
gateway-side engine ``app/healing/db_backup.py``.

Two responsibilities:

  1. **Run the gateway-side backup** on internal weekly cadence. The
     monitor driver ticks this monitor daily (cheap), but the run
     itself is gated by ``DB_BACKUP_RUN_INTERVAL_S`` (default 7
     days, min 1 day). When ``DB_BACKUP_HOST_MANAGED=1`` the gateway
     only runs ChromaDB; pg + neo4j are owned by the host LaunchAgent
     (see ``crewai-team/scripts/install_db_backup.sh``).

  2. **Alert on per-component staleness**. The monitor walks the
     manifest for each of ``postgres``, ``neo4j``, ``chromadb`` and
     finds the most recent NON-SKIPPED success. If any component is
     older than ``DB_BACKUP_STALE_DAYS`` (default 14 days), the alert
     fires naming the stale components and pointing the operator at
     the right repair lever (LaunchAgent for pg/neo4j, gateway logs
     for chromadb). Per-monitor cooldown so a chronic failure doesn't
     spam.

The per-component check is the post-2026-05-16 fix that prevents a
dead host LaunchAgent from being silently masked by happy gateway
runs that only cover ChromaDB.

Default OFF because:
  * Laptop dev rarely needs automated DB backups.
  * Production K8s should use a dedicated CronJob with proper
    alerting, not the gateway monitor.

Operators turn it on once they've decided "the gateway is the
chromadb backup runner." The freshness alert still fires either
way — even with backups disabled, if a stale manifest indicates the
host script also stopped, the alert nudges them.
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


_COMPONENTS = ("postgres", "neo4j", "chromadb")


def _read_manifest_runs() -> list[dict]:
    if not _MANIFEST_PATH.exists():
        return []
    try:
        import json
        manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
        runs = manifest.get("runs", [])
        return list(runs) if isinstance(runs, list) else []
    except Exception:
        logger.debug("db_backup_monitor: manifest read failed", exc_info=True)
        return []


def _entry_age_s(entry: dict, now: float) -> float | None:
    """Parse the entry's completed_at; tolerant of Z suffix and missing field."""
    completed_at = entry.get("completed_at", "")
    if not completed_at:
        return None
    try:
        from datetime import datetime
        normalized = completed_at.replace("Z", "+00:00")
        return now - datetime.fromisoformat(normalized).timestamp()
    except Exception:
        return None


def _last_real_success_age_s(component: str, now: float) -> float | None:
    """Age in seconds of the most recent entry where this component
    actually ran and succeeded (ok=True AND skipped is not True).

    Returns None if no manifest / no real success for this component.
    Used by the post-split freshness check — a chromadb-only gateway
    entry must NOT count as a postgres backup, even though the
    skipped placeholder records ok=True.
    """
    for entry in reversed(_read_manifest_runs()):
        comp = entry.get(component, {})
        if not isinstance(comp, dict):
            continue
        if not comp.get("ok"):
            continue
        if comp.get("skipped"):
            continue
        age = _entry_age_s(entry, now)
        if age is not None:
            return age
    return None


def _stale_components(now: float, threshold_s: float) -> list[tuple[str, float | None]]:
    """Return list of (component, age_s) pairs whose last real success
    is older than ``threshold_s`` (or has never been observed)."""
    out: list[tuple[str, float | None]] = []
    for component in _COMPONENTS:
        age = _last_real_success_age_s(component, now)
        if age is None or age > threshold_s:
            out.append((component, age))
    return out


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

    # --- Phase 2: alert on per-component staleness -----------------------
    # Post-2026-05-16 split: gateway only writes chromadb entries when
    # DB_BACKUP_HOST_MANAGED is on; the host launchd LaunchAgent writes
    # postgres + neo4j entries. We need to alert separately so a dead
    # host agent (pg/n4j stale) doesn't get masked by the gateway's
    # otherwise-happy chromadb runs.
    threshold_s = _stale_days() * 24 * 3600
    stale = _stale_components(now, threshold_s)
    if stale and now - float(state.get("last_stale_alert_at", 0)) >= _STALE_ALERT_COOLDOWN_S:
        state["last_stale_alert_at"] = now
        parts = []
        for comp, age in stale:
            human = "never" if age is None else f"{int(age // 86400)}d"
            parts.append(f"{comp}={human}")
        # Hint operators where to look. pg/neo4j → host agent;
        # chromadb → gateway. If both, advise both.
        stale_names = {c for c, _ in stale}
        host_owned = {"postgres", "neo4j"} & stale_names
        gw_owned = {"chromadb"} & stale_names
        hints = []
        if host_owned:
            hints.append(
                "Check the host LaunchAgent: "
                "`launchctl list | grep db-backup` or "
                "`bash deploy/scripts/backup.sh` from the host."
            )
        if gw_owned:
            hints.append(
                "ChromaDB is gateway-owned; check gateway logs and "
                "DB_BACKUP_RUN_INTERVAL_S."
            )
        send_signal_alert(
            f"💾 Self-heal: DB backup stale (threshold {_stale_days()}d) — "
            f"{', '.join(parts)}. Manifest: workspace/backups/manifest.json. "
            + " ".join(hints),
            tag="db_backup:stale",
        )
        audit_event(
            "db_backup_stale_alert",
            stale_components=[c for c, _ in stale],
            threshold_s=threshold_s,
        )

    write_state_json(_STATE_FILE, state)
