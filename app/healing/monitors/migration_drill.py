"""migration_drill — schema-migration drill freshness monitor.

PROGRAM §48 — Q13.1 (year-2+ resilience #2.2). Companion to
``deploy/scripts/migration-drill.sh``.

Distinct from siblings:

  * :mod:`app.healing.monitors.restore_drill` — backup restores
    against CURRENT versions. Catches "can we restore at all?"
  * :mod:`app.healing.monitors.version_upgrade_drill` — backup
    restores against NEWER versions of PG/Neo4j/Chroma. Catches
    "does pg_upgrade work on real data?"
  * **THIS** — backup restores against current versions then
    applies any ``migrations/*.sql`` past the snapshot's version,
    then runs ``startup_migrations.apply_all``, then runs a tiny
    end-to-end smoke (one Commander.handle dispatch against a
    fixture). Catches "does TODAY's code read a 6-month-old
    backup?" — the user's exact §2.2 concern.

All three drills are needed because they break for different
reasons. This monitor watches the schema-migration drill's
manifest:

  * the manifest is missing entirely (no migration drill ever); or
  * the most recent drill is older than
    ``MIGRATION_DRILL_STALE_DAYS`` (default 100 days — quarterly
    cadence with slack); or
  * the most recent drill failed (``last_drill_ok: false``).

Never RUNS the drill itself (the drill spins up scratch containers
+ applies SQL; running that from inside the gateway is risky).
Operators schedule:

::

    @quarterly cd /path/to/crewai-team && \
        bash deploy/scripts/migration-drill.sh

Cadence: daily probe, alert dedup 14 days.

Master switch: ``migration_drill_monitor_enabled`` (default ON).
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from app.healing.handlers._common import (
    audit_event,
    read_state_json,
    send_signal_alert,
    write_state_json,
)

logger = logging.getLogger(__name__)


NAME = "migration_drill"
CADENCE_SECONDS = 24 * 3600
MASTER_SWITCH_KEY = "migration_drill_monitor_enabled"

_STATE_FILE = "migration_drill_monitor.json"
_DEFAULT_MANIFEST_PATH = Path(
    "/app/workspace/backups/migration_drill_manifest.json"
)
_DEFAULT_STALE_DAYS = 100
_DEDUP_WINDOW_S = 14 * 86400


def _enabled() -> bool:
    # Mirror runtime_settings + env-var pattern from sibling monitors.
    try:
        from app.runtime_settings import get_migration_drill_monitor_enabled
        return get_migration_drill_monitor_enabled()
    except Exception:
        return os.getenv("MIGRATION_DRILL_MONITOR_ENABLED", "true").lower() in (
            "true", "1", "yes", "on",
        )


def _stale_days() -> int:
    raw = os.getenv(
        "MIGRATION_DRILL_STALE_DAYS",
        str(_DEFAULT_STALE_DAYS),
    ).strip()
    try:
        return max(7, int(raw))
    except ValueError:
        return _DEFAULT_STALE_DAYS


def _read_manifest(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("migration_drill: manifest unreadable", exc_info=True)
        return None


def _last_drill_age_days(manifest: dict, now: float) -> float | None:
    last = manifest.get("last_drill_at")
    if not last:
        return None
    try:
        dt = datetime.fromisoformat(str(last).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return (now - dt.timestamp()) / 86400


def _emit_continuity_ledger_event(
    *,
    summary: str,
    detail: dict[str, Any],
) -> None:
    """Best-effort emit to identity continuity ledger on landmark
    transitions (first drill ever / drill failed / drill recovered)."""
    try:
        from app.identity.continuity_ledger import record_event
        record_event(
            kind="schema_migration_drill",
            actor="migration_drill_monitor",
            summary=summary,
            detail=detail,
        )
    except Exception:
        logger.debug("migration_drill: ledger emit failed", exc_info=True)


def run(
    *,
    manifest_path: Path | str | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Single-pass freshness probe. Returns a structured summary.

    Test/operator hook: ``manifest_path`` overrides the storage
    location. ``now`` overrides ``time.time()`` for deterministic
    tests.
    """
    summary: dict[str, Any] = {
        "ran": False,
        "manifest_present": False,
        "last_drill_age_days": None,
        "last_drill_ok": None,
        "alert_fired": False,
        "alert_tag": None,
    }
    if not _enabled():
        return summary

    mp = Path(manifest_path) if manifest_path else _DEFAULT_MANIFEST_PATH
    cur = float(now) if now is not None else time.time()

    state = read_state_json(_STATE_FILE, {
        "last_run_at": 0.0,
        "last_alert_at": {},  # tag → timestamp
        "last_landmark": None,  # for continuity-ledger dedup
    })

    if cur - float(state.get("last_run_at", 0)) < CADENCE_SECONDS:
        return summary
    state["last_run_at"] = cur
    summary["ran"] = True

    manifest = _read_manifest(mp)
    threshold_days = _stale_days()

    def _maybe_alert(tag: str, body: str) -> None:
        last_alerts = state.setdefault("last_alert_at", {})
        if not isinstance(last_alerts, dict):
            last_alerts = {}
            state["last_alert_at"] = last_alerts
        last = float(last_alerts.get(tag, 0))
        if cur - last < _DEDUP_WINDOW_S:
            return
        try:
            send_signal_alert(body, tag=tag)
        except Exception:
            logger.debug("migration_drill: send_signal_alert raised", exc_info=True)
        last_alerts[tag] = cur
        summary["alert_fired"] = True
        summary["alert_tag"] = tag

    if manifest is None:
        summary["manifest_present"] = False
        _maybe_alert(
            "migration_drill:never_run",
            "🛑 Schema-migration drill: no manifest found at "
            "`workspace/backups/migration_drill_manifest.json`. The "
            "*backups+migrations+smoke* path has never been tested — "
            "today's code might fail to read a 6-month-old backup "
            "(the migrations 030-035 added new tables). Run "
            "`bash deploy/scripts/migration-drill.sh` and add it to "
            "cron (quarterly).",
        )
        if state.get("last_landmark") != "never_run":
            _emit_continuity_ledger_event(
                summary="Schema-migration drill never run — alerted.",
                detail={"manifest_path": str(mp)},
            )
            state["last_landmark"] = "never_run"
    else:
        summary["manifest_present"] = True
        age = _last_drill_age_days(manifest, cur)
        summary["last_drill_age_days"] = age
        summary["last_drill_ok"] = manifest.get("last_drill_ok")

        if age is None:
            _maybe_alert(
                "migration_drill:malformed_manifest",
                "⚠️ Schema-migration drill manifest is missing a valid "
                "`last_drill_at` field. Run the drill again to refresh.",
            )
        elif age > threshold_days:
            _maybe_alert(
                "migration_drill:stale",
                f"🛑 Schema-migration drill is {age:.0f}d stale "
                f"(threshold: {threshold_days}d, quarterly cadence). "
                f"Run `bash deploy/scripts/migration-drill.sh`.",
            )
            if state.get("last_landmark") != "stale":
                _emit_continuity_ledger_event(
                    summary=(
                        f"Schema-migration drill is {age:.0f}d stale — "
                        f"alerted."
                    ),
                    detail={"age_days": age, "threshold_days": threshold_days},
                )
                state["last_landmark"] = "stale"
        elif manifest.get("last_drill_ok") is False:
            _maybe_alert(
                "migration_drill:failed",
                "🚨 Most-recent schema-migration drill FAILED. "
                "Today's code may not read a 6-month-old backup "
                "correctly. Check "
                f"`{mp.parent / manifest.get('log', '?')}` for the "
                "failure trace.",
            )
            if state.get("last_landmark") != "failed":
                _emit_continuity_ledger_event(
                    summary="Schema-migration drill FAILED — alerted.",
                    detail={"manifest": manifest},
                )
                state["last_landmark"] = "failed"
        else:
            # Healthy state — clear landmark for next recovery event.
            if state.get("last_landmark"):
                _emit_continuity_ledger_event(
                    summary=(
                        f"Schema-migration drill recovered: last drill "
                        f"{age:.0f}d old, ok=True."
                    ),
                    detail={"age_days": age, "manifest": manifest},
                )
            state["last_landmark"] = None

    write_state_json(_STATE_FILE, state)
    try:
        audit_event(
            "migration_drill_probe",
            {
                "manifest_present": summary["manifest_present"],
                "age_days": summary["last_drill_age_days"],
                "alert_fired": summary["alert_fired"],
            },
        )
    except Exception:
        logger.debug("migration_drill: audit_event failed", exc_info=True)
    return summary
