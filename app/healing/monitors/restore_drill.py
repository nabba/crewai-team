"""Restore-drill freshness monitor (Phase H #1, 2026-05-10).

Companion to the DB backup engine (Phase A #A1). Backups exist;
this module ensures the *restore-from-backup* path is exercised
periodically so the operator finds out about a broken restore
BEFORE they need it.

Reads ``workspace/backups/restore_drill_manifest.json`` (written by
``deploy/scripts/restore-drill.sh``). Alerts when:

  * the manifest is missing entirely (no drill has ever run); or
  * the most recent drill is older than ``RESTORE_DRILL_STALE_DAYS``
    (default 100 days — comfortably within a quarterly cadence); or
  * the most recent drill failed (``all_ok: false``).

This module never RUNS the drill (the drill spins up scratch
containers; doing that from inside the gateway risks cross-resource
issues). It only watches the manifest. Operators run the drill from
cron / launchd:

```
@quarterly cd /path/to/crewai-team && bash deploy/scripts/restore-drill.sh
```

Cadence: daily probe, alert dedup 14 days.

Master switch: ``RESTORE_DRILL_MONITOR_ENABLED`` (default ON).
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.healing.handlers._common import (
    audit_event,
    read_state_json,
    send_signal_alert,
    write_state_json,
)

logger = logging.getLogger(__name__)


_STATE_FILE = "restore_drill_monitor.json"
_MANIFEST_PATH = Path("/app/workspace/backups/restore_drill_manifest.json")
_RUN_CADENCE_S = 24 * 3600
_DEFAULT_STALE_DAYS = 100
_DEDUP_WINDOW_S = 14 * 86400


def _enabled() -> bool:
    return os.getenv("RESTORE_DRILL_MONITOR_ENABLED", "true").lower() in (
        "true", "1", "yes", "on",
    )


def _stale_days() -> int:
    raw = os.getenv("RESTORE_DRILL_STALE_DAYS", str(_DEFAULT_STALE_DAYS)).strip()
    try:
        return max(7, int(raw))
    except ValueError:
        return _DEFAULT_STALE_DAYS


def _read_manifest() -> dict | None:
    if not _MANIFEST_PATH.exists():
        return None
    try:
        return json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("restore_drill: manifest unreadable", exc_info=True)
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


def run() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "ran": False, "manifest_present": False,
        "last_drill_age_days": None, "last_drill_ok": None,
        "alert_fired": False,
    }
    if not _enabled():
        return summary

    state = read_state_json(_STATE_FILE, {
        "last_run_at": 0.0, "last_alert_at": 0.0,
    })
    now = time.time()
    if now - float(state.get("last_run_at", 0)) < _RUN_CADENCE_S:
        return summary
    state["last_run_at"] = now
    summary["ran"] = True

    manifest = _read_manifest()
    threshold_days = _stale_days()

    if manifest is None:
        # No drill has ever run.
        summary["manifest_present"] = False
        if now - float(state.get("last_alert_at", 0)) >= _DEDUP_WINDOW_S:
            state["last_alert_at"] = now
            try:
                send_signal_alert(
                    "🛑 Restore drill: no drill manifest found "
                    "(`workspace/backups/restore_drill_manifest.json`). "
                    "Backups exist but the restore path has never been "
                    "tested. Run "
                    "`bash deploy/scripts/restore-drill.sh` and add to "
                    "cron (quarterly). See `deploy/RESTORE.md`.",
                    tag="restore_drill:never_run",
                )
                summary["alert_fired"] = True
            except Exception:
                logger.debug("restore_drill: alert send failed", exc_info=True)
        write_state_json(_STATE_FILE, state)
        audit_event(
            "restore_drill_check", manifest_present=False, alerted=summary["alert_fired"],
        )
        return summary

    summary["manifest_present"] = True
    age = _last_drill_age_days(manifest, now)
    summary["last_drill_age_days"] = round(age, 1) if age is not None else None
    summary["last_drill_ok"] = manifest.get("last_drill_ok")

    needs_alert = (
        age is None
        or age > threshold_days
        or manifest.get("last_drill_ok") is False
    )

    if needs_alert and now - float(state.get("last_alert_at", 0)) >= _DEDUP_WINDOW_S:
        state["last_alert_at"] = now
        if manifest.get("last_drill_ok") is False:
            body = (
                f"🛑 Restore drill: most recent drill FAILED "
                f"({manifest.get('last_drill_at', '?')}). "
                f"Backups may not be restorable. Check "
                f"`workspace/backups/restore_drill_manifest.json` for "
                f"the smoke-check details."
            )
        else:
            human_age = (
                f"{int(age)} d" if age is not None else "never"
            )
            body = (
                f"⏰ Restore drill: last successful drill was "
                f"{human_age} ago (threshold {threshold_days} d). "
                f"Run `bash deploy/scripts/restore-drill.sh` to keep "
                f"the restore path tested."
            )
        try:
            send_signal_alert(body, tag="restore_drill:stale")
            summary["alert_fired"] = True
        except Exception:
            logger.debug("restore_drill: alert send failed", exc_info=True)

    write_state_json(_STATE_FILE, state)
    audit_event(
        "restore_drill_check",
        manifest_present=True,
        last_drill_age_days=summary["last_drill_age_days"],
        last_drill_ok=summary["last_drill_ok"],
        alerted=summary["alert_fired"],
    )
    return summary
