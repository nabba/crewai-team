"""Version-upgrade drill freshness monitor (§2.5).

Companion to ``deploy/scripts/version-upgrade-drill.sh``. The drill
proves that the freshest backup set can be restored into NEWER
versions of Postgres + Neo4j + ChromaDB and survive whatever
forward-version migrations those versions require. This module
ensures the drill is run on cadence and alerts when it isn't or
when its most recent run failed.

Distinct from :mod:`app.healing.monitors.restore_drill` — that
monitor watches the *current-version* restore path; this one
watches the *forward-version-migration* path. Both are needed:
restore breaks for different reasons than upgrade does.

Reads ``workspace/backups/version_upgrade_drill_manifest.json``.
Alerts when:

  * the manifest is missing entirely (no upgrade drill ever); or
  * the most recent drill is older than
    ``VERSION_UPGRADE_DRILL_STALE_DAYS`` (default 100 days —
    quarterly cadence with slack); or
  * the most recent drill failed (``last_drill_ok: false``).

Never RUNS the drill itself (the drill spins up scratch containers;
running that from inside the gateway is risky). Operators schedule:

::

    @quarterly cd /path/to/crewai-team && \
        bash deploy/scripts/version-upgrade-drill.sh

Cadence: daily probe, alert dedup 14 days.

Master switch: ``VERSION_UPGRADE_DRILL_MONITOR_ENABLED`` (default ON).
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


_STATE_FILE = "version_upgrade_drill_monitor.json"
_DEFAULT_MANIFEST_PATH = Path(
    "/app/workspace/backups/version_upgrade_drill_manifest.json"
)
_RUN_CADENCE_S = 24 * 3600
_DEFAULT_STALE_DAYS = 100
_DEDUP_WINDOW_S = 14 * 86400


def _enabled() -> bool:
    return os.getenv("VERSION_UPGRADE_DRILL_MONITOR_ENABLED", "true").lower() in (
        "true", "1", "yes", "on",
    )


def _stale_days() -> int:
    raw = os.getenv(
        "VERSION_UPGRADE_DRILL_STALE_DAYS",
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
        logger.debug(
            "version_upgrade_drill: manifest unreadable", exc_info=True,
        )
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


def _format_targets(manifest: dict) -> str:
    """Human-readable target-version line for alert bodies."""
    targets = manifest.get("last_target_versions") or {}
    if not targets:
        return "(unknown)"
    parts = [f"{k}={v}" for k, v in sorted(targets.items())]
    return ", ".join(parts)


def run(
    *,
    manifest_path: Path | str | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Single-pass freshness probe. Returns a structured summary.

    Test/operator hook: ``manifest_path`` overrides the storage
    location. ``now`` overrides time.time() for deterministic tests.
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
    })

    # Cadence guard: skip if we ran within the last 24h. Tests bypass
    # by providing ``now`` outside the cadence window.
    if cur - float(state.get("last_run_at", 0)) < _RUN_CADENCE_S:
        return summary
    state["last_run_at"] = cur
    summary["ran"] = True

    manifest = _read_manifest(mp)
    threshold_days = _stale_days()

    def _maybe_alert(tag: str, body: str) -> None:
        last_alerts = state.setdefault("last_alert_at", {})
        if not isinstance(last_alerts, dict):  # state-file schema migration
            last_alerts = {}
            state["last_alert_at"] = last_alerts
        last = float(last_alerts.get(tag, 0))
        if cur - last < _DEDUP_WINDOW_S:
            return
        try:
            send_signal_alert(body, tag=tag)
        except Exception:
            logger.debug(
                "version_upgrade_drill: send_signal_alert raised",
                exc_info=True,
            )
        last_alerts[tag] = cur
        summary["alert_fired"] = True
        summary["alert_tag"] = tag

    if manifest is None:
        summary["manifest_present"] = False
        _maybe_alert(
            "version_upgrade_drill:never_run",
            "🛑 Version-upgrade drill: no manifest found at "
            "`workspace/backups/version_upgrade_drill_manifest.json`. "
            "Backups exist but the *forward-version-migration* path has "
            "never been tested — when you next bump Postgres / Neo4j / "
            "ChromaDB you may discover the migration breaks on real data. "
            "Run `bash deploy/scripts/version-upgrade-drill.sh` and add "
            "it to cron (quarterly).",
        )
    else:
        summary["manifest_present"] = True
        age = _last_drill_age_days(manifest, cur)
        summary["last_drill_age_days"] = age
        summary["last_drill_ok"] = manifest.get("last_drill_ok")

        if manifest.get("last_drill_ok") is False:
            _maybe_alert(
                "version_upgrade_drill:failed",
                f"❌ Version-upgrade drill FAILED on the most recent run.\n"
                f"Target versions: {_format_targets(manifest)}\n"
                f"Drill manifest: `workspace/backups/"
                f"version_upgrade_drill_manifest.json`\n"
                f"Most-recent log path is in the manifest's runs[-1].log.\n"
                f"Do NOT bump production versions until the drill passes.",
            )
        elif age is None:
            _maybe_alert(
                "version_upgrade_drill:stale",
                "⚠️ Version-upgrade drill: manifest exists but lacks a "
                "parseable `last_drill_at` timestamp. Re-run the drill.",
            )
        elif age > threshold_days:
            _maybe_alert(
                "version_upgrade_drill:stale",
                f"⚠️ Version-upgrade drill is stale: last run "
                f"{int(age)} days ago (threshold "
                f"{threshold_days}d). Run `bash "
                f"deploy/scripts/version-upgrade-drill.sh` to refresh.",
            )

    audit_event(
        "version_upgrade_drill_pass",
        manifest_present=summary["manifest_present"],
        last_drill_age_days=(
            int(summary["last_drill_age_days"])
            if summary["last_drill_age_days"] is not None else None
        ),
        last_drill_ok=summary["last_drill_ok"],
        alert_fired=summary["alert_fired"],
        alert_tag=summary["alert_tag"],
    )

    write_state_json(_STATE_FILE, state)
    return summary
