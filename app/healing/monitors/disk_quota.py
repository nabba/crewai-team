"""Disk-quota guard — proactive Signal alert when free space gets tight.

The atomic-write helpers in ``app/safe_io.py`` don't pre-check free
space. On a multi-year deployment, a slowly-filling workspace volume
will eventually corrupt half-written files. This monitor runs every
~5 min and alerts the operator when free space drops below a tunable
threshold.

Tunables via env vars:

  * ``HEALING_DISK_FREE_WARN_GB``  — alert threshold (default 5 GB)
  * ``HEALING_DISK_FREE_CRIT_GB``  — critical alert threshold (default 1 GB)

Alerts are deduped: at most one alert per (level, mountpoint) per 12 h.
"""
from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path

from app.healing.handlers._common import (
    audit_event,
    read_state_json,
    send_signal_alert,
    write_state_json,
)

logger = logging.getLogger(__name__)

_STATE_FILE = "disk_quota_alerts.json"
_ALERT_WINDOW_S = 12 * 3600


def _threshold_gb(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _alert_with_cooldown(level: str, mountpoint: str, free_gb: float, total_gb: float) -> bool:
    state = read_state_json(_STATE_FILE, {"alerts": {}})
    key = f"{level}:{mountpoint}"
    entry = state.setdefault("alerts", {}).setdefault(key, {"last_alert_at": 0})
    now = time.time()
    if now - entry.get("last_alert_at", 0) < _ALERT_WINDOW_S:
        return False
    entry["last_alert_at"] = now
    entry["free_gb"] = round(free_gb, 2)
    entry["total_gb"] = round(total_gb, 2)
    write_state_json(_STATE_FILE, state)
    return True


def run() -> None:
    workspace = Path(__file__).resolve().parents[3] / "workspace"
    if not workspace.exists():
        return

    try:
        usage = shutil.disk_usage(str(workspace))
    except Exception:
        logger.debug("disk_quota: shutil.disk_usage failed", exc_info=True)
        return

    free_gb = usage.free / (1024 ** 3)
    total_gb = usage.total / (1024 ** 3)
    used_pct = (1 - usage.free / usage.total) * 100 if usage.total else 0

    crit_gb = _threshold_gb("HEALING_DISK_FREE_CRIT_GB", 1.0)
    warn_gb = _threshold_gb("HEALING_DISK_FREE_WARN_GB", 5.0)

    audit_event(
        "disk_quota_check",
        free_gb=round(free_gb, 2), total_gb=round(total_gb, 2),
        used_pct=round(used_pct, 1),
    )

    if free_gb < crit_gb:
        if _alert_with_cooldown("critical", str(workspace), free_gb, total_gb):
            send_signal_alert(
                f"🚨 CRITICAL — workspace disk free is **{free_gb:.1f} GB** "
                f"(of {total_gb:.0f} GB, {used_pct:.0f}% used). Below "
                f"critical threshold {crit_gb:.0f} GB. Risk of corruption "
                f"on next write. Free space NOW.",
                tag="disk_quota",
            )
    elif free_gb < warn_gb:
        if _alert_with_cooldown("warn", str(workspace), free_gb, total_gb):
            send_signal_alert(
                f"⚠️  Workspace disk free is **{free_gb:.1f} GB** "
                f"(of {total_gb:.0f} GB, {used_pct:.0f}% used). Below "
                f"warn threshold {warn_gb:.0f} GB. Plan cleanup.",
                tag="disk_quota",
            )
