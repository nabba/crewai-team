"""Disk-quota guard — proactive alert + immediate-retention auto-action.

The atomic-write helpers in ``app/safe_io.py`` don't pre-check free
space. On a multi-year deployment, a slowly-filling workspace volume
will eventually corrupt half-written files. This monitor runs every
~5 min and:

  1. **Alerts** the operator via Signal when free space drops below a
     tunable threshold (deduped 1 per 12 h per level).
  2. **Auto-actions** at WARN-level: invokes the three retention
     monitors (``retention.run_chromadb``, ``run_worktrees``,
     ``run_attachments``) immediately rather than waiting for their
     own cadence (which could be days). This is the single highest-
     leverage clean auto-action: retention only deletes already-stale
     data, so the worst case is "we did the next scheduled cleanup
     a few days early." Disk pressure is the trigger; retention is
     the response.

Tunables via env vars:

  * ``HEALING_DISK_FREE_WARN_GB``  — alert threshold (default 5 GB)
  * ``HEALING_DISK_FREE_CRIT_GB``  — critical alert threshold (default 1 GB)
  * ``HEALING_DISK_AUTO_RETENTION_ENABLED`` — master switch for the
    auto-action (default ``true``). Disable to revert to alert-only
    behaviour.

Audit + Signal alert both fire whether or not the auto-action did
anything; the operator always sees disk pressure events.
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

    if free_gb < warn_gb:
        # Auto-action: run retention immediately. Always attempted on
        # WARN+ regardless of the alert cooldown — retention is cheap
        # and idempotent. A separate enabled flag lets the operator
        # revert to alert-only behaviour if needed.
        retention_summary = _try_immediate_retention()
    else:
        retention_summary = None

    if free_gb < crit_gb:
        if _alert_with_cooldown("critical", str(workspace), free_gb, total_gb):
            tail = _format_retention_tail(retention_summary)
            send_signal_alert(
                f"🚨 CRITICAL — workspace disk free is **{free_gb:.1f} GB** "
                f"(of {total_gb:.0f} GB, {used_pct:.0f}% used). Below "
                f"critical threshold {crit_gb:.0f} GB. Risk of corruption "
                f"on next write. Free space NOW.{tail}",
                tag="disk_quota",
            )
    elif free_gb < warn_gb:
        if _alert_with_cooldown("warn", str(workspace), free_gb, total_gb):
            tail = _format_retention_tail(retention_summary)
            send_signal_alert(
                f"⚠️  Workspace disk free is **{free_gb:.1f} GB** "
                f"(of {total_gb:.0f} GB, {used_pct:.0f}% used). Below "
                f"warn threshold {warn_gb:.0f} GB. Plan cleanup.{tail}",
                tag="disk_quota",
            )


# ── Auto-action: immediate retention ─────────────────────────────────


def _auto_retention_enabled() -> bool:
    return os.getenv("HEALING_DISK_AUTO_RETENTION_ENABLED", "true").lower() in (
        "true", "1", "yes", "on",
    )


def _try_immediate_retention() -> dict | None:
    """Invoke the three retention monitors in-line. Returns a per-
    target outcome dict or None when the auto-action is disabled.

    Failures are isolated per-target: chromadb failing doesn't stop
    worktrees from running. Each retention.run_* function is already
    designed to be safely re-callable (state-aware, no-op when nothing
    to do).
    """
    if not _auto_retention_enabled():
        return None

    summary: dict = {}
    targets = ("chromadb", "worktrees", "attachments")
    try:
        from app.healing.monitors import retention
    except Exception:
        logger.debug(
            "disk_quota.auto_retention: retention import failed",
            exc_info=True,
        )
        return None

    for name in targets:
        runner = getattr(retention, f"run_{name}", None)
        if runner is None:
            summary[name] = "no_runner"
            continue
        try:
            runner()
            summary[name] = "ok"
        except Exception as exc:  # noqa: BLE001
            summary[name] = f"failed: {type(exc).__name__}"
            logger.debug(
                "disk_quota.auto_retention: %s raised", name, exc_info=True,
            )

    audit_event(
        "disk_quota_auto_retention",
        outcomes=summary,
    )
    return summary


def _format_retention_tail(summary: dict | None) -> str:
    """Append retention outcome to the Signal alert tail. Empty
    string when the auto-action is off; ``ok``/``failed`` summary
    when it ran."""
    if not summary:
        return ""
    parts = [f"{k}={v}" for k, v in summary.items()]
    return f"\nAuto-retention ran: {', '.join(parts)}."
