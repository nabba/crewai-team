"""backup_freshness — proactive monitor for local DR tarball freshness.

PROGRAM §44.5 — Q6.5 P2#3. Catches the common failure mode "the
operator's backup-sync cron silently died." The Q6 posture
(`docs/RESILIENCE_POSTURE.md`) commits to dual-target off-host
backups (S3 + Google Drive) but the integrity verification of those
off-host targets is operator-managed.

What this monitor verifies (light, local-only):

  * `workspace/backups/dr/` exists
  * The directory has been touched (a tarball written) within
    ``2 × target_backup_age_days`` (default: 14 days)

What this monitor does NOT verify (deferred to a future off-host
drill, requires cloud SDKs):

  * S3 has a recent tarball
  * Google Drive has a recent tarball
  * Off-host tarballs match local SHA-256

The local-freshness check is the cheap proxy. If the operator's
sync script crashed, local-freshness is the FIRST signal — the
sync produces fresh local tarballs that get pushed off-host.
If local-freshness fails, off-host is even worse.

Master switch: ``backup_freshness_monitor_enabled`` (default ON).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


NAME = "backup_freshness"
CADENCE_SECONDS = 24 * 3600          # daily probe
MASTER_SWITCH_KEY = "backup_freshness_monitor_enabled"

# Default staleness threshold = 2× the posture's target_backup_age_days.
# At 7d target this means alert after 14d of no fresh backup.
_DEFAULT_STALENESS_MULTIPLIER = 2


def _default_backup_dir() -> Path:
    """Path to the local DR backup directory."""
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT) / "backups" / "dr"
    except Exception:
        return Path("/app/workspace/backups/dr")


def _newest_tarball_age_days(backup_dir: Path) -> float | None:
    """Return age in days of the newest .tar.gz under backup_dir,
    or None if no tarball found / directory missing.

    Uses mtime; tolerant of non-tarball files in the directory."""
    if not backup_dir.exists():
        return None
    newest_mtime: float | None = None
    try:
        for entry in backup_dir.iterdir():
            if not entry.is_file():
                continue
            name = entry.name.lower()
            if not (name.endswith(".tar.gz") or name.endswith(".tar")):
                continue
            try:
                mtime = entry.stat().st_mtime
            except OSError:
                continue
            if newest_mtime is None or mtime > newest_mtime:
                newest_mtime = mtime
    except OSError:
        return None
    if newest_mtime is None:
        return None
    return (time.time() - newest_mtime) / 86_400.0


def run() -> dict[str, Any]:
    """One monitor probe. Alerts via the canonical notify channel
    when no local tarball update within the staleness window.

    Failure-isolated."""
    summary: dict[str, Any] = {
        "checked": False,
        "newest_tarball_age_days": None,
        "stale": False,
        "alerts": 0,
        "errors": 0,
    }
    try:
        from app.runtime_settings import get_backup_freshness_monitor_enabled
        if not get_backup_freshness_monitor_enabled():
            summary["skipped"] = True
            return summary
    except Exception:
        pass  # fall through — default ON

    # Resolve staleness threshold from the posture decision.
    try:
        from app.resilience_drills.posture import POSTURE
        target_days = POSTURE.target_backup_age_days
    except Exception:
        target_days = 7
    threshold_days = target_days * _DEFAULT_STALENESS_MULTIPLIER

    backup_dir = _default_backup_dir()
    age_days = _newest_tarball_age_days(backup_dir)
    summary["checked"] = True
    summary["newest_tarball_age_days"] = age_days
    summary["threshold_days"] = threshold_days

    stale = age_days is None or age_days > threshold_days
    summary["stale"] = stale

    if stale:
        try:
            from app.notify import notify
            if age_days is None:
                body = (
                    f"No local DR tarball found at {backup_dir!s}. "
                    f"Either no backup has ever been written, or the "
                    f"backup directory has been deleted. Off-host "
                    f"backups (S3 + Google Drive) are operator-managed; "
                    f"this monitor catches the local-side prerequisite. "
                    f"See docs/RESILIENCE_POSTURE.md."
                )
            else:
                body = (
                    f"Local DR tarball is {age_days:.0f}d old "
                    f"(threshold: {threshold_days}d, target cadence: "
                    f"{target_days}d). The operator's backup-sync "
                    f"script may have stopped. See "
                    f"docs/RESILIENCE_POSTURE.md."
                )
            notify(
                title="🛡 Backup freshness: stale local tarball",
                body=body,
                url="/cp/drills",
                topic="backup_freshness_stale",
                critical=False,
                arbitrate=True,
            )
            summary["alerts"] = 1
        except Exception:
            logger.debug("backup_freshness: notify failed", exc_info=True)
            summary["errors"] = 1
    return summary
