"""Cron-liveness monitor — detect APScheduler cron jobs that haven't fired.

Eight cron jobs run on schedule out of ``app/main.py:lifespan()`` (e.g.
``self_improve``, ``code_audit``, ``error_resolution``, ``evolution``,
``retrospective``, ``benchmark_snapshot``, ``workspace_sync``). Each
job leaves a footprint on disk when it runs. If a footprint hasn't been
touched in N × the expected interval, something is wrong with the
scheduler.

We don't have a direct way to introspect APScheduler's job state from
outside the gateway process, so we use file mtimes as proxies. The
config is intentionally generous (3× the expected interval) so a slow
run doesn't fire false alarms.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from app.healing.handlers._common import (
    audit_event,
    read_state_json,
    send_signal_alert,
    write_state_json,
)

logger = logging.getLogger(__name__)

_STATE_FILE = "cron_liveness_alerts.json"
_ALERT_WINDOW_S = 12 * 3600

# Job-name → (relative-footprint-path, expected-interval-seconds).
# Multiplied by 3× before alerting (allows for slow runs / clock skew).
_JOBS = [
    # (name, footprint_relpath, expected_seconds)
    ("error_resolution", "workspace/error_tracker.json", 30 * 60),
    ("code_audit", "workspace/audit_journal.json", 4 * 3600),
    ("workspace_sync", "workspace/.git/HEAD", 1 * 3600),
    ("retrospective", "workspace/retrospective", 24 * 3600),
    ("self_improve", "workspace/self_improvement", 24 * 3600),
]


def run() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    now = time.time()
    state = read_state_json(_STATE_FILE, {"jobs": {}})

    stale_jobs: list[dict] = []

    for name, footprint, interval_s in _JOBS:
        p = repo_root / footprint
        if not p.exists():
            # Footprint never written — could be a fresh install OR
            # a job that never ran. Don't alert; just record.
            continue
        try:
            age = now - p.stat().st_mtime
        except Exception:
            continue
        threshold = interval_s * 3
        if age > threshold:
            stale_jobs.append({
                "name": name,
                "footprint": footprint,
                "age_s": int(age),
                "expected_interval_s": interval_s,
                "threshold_s": threshold,
            })

    audit_event(
        "cron_liveness_check",
        n_stale=len(stale_jobs),
        n_checked=len(_JOBS),
    )

    if not stale_jobs:
        return

    # De-dup: alert only once per (job-set, 12 h).
    key = "+".join(sorted(j["name"] for j in stale_jobs))
    entry = state.setdefault("jobs", {}).setdefault(key, {"last_alert_at": 0})
    if now - entry.get("last_alert_at", 0) < _ALERT_WINDOW_S:
        return

    entry["last_alert_at"] = now
    entry["stale_jobs"] = stale_jobs
    write_state_json(_STATE_FILE, state)

    lines = [
        f"  • `{j['name']}` — last run {j['age_s'] // 60} min ago "
        f"(expected ≤ {j['expected_interval_s'] // 60} min)"
        for j in stale_jobs
    ]
    send_signal_alert(
        "⏰ Self-heal: cron job(s) appear stale — last footprints are "
        "older than 3× their expected interval:\n\n"
        + "\n".join(lines)
        + "\n\nCheck APScheduler / gateway logs.",
        tag="cron_liveness",
    )
