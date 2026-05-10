"""Silent regression detector (Phase C #2, 2026-05-09).

The classic failure mode for an autonomous system: a CR landed, a
ratchet flipped, an adapter promoted — and afterward, things "just
work less." No errors thrown, no Signal alerts, just a quiet
degradation in throughput, completion rate, or latency. The audit
log shows the change went through; the cron-liveness monitor sees
the cron still firing; nobody notices for weeks.

This module's job: walk recent audit-journal events, compute a
per-event-type baseline, and alert when the last 24 h regress
below that baseline AND there's a recent change (commit, CR, or
ratchet) on the same window — pointing the operator at the suspect
change rather than just the symptom.

Cadence: 4 h (the daemon driver pings daily; we self-cadence to ≤6×/day).

Algorithm:
  1. Walk the rolled audit journal (``workspace/audit_journal/``) for the last 14 days.
  2. For each "cron-like" event type (the ones with regular cadence),
     compute: count_last_24h, count_prior_baseline (mean over the
     13-day window before the trailing 24h).
  3. Flag any event whose last-24h count dropped > REGRESSION_PCT
     from baseline (default 30%).
  4. For each flag, gather recent suspects (last 48 h):
        * git commits (via ``git log``)
        * change-requests applied (via PG ``control_plane.audit_log``
          where action='change_request_apply' OR 'ratchet_set')
        * governance ratchet changes
  5. Send one Signal alert per flag, listing suspects.
  6. Per-event de-dup window: 24 h. Same regression won't re-alert.

State at ``workspace/self_heal/silent_regression_alerts.json``::

    {"flags": {<event_type>: {"last_alert_at": ts, "ratio": 0.6}}}
"""
from __future__ import annotations

import logging
import os
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.audit import journal as audit_journal
from app.healing.handlers._common import (
    audit_event,
    read_state_json,
    send_signal_alert,
    write_state_json,
)

logger = logging.getLogger(__name__)


_STATE_FILE = "silent_regression_alerts.json"
_RUN_CADENCE_S = 4 * 3600
_BASELINE_DAYS = 13
_RECENT_WINDOW_S = 24 * 3600
_DEDUP_WINDOW_S = 24 * 3600

# Event types we treat as "should be running on regular cadence".
# Hand-picked from the audit-journal taxonomy. Adding a new one =
# observe it run regularly, then drop the name in here.
_CRON_LIKE_EVENTS: tuple[str, ...] = (
    "error_resolution",
    "code_audit",
    "self_improve",
    "evolution",
    "retrospective",
    "benchmark_snapshot",
    "workspace_sync",
)

# Min count for a baseline to be trusted. With <8 baseline samples
# the variance is too high — a single missing run looks like a
# regression. Bumps for any baseline-thin event silently ignored.
_MIN_BASELINE_SAMPLES = 8


def _enabled() -> bool:
    return os.getenv("HEALING_SILENT_REGRESSION_ENABLED", "true").lower() in (
        "true", "1", "yes", "on",
    )


def _regression_pct() -> float:
    raw = os.getenv("SILENT_REGRESSION_PCT", "0.30").strip()
    try:
        return max(0.05, min(0.9, float(raw)))
    except ValueError:
        return 0.30


def _read_audit_journal(lookback_days: int) -> list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    out: list[dict] = []
    for row in audit_journal.read_since(cutoff):
        ts = (row.get("ts") or "").replace("Z", "+00:00")
        try:
            t = datetime.fromisoformat(ts)
        except (ValueError, TypeError):
            continue
        out.append({**row, "_ts": t})
    return out


# ── Counter ──────────────────────────────────────────────────────────────


def _bin_counts(rows: list[dict], now: datetime) -> dict[str, dict[str, int]]:
    """For each event type, return ``{recent_24h, baseline_total}``."""
    cutoff_recent = now - timedelta(seconds=_RECENT_WINDOW_S)
    cutoff_baseline = cutoff_recent - timedelta(days=_BASELINE_DAYS)

    counts: dict[str, dict[str, int]] = {}
    for ev in rows:
        et = ev.get("event") or ""
        if et not in _CRON_LIKE_EVENTS:
            continue
        bucket = counts.setdefault(et, {"recent_24h": 0, "baseline_total": 0})
        if ev["_ts"] >= cutoff_recent:
            bucket["recent_24h"] += 1
        elif ev["_ts"] >= cutoff_baseline:
            bucket["baseline_total"] += 1
    return counts


def _ratio_against_baseline(recent: int, baseline_total: int) -> float | None:
    """Return the ratio of recent rate to baseline rate. None if baseline is too thin."""
    if baseline_total < _MIN_BASELINE_SAMPLES:
        return None
    # Convert to events-per-day for both windows.
    recent_per_day = recent  # last 24 h IS one day
    baseline_per_day = baseline_total / _BASELINE_DAYS
    if baseline_per_day <= 0:
        return None
    return recent_per_day / baseline_per_day


# ── Suspect gathering ────────────────────────────────────────────────────


def _recent_git_commits(hours: int = 48, max_n: int = 10) -> list[str]:
    """Return short summary lines for git commits in the last ``hours``."""
    repo = Path("/app")
    if not (repo / ".git").exists():
        # Try the host path the gateway sometimes runs at.
        repo = Path("/app/workspace")
        if not (repo / ".git").exists():
            return []
    try:
        result = subprocess.run(
            ["git", "log", f"--since={hours} hours ago",
             "--pretty=format:%h %s", f"-{max_n}"],
            cwd=str(repo), capture_output=True, text=True, timeout=8,
        )
        if result.returncode != 0:
            return []
        return [ln for ln in result.stdout.splitlines() if ln.strip()]
    except Exception:
        logger.debug("silent_regression: git log failed", exc_info=True)
        return []


def _recent_change_requests(hours: int = 48) -> list[str]:
    """Recent CRs / ratchet changes from PG audit_log. [] on PG miss."""
    try:
        from app.control_plane.audit import AuditTrail
    except Exception:
        return []
    try:
        trail = AuditTrail()
        since = datetime.now(timezone.utc) - timedelta(hours=hours)
        rows = trail.query(action_prefix="change_request_apply", since=since, limit=20)
        rows += trail.query(action_prefix="ratchet_", since=since, limit=20)
        rows += trail.query(action_prefix="amendment_", since=since, limit=20)
    except Exception:
        return []
    out: list[str] = []
    for r in rows:
        action = r.get("action") or ""
        actor = r.get("actor") or ""
        rid = r.get("resource_id") or ""
        out.append(f"{action} by {actor} (resource={rid})")
    return out


# ── Alerting ─────────────────────────────────────────────────────────────


def _format_alert(event_type: str, recent: int, baseline_total: int,
                  ratio: float, suspects: list[str], commits: list[str]) -> str:
    drop_pct = int((1.0 - ratio) * 100)
    baseline_per_day = baseline_total / _BASELINE_DAYS
    lines = [
        f"📉 Silent regression: `{event_type}` ran {recent}× in the last 24 h, "
        f"{drop_pct}% below the {_BASELINE_DAYS}-day baseline of "
        f"{baseline_per_day:.1f}×/day.",
    ]
    if commits:
        lines.append("\nRecent git commits (last 48 h):")
        for c in commits[:5]:
            lines.append(f"  • {c}")
    if suspects:
        lines.append("\nRecent control-plane changes:")
        for s in suspects[:5]:
            lines.append(f"  • {s}")
    if not commits and not suspects:
        lines.append("\nNo recent commits / CRs / ratchet — likely an "
                     "external degradation (provider, network, scheduler).")
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────


def run() -> dict[str, Any]:
    """One pass — cadence-guarded internally. Returns a small summary."""
    summary: dict[str, Any] = {
        "ran": False, "regressed": [], "alerted": [],
    }
    if not _enabled():
        return summary

    state = read_state_json(_STATE_FILE, {"last_run_at": 0.0, "flags": {}})
    now_ts = time.time()
    if now_ts - float(state.get("last_run_at", 0)) < _RUN_CADENCE_S:
        return summary
    state["last_run_at"] = now_ts
    summary["ran"] = True

    rows = _read_audit_journal(lookback_days=_BASELINE_DAYS + 1)
    if not rows:
        write_state_json(_STATE_FILE, state)
        return summary

    now = datetime.now(timezone.utc)
    counts = _bin_counts(rows, now)
    threshold = 1.0 - _regression_pct()  # e.g. 0.70 means alert if recent < 70% of baseline

    flags: dict = state.setdefault("flags", {})
    suspects: list[str] = []
    commits: list[str] = []
    suspects_loaded = False

    for event_type, c in counts.items():
        ratio = _ratio_against_baseline(c["recent_24h"], c["baseline_total"])
        if ratio is None or ratio >= threshold:
            continue
        summary["regressed"].append({
            "event_type": event_type, "ratio": round(ratio, 2),
            "recent_24h": c["recent_24h"], "baseline_total": c["baseline_total"],
        })

        # Per-event de-dup.
        prev = flags.setdefault(event_type, {"last_alert_at": 0})
        if now_ts - float(prev.get("last_alert_at", 0)) < _DEDUP_WINDOW_S:
            continue

        # Lazily load suspects once per pass (git log + audit_log are not free).
        if not suspects_loaded:
            commits = _recent_git_commits()
            suspects = _recent_change_requests()
            suspects_loaded = True

        body = _format_alert(event_type, c["recent_24h"], c["baseline_total"],
                             ratio, suspects, commits)
        try:
            send_signal_alert(body, tag=f"silent_regression:{event_type}")
            prev["last_alert_at"] = now_ts
            prev["ratio"] = round(ratio, 2)
            summary["alerted"].append(event_type)
        except Exception:
            logger.debug("silent_regression: send failed", exc_info=True)

    write_state_json(_STATE_FILE, state)
    audit_event(
        "silent_regression_pass",
        regressed_count=len(summary["regressed"]),
        alerted_count=len(summary["alerted"]),
    )
    return summary
