"""lock_contention — weekly per-resource p99 write-latency probe.

PROGRAM §49 — Q14.6 (year-2+ resilience §10.6). Reads the JSONL
written by :mod:`app.utils.lock_metrics` (slow writes ≥50ms), groups
by resource path, computes p99 wait time, alerts on any resource
whose p99 exceeds ``_P99_ALERT_MS``.

The data source is passive — :mod:`app.safe_io.safe_write` and
:mod:`app.safe_io.safe_append` record their own elapsed time when
the call takes >50ms. This monitor never writes anything except
its own state file; never blocks; never enforces any locking.

What "high p99" means here: an outlier write took N ms longer than
expected. Possible causes (in order of likelihood):

  1. Another process was writing the same file concurrently
     (kernel I/O queueing). The user's actual concern.
  2. Slow underlying disk (NFS, SSD wear).
  3. Filesystem fsync amplification.
  4. Process-level GIL contention with another tight loop.

Whichever it is, p99 > 500ms is a signal worth surfacing. The
operator can then investigate via:

  * ``less workspace/healing/lock_waits.jsonl`` — see which writes
    are slow.
  * ``lsof workspace/...path...`` — see who else has it open.
  * If the resource is a frequent JSONL append, consider migrating
    to the rolled_log primitive (which uses fcntl locking per
    segment, eliminating cross-writer interleaving).

Cadence: weekly. Master switch: ``lock_contention_monitor_enabled``
(default ON). Alert dedup: 14 days per resource.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


NAME = "lock_contention"
CADENCE_SECONDS = 7 * 24 * 3600
MASTER_SWITCH_KEY = "lock_contention_monitor_enabled"

_P99_ALERT_MS = 500
_MIN_SAMPLES_FOR_P99 = 10
_DEDUP_WINDOW_S = 14 * 86400
_STATE_FILE_NAME = "lock_contention_state.json"
_LOOKBACK_DAYS = 7


def _enabled() -> bool:
    try:
        from app.runtime_settings import get_lock_contention_monitor_enabled
        return get_lock_contention_monitor_enabled()
    except Exception:
        return os.getenv("LOCK_CONTENTION_MONITOR_ENABLED", "true").lower() in (
            "true", "1", "yes", "on",
        )


def _workspace() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT)
    except Exception:
        return Path("/app/workspace")


def _state_path() -> Path:
    return _workspace() / "healing" / _STATE_FILE_NAME


def _log_path() -> Path:
    return _workspace() / "healing" / "lock_waits.jsonl"


def _read_state() -> dict[str, Any]:
    p = _state_path()
    if not p.exists():
        return {"last_run_at": 0.0, "last_alert_at": {}}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"last_run_at": 0.0, "last_alert_at": {}}


def _write_state(state: dict[str, Any]) -> None:
    p = _state_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(state, indent=2, sort_keys=True), encoding="utf-8",
        )
    except Exception:
        logger.debug("lock_contention: state write failed", exc_info=True)


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Compute the percentile via linear interpolation. Assumes the
    input is already sorted ascending."""
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    k = (n - 1) * pct
    f = int(k)
    c = min(f + 1, n - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def _read_recent_rows(*, now_ts: float) -> list[dict[str, Any]]:
    """Read JSONL rows from the last ``_LOOKBACK_DAYS``. Tolerant of
    broken lines (skips silently)."""
    p = _log_path()
    if not p.exists():
        return []
    cutoff = now_ts - _LOOKBACK_DAYS * 86400
    out: list[dict[str, Any]] = []
    try:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                ts = row.get("ts")
                if not isinstance(ts, (int, float)):
                    continue
                if ts < cutoff:
                    continue
                if "resource" not in row or "elapsed_ms" not in row:
                    continue
                out.append(row)
    except OSError:
        return []
    return out


def run(*, now: Optional[float] = None) -> dict[str, Any]:
    """One weekly pass. Reads slow-write log, computes p99 per
    resource, alerts on outliers. Returns summary dict.

    Failure-isolated."""
    summary: dict[str, Any] = {
        "ran": False,
        "n_rows": 0,
        "n_resources": 0,
        "high_p99_resources": [],
        "alerts": 0,
    }
    if not _enabled():
        summary["skipped"] = True
        return summary

    cur = float(now) if now is not None else time.time()
    state = _read_state()
    last_run = float(state.get("last_run_at", 0))
    if last_run > 0 and cur - last_run < CADENCE_SECONDS:
        return summary
    state["last_run_at"] = cur
    summary["ran"] = True

    rows = _read_recent_rows(now_ts=cur)
    summary["n_rows"] = len(rows)
    if not rows:
        _write_state(state)
        return summary

    # Group by resource.
    by_resource: dict[str, list[float]] = {}
    for row in rows:
        r = row.get("resource", "unknown")
        e = float(row.get("elapsed_ms", 0))
        by_resource.setdefault(r, []).append(e)
    summary["n_resources"] = len(by_resource)

    # Compute p99 per resource and find outliers.
    high_p99: list[dict[str, Any]] = []
    for resource, elapsed_list in by_resource.items():
        if len(elapsed_list) < _MIN_SAMPLES_FOR_P99:
            continue
        elapsed_list.sort()
        p99 = _percentile(elapsed_list, 0.99)
        if p99 >= _P99_ALERT_MS:
            high_p99.append({
                "resource": resource,
                "p99_ms": round(p99, 1),
                "p50_ms": round(_percentile(elapsed_list, 0.50), 1),
                "n_samples": len(elapsed_list),
            })
    summary["high_p99_resources"] = high_p99

    if high_p99:
        # Top 5 worst (highest p99).
        high_p99.sort(key=lambda x: x["p99_ms"], reverse=True)
        top5 = high_p99[:5]
        last_alerts = state.setdefault("last_alert_at", {})
        if not isinstance(last_alerts, dict):
            last_alerts = {}
            state["last_alert_at"] = last_alerts
        last = float(last_alerts.get("any", 0))
        if cur - last >= _DEDUP_WINDOW_S:
            try:
                from app.notify import notify
                lines = [
                    f"🔒 Lock contention detected on "
                    f"{len(high_p99)} resource(s) in last "
                    f"{_LOOKBACK_DAYS}d (p99 > {_P99_ALERT_MS}ms):",
                    "",
                ]
                for h in top5:
                    lines.append(
                        f"  • {h['resource']:55s} "
                        f"p99={h['p99_ms']:.0f}ms "
                        f"p50={h['p50_ms']:.0f}ms "
                        f"(n={h['n_samples']})"
                    )
                lines.append("")
                lines.append(
                    "Consider migrating high-traffic JSONL writers "
                    "to app/audit/rolled_log.py (fcntl-locked "
                    "segments) if a single resource dominates."
                )
                notify(
                    title="🔒 Lock contention p99 spike",
                    body="\n".join(lines),
                    url="/cp/health",
                    topic="lock_contention",
                    critical=False,
                    arbitrate=True,
                )
                summary["alerts"] = 1
                last_alerts["any"] = cur
            except Exception:
                logger.debug("lock_contention: notify failed", exc_info=True)

    _write_state(state)
    return summary
