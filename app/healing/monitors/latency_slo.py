"""latency_slo — operator-facing latency SLO surface.

PROGRAM §51 — Q16 Theme 6 (decade-resilience hardening, QoS
measurement). "Fast service and quality answers" was a stated goal,
never measured against an SLO. This monitor computes p50 / p95 / p99
weekly from existing ``workspace/audit.log`` request_received +
response_sent pairs, persists a rolling history of weekly
percentiles, and alerts when any percentile regresses to ≥2× the
4-week trailing baseline.

What this monitor observes
==========================

  * **Latency p50 / p95 / p99** for Signal-routed requests, computed
    from the difference between matched ``request_received`` and
    ``response_sent`` audit rows. Matched by ``trace_id``.
  * **Sample-size sanity.** Below ``_MIN_SAMPLES_FOR_PERCENTILE``
    the percentile is reported but not regression-checked (small
    sample = high variance).
  * **Regression alerts.** When p95 OR p99 ≥ 2× trailing 4-week
    median for that percentile → Signal alert, 14-day dedup.

What this monitor deliberately doesn't do
=========================================

  * No per-request alerting. The arbiter handles individual
    failures; this is the slow-burn trend surface.
  * No latency MUTATION. No timeouts adjusted, no retries
    cancelled. Observation only.
  * No reading of message content (audit.log doesn't have it
    anyway).

Cadence: daily probe; internal weekly cadence for emission.
Master switch: ``latency_slo_monitor_enabled`` (default ON).
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


NAME = "latency_slo"
CADENCE_SECONDS = 24 * 3600
MASTER_SWITCH_KEY = "latency_slo_monitor_enabled"

_INTERNAL_CADENCE_S = 7 * 24 * 3600
_DEDUP_WINDOW_S = 14 * 86400
_STATE_FILE_NAME = "latency_slo_state.json"

_LOOKBACK_DAYS_FOR_SAMPLE = 7    # this week's percentiles
_BASELINE_WEEKS = 4              # trailing weeks for regression check
_MIN_SAMPLES_FOR_PERCENTILE = 30  # below this, skip regression alerting
_MAX_HISTORY_ENTRIES = 52        # keep 1 year of weekly rollups
_REGRESSION_RATIO_WARN = 2.0     # alert at 2× baseline median


def _enabled() -> bool:
    try:
        from app.runtime_settings import get_latency_slo_monitor_enabled
        return get_latency_slo_monitor_enabled()
    except Exception:
        return os.getenv(
            "LATENCY_SLO_MONITOR_ENABLED", "true",
        ).lower() in ("true", "1", "yes", "on")


def _workspace() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT)
    except Exception:
        return Path("/app/workspace")


def _state_path() -> Path:
    return _workspace() / "healing" / _STATE_FILE_NAME


def _audit_log_path() -> Path:
    return _workspace() / "audit.log"


def _read_state() -> dict[str, Any]:
    p = _state_path()
    if not p.exists():
        return {
            "last_run_at": 0.0,
            "history": [],          # rolling list of weekly rollups
            "last_alert_at": {},
        }
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {
            "last_run_at": 0.0,
            "history": [],
            "last_alert_at": {},
        }


def _write_state(state: dict[str, Any]) -> None:
    p = _state_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(state, indent=2, sort_keys=True), encoding="utf-8",
        )
    except Exception:
        logger.debug("latency_slo: state write failed", exc_info=True)


def _parse_iso(s: Any) -> Optional[float]:
    if not isinstance(s, str):
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    k = (n - 1) * pct
    f = int(k)
    c = min(f + 1, n - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def _median(values: list[float]) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    return _percentile(s, 0.5)


def _collect_latency_samples(*, now: float) -> list[float]:
    """Walk audit.log; pair request_received with response_sent by
    trace_id; return per-request latencies (seconds) within the
    lookback window. Tolerant of broken lines."""
    p = _audit_log_path()
    if not p.exists():
        return []
    cutoff = now - _LOOKBACK_DAYS_FOR_SAMPLE * 86400
    requests: dict[str, float] = {}
    latencies: list[float] = []
    try:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                event = row.get("event")
                if event not in ("request_received", "response_sent"):
                    continue
                ts = _parse_iso(row.get("ts"))
                if ts is None or ts < cutoff:
                    continue
                trace_id = row.get("trace_id")
                if not trace_id:
                    continue
                if event == "request_received":
                    requests[trace_id] = ts
                elif event == "response_sent":
                    started = requests.pop(trace_id, None)
                    if started is None:
                        continue
                    latency = ts - started
                    if 0 <= latency < 600:  # 10-min hard cap on outliers
                        latencies.append(latency)
    except OSError:
        return []
    return latencies


def _compute_rollup(
    latencies: list[float],
    *,
    now: float,
) -> dict[str, Any]:
    """Compute one weekly rollup. Returns even when n is small —
    consumers gate on ``n``."""
    latencies = sorted(latencies)
    n = len(latencies)
    if n == 0:
        return {
            "ts": now,
            "iso": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
            "n": 0,
            "p50_s": None,
            "p95_s": None,
            "p99_s": None,
        }
    return {
        "ts": now,
        "iso": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
        "n": n,
        "p50_s": round(_percentile(latencies, 0.5), 3),
        "p95_s": round(_percentile(latencies, 0.95), 3),
        "p99_s": round(_percentile(latencies, 0.99), 3),
    }


def _baseline_medians(history: list[dict[str, Any]]) -> dict[str, Optional[float]]:
    """Compute trailing-4-week median of each percentile from the
    history list (most recent last). Returns ``{"p50": x, "p95": y,
    "p99": z}`` or None per key when too few samples."""
    recent = history[-_BASELINE_WEEKS:]
    out: dict[str, Optional[float]] = {"p50_s": None, "p95_s": None, "p99_s": None}
    for key in ("p50_s", "p95_s", "p99_s"):
        values = [
            float(r[key]) for r in recent
            if isinstance(r.get(key), (int, float))
        ]
        out[key] = _median(values)
    return out


def _alert_if_due(
    state: dict[str, Any],
    *,
    key: str,
    title: str,
    body: str,
    now: float,
) -> bool:
    last_alerts = state.setdefault("last_alert_at", {})
    if not isinstance(last_alerts, dict):
        last_alerts = {}
        state["last_alert_at"] = last_alerts
    last = float(last_alerts.get(key, 0))
    if now - last < _DEDUP_WINDOW_S:
        return False
    last_alerts[key] = now
    try:
        from app.notify import notify
        notify(
            title=title,
            body=body,
            url="/cp/monitor",
            topic=f"latency_slo:{key}",
            critical=False,
            arbitrate=True,
        )
        return True
    except Exception:
        logger.debug("latency_slo: notify failed", exc_info=True)
        return False


def _check_regression(
    history: list[dict[str, Any]],
    current: dict[str, Any],
    state: dict[str, Any],
    *,
    now: float,
) -> list[dict[str, Any]]:
    """Compare current rollup to trailing-4-week baseline medians.
    Emit alerts for any percentile ≥ _REGRESSION_RATIO_WARN × baseline.
    Returns list of alert payloads."""
    if current.get("n", 0) < _MIN_SAMPLES_FOR_PERCENTILE:
        return []
    if len(history) < _BASELINE_WEEKS:
        return []
    baselines = _baseline_medians(history)
    alerts: list[dict[str, Any]] = []
    for key in ("p95_s", "p99_s"):  # alert only on tails — p50 too noisy
        cur_val = current.get(key)
        baseline = baselines.get(key)
        if not isinstance(cur_val, (int, float)) or not isinstance(baseline, (int, float)):
            continue
        if baseline <= 0:
            continue
        ratio = cur_val / baseline
        if ratio < _REGRESSION_RATIO_WARN:
            continue
        body = (
            f"⏱️ Latency regression on {key}: "
            f"{cur_val:.2f}s this week vs {baseline:.2f}s "
            f"trailing {_BASELINE_WEEKS}-week median "
            f"({ratio:.1f}× baseline).\n\n"
            f"  • n samples this week: {current.get('n')}\n"
            f"  • p50 this week: {current.get('p50_s')}s\n"
            f"  • baseline thresholds: {baselines}\n\n"
            f"Possible causes: a slow upstream provider, a new code "
            f"path with extra round-trips, an idle daemon stealing "
            f"CPU. Investigate via /cp/monitor + recent crew_dispatch "
            f"rows."
        )
        sent = _alert_if_due(
            state,
            key=f"regression_{key}",
            title=f"⏱️ {key} regression {ratio:.1f}× baseline",
            body=body,
            now=now,
        )
        alerts.append({
            "kind": f"regression_{key}",
            "current_s": round(cur_val, 3),
            "baseline_s": round(baseline, 3),
            "ratio": round(ratio, 2),
            "alert_sent": sent,
        })
    return alerts


def run(*, now: Optional[float] = None) -> dict[str, Any]:
    """One probe pass. Daily wake-up gates on weekly internal
    cadence. Returns a summary dict."""
    summary: dict[str, Any] = {
        "ran": False,
        "rollup": None,
        "alerts": [],
        "history_size": 0,
    }
    if not _enabled():
        summary["skipped"] = True
        return summary

    cur = float(now) if now is not None else time.time()
    state = _read_state()
    last = float(state.get("last_run_at", 0))
    if last > 0 and cur - last < _INTERNAL_CADENCE_S:
        return summary
    state["last_run_at"] = cur
    summary["ran"] = True

    latencies = _collect_latency_samples(now=cur)
    rollup = _compute_rollup(latencies, now=cur)
    summary["rollup"] = rollup

    history = state.setdefault("history", [])
    if not isinstance(history, list):
        history = []
    # Build alerts BEFORE appending the new rollup so the baseline
    # excludes the current week.
    alerts = _check_regression(history, rollup, state, now=cur)
    summary["alerts"] = alerts

    history.append(rollup)
    if len(history) > _MAX_HISTORY_ENTRIES:
        del history[: len(history) - _MAX_HISTORY_ENTRIES]
    state["history"] = history
    summary["history_size"] = len(history)
    _write_state(state)
    return summary


def history_snapshot() -> dict[str, Any]:
    """Read-only accessor for the rolling history. Used by the REST
    surface (Theme 6.3 sibling endpoint)."""
    state = _read_state()
    history = state.get("history", [])
    return {
        "history": list(history) if isinstance(history, list) else [],
        "baselines": _baseline_medians(history if isinstance(history, list) else []),
    }
