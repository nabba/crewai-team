"""Rolling-window anomaly detection over health records.

The detector compares a *recent* window (default last 3 days) against
a *baseline* window (default 30 days, excluding the recent window).
Anything that deviates by ≥ ``z_threshold`` (default 2.0) flags an
anomaly the daily briefing can surface.

Conservative discipline:

  - Pure descriptive stats (mean / stdev). No model, no LLM, no
    external call.
  - Returns observational ``HealthAnomaly`` records — never modifies
    health data, never auto-routes anywhere. The briefing decides
    whether to surface them; the operator decides what to do.
  - Every anomaly carries the metric, the baseline mean, the recent
    mean, the z-score, and a one-line description. The user sees the
    raw number, never just "something's off."
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from app.health import store


@dataclass(frozen=True)
class HealthAnomaly:
    """One observation worth surfacing."""

    metric: str          # "resting_hr" | "steps_per_day" | "sleep_hours_per_night"
    direction: str       # "up" | "down"
    baseline_mean: float
    recent_mean: float
    z_score: float
    description: str     # one-line human summary


def _stdev(values: list[float], mean: float) -> float:
    if len(values) < 2:
        return 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


def _z_score(recent_mean: float, baseline: list[float]) -> tuple[float, float, float]:
    """Return (z_score, baseline_mean, baseline_std)."""
    if not baseline:
        return (0.0, 0.0, 0.0)
    mean = sum(baseline) / len(baseline)
    sd = _stdev(baseline, mean)
    if sd == 0.0:
        return (0.0, mean, 0.0)
    return ((recent_mean - mean) / sd, mean, sd)


def _aggregate_per_day(
    records: list[dict],
    *,
    value_key: str,
    aggregator: str = "sum",
) -> dict[str, float]:
    """Group by ISO date of start_iso; aggregate values."""
    by_date: dict[str, list[float]] = {}
    for r in records:
        start = str(r.get("start_iso", ""))
        if not start:
            continue
        try:
            t0 = datetime.fromisoformat(start.replace("Z", "+00:00"))
        except ValueError:
            continue
        date_key = t0.date().isoformat()
        try:
            v = float(r.get(value_key, 0.0))
        except (ValueError, TypeError):
            continue
        by_date.setdefault(date_key, []).append(v)
    out: dict[str, float] = {}
    for day, values in by_date.items():
        if not values:
            continue
        if aggregator == "sum":
            out[day] = sum(values)
        elif aggregator == "mean":
            out[day] = sum(values) / len(values)
        elif aggregator == "p10":
            s = sorted(values)
            out[day] = s[len(s) // 10] if len(s) >= 10 else s[0]
    return out


def _split_recent_vs_baseline(
    daily: dict[str, float],
    *,
    recent_days: int,
    baseline_days: int,
    now: datetime,
) -> tuple[list[float], list[float]]:
    """Slice the daily series into (recent, baseline) lists. Recent is
    the last ``recent_days``; baseline is the ``baseline_days`` *before*
    the recent window."""
    cur_date = now.date()
    recent_dates = {
        (cur_date - timedelta(days=i)).isoformat()
        for i in range(recent_days)
    }
    baseline_dates = {
        (cur_date - timedelta(days=i)).isoformat()
        for i in range(recent_days, recent_days + baseline_days)
    }
    recent_vals = [v for d, v in daily.items() if d in recent_dates]
    baseline_vals = [v for d, v in daily.items() if d in baseline_dates]
    return recent_vals, baseline_vals


def _flag_if_anomalous(
    metric: str,
    daily: dict[str, float],
    *,
    recent_days: int,
    baseline_days: int,
    z_threshold: float,
    now: datetime,
    describe_up: Callable[[float, float, float], str],
    describe_down: Callable[[float, float, float], str],
) -> HealthAnomaly | None:
    """Compute z-score on daily series; return an anomaly when |z| ≥ threshold."""
    recent, baseline = _split_recent_vs_baseline(
        daily, recent_days=recent_days, baseline_days=baseline_days, now=now,
    )
    if not (recent and baseline):
        return None
    recent_mean = sum(recent) / len(recent)
    z, base_mean, _ = _z_score(recent_mean, baseline)
    if abs(z) < z_threshold:
        return None
    direction = "up" if z > 0 else "down"
    describe = describe_up if z > 0 else describe_down
    return HealthAnomaly(
        metric=metric,
        direction=direction,
        baseline_mean=base_mean,
        recent_mean=recent_mean,
        z_score=z,
        description=describe(recent_mean, base_mean, z),
    )


def _sleep_hours_per_night(records: list[dict]) -> dict[str, float]:
    """Group asleep* stage durations by start-date.

    The sleep is attributed to its **start** date (so a session that
    begins 23:30 May 9 and ends 06:30 May 10 is recorded as May 9). This
    is a deliberate simplification — Apple Health emits multiple stage
    fragments per session, and a session-merge layer would be the right
    long-term fix. For descriptive trend detection (the only consumer
    today) the start-date attribution is consistent enough.
    """
    per_night: dict[str, float] = {}
    for r in records:
        stage = str(r.get("stage", ""))
        if not stage.startswith("asleep"):
            continue
        start = str(r.get("start_iso", ""))
        end = str(r.get("end_iso", ""))
        if not start or not end:
            continue
        try:
            t0 = datetime.fromisoformat(start.replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(end.replace("Z", "+00:00"))
        except ValueError:
            continue
        day = t0.date().isoformat()
        per_night[day] = per_night.get(day, 0.0) + max(
            0.0, (t1 - t0).total_seconds() / 3600.0,
        )
    return per_night


def detect_anomalies(
    *,
    recent_days: int = 3,
    baseline_days: int = 30,
    z_threshold: float = 2.0,
    now: datetime | None = None,
    base: Path | str | None = None,
) -> list[HealthAnomaly]:
    """Walk the per-kind JSONL stores and return any anomalies past
    ``z_threshold``. Returns ``[]`` when health ingestion is disabled
    or no data exists yet."""
    cur = now or datetime.now(timezone.utc)
    out: list[HealthAnomaly] = []
    window = recent_days + baseline_days

    def _check(
        metric: str,
        kind: str,
        build_daily: Callable[[list[dict]], dict[str, float]],
        describe_up: Callable[[float, float, float], str],
        describe_down: Callable[[float, float, float], str],
    ) -> None:
        records = store.list_window(kind, days=window, now=cur, base=base)
        if not records:
            return
        daily = build_daily(records)
        anomaly = _flag_if_anomalous(
            metric, daily,
            recent_days=recent_days, baseline_days=baseline_days,
            z_threshold=z_threshold, now=cur,
            describe_up=describe_up, describe_down=describe_down,
        )
        if anomaly:
            out.append(anomaly)

    _check(
        metric="resting_hr",
        kind="heart_rate",
        build_daily=lambda rs: _aggregate_per_day(rs, value_key="bpm", aggregator="p10"),
        describe_up=lambda r, b, z: f"resting heart rate up ({r:.0f} vs {b:.0f} bpm baseline; z={z:+.1f})",
        describe_down=lambda r, b, z: f"resting heart rate down ({r:.0f} vs {b:.0f} bpm baseline; z={z:+.1f})",
    )
    _check(
        metric="steps_per_day",
        kind="steps",
        build_daily=lambda rs: _aggregate_per_day(rs, value_key="count", aggregator="sum"),
        describe_up=lambda r, b, z: f"daily steps up ({r:,.0f} vs {b:,.0f} baseline; z={z:+.1f})",
        describe_down=lambda r, b, z: f"daily steps down ({r:,.0f} vs {b:,.0f} baseline; z={z:+.1f})",
    )
    _check(
        metric="sleep_hours_per_night",
        kind="sleep",
        build_daily=_sleep_hours_per_night,
        describe_up=lambda r, b, z: f"sleep duration up ({r:.1f}h vs {b:.1f}h baseline; z={z:+.1f})",
        describe_down=lambda r, b, z: f"sleep duration down ({r:.1f}h vs {b:.1f}h baseline; z={z:+.1f})",
    )

    return out
