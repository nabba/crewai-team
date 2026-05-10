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
from datetime import datetime, timezone
from pathlib import Path

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
        (cur_date.fromordinal(cur_date.toordinal() - i)).isoformat()
        for i in range(recent_days)
    }
    baseline_dates = {
        (cur_date.fromordinal(cur_date.toordinal() - i)).isoformat()
        for i in range(recent_days, recent_days + baseline_days)
    }
    recent_vals = [v for d, v in daily.items() if d in recent_dates]
    baseline_vals = [v for d, v in daily.items() if d in baseline_dates]
    return recent_vals, baseline_vals


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

    # Resting HR (10th percentile per day) — a low resting HR drop or
    # spike is physiologically meaningful.
    hr_records = store.list_window("heart_rate", days=window, now=cur, base=base)
    if hr_records:
        daily = _aggregate_per_day(hr_records, value_key="bpm", aggregator="p10")
        recent, baseline = _split_recent_vs_baseline(
            daily, recent_days=recent_days, baseline_days=baseline_days, now=cur,
        )
        if recent and baseline:
            recent_mean = sum(recent) / len(recent)
            z, base_mean, _ = _z_score(recent_mean, baseline)
            if abs(z) >= z_threshold:
                direction = "up" if z > 0 else "down"
                out.append(HealthAnomaly(
                    metric="resting_hr",
                    direction=direction,
                    baseline_mean=base_mean,
                    recent_mean=recent_mean,
                    z_score=z,
                    description=(
                        f"resting heart rate {direction} "
                        f"({recent_mean:.0f} vs {base_mean:.0f} bpm baseline; "
                        f"z={z:+.1f})"
                    ),
                ))

    # Steps per day
    steps_records = store.list_window("steps", days=window, now=cur, base=base)
    if steps_records:
        daily = _aggregate_per_day(steps_records, value_key="count", aggregator="sum")
        recent, baseline = _split_recent_vs_baseline(
            daily, recent_days=recent_days, baseline_days=baseline_days, now=cur,
        )
        if recent and baseline:
            recent_mean = sum(recent) / len(recent)
            z, base_mean, _ = _z_score(recent_mean, baseline)
            if abs(z) >= z_threshold:
                direction = "up" if z > 0 else "down"
                out.append(HealthAnomaly(
                    metric="steps_per_day",
                    direction=direction,
                    baseline_mean=base_mean,
                    recent_mean=recent_mean,
                    z_score=z,
                    description=(
                        f"daily steps {direction} "
                        f"({recent_mean:,.0f} vs {base_mean:,.0f} baseline; "
                        f"z={z:+.1f})"
                    ),
                ))

    # Sleep hours per night (aggregate of asleep* stages).
    sleep_records = store.list_window("sleep", days=window, now=cur, base=base)
    if sleep_records:
        per_night: dict[str, float] = {}
        for r in sleep_records:
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
        recent, baseline = _split_recent_vs_baseline(
            per_night, recent_days=recent_days, baseline_days=baseline_days, now=cur,
        )
        if recent and baseline:
            recent_mean = sum(recent) / len(recent)
            z, base_mean, _ = _z_score(recent_mean, baseline)
            if abs(z) >= z_threshold:
                direction = "up" if z > 0 else "down"
                out.append(HealthAnomaly(
                    metric="sleep_hours_per_night",
                    direction=direction,
                    baseline_mean=base_mean,
                    recent_mean=recent_mean,
                    z_score=z,
                    description=(
                        f"sleep duration {direction} "
                        f"({recent_mean:.1f}h vs {base_mean:.1f}h baseline; "
                        f"z={z:+.1f})"
                    ),
                ))

    return out
