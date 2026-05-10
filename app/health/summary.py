"""Roll-up summaries over the JSONL stores.

The daily briefing reads the summary, not raw records. This keeps
high-frequency data (e.g. heart-rate every 3 minutes) out of the
LLM prompt while preserving the signal (mean / trend / threshold-cross).

All summaries are pure functions over the JSONL store — no external
calls, no LLM inference, no embedding. Health data stays local.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.health import store


@dataclass(frozen=True)
class HealthSummary:
    """One-day or N-day rollup the daily briefing can read."""

    window_days: int
    as_of_iso: str

    # Heart rate
    hr_mean_bpm: float | None = None
    hr_resting_p10_bpm: float | None = None  # 10th-percentile proxy for resting

    # Steps + active energy
    steps_total: int = 0
    steps_per_day_mean: float = 0.0
    active_kcal_total: float = 0.0
    active_kcal_per_day_mean: float = 0.0

    # Body mass
    body_mass_latest_kg: float | None = None
    body_mass_change_kg: float | None = None  # last - first in window

    # Sleep
    sleep_hours_per_night_mean: float | None = None
    sleep_nights_observed: int = 0

    # Workouts
    workouts_count: int = 0
    workouts_distance_km_total: float = 0.0

    # Coverage diagnostics
    record_counts: dict[str, int] = field(default_factory=dict)


def _percentile(values: list[float], p: float) -> float | None:
    """Linear-interpolation percentile for small arrays. Returns None
    for empty input. Not statistically rigorous — the 10th percentile
    of a day's heart rate readings is a workable resting-HR proxy
    without needing a real percentile algo."""
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    rank = max(0.0, min(1.0, p / 100.0)) * (len(s) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    frac = rank - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _sleep_seconds_per_night(records: list[dict[str, Any]]) -> dict[str, float]:
    """Group sleep records by ISO date of start; sum any 'asleep*' stage
    durations. Returns date-string → seconds."""
    by_date: dict[str, float] = {}
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
        date_key = t0.date().isoformat()
        by_date[date_key] = by_date.get(date_key, 0.0) + max(
            0.0, (t1 - t0).total_seconds(),
        )
    return by_date


def summarise_window(
    days: int = 7,
    *,
    now: datetime | None = None,
    base: Path | str | None = None,
) -> HealthSummary:
    """Build a :class:`HealthSummary` over the last ``days``.

    Returns an empty-stats summary when health ingestion is disabled
    or no data exists yet — the caller can guard on
    ``record_counts == {}`` to detect that case.
    """
    cur = now or datetime.now(timezone.utc)
    window = max(1, days)

    # Heart rate
    hr_records = store.list_window("heart_rate", days=window, now=cur, base=base)
    hr_values = []
    for r in hr_records:
        try:
            hr_values.append(float(r.get("bpm", 0.0)))
        except (ValueError, TypeError):
            continue
    hr_mean = (sum(hr_values) / len(hr_values)) if hr_values else None
    hr_resting = _percentile(hr_values, 10) if hr_values else None

    # Steps
    steps_records = store.list_window("steps", days=window, now=cur, base=base)
    steps_total = 0
    for r in steps_records:
        try:
            steps_total += int(float(r.get("count", 0)))
        except (ValueError, TypeError):
            continue
    steps_per_day = steps_total / window

    # Active energy
    ae_records = store.list_window("active_energy", days=window, now=cur, base=base)
    ae_total = 0.0
    for r in ae_records:
        try:
            ae_total += float(r.get("kcal", 0.0))
        except (ValueError, TypeError):
            continue
    ae_per_day = ae_total / window

    # Body mass
    bm_records = store.list_window("body_mass", days=window, now=cur, base=base)
    bm_records_sorted = sorted(
        bm_records, key=lambda r: str(r.get("start_iso", "")),
    )
    bm_latest = None
    bm_change = None
    if bm_records_sorted:
        try:
            bm_latest = float(bm_records_sorted[-1].get("kg", 0.0))
            bm_change = bm_latest - float(bm_records_sorted[0].get("kg", 0.0))
        except (ValueError, TypeError):
            pass

    # Sleep
    sleep_records = store.list_window("sleep", days=window, now=cur, base=base)
    night_seconds = _sleep_seconds_per_night(sleep_records)
    nights_observed = len(night_seconds)
    if night_seconds:
        sleep_hours_mean = (
            sum(night_seconds.values()) / 3600.0 / nights_observed
        )
    else:
        sleep_hours_mean = None

    # Workouts
    wo_records = store.list_window("workouts", days=window, now=cur, base=base)
    wo_count = len(wo_records)
    wo_distance = 0.0
    for r in wo_records:
        try:
            wo_distance += float(r.get("distance_km", 0.0))
        except (ValueError, TypeError):
            continue

    return HealthSummary(
        window_days=window,
        as_of_iso=cur.isoformat(),
        hr_mean_bpm=hr_mean,
        hr_resting_p10_bpm=hr_resting,
        steps_total=steps_total,
        steps_per_day_mean=steps_per_day,
        active_kcal_total=ae_total,
        active_kcal_per_day_mean=ae_per_day,
        body_mass_latest_kg=bm_latest,
        body_mass_change_kg=bm_change,
        sleep_hours_per_night_mean=sleep_hours_mean,
        sleep_nights_observed=nights_observed,
        workouts_count=wo_count,
        workouts_distance_km_total=wo_distance,
        record_counts={
            "heart_rate": len(hr_records),
            "steps": len(steps_records),
            "active_energy": len(ae_records),
            "body_mass": len(bm_records),
            "sleep": len(sleep_records),
            "workouts": wo_count,
        },
    )
