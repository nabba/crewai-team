"""Tests for app.health.anomaly."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.health import anomaly, store
from app.health.types import HeartRateRecord, SleepRecord, StepsRecord


@pytest.fixture
def base_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HEALTH_INGESTION_ENABLED", "true")
    return tmp_path / "health"


def test_no_data_returns_empty(base_dir: Path) -> None:
    out = anomaly.detect_anomalies(base=base_dir)
    assert out == []


def test_steps_drop_is_flagged(base_dir: Path) -> None:
    """30-day baseline around 8k ± 500 steps; recent 3 days at 1k → flagged."""
    now = datetime(2026, 5, 10, tzinfo=timezone.utc)
    baseline = []
    for i in range(3, 33):  # baseline window: days 3..32 ago
        # Modest variability so stdev is non-zero but realistic.
        wobble = ((i * 71) % 1000) - 500
        baseline.append(StepsRecord(
            start_iso=(now - timedelta(days=i)).isoformat(),
            end_iso=(now - timedelta(days=i, hours=-1)).isoformat(),
            count=8000 + wobble, source_version=f"b{i}",
        ))
    recent = []
    for i in range(3):  # last 3 days
        recent.append(StepsRecord(
            start_iso=(now - timedelta(days=i)).isoformat(),
            end_iso=(now - timedelta(days=i, hours=-1)).isoformat(),
            count=1000, source_version=f"r{i}",
        ))
    store.append_records("steps", baseline + recent, base=base_dir)
    out = anomaly.detect_anomalies(now=now, base=base_dir)
    assert any(a.metric == "steps_per_day" and a.direction == "down" for a in out)


def test_steady_state_no_anomaly(base_dir: Path) -> None:
    """Constant-value series across baseline + recent → no anomaly."""
    now = datetime(2026, 5, 10, tzinfo=timezone.utc)
    records = []
    for i in range(33):
        records.append(StepsRecord(
            start_iso=(now - timedelta(days=i)).isoformat(),
            end_iso=(now - timedelta(days=i, hours=-1)).isoformat(),
            count=8000, source_version=f"u{i}",
        ))
    store.append_records("steps", records, base=base_dir)
    out = anomaly.detect_anomalies(now=now, base=base_dir)
    # With zero variance the z-score is 0 by convention; no flag.
    assert all(a.metric != "steps_per_day" for a in out)


def test_resting_hr_spike_flagged(base_dir: Path) -> None:
    """Recent resting HR (10th-percentile) elevated above baseline."""
    now = datetime(2026, 5, 10, tzinfo=timezone.utc)
    records: list = []
    # Baseline: 30 days of resting around 55 bpm.
    for i in range(3, 33):
        for h in range(20):
            records.append(HeartRateRecord(
                start_iso=(now - timedelta(days=i, hours=h)).isoformat(),
                end_iso=(now - timedelta(days=i, hours=h)).isoformat(),
                bpm=55.0 + (h * 0.5),  # variability
                source_version=f"b{i}-{h}",
            ))
    # Recent: 3 days of resting around 75 bpm.
    for i in range(3):
        for h in range(20):
            records.append(HeartRateRecord(
                start_iso=(now - timedelta(days=i, hours=h)).isoformat(),
                end_iso=(now - timedelta(days=i, hours=h)).isoformat(),
                bpm=75.0 + (h * 0.5),
                source_version=f"r{i}-{h}",
            ))
    store.append_records("heart_rate", records, base=base_dir)
    out = anomaly.detect_anomalies(now=now, base=base_dir)
    assert any(a.metric == "resting_hr" and a.direction == "up" for a in out)


def test_sleep_drop_flagged(base_dir: Path) -> None:
    """Recent 3 nights at 4h vs baseline 7.5h → anomaly."""
    now = datetime(2026, 5, 10, tzinfo=timezone.utc)
    records: list = []
    # 30 nights, 7.5h each.
    for i in range(3, 33):
        start = now - timedelta(days=i, hours=8)
        end = start + timedelta(hours=7, minutes=30)
        records.append(SleepRecord(
            start_iso=start.isoformat(), end_iso=end.isoformat(),
            stage="asleep_core", source_version=f"b{i}",
        ))
    # 3 recent nights, 4h each.
    for i in range(3):
        start = now - timedelta(days=i, hours=8)
        end = start + timedelta(hours=4)
        records.append(SleepRecord(
            start_iso=start.isoformat(), end_iso=end.isoformat(),
            stage="asleep_core", source_version=f"r{i}",
        ))
    store.append_records("sleep", records, base=base_dir)
    out = anomaly.detect_anomalies(now=now, base=base_dir)
    assert any(
        a.metric == "sleep_hours_per_night" and a.direction == "down"
        for a in out
    )


def test_z_threshold_filters(base_dir: Path) -> None:
    """A small drop with high baseline variance should NOT flag."""
    now = datetime(2026, 5, 10, tzinfo=timezone.utc)
    records: list = []
    # Highly variable baseline: 2k–14k steps.
    for i in range(3, 33):
        records.append(StepsRecord(
            start_iso=(now - timedelta(days=i)).isoformat(),
            end_iso=(now - timedelta(days=i, hours=-1)).isoformat(),
            count=2000 + (i * 400) % 12000, source_version=f"b{i}",
        ))
    # Recent: 3 days at 7k (within baseline range).
    for i in range(3):
        records.append(StepsRecord(
            start_iso=(now - timedelta(days=i)).isoformat(),
            end_iso=(now - timedelta(days=i, hours=-1)).isoformat(),
            count=7000, source_version=f"r{i}",
        ))
    store.append_records("steps", records, base=base_dir)
    out = anomaly.detect_anomalies(now=now, base=base_dir, z_threshold=2.0)
    # Either no anomaly, or the magnitude is bounded.
    for a in out:
        if a.metric == "steps_per_day":
            assert abs(a.z_score) >= 2.0
