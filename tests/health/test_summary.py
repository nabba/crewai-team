"""Tests for app.health.summary."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.health import store, summary
from app.health.types import (
    ActiveEnergyRecord,
    BodyMassRecord,
    HeartRateRecord,
    SleepRecord,
    StepsRecord,
    WorkoutRecord,
)


@pytest.fixture
def base_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HEALTH_INGESTION_ENABLED", "true")
    return tmp_path / "health"


def test_empty_summary(base_dir: Path) -> None:
    s = summary.summarise_window(days=7, base=base_dir)
    assert s.window_days == 7
    assert s.steps_total == 0
    assert s.hr_mean_bpm is None
    assert s.body_mass_latest_kg is None
    assert s.sleep_nights_observed == 0
    assert s.record_counts == {
        "heart_rate": 0, "steps": 0, "active_energy": 0,
        "body_mass": 0, "sleep": 0, "workouts": 0,
    }


def test_heart_rate_mean(base_dir: Path) -> None:
    now = datetime(2026, 5, 10, tzinfo=timezone.utc)
    records = [
        HeartRateRecord(
            start_iso=(now - timedelta(hours=i)).isoformat(),
            end_iso=(now - timedelta(hours=i)).isoformat(),
            bpm=70.0 + i, source_uuid=f"u{i}",
        )
        for i in range(5)
    ]
    store.append_records("heart_rate", records, base=base_dir)
    s = summary.summarise_window(days=7, now=now, base=base_dir)
    assert s.hr_mean_bpm is not None
    assert 70.0 <= s.hr_mean_bpm <= 75.0


def test_steps_total(base_dir: Path) -> None:
    now = datetime(2026, 5, 10, tzinfo=timezone.utc)
    records = [
        StepsRecord(
            start_iso=(now - timedelta(days=i)).isoformat(),
            end_iso=(now - timedelta(days=i, hours=1)).isoformat(),
            count=1000 * (i + 1), source_uuid=f"u{i}",
        )
        for i in range(3)
    ]
    store.append_records("steps", records, base=base_dir)
    s = summary.summarise_window(days=7, now=now, base=base_dir)
    assert s.steps_total == 1000 + 2000 + 3000
    assert s.steps_per_day_mean == 6000 / 7


def test_body_mass_change(base_dir: Path) -> None:
    now = datetime(2026, 5, 10, tzinfo=timezone.utc)
    records = [
        BodyMassRecord(
            start_iso=(now - timedelta(days=6)).isoformat(),
            kg=75.0, source_uuid="bm1",
        ),
        BodyMassRecord(
            start_iso=(now - timedelta(days=2)).isoformat(),
            kg=74.5, source_uuid="bm2",
        ),
    ]
    store.append_records("body_mass", records, base=base_dir)
    s = summary.summarise_window(days=7, now=now, base=base_dir)
    assert s.body_mass_latest_kg == 74.5
    assert s.body_mass_change_kg == pytest.approx(-0.5)


def test_sleep_hours_per_night(base_dir: Path) -> None:
    now = datetime(2026, 5, 10, tzinfo=timezone.utc)
    records = [
        # Two 4-hour asleep_core stages on the same night = 8h.
        SleepRecord(
            start_iso=(now - timedelta(days=1, hours=8)).isoformat(),
            end_iso=(now - timedelta(days=1, hours=4)).isoformat(),
            stage="asleep_core", source_uuid="s1",
        ),
        SleepRecord(
            start_iso=(now - timedelta(days=1, hours=4)).isoformat(),
            end_iso=(now - timedelta(days=1, hours=0)).isoformat(),
            stage="asleep_core", source_uuid="s2",
        ),
        # An "awake" stage shouldn't count.
        SleepRecord(
            start_iso=(now - timedelta(days=1, hours=2)).isoformat(),
            end_iso=(now - timedelta(days=1, hours=1)).isoformat(),
            stage="awake", source_uuid="s3",
        ),
    ]
    store.append_records("sleep", records, base=base_dir)
    s = summary.summarise_window(days=7, now=now, base=base_dir)
    assert s.sleep_nights_observed == 1
    assert s.sleep_hours_per_night_mean is not None
    assert 7.5 <= s.sleep_hours_per_night_mean <= 8.5


def test_workouts_count_and_distance(base_dir: Path) -> None:
    now = datetime(2026, 5, 10, tzinfo=timezone.utc)
    records = [
        WorkoutRecord(
            start_iso=(now - timedelta(days=2)).isoformat(),
            end_iso=(now - timedelta(days=2, hours=-1)).isoformat(),
            activity="running", duration_s=1800, distance_km=6.1,
            kcal=320.0, source_uuid="w1",
        ),
        WorkoutRecord(
            start_iso=(now - timedelta(days=4)).isoformat(),
            end_iso=(now - timedelta(days=4, hours=-1)).isoformat(),
            activity="cycling", duration_s=3600, distance_km=20.0,
            kcal=600.0, source_uuid="w2",
        ),
    ]
    store.append_records("workouts", records, base=base_dir)
    s = summary.summarise_window(days=7, now=now, base=base_dir)
    assert s.workouts_count == 2
    assert s.workouts_distance_km_total == pytest.approx(26.1)


def test_active_energy_per_day(base_dir: Path) -> None:
    now = datetime(2026, 5, 10, tzinfo=timezone.utc)
    records = [
        ActiveEnergyRecord(
            start_iso=(now - timedelta(days=i)).isoformat(),
            end_iso=(now - timedelta(days=i, hours=-1)).isoformat(),
            kcal=400.0, source_uuid=f"e{i}",
        )
        for i in range(7)
    ]
    store.append_records("active_energy", records, base=base_dir)
    s = summary.summarise_window(days=7, now=now, base=base_dir)
    assert s.active_kcal_total == pytest.approx(2800.0)
    assert s.active_kcal_per_day_mean == pytest.approx(400.0)
