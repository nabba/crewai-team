"""Tests that the daily briefing actually reads health summary data."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.health import store
from app.health.types import HeartRateRecord, SleepRecord, StepsRecord
from app.life_companion import daily_briefing


@pytest.fixture
def health_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Populate a small health store and pin it via HEALTH_BASE_DIR."""
    monkeypatch.setenv("HEALTH_INGESTION_ENABLED", "true")
    base = tmp_path / "health"
    monkeypatch.setenv("HEALTH_BASE_DIR", str(base))
    now = datetime(2026, 5, 10, tzinfo=timezone.utc)
    store.append_records("steps", [
        StepsRecord(
            start_iso=(now - timedelta(days=i)).isoformat(),
            end_iso=(now - timedelta(days=i, hours=-1)).isoformat(),
            count=8000, source_version=f"v{i}",
        ) for i in range(7)
    ], base=base)
    store.append_records("heart_rate", [
        HeartRateRecord(
            start_iso=(now - timedelta(hours=i)).isoformat(),
            end_iso=(now - timedelta(hours=i)).isoformat(),
            bpm=60.0 + i, source_version=f"hr{i}",
        ) for i in range(20)
    ], base=base)
    store.append_records("sleep", [
        SleepRecord(
            start_iso=(now - timedelta(days=1, hours=8)).isoformat(),
            end_iso=(now - timedelta(days=1)).isoformat(),
            stage="asleep_core", source_version="s1",
        ),
    ], base=base)
    return base


def test_gather_health_summary_returns_bullets(health_data: Path) -> None:
    """When data exists, the gatherer returns formatted bullet lines."""
    lines = daily_briefing._gather_health_summary()
    assert lines, "expected at least one line of health summary"
    assert any("steps/day" in line for line in lines)
    assert any("sleep" in line.lower() for line in lines)


def test_gather_health_summary_empty_without_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No store, no records → empty list (briefing reads as before §5.1)."""
    monkeypatch.setenv("HEALTH_INGESTION_ENABLED", "true")
    monkeypatch.setenv("HEALTH_BASE_DIR", str(tmp_path / "empty"))
    lines = daily_briefing._gather_health_summary()
    assert lines == []


def test_gather_health_summary_when_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disabled → no error, just an empty list."""
    monkeypatch.setenv("HEALTH_INGESTION_ENABLED", "false")
    monkeypatch.setenv("HEALTH_BASE_DIR", str(tmp_path / "disabled"))
    lines = daily_briefing._gather_health_summary()
    assert lines == []


def test_compose_morning_includes_health_section(health_data: Path) -> None:
    """Morning briefing actually includes the ❤️ Health section."""
    text = daily_briefing._compose_morning()
    assert "❤️" in text
    assert "Health (7d)" in text
    assert "steps/day" in text


def test_compose_evening_includes_health_section(health_data: Path) -> None:
    text = daily_briefing._compose_evening()
    assert "❤️" in text
    assert "Health (7d)" in text


def test_compose_weekly_includes_health_section(health_data: Path) -> None:
    text = daily_briefing._compose_weekly()
    assert "❤️" in text
    assert "Health (7d)" in text


def test_briefing_unchanged_without_health_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sanity: briefing without health data does NOT show the Health
    section (no empty header)."""
    monkeypatch.setenv("HEALTH_BASE_DIR", str(tmp_path / "empty"))
    text = daily_briefing._compose_morning()
    assert "Health (7d)" not in text
