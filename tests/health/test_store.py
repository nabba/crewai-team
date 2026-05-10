"""Tests for app.health.store."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.health import store
from app.health.types import HeartRateRecord, StepsRecord


@pytest.fixture
def base_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HEALTH_INGESTION_ENABLED", "true")
    return tmp_path / "health"


def test_append_round_trip(base_dir: Path) -> None:
    records = [
        HeartRateRecord(
            start_iso="2026-05-10T08:30:00+00:00",
            end_iso="2026-05-10T08:31:00+00:00",
            bpm=72.0, source="Watch", source_uuid="u1",
        ),
    ]
    n = store.append_records("heart_rate", records, base=base_dir)
    assert n == 1
    out = store.list_records("heart_rate", base=base_dir)
    assert len(out) == 1
    assert out[0]["bpm"] == 72.0


def test_append_disabled_short_circuits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HEALTH_INGESTION_ENABLED", "false")
    rec = HeartRateRecord(
        start_iso="2026-05-10T08:30:00+00:00",
        end_iso="2026-05-10T08:31:00+00:00",
        bpm=72.0,
    )
    n = store.append_records("heart_rate", [rec], base=tmp_path)
    assert n == 0
    assert not (tmp_path / "heart_rate.jsonl").exists()


def test_append_dedupe_on_start_and_uuid(base_dir: Path) -> None:
    rec = HeartRateRecord(
        start_iso="2026-05-10T08:30:00+00:00",
        end_iso="2026-05-10T08:31:00+00:00",
        bpm=72.0, source_uuid="u1",
    )
    n1 = store.append_records("heart_rate", [rec], base=base_dir)
    n2 = store.append_records("heart_rate", [rec], base=base_dir)
    assert n1 == 1
    assert n2 == 0
    out = store.list_records("heart_rate", base=base_dir)
    assert len(out) == 1


def test_list_window_filters_by_age(base_dir: Path) -> None:
    old = HeartRateRecord(
        start_iso="2026-01-01T00:00:00+00:00",
        end_iso="2026-01-01T00:01:00+00:00",
        bpm=70.0, source_uuid="old",
    )
    fresh = HeartRateRecord(
        start_iso="2026-05-08T00:00:00+00:00",
        end_iso="2026-05-08T00:01:00+00:00",
        bpm=75.0, source_uuid="fresh",
    )
    store.append_records("heart_rate", [old, fresh], base=base_dir)
    out = store.list_window(
        "heart_rate", days=7,
        now=datetime(2026, 5, 10, tzinfo=timezone.utc),
        base=base_dir,
    )
    assert len(out) == 1
    assert out[0]["source_uuid"] == "fresh"


def test_list_records_missing_file(tmp_path: Path) -> None:
    out = store.list_records("does_not_exist", base=tmp_path)
    assert out == []


def test_list_records_skips_malformed(base_dir: Path) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "heart_rate.jsonl").write_text(
        '{"bpm": 70, "start_iso": "2026-05-10T00:00:00+00:00"}\n'
        'not json\n'
        '\n'
        '{"bpm": 75, "start_iso": "2026-05-10T01:00:00+00:00"}\n',
        encoding="utf-8",
    )
    out = store.list_records("heart_rate", base=base_dir)
    assert len(out) == 2


def test_append_multiple_kinds(base_dir: Path) -> None:
    hr = HeartRateRecord(
        start_iso="2026-05-10T08:30:00+00:00",
        end_iso="2026-05-10T08:31:00+00:00",
        bpm=72.0, source_uuid="hr1",
    )
    steps = StepsRecord(
        start_iso="2026-05-10T08:00:00+00:00",
        end_iso="2026-05-10T08:15:00+00:00",
        count=512, source_uuid="s1",
    )
    store.append_records("heart_rate", [hr], base=base_dir)
    store.append_records("steps", [steps], base=base_dir)
    assert (base_dir / "heart_rate.jsonl").exists()
    assert (base_dir / "steps.jsonl").exists()
    assert len(store.list_records("heart_rate", base=base_dir)) == 1
    assert len(store.list_records("steps", base=base_dir)) == 1
