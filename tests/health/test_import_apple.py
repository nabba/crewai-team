"""Tests for app.health.import_apple."""
from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from app.health import import_apple


_SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<HealthData>
  <Record type="HKQuantityTypeIdentifierHeartRate"
          sourceName="Apple Watch" sourceVersion="u-hr-1"
          unit="count/min"
          startDate="2026-05-10 08:30:00 +0300"
          endDate="2026-05-10 08:31:00 +0300"
          value="72"/>
  <Record type="HKQuantityTypeIdentifierHeartRate"
          sourceName="Apple Watch" sourceVersion="u-hr-2"
          unit="count/min"
          startDate="2026-05-10 09:00:00 +0300"
          endDate="2026-05-10 09:01:00 +0300"
          value="80"/>
  <Record type="HKQuantityTypeIdentifierStepCount"
          sourceName="iPhone" sourceVersion="u-s-1"
          unit="count"
          startDate="2026-05-10 08:00:00 +0300"
          endDate="2026-05-10 08:15:00 +0300"
          value="512"/>
  <Record type="HKQuantityTypeIdentifierActiveEnergyBurned"
          sourceName="Watch" sourceVersion="u-e-1"
          unit="kcal"
          startDate="2026-05-10 08:00:00 +0300"
          endDate="2026-05-10 08:30:00 +0300"
          value="42.5"/>
  <Record type="HKQuantityTypeIdentifierBodyMass"
          sourceName="Scale" sourceVersion="u-bm-1"
          unit="kg"
          startDate="2026-05-10 07:00:00 +0300"
          endDate="2026-05-10 07:00:00 +0300"
          value="74.2"/>
  <Record type="HKCategoryTypeIdentifierSleepAnalysis"
          sourceName="Watch" sourceVersion="u-sl-1"
          startDate="2026-05-09 23:00:00 +0300"
          endDate="2026-05-10 06:30:00 +0300"
          value="HKCategoryValueSleepAnalysisAsleepCore"/>
  <Workout workoutActivityType="HKWorkoutActivityTypeRunning"
           sourceName="Watch" sourceVersion="u-w-1"
           duration="32.5" durationUnit="min"
           totalDistance="6.1" totalDistanceUnit="km"
           totalEnergyBurned="320"
           startDate="2026-05-10 18:00:00 +0300"
           endDate="2026-05-10 18:32:30 +0300"/>
</HealthData>
"""


def _make_apple_zip(tmp_path: Path) -> Path:
    """Build a fake apple_health_export.zip with a minimal export.xml."""
    extract_root = tmp_path / "src"
    extract_root.mkdir()
    xml_dir = extract_root / "apple_health_export"
    xml_dir.mkdir()
    (xml_dir / "export.xml").write_text(_SAMPLE_XML, encoding="utf-8")
    zip_path = tmp_path / "apple_health_export.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(xml_dir / "export.xml", "apple_health_export/export.xml")
    return zip_path


def test_import_zip_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HEALTH_INGESTION_ENABLED", "true")
    zip_path = _make_apple_zip(tmp_path)
    base = tmp_path / "health"

    result = import_apple.import_apple_export(zip_path, base=base)
    assert result.status == "ok"
    assert result.total_written == 7
    assert result.records_written["heart_rate"] == 2
    assert result.records_written["steps"] == 1
    assert result.records_written["active_energy"] == 1
    assert result.records_written["body_mass"] == 1
    assert result.records_written["sleep"] == 1
    assert result.records_written["workouts"] == 1


def test_import_extracted_xml_directly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HEALTH_INGESTION_ENABLED", "true")
    xml_path = tmp_path / "export.xml"
    xml_path.write_text(_SAMPLE_XML, encoding="utf-8")
    base = tmp_path / "health"

    result = import_apple.import_apple_export(xml_path, base=base)
    assert result.status == "ok"
    assert result.total_written == 7


def test_import_disabled_short_circuits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HEALTH_INGESTION_ENABLED", "false")
    zip_path = _make_apple_zip(tmp_path)
    result = import_apple.import_apple_export(zip_path, base=tmp_path / "health")
    assert result.status == "skipped_disabled"
    assert result.total_written == 0


def test_import_idempotent_on_dedup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HEALTH_INGESTION_ENABLED", "true")
    zip_path = _make_apple_zip(tmp_path)
    base = tmp_path / "health"
    r1 = import_apple.import_apple_export(zip_path, base=base)
    r2 = import_apple.import_apple_export(zip_path, base=base)
    assert r1.total_written == 7
    assert r2.total_written == 0  # everything deduped


def test_import_missing_zip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HEALTH_INGESTION_ENABLED", "true")
    bogus = tmp_path / "no_such.zip"
    bogus.write_bytes(b"not a zip")
    result = import_apple.import_apple_export(bogus, base=tmp_path / "health")
    assert result.status == "failed_zip"
    assert "BadZipFile" in result.failure_reason


def test_import_missing_xml_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HEALTH_INGESTION_ENABLED", "true")
    result = import_apple.import_apple_export(
        tmp_path / "missing.xml", base=tmp_path / "health",
    )
    assert result.status == "failed_missing_xml"


def test_normalize_apple_dt() -> None:
    """Apple's '2026-05-10 08:30:00 +0300' should become ISO-8601."""
    iso = import_apple._normalize_apple_dt("2026-05-10 08:30:00 +0300")
    assert iso == "2026-05-10T08:30:00+03:00"


def test_normalize_apple_dt_empty() -> None:
    assert import_apple._normalize_apple_dt("") == ""


def test_zip_without_export_xml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HEALTH_INGESTION_ENABLED", "true")
    zip_path = tmp_path / "stuff.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("readme.txt", "not health")
    result = import_apple.import_apple_export(zip_path, base=tmp_path / "health")
    assert result.status == "failed_missing_xml"


def test_malformed_record_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A record with non-numeric value should be skipped, not crash."""
    monkeypatch.setenv("HEALTH_INGESTION_ENABLED", "true")
    bad_xml = """<?xml version="1.0" encoding="UTF-8"?>
<HealthData>
  <Record type="HKQuantityTypeIdentifierHeartRate"
          sourceName="X" sourceVersion="u1"
          startDate="2026-05-10 08:30:00 +0300"
          endDate="2026-05-10 08:31:00 +0300"
          value="not a number"/>
  <Record type="HKQuantityTypeIdentifierHeartRate"
          sourceName="X" sourceVersion="u2"
          startDate="2026-05-10 08:35:00 +0300"
          endDate="2026-05-10 08:36:00 +0300"
          value="80"/>
</HealthData>
"""
    xml_path = tmp_path / "export.xml"
    xml_path.write_text(bad_xml, encoding="utf-8")
    result = import_apple.import_apple_export(xml_path, base=tmp_path / "health")
    assert result.status == "ok"
    assert result.skipped_malformed == 1
    assert result.records_written.get("heart_rate", 0) == 1
