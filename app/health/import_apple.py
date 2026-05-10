"""Apple Health export parser.

Apple Health exports as a ``.zip`` containing ``apple_health_export/
export.xml`` (plus optional ECG / route folders we skip). The XML is
flat: a list of ``<Record>`` elements (vital signs / activity counts)
plus ``<Workout>`` elements for sessions.

We use ``xml.etree.ElementTree.iterparse`` so the ~50–500 MB XML
parses in bounded memory, even for years-of-data exports.

Failure modes
-------------

  * Malformed zip → returns ``ImportResult`` with status="failed_zip".
  * Missing export.xml → status="failed_missing_xml".
  * One bad ``<Record>`` element → skipped + counted under
    ``skipped_malformed``; the import continues.
  * Disabled (``HEALTH_INGESTION_ENABLED=false``) → status="skipped_disabled",
    no parsing happens.

The function never raises — every error path returns a typed result.
"""
from __future__ import annotations

import logging
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator
from xml.etree import ElementTree as ET

from app.health import store
from app.health.types import (
    ActiveEnergyRecord,
    BodyMassRecord,
    HeartRateRecord,
    SleepRecord,
    StepsRecord,
    WorkoutRecord,
    record_type_for_apple_kind,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImportResult:
    """Structured outcome of one ``import_apple_export`` call."""

    status: str          # "ok" | "skipped_disabled" | "failed_zip" |
                         # "failed_missing_xml" | "failed_unexpected_error"
    records_written: dict[str, int] = field(default_factory=dict)
    records_seen: dict[str, int] = field(default_factory=dict)
    skipped_malformed: int = 0
    failure_reason: str = ""

    @property
    def total_written(self) -> int:
        return sum(self.records_written.values())


def _enabled() -> bool:
    return os.getenv("HEALTH_INGESTION_ENABLED", "false").lower() in (
        "true", "1", "yes", "on",
    )


# ── Apple datetime → ISO-8601 UTC ─────────────────────────────────────


def _normalize_apple_dt(value: str) -> str:
    """Apple writes timestamps as ``2026-05-09 17:30:00 +0300``. Convert
    to ISO-8601 UTC. Empty/unparseable → empty string (record skipped)."""
    if not value:
        return ""
    s = value.strip()
    # Replace the space between date and time with 'T'.
    if " " in s:
        parts = s.split(" ", 2)
        if len(parts) == 3:
            date_part, time_part, tz_part = parts
            # tz_part is e.g. "+0300" — turn into "+03:00".
            if len(tz_part) == 5 and tz_part[0] in ("+", "-"):
                tz_part = f"{tz_part[:3]}:{tz_part[3:]}"
            iso = f"{date_part}T{time_part}{tz_part}"
        elif len(parts) == 2:
            iso = f"{parts[0]}T{parts[1]}"
        else:
            iso = s.replace(" ", "T", 1)
    else:
        iso = s
    return iso


# ── XML iteration ─────────────────────────────────────────────────────


def _iter_records(xml_path: Path) -> Iterator[ET.Element]:
    """Yield each ``<Record>`` and ``<Workout>`` element while clearing
    parsed siblings to keep memory bounded."""
    context = ET.iterparse(str(xml_path), events=("end",))
    for event, elem in context:
        if elem.tag in ("Record", "Workout"):
            yield elem
            elem.clear()


# ── Per-kind parsers ──────────────────────────────────────────────────


def _parse_quantity_record(elem: ET.Element) -> dict[str, Any] | None:
    """Common fields for HKQuantityType records."""
    apple_kind = elem.attrib.get("type", "")
    record_kind = record_type_for_apple_kind(apple_kind)
    if record_kind is None:
        return None
    start = _normalize_apple_dt(elem.attrib.get("startDate", ""))
    end = _normalize_apple_dt(elem.attrib.get("endDate", ""))
    if not start or not end:
        return None
    return {
        "record_kind": record_kind,
        "start_iso": start,
        "end_iso": end,
        "value": elem.attrib.get("value", ""),
        "unit": elem.attrib.get("unit", ""),
        "source": elem.attrib.get("sourceName", ""),
        "source_uuid": elem.attrib.get("sourceVersion", "") or elem.attrib.get("device", "") or "",
    }


def _build_typed_record(parsed: dict[str, Any]) -> Any | None:
    """Promote a parsed dict to the right typed dataclass."""
    kind = parsed["record_kind"]
    try:
        if kind == "heart_rate":
            return HeartRateRecord(
                start_iso=parsed["start_iso"],
                end_iso=parsed["end_iso"],
                bpm=float(parsed["value"]),
                source=parsed["source"],
                source_uuid=parsed["source_uuid"],
            )
        if kind == "steps":
            return StepsRecord(
                start_iso=parsed["start_iso"],
                end_iso=parsed["end_iso"],
                count=int(float(parsed["value"])),
                source=parsed["source"],
                source_uuid=parsed["source_uuid"],
            )
        if kind == "active_energy":
            return ActiveEnergyRecord(
                start_iso=parsed["start_iso"],
                end_iso=parsed["end_iso"],
                kcal=float(parsed["value"]),
                source=parsed["source"],
                source_uuid=parsed["source_uuid"],
            )
        if kind == "body_mass":
            return BodyMassRecord(
                start_iso=parsed["start_iso"],
                kg=float(parsed["value"]),
                source=parsed["source"],
                source_uuid=parsed["source_uuid"],
            )
        if kind == "sleep":
            return SleepRecord(
                start_iso=parsed["start_iso"],
                end_iso=parsed["end_iso"],
                stage=_sleep_stage(parsed.get("value", "")),
                source=parsed["source"],
                source_uuid=parsed["source_uuid"],
            )
    except (ValueError, KeyError):
        return None
    return None


def _sleep_stage(value: str) -> str:
    """Map Apple sleep-analysis values to our stable stage names."""
    mapping = {
        "HKCategoryValueSleepAnalysisInBed": "in_bed",
        "HKCategoryValueSleepAnalysisAsleep": "asleep",
        "HKCategoryValueSleepAnalysisAsleepCore": "asleep_core",
        "HKCategoryValueSleepAnalysisAsleepDeep": "asleep_deep",
        "HKCategoryValueSleepAnalysisAsleepREM": "asleep_rem",
        "HKCategoryValueSleepAnalysisAwake": "awake",
        "HKCategoryValueSleepAnalysisAsleepUnspecified": "asleep",
    }
    return mapping.get(value, value or "unknown")


def _parse_workout(elem: ET.Element) -> WorkoutRecord | None:
    start = _normalize_apple_dt(elem.attrib.get("startDate", ""))
    end = _normalize_apple_dt(elem.attrib.get("endDate", ""))
    if not start or not end:
        return None
    activity = elem.attrib.get("workoutActivityType", "").replace(
        "HKWorkoutActivityType", "",
    ).lower()
    duration_unit = elem.attrib.get("durationUnit", "")
    duration_raw = elem.attrib.get("duration", "0")
    try:
        duration = float(duration_raw)
    except ValueError:
        duration = 0.0
    duration_s = duration * 60.0 if duration_unit == "min" else duration
    distance_raw = elem.attrib.get("totalDistance", "0")
    try:
        distance_km = float(distance_raw)
    except ValueError:
        distance_km = 0.0
    if elem.attrib.get("totalDistanceUnit", "") == "mi":
        distance_km *= 1.609344
    kcal_raw = elem.attrib.get("totalEnergyBurned", "0")
    try:
        kcal = float(kcal_raw)
    except ValueError:
        kcal = 0.0
    return WorkoutRecord(
        start_iso=start,
        end_iso=end,
        activity=activity or "unknown",
        duration_s=duration_s,
        distance_km=distance_km,
        kcal=kcal,
        source=elem.attrib.get("sourceName", ""),
        source_uuid=elem.attrib.get("sourceVersion", "")
            or elem.attrib.get("device", "") or "",
    )


# ── Public entry point ────────────────────────────────────────────────


def import_apple_export(
    path: Path | str,
    *,
    base: Path | str | None = None,
) -> ImportResult:
    """Parse one ``apple_health_export.zip`` (or extracted ``export.xml``)
    and append typed records to the per-kind JSONL store.

    Idempotent on dedup-key ``(start_iso, source_uuid)`` — re-importing
    the same export adds no duplicates. Never raises.
    """
    if not _enabled():
        return ImportResult(status="skipped_disabled")

    src = Path(path)
    xml_path: Path
    extracted_dir: Path | None = None

    try:
        if src.suffix == ".zip":
            try:
                with zipfile.ZipFile(src) as zf:
                    candidates = [
                        n for n in zf.namelist()
                        if n.endswith("export.xml")
                    ]
                    if not candidates:
                        return ImportResult(
                            status="failed_missing_xml",
                            failure_reason="no export.xml in zip",
                        )
                    extracted_dir = src.parent / f".health_extract_{src.stem}"
                    extracted_dir.mkdir(parents=True, exist_ok=True)
                    member = candidates[0]
                    zf.extract(member, path=extracted_dir)
                    xml_path = extracted_dir / member
            except zipfile.BadZipFile as exc:
                return ImportResult(
                    status="failed_zip",
                    failure_reason=f"BadZipFile: {exc}",
                )
        else:
            xml_path = src
            if not xml_path.exists():
                return ImportResult(
                    status="failed_missing_xml",
                    failure_reason=f"file not found: {src}",
                )
    except OSError as exc:
        return ImportResult(
            status="failed_unexpected_error",
            failure_reason=f"OSError: {exc}",
        )

    seen: dict[str, int] = {}
    by_kind: dict[str, list[Any]] = {}
    skipped = 0

    try:
        for elem in _iter_records(xml_path):
            if elem.tag == "Workout":
                rec = _parse_workout(elem)
                if rec is None:
                    skipped += 1
                    continue
                seen["workouts"] = seen.get("workouts", 0) + 1
                by_kind.setdefault("workouts", []).append(rec)
            elif elem.tag == "Record":
                parsed = _parse_quantity_record(elem)
                if parsed is None:
                    continue  # not a record-kind we ingest; not "malformed"
                rec = _build_typed_record(parsed)
                if rec is None:
                    skipped += 1
                    continue
                kind = parsed["record_kind"]
                seen[kind] = seen.get(kind, 0) + 1
                by_kind.setdefault(kind, []).append(rec)
    except ET.ParseError as exc:
        return ImportResult(
            status="failed_unexpected_error",
            failure_reason=f"XML parse error: {exc}",
            records_seen=seen,
            skipped_malformed=skipped,
        )
    except OSError as exc:
        return ImportResult(
            status="failed_unexpected_error",
            failure_reason=f"OSError reading XML: {exc}",
            records_seen=seen,
            skipped_malformed=skipped,
        )

    written: dict[str, int] = {}
    for kind, records in by_kind.items():
        n = store.append_records(kind, records, base=base)
        written[kind] = n

    # Best-effort cleanup of the temp extraction dir.
    if extracted_dir is not None:
        try:
            for child in extracted_dir.rglob("*"):
                if child.is_file():
                    child.unlink()
            for child in sorted(extracted_dir.rglob("*"), reverse=True):
                if child.is_dir():
                    child.rmdir()
            extracted_dir.rmdir()
        except OSError:
            pass

    return ImportResult(
        status="ok",
        records_written=written,
        records_seen=seen,
        skipped_malformed=skipped,
    )
