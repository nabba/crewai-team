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

import contextlib
import logging
import os
import tempfile
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


class _BadZip(Exception):
    """Raised inside ``_resolve_xml`` when the input zip is malformed."""


class _MissingXml(Exception):
    """Raised inside ``_resolve_xml`` when no export.xml is reachable."""


@contextlib.contextmanager
def _resolve_xml(src: Path) -> Iterator[Path]:
    """Yield the path to ``export.xml``. For a ``.zip`` source the file
    is extracted to a system temp dir cleaned up on context exit; for a
    bare ``.xml`` source the path is yielded directly.
    """
    if src.suffix == ".zip":
        try:
            with zipfile.ZipFile(src) as zf, tempfile.TemporaryDirectory() as tmp:
                candidates = [
                    n for n in zf.namelist() if n.endswith("export.xml")
                ]
                if not candidates:
                    raise _MissingXml("no export.xml in zip")
                tmp_root = Path(tmp)
                zf.extract(candidates[0], path=tmp_root)
                yield tmp_root / candidates[0]
        except zipfile.BadZipFile as exc:
            raise _BadZip(f"BadZipFile: {exc}") from exc
    else:
        if not src.exists():
            raise _MissingXml(f"file not found: {src}")
        yield src


def _iter_records(xml_path: Path) -> Iterator[ET.Element]:
    """Yield each ``<Record>`` and ``<Workout>`` element while clearing
    parsed siblings AND removing them from the root so memory stays
    bounded across multi-million-record exports.

    Plain ``elem.clear()`` empties the element but leaves a (now empty)
    reference attached to the root — fine for short files, but for a
    decade-scale export the root's children list grows unbounded. We
    capture the root from the first ``start`` event and ``root.remove``
    each processed child after yielding.
    """
    context = ET.iterparse(str(xml_path), events=("start", "end"))
    iterator = iter(context)
    try:
        _, root = next(iterator)
    except StopIteration:
        return
    for event, elem in iterator:
        if event != "end":
            continue
        if elem.tag in ("Record", "Workout"):
            yield elem
            elem.clear()
            try:
                root.remove(elem)
            except ValueError:
                pass


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
        "source_version": elem.attrib.get("sourceVersion", "") or elem.attrib.get("device", "") or "",
    }


# Per-kind builders. Each takes the parsed dict + returns the typed
# record, or raises (ValueError|KeyError) — the caller catches.
_BUILDERS: dict[str, Any] = {
    "heart_rate": lambda p: HeartRateRecord(
        start_iso=p["start_iso"], end_iso=p["end_iso"],
        bpm=float(p["value"]),
        source=p["source"], source_version=p["source_version"],
    ),
    "steps": lambda p: StepsRecord(
        start_iso=p["start_iso"], end_iso=p["end_iso"],
        count=int(float(p["value"])),
        source=p["source"], source_version=p["source_version"],
    ),
    "active_energy": lambda p: ActiveEnergyRecord(
        start_iso=p["start_iso"], end_iso=p["end_iso"],
        kcal=float(p["value"]),
        source=p["source"], source_version=p["source_version"],
    ),
    "body_mass": lambda p: BodyMassRecord(
        start_iso=p["start_iso"], kg=float(p["value"]),
        source=p["source"], source_version=p["source_version"],
    ),
    "sleep": lambda p: SleepRecord(
        start_iso=p["start_iso"], end_iso=p["end_iso"],
        stage=_sleep_stage(p.get("value", "")),
        source=p["source"], source_version=p["source_version"],
    ),
}


def _build_typed_record(parsed: dict[str, Any]) -> Any | None:
    """Promote a parsed dict to the right typed dataclass."""
    builder = _BUILDERS.get(parsed["record_kind"])
    if builder is None:
        return None
    try:
        return builder(parsed)
    except (ValueError, KeyError):
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
        source_version=elem.attrib.get("sourceVersion", "")
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

    Idempotent on dedup-key ``(start_iso, source_version)`` — re-importing
    the same export adds no duplicates. Never raises.
    """
    if not _enabled():
        return ImportResult(status="skipped_disabled")

    src = Path(path)
    seen: dict[str, int] = {}
    by_kind: dict[str, list[Any]] = {}
    skipped = 0

    try:
        with _resolve_xml(src) as xml_path:
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
                        continue  # kind we don't ingest; not "malformed"
                    rec = _build_typed_record(parsed)
                    if rec is None:
                        skipped += 1
                        continue
                    kind = parsed["record_kind"]
                    seen[kind] = seen.get(kind, 0) + 1
                    by_kind.setdefault(kind, []).append(rec)
    except _BadZip as exc:
        return ImportResult(
            status="failed_zip",
            failure_reason=str(exc),
        )
    except _MissingXml as exc:
        return ImportResult(
            status="failed_missing_xml",
            failure_reason=str(exc),
        )
    except (ET.ParseError, OSError) as exc:
        return ImportResult(
            status="failed_unexpected_error",
            failure_reason=f"{type(exc).__name__}: {exc}",
            records_seen=seen,
            skipped_malformed=skipped,
        )

    written: dict[str, int] = {}
    for kind, records in by_kind.items():
        n = store.append_records(kind, records, base=base)
        written[kind] = n

    return ImportResult(
        status="ok",
        records_written=written,
        records_seen=seen,
        skipped_malformed=skipped,
    )
