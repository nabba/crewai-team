"""Typed health records.

Apple Health stores everything as ``HKQuantityTypeIdentifier...`` /
``HKCategoryTypeIdentifier...`` strings. We map the high-value subset
to typed dataclasses with stable field names — the rest of the system
should never see raw HealthKit identifiers.

Only the high-leverage record types are modelled. The long tail
(menstrual flow, audiogram, environmental audio exposure, ...) is
deliberately out of scope — adding a record type is one dataclass
plus one entry in :func:`record_type_for_apple_kind`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class HeartRateRecord:
    """One heart-rate reading. Apple emits these every few minutes
    when the watch is worn — expect 200-400/day."""

    start_iso: str       # ISO-8601 UTC; "2026-05-10T08:30:00+00:00"
    end_iso: str
    bpm: float
    source: str = ""     # "Apple Watch", "iPhone", etc.
    source_uuid: str = ""

    record_kind: str = "heart_rate"

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_iso": self.start_iso,
            "end_iso": self.end_iso,
            "bpm": self.bpm,
            "source": self.source,
            "source_uuid": self.source_uuid,
            "record_kind": self.record_kind,
        }


@dataclass(frozen=True)
class StepsRecord:
    """Step count over an interval (typically 5–15 minute buckets)."""

    start_iso: str
    end_iso: str
    count: int
    source: str = ""
    source_uuid: str = ""

    record_kind: str = "steps"

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_iso": self.start_iso,
            "end_iso": self.end_iso,
            "count": self.count,
            "source": self.source,
            "source_uuid": self.source_uuid,
            "record_kind": self.record_kind,
        }


@dataclass(frozen=True)
class ActiveEnergyRecord:
    """Active energy burned (kcal) over an interval."""

    start_iso: str
    end_iso: str
    kcal: float
    source: str = ""
    source_uuid: str = ""

    record_kind: str = "active_energy"

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_iso": self.start_iso,
            "end_iso": self.end_iso,
            "kcal": self.kcal,
            "source": self.source,
            "source_uuid": self.source_uuid,
            "record_kind": self.record_kind,
        }


@dataclass(frozen=True)
class BodyMassRecord:
    """Body weight (kg). One per measurement event."""

    start_iso: str
    kg: float
    source: str = ""
    source_uuid: str = ""

    record_kind: str = "body_mass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_iso": self.start_iso,
            "kg": self.kg,
            "source": self.source,
            "source_uuid": self.source_uuid,
            "record_kind": self.record_kind,
        }


@dataclass(frozen=True)
class SleepRecord:
    """One sleep stage interval. Apple emits multiple per night
    (asleep_core / asleep_deep / asleep_rem / awake)."""

    start_iso: str
    end_iso: str
    stage: str           # "asleep" | "asleep_core" | "asleep_deep" | "asleep_rem" | "awake" | "in_bed"
    source: str = ""
    source_uuid: str = ""

    record_kind: str = "sleep"

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_iso": self.start_iso,
            "end_iso": self.end_iso,
            "stage": self.stage,
            "source": self.source,
            "source_uuid": self.source_uuid,
            "record_kind": self.record_kind,
        }


@dataclass(frozen=True)
class WorkoutRecord:
    """One workout session."""

    start_iso: str
    end_iso: str
    activity: str        # "running", "cycling", "walking", ...
    duration_s: float
    distance_km: float = 0.0
    kcal: float = 0.0
    source: str = ""
    source_uuid: str = ""

    record_kind: str = "workouts"

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_iso": self.start_iso,
            "end_iso": self.end_iso,
            "activity": self.activity,
            "duration_s": self.duration_s,
            "distance_km": self.distance_km,
            "kcal": self.kcal,
            "source": self.source,
            "source_uuid": self.source_uuid,
            "record_kind": self.record_kind,
        }


# All record-kind strings. The store layer uses these to pick the
# JSONL file path (one file per kind).
ALL_RECORD_KINDS: frozenset[str] = frozenset({
    "heart_rate",
    "steps",
    "active_energy",
    "body_mass",
    "sleep",
    "workouts",
})


# Apple HealthKit identifier → our record_kind. Adding a new kind is
# one new dataclass + one entry here + one branch in
# ``import_apple._parse_record``.
APPLE_KIND_TO_RECORD_KIND: dict[str, str] = {
    "HKQuantityTypeIdentifierHeartRate": "heart_rate",
    "HKQuantityTypeIdentifierStepCount": "steps",
    "HKQuantityTypeIdentifierActiveEnergyBurned": "active_energy",
    "HKQuantityTypeIdentifierBodyMass": "body_mass",
    "HKCategoryTypeIdentifierSleepAnalysis": "sleep",
    # Workouts are <Workout> elements, not <Record> — handled separately.
}


def record_type_for_apple_kind(apple_kind: str) -> str | None:
    """Map an Apple HealthKit identifier to our record-kind string,
    or ``None`` if we don't ingest this kind."""
    return APPLE_KIND_TO_RECORD_KIND.get(apple_kind)
