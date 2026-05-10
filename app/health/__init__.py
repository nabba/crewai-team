"""Personal health-data ingestion (§5.1 — decade-class roadmap).

This package ingests the user's own health data (Apple Health export
today; other formats can plug in later) into typed JSONL streams the
companion can read for daily-briefing summaries and anomaly nudges.

Design discipline
-----------------

  - **Privacy.** Health data NEVER leaves the system. No ChromaDB
    embedding, no external-API calls, no LLM inference over raw
    records. The composer in :mod:`app.life_companion.daily_briefing`
    receives summary statistics (mean / trend / threshold-cross), not
    individual records.

  - **Default OFF.** ``HEALTH_INGESTION_ENABLED`` defaults to ``false``.
    The user must explicitly opt in — health data is high-leverage and
    the system shouldn't enable it implicitly.

  - **Append-only JSONL.** Per-record-type at
    ``workspace/health/<type>.jsonl`` (heart_rate, sleep, steps,
    active_energy, body_mass, workouts). Mirrors the audit-log
    discipline: append on import, never delete, dedupe on
    ``(start_iso, source_uuid)`` tuple.

  - **Failure-isolated.** Importer exposes ``import_apple_export(path)``
    that returns a typed :class:`ImportResult`; never raises. A
    malformed XML node is skipped + logged, not fatal.

Public API
----------

  * :func:`app.health.import_apple_export` — parse one
    ``apple_health_export.zip`` (or extracted ``export.xml``) into
    typed records and append to the per-type JSONL files.
  * :func:`app.health.summarise_window` — read the last N days of
    each record type and return a :class:`HealthSummary` for the
    daily briefing.
  * :func:`app.health.detect_anomalies` — rolling-window outlier
    detection per record type; returns :class:`HealthAnomaly` list.
"""

from app.health.anomaly import HealthAnomaly, detect_anomalies
from app.health.import_apple import ImportResult, import_apple_export
from app.health.summary import HealthSummary, summarise_window
from app.health.types import (
    ActiveEnergyRecord,
    BodyMassRecord,
    HeartRateRecord,
    SleepRecord,
    StepsRecord,
    WorkoutRecord,
)

__all__ = [
    "ActiveEnergyRecord",
    "BodyMassRecord",
    "HealthAnomaly",
    "HealthSummary",
    "HeartRateRecord",
    "ImportResult",
    "SleepRecord",
    "StepsRecord",
    "WorkoutRecord",
    "detect_anomalies",
    "import_apple_export",
    "summarise_window",
]
