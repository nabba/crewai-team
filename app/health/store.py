"""JSONL store for typed health records.

One file per record kind at ``workspace/health/<kind>.jsonl``. Append-
only; dedupe on ``(start_iso, source_uuid)`` so re-importing the same
Apple Health export is idempotent.

All file I/O is failure-isolated — a write that fails is logged and
returns ``False``. The importer never raises into the caller.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


_DEFAULT_BASE = Path("/app/workspace/health")
_path_override: Path | None = None


def _enabled() -> bool:
    return os.getenv("HEALTH_INGESTION_ENABLED", "false").lower() in (
        "true", "1", "yes", "on",
    )


def _resolve_base() -> Path:
    if _path_override:
        return _path_override
    raw = os.getenv("HEALTH_BASE_DIR")
    if raw:
        return Path(raw)
    return _DEFAULT_BASE


def _path_for(kind: str, base: Path | str | None = None) -> Path:
    root = Path(base) if base else _resolve_base()
    return root / f"{kind}.jsonl"


def _existing_keys(kind: str, base: Path | str | None = None) -> set[tuple[str, str]]:
    """Return the (start_iso, source_uuid) tuples already present for
    this kind. Used by the importer to dedupe additive runs."""
    p = _path_for(kind, base=base)
    if not p.exists():
        return set()
    out: set[tuple[str, str]] = set()
    try:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (
                    str(raw.get("start_iso", "")),
                    str(raw.get("source_uuid", "")),
                )
                out.add(key)
    except OSError:
        return set()
    return out


def append_records(
    kind: str,
    records: Iterable[Any],
    *,
    base: Path | str | None = None,
) -> int:
    """Append records (each ``.to_dict()``-able) for ``kind``. Dedupes
    against existing entries by ``(start_iso, source_uuid)``. Returns
    the number of records actually written.

    Disabled short-circuit: if ``HEALTH_INGESTION_ENABLED`` is false,
    the call returns 0 without touching disk.
    """
    if not _enabled():
        return 0
    p = _path_for(kind, base=base)
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = _existing_keys(kind, base=base)
    written = 0
    try:
        with open(p, "a", encoding="utf-8") as f:
            for r in records:
                d = r.to_dict() if hasattr(r, "to_dict") else dict(r)
                key = (str(d.get("start_iso", "")), str(d.get("source_uuid", "")))
                if key in existing:
                    continue
                f.write(json.dumps(d, sort_keys=True) + "\n")
                existing.add(key)
                written += 1
    except OSError as exc:
        logger.warning("health.store: append to %s failed: %s", p, exc)
        return written
    return written


def list_records(
    kind: str,
    *,
    since_iso: str | None = None,
    until_iso: str | None = None,
    base: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Read all records for ``kind``, optionally bounded by ISO range.

    Records are returned in file order (which matches arrival order
    since the file is append-only). Failure-isolated — a missing or
    malformed file yields ``[]``.
    """
    p = _path_for(kind, base=base)
    if not p.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                start = str(raw.get("start_iso", ""))
                if since_iso and start < since_iso:
                    continue
                if until_iso and start >= until_iso:
                    continue
                out.append(raw)
    except OSError:
        return []
    return out


def list_window(
    kind: str,
    *,
    days: int,
    now: datetime | None = None,
    base: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Records from the last ``days`` (inclusive). Convenience over
    :func:`list_records`."""
    cur = now or datetime.now(timezone.utc)
    cutoff = (cur - timedelta(days=days)).isoformat()
    return list_records(kind, since_iso=cutoff, base=base)


def _reset_for_tests(path: Path | None = None) -> None:
    """Inject a base path for tests. Internal use only."""
    global _path_override
    _path_override = path
