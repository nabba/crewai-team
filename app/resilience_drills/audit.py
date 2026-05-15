"""Drill audit log: append-only JSONL at
``workspace/resilience/drill_audit.jsonl``.

PROGRAM §44.1 — Q6.1 foundation.

Two responsibilities:

  1. **Append** every ``DrillResult`` so the operator (and the
     freshness monitor) can see what's been run.
  2. **Emit** continuity-ledger ``resilience_drill`` events on
     landmark outcomes (failures, first-ever runs of a drill,
     status transitions).

Per operator decision (#22 plan response): drill audit is INCLUDED
in DR export tarballs. The system's resilience history is part of
its identity — restoring from backup should preserve the operator's
view of "we ran these drills these dates."

Storage cap via existing ``append_with_archive_rotate`` from Q3:
live file holds recent entries, older entries archive to monthly
buckets. Decade-scale safe.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from app.resilience_drills.protocol import DrillResult, DrillStatus

logger = logging.getLogger(__name__)


_AUDIT_LOG_MAX_LINES = 5_000  # archive-rotates older entries


def _default_audit_path() -> Path:
    """Path to the live drill_audit.jsonl. Honors WORKSPACE_ROOT."""
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT) / "resilience" / "drill_audit.jsonl"
    except Exception:
        return Path("/app/workspace/resilience/drill_audit.jsonl")


def append_result(result: DrillResult) -> bool:
    """Append a DrillResult to the audit log. Returns True on success.

    Uses archive-rotation so the file is bounded. Failure-isolated."""
    path = _default_audit_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.debug("drill_audit: mkdir failed", exc_info=True)
        return False
    try:
        from app.utils.jsonl_retention import append_with_archive_rotate
        append_with_archive_rotate(
            path,
            json.dumps(result.to_dict(), sort_keys=True),
            max_lines=_AUDIT_LOG_MAX_LINES,
        )
        return True
    except Exception:
        # Fall back to a simple append so audit never silently drops.
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result.to_dict(), sort_keys=True) + "\n")
            return True
        except OSError:
            logger.debug("drill_audit: fallback append failed", exc_info=True)
            return False


def iter_results(*, since_iso: str | None = None) -> Iterator[dict[str, Any]]:
    """Yield result dicts from the live audit file, optionally filtered."""
    path = _default_audit_path()
    if not path.exists():
        return
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if since_iso and (row.get("started_at") or "") < since_iso:
                    continue
                yield row
    except OSError:
        return


def last_result_for(drill_name: str) -> dict[str, Any] | None:
    """Most recent result for a given drill (any status)."""
    latest: dict[str, Any] | None = None
    for row in iter_results():
        if row.get("drill_name") != drill_name:
            continue
        if latest is None or (row.get("started_at") or "") > (
            latest.get("started_at") or ""
        ):
            latest = row
    return latest


def last_successful_for(drill_name: str) -> dict[str, Any] | None:
    """Most recent PASS result for a given drill."""
    latest: dict[str, Any] | None = None
    for row in iter_results():
        if row.get("drill_name") != drill_name:
            continue
        if (row.get("status") or "") != DrillStatus.PASS.value:
            continue
        if latest is None or (row.get("started_at") or "") > (
            latest.get("started_at") or ""
        ):
            latest = row
    return latest


def days_since_last_success(drill_name: str, *, now: datetime | None = None) -> float | None:
    """Days since the last successful run, or None if never run.

    Used by the drill_staleness monitor to detect overdue drills."""
    last = last_successful_for(drill_name)
    if not last:
        return None
    try:
        ts = datetime.fromisoformat(
            (last.get("started_at") or "").replace("Z", "+00:00"),
        )
    except (ValueError, TypeError):
        return None
    now_dt = now or datetime.now(timezone.utc)
    return (now_dt - ts).total_seconds() / 86400.0


# ── Per-drill in-flight lock (Q6.4 P1#3) ────────────────────────────────


# Stale lock TTL: if a lock file is older than this, treat the
# previous run as crashed and allow a new run. Drills typically
# complete in < 5 min; an hour is a generous upper bound.
_LOCK_STALE_TTL_SECONDS = 3600


def _default_lock_path(drill_name: str) -> Path:
    """Path for the per-drill in-flight lock file."""
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT) / "resilience" / f".{drill_name}.lock"
    except Exception:
        return Path(f"/app/workspace/resilience/.{drill_name}.lock")


def is_drill_in_flight(drill_name: str) -> bool:
    """Return True if a recent lock file exists for this drill.

    Q6.4 — defensive guard against concurrent drill execution. If the
    idle scheduler invokes a drill while a prior run is still
    executing (e.g. backup_restore can take 30+ seconds), the
    second invocation should bail out rather than start a duplicate.

    Stale locks (>1h old) are ignored — treats them as crash residue.
    """
    path = _default_lock_path(drill_name)
    if not path.exists():
        return False
    try:
        age_s = (
            datetime.now(timezone.utc).timestamp() - path.stat().st_mtime
        )
    except OSError:
        return False
    return age_s < _LOCK_STALE_TTL_SECONDS


def acquire_drill_lock(drill_name: str) -> bool:
    """Create the lock file. Returns True on acquisition, False if
    already in-flight (callers should skip in that case).

    Best-effort: filesystem errors fall through to True so the drill
    runs anyway (better to risk duplicate than block on FS issue)."""
    if is_drill_in_flight(drill_name):
        return False
    path = _default_lock_path(drill_name)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({
                "drill_name": drill_name,
                "acquired_at": datetime.now(timezone.utc).isoformat(),
            }),
            encoding="utf-8",
        )
        return True
    except OSError:
        logger.debug("drill lock: acquire failed for %s", drill_name)
        return True  # fail-open — drill runs


def release_drill_lock(drill_name: str) -> None:
    """Remove the lock file. Failure-isolated — drill completion is
    audited via append_result regardless."""
    path = _default_lock_path(drill_name)
    try:
        if path.exists():
            path.unlink()
    except OSError:
        logger.debug("drill lock: release failed for %s", drill_name)


# ── Continuity ledger emission ───────────────────────────────────────────


def emit_landmark_for(
    result: DrillResult,
    *,
    is_first_run: bool = False,
    prior_status: str | None = None,
) -> bool:
    """Emit a continuity-ledger ``resilience_drill`` event for landmark
    outcomes:

    * ``FAIL`` or ``ERROR`` outcomes — operator needs to know
    * First-ever PASS for a drill — identity-shaping event
    * Recovery: ``prior_status ∈ {fail, error}`` AND current is PASS

    Routine PASS-then-PASS runs are NOT ledger-worthy (they go in the
    audit log; only the audit log). Returns True on emission, False
    when not emitted (not landmark, ledger disabled, etc.).

    Q6.4 (PROGRAM §44.4) — ``prior_status`` is now an EXPLICIT parameter
    passed by the drill module BEFORE it appends the new result. The
    original Q6.2 ship read ``prior_status`` from the audit log AFTER
    append, which meant ``last_result_for`` returned the just-appended
    row and recovery detection silently never fired. Caller order now:

        prior = last_successful_for(SPEC.name)
        prior_status_str = (last_any_result or {}).get("status")
        is_first_run = prior is None
        append_result(result)
        emit_landmark_for(result, is_first_run=is_first_run,
                          prior_status=prior_status_str)
    """
    if result.status in (DrillStatus.FAIL, DrillStatus.ERROR):
        return _emit_event(result, kind_suffix="failed")
    if is_first_run and result.status == DrillStatus.PASS:
        return _emit_event(result, kind_suffix="first_pass")
    # Q6.4 — use the explicit prior_status passed in by the caller;
    # do NOT consult the audit log here (caller already appended).
    if result.status == DrillStatus.PASS and prior_status:
        if prior_status in (DrillStatus.FAIL.value, DrillStatus.ERROR.value):
            return _emit_event(result, kind_suffix="recovered")
    return False


def _emit_event(result: DrillResult, *, kind_suffix: str) -> bool:
    """Internal — fire the continuity-ledger event."""
    try:
        from app.identity.continuity_ledger import record_event
        summary = (
            f"resilience drill {result.drill_name} {kind_suffix} "
            f"(duration {result.duration_s:.1f}s)"
        )
        detail = {
            "drill_name": result.drill_name,
            "status": getattr(result.status, "value", str(result.status)),
            "duration_s": float(result.duration_s),
            "dry_run": bool(result.dry_run),
            "landmark_kind": kind_suffix,
            # Don't leak full detail blob — opaque counts only.
            # Include error count for failed/error rows so operator
            # sees "this drill failed with N errors" at the ledger
            # level.
            "n_errors": len(result.errors or []),
        }
        ok = record_event(
            kind="resilience_drill",
            actor=f"drill:{result.drill_name}",
            summary=summary[:200],
            detail=detail,
        )
        return bool(ok)
    except Exception:
        logger.debug(
            "drill_audit: continuity-ledger emit failed for %s",
            result.drill_name, exc_info=True,
        )
        return False
