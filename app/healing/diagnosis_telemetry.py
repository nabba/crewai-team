"""Telemetry ledger for structured-diagnosis CRs (Q2 §39).

Three event kinds, persisted to
``workspace/healing/structured_diagnosis_telemetry.jsonl``:

  * ``filed`` — the structured diagnosis produced a fix above the
    confidence threshold and a CR was created.
  * ``declined`` — the LLM declined OR the confidence was below
    threshold OR a guard fired (file too large, rate limit,
    multi-site bug). No CR created.
  * ``resolution`` — a previously-filed CR resolved (operator
    approved/rejected, or it timed out, or it was rolled back).

The auto-tuner (``diagnosis_auto_tune``) reads ``filed`` events
joined with their ``resolution`` events to compute a rolling
approval rate. The React dashboard reads the same telemetry for
operator-facing visibility.

Append-only with a 5000-line cap (≈3 years of observed cadence)
via ``app.utils.jsonl_retention.append_with_cap``. No retention
sweep — the cap handles it inline.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────


_MAX_LINES = 5000
_DEFAULT_LOG_RELATIVE = "workspace/healing/structured_diagnosis_telemetry.jsonl"


def _log_path() -> Path:
    """Honours an env override for tests."""
    override = os.environ.get("STRUCTURED_DIAGNOSIS_TELEMETRY_LOG")
    if override:
        return Path(override)
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / _DEFAULT_LOG_RELATIVE


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append(payload: dict[str, Any]) -> None:
    """Best-effort append. Failure logs at debug; never raises into
    the caller (telemetry is observability, not load-bearing)."""
    try:
        from app.utils.jsonl_retention import append_with_cap
        append_with_cap(_log_path(), json.dumps(payload, sort_keys=True), _MAX_LINES)
    except Exception:
        logger.debug("diagnosis_telemetry: append failed", exc_info=True)


# ── Public API: emit ──────────────────────────────────────────────────


def record_filed(
    *,
    cr_id: str,
    pattern_signature: str,
    file_path: str,
    error_class: str,
    confidence: float,
    threshold: float,
    delta_added: int,
    delta_removed: int,
) -> None:
    _append({
        "ts": _now_iso(),
        "event_kind": "filed",
        "cr_id": cr_id,
        "originating_pattern_signature": pattern_signature,
        "file_path": file_path,
        "error_class": error_class,
        "llm_confidence": float(confidence),
        "active_threshold_at_decision": float(threshold),
        "delta_lines_added": int(delta_added),
        "delta_lines_removed": int(delta_removed),
        "decline_reason": None,
        "resolution": None,
    })


def record_declined(
    *,
    pattern_signature: str,
    file_path: str,
    error_class: str,
    confidence: float,
    threshold: float,
    decline_reason: str,
) -> None:
    _append({
        "ts": _now_iso(),
        "event_kind": "declined",
        "cr_id": None,
        "originating_pattern_signature": pattern_signature,
        "file_path": file_path,
        "error_class": error_class,
        "llm_confidence": float(confidence),
        "active_threshold_at_decision": float(threshold),
        "delta_lines_added": 0,
        "delta_lines_removed": 0,
        "decline_reason": decline_reason,
        "resolution": None,
    })


def record_resolution(
    *,
    cr_id: str,
    decided_by: str,
    approved: bool,
    decided_at: str | None = None,
    rejection_reason: str | None = None,
) -> None:
    _append({
        "ts": _now_iso(),
        "event_kind": "resolution",
        "cr_id": cr_id,
        "resolution": {
            "decided_by": decided_by,
            "approved": bool(approved),
            "decided_at": decided_at or _now_iso(),
            "rejection_reason": rejection_reason,
        },
    })


# ── Public API: query ─────────────────────────────────────────────────


def _read_all() -> list[dict]:
    path = _log_path()
    if not path.exists():
        return []
    out: list[dict] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return out


def attempts_for_pattern_in_window(
    pattern_signature: str, *, window_seconds: int,
) -> int:
    """Count filed + declined events for ``pattern_signature`` within
    the trailing ``window_seconds``. Used by the structured-diagnosis
    rate limiter."""
    if not pattern_signature:
        return 0
    cutoff = time.time() - window_seconds
    n = 0
    for row in _read_all():
        if row.get("event_kind") not in ("filed", "declined"):
            continue
        if row.get("originating_pattern_signature") != pattern_signature:
            continue
        ts = row.get("ts") or ""
        try:
            row_ts = datetime.fromisoformat(ts).timestamp()
        except (TypeError, ValueError):
            continue
        if row_ts >= cutoff:
            n += 1
    return n


def rolling_window_with_resolutions(*, window: int = 20) -> list[dict]:
    """Return the last ``window`` filed events with their resolution
    joined (when known). Pending entries get ``resolution=None``.

    Sorted oldest→newest. Used by the React dashboard + the auto-
    tuner's approval-rate calculation.
    """
    rows = _read_all()
    if not rows:
        return []

    resolutions_by_cr: dict[str, dict] = {}
    for row in rows:
        if row.get("event_kind") == "resolution":
            cr_id = row.get("cr_id")
            if cr_id:
                resolutions_by_cr[cr_id] = row.get("resolution") or {}

    filed = [row for row in rows if row.get("event_kind") == "filed"]
    filed.sort(key=lambda r: r.get("ts", ""))
    tail = filed[-window:]

    joined: list[dict] = []
    for row in tail:
        cr_id = row.get("cr_id") or ""
        joined.append({
            **row,
            "resolution": resolutions_by_cr.get(cr_id),
        })
    return joined


def approval_rate(*, window: int = 20) -> float | None:
    """Approval rate over the last ``window`` resolved CRs.

    Returns:
      * float in [0.0, 1.0] when at least ``window/2`` resolved
        events are present in the window
      * None when insufficient data (auto-tuner skips adjustments)
    """
    joined = rolling_window_with_resolutions(window=window)
    if not joined:
        return None
    resolved = [r for r in joined if r.get("resolution") is not None]
    if len(resolved) < max(1, window // 2):
        return None
    approved = sum(
        1 for r in resolved if (r["resolution"] or {}).get("approved") is True
    )
    return approved / len(resolved)


def summary(*, window: int = 30) -> dict:
    """Compact summary for the React dashboard."""
    joined = rolling_window_with_resolutions(window=window)
    n_filed = len(joined)
    n_resolved = sum(1 for r in joined if r.get("resolution") is not None)
    n_approved = sum(
        1 for r in joined
        if (r.get("resolution") or {}).get("approved") is True
    )
    n_rejected = sum(
        1 for r in joined
        if r.get("resolution") is not None
        and (r.get("resolution") or {}).get("approved") is False
    )
    return {
        "window": window,
        "filed": n_filed,
        "resolved": n_resolved,
        "approved": n_approved,
        "rejected": n_rejected,
        "pending": n_filed - n_resolved,
        "approval_rate": (n_approved / n_resolved) if n_resolved else None,
    }


def n_resolutions_since(iso_ts: str | None) -> int:
    """Count resolution events strictly after ``iso_ts``. Used by the
    auto-tuner's hysteresis check (need ≥5 new resolutions before the
    next adjustment)."""
    if not iso_ts:
        # No prior adjustment recorded; everything counts.
        iso_ts = ""
    n = 0
    for row in _read_all():
        if row.get("event_kind") != "resolution":
            continue
        ts = row.get("ts") or ""
        if ts > iso_ts:
            n += 1
    return n
