"""HOT-1 outcome reconciler — fills in the ``outcome`` field on
metacognitive-repair observations.

PROGRAM §51 Q16.1 — closes the open loop flagged in the Theme 8.1
review. Without a reconciler, the ``outcome`` field on every HOT-1
row stays ``None`` forever. That breaks
:func:`app.healing.hot1_consultation.consult` because the
``proceed_normally`` recommendation path (which requires
``n_applied >= 1``) is unreachable in production — the consultation
module only ever returns ``skip`` (chronic decline) or
``proceed_with_caveat`` (≥2 attempts, no outcome).

What this reconciler does
=========================

  1. Walks ``workspace/change_requests/audit.jsonl`` (the CR audit
     log) for terminal events (``applied``, ``rejected``,
     ``rolled_back``, ``timeout``) on CRs filed by
     ``requestor=error_diagnosis``.
  2. For each terminal CR, finds its ``origin_pattern_signature``
     and ``created_at``.
  3. Locates the most recent HOT-1 observation row with matching
     ``pattern_signature`` AND ``ts < created_at`` (the structured-
     diagnosis emission preceded the CR creation by milliseconds).
  4. Records ``(pattern_signature, hot1_row_ts) → outcome`` in
     ``workspace/healing/hot1_outcomes_overlay.json`` — a side
     index that overlays on top of the append-only HOT-1 log.

What this reconciler deliberately doesn't do
============================================

  * NEVER rewrites the original HOT-1 JSONL. Identity-preserving
    consciousness data is append-only by contract; outcome
    enrichment is overlay-only.
  * NEVER files any CR or alters the CR audit log.
  * Idempotent — re-running with the same data produces the same
    overlay (last-write-wins is the natural semantics for
    ``(pattern, ts)`` keys).

Cadence: daily probe; internal weekly cadence for full reconciliation.
Master switch: ``hot1_outcome_reconciler_enabled`` (default ON).
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


_OVERLAY_FILE = "hot1_outcomes_overlay.json"
_RECONCILER_STATE_FILE = "hot1_outcome_reconciler_state.json"
_CR_AUDIT_PATH_OVERRIDE_ENV = "HOT1_RECONCILER_CR_AUDIT_PATH"

# Terminal CR statuses we care about.
_TERMINAL_EVENTS = frozenset({
    "applied", "rejected", "rolled_back", "rolled-back",
    "timeout", "auto-approved",
})

# Map CR audit event → HOT-1 outcome string. The HOT-1 consultation
# reader uses these strings literally; see
# ``hot1_consultation.consult`` recommendation logic.
_EVENT_TO_OUTCOME = {
    "applied": "applied",
    "auto-approved": "applied",     # auto-approved CRs auto-apply
    "rejected": "rejected",
    "rolled_back": "rolled_back",
    "rolled-back": "rolled_back",
    "timeout": "rejected",          # timeout = de-facto rejected
}


def _enabled() -> bool:
    try:
        from app.runtime_settings import get_hot1_outcome_reconciler_enabled
        return get_hot1_outcome_reconciler_enabled()
    except Exception:
        return os.getenv(
            "HOT1_OUTCOME_RECONCILER_ENABLED", "true",
        ).lower() in ("true", "1", "yes", "on")


def _workspace() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT)
    except Exception:
        return Path("/app/workspace")


def _cr_audit_path() -> Path:
    override = os.environ.get(_CR_AUDIT_PATH_OVERRIDE_ENV)
    if override:
        return Path(override)
    return _workspace() / "change_requests" / "audit.jsonl"


def _hot1_observation_path() -> Path:
    return Path(
        os.environ.get("HOT1_OBSERVATION_LOG")
        or str(_workspace() / "subia" / "observations" / "metacognitive_repair.jsonl")
    )


def _overlay_path() -> Path:
    return _workspace() / "healing" / _OVERLAY_FILE


def _state_path() -> Path:
    return _workspace() / "healing" / _RECONCILER_STATE_FILE


def _parse_iso(s: Any) -> Optional[float]:
    if not isinstance(s, str):
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def _read_overlay() -> dict[str, str]:
    p = _overlay_path()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception:
        logger.debug("hot1_reconciler: overlay read failed", exc_info=True)
    return {}


def _write_overlay(overlay: dict[str, str]) -> None:
    p = _overlay_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(overlay, indent=2, sort_keys=True), encoding="utf-8",
        )
    except Exception:
        logger.debug("hot1_reconciler: overlay write failed", exc_info=True)


def _read_state() -> dict[str, Any]:
    p = _state_path()
    if not p.exists():
        return {"last_run_at": 0.0, "last_cr_audit_ts": 0.0}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"last_run_at": 0.0, "last_cr_audit_ts": 0.0}


def _write_state(state: dict[str, Any]) -> None:
    p = _state_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(state, indent=2, sort_keys=True), encoding="utf-8",
        )
    except Exception:
        logger.debug("hot1_reconciler: state write failed", exc_info=True)


def _load_terminal_cr_events() -> list[dict[str, Any]]:
    """Walk the CR audit and return one row per terminal event on a
    requestor=error_diagnosis CR. Each row has:
        {pattern_signature, created_at_ts, event, request_id}
    """
    p = _cr_audit_path()
    if not p.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                event = row.get("audit_event") or row.get("event")
                if event not in _TERMINAL_EVENTS:
                    continue
                requestor = row.get("requestor")
                if requestor != "error_diagnosis":
                    continue
                pattern_sig = (
                    row.get("origin_pattern_signature")
                    or row.get("pattern_signature")
                )
                if not pattern_sig:
                    continue
                ts = (
                    _parse_iso(row.get("decided_at"))
                    or _parse_iso(row.get("created_at"))
                    or _parse_iso(row.get("ts"))
                )
                if ts is None:
                    continue
                out.append({
                    "pattern_signature": pattern_sig,
                    "event_ts": ts,
                    "event": event,
                    "request_id": row.get("request_id") or row.get("id"),
                })
    except OSError:
        return []
    return out


def _load_hot1_observations() -> list[dict[str, Any]]:
    """Walk the HOT-1 log; return rows with ``pattern_signature`` +
    ``ts``. We do NOT load full bodies — only the join keys."""
    p = _hot1_observation_path()
    if not p.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                origin = row.get("originating_error") or {}
                pattern_sig = origin.get("pattern_signature")
                if not pattern_sig:
                    continue
                ts = _parse_iso(row.get("ts"))
                if ts is None:
                    continue
                out.append({
                    "pattern_signature": pattern_sig,
                    "ts": ts,
                    "ts_iso": row.get("ts"),
                })
    except OSError:
        return []
    return out


def _overlay_key(pattern_signature: str, ts_iso: str) -> str:
    """Stable join key for the overlay file. Mirrors the way the
    consultation reader will look up outcomes."""
    return f"{pattern_signature}::{ts_iso}"


def reconcile_once(*, now: Optional[float] = None) -> dict[str, Any]:
    """One reconciliation pass. Returns summary dict. Failure-isolated."""
    summary: dict[str, Any] = {
        "ran": False,
        "n_terminal_events": 0,
        "n_observations": 0,
        "n_overlay_entries_before": 0,
        "n_overlay_entries_after": 0,
        "n_new_outcomes": 0,
    }
    if not _enabled():
        summary["skipped"] = True
        return summary
    cur = float(now) if now is not None else time.time()
    summary["ran"] = True
    state = _read_state()
    state["last_run_at"] = cur

    overlay = _read_overlay()
    summary["n_overlay_entries_before"] = len(overlay)

    cr_events = _load_terminal_cr_events()
    obs = _load_hot1_observations()
    summary["n_terminal_events"] = len(cr_events)
    summary["n_observations"] = len(obs)
    if not cr_events or not obs:
        _write_state(state)
        _write_overlay(overlay)
        summary["n_overlay_entries_after"] = len(overlay)
        return summary

    # Group HOT-1 observations by pattern_signature, sorted ascending by ts.
    by_pattern: dict[str, list[dict[str, Any]]] = {}
    for row in obs:
        by_pattern.setdefault(row["pattern_signature"], []).append(row)
    for pattern in by_pattern:
        by_pattern[pattern].sort(key=lambda r: r["ts"])

    n_new = 0
    for evt in cr_events:
        pattern = evt["pattern_signature"]
        event_ts = evt["event_ts"]
        candidates = by_pattern.get(pattern, [])
        # Most recent HOT-1 row whose ts < event_ts.
        match: Optional[dict[str, Any]] = None
        for row in reversed(candidates):
            if row["ts"] < event_ts:
                match = row
                break
        if match is None:
            continue
        key = _overlay_key(pattern, match["ts_iso"])
        outcome = _EVENT_TO_OUTCOME.get(evt["event"])
        if outcome is None:
            continue
        if overlay.get(key) != outcome:
            overlay[key] = outcome
            n_new += 1

    summary["n_new_outcomes"] = n_new
    summary["n_overlay_entries_after"] = len(overlay)
    _write_overlay(overlay)
    _write_state(state)
    return summary


def lookup_outcome(*, pattern_signature: str, ts_iso: str) -> Optional[str]:
    """Public read API for the consultation module. Returns the
    overlay outcome for a given HOT-1 row, or None."""
    overlay = _read_overlay()
    return overlay.get(_overlay_key(pattern_signature, ts_iso))


def run(*, now: Optional[float] = None) -> dict[str, Any]:
    """Healing-monitor entry point with weekly internal cadence."""
    if not _enabled():
        return {"skipped": True}
    cur = float(now) if now is not None else time.time()
    state = _read_state()
    last_run = float(state.get("last_run_at", 0))
    if last_run > 0 and cur - last_run < 7 * 86400:
        return {"ran": False, "reason": "cadence"}
    return reconcile_once(now=cur)
