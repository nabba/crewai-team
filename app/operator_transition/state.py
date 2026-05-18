"""state — operator-presence phase machine (Q17.4)."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_PHASE_FILE = "state.json"
_ABSENT_30D = 30
_ABSENT_90D = 90
_READ_MOSTLY_AUTOTRIGGER_DAYS = 180


class OperatorPhase(str, Enum):
    ACTIVE = "active"
    ABSENT_30D = "absent_30d"
    ABSENT_90D = "absent_90d"
    READ_MOSTLY = "read_mostly"
    TRANSITIONED = "transitioned"


def _workspace_root() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT)
    except Exception:
        return Path(os.environ.get("WORKSPACE_ROOT", "/app/workspace"))


def _state_path() -> Path:
    return _workspace_root() / "operator_transition" / _PHASE_FILE


def _audit_path() -> Path:
    for candidate in (_workspace_root() / "audit.log", _workspace_root() / "audit_log.jsonl"):
        if candidate.exists():
            return candidate
    return _workspace_root() / "audit.log"


def _read_state() -> dict[str, Any]:
    p = _state_path()
    if not p.exists():
        return {"phase": OperatorPhase.ACTIVE.value, "since": None, "last_activity_ts": None}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"phase": OperatorPhase.ACTIVE.value, "since": None, "last_activity_ts": None}


def _write_state(state: dict[str, Any]) -> None:
    p = _state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(p)
    except Exception:
        logger.debug("operator_transition.state: write failed", exc_info=True)


def _is_operator_row(row: dict[str, Any]) -> bool:
    kind = (row.get("kind") or row.get("event") or "").lower()
    if "request" not in kind and "received" not in kind:
        return False
    sender = row.get("sender_id") or row.get("from") or row.get("sender") or ""
    return isinstance(sender, str) and bool(sender)


def _read_last_operator_ts() -> str | None:
    """Return the most recent operator-event timestamp.

    Walks the live audit log first; if no operator row is found there
    (i.e. the operator has been silent for the entire window the live
    file covers), escalates to the rotated archive via
    ``jsonl_retention.read_archive``. Q3.1 §40.1.1c pattern — once
    retention rotation kicks in, a long-absent operator's last visit
    can be entirely in archived months.
    """
    audit = _audit_path()
    if not audit.exists():
        return None
    last: str | None = None

    def _scan(iterable) -> None:
        nonlocal last
        for line in iterable:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not _is_operator_row(row):
                continue
            ts = row.get("ts") or row.get("timestamp")
            if isinstance(ts, str) and (last is None or ts > last):
                last = ts

    try:
        with open(audit, "r", encoding="utf-8", errors="replace") as f:
            _scan(f)
    except OSError:
        return None
    if last is not None:
        return last

    # Live file holds no operator row at all — escalate to archive.
    try:
        from app.utils.jsonl_retention import read_archive
        _scan(read_archive(audit, include_live=False))
    except Exception:
        logger.debug("operator_transition: archive scan failed", exc_info=True)
    return last


def operator_active_within_days(days: int) -> bool:
    last = _read_last_operator_ts()
    if not last:
        return False
    try:
        prev = datetime.fromisoformat(last.replace("Z", "+00:00"))
    except ValueError:
        return False
    if prev.tzinfo is None:
        prev = prev.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - prev) < timedelta(days=days)


def _classify_phase(last_ts: str | None, *, manually_set: str | None) -> OperatorPhase:
    if manually_set:
        try:
            return OperatorPhase(manually_set)
        except ValueError:
            pass
    if not last_ts:
        return OperatorPhase.ACTIVE
    try:
        prev = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
    except ValueError:
        return OperatorPhase.ACTIVE
    if prev.tzinfo is None:
        prev = prev.replace(tzinfo=timezone.utc)
    delta = (datetime.now(timezone.utc) - prev).days
    if delta >= _READ_MOSTLY_AUTOTRIGGER_DAYS:
        return OperatorPhase.READ_MOSTLY
    if delta >= _ABSENT_90D:
        return OperatorPhase.ABSENT_90D
    if delta >= _ABSENT_30D:
        return OperatorPhase.ABSENT_30D
    return OperatorPhase.ACTIVE


def current_phase() -> dict[str, Any]:
    state = _read_state()
    last_ts = _read_last_operator_ts()
    manual = state.get("manually_set_phase")
    phase = _classify_phase(last_ts, manually_set=manual)
    prev_phase = state.get("phase")
    if phase.value != prev_phase:
        state["phase"] = phase.value
        state["since"] = datetime.now(timezone.utc).isoformat()
        state["last_activity_ts"] = last_ts
        _write_state(state)
        record_phase_transition(prev_phase, phase, last_ts=last_ts)
    else:
        state["last_activity_ts"] = last_ts
        _write_state(state)
    return {
        "phase": phase.value,
        "last_activity_ts": last_ts,
        "since": state.get("since"),
        "manually_set": bool(manual),
    }


def record_phase_transition(prev: str | None, new: OperatorPhase, *, last_ts: str | None) -> None:
    try:
        from app.notify import notify
        notify(
            title="🚪 Operator presence change",
            body=f"Operator presence: {prev or 'init'} → {new.value}. Last activity: {last_ts or 'unknown'}.",
            url="/cp/operator-transition",
            topic=f"operator_transition:{new.value}",
            critical=(new == OperatorPhase.READ_MOSTLY),
            arbitrate=False,
        )
    except Exception:
        logger.debug("operator_transition: notify failed", exc_info=True)
    try:
        from app.identity.continuity_ledger import record_event
        record_event(
            kind="q17_landmark",
            actor="operator_transition",
            summary=f"operator phase {prev or 'init'} → {new.value}",
            detail={"subsystem": "operator_transition", "prev": prev, "new": new.value, "last_activity_ts": last_ts},
        )
    except Exception:
        logger.debug("operator_transition: ledger emit failed", exc_info=True)


def manually_set_phase(phase: OperatorPhase) -> dict[str, Any]:
    state = _read_state()
    state["manually_set_phase"] = phase.value
    state["phase"] = phase.value
    state["since"] = datetime.now(timezone.utc).isoformat()
    _write_state(state)
    return state


def clear_manual_override() -> dict[str, Any]:
    state = _read_state()
    state.pop("manually_set_phase", None)
    _write_state(state)
    return state
