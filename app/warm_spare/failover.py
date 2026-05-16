"""failover — canonical/standby state machine (Q17.1).

Five states: CANONICAL / STANDBY / CLAIMING / DEMOTED / UNKNOWN.
``claim_canonical("CLAIM CANONICAL")`` requires a typed phrase + a
heartbeat-silence threshold. Split-brain is too costly to let the
system resolve unilaterally — operator must observe partner is
genuinely down.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_CLAIM_PHRASE = "CLAIM CANONICAL"
_MIN_SILENCE_MINUTES = 15
_CONFIRM_AFTER_MINUTES = 5


class FailoverState(str, Enum):
    CANONICAL = "canonical"
    STANDBY = "standby"
    CLAIMING = "claiming"
    DEMOTED = "demoted"
    UNKNOWN = "unknown"


def _workspace_root() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT)
    except Exception:
        return Path(os.environ.get("WORKSPACE_ROOT", "/app/workspace"))


def _state_path() -> Path:
    return _workspace_root() / "warm_spare" / "failover_state.json"


def _heartbeat_path() -> Path:
    return _workspace_root() / "warm_spare" / "canonical_heartbeat.json"


def _read_state() -> dict[str, Any]:
    p = _state_path()
    if not p.exists():
        return {"state": FailoverState.UNKNOWN.value, "since": None}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"state": FailoverState.UNKNOWN.value, "since": None}


def _write_state(state: dict[str, Any]) -> None:
    p = _state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(p)
    except Exception:
        logger.debug("warm_spare.failover: state write failed", exc_info=True)


def _read_heartbeat() -> dict[str, Any] | None:
    p = _heartbeat_path()
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def record_heartbeat(*, now: datetime | None = None) -> None:
    cur = now or datetime.now(timezone.utc)
    p = _heartbeat_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        p.write_text(json.dumps({
            "ts": cur.isoformat(),
            "hostname": os.uname().nodename if hasattr(os, "uname") else "unknown",
            "pid": os.getpid(),
        }, sort_keys=True), encoding="utf-8")
    except OSError:
        logger.debug("warm_spare.failover: heartbeat write failed", exc_info=True)


def _silence_minutes_from_heartbeat(hb: dict[str, Any] | None) -> float | None:
    if not hb:
        return None
    ts = hb.get("ts")
    if not isinstance(ts, str):
        return None
    try:
        prev = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    now = datetime.now(timezone.utc)
    if prev.tzinfo is None:
        prev = prev.replace(tzinfo=timezone.utc)
    return (now - prev).total_seconds() / 60.0


def claim_canonical(typed_phrase: str, *, min_silence_minutes: int = _MIN_SILENCE_MINUTES) -> dict[str, Any]:
    if typed_phrase.strip() != _CLAIM_PHRASE:
        return {"accepted": False, "reason": f"typed phrase must be {_CLAIM_PHRASE!r}", "state": _read_state().get("state"), "silence_minutes": None}
    hb = _read_heartbeat()
    silence = _silence_minutes_from_heartbeat(hb)
    if silence is not None and silence < min_silence_minutes:
        return {"accepted": False, "reason": f"canonical heartbeat is only {silence:.1f}min old (threshold {min_silence_minutes}min)", "state": _read_state().get("state"), "silence_minutes": silence}
    state = {
        "state": FailoverState.CLAIMING.value,
        "since": datetime.now(timezone.utc).isoformat(),
        "claim_silence_minutes": silence,
    }
    _write_state(state)
    return {"accepted": True, "reason": "claim accepted; entering CLAIMING window", "state": FailoverState.CLAIMING.value, "silence_minutes": silence}


def current_state() -> dict[str, Any]:
    state = _read_state()
    if state.get("state") == FailoverState.CLAIMING.value:
        since = state.get("since")
        if isinstance(since, str):
            try:
                prev = datetime.fromisoformat(since.replace("Z", "+00:00"))
            except ValueError:
                prev = None
            if prev is not None:
                if prev.tzinfo is None:
                    prev = prev.replace(tzinfo=timezone.utc)
                elapsed_min = (datetime.now(timezone.utc) - prev).total_seconds() / 60.0
                if elapsed_min >= _CONFIRM_AFTER_MINUTES:
                    state = {
                        "state": FailoverState.CANONICAL.value,
                        "since": datetime.now(timezone.utc).isoformat(),
                        "auto_promoted_from": "claiming",
                    }
                    _write_state(state)
    return state


def demote() -> dict[str, Any]:
    state = {
        "state": FailoverState.DEMOTED.value,
        "since": datetime.now(timezone.utc).isoformat(),
    }
    _write_state(state)
    return state
