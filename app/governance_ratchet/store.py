"""JSON state for the governance ratchet.

Lives at ``workspace/governance/ratchet_state.json``. Atomic writes;
malformed file → empty state (the hardcoded floors in
``governance.py`` provide the safety net).
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

from app.governance_ratchet._state import ThresholdName, ThresholdState

logger = logging.getLogger(__name__)


_STATE_PATH = (
    Path(__file__).resolve().parents[2] / "workspace" / "governance"
    / "ratchet_state.json"
)
_lock = threading.Lock()


def _ensure_dir() -> None:
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_all() -> dict[str, ThresholdState]:
    """Read every threshold's current state. Returns empty dict on
    missing or malformed file (caller layers on the floor).
    """
    if not _STATE_PATH.exists():
        return {}
    try:
        data = json.loads(_STATE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        logger.warning(
            "governance_ratchet: ratchet_state.json malformed — treating as empty",
            exc_info=True,
        )
        return {}
    out: dict[str, ThresholdState] = {}
    for name, payload in (data or {}).items():
        try:
            out[name] = ThresholdState.from_dict(payload)
        except Exception:
            logger.debug(
                "governance_ratchet: cannot decode entry %s", name,
                exc_info=True,
            )
    return out


def save_all(state: dict[str, ThresholdState]) -> None:
    """Atomic write."""
    _ensure_dir()
    payload = {name: s.to_dict() for name, s in state.items()}
    with _lock:
        tmp = _STATE_PATH.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
            try:
                os.chmod(tmp, 0o600)
            except OSError:
                pass
            tmp.replace(_STATE_PATH)
        except Exception:
            logger.warning(
                "governance_ratchet: state-write failed", exc_info=True,
            )


def get(name: str) -> ThresholdState | None:
    return load_all().get(name)


def set_one(name: str, state: ThresholdState) -> None:
    """Update a single threshold and persist."""
    everything = load_all()
    everything[name] = state
    save_all(everything)


def known_thresholds() -> list[str]:
    """Names that the protocol allows operations on."""
    return [t.value for t in ThresholdName]
