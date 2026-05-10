"""Auto-tuning for the structured-diagnosis confidence threshold (Q2 §39).

The structured-diagnosis pipeline files CRs only when the LLM's
self-assessed confidence is ≥ ``current_threshold()``. A static
threshold is brittle — set too low we flood the operator with
rejected CRs; set too high we're back to the "0 resolved" state.

This module observes the rolling approval rate from the telemetry
ledger and adjusts the threshold within ``[floor, ceiling]`` from
``runtime_settings``. Algorithm:

  target band:  approval_rate ∈ [0.65, 0.85]
  step:         0.02 per adjustment
  cadence:      at most one adjustment per UTC day
  hysteresis:   ≥ 5 NEW resolutions since last adjustment

  if approval_rate < 0.65:  threshold += step  (be more conservative)
  elif approval_rate > 0.85: threshold -= step  (be more aggressive)
  else:                      no change

  clamp: max(floor, min(ceiling, threshold))

Signal-alerting: option B from the design pass. The auto-tuner
runs silently in the operator's background. We only alert when the
auto-tune wants to move beyond the operator-set band — i.e., it's
pinned at floor and approval is still below band, OR pinned at
ceiling and approval is still above band. That's actionable signal
("consider widening the band") instead of daily noise.

State persists at ``workspace/healing/structured_diagnosis_threshold.json``.
Read precedence in ``current_state()``:
  1. ``runtime_settings.structured_diagnosis_threshold_override`` —
     operator-pinned value (overrides everything)
  2. state file ``current``
  3. fallback default 0.70
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────


_TARGET_BAND_LOW = 0.65
_TARGET_BAND_HIGH = 0.85
_ADJUSTMENT_STEP = 0.02
_MIN_HOURS_BETWEEN_ADJUSTMENTS = 24
_MIN_RESOLUTIONS_FOR_HYSTERESIS = 5
_ROLLING_WINDOW = 20

_DEFAULT_FLOOR = 0.50
_DEFAULT_CEILING = 0.95
_DEFAULT_INITIAL = 0.70


def _state_path() -> Path:
    """Honours an env override for tests."""
    override = os.environ.get("STRUCTURED_DIAGNOSIS_THRESHOLD_STATE")
    if override:
        return Path(override)
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "workspace/healing/structured_diagnosis_threshold.json"


# ── State load / save ─────────────────────────────────────────────────


def _initial_state() -> dict:
    return {
        "current": _DEFAULT_INITIAL,
        "floor": _DEFAULT_FLOOR,
        "ceiling": _DEFAULT_CEILING,
        "last_adjusted_at": None,
        "last_adjusted_from": None,
        "last_adjusted_to": None,
        "last_approval_rate_at_adjustment": None,
        "auto_tune_enabled_seen": True,
    }


def _load_state() -> dict:
    path = _state_path()
    if not path.exists():
        return _initial_state()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return _initial_state()
        # Merge defaults so missing keys don't crash readers.
        merged = _initial_state()
        merged.update({k: v for k, v in data.items() if k in merged})
        return merged
    except (OSError, json.JSONDecodeError):
        return _initial_state()


def _save_state(state: dict) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        logger.debug("diagnosis_auto_tune: state write failed", exc_info=True)


# ── Runtime settings reads ────────────────────────────────────────────


def _runtime_floor_and_ceiling() -> tuple[float, float]:
    """Read floor + ceiling from runtime_settings, falling back to
    defaults on any read failure."""
    floor, ceiling = _DEFAULT_FLOOR, _DEFAULT_CEILING
    try:
        from app.runtime_settings import (
            get_structured_diagnosis_threshold_floor,
            get_structured_diagnosis_threshold_ceiling,
        )
        floor = float(get_structured_diagnosis_threshold_floor())
        ceiling = float(get_structured_diagnosis_threshold_ceiling())
    except Exception:
        pass
    # Sanity clamp — operator-set values are validated upstream but
    # belt-and-braces here.
    floor = max(0.0, min(0.99, floor))
    ceiling = max(floor + 0.01, min(1.0, ceiling))
    return floor, ceiling


def _auto_tune_enabled() -> bool:
    try:
        from app.runtime_settings import get_structured_diagnosis_auto_tune_enabled
        return bool(get_structured_diagnosis_auto_tune_enabled())
    except Exception:
        return True


def _operator_override() -> float | None:
    try:
        from app.runtime_settings import get_structured_diagnosis_threshold_override
        return get_structured_diagnosis_threshold_override()
    except Exception:
        return None


# ── Public API: read ──────────────────────────────────────────────────


def current_state() -> dict:
    """Return the full state dict + the resolved-effective threshold.

    The dashboard + structured_diagnosis read this. Fields:
      * ``current`` — the auto-tune state-file value
      * ``effective`` — the active threshold accounting for override
      * ``floor`` / ``ceiling`` — runtime_settings band
      * ``override`` — operator pin (None when unset)
      * ``auto_tune_enabled``
      * ``last_*`` — adjustment audit fields
    """
    state = _load_state()
    floor, ceiling = _runtime_floor_and_ceiling()
    state["floor"] = floor
    state["ceiling"] = ceiling
    override = _operator_override()
    state["override"] = override
    state["effective"] = float(override) if override is not None else state["current"]
    state["auto_tune_enabled"] = _auto_tune_enabled()
    return state


# ── Adjustment driver ─────────────────────────────────────────────────


def maybe_adjust_threshold() -> dict:
    """Single pass of the auto-tuner. Returns the resulting state.

    Wired into the healing/monitors daemon as a cadence-guarded
    monitor. Cadence guard inside this function (24h since last
    adjustment) means even if the monitor schedules us hourly, we
    only adjust once a day.
    """
    if not _auto_tune_enabled():
        return current_state()

    if _operator_override() is not None:
        # Override is active — auto-tune is a no-op until override clears.
        return current_state()

    state = _load_state()
    floor, ceiling = _runtime_floor_and_ceiling()

    # Cadence gate.
    if not _at_least_24h_since(state.get("last_adjusted_at")):
        return current_state()

    # Hysteresis gate: need new resolutions since last adjustment.
    try:
        from app.healing.diagnosis_telemetry import (
            approval_rate, n_resolutions_since,
        )
    except Exception:
        return current_state()
    if n_resolutions_since(state.get("last_adjusted_at")) < _MIN_RESOLUTIONS_FOR_HYSTERESIS:
        return current_state()

    rate = approval_rate(window=_ROLLING_WINDOW)
    if rate is None:
        # Insufficient data for a confident adjustment.
        return current_state()

    new_value, alert_kind = _compute_adjustment(
        current=state["current"], rate=rate,
        floor=floor, ceiling=ceiling,
    )

    if new_value == state["current"]:
        # In band — no change. Option B alerting fires only when we
        # WANTED to move but were pinned by the operator-set band.
        if alert_kind in ("pinned_at_floor", "pinned_at_ceiling"):
            _alert_pinned(
                kind=alert_kind, current=state["current"],
                rate=rate, floor=floor, ceiling=ceiling,
            )
        return current_state()

    # Real adjustment: persist + audit.
    prior = state["current"]
    state["current"] = new_value
    state["last_adjusted_at"] = datetime.now(timezone.utc).isoformat()
    state["last_adjusted_from"] = prior
    state["last_adjusted_to"] = new_value
    state["last_approval_rate_at_adjustment"] = rate
    _save_state(state)
    logger.info(
        "diagnosis_auto_tune: threshold %.2f → %.2f (approval_rate=%.2f, band=[%.2f, %.2f])",
        prior, new_value, rate, floor, ceiling,
    )
    return current_state()


def _compute_adjustment(
    *, current: float, rate: float, floor: float, ceiling: float,
) -> tuple[float, str | None]:
    """Decide the adjustment.

    Returns (new_threshold, alert_kind):
      * alert_kind is "pinned_at_floor" / "pinned_at_ceiling" when
        we wanted to move but were pinned by the band
      * otherwise None
    """
    if rate < _TARGET_BAND_LOW:
        # Want more conservative (raise threshold)
        proposed = current + _ADJUSTMENT_STEP
        if proposed > ceiling:
            return ceiling if current != ceiling else current, (
                "pinned_at_ceiling" if current == ceiling else None
            )
        return proposed, None
    if rate > _TARGET_BAND_HIGH:
        # Want more aggressive (lower threshold)
        proposed = current - _ADJUSTMENT_STEP
        if proposed < floor:
            return floor if current != floor else current, (
                "pinned_at_floor" if current == floor else None
            )
        return proposed, None
    # In band — no change.
    return current, None


def _at_least_24h_since(iso_ts: str | None) -> bool:
    if not iso_ts:
        return True
    try:
        last = datetime.fromisoformat(iso_ts)
    except (TypeError, ValueError):
        return True
    return (datetime.now(timezone.utc) - last) >= timedelta(
        hours=_MIN_HOURS_BETWEEN_ADJUSTMENTS,
    )


# ── Pinned-at-band Signal alert (option B) ────────────────────────────


def _alert_pinned(*, kind: str, current: float, rate: float,
                   floor: float, ceiling: float) -> None:
    """Fire ONLY when auto-tune wants to move beyond the operator-set
    band. Daily-cadence noise is intentionally suppressed.

    Dedup window: 7 days per kind. If the band stays bad for a week,
    we re-alert; otherwise the operator gets one alert per band-stuck
    spell.
    """
    if not _alert_dedup_ok(kind):
        return

    if kind == "pinned_at_ceiling":
        body = (
            f"⚠️  Structured-diagnosis threshold pinned at ceiling "
            f"{ceiling:.2f}. Auto-tuner wants higher — recent approval "
            f"rate is {rate:.2f}, below the [{_TARGET_BAND_LOW:.2f}, "
            f"{_TARGET_BAND_HIGH:.2f}] target band. Consider raising "
            f"the ceiling at /cp/settings, or accept the current "
            f"reject rate."
        )
    elif kind == "pinned_at_floor":
        body = (
            f"⚠️  Structured-diagnosis threshold pinned at floor "
            f"{floor:.2f}. Auto-tuner wants lower — recent approval "
            f"rate is {rate:.2f}, above the [{_TARGET_BAND_LOW:.2f}, "
            f"{_TARGET_BAND_HIGH:.2f}] target band, suggesting we "
            f"could be more aggressive. Consider lowering the floor "
            f"at /cp/settings."
        )
    else:
        return

    try:
        from app.signal_client import send_message
        from app.config import get_settings
        recipient = (get_settings().signal_owner_number or "").strip()
        if not recipient:
            return
        send_message(recipient, body)
    except Exception:
        logger.debug("diagnosis_auto_tune: alert send failed", exc_info=True)

    try:
        from app.workspace_publish import publish_to_workspace
        publish_to_workspace(
            source="structured-diagnosis-auto-tune",
            content=body[:200],
            salience=0.5,
            signal_type="disposition",
        )
    except Exception:
        logger.debug("diagnosis_auto_tune: GW publish failed", exc_info=True)


_ALERT_DEDUP_FILE_RELATIVE = "workspace/healing/auto_tune_pin_alerts.json"
_ALERT_DEDUP_WINDOW_S = 7 * 24 * 3600


def _alert_dedup_ok(kind: str) -> bool:
    """Returns True iff we should fire (i.e., last alert for this kind
    was > 7 days ago)."""
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / _ALERT_DEDUP_FILE_RELATIVE
    state: dict[str, float] = {}
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f) or {}
        except (OSError, json.JSONDecodeError):
            state = {}
    import time
    now = time.time()
    last = state.get(kind, 0.0)
    if (now - last) < _ALERT_DEDUP_WINDOW_S:
        return False
    state[kind] = now
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        pass
    return True
