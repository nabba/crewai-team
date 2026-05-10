"""Operator-controlled ratchet for governance thresholds.

The TIER_IMMUTABLE ``governance.py`` module defines the FLOOR for each
threshold (``SAFETY_MINIMUM_FLOOR = 0.95``, ``QUALITY_MINIMUM_FLOOR =
0.70``). The ratchet *raises* the effective threshold above the floor
as the system earns trust, and (with the operator's typed
confirmation) can *relax* it back down — but never below the floor.

Two operations:

    set_ratchet(name, new_value, source, reason)
        Raises the threshold. ``new_value > current`` required;
        ``new_value <= 1.0`` (sanity ceiling).

    relax_ratchet(name, new_value, source, reason)
        Lowers the threshold. ``new_value < current`` required;
        ``new_value >= floor`` enforced. ``reason`` is mandatory and
        recorded in the audit trail. UI must present a typed-phrase
        confirmation gate before calling this.

Effective value:

    effective_value(name)  →  max(FLOOR, ratcheted_current)

This is the function ``governance.py`` reads. Even if the JSON state
file is corrupted or maliciously edited to drop ``current`` below the
floor, the ``max`` clamp is type-level enforcement that the
post-bootstrap safety contract is preserved.

V1 is operator-only. Agents have NO API path here — the only
mutating entry is via the React ``/cp/settings`` page, which talks to
the ``/config/governance_ratchet/*`` endpoints (gateway-bearer-secret
gated) which call this module.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from app.governance_ratchet import audit, store
from app.governance_ratchet._state import (
    CeilingViolation,
    Direction,
    FloorViolation,
    MonotonicViolation,
    RatchetEntry,
    ThresholdName,
    ThresholdState,
    UnknownThresholdViolation,
)

logger = logging.getLogger(__name__)


# ── Floor lookup ─────────────────────────────────────────────────────────


def _floor(name: str) -> float:
    """Read the hardcoded floor from ``app/governance.py``. Imported lazily
    so a circular bootstrap during gateway startup doesn't trip up.
    """
    from app import governance as _g
    return float(_g.threshold_floor(name))


def _validate_name(name: str) -> str:
    try:
        return ThresholdName(name).value
    except ValueError:
        known = ", ".join(t.value for t in ThresholdName)
        raise UnknownThresholdViolation(
            f"unknown threshold {name!r}; known: {known}"
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Read API ─────────────────────────────────────────────────────────────


def effective_value(name: str) -> float:
    """The value the gates should USE: ``max(FLOOR, ratcheted_current)``.

    Called from ``governance.evaluate_promotion`` for both safety and
    quality minimums. Always returns a value ≥ floor — even with a
    missing or corrupted state file.
    """
    name = _validate_name(name)
    floor = _floor(name)
    state = store.get(name)
    if state is None:
        return floor
    return max(floor, float(state.current))


def get_state(name: str) -> ThresholdState | None:
    """Return the current ratchet state for one threshold (or None if
    nothing has been ratcheted for it yet — caller should treat as
    ``current = floor``).
    """
    name = _validate_name(name)
    return store.get(name)


def list_thresholds() -> list[dict]:
    """Snapshot of all known thresholds + their current effective values
    + history. Designed for the React UI.
    """
    out = []
    all_state = store.load_all()
    for name in store.known_thresholds():
        floor = _floor(name)
        s = all_state.get(name)
        current = float(s.current) if s else floor
        history = [e.to_dict() for e in (s.history if s else [])]
        out.append({
            "name": name,
            "floor": floor,
            "current": current,
            "effective": max(floor, current),
            "history": history,
        })
    return out


# ── Write API (operator-only) ────────────────────────────────────────────


def set_ratchet(*, name: str, new_value: float, source: str, reason: str = "") -> ThresholdState:
    """Raise the threshold floor. Validates monotonicity + sanity ceiling.

    Args:
        name: Threshold name (``ThresholdName.value``).
        new_value: New value, MUST be > current and <= 1.0.
        source: Audit identifier — typically ``"operator_react"``.
        reason: Optional rationale for the audit trail.

    Returns:
        The persisted ``ThresholdState`` after the change.
    """
    name = _validate_name(name)
    if not isinstance(new_value, (int, float)):
        raise CeilingViolation(f"new_value must be numeric, got {type(new_value).__name__}")
    new_value = float(new_value)
    if new_value > 1.0:
        raise CeilingViolation(f"new_value {new_value} > 1.0 sanity ceiling")
    if new_value < 0.0:
        raise CeilingViolation(f"new_value {new_value} < 0.0")

    state = store.get(name)
    floor = _floor(name)
    current = float(state.current) if state else floor

    if new_value <= current:
        raise MonotonicViolation(
            f"set_ratchet on {name}: new_value {new_value} must be > current {current}"
        )

    chain = audit.append(
        action="ratchet_up",
        threshold=name,
        old_value=current,
        new_value=new_value,
        source=source,
        reason=reason,
    )

    entry = RatchetEntry(
        ts=_now_iso(),
        direction=Direction.UP,
        old_value=current,
        new_value=new_value,
        source=source,
        reason=reason,
        audit_chain=chain,
    )
    if state is None:
        state = ThresholdState(name=name, current=new_value, history=[entry])
    else:
        state.current = new_value
        state.history.append(entry)
    store.set_one(name, state)
    logger.info(
        "governance_ratchet: %s ratcheted %s → %s by %s",
        name, current, new_value, source,
    )
    try:
        from app.identity.continuity_ledger import record_event
        record_event(
            kind="governance_ratchet",
            actor=source,
            summary=f"ratcheted up {name} {current} → {new_value}",
            detail={
                "threshold": name,
                "direction": "up",
                "old_value": current,
                "new_value": new_value,
                "reason": reason,
            },
        )
    except Exception:
        logger.debug("identity ledger emission failed", exc_info=True)
    return state


def relax_ratchet(*, name: str, new_value: float, source: str, reason: str) -> ThresholdState:
    """Lower the threshold. Validates floor + monotonic-down direction.

    ``reason`` is MANDATORY (caller must pass a non-empty string).
    The UI is responsible for the typed-phrase confirmation gate
    before calling this — this function is purely the validation +
    persistence layer.
    """
    name = _validate_name(name)
    if not isinstance(new_value, (int, float)):
        raise CeilingViolation(f"new_value must be numeric, got {type(new_value).__name__}")
    new_value = float(new_value)
    if not (reason or "").strip():
        raise ValueError("relax_ratchet requires a non-empty reason")
    if new_value < 0.0:
        raise CeilingViolation(f"new_value {new_value} < 0.0")

    floor = _floor(name)
    if new_value < floor:
        raise FloorViolation(
            f"relax_ratchet on {name}: new_value {new_value} < FLOOR {floor}"
        )

    state = store.get(name)
    current = float(state.current) if state else floor

    if new_value >= current:
        raise MonotonicViolation(
            f"relax_ratchet on {name}: new_value {new_value} must be < current {current}; "
            f"use set_ratchet for upward changes"
        )

    chain = audit.append(
        action="ratchet_down",
        threshold=name,
        old_value=current,
        new_value=new_value,
        floor=floor,
        source=source,
        reason=reason,
    )

    entry = RatchetEntry(
        ts=_now_iso(),
        direction=Direction.DOWN,
        old_value=current,
        new_value=new_value,
        source=source,
        reason=reason,
        audit_chain=chain,
    )
    if state is None:
        state = ThresholdState(name=name, current=new_value, history=[entry])
    else:
        state.current = new_value
        state.history.append(entry)
    store.set_one(name, state)
    logger.warning(
        "governance_ratchet: %s RELAXED %s → %s by %s (reason: %s)",
        name, current, new_value, source, reason,
    )
    try:
        from app.identity.continuity_ledger import record_event
        record_event(
            kind="governance_ratchet",
            actor=source,
            summary=f"relaxed {name} {current} → {new_value}",
            detail={
                "threshold": name,
                "direction": "down",
                "old_value": current,
                "new_value": new_value,
                "floor": floor,
                "reason": reason,
            },
        )
    except Exception:
        logger.debug("identity ledger emission failed", exc_info=True)
    return state
