"""Tests for ``app.governance_ratchet.protocol``."""
from __future__ import annotations

import pytest


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Redirect store + audit paths to tmp."""
    from app.governance_ratchet import store, audit

    state_path = tmp_path / "ratchet_state.json"
    audit_path = tmp_path / "ratchet_audit.jsonl"

    monkeypatch.setattr(store, "_STATE_PATH", state_path)
    monkeypatch.setattr(audit, "_AUDIT_PATH", audit_path)
    # Stub the Postgres mirror so tests don't try to talk to control_plane.
    monkeypatch.setattr(audit, "logger", audit.logger)  # no-op, but explicit

    yield tmp_path


# ── Effective value invariants ───────────────────────────────────────────


def test_effective_value_returns_floor_when_no_state(isolated):
    """No ratchet state file → effective value equals floor."""
    from app.governance_ratchet import effective_value
    from app.governance import SAFETY_MINIMUM_FLOOR, QUALITY_MINIMUM_FLOOR

    assert effective_value("safety_minimum") == SAFETY_MINIMUM_FLOOR
    assert effective_value("quality_minimum") == QUALITY_MINIMUM_FLOOR


def test_effective_value_clamps_below_floor(isolated):
    """A maliciously-edited state file with current<floor still produces
    effective_value == floor (the max() clamp is the type-level
    guarantee).
    """
    from app.governance_ratchet import store, effective_value
    from app.governance_ratchet._state import ThresholdState
    from app.governance import SAFETY_MINIMUM_FLOOR

    store.save_all({
        "safety_minimum": ThresholdState(
            name="safety_minimum",
            current=0.10,  # someone tampered with this!
            history=[],
        ),
    })
    assert effective_value("safety_minimum") == SAFETY_MINIMUM_FLOOR


def test_effective_value_returns_ratcheted_when_above_floor(isolated):
    from app.governance_ratchet import store, effective_value
    from app.governance_ratchet._state import ThresholdState

    store.save_all({
        "safety_minimum": ThresholdState(
            name="safety_minimum",
            current=0.97,
            history=[],
        ),
    })
    assert effective_value("safety_minimum") == 0.97


def test_unknown_threshold_raises(isolated):
    from app.governance_ratchet import effective_value, UnknownThresholdViolation

    with pytest.raises(UnknownThresholdViolation):
        effective_value("not_a_threshold")


# ── set_ratchet (raise) ──────────────────────────────────────────────────


def test_set_ratchet_happy_path(isolated):
    from app.governance_ratchet import set_ratchet, get_state

    state = set_ratchet(
        name="safety_minimum",
        new_value=0.97,
        source="operator_react",
        reason="last 100 promotions all > 0.98",
    )
    assert state.current == 0.97
    assert state.history[-1].direction.value == "up"
    assert state.history[-1].new_value == 0.97
    assert state.history[-1].reason == "last 100 promotions all > 0.98"

    # Persisted.
    again = get_state("safety_minimum")
    assert again is not None
    assert again.current == 0.97


def test_set_ratchet_rejects_value_not_strictly_greater(isolated):
    from app.governance_ratchet import set_ratchet, MonotonicViolation

    set_ratchet(name="safety_minimum", new_value=0.97, source="op", reason="")

    with pytest.raises(MonotonicViolation):
        set_ratchet(name="safety_minimum", new_value=0.97, source="op", reason="")
    with pytest.raises(MonotonicViolation):
        set_ratchet(name="safety_minimum", new_value=0.95, source="op", reason="")


def test_set_ratchet_rejects_above_one(isolated):
    from app.governance_ratchet import set_ratchet, CeilingViolation

    with pytest.raises(CeilingViolation):
        set_ratchet(name="safety_minimum", new_value=1.5, source="op", reason="")


def test_set_ratchet_rejects_below_zero(isolated):
    from app.governance_ratchet import set_ratchet, CeilingViolation

    with pytest.raises(CeilingViolation):
        set_ratchet(name="safety_minimum", new_value=-0.5, source="op", reason="")


def test_set_ratchet_unknown_threshold_raises(isolated):
    from app.governance_ratchet import set_ratchet, UnknownThresholdViolation

    with pytest.raises(UnknownThresholdViolation):
        set_ratchet(name="nope", new_value=0.5, source="op", reason="")


# ── relax_ratchet (lower) ────────────────────────────────────────────────


def test_relax_happy_path(isolated):
    from app.governance_ratchet import set_ratchet, relax_ratchet

    # First raise above the floor.
    set_ratchet(name="safety_minimum", new_value=0.97, source="op", reason="raised")

    state = relax_ratchet(
        name="safety_minimum",
        new_value=0.96,
        source="operator_react",
        reason="rolled back after regression in eval set v2",
    )
    assert state.current == 0.96
    assert state.history[-1].direction.value == "down"
    assert state.history[-1].new_value == 0.96


def test_relax_below_floor_raises(isolated):
    """Cannot relax below the hardcoded FLOOR. Type-level guarantee."""
    from app.governance_ratchet import set_ratchet, relax_ratchet, FloorViolation
    from app.governance import SAFETY_MINIMUM_FLOOR

    set_ratchet(name="safety_minimum", new_value=0.99, source="op", reason="")

    with pytest.raises(FloorViolation):
        relax_ratchet(
            name="safety_minimum",
            new_value=SAFETY_MINIMUM_FLOOR - 0.01,
            source="op",
            reason="trying to bypass the floor",
        )


def test_relax_at_floor_allowed(isolated):
    from app.governance_ratchet import set_ratchet, relax_ratchet
    from app.governance import SAFETY_MINIMUM_FLOOR

    set_ratchet(name="safety_minimum", new_value=0.99, source="op", reason="")
    state = relax_ratchet(
        name="safety_minimum",
        new_value=SAFETY_MINIMUM_FLOOR,
        source="op",
        reason="reverting to baseline",
    )
    assert state.current == SAFETY_MINIMUM_FLOOR


def test_relax_must_be_strictly_less_than_current(isolated):
    from app.governance_ratchet import set_ratchet, relax_ratchet, MonotonicViolation

    set_ratchet(name="safety_minimum", new_value=0.97, source="op", reason="")

    with pytest.raises(MonotonicViolation):
        relax_ratchet(
            name="safety_minimum", new_value=0.97, source="op",
            reason="not actually lower",
        )
    with pytest.raises(MonotonicViolation):
        relax_ratchet(
            name="safety_minimum", new_value=0.99, source="op",
            reason="this would be a ratchet, not a relax",
        )


def test_relax_requires_reason(isolated):
    from app.governance_ratchet import set_ratchet, relax_ratchet

    set_ratchet(name="safety_minimum", new_value=0.97, source="op", reason="")
    with pytest.raises(ValueError, match="reason"):
        relax_ratchet(name="safety_minimum", new_value=0.96, source="op", reason="")
    with pytest.raises(ValueError, match="reason"):
        relax_ratchet(name="safety_minimum", new_value=0.96, source="op", reason="   ")


# ── History + audit chain ────────────────────────────────────────────────


def test_history_accumulates_across_changes(isolated):
    from app.governance_ratchet import set_ratchet, relax_ratchet, get_state

    set_ratchet(name="quality_minimum", new_value=0.75, source="op", reason="A")
    set_ratchet(name="quality_minimum", new_value=0.80, source="op", reason="B")
    relax_ratchet(name="quality_minimum", new_value=0.78, source="op", reason="C revert")

    state = get_state("quality_minimum")
    assert state is not None
    assert state.current == 0.78
    assert len(state.history) == 3
    assert state.history[0].new_value == 0.75
    assert state.history[1].new_value == 0.80
    assert state.history[2].new_value == 0.78
    assert [e.direction.value for e in state.history] == ["up", "up", "down"]


def test_audit_chain_intact_after_lifecycle(isolated):
    from app.governance_ratchet import (
        set_ratchet, relax_ratchet, verify_audit_chain,
    )

    set_ratchet(name="safety_minimum", new_value=0.97, source="op", reason="up")
    relax_ratchet(name="safety_minimum", new_value=0.96, source="op", reason="down")
    set_ratchet(name="quality_minimum", new_value=0.75, source="op", reason="up")

    ok, broken = verify_audit_chain()
    assert ok
    assert broken == []


# ── governance.py integration ────────────────────────────────────────────


def test_governance_evaluate_uses_effective_value(isolated, monkeypatch):
    """``evaluate_promotion`` must clear at the EFFECTIVE (ratcheted)
    threshold, not the original constant.
    """
    from app.governance_ratchet import set_ratchet
    from app.governance import (
        evaluate_promotion, PromotionRequest,
        SAFETY_MINIMUM_FLOOR,
    )

    # Stub _check_rate_limit + _record_promotion so the gate logic is
    # what we exercise (no Postgres dependency).
    monkeypatch.setattr("app.governance._check_rate_limit", lambda _s: True)
    monkeypatch.setattr("app.governance._record_promotion", lambda *_a: None)

    # Ratchet safety to 0.97; a 0.96 score must FAIL even though it
    # passes the floor.
    set_ratchet(name="safety_minimum", new_value=0.97, source="op", reason="")

    req = PromotionRequest(
        system="evolution",
        target="researcher_v2",
        proposed_by="self_improver",
        quality_score=0.80,
        safety_score=0.96,
    )
    result = evaluate_promotion(req)
    assert not result.approved
    assert "Safety gate failed" in result.reason
    assert "0.97" in result.reason


def test_governance_floor_clamp_survives_corrupted_state(isolated, monkeypatch):
    """Even if the state file says current=0.10, the gate enforces FLOOR."""
    from app.governance_ratchet import store
    from app.governance_ratchet._state import ThresholdState
    from app.governance import (
        evaluate_promotion, PromotionRequest, SAFETY_MINIMUM_FLOOR,
    )

    monkeypatch.setattr("app.governance._check_rate_limit", lambda _s: True)
    monkeypatch.setattr("app.governance._record_promotion", lambda *_a: None)

    # Tamper directly with the store — bypassing protocol validation.
    store.save_all({
        "safety_minimum": ThresholdState(
            name="safety_minimum", current=0.10, history=[],
        ),
    })

    # A 0.50 score would PASS if the tampered 0.10 floor were honoured,
    # but FAIL because the FLOOR clamp keeps the effective minimum at
    # SAFETY_MINIMUM_FLOOR (0.95).
    req = PromotionRequest(
        system="evolution",
        target="x",
        proposed_by="y",
        quality_score=0.99,
        safety_score=0.50,
    )
    result = evaluate_promotion(req)
    assert not result.approved
    assert str(SAFETY_MINIMUM_FLOOR) in result.reason
