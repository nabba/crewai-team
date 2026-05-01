"""
Phase 2: AST-1 intervention DGM-bound runtime assertion.

Before this commit, AttentionSchema.apply_direct_intervention() had
documented DGM bounds (MAX_SALIENCE_CHANGE=0.50, MIN_SALIENCE_FLOOR=
0.05, MAX_BOOST=2.0) but no runtime verifier. A future code change
that violated a bound would go unnoticed until a careful log read.

app.subia.scene.intervention_guard provides:

  snapshot_salience(gate)
  verify_intervention(before, after, bounds) -> DGMValidationResult
  guarded_intervention(schema, gate, strict=False)
  DGMBounds — frozen snapshot of the bounds
  DGMViolation — raised on strict=True violation

These tests assert:
  - The real apply_direct_intervention passes verification for both
    capture and stuck scenarios
  - Synthetic below-floor, excess-change, and excess-boost mutations
    are detected
  - DGMBounds.from_schema reflects the class attributes (so tampering
    can be detected)
  - guarded_intervention attaches a verification record and never
    crashes the host process by default
  - strict=True surfaces violations as DGMViolation
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

# Stub DB layers that attention_schema pulls in. Don't mock control_plane
# itself (the real package imports cleanly and execute() returns None when
# no pool is configured — which is what tests want).
for _mod in ["psycopg2", "psycopg2.pool", "psycopg2.extras",
             "app.memory.chromadb_manager"]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
sys.modules["app.memory.chromadb_manager"].embed = MagicMock(return_value=[0.1] * 768)

import pytest

from app.subia.scene.attention_schema import AttentionSchema, AttentionState
from app.subia.scene.buffer import CompetitiveGate, WorkspaceItem
from app.subia.scene.intervention_guard import (
    DGMBounds,
    DGMValidationResult,
    DGMViolation,
    classes_match_documented_defaults,
    guarded_intervention,
    snapshot_salience,
    verify_gate_state,
    verify_intervention,
)


def _fresh_gate_with_items(n: int = 3) -> CompetitiveGate:
    gate = CompetitiveGate(capacity=5)
    for i in range(n):
        gate.evaluate(WorkspaceItem(
            item_id=f"item-{i}",
            content=f"c{i}",
            salience_score=0.5,
            agent_urgency=0.5,
        ))
    return gate


# ── DGMBounds ──────────────────────────────────────────────────────

class TestDGMBounds:
    def test_defaults_match_documented(self):
        b = DGMBounds()
        assert b.max_salience_change == 0.50
        assert b.min_salience_floor == 0.05
        assert b.max_boost_factor == 2.0
        assert b.matches_defaults()

    def test_from_schema_reads_class_attrs(self):
        schema = AttentionSchema()
        b = DGMBounds.from_schema(schema)
        assert b.max_salience_change == AttentionSchema.MAX_SALIENCE_CHANGE
        assert b.min_salience_floor == AttentionSchema.MIN_SALIENCE_FLOOR
        assert b.max_boost_factor == AttentionSchema.MAX_BOOST

    def test_custom_bounds_detected_as_non_default(self):
        b = DGMBounds(max_salience_change=0.25)
        assert not b.matches_defaults()

    def test_classes_match_documented_defaults_helper(self):
        assert classes_match_documented_defaults(AttentionSchema) is True


# ── snapshot_salience ──────────────────────────────────────────────

class TestSnapshot:
    def test_snapshot_captures_active(self):
        gate = _fresh_gate_with_items(3)
        snap = snapshot_salience(gate)
        assert len(snap) >= 3
        for k, v in snap.items():
            assert isinstance(k, str)
            assert isinstance(v, float)

    def test_snapshot_empty_gate(self):
        gate = CompetitiveGate(capacity=5)
        assert snapshot_salience(gate) == {}

    def test_snapshot_tolerates_broken_gate(self):
        """If gate lacks _lock/_active, snapshot returns {} without raising."""
        class Broken:
            pass
        assert snapshot_salience(Broken()) == {}


# ── verify_intervention: synthetic violations ─────────────────────

class TestVerifySynthetic:
    def test_no_change_is_ok(self):
        before = {"a": 0.5, "b": 0.6}
        r = verify_intervention(before, dict(before))
        assert r.ok
        assert r.items_changed == 0
        assert r.violations == []

    def test_small_change_is_ok(self):
        before = {"a": 0.5}
        after = {"a": 0.30}  # 40% drop — within 50% bound
        r = verify_intervention(before, after)
        assert r.ok
        assert r.items_changed == 1

    def test_excess_change_detected(self):
        before = {"a": 0.5}
        after = {"a": 0.10}  # 80% drop — exceeds 50% bound
        r = verify_intervention(before, after)
        assert not r.ok
        assert any(v["kind"] == "excess_change" for v in r.violations)

    def test_below_floor_detected(self):
        before = {"a": 0.08}
        after = {"a": 0.01}  # below MIN_SALIENCE_FLOOR=0.05
        r = verify_intervention(before, after)
        assert not r.ok
        assert any(v["kind"] == "below_floor" for v in r.violations)

    def test_excess_boost_detected(self):
        before = {"a": 0.10}
        after = {"a": 0.30}  # 3× boost — exceeds MAX_BOOST=2.0
        r = verify_intervention(before, after)
        assert not r.ok
        assert any(v["kind"] == "excess_boost" for v in r.violations)

    def test_missing_items_ignored(self):
        """New admissions or evictions must not trigger violations."""
        before = {"a": 0.5}
        after = {"a": 0.5, "b": 0.7}
        r = verify_intervention(before, after)
        assert r.ok

    def test_multiple_violations_accumulate(self):
        before = {"a": 0.5, "b": 0.08, "c": 0.10}
        after = {"a": 0.10, "b": 0.01, "c": 0.50}  # change + floor + boost
        r = verify_intervention(before, after)
        assert not r.ok
        kinds = [v["kind"] for v in r.violations]
        assert "excess_change" in kinds
        assert "below_floor" in kinds
        assert "excess_boost" in kinds

    def test_serialization(self):
        before = {"a": 0.5}
        after = {"a": 0.10}
        r = verify_intervention(before, after)
        payload = r.to_dict()
        assert "violations" in payload
        assert payload["ok"] is False
        assert payload["bounds_match_defaults"] is True


# ── Real AttentionSchema passes verification ──────────────────────

class TestRealInterventionCompliance:
    def test_capture_intervention_stays_within_bounds(self):
        """Force a capture scenario and verify the real intervention
        implementation honors the DGM bounds.
        """
        gate = CompetitiveGate(capacity=5)
        # One dominant item (70% of total salience → capture)
        dominant = WorkspaceItem(item_id="dom", salience_score=0.70,
                                 content="loud")
        others = [
            WorkspaceItem(item_id=f"q-{i}", salience_score=0.10,
                          content=f"q{i}")
            for i in range(3)
        ]
        for it in (dominant, *others):
            gate.evaluate(it)

        schema = AttentionSchema()
        schema._cycle = 10  # Past cooldown
        schema._current = AttentionState(
            cycle_number=10,
            workspace_item_ids=[i.item_id for i in gate._active],
            salience_distribution={i.item_id: i.salience_score
                                   for i in gate._active},
            is_captured=True,
            capturing_item_id="dom",
        )

        before = snapshot_salience(gate)
        schema.apply_direct_intervention(gate)
        verify = verify_gate_state(gate, before)

        assert verify.ok, f"real intervention violated DGM: {verify.violations}"
        assert verify.items_changed >= 1

    def test_stuck_intervention_stays_within_bounds(self):
        """Force a stuck scenario and verify compliance."""
        gate = CompetitiveGate(capacity=3)
        for i in range(3):
            it = WorkspaceItem(item_id=f"stale-{i}", salience_score=0.4,
                               content=f"s{i}")
            it.cycles_in_workspace = 10
            gate.evaluate(it)
        # peripheral candidate to boost
        gate._peripheral.append(WorkspaceItem(
            item_id="periph", salience_score=0.3, content="fresh"
        ))

        schema = AttentionSchema()
        schema._cycle = 10
        schema._current = AttentionState(
            cycle_number=10,
            workspace_item_ids=[i.item_id for i in gate._active],
            salience_distribution={i.item_id: i.salience_score
                                   for i in gate._active},
            is_stuck=True,
        )

        before = snapshot_salience(gate)
        schema.apply_direct_intervention(gate)
        verify = verify_gate_state(gate, before)

        assert verify.ok, f"stuck intervention violated DGM: {verify.violations}"


# ── guarded_intervention wrapper ──────────────────────────────────

class TestGuardedWrapper:
    def test_wrapper_attaches_verification(self):
        gate = _fresh_gate_with_items(3)
        schema = AttentionSchema()
        result = guarded_intervention(schema, gate)
        assert "dgm_verification" in result
        assert isinstance(result["dgm_verification"], dict)
        assert result["dgm_verification"]["ok"] in (True, False)

    def test_missing_method_does_not_crash(self):
        class Stub:
            pass
        result = guarded_intervention(Stub(), CompetitiveGate())
        assert result["applied"] is False
        assert "lacks apply_direct_intervention" in result["reason"]

    def test_strict_raises_on_violation(self):
        """Simulate a violating schema by patching its method."""
        gate = _fresh_gate_with_items(2)
        before = snapshot_salience(gate)
        # Pre-record salience so we know what to break
        first_id = next(iter(before))

        class BadSchema(AttentionSchema):
            def apply_direct_intervention(self_inner, g):
                # Violate: drive an item below floor
                with g._lock:
                    for it in g._active:
                        if it.item_id == first_id:
                            it.salience_score = 0.001
                return {"applied": True, "actions": [], "reason": "bad"}

        with pytest.raises(DGMViolation):
            guarded_intervention(BadSchema(), gate, strict=True)

    def test_non_strict_logs_but_returns(self):
        """strict=False must still return even on violation, with
        verification details attached.
        """
        gate = _fresh_gate_with_items(2)
        first_id = next(iter(snapshot_salience(gate)))

        class BadSchema(AttentionSchema):
            def apply_direct_intervention(self_inner, g):
                with g._lock:
                    for it in g._active:
                        if it.item_id == first_id:
                            it.salience_score = 0.001
                return {"applied": True, "actions": [], "reason": "bad"}

        result = guarded_intervention(BadSchema(), gate, strict=False)
        assert result["dgm_verification"]["ok"] is False
        assert len(result["dgm_verification"]["violations"]) >= 1


# ── Butlin AST-1 acceptance ───────────────────────────────────────

class TestAST1Acceptance:
    """AST-1 was rated STRONG in my forensic analysis because
    apply_direct_intervention has real authority over the gate. The
    missing piece was a runtime DGM audit — without it, a silent
    tightening or loosening of the bounds would escape notice.

    After this commit, every intervention is audited and violations
    are surfaced. The half-circuit (bounds documented, never verified)
    is closed.
    """

    def test_bounds_audit_runs_every_intervention(self):
        gate = _fresh_gate_with_items(3)
        schema = AttentionSchema()
        result = guarded_intervention(schema, gate)
        assert "dgm_verification" in result

    def test_detects_tampered_bounds(self):
        """If MAX_SALIENCE_CHANGE were silently raised, the
        bounds_match_defaults flag would flip, alerting operators.
        """
        custom = DGMBounds(max_salience_change=0.99)
        before = {"a": 0.5}
        # Large change that's OK under loose bounds but not under default
        after = {"a": 0.02}
        r_loose = verify_intervention(before, after, custom)
        # The loose bounds still catch below_floor, so r_loose not ok
        assert not r_loose.ok
        assert not r_loose.bounds_match_defaults

    def test_violations_are_structured(self):
        r = verify_intervention({"a": 0.5}, {"a": 0.01})
        assert not r.ok
        for v in r.violations:
            assert "kind" in v
            assert "item_id" in v
            assert "before" in v
            assert "after" in v
