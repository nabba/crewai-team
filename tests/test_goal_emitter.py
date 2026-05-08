"""Tests for app.affect.goal_emitter — viability → current_goals connector.

Consciousness-roadmap §3.G1. The SCORECARD's diagnosis of AE-1 PARTIAL is
*"Goals are still user-dispatched, not autonomously generated"*; this
module's tests pin the contract that closes that gap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from app.affect.goal_emitter import (
    EmitterResult,
    GoalProposal,
    derive_proposals,
    dedup_against_existing,
    fifo_evict_to_cap,
    run_pass,
    _reset_rate_limit_for_tests,
)
from app.affect.schemas import ViabilityFrame, ViabilityVariable


# ── Helpers ──────────────────────────────────────────────────────────────


def _frame(*, errors: dict[str, float], setpoint: float = 0.5) -> ViabilityFrame:
    """Build a ViabilityFrame where each given variable has the specified
    |error| from its setpoint. All other variables are at setpoint (zero
    error).
    """
    setpoints = {v.value: setpoint for v in ViabilityVariable}
    values = dict(setpoints)
    for var, err in errors.items():
        # Add error symmetrically — direction doesn't matter, magnitude does.
        values[var] = min(1.0, setpoints[var] + err)
    return ViabilityFrame(
        values=values,
        setpoints=setpoints,
        weights={v.value: 1.0 for v in ViabilityVariable},
        total_error=sum(errors.values()),
        sources={v.value: "test" for v in ViabilityVariable},
        ts="2026-05-08T12:00:00Z",
    )


@dataclass
class _FakeSelfState:
    current_goals: list = field(default_factory=list)


@dataclass
class _FakeKernel:
    self_state: _FakeSelfState = field(default_factory=_FakeSelfState)


@pytest.fixture(autouse=True)
def _reset_rate_limit():
    """Each test starts with a fresh rate-limit window."""
    _reset_rate_limit_for_tests()
    yield


# ── Pure-logic: derive_proposals ─────────────────────────────────────────


def test_derive_skips_when_too_few_frames():
    frames = [_frame(errors={"compute_reserve": 0.5})]
    assert derive_proposals(frames, n_consecutive=3) == []


def test_derive_skips_when_error_below_threshold():
    """All variables in healthy band → no proposals."""
    frames = [_frame(errors={"compute_reserve": 0.05}) for _ in range(5)]
    assert derive_proposals(frames, n_consecutive=3, error_threshold=0.25) == []


def test_derive_skips_when_error_dips_within_window():
    """Sustained pressure required: even 4 of 5 high frames don't qualify
    if any frame in the consecutive window dips below threshold."""
    frames = [
        _frame(errors={"compute_reserve": 0.5}),
        _frame(errors={"compute_reserve": 0.5}),
        _frame(errors={"compute_reserve": 0.0}),  # the dip
        _frame(errors={"compute_reserve": 0.5}),
        _frame(errors={"compute_reserve": 0.5}),  # last 3 frames are 0.0/0.5/0.5 → not all ≥ threshold
    ]
    proposals = derive_proposals(frames, n_consecutive=3, error_threshold=0.25)
    assert proposals == []


def test_derive_emits_when_sustained_above_threshold():
    frames = [_frame(errors={"compute_reserve": 0.4}) for _ in range(5)]
    proposals = derive_proposals(frames, n_consecutive=3, error_threshold=0.25)
    assert len(proposals) == 1
    assert proposals[0].triggered_by == "compute_reserve"
    assert proposals[0].sustained_error == pytest.approx(0.4)


def test_derive_caps_at_max_proposals():
    """Many variables in trouble at once → still bounded by max_proposals."""
    errors = {v.value: 0.4 for v in ViabilityVariable}
    frames = [_frame(errors=errors) for _ in range(5)]
    proposals = derive_proposals(
        frames, n_consecutive=3, error_threshold=0.25, max_proposals=2,
    )
    assert len(proposals) == 2


def test_derive_orders_by_sustained_error_descending():
    """Highest-error variable should rank first."""
    frames = [
        _frame(errors={
            "compute_reserve": 0.30,
            "task_coherence": 0.50,    # higher
            "memory_pressure": 0.40,
        }) for _ in range(5)
    ]
    proposals = derive_proposals(frames, n_consecutive=3, max_proposals=3)
    triggers = [p.triggered_by for p in proposals]
    assert triggers[0] == "task_coherence"


def test_derive_proposal_has_template_text():
    frames = [_frame(errors={"epistemic_uncertainty": 0.4}) for _ in range(5)]
    proposals = derive_proposals(frames, n_consecutive=3)
    assert "epistemic uncertainty" in proposals[0].text.lower()


# ── Pure-logic: dedup_against_existing ───────────────────────────────────


def test_dedup_skips_existing_active_trigger():
    proposals = [
        GoalProposal(id="g1", text="x", triggered_by="compute_reserve",
                     sustained_error=0.4, proposed_at="t"),
    ]
    existing = [{"triggered_by": "compute_reserve", "source": "viability-goal-emitter"}]
    assert dedup_against_existing(proposals, existing) == []


def test_dedup_passes_through_when_trigger_not_active():
    proposals = [
        GoalProposal(id="g1", text="x", triggered_by="compute_reserve",
                     sustained_error=0.4, proposed_at="t"),
    ]
    existing = [{"triggered_by": "memory_pressure"}]
    out = dedup_against_existing(proposals, existing)
    assert len(out) == 1


def test_dedup_ignores_non_dict_existing_entries():
    """current_goals may already contain legacy/string entries."""
    proposals = [
        GoalProposal(id="g1", text="x", triggered_by="compute_reserve",
                     sustained_error=0.4, proposed_at="t"),
    ]
    existing = ["a legacy string goal", 42, None]
    out = dedup_against_existing(proposals, existing)
    assert len(out) == 1


# ── Pure-logic: fifo_evict_to_cap ────────────────────────────────────────


def test_fifo_evicts_oldest_emitter_goals_when_above_cap():
    existing = [
        {"triggered_by": "var1", "source": "viability-goal-emitter"},
        {"triggered_by": "var2", "source": "viability-goal-emitter"},
        {"triggered_by": "var3", "source": "viability-goal-emitter"},
    ]
    new = [
        GoalProposal(id="g4", text="x", triggered_by="var4",
                     sustained_error=0.4, proposed_at="t"),
        GoalProposal(id="g5", text="y", triggered_by="var5",
                     sustained_error=0.4, proposed_at="t"),
    ]
    out = fifo_evict_to_cap(existing, new, cap=3)
    assert len(out) == 3
    triggers = [g.get("triggered_by") for g in out]
    # Oldest two ours dropped; var3, var4, var5 survive.
    assert triggers == ["var3", "var4", "var5"]


def test_fifo_preserves_non_emitter_entries():
    """Legacy goals or grand_task entries (no source=='viability-goal-emitter')
    must NOT be evicted by our cap logic."""
    existing = [
        "legacy string goal",
        {"triggered_by": "var1", "source": "viability-goal-emitter"},
        {"text": "grand-task proposal", "source": "companion.grand_task"},
    ]
    new = [
        GoalProposal(id="g2", text="x", triggered_by="var2",
                     sustained_error=0.4, proposed_at="t"),
        GoalProposal(id="g3", text="y", triggered_by="var3",
                     sustained_error=0.4, proposed_at="t"),
    ]
    out = fifo_evict_to_cap(existing, new, cap=3)
    # Non-emitter entries preserved; only oldest emitter goal evicted.
    assert "legacy string goal" in out
    assert any(g.get("source") == "companion.grand_task" for g in out
               if isinstance(g, dict))


def test_fifo_no_eviction_when_under_cap():
    existing = [{"triggered_by": "var1", "source": "viability-goal-emitter"}]
    new = [GoalProposal(id="g2", text="x", triggered_by="var2",
                        sustained_error=0.4, proposed_at="t")]
    out = fifo_evict_to_cap(existing, new, cap=5)
    assert len(out) == 2


# ── Integration: run_pass ────────────────────────────────────────────────


def test_run_pass_skips_with_no_kernel():
    result = run_pass(kernel=None, frames=[], force=True)
    assert result.skipped


def test_run_pass_skips_with_too_few_frames():
    kernel = _FakeKernel()
    frames = [_frame(errors={"compute_reserve": 0.5})]
    result = run_pass(kernel=kernel, frames=frames, force=True)
    assert result.skipped
    assert "insufficient" in (result.skip_reason or "")


def test_run_pass_writes_to_kernel_on_qualifying_frames():
    kernel = _FakeKernel()
    frames = [_frame(errors={"compute_reserve": 0.4}) for _ in range(5)]
    result = run_pass(kernel=kernel, frames=frames, force=True)

    assert not result.skipped
    assert len(result.written) == 1
    assert kernel.self_state.current_goals
    assert kernel.self_state.current_goals[0]["triggered_by"] == "compute_reserve"
    assert kernel.self_state.current_goals[0]["source"] == "viability-goal-emitter"


def test_run_pass_dedups_against_existing_goals():
    kernel = _FakeKernel()
    kernel.self_state.current_goals = [
        {"triggered_by": "compute_reserve", "source": "viability-goal-emitter"},
    ]
    frames = [_frame(errors={"compute_reserve": 0.4}) for _ in range(5)]
    result = run_pass(kernel=kernel, frames=frames, force=True)

    assert len(result.written) == 0
    assert "dedup_existing" in result.skipped_reasons


def test_run_pass_respects_rate_limit():
    """Without `force=True`, back-to-back invocations after the first
    successful one are rate-limited."""
    kernel = _FakeKernel()
    frames = [_frame(errors={"compute_reserve": 0.4}) for _ in range(5)]

    first = run_pass(kernel=kernel, frames=frames, force=False)
    assert not first.skipped
    second = run_pass(kernel=kernel, frames=frames, force=False)
    assert second.skipped
    assert "rate-limited" in (second.skip_reason or "")


def test_run_pass_handles_kernel_without_self_state():
    """A bare object missing self_state shouldn't crash — skipped cleanly."""
    bare = SimpleNamespace()
    frames = [_frame(errors={"compute_reserve": 0.4}) for _ in range(5)]
    result = run_pass(kernel=bare, frames=frames, force=True)
    assert result.skipped


def test_emitter_result_to_dict_is_json_safe():
    import json
    kernel = _FakeKernel()
    frames = [_frame(errors={"compute_reserve": 0.4}) for _ in range(5)]
    result = run_pass(kernel=kernel, frames=frames, force=True)
    payload = json.dumps(result.to_dict())
    assert "written_count" in payload


def test_t1_threshold_documented_invariant():
    """G1's effect on AE-1 SCORECARD rating: a successful goal write means
    `current_goals` is no longer empty/dead. This test pins that invariant
    so the AE-1 → STRONG case is testable."""
    kernel = _FakeKernel()
    assert kernel.self_state.current_goals == []  # dead-field starting state

    frames = [_frame(errors={"compute_reserve": 0.4}) for _ in range(5)]
    result = run_pass(kernel=kernel, frames=frames, force=True)

    # After a qualifying viability event, autonomous goals exist.
    assert kernel.self_state.current_goals != []
    assert any(
        g.get("source") == "viability-goal-emitter"
        for g in kernel.self_state.current_goals
        if isinstance(g, dict)
    ), "at least one goal must be autonomously generated"
