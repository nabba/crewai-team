"""Drill orchestrator — invokes drills, threads state + baseline + audit.

PROGRAM §57 — Q18 resilience-drill v2. The orchestrator is the
heart of the v2 system: every drill invocation (from the scheduler,
REST, or CLI) goes through ``invoke_drill`` so state transitions,
baseline comparisons, audit appends, and operator notifications are
threaded consistently.

This module also enforces the safety contract: a drill that has
been QUARANTINED or MUTED, or whose backoff has not yet elapsed,
will NOT have its runner called. The hot-loop pattern from
2026-05-16 is structurally impossible — the scheduler can call
``invoke_drill`` every second and the orchestrator will return
SKIPPED results until the state machine permits a real run.
"""
from __future__ import annotations

import logging
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Callable

from app.resilience_drills import baseline as bl
from app.resilience_drills import state as st
from app.resilience_drills.audit import (
    acquire_drill_lock,
    append_result,
    emit_landmark_for,
    last_result_for,
    last_successful_for,
    release_drill_lock,
)
from app.resilience_drills.protocol import (
    DrillResult,
    DrillRisk,
    DrillSpec,
    DrillStatus,
    FailureClass,
    drill_enabled,
)

logger = logging.getLogger(__name__)


# ── Public entry point ──────────────────────────────────────────────────


def invoke_drill(
    spec: DrillSpec,
    runner_fn: Callable[..., DrillResult],
    *,
    dry_run: bool = True,
    triggered_by: str = "scheduler",
) -> DrillResult:
    """Invoke a drill end-to-end with state threading.

    Args:
        spec: The drill's :class:`DrillSpec` (defines cadence, warmup,
            master switch, etc.).
        runner_fn: The drill's ``run(*, dry_run=...)`` callable. Must
            return a :class:`DrillResult`.
        dry_run: Passed through to the runner. Most LOW-risk drills
            ignore this; HIGH-risk drills should respect it.
        triggered_by: ``"scheduler"`` (cadence-driven), ``"operator"``
            (manual invocation via REST or CLI), or ``"test"``.

    Returns:
        A :class:`DrillResult`. If the state machine declined to run
        the drill (muted/quarantined/backoff), the returned result
        has ``status=SKIPPED`` and ``detail["skip_reason"]`` populated.

    The orchestrator never raises. Runner exceptions become
    ``DrillStatus.ERROR`` with ``failure_class=CODE_ERROR``; the
    state machine then escalates per the ``QUARANTINE_THRESHOLD``
    rule.
    """
    started_at_iso = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()

    # ── Step 1: master switch ────────────────────────────────────────
    if not drill_enabled(spec):
        return _make_skipped(spec, started_at_iso, t0, dry_run, "master switch off")

    # ── Step 2: state machine permission ─────────────────────────────
    state = st.load_or_initialize(spec.name, warmup_days=spec.warmup_days)
    ok, why = st.is_runnable_now(state)
    if not ok and triggered_by == "scheduler":
        # Operator-triggered invocations bypass the backoff/quarantine
        # checks — the operator explicitly asked. Only the cadence-driven
        # scheduler is gated.
        return _make_skipped(spec, started_at_iso, t0, dry_run, why)

    # ── Step 3: in-flight lock (defends against concurrent invocation) ─
    if not acquire_drill_lock(spec.name):
        return _make_skipped(spec, started_at_iso, t0, dry_run, "already in-flight")

    try:
        # ── Step 4: snapshot prior status BEFORE running ───────────
        prior_any = last_result_for(spec.name)
        prior_status_str = (prior_any or {}).get("status") if prior_any else None
        is_first_run = last_successful_for(spec.name) is None

        # ── Step 5: invoke the runner ──────────────────────────────
        result = _safe_run(spec, runner_fn, dry_run, started_at_iso, t0)

        # ── Step 6: baseline comparison (if observation present) ───
        result = _apply_baseline_check(spec, result)

        # ── Step 7: thread state transitions ───────────────────────
        _apply_state_transitions(spec, state, result)

        # ── Step 8: persist audit + emit landmark ──────────────────
        append_result(result)
        if not st.is_warming_up(state):
            # Suppress landmark emissions during warmup — they're
            # observational only, not identity-shaping.
            emit_landmark_for(
                result,
                is_first_run=is_first_run,
                prior_status=prior_status_str,
            )

        return result
    finally:
        release_drill_lock(spec.name)


# ── Step helpers ────────────────────────────────────────────────────────


def _make_skipped(spec: DrillSpec, started_at: str, t0: float,
                  dry_run: bool, reason: str) -> DrillResult:
    """Build a SKIPPED result. Used at every short-circuit return point
    so the caller sees uniform shape."""
    completed = datetime.now(timezone.utc).isoformat()
    return DrillResult(
        drill_name=spec.name,
        status=DrillStatus.SKIPPED,
        started_at=started_at,
        completed_at=completed,
        duration_s=round(time.monotonic() - t0, 4),
        dry_run=dry_run,
        detail={"skip_reason": reason},
    )


def _safe_run(spec: DrillSpec, runner_fn: Callable[..., DrillResult],
              dry_run: bool, started_at: str, t0: float) -> DrillResult:
    """Call the drill's runner, converting any uncaught exception into
    a CODE_ERROR result with traceback captured. The runner is
    expected to return a DrillResult; anything else is treated as a
    drill code bug (CODE_ERROR)."""
    try:
        result = runner_fn(dry_run=dry_run)
    except Exception as exc:  # noqa: BLE001
        completed = datetime.now(timezone.utc).isoformat()
        tb = traceback.format_exc(limit=10)
        logger.warning("drill %s raised: %s", spec.name, exc, exc_info=True)
        return DrillResult(
            drill_name=spec.name,
            status=DrillStatus.ERROR,
            started_at=started_at,
            completed_at=completed,
            duration_s=round(time.monotonic() - t0, 4),
            dry_run=dry_run,
            detail={"exception_type": type(exc).__name__},
            errors=[f"{type(exc).__name__}: {exc}"[:500]],
            failure_class=FailureClass.CODE_ERROR,
        )

    if not isinstance(result, DrillResult):
        # Defensive: a drill that returns something else is a code bug.
        completed = datetime.now(timezone.utc).isoformat()
        return DrillResult(
            drill_name=spec.name,
            status=DrillStatus.ERROR,
            started_at=started_at,
            completed_at=completed,
            duration_s=round(time.monotonic() - t0, 4),
            dry_run=dry_run,
            detail={"returned_type": type(result).__name__},
            errors=[f"runner returned {type(result).__name__}, expected DrillResult"],
            failure_class=FailureClass.CODE_ERROR,
        )

    # If the runner produced a FAIL/ERROR but didn't set failure_class,
    # default to STRUCTURAL_FAIL / CODE_ERROR respectively. Pre-Q18
    # drills don't know about FailureClass; this back-compat shim
    # lets them coexist with v2 drills while migration proceeds.
    if result.status == DrillStatus.FAIL and result.failure_class is None:
        result.failure_class = FailureClass.STRUCTURAL_FAIL
    elif result.status == DrillStatus.ERROR and result.failure_class is None:
        result.failure_class = FailureClass.CODE_ERROR

    return result


def _apply_baseline_check(spec: DrillSpec, result: DrillResult) -> DrillResult:
    """If the drill produced an observation AND a baseline is ratified,
    compare. A regression upgrades the result to FAIL with
    failure_class=BASELINE_REGRESSION; a clean comparison leaves the
    result alone.

    Crucially: a passing observation against a ratified baseline can
    *downgrade* a STRUCTURAL_FAIL the drill produced. Example: the
    legacy vendor_independence drill returns FAIL because
    ``n_fallbacks < 2``, but the operator's ratified baseline says
    ``n_fallbacks >= 1`` is acceptable. The orchestrator promotes the
    FAIL to PASS — the operator's policy wins over the drill's
    built-in opinion.
    """
    if result.observation is None:
        return result
    baseline = bl.load(spec.name)
    if baseline is None:
        # No baseline → use the drill's pass/fail verdict as-is.
        return result
    from app.resilience_drills.baseline import Observation
    obs = Observation(
        drill_name=spec.name,
        observed_at=result.started_at,
        measurements=dict(result.observation),
    )
    comparison = bl.compare(obs, baseline)

    # Embed comparison verdict in result.detail for operator review.
    if "baseline_comparison" not in result.detail:
        result.detail["baseline_comparison"] = comparison.to_dict()

    if comparison.ok:
        # Operator-ratified baseline says this state is fine →
        # promote to PASS regardless of what the drill thought.
        if result.status in (DrillStatus.FAIL, DrillStatus.ERROR):
            logger.info(
                "drill %s: drill-internal verdict was %s but matches "
                "operator-ratified baseline → promoted to PASS",
                spec.name, result.status.value,
            )
        result.status = DrillStatus.PASS
        result.failure_class = None
        # Note: keep errors[] as-is — they're the drill's own
        # report. The operator can see them in /cp/drills.
    else:
        # Operator's baseline was violated. Upgrade to BASELINE_REGRESSION.
        if result.status == DrillStatus.PASS:
            result.status = DrillStatus.FAIL
        result.failure_class = FailureClass.BASELINE_REGRESSION
        # Include compact regression summary in errors[]
        reg_summaries = [
            f"{r.key}: {r.detail}"
            for r in comparison.regressions[:5]
        ]
        if comparison.missing_keys:
            reg_summaries.append(
                f"missing baseline keys: {comparison.missing_keys[:5]}"
            )
        if reg_summaries:
            result.errors = result.errors + reg_summaries
    return result


def _apply_state_transitions(spec: DrillSpec, state: st.DrillStateRecord,
                              result: DrillResult) -> None:
    """Feed the result back into the state machine + persist."""
    if result.status == DrillStatus.PASS:
        st.record_pass(state, cadence_days=spec.cadence_days)
    elif result.status in (DrillStatus.FAIL, DrillStatus.ERROR):
        fc = (
            result.failure_class.value
            if isinstance(result.failure_class, FailureClass)
            else (result.failure_class or "structural_fail")
        )
        summary = "; ".join(result.errors[:3]) if result.errors else ""
        traceback_text = result.detail.get("traceback", "") if result.detail else ""
        st.record_failure(
            state,
            cadence_days=spec.cadence_days,
            failure_class=fc,
            summary=summary,
            traceback_text=traceback_text,
        )
    # SKIPPED doesn't change state.
    st.save(state)


# ── Convenience: by-name invocation ─────────────────────────────────────


def invoke_drill_by_name(name: str, *, dry_run: bool = True,
                         triggered_by: str = "operator") -> DrillResult | None:
    """Operator/REST entry point: invoke by drill name.

    Returns None if the drill name isn't registered. Default
    ``triggered_by="operator"`` bypasses the backoff gate (operator
    explicitly asked); pass ``triggered_by="scheduler"`` to keep the
    gate engaged.
    """
    from app.resilience_drills.protocol import get_registry
    registry = get_registry()
    spec = registry.get(name)
    runner_fn = registry.runner_for(name)
    if spec is None or runner_fn is None:
        return None
    return invoke_drill(spec, runner_fn, dry_run=dry_run, triggered_by=triggered_by)
