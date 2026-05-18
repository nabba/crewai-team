"""Drill scheduler — cadence + state-machine driven invocation.

PROGRAM §57 — Q18 resilience-drill v2. Rewrites the §44 scheduler
to be state-driven. The legacy "if past-due, auto-run" logic that
produced the 2026-05-16 hot-loop (3580 drill rows in 48h) is gone;
every invocation goes through :func:`app.resilience_drills.runner.
invoke_drill`, which respects the state machine's backoff /
quarantine / mute decisions.

Responsibilities of this module:

  * Walk the drill registry on each idle pass.
  * For each registered drill, invoke via the orchestrator with
    ``triggered_by="scheduler"`` (so backoff gate is enforced).
  * For HIGH-risk drills, NEVER auto-invoke — only emit
    notifications when due. Operator runs the external script.
  * Surface notifications on FAIL / ERROR / BASELINE_REGRESSION
    (suppressed during WARMING_UP).
"""
from __future__ import annotations

import logging
from typing import Any

from app.resilience_drills import state as st
from app.resilience_drills.protocol import (
    DrillRisk,
    DrillStatus,
    FailureClass,
    drill_enabled,
    get_registry,
    master_enabled,
)
from app.resilience_drills.runner import invoke_drill

logger = logging.getLogger(__name__)


def _notify_drill_failed(name: str, status: str, failure_class: str,
                          errors: list[str]) -> None:
    """Best-effort Signal notification on FAIL/ERROR. Topic-keyed so
    the notify arbiter dedupes repeated failures of the same drill
    within 24 h.

    Q18 extension: includes ``failure_class`` so the operator can
    distinguish a drill bug (CODE_ERROR) from a real-world finding
    (STRUCTURAL_FAIL) from a baseline regression (the actionable
    signal)."""
    try:
        from app.notify import notify
        err_summary = "; ".join(errors[:3])[:200] if errors else "(no error detail)"
        title_prefix = {
            "code_error": "🐛",
            "structural_fail": "❌",
            "transient_fail": "⚠️",
            "baseline_regression": "📉",
        }.get(failure_class, "❌")
        notify(
            title=f"{title_prefix} Resilience drill {failure_class}: {name}",
            body=(
                f"Drill {name!r} returned status={status} "
                f"failure_class={failure_class}. "
                f"Errors: {err_summary}. See /cp/drills/{name} for state + baseline."
            ),
            url=f"/cp/drills/{name}",
            topic=f"resilience_drill_failed:{name}:{failure_class}",
            critical=False,
            arbitrate=True,
        )
    except Exception:
        logger.debug("scheduler: drill-failed notify failed", exc_info=True)


def _notify_drill_quarantined(name: str, reason: str) -> None:
    """Loud one-shot Signal when a drill transitions to QUARANTINED.

    Topic-keyed without failure_class suffix so subsequent
    re-quarantines after operator un-quarantine fire fresh alerts."""
    try:
        from app.notify import notify
        notify(
            title=f"🚨 Drill quarantined: {name}",
            body=(
                f"Drill {name!r} has been quarantined after 3 consecutive "
                f"code errors. The scheduler will NOT auto-run it. Visit "
                f"/cp/drills/{name} to inspect the traceback and unquarantine."
                f"\n\n{reason[:300]}"
            ),
            url=f"/cp/drills/{name}",
            topic=f"resilience_drill_quarantined:{name}",
            critical=False,
            arbitrate=True,
        )
    except Exception:
        logger.debug("scheduler: quarantine notify failed", exc_info=True)


def run_once() -> dict[str, Any]:
    """One scheduler pass. For each registered drill, invoke via the
    orchestrator. The orchestrator respects state — drills in
    backoff/muted/quarantined return SKIPPED, no runner invocation.

    Returns a summary suitable for idle-job logging.
    """
    summary: dict[str, Any] = {
        "checked": 0,
        "ran": 0,
        "skipped": 0,
        "passed": 0,
        "failed": 0,
        "errored": 0,
        "regressions": 0,
        "quarantined_now": 0,
    }
    if not master_enabled():
        summary["skipped"] = 1
        summary["reason"] = "master switch off"
        return summary

    registry = get_registry()

    # First-time hook: ingest any kill-the-gateway external report
    # written while the gateway was rebooting.
    try:
        from app.resilience_drills.drills.kill_the_gateway import (
            ingest_external_report,
        )
        ingested = ingest_external_report()
        if ingested is not None:
            summary["external_kill_drill_ingested"] = ingested.status.value
    except Exception:
        logger.debug("scheduler: kill-drill ingest failed", exc_info=True)

    for spec in registry.list_specs():
        summary["checked"] += 1

        # HIGH-risk drills are never auto-invoked — emit a "due"
        # notification when past-cadence and rely on the operator to
        # run the external script.
        if spec.risk == DrillRisk.HIGH:
            _maybe_notify_high_risk_due(spec)
            summary["skipped"] += 1
            continue

        runner_fn = registry.runner_for(spec.name)
        if runner_fn is None:
            summary["skipped"] += 1
            continue

        # Pre-invocation: snapshot state so we can detect a
        # quarantine transition.
        pre_state = st.load_or_initialize(spec.name, warmup_days=spec.warmup_days)
        pre_state_value = pre_state.state

        result = invoke_drill(
            spec, runner_fn,
            dry_run=True,
            triggered_by="scheduler",
        )

        if result.status == DrillStatus.SKIPPED:
            summary["skipped"] += 1
            continue

        summary["ran"] += 1

        if result.status == DrillStatus.PASS:
            summary["passed"] += 1
        elif result.status == DrillStatus.FAIL:
            summary["failed"] += 1
            fc = (
                result.failure_class.value
                if isinstance(result.failure_class, FailureClass)
                else str(result.failure_class or "structural_fail")
            )
            if fc == "baseline_regression":
                summary["regressions"] += 1
            # Suppress alerts during warmup (observational only).
            post_state = st.load(spec.name)
            if post_state is None or not st.is_warming_up(post_state):
                _notify_drill_failed(
                    spec.name, result.status.value, fc, list(result.errors or []),
                )
        elif result.status == DrillStatus.ERROR:
            summary["errored"] += 1
            post_state = st.load(spec.name)
            if post_state is None or not st.is_warming_up(post_state):
                _notify_drill_failed(
                    spec.name, result.status.value, "code_error",
                    list(result.errors or []),
                )

        # Detect transitions to QUARANTINED for a loud one-shot alert.
        post_state = st.load(spec.name)
        if (post_state is not None
                and post_state.state == st.DrillState.QUARANTINED
                and pre_state_value != st.DrillState.QUARANTINED):
            summary["quarantined_now"] += 1
            _notify_drill_quarantined(spec.name, post_state.quarantined_reason)

    return summary


def _maybe_notify_high_risk_due(spec) -> None:
    """For HIGH-risk drills, emit a 'due' notification when past
    cadence. Never auto-runs."""
    if not drill_enabled(spec):
        return
    try:
        from app.resilience_drills.audit import days_since_last_success
        days = days_since_last_success(spec.name)
    except Exception:
        return
    if days is None:
        # Never run — emit one-shot
        body = (
            f"HIGH-risk resilience drill {spec.name!r} has never run. "
            f"Run the external script when ready (typed-phrase required)."
        )
    else:
        if days < spec.cadence_days:
            return
        body = (
            f"HIGH-risk resilience drill {spec.name!r} is due — last "
            f"successful run was {days:.0f}d ago (cadence {spec.cadence_days}d). "
            f"Run the external script when ready (typed-phrase required)."
        )
    try:
        from app.notify import notify
        notify(
            title="🛡 Resilience drill due (HIGH risk)",
            body=body,
            url=f"/cp/drills/{spec.name}",
            topic=f"resilience_drill_due:{spec.name}",
            critical=False,
            arbitrate=True,
        )
    except Exception:
        logger.debug("scheduler: high-risk due-notify failed", exc_info=True)
