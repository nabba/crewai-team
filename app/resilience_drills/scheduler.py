"""Drill scheduler — cadence tracking + due-notifications.

PROGRAM §44.2 — Q6.2. Runs as an idle job. For each registered
drill, computes "days since last successful run" against the
drill's cadence. When a drill is past-due, emits an opaque Signal
notification AND optionally auto-runs LOW-risk drills.

Architecture decision: LOW-risk drills auto-run on schedule (they
never disrupt prod). MEDIUM-risk drills auto-run on schedule too
(the embedding-migration dry_run is bounded by sandbox isolation).
HIGH-risk drills (kill_the_gateway) DO NOT auto-run — operator
must explicitly run the external script after seeing the "due"
notification.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from app.resilience_drills.audit import days_since_last_success
from app.resilience_drills.protocol import (
    DrillRisk,
    DrillStatus,
    drill_enabled,
    get_registry,
    master_enabled,
)

logger = logging.getLogger(__name__)


def _notify_drill_due(name: str, days_since: float | None, cadence_days: int) -> None:
    """Best-effort Signal notification when a drill is due. Failure-
    isolated. Uses the existing notify infrastructure."""
    try:
        from app.notify import notify
        if days_since is None:
            body = (
                f"Resilience drill {name!r} has never run "
                f"(target cadence: {cadence_days}d)."
            )
        else:
            body = (
                f"Resilience drill {name!r} is due — last successful "
                f"run was {days_since:.0f}d ago "
                f"(target cadence: {cadence_days}d)."
            )
        notify(
            title="🛡 Resilience drill due",
            body=body,
            url="/cp/drills",
            topic=f"resilience_drill_due:{name}",
            critical=False,
            arbitrate=True,
        )
    except Exception:
        logger.debug("scheduler: drill-due notify failed", exc_info=True)


def run_once() -> dict[str, Any]:
    """One scheduler pass. For each registered drill:

      * If past-due, emit a "due" notification.
      * If LOW or MEDIUM risk AND auto-runnable, invoke the drill.
      * HIGH-risk drills are NEVER auto-run — operator runs the
        external script.

    Returns a summary suitable for idle-job logging.
    """
    summary: dict[str, Any] = {
        "checked": 0,
        "auto_ran": 0,
        "due_notified": 0,
        "skipped": 0,
        "errors": 0,
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
        if not drill_enabled(spec):
            summary["skipped"] += 1
            continue
        days = days_since_last_success(spec.name)
        past_due_threshold = spec.cadence_days
        if days is not None and days < past_due_threshold:
            # Still within cadence — nothing to do.
            continue

        # The drill IS due. Emit a notification (always).
        _notify_drill_due(spec.name, days, spec.cadence_days)
        summary["due_notified"] += 1

        # HIGH-risk drills: never auto-run; operator must run script.
        if spec.risk == DrillRisk.HIGH:
            continue

        # LOW + MEDIUM risk: auto-run.
        runner = registry.runner_for(spec.name)
        if runner is None:
            continue
        try:
            result = runner(dry_run=True)
            summary["auto_ran"] += 1
            if getattr(result, "status", None) == DrillStatus.PASS:
                pass  # routine pass; nothing further
            elif getattr(result, "status", None) in (DrillStatus.FAIL, DrillStatus.ERROR):
                summary["errors"] += 1
        except Exception:
            logger.debug("scheduler: auto-run failed for %s", spec.name, exc_info=True)
            summary["errors"] += 1

    return summary
