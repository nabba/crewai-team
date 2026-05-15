"""embedding_migration drill — wraps app.memory.embedding_migration.dry_run.

PROGRAM §44.2 — Q6.2. The existing dry_run.py is already drill-shaped
(operates on isolated _dry_run_sandbox_collection; real HTTP calls
to embedding model but no production-KB writes). This module is the
Q6 shim.

Risk LOW: dry_run never touches real KBs, only the sandbox collection.

Configuration: a default target spec is stored in the drill detail.
Operator can override per-run via the REST endpoint by passing
``{"target_provider": "...", "target_model": "...", "target_dim": ...}``.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

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
    drill_enabled,
    register,
)

logger = logging.getLogger(__name__)


# Default target spec. Mirrors a realistic candidate migration the
# operator might consider; quarterly drill verifies the procedure
# still works against this candidate.
_DEFAULT_TARGET = {
    "target_provider": "ollama",
    "target_model": "mxbai-embed-large",
    "target_dim": 1024,
}


SPEC = DrillSpec(
    name="embedding_migration",
    cadence_days=90,
    grace_days=30,
    risk=DrillRisk.LOW,
    description=(
        "Run the full embedding-migration state machine against an "
        "isolated sandbox collection; verify dual-write + shadow-read "
        "+ verifier all succeed."
    ),
    requires_master_switch="drill_embedding_migration_enabled",
)


def run(*, dry_run: bool = True, target: dict[str, Any] | None = None) -> DrillResult:
    """Run the embedding-migration drill.

    ``dry_run`` is informational here — dry_run.py is itself a
    drill (never touches production KBs), so the parameter is
    always functionally True.
    """
    started_dt = datetime.now(timezone.utc)
    started_at = started_dt.isoformat()
    t0 = time.monotonic()
    target_spec = {**_DEFAULT_TARGET, **(target or {})}

    if not drill_enabled(SPEC):
        return DrillResult(
            drill_name=SPEC.name,
            status=DrillStatus.SKIPPED,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc).isoformat(),
            duration_s=time.monotonic() - t0,
            dry_run=dry_run,
            detail={"reason": "master switch off"},
        )

    # Q6.4 P1#3 — in-flight lock.
    if not acquire_drill_lock(SPEC.name):
        return DrillResult(
            drill_name=SPEC.name,
            status=DrillStatus.SKIPPED,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc).isoformat(),
            duration_s=time.monotonic() - t0,
            dry_run=dry_run,
            detail={"reason": "drill already in-flight"},
        )

    try:
        # Q6.4 P0#1 + P1#4 — prior-state snapshot BEFORE append.
        prior_any = last_result_for(SPEC.name)
        is_first_run = last_successful_for(SPEC.name) is None
        prior_status = (prior_any or {}).get("status") if prior_any else None

        detail: dict[str, Any] = {"dry_run": True, "target": target_spec}
        errors: list[str] = []
        status = DrillStatus.PASS
        try:
            from app.memory.embedding_migration.dry_run import run_dry_run
            report = run_dry_run(
                target_provider=target_spec["target_provider"],
                target_model=target_spec["target_model"],
                target_dim=target_spec["target_dim"],
            )
            steps = list(getattr(report, "steps", None) or [])
            detail["steps"] = [
                {"name": s.get("name"), "ok": s.get("ok")}
                for s in steps
            ]
            detail["n_steps_ok"] = sum(1 for s in steps if s.get("ok"))
            detail["n_steps_total"] = len(steps)
            failed_steps = [
                s.get("name") for s in steps if not s.get("ok")
            ]
            if failed_steps:
                status = DrillStatus.FAIL
                errors.append(f"failed steps: {', '.join(failed_steps[:5])}")
        except Exception as exc:  # noqa: BLE001
            status = DrillStatus.ERROR
            errors.append(f"{type(exc).__name__}: {exc}")
            logger.debug("embedding_migration drill raised", exc_info=True)

        completed_dt = datetime.now(timezone.utc)
        result = DrillResult(
            drill_name=SPEC.name,
            status=status,
            started_at=started_at,
            completed_at=completed_dt.isoformat(),
            duration_s=round(time.monotonic() - t0, 3),
            dry_run=dry_run,
            detail=detail,
            errors=errors,
        )
        append_result(result)
        emit_landmark_for(
            result,
            is_first_run=is_first_run,
            prior_status=prior_status,
        )
        return result
    finally:
        release_drill_lock(SPEC.name)


register(SPEC, run)
