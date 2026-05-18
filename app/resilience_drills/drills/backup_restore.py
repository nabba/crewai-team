"""backup_restore drill — wraps app.dr.boot_drill.run_drill.

PROGRAM §44.2 — Q6.2 (original). PROGRAM §57 — Q18 (v2 conversion).
Returns a bare DrillResult; the orchestrator handles lock + audit +
landmark + state.

Risk LOW: boot_drill operates on a tarball + ephemeral target
directory; production state is untouched.
"""
from __future__ import annotations

import logging
import time
import traceback
from datetime import datetime, timezone

from app.resilience_drills.protocol import (
    DrillResult,
    DrillRisk,
    DrillSpec,
    DrillStatus,
    FailureClass,
    register,
)

logger = logging.getLogger(__name__)


SPEC = DrillSpec(
    name="backup_restore",
    cadence_days=90,
    grace_days=30,
    warmup_days=0,
    risk=DrillRisk.LOW,
    description=(
        "Verify the most recent DR export imports cleanly to an "
        "ephemeral directory; check ChromaDB collections + ledger "
        "round-trip integrity."
    ),
    requires_master_switch="drill_backup_restore_enabled",
)


def run(*, dry_run: bool = True) -> DrillResult:
    """Run the backup-restore drill. Q18 runner contract: return bare
    DrillResult."""
    started_dt = datetime.now(timezone.utc)
    started_at = started_dt.isoformat()
    t0 = time.monotonic()

    detail: dict = {"dry_run": dry_run}
    errors: list[str] = []
    status = DrillStatus.PASS
    failure_class: FailureClass | None = None
    observation: dict = {}

    try:
        from app.dr.boot_drill import run_drill as _run_boot_drill
        report = _run_boot_drill(export_fresh=False, keep_target=False)
        detail["tarball"] = getattr(report, "tarball", None)
        detail["overall_ok"] = getattr(report, "overall_ok", False)
        detail["collections_checked"] = len(
            getattr(report, "collections", []) or [],
        )
        detail["fresh_export"] = getattr(report, "fresh_export", False)
        observation["overall_ok"] = detail["overall_ok"]
        observation["collections_checked"] = detail["collections_checked"]
        report_errors = list(getattr(report, "errors", []) or [])
        if report_errors:
            errors.extend(report_errors[:10])
        if not detail["overall_ok"]:
            status = DrillStatus.FAIL
            failure_class = FailureClass.STRUCTURAL_FAIL
    except Exception as exc:  # noqa: BLE001
        status = DrillStatus.ERROR
        failure_class = FailureClass.CODE_ERROR
        errors.append(f"{type(exc).__name__}: {exc}")
        detail["traceback"] = traceback.format_exc(limit=10)
        logger.debug("backup_restore drill raised", exc_info=True)

    completed_dt = datetime.now(timezone.utc)
    return DrillResult(
        drill_name=SPEC.name,
        status=status,
        started_at=started_at,
        completed_at=completed_dt.isoformat(),
        duration_s=round(time.monotonic() - t0, 3),
        dry_run=dry_run,
        detail=detail,
        errors=errors,
        failure_class=failure_class,
        observation=observation,
    )


register(SPEC, run)
