"""backup_restore drill — wraps app.dr.boot_drill.run_drill.

PROGRAM §44.2 — Q6.2. The existing boot_drill.py is already
drill-shaped (never touches live workspace, writes structured
report, emits Signal + ledger events). This module is the
thin shim that integrates it with the Q6 registry + audit
infrastructure.

Risk LOW: boot_drill operates on a tarball + ephemeral target
directory; production state is untouched.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from app.resilience_drills.audit import (
    append_result,
    emit_landmark_for,
    last_result_for,
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


SPEC = DrillSpec(
    name="backup_restore",
    cadence_days=90,
    grace_days=30,
    risk=DrillRisk.LOW,
    description=(
        "Verify the most recent DR export imports cleanly to an "
        "ephemeral directory; check ChromaDB collections + ledger "
        "round-trip integrity."
    ),
    requires_master_switch="drill_backup_restore_enabled",
)


def run(*, dry_run: bool = True) -> DrillResult:
    """Run the backup-restore drill.

    ``dry_run`` parameter exists for the protocol uniformity but
    is functionally a no-op here — boot_drill is ALWAYS effectively
    a dry-run from the live system's perspective (it never touches
    live workspace).
    """
    started_dt = datetime.now(timezone.utc)
    started_at = started_dt.isoformat()
    t0 = time.monotonic()

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

    is_first_run = last_result_for(SPEC.name) is None
    detail: dict = {"dry_run": dry_run}
    errors: list[str] = []
    status = DrillStatus.PASS
    try:
        from app.dr.boot_drill import run_drill as _run_boot_drill
        report = _run_boot_drill(export_fresh=False, keep_target=False)
        detail["tarball"] = getattr(report, "tarball", None)
        detail["overall_ok"] = getattr(report, "overall_ok", False)
        detail["collections_checked"] = len(
            getattr(report, "collections", []) or [],
        )
        detail["fresh_export"] = getattr(report, "fresh_export", False)
        # Propagate any boot_drill errors into the drill result.
        report_errors = list(getattr(report, "errors", []) or [])
        if report_errors:
            errors.extend(report_errors[:10])  # cap for surface readability
        if not detail["overall_ok"]:
            status = DrillStatus.FAIL
    except Exception as exc:  # noqa: BLE001
        status = DrillStatus.ERROR
        errors.append(f"{type(exc).__name__}: {exc}")
        logger.debug("backup_restore drill raised", exc_info=True)

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
    emit_landmark_for(result, is_first_run=is_first_run)
    return result


# Register at import.
register(SPEC, run)
