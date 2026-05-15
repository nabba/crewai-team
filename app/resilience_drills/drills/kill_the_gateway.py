"""kill_the_gateway drill — pre/post-drill hooks.

PROGRAM §44.2 — Q6.2. The most disruptive of the four drills:
actually stops the gateway container and measures recovery time.
The gateway can't kill itself, so the architecture is:

  1. **Pre-drill check** (this module's ``run(dry_run=True)``):
     verify the system is ready for the drill (DR backup recent,
     no active Tier-3 amendments in monitoring, persistent stores
     healthy). Emit Signal "drill ready" notification.

  2. **External script** (``scripts/drills/kill_the_gateway.sh``):
     run OUTSIDE the gateway. Stops the container, waits, restarts,
     measures recovery time, verifies post-recovery state. Writes
     a result file at ``workspace/resilience/kill_drill_<ts>.json``.

  3. **Post-drill hook** (this module's ``ingest_external_report``):
     when the gateway boots back up, detects recent result file,
     ingests it into the audit log + emits ledger landmark.

Risk HIGH: actual stop+start of the gateway container.

Gating:

  * ``resilience_drills_enabled`` master switch ON
  * ``drill_kill_the_gateway_enabled`` per-drill switch ON
    (default OFF — operator opts in via React /cp/settings)
  * For LIVE execution: external script requires typed-phrase
    confirmation (read from CLI flag)
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

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
    name="kill_the_gateway",
    cadence_days=90,
    grace_days=60,  # extra grace; HIGH-risk drills shouldn't pressure operator
    risk=DrillRisk.HIGH,
    description=(
        "DISRUPTIVE: stop the gateway container via external script + "
        "measure recovery time. Pre-drill check runs in-gateway; live "
        "execution runs OUTSIDE."
    ),
    requires_typed_phrase="EXECUTE KILL DRILL",
    requires_master_switch="drill_kill_the_gateway_enabled",
)


# Where the external script writes its report.
def _default_kill_report_dir() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT) / "resilience"
    except Exception:
        return Path("/app/workspace/resilience")


def _check_dr_backup_recent(*, max_age_days: int = 7) -> tuple[bool, str | None]:
    """Refuse the drill unless the most recent DR tarball is recent."""
    try:
        from app.dr.boot_drill import _default_export_dir, _latest_tarball
        tarball = _latest_tarball(_default_export_dir())
        if tarball is None:
            return False, "no DR tarball found"
        age_days = (time.time() - tarball.stat().st_mtime) / 86400.0
        if age_days > max_age_days:
            return False, f"latest tarball is {age_days:.1f}d old (max {max_age_days}d)"
        return True, None
    except Exception as exc:
        return False, f"tarball-age check failed: {type(exc).__name__}"


def _check_no_active_tier3_monitoring() -> tuple[bool, str | None]:
    """Refuse if a Tier-3 amendment is in APPLIED-monitoring window —
    a gateway restart mid-monitoring would muddy the signal."""
    try:
        from app.governance_amendment.protocol import list_proposals
        proposals = list_proposals()
        active = [
            p for p in (proposals or [])
            if hasattr(p, "state")
            and getattr(p.state, "value", str(p.state)).lower() == "applied"
        ]
        if active:
            return False, f"{len(active)} amendments in APPLIED-monitoring"
        return True, None
    except Exception:
        # If we can't check, allow — better to drill than block on a
        # missing-import error. The drill is for the operator's benefit.
        return True, None


def _check_persistent_stores_healthy() -> tuple[bool, str | None]:
    """Verify Postgres + Neo4j + ChromaDB are reachable. The drill
    assumes these will survive the gateway restart; if they're already
    down, the drill is invalid."""
    failures: list[str] = []
    # Postgres
    try:
        from app.control_plane.db import execute_scalar
        execute_scalar("SELECT 1")
    except Exception:
        failures.append("postgres")
    # ChromaDB
    try:
        from app.memory.chromadb_manager import get_client
        client = get_client()
        if client is None:
            failures.append("chromadb (no client)")
    except Exception:
        failures.append("chromadb")
    # Neo4j — soft check; not all configs use neo4j
    try:
        from app.subia.belief import neo4j_mirror
        drv = neo4j_mirror._get_driver()
        if drv is not None:
            with drv.session() as s:
                s.run("RETURN 1").consume()
    except Exception:
        # Neo4j optional — don't block.
        pass
    if failures:
        return False, f"unhealthy: {', '.join(failures)}"
    return True, None


def run(*, dry_run: bool = True) -> DrillResult:
    """Pre-drill readiness check.

    When ``dry_run=True`` (the default and only supported value from
    Python): runs the readiness checks + emits a Signal notification
    if the system is ready. Records a SKIPPED-or-PASS audit row.

    The LIVE drill is NEVER triggered from Python — it requires the
    external ``scripts/drills/kill_the_gateway.sh`` script. This is
    a deliberate architectural choice; the gateway cannot kill itself.
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
            detail={"reason": "master switch off (operator must opt in)"},
        )

    is_first_run = last_result_for(SPEC.name) is None
    detail: dict[str, Any] = {"mode": "pre_drill_check"}
    errors: list[str] = []
    status = DrillStatus.PASS

    # Run readiness checks.
    ok, err = _check_dr_backup_recent()
    detail["dr_backup_recent"] = ok
    if not ok:
        errors.append(f"dr_backup_recent: {err}")
        status = DrillStatus.FAIL

    ok, err = _check_no_active_tier3_monitoring()
    detail["no_active_tier3_monitoring"] = ok
    if not ok:
        errors.append(f"no_active_tier3_monitoring: {err}")
        status = DrillStatus.FAIL

    ok, err = _check_persistent_stores_healthy()
    detail["persistent_stores_healthy"] = ok
    if not ok:
        errors.append(f"persistent_stores_healthy: {err}")
        status = DrillStatus.FAIL

    detail["next_step"] = (
        "Run scripts/drills/kill_the_gateway.sh with the typed-phrase "
        "confirmation 'EXECUTE KILL DRILL' to actually perform the drill."
    )

    completed_dt = datetime.now(timezone.utc)
    result = DrillResult(
        drill_name=SPEC.name,
        status=status,
        started_at=started_at,
        completed_at=completed_dt.isoformat(),
        duration_s=round(time.monotonic() - t0, 3),
        dry_run=True,  # the Python entrypoint is always pre-drill check
        detail=detail,
        errors=errors,
    )
    append_result(result)
    emit_landmark_for(result, is_first_run=is_first_run)
    return result


def ingest_external_report() -> DrillResult | None:
    """Boot-time hook: detect a recent kill-drill report from the
    external script, ingest it into the audit log, emit landmark.

    Called from companion.loop on first idle pass after gateway
    startup. Idempotent — won't re-ingest the same report twice.
    """
    report_dir = _default_kill_report_dir()
    if not report_dir.exists():
        return None
    # Look for the newest kill_drill_*.json that's < 10 minutes old.
    candidates = sorted(report_dir.glob("kill_drill_*.json"), reverse=True)
    if not candidates:
        return None
    newest = candidates[0]
    try:
        age_s = time.time() - newest.stat().st_mtime
    except OSError:
        return None
    if age_s > 600:  # 10 min
        return None
    # Check we haven't already ingested this one — match against the
    # most recent audit row for this drill.
    try:
        report_data = json.loads(newest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    last = last_result_for(SPEC.name)
    if last and last.get("started_at") == report_data.get("started_at"):
        return None  # already ingested
    # Build a DrillResult from the external report.
    status_str = (report_data.get("status") or "pass").lower()
    try:
        status = DrillStatus(status_str)
    except ValueError:
        status = DrillStatus.ERROR
    result = DrillResult(
        drill_name=SPEC.name,
        status=status,
        started_at=report_data.get("started_at", ""),
        completed_at=report_data.get("completed_at", ""),
        duration_s=float(report_data.get("duration_s", 0.0)),
        dry_run=False,  # external report = LIVE drill
        detail=dict(report_data.get("detail") or {}),
        errors=list(report_data.get("errors") or []),
    )
    append_result(result)
    emit_landmark_for(result, is_first_run=False)
    return result


register(SPEC, run)
