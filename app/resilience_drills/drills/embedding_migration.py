"""embedding_migration drill — wraps app.memory.embedding_migration.dry_run.

PROGRAM §44.2 — Q6.2 (original). PROGRAM §57 — Q18 (v2 conversion).
Converted to the Q18 contract: the runner returns the bare
DrillResult; the orchestrator handles lock acquisition, audit
append, landmark emission, state machine, and baseline comparison.

Risk LOW: dry_run never touches real KBs, only the sandbox collection.
"""
from __future__ import annotations

import logging
import time
import traceback
from datetime import datetime, timezone
from typing import Any

from app.resilience_drills.protocol import (
    DrillResult,
    DrillRisk,
    DrillSpec,
    DrillStatus,
    FailureClass,
    register,
)

logger = logging.getLogger(__name__)


_DEFAULT_TARGET = {
    "target_provider": "ollama",
    "target_model": "mxbai-embed-large",
    "target_dim": 1024,
}


SPEC = DrillSpec(
    name="embedding_migration",
    cadence_days=90,
    grace_days=30,
    warmup_days=0,  # existing drill — no warmup
    risk=DrillRisk.LOW,
    description=(
        "Run the full embedding-migration state machine against an "
        "isolated sandbox collection; verify dual-write + shadow-read "
        "+ verifier all succeed."
    ),
    requires_master_switch="drill_embedding_migration_enabled",
)


def _step_field(step: Any, name: str) -> Any:
    """Extract a named field from a DryRunStep instance.

    Q18 bug fix (PROGRAM §57): the pre-Q18 drill assumed ``step`` was a
    dict and called ``step.get(...)``. ``DryRunStep`` is a dataclass,
    not a dict — that triggered ``AttributeError: 'DryRunStep' object
    has no attribute 'get'`` and was a chronic contributor to the
    2026-05-16 hot-loop. This shim accepts either shape.
    """
    if hasattr(step, name):
        return getattr(step, name)
    if isinstance(step, dict):
        return step.get(name)
    return None


def run(*, dry_run: bool = True, target: dict[str, Any] | None = None) -> DrillResult:
    """Run the embedding-migration drill. Returns a bare DrillResult;
    the orchestrator threads state, baseline, audit, and landmarks."""
    started_dt = datetime.now(timezone.utc)
    started_at = started_dt.isoformat()
    t0 = time.monotonic()
    target_spec = {**_DEFAULT_TARGET, **(target or {})}

    detail: dict[str, Any] = {"dry_run": True, "target": target_spec}
    errors: list[str] = []
    observation: dict[str, Any] = {
        "target_provider": target_spec.get("target_provider"),
        "target_model": target_spec.get("target_model"),
        "target_dim": target_spec.get("target_dim"),
    }
    status = DrillStatus.PASS
    failure_class: FailureClass | None = None

    try:
        from app.memory.embedding_migration.dry_run import run_dry_run
        report = run_dry_run(
            target_provider=target_spec["target_provider"],
            target_model=target_spec["target_model"],
            target_dim=target_spec["target_dim"],
        )
        steps = list(getattr(report, "steps", None) or [])
        # Q18 bug fix: use _step_field instead of s.get
        detail["steps"] = [
            {"name": _step_field(s, "name"), "ok": _step_field(s, "ok")}
            for s in steps
        ]
        n_ok = sum(1 for s in steps if _step_field(s, "ok"))
        detail["n_steps_ok"] = n_ok
        detail["n_steps_total"] = len(steps)
        observation["n_steps_ok"] = n_ok
        observation["n_steps_total"] = len(steps)
        failed_steps = [
            _step_field(s, "name") for s in steps if not _step_field(s, "ok")
        ]
        if failed_steps:
            status = DrillStatus.FAIL
            failure_class = FailureClass.STRUCTURAL_FAIL
            errors.append(f"failed steps: {', '.join(filter(None, failed_steps[:5]))}")
            observation["failed_steps"] = list(failed_steps)
    except Exception as exc:  # noqa: BLE001
        status = DrillStatus.ERROR
        failure_class = FailureClass.CODE_ERROR
        errors.append(f"{type(exc).__name__}: {exc}")
        detail["traceback"] = traceback.format_exc(limit=10)
        logger.debug("embedding_migration drill raised", exc_info=True)

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
