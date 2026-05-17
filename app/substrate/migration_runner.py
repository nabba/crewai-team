"""
migration_runner — background-thread orchestrator for React-driven migrations.

Productization plan WP D Phase 5a (2026-05-17). The CLI version of
``run_migration_live`` runs synchronously in the operator's terminal
(30+ min wallclock). For a React button, the gateway can't block its
event loop that long. This module wraps the orchestrator in a worker
thread, persists progress to ``workspace/migrations/<run_id>/run_state.json``,
and gives REST endpoints a polling interface.

Key design choices:

  * **One run per process at a time.** Multi-migration is a misfeature
    for a single-operator system; a module-level lock prevents two
    React clicks from racing.

  * **Status is file-persisted, not memory.** A gateway restart
    mid-migration leaves the orchestrator dead (it's a foreground
    thread) but the report-on-disk survives so the operator can see
    where it stopped + run ``terraform destroy`` to recover.

  * **Cancel is cooperative.** The orchestrator checks a cancel flag
    between steps. Mid-step cancellation (e.g., during a 15-min
    terraform apply) isn't supported — operator can ``terraform destroy``
    out-of-band if they really need to abort.

  * **No new safety gates.** Every gate evaluate_live_gates already
    enforces is inherited; the runner just transports its outcomes.
    The REST layer (migrate_api.py) adds API-level gates on top.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

logger = logging.getLogger(__name__)

RunStatus = Literal[
    "queued",
    "preparing",
    "preflight_failed",
    "running",
    "succeeded",
    "failed",
    "cancelled",
]


@dataclass
class AsyncRunRecord:
    """Persisted snapshot of an async migration run.

    The orchestrator updates this in-memory + flushes to
    ``workspace/migrations/<run_id>/run_state.json`` after every state
    transition. React polls the file via the REST endpoint.
    """
    run_id: str
    status: RunStatus = "queued"
    target: str = "gcp"
    tier: str = "cheapest"
    region: str = ""
    project_id: str = ""
    active_account: str = ""
    started_at: str = ""
    updated_at: str = ""
    completed_at: str = ""
    # Step labels match the orchestrator's step names so React UI can
    # render a consistent progress bar.
    current_step: str = ""
    progress_pct: int = 0
    detail: str = ""
    # The final MigrationRun report.json path (only after succeeded/failed)
    report_path: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Persistence ────────────────────────────────────────────────────


def _run_state_path(run_id: str) -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT) / "migrations" / run_id / "run_state.json"
    except Exception:
        return Path("/tmp/botarmy_migrations") / run_id / "run_state.json"


def _persist(record: AsyncRunRecord) -> None:
    """Atomic write of the run record. Failure-isolated."""
    try:
        path = _run_state_path(record.run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        record.updated_at = datetime.now(timezone.utc).isoformat()
        # Atomic via .tmp + rename
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record.to_dict(), indent=2, default=str))
        tmp.replace(path)
    except Exception:
        logger.debug("migration_runner: persist failed (non-fatal)", exc_info=True)


def load_run_record(run_id: str) -> AsyncRunRecord | None:
    """Read the persisted run state. Returns None if absent or corrupt."""
    path = _run_state_path(run_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return AsyncRunRecord(**{
            k: v for k, v in data.items()
            if k in AsyncRunRecord.__dataclass_fields__
        })
    except Exception:
        logger.debug("migration_runner: load failed", exc_info=True)
        return None


def list_recent_runs(limit: int = 20) -> list[AsyncRunRecord]:
    """Walk workspace/migrations/*/run_state.json, newest first."""
    try:
        from app.paths import WORKSPACE_ROOT
        root = Path(WORKSPACE_ROOT) / "migrations"
    except Exception:
        root = Path("/tmp/botarmy_migrations")
    if not root.exists():
        return []
    records: list[AsyncRunRecord] = []
    for run_dir in root.iterdir():
        if not run_dir.is_dir():
            continue
        state = run_dir / "run_state.json"
        if not state.exists():
            continue
        try:
            data = json.loads(state.read_text())
            rec = AsyncRunRecord(**{
                k: v for k, v in data.items()
                if k in AsyncRunRecord.__dataclass_fields__
            })
            records.append(rec)
        except Exception:
            continue
    # Newest first by started_at
    records.sort(key=lambda r: r.started_at, reverse=True)
    return records[:limit]


# ── Singleton manager ──────────────────────────────────────────────


class _RunnerSingleton:
    """Module-level state — at most one active migration per process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active_run_id: str | None = None
        self._active_thread: threading.Thread | None = None
        self._cancel_flag = threading.Event()

    def active_run_id(self) -> str | None:
        with self._lock:
            return self._active_run_id

    def request_cancel(self) -> bool:
        """Cooperative cancel — the orchestrator checks this between steps."""
        with self._lock:
            if self._active_run_id is None:
                return False
            self._cancel_flag.set()
            return True

    def _start(self, run_id: str, thread: threading.Thread) -> None:
        with self._lock:
            self._active_run_id = run_id
            self._active_thread = thread
            self._cancel_flag.clear()

    def _finish(self) -> None:
        with self._lock:
            self._active_run_id = None
            self._active_thread = None
            self._cancel_flag.clear()

    def cancel_requested(self) -> bool:
        return self._cancel_flag.is_set()


_RUNNER = _RunnerSingleton()


# ── Public scheduler ───────────────────────────────────────────────


class RunnerBusyError(Exception):
    """Raised when a second start() is attempted while another run is active."""


def active_run_id() -> str | None:
    """Return the run_id of the currently in-flight migration, or None."""
    return _RUNNER.active_run_id()


def cancel_active() -> bool:
    """Request cancellation of the in-flight run. Returns True iff a
    run was in flight to cancel. Cancellation is cooperative —
    the orchestrator checks between steps."""
    return _RUNNER.request_cancel()


def start_async_migration(
    *,
    target: Literal["gcp", "aws"] = "gcp",
    tier: Literal["cheapest", "prod"] = "cheapest",
    region: str | None = None,
    project_id: str,
    active_account: str,
    confirm_phrase: str,
    budget_cap_usd: float = 200.0,
    execute_subprocess: bool = False,
) -> AsyncRunRecord:
    """Kick off a migration in a background thread.

    Caller (REST handler) gets an immediate AsyncRunRecord with
    status='queued'; the React side polls /api/cp/migrate/runs/<run_id>
    for progress.

    Raises RunnerBusyError if a migration is already in flight.
    """
    if active_run_id() is not None:
        raise RunnerBusyError(
            f"another migration ({active_run_id()}) is already in flight"
        )
    if region is None:
        region = "europe-north1" if target == "gcp" else "eu-north-1"

    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_" + uuid.uuid4().hex[:8]
    )
    record = AsyncRunRecord(
        run_id=run_id,
        status="queued",
        target=target,
        tier=tier,
        region=region,
        project_id=project_id,
        active_account=active_account,
        started_at=datetime.now(timezone.utc).isoformat(),
        current_step="queued",
        progress_pct=0,
    )
    _persist(record)

    def _worker() -> None:
        try:
            _run_pipeline(
                record=record,
                confirm_phrase=confirm_phrase,
                budget_cap_usd=budget_cap_usd,
                execute_subprocess=execute_subprocess,
            )
        except Exception as exc:
            logger.exception("migration_runner: worker crashed")
            record.status = "failed"
            record.error = f"worker crash: {type(exc).__name__}: {exc}"
            record.completed_at = datetime.now(timezone.utc).isoformat()
            _persist(record)
        finally:
            _RUNNER._finish()

    thread = threading.Thread(
        target=_worker,
        name=f"migration-{run_id[:12]}",
        daemon=True,
    )
    _RUNNER._start(run_id, thread)
    thread.start()
    return record


# ── Pipeline ───────────────────────────────────────────────────────


# Step progress markers (0-100). React shows these as a progress bar.
_PROGRESS = {
    "queued":       0,
    "preparing":    5,
    "cloud_prep":  10,
    "preflight":   20,
    "provision":   25,   # bulk of the time
    "transfer":    80,
    "restore":     85,
    "verify":      95,
    "done":       100,
}


def _step_progress(record: AsyncRunRecord, step: str, detail: str = "") -> None:
    record.current_step = step
    record.progress_pct = _PROGRESS.get(step, record.progress_pct)
    if detail:
        record.detail = detail[:300]
    _persist(record)


def _run_pipeline(
    *,
    record: AsyncRunRecord,
    confirm_phrase: str,
    budget_cap_usd: float,
    execute_subprocess: bool,
) -> None:
    """The actual migration sequence. Updates record in place.

    Stages:
      1. preparing       — verify run_state, set up env
      2. cloud_prep      — auto-unblock (set account, enable APIs, mint token)
      3. preflight       — cloud_doctor refresh (must be OK)
      4. provision       — terraform apply
      5. transfer        — bundle upload
      6. restore         — kubectl exec import
      7. verify          — boot_drill + integrity
      8. done            — mark succeeded
    """
    # 1. Preparing
    record.status = "preparing"
    _step_progress(record, "preparing", "validating inputs")

    if _RUNNER.cancel_requested():
        record.status = "cancelled"
        record.completed_at = datetime.now(timezone.utc).isoformat()
        _persist(record)
        return

    # 2. Cloud prep — auto-unblock
    from app.substrate.cloud_prep import prepare_gcp_for_migrate, REQUIRED_GCP_APIS
    _step_progress(record, "cloud_prep", "switching gcloud account + enabling APIs")
    prep = prepare_gcp_for_migrate(
        active_account=record.active_account,
        project_id=record.project_id,
        apis=REQUIRED_GCP_APIS,
    )
    if not prep.succeeded:
        record.status = "preflight_failed"
        record.error = f"cloud_prep: {prep.fail_reason}"
        record.completed_at = datetime.now(timezone.utc).isoformat()
        _persist(record)
        return

    if _RUNNER.cancel_requested():
        record.status = "cancelled"
        record.completed_at = datetime.now(timezone.utc).isoformat()
        _persist(record)
        return

    # 3. Preflight — cloud doctor
    from app.substrate.cloud_doctor import check_readiness
    _step_progress(record, "preflight", "running cloud_doctor")
    readiness = check_readiness(target=record.target)
    if readiness.overall not in ("OK", "DEGRADED"):
        record.status = "preflight_failed"
        record.error = f"cloud_doctor.overall={readiness.overall}"
        record.completed_at = datetime.now(timezone.utc).isoformat()
        _persist(record)
        return

    if _RUNNER.cancel_requested():
        record.status = "cancelled"
        record.completed_at = datetime.now(timezone.utc).isoformat()
        _persist(record)
        return

    # 4-7. The actual live migrate. We delegate to run_migration_live
    # which executes provision/transfer/restore/verify. Progress
    # granularity is coarse (we don't get per-step callbacks from
    # the synchronous run); we mark "running" and update only at
    # completion.
    record.status = "running"
    _step_progress(
        record, "provision",
        f"running terraform apply (~15min) + transfer + restore + verify",
    )

    # Set the OAuth token for terraform via env. ``run_migration_live``
    # internally invokes scripts/install/gcp.sh which inherits env.
    import os
    for k, v in prep.terraform_env.items():
        os.environ[k] = v

    try:
        from app.substrate.migration import run_migration_live
        live_run = run_migration_live(
            target=record.target,
            tier=record.tier,
            region=record.region,
            project_id=record.project_id,
            confirm_phrase=confirm_phrase,
            budget_cap_usd=budget_cap_usd,
            run_id=record.run_id,   # carry our run_id through so report.json lands at the same dir
            execute_subprocess=execute_subprocess,
        )
    except Exception as exc:
        record.status = "failed"
        record.error = f"run_migration_live crashed: {type(exc).__name__}: {exc}"
        record.completed_at = datetime.now(timezone.utc).isoformat()
        _persist(record)
        return
    finally:
        # Clear the in-process token env so subsequent operations don't
        # accidentally reuse it.
        for k in prep.terraform_env:
            os.environ.pop(k, None)

    # 8. Done — translate the live_run outcome into our record
    record.completed_at = datetime.now(timezone.utc).isoformat()
    record.report_path = str(_run_state_path(record.run_id).parent / "report.json")
    if live_run.ready_for_live:
        record.status = "succeeded"
        _step_progress(record, "done", "migration complete — ready for cutover")
    else:
        record.status = "failed"
        record.error = "; ".join(live_run.blockers[:3]) if live_run.blockers else "(no blockers; check report.json)"
        _persist(record)
