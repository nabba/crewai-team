"""Workflow run queue — async execution via a background thread pool.

PROGRAM §46.3 — Q8.3. The operator's spec called for "async + queue
+ status endpoint": REST callers get a ``run_id`` back immediately
and poll ``GET /api/cp/workflows/runs/<id>`` for status.

Implementation:

  * A module-level :class:`concurrent.futures.ThreadPoolExecutor`
    sized at ``_MAX_CONCURRENT_RUNS`` (default 4) handles execution.
  * Each enqueued run is persisted as ``QUEUED`` immediately so a
    crash doesn't lose it.
  * The executor mutates the run record through every transition;
    every transition is persisted via ``_persist_run`` to disk.
  * Cancel: best-effort flag-flip — running nodes finish, then the
    next step sees the flag.

This is a *small* queue — no Redis, no distributed workers. It
matches the rest of the codebase's "single gateway process owns
its background work" pattern (see ``app.idle_scheduler`` and
``app.healing.monitors``).
"""
from __future__ import annotations

import json
import logging
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.workflows import store
from app.workflows.executor import execute_run
from app.workflows.models import RunStatus, WorkflowRun, WorkflowTemplate

logger = logging.getLogger(__name__)


_MAX_CONCURRENT_RUNS = 4
_executor: ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()
_runs_in_flight: dict[str, Future] = {}
_cancel_flags: dict[str, bool] = {}


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is not None:
        return _executor
    with _executor_lock:
        if _executor is not None:
            return _executor
        _executor = ThreadPoolExecutor(
            max_workers=_MAX_CONCURRENT_RUNS,
            thread_name_prefix="workflow-run",
        )
        return _executor


def enqueue(
    template: WorkflowTemplate,
    *,
    inputs: dict[str, Any] | None = None,
    tool_dispatcher=None,
) -> WorkflowRun:
    """Persist a QUEUED run, submit it to the executor pool, return
    the run record.

    Callers receive the ``run_id`` via the returned record and poll
    via :func:`get_run`.

    ``tool_dispatcher`` is an injection seam — unit tests pass a
    fake; production leaves it ``None``.
    """
    run = WorkflowRun(
        id=str(uuid.uuid4()),
        template_id=template.id,
        status=RunStatus.QUEUED,
        started_at=_now_iso(),
        inputs=dict(inputs or {}),
    )
    _persist_run(run)
    fut = _get_executor().submit(
        _run_worker, template, run, tool_dispatcher,
    )
    _runs_in_flight[run.id] = fut
    return run


def get_run(run_id: str) -> WorkflowRun | None:
    """Read a run by id. Disk-first (works across restarts)."""
    path = _run_path(run_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return WorkflowRun.from_dict(data)
    except Exception:
        logger.warning("workflows: cannot read run %s", run_id, exc_info=True)
        return None


def list_runs(
    *,
    template_id: str | None = None,
    limit: int = 50,
) -> list[WorkflowRun]:
    """Return recent runs, newest first. Optionally filter by template."""
    runs_dir = store.runs_dir()
    files = sorted(
        runs_dir.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    out: list[WorkflowRun] = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            run = WorkflowRun.from_dict(data)
        except Exception:
            continue
        if template_id is not None and run.template_id != template_id:
            continue
        out.append(run)
        if len(out) >= limit:
            break
    return out


def cancel_run(run_id: str) -> bool:
    """Flag the run for cancellation. Returns True if the run was
    found AND not yet terminal."""
    run = get_run(run_id)
    if run is None or run.is_terminal:
        return False
    _cancel_flags[run_id] = True
    # If still queued, we can mark it cancelled inline.
    if run.status is RunStatus.QUEUED:
        run.status = RunStatus.CANCELLED
        run.finished_at = _now_iso()
        _persist_run(run)
    return True


# ─────────────────────────────────────────────────────────────────────
#   Worker
# ─────────────────────────────────────────────────────────────────────


def _run_worker(
    template: WorkflowTemplate,
    run: WorkflowRun,
    tool_dispatcher,
) -> None:
    """Executed on a background thread. Drives the executor + writes
    per-node-state persistence + records outcome on the template."""
    try:
        # Honour cancel flag set while still queued.
        if _cancel_flags.pop(run.id, False):
            run.status = RunStatus.CANCELLED
            run.finished_at = _now_iso()
            _persist_run(run)
            return
        execute_run(
            template, run,
            tool_dispatcher=tool_dispatcher,
            persist_callback=_persist_run,
        )
    except Exception as exc:
        logger.warning("workflows: worker raised: %s", exc, exc_info=True)
        run.status = RunStatus.FAILED
        run.error = f"worker exception: {exc}"
        run.finished_at = _now_iso()
        _persist_run(run)
    finally:
        _runs_in_flight.pop(run.id, None)
        # Bump template counters on terminal status
        try:
            store.record_run_outcome(
                run.template_id,
                success=(run.status is RunStatus.SUCCEEDED),
                finished_at=run.finished_at or _now_iso(),
            )
        except Exception:
            logger.debug("workflows: outcome record failed", exc_info=True)


def _persist_run(run: WorkflowRun) -> None:
    path = _run_path(run.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(run.to_dict(), indent=2, default=str))
        tmp.replace(path)
    except OSError:
        logger.debug("workflows: run persist failed", exc_info=True)


def _run_path(run_id: str) -> Path:
    return store.runs_dir() / f"{run_id}.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def wait_for_run(run_id: str, *, timeout: float = 30.0) -> WorkflowRun | None:
    """Synchronous wait helper for tests + occasional CLI use.

    Blocks until the run terminates or ``timeout`` elapses. Returns
    the final run record OR None if the run never appeared (bogus
    id) OR a partial record on timeout."""
    fut = _runs_in_flight.get(run_id)
    if fut is not None:
        try:
            fut.result(timeout=timeout)
        except Exception:
            pass
    return get_run(run_id)
