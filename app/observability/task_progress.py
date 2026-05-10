"""
task_progress — Per-task "user-visible output was just produced" heartbeat.

Complements :mod:`app.rate_throttle.record_llm_activity`:

* **LLM activity**    — "something LLM-shaped returned" (cheap, cross-cutting,
                        fires whether the LLM produced useful output or an API
                        error).  Good at detecting hung threads.  **Bad** at
                        detecting LLM retry-loops that look busy but produce
                        no deliverable — every 403/429 still counts as
                        "activity" because the LLM did, in fact, cycle.

* **Output progress** — "something the user would actually see was just
                        produced".  Tools call this every time they ship a
                        partial row / chunk / finding.  Strict: a retry loop
                        does **not** advance it, a structurally-stuck task
                        does **not** advance it, only real deliverables
                        advance it.

The progress-gated timeout in :func:`app.main.handle_task` should prefer
output-progress when available and fall back to LLM activity when not
(so un-instrumented tasks don't starve).

Thread safety
-------------
All functions are thread-safe.  Tools invoked from CrewAI's worker thread
pool can record progress without coordinating with the asyncio loop.

Scoping
-------
Progress is scoped by ``task_id``.  For Signal-originating tasks, the
natural task_id is the sender phone number — it's stable for the lifetime
of the handle_task call.  The :data:`current_task_id` :class:`ContextVar`
lets tools read the active task_id without threading it through every
call signature; :func:`app.main.handle_task` sets it before dispatching
to the commander.
"""

from __future__ import annotations

import logging
import threading
import time
from contextvars import ContextVar

logger = logging.getLogger(__name__)

# ── State ────────────────────────────────────────────────────────────
_last_progress: dict[str, float] = {}   # task_id → monotonic ts
_progress_count: dict[str, int] = {}    # task_id → count

# Cure C (2026-05-10) — failure context tracker.  Distinct from the
# heartbeat / count tracker above: this records the *most recent
# failure signal* a task has hit (vetting rejection, artifact-not-
# produced, completion truncation, …), so when the watchdog fires
# its apology can include the actual reason instead of generic
# "narrow your question" advice.  Pre-fix the watchdog had no way
# to read the failure context — the apology message at
# app/main.py:1894-1900 conflated "agent stuck without progress"
# with "agent hit explicit vetting failures and exhausted retries"
# and gave the same misleading message in both cases.
#
# Schema: {task_id: {"kind": str, "detail": str, "ts": monotonic_float}}.
# kind is a short stable enum-ish string ("vetting_fail",
# "artifact_missing", "completion_truncated", "exception", …)
# that the apology formatter can switch on.
_failure_context: dict[str, dict] = {}
_lock = threading.Lock()

# The task id of the request currently being processed in this context.
# Set by ``handle_task`` before it dispatches to the commander so tools
# can read it lazily without per-call threading.
current_task_id: ContextVar[str] = ContextVar("current_task_id", default="")

# Drop entries older than this during GC passes to cap memory growth.
_GC_MAX_AGE_SECONDS = 3600  # 1 hour


# ── Public API ───────────────────────────────────────────────────────
def record_output_progress(task_id: str | None = None, *, note: str = "") -> None:
    """Mark that the given task just produced a user-visible partial
    result (a table row, a paragraph, a search hit, etc.).

    Safe to call from any thread.  A ``task_id`` of ``None`` resolves
    via the :data:`current_task_id` context-var — tools called from
    inside :func:`app.main.handle_task` pick up the right id
    automatically.

    Parameters
    ----------
    task_id : explicit task id; falls back to the context-var if None
              or empty.  A falsy id is a silent no-op (useful for call
              sites that might run outside a request, e.g. tests).
    note    : optional short label for the debug log (not persisted).
    """
    tid = task_id or current_task_id.get()
    if not tid:
        return
    with _lock:
        now = time.monotonic()
        _last_progress[tid] = now
        _progress_count[tid] = _progress_count.get(tid, 0) + 1
        _gc_locked(now)
    if note:
        logger.debug("task_progress: %s → %s", tid[-4:], note[:120])


def seconds_since_last_output_progress(task_id: str) -> float | None:
    """Return seconds since the task last produced output-progress, or
    ``None`` if the task has never emitted (bootstrap / not yet
    instrumented) — callers should treat ``None`` as "no signal
    available, fall back to LLM-activity".
    """
    if not task_id:
        return None
    with _lock:
        ts = _last_progress.get(task_id)
        if ts is None:
            return None
        return time.monotonic() - ts


def output_progress_count(task_id: str) -> int:
    """Total partial results emitted for this task since the tracker
    started.  Useful for "did the task produce at least N rows?" gates."""
    if not task_id:
        return 0
    with _lock:
        return _progress_count.get(task_id, 0)


def reset_task(task_id: str) -> None:
    """Drop a task's counters.  Called at the end of handle_task so
    stale entries from crashed threads don't accumulate."""
    if not task_id:
        return
    with _lock:
        _last_progress.pop(task_id, None)
        _progress_count.pop(task_id, None)
        _failure_context.pop(task_id, None)


# ── Failure-context API (Cure C, 2026-05-10) ─────────────────────────


def record_failure_context(
    kind: str,
    detail: str = "",
    *,
    task_id: str | None = None,
) -> None:
    """Stash the most recent failure signal for a task.

    Called from the points where failures are first detected
    (vetting rejection, artifact verification, completion truncation,
    typed exception in dispatch).  The watchdog's user-facing
    apology reads this so the message can name the actual cause
    instead of falling back to "please re-send a narrower question".

    Parameters
    ----------
    kind   : short stable identifier (``vetting_fail`` /
             ``artifact_missing`` / ``completion_truncated`` /
             ``exception`` / …).  The apology formatter switches
             on this; new values are accepted but render as
             generic.
    detail : human-readable specifics (vetting issues list,
             missing artifact path, truncated model id, …).
             Truncated to 500 chars to keep the apology bounded.
    task_id: explicit task id; defaults to the ContextVar.

    Multiple calls overwrite — the LATEST failure wins.  Cleared
    by ``reset_task`` at request end.
    """
    tid = task_id or current_task_id.get()
    if not tid:
        return
    with _lock:
        _failure_context[tid] = {
            "kind": kind[:64],
            "detail": (detail or "")[:500],
            "ts": time.monotonic(),
        }


def get_failure_context(task_id: str) -> dict | None:
    """Return the latest failure context for a task, or None.

    Returns a dict with ``kind`` / ``detail`` / ``age_s`` (seconds
    since the failure was recorded).  ``None`` means "no failure
    has been explicitly recorded for this task" — callers should
    distinguish that from "the task is fine" (it might still be
    stuck for a different reason; failure-context is best-effort).
    """
    if not task_id:
        return None
    with _lock:
        entry = _failure_context.get(task_id)
        if entry is None:
            return None
        return {
            "kind": entry["kind"],
            "detail": entry["detail"],
            "age_s": time.monotonic() - entry["ts"],
        }


def snapshot_all() -> dict[str, dict[str, float | int]]:
    """Observability helper: return ``{task_id: {last_ts_age, count}}``
    for every currently-tracked task.  Used by tests and by future
    ``/api/cp/*`` debug endpoints."""
    with _lock:
        now = time.monotonic()
        return {
            tid: {
                "seconds_since_last": now - ts,
                "count": _progress_count.get(tid, 0),
            }
            for tid, ts in _last_progress.items()
        }


# ── Internal ─────────────────────────────────────────────────────────
def _gc_locked(now: float) -> None:
    """Drop entries older than ``_GC_MAX_AGE_SECONDS``.  Caller holds
    ``_lock``."""
    cutoff = now - _GC_MAX_AGE_SECONDS
    stale = [tid for tid, ts in _last_progress.items() if ts < cutoff]
    for tid in stale:
        _last_progress.pop(tid, None)
        _progress_count.pop(tid, None)
