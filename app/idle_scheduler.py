"""
idle_scheduler.py — Run background work (self-improvement, retrospective, evolution)
during idle time when no user tasks are active.

Architecture:
  - Tracks user task activity via notify_task_start() / notify_task_end()
  - After IDLE_DELAY_SECONDS of no activity, starts cycling through background jobs
  - Each job runs one iteration, then yields back (cooperative multitasking)
  - If a user task arrives, background work is interrupted at the next yield point
  - Kill switch via Firestore config/background_tasks {enabled: bool}
  - Dashboard toggle controls the kill switch in real time
  - Background LLM calls are marked as low-priority (rate_throttle yields to user calls)
  - Long-running jobs can check should_yield() to abort mid-execution when user arrives

The idle loop does NOT replace cron jobs — cron jobs are the guaranteed baseline.
Idle scheduling is opportunistic: it fills dead time between user requests.
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from app.workspace_publish import publish_idle_outcome, publish_to_workspace

logger = logging.getLogger(__name__)

# How long to wait after last user task before starting background work
IDLE_DELAY_SECONDS = 30

# Pause between background job iterations (brief cooldown, then next job)
INTER_JOB_PAUSE_SECONDS = 2  # Reduced from 5 — lightweight jobs don't need long pauses

# Per-job failure handling (named so operators can tune without reading
# the body of _run_single_job). Phase G4 — pre-existing literals lifted
# to module constants. NO behavioural change.
MAX_CONSECUTIVE_FAILURES = 3            # job is skipped for cooldown after this many
JOB_COOLDOWN_AFTER_FAILURES_S = 3600    # 1 h cooldown after MAX_CONSECUTIVE_FAILURES
TRAINING_LOOP_INTERVAL_S = 3600         # cadence of the training-pipeline loop


# ── Job weight classification ────────────────────────────────────────────────

class JobWeight:
    LIGHT = "light"    # <30s: monitoring, snapshots, indexing
    MEDIUM = "medium"  # 30s-3min: feedback, safety, cogito
    HEAVY = "heavy"    # 3min+: evolution, training, retrospective

# Time caps per weight class (seconds) — cooperative via should_yield()
TIME_CAPS = {
    JobWeight.LIGHT: 60,
    JobWeight.MEDIUM: 180,
    JobWeight.HEAVY: 600,
}


def _substrate_defer_reason(weight: str) -> str | None:
    """Productization plan T2.5 — consult the substrate resource policy.

    Returns a defer_reason string if heavy/medium work should pause this
    cycle (disk pressure, host_substrate alert, etc.), or None to proceed.
    LIGHT jobs are never deferred — they're observability/reconciler work
    that needs to keep running even under pressure.

    Fail-safe: if the substrate package isn't importable, returns None
    (run normally). The whole point of substrate is to be additive — a
    broken policy module must never stall the system.
    """
    if weight == JobWeight.LIGHT:
        return None
    try:
        from app.substrate.policy import should_defer_heavy_work
        return should_defer_heavy_work()
    except Exception:
        logger.debug("idle_scheduler: substrate policy probe failed", exc_info=True)
        return None


def _publish_deferral(job_name: str, weight: str, reason: str) -> None:
    """Emit a visible event when an idle job is deferred for resource posture.

    Silent deferral is the failure mode the substrate policy exists to
    prevent. Every defer fires through workspace_publish so the dashboard
    + chronicle can render the operator-visible event.
    """
    logger.info(
        "idle_scheduler: deferred %s (%s) — %s", job_name, weight, reason,
    )
    try:
        from app.workspace_publish import publish_idle_outcome
        publish_idle_outcome(
            source="idle_scheduler",
            signal_type="resource_pressure",
            counts={"deferred": 1, "weight": weight},
            salience_key="deferred",
            content_template=f"idle_job_deferred: {job_name} ({weight}) — {reason}",
        )
    except Exception:
        logger.debug("idle_scheduler: publish_deferral failed", exc_info=True)

# Global state
_last_task_end: float = 0.0  # monotonic timestamp of last user task completion
_active_tasks: int = 0
_lock = threading.Lock()
_stop_event = threading.Event()
_enabled = True  # kill switch — toggled from Firestore
_enabled_lock = threading.Lock()
_idle_thread: threading.Thread | None = None

# Snapshot of the jobs passed to start(), exposed read-only via
# list_jobs() so dashboards can publish the full registry alongside
# APScheduler cron jobs. Stored as (name, weight) — the callable is
# intentionally dropped (not serialisable, not interesting to readers).
_active_jobs_snapshot: list[tuple[str, str]] = []


def notify_task_start() -> None:
    """Called when a user task begins processing."""
    global _active_tasks
    with _lock:
        _active_tasks += 1


def notify_task_end() -> None:
    """Called when a user task finishes (success or failure)."""
    global _active_tasks, _last_task_end
    with _lock:
        _active_tasks = max(0, _active_tasks - 1)
        _last_task_end = time.monotonic()


def is_idle() -> bool:
    """Check if the system is idle (no active tasks + idle delay elapsed)."""
    with _lock:
        if _active_tasks > 0:
            return False
        if _last_task_end == 0:
            # No tasks ever ran — consider idle after startup delay
            return True
        return (time.monotonic() - _last_task_end) >= IDLE_DELAY_SECONDS


_job_timeout = threading.Event()  # Set when a job exceeds its time cap


def should_yield() -> bool:
    """Check if a background job should abort because a user task arrived
    or the job exceeded its time cap.

    Long-running background functions (evolution iterations, self-improvement)
    should call this between units of work and return early if True.
    """
    with _lock:
        return _active_tasks > 0 or _job_timeout.is_set()


def set_enabled(enabled: bool) -> None:
    """Kill switch — disable/enable background work."""
    global _enabled
    with _enabled_lock:
        _enabled = enabled
    logger.info(f"idle_scheduler: background tasks {'enabled' if enabled else 'disabled'}")


def is_enabled() -> bool:
    with _enabled_lock:
        return _enabled


# ── Persistent job failure state (survives restarts via dbm.sqlite3) ──────────
# Before Python 3.13 these were in-memory dicts lost on every restart.
# Now persisted so a job in 1-hour cooldown stays in cooldown after restart.
_JOB_STATE_PATH = "/app/workspace/memory/idle_job_state"

_job_failure_counts: dict[str, int] = {}  # Per-job consecutive failure counter
_job_skip_until: dict[str, float] = {}    # Per-job skip-until timestamp (wall clock)


def _load_job_state() -> None:
    """Load persisted job failure counts and skip-until timestamps."""
    global _job_failure_counts, _job_skip_until
    try:
        import dbm.sqlite3
        import os
        os.makedirs(os.path.dirname(_JOB_STATE_PATH), exist_ok=True)
        with dbm.sqlite3.open(_JOB_STATE_PATH, "c") as db:
            for key in db.keys():
                k = key.decode() if isinstance(key, bytes) else key
                val = db[key].decode() if isinstance(db[key], bytes) else db[key]
                if k.startswith("fail:"):
                    _job_failure_counts[k[5:]] = int(val)
                elif k.startswith("skip:"):
                    ts = float(val)
                    if ts > time.time():  # Only load if still in the future
                        _job_skip_until[k[5:]] = ts
        if _job_failure_counts or _job_skip_until:
            logger.info(f"idle_scheduler: restored job state — "
                        f"{len(_job_failure_counts)} failure counts, "
                        f"{len(_job_skip_until)} active cooldowns")
    except Exception:
        logger.debug("idle_scheduler: job state load failed (starting fresh)", exc_info=True)


def _persist_job_failure(name: str, count: int) -> None:
    """Persist failure count for a job."""
    try:
        import dbm.sqlite3
        with dbm.sqlite3.open(_JOB_STATE_PATH, "c") as db:
            db[f"fail:{name}"] = str(count)
    except Exception:
        pass


def _persist_job_skip(name: str, until: float) -> None:
    """Persist skip-until timestamp for a job."""
    try:
        import dbm.sqlite3
        with dbm.sqlite3.open(_JOB_STATE_PATH, "c") as db:
            db[f"skip:{name}"] = str(until)
    except Exception:
        pass


def _persist_clear_skip(name: str) -> None:
    """Remove skip-until persistence for ``name``. Phase H #3 — half-open
    probe success path needs to clear the cooldown across restarts."""
    try:
        import dbm.sqlite3
        with dbm.sqlite3.open(_JOB_STATE_PATH, "c") as db:
            key = f"skip:{name}"
            if key in db:
                del db[key]
    except Exception:
        pass


# Phase H #3 (2026-05-10) — half-open probe state. In-memory only;
# losing it on restart just means probes re-arm afresh, which is
# fine. Tracks which probe points (0.25, 0.5, 0.75 of the cooldown
# window) have been consumed so we don't probe twice at the same
# fraction.
_job_half_open_used: dict[str, set[float]] = {}


def _half_open_probe_allowed(name: str, skip_until: float, now: float) -> bool:
    """True iff a half-open probe at the current cooldown fraction is allowed.

    Probes at 1/4, 1/2, 3/4 of the cooldown window. Each probe-point
    fires AT MOST ONCE per cooldown — so a transient outage can be
    detected as cleared without waiting for the full hour.
    """
    cooldown_started = skip_until - JOB_COOLDOWN_AFTER_FAILURES_S
    elapsed = now - cooldown_started
    if elapsed <= 0 or JOB_COOLDOWN_AFTER_FAILURES_S <= 0:
        return False
    fraction = elapsed / JOB_COOLDOWN_AFTER_FAILURES_S
    used = _job_half_open_used.setdefault(name, set())
    for probe in (0.25, 0.5, 0.75):
        if fraction >= probe and probe not in used:
            used.add(probe)
            return True
    return False


def _clear_cooldown(name: str) -> None:
    """Reset all cooldown state for ``name``. Called on probe success."""
    _job_skip_until.pop(name, None)
    _job_failure_counts[name] = 0
    _job_half_open_used.pop(name, None)
    _persist_clear_skip(name)
    _persist_job_failure(name, 0)


# Load on module init
_load_job_state()


# ── Phase E3: introspection snapshot for the dashboard ───────────────
# Read-only view of the scheduler's state. Exposes which jobs are
# currently in cooldown, their failure counts, and (where available)
# the names of jobs the scheduler considers "registered" via the
# ``_default_jobs()`` factory. The dashboard surface uses this so
# operators can answer "why hasn't job X run for 4 hours?" without
# attaching a debugger.

_last_job_success_ts: dict[str, float] = {}    # name -> monotonic ts
_last_job_failure_ts: dict[str, float] = {}    # name -> monotonic ts
_currently_running_job: str | None = None       # set/cleared in _run_single_job


def _record_job_outcome(name: str, success: bool) -> None:
    """Update last-success / last-failure timestamps. Called by _run_single_job."""
    now = time.monotonic()
    if success:
        _last_job_success_ts[name] = now
    else:
        _last_job_failure_ts[name] = now


def get_job_snapshot() -> dict[str, dict]:
    """Return a JSON-serializable snapshot of every known job's state.

    Format::

        {
          "<job_name>": {
            "failure_count": int,
            "in_cooldown": bool,
            "cooldown_until_ts": float | None,
            "seconds_since_last_success": float | None,
            "seconds_since_last_failure": float | None,
            "currently_running": bool,
          },
          ...
        }

    Job names are the union of every name we've seen — failure
    counters, cooldowns, success/failure timestamps. The scheduler's
    static job-list registry is not exposed here (it's a closure local
    inside ``_run_idle_loop``); callers wanting that should consult
    the source. This snapshot is enough to answer "what's stuck?".
    """
    now_mono = time.monotonic()
    now_wall = time.time()
    names = (
        set(_job_failure_counts.keys())
        | set(_job_skip_until.keys())
        | set(_last_job_success_ts.keys())
        | set(_last_job_failure_ts.keys())
    )
    if _currently_running_job:
        names.add(_currently_running_job)
    out: dict[str, dict] = {}
    for name in sorted(names):
        cooldown_until = _job_skip_until.get(name)
        in_cooldown = bool(cooldown_until and cooldown_until > now_wall)
        last_succ = _last_job_success_ts.get(name)
        last_fail = _last_job_failure_ts.get(name)
        out[name] = {
            "failure_count": _job_failure_counts.get(name, 0),
            "in_cooldown": in_cooldown,
            "cooldown_until_ts": cooldown_until if in_cooldown else None,
            "seconds_since_last_success": (now_mono - last_succ) if last_succ else None,
            "seconds_since_last_failure": (now_mono - last_fail) if last_fail else None,
            "currently_running": (name == _currently_running_job),
        }
    return out


def _run_single_job(name: str, fn: Callable, timeout_s: int = 60) -> bool:
    """Run a single job with time cap, retry on failure, and skip-after-3-failures.

    Returns True if job succeeded, False if failed.
    Jobs that fail 3 consecutive times are skipped for 1 hour.

    Phase H #3 (2026-05-10): the cooldown allows probe attempts at
    1/4, 1/2, 3/4 of the window. A transient outage at T+0 with the
    cooldown set to T+1h is now detected as cleared at T+15min on
    the first probe, instead of waiting the full hour. Probe success
    clears all cooldown state; probe failure leaves cooldown in
    place but consumes that probe-point.
    """
    # Check if job is in skip cooldown (wall clock — survives restarts).
    skip_until = _job_skip_until.get(name, 0)
    is_half_open_probe = False
    now = time.time()
    if skip_until and now < skip_until:
        if _half_open_probe_allowed(name, skip_until, now):
            is_half_open_probe = True
            logger.info(
                "idle_scheduler: '%s' half-open probe (cooldown until %s)",
                name, time.strftime("%H:%M", time.localtime(skip_until)),
            )
        else:
            return False

    _job_timeout.clear()
    timer = threading.Timer(timeout_s, _job_timeout.set)
    timer.daemon = True
    timer.start()
    global _currently_running_job
    _currently_running_job = name
    # Tag every nested LLM call with the job name so reconcile_actual_spend
    # writes to a budgets row keyed on it (e.g. "llm-discovery",
    # "fiction-ingest", "training-collector") instead of the generic
    # "unknown" fallback. This is what splits the historical "unknown"
    # bucket into per-job line items.
    from app.project_context import agent_scope
    try:
        _report_background_activity(name, "running")
        with agent_scope(name):
            fn()
        logger.info(f"idle_scheduler: '{name}' completed")
        _report_background_activity(name, "completed")
        _job_failure_counts[name] = 0  # Reset on success
        _persist_job_failure(name, 0)
        _record_job_outcome(name, success=True)
        if is_half_open_probe:
            # Phase H #3 — outage cleared. Wipe cooldown state so the
            # job returns to normal cadence.
            _clear_cooldown(name)
            logger.info(
                "idle_scheduler: '%s' half-open probe SUCCEEDED — "
                "cooldown cleared",
                name,
            )
        return True
    except Exception as exc:
        _job_failure_counts[name] = _job_failure_counts.get(name, 0) + 1
        consec = _job_failure_counts[name]
        _persist_job_failure(name, consec)
        logger.warning(f"idle_scheduler: '{name}' failed ({consec} consecutive): {exc}")
        _report_background_activity(name, "failed")
        _record_job_outcome(name, success=False)

        # After MAX_CONSECUTIVE_FAILURES, skip job for the cooldown window.
        if consec >= MAX_CONSECUTIVE_FAILURES:
            skip_ts = time.time() + JOB_COOLDOWN_AFTER_FAILURES_S
            _job_skip_until[name] = skip_ts
            _persist_job_skip(name, skip_ts)
            logger.warning(
                f"idle_scheduler: '{name}' skipped for "
                f"{JOB_COOLDOWN_AFTER_FAILURES_S}s after {consec} "
                f"consecutive failures"
            )

        try:
            from app.firebase_reporter import detect_credit_error, report_credit_alert
            provider = detect_credit_error(exc)
            if provider:
                report_credit_alert(provider, str(exc)[:300])
        except Exception:
            pass
        return False
    finally:
        timer.cancel()
        _job_timeout.clear()
        # Phase E3: clear "currently running" so the snapshot is accurate
        # the moment the job exits (success, failure, or via cooldown skip).
        _currently_running_job = None


def _drain_futures_with_timeout(
    futures: dict, timeout_s: float,
) -> None:
    """Drain a futures dict, swallowing per-future errors AND iterator timeout.

    PR 1 fix (2026-05-16): the previous inline code wrapped only
    ``future.result()`` in ``try/except Exception``. ``as_completed``
    itself raises ``concurrent.futures.TimeoutError`` at the *generator*
    level when at least one future still hasn't completed within
    ``timeout_s`` seconds — that exception escaped the for-loop and
    killed the idle-scheduler daemon thread, so one stuck job would
    silently stop ALL background work until the gateway restarted.

    On iterator timeout we cancel still-pending futures so a single
    stuck job doesn't permanently consume a slot in the calling pool.

    Args:
        futures: dict mapping ``Future`` → human-readable job name
        timeout_s: total seconds to wait for the slowest future
    """
    import concurrent.futures as _cf
    from concurrent.futures import as_completed
    try:
        for future in as_completed(futures, timeout=timeout_s):
            try:
                future.result()
            except Exception:
                pass
    except _cf.TimeoutError:
        stuck = [futures[f] for f in futures if not f.done()]
        logger.warning(
            "idle_scheduler: light-job phase timed out with %d "
            "futures still pending (%s) — cancelling",
            len(stuck),
            ", ".join(stuck[:5]) + ("…" if len(stuck) > 5 else ""),
        )
        for f in futures:
            if not f.done():
                f.cancel()


def _run_idle_loop(jobs) -> None:
    """Main idle loop — dual-queue architecture with parallel lightweight execution.

    Jobs are classified by weight (LIGHT/MEDIUM/HEAVY):
    - LIGHT jobs run in parallel (3 workers) — monitoring, snapshots, indexing
    - MEDIUM jobs run one at a time — feedback, cogito, probes
    - HEAVY jobs run one at a time, only after 2+ min idle — evolution, training

    Each class has a time cap enforced via should_yield() + _job_timeout event.
    """
    from app.rate_throttle import set_background_caller
    set_background_caller(True)

    # Classify jobs by weight
    light_jobs = [(n, fn) for n, fn, *w in jobs if (w[0] if w else JobWeight.MEDIUM) == JobWeight.LIGHT]
    medium_jobs = [(n, fn) for n, fn, *w in jobs if (w[0] if w else JobWeight.MEDIUM) == JobWeight.MEDIUM]
    heavy_jobs = [(n, fn) for n, fn, *w in jobs if (w[0] if w else JobWeight.MEDIUM) == JobWeight.HEAVY]

    logger.info(
        f"idle_scheduler: started — {len(light_jobs)} light, "
        f"{len(medium_jobs)} medium, {len(heavy_jobs)} heavy jobs"
    )

    try:
        _n_light_workers = __import__("app.config", fromlist=["get_settings"]).get_settings().idle_lightweight_workers
    except Exception:
        _n_light_workers = 3

    light_pool = ThreadPoolExecutor(max_workers=_n_light_workers, thread_name_prefix="idle-light")
    medium_idx = 0
    heavy_idx = 0
    _last_training_run = 0.0

    try:
        _training_interval = __import__("app.config", fromlist=["get_settings"]).get_settings().idle_training_interval_s
        _heavy_cap = __import__("app.config", fromlist=["get_settings"]).get_settings().idle_heavy_time_cap_s
    except Exception:
        _training_interval = TRAINING_LOOP_INTERVAL_S
        _heavy_cap = 600

    while not _stop_event.is_set():
        # Wait until idle
        while not _stop_event.is_set():
            if is_enabled() and is_idle():
                break
            _stop_event.wait(5)

        if _stop_event.is_set():
            break
        if not is_enabled() or not is_idle():
            continue

        # ── Phase 1: Run lightweight jobs in parallel ────────────────────
        if light_jobs:
            futures = {}
            for name, fn in light_jobs:
                if _stop_event.is_set() or not is_idle():
                    break
                futures[light_pool.submit(_run_single_job, name, fn, TIME_CAPS[JobWeight.LIGHT])] = name
            # Wait for all lightweight jobs (bounded by their time caps).
            _drain_futures_with_timeout(
                futures, TIME_CAPS[JobWeight.LIGHT] + 10,
            )

        if _stop_event.is_set() or not is_idle():
            continue

        # ── Phase 2: Run ONE medium job ──────────────────────────────────
        if medium_jobs:
            name, fn = medium_jobs[medium_idx % len(medium_jobs)]
            medium_idx += 1
            # Productization plan T2.5 — consult substrate resource policy.
            # Heavy/medium work defers under disk pressure or host alerts;
            # the deferral is published as a visible event (no silent skip).
            defer_reason = _substrate_defer_reason(JobWeight.MEDIUM)
            if defer_reason:
                _publish_deferral(name, JobWeight.MEDIUM, defer_reason)
            else:
                _run_single_job(name, fn, TIME_CAPS[JobWeight.MEDIUM])

        if _stop_event.is_set() or not is_idle():
            continue

        # ── Phase 3: Run ONE heavy job (only if idle for >2 min) ─────────
        if heavy_jobs and (time.monotonic() - _last_task_end) > 120:
            name, fn = heavy_jobs[heavy_idx % len(heavy_jobs)]
            heavy_idx += 1

            # Training pipeline: hourly cadence (not every cycle)
            if name == "training-pipeline":
                if time.monotonic() - _last_training_run < _training_interval:
                    continue  # Skip: ran less than 1 hour ago
                _last_training_run = time.monotonic()

            # Substrate resource policy — productization plan T2.5.
            defer_reason = _substrate_defer_reason(JobWeight.HEAVY)
            if defer_reason:
                _publish_deferral(name, JobWeight.HEAVY, defer_reason)
            else:
                _run_single_job(name, fn, _heavy_cap)

        # Brief pause between cycles
        for _ in range(INTER_JOB_PAUSE_SECONDS):
            if _stop_event.is_set() or not is_idle():
                break
            time.sleep(1)

    light_pool.shutdown(wait=False)
    logger.info("idle_scheduler: stopped")


def _report_background_activity(job_name: str, status: str) -> None:
    """Report background activity to Firestore for dashboard visibility."""
    try:
        from app.firebase_reporter import _fire, _get_db
        def _write() -> None:
            db = _get_db()
            if not db:
                return
            db.collection("status").document("background").set({
                "current_job": job_name,
                "status": status,
                "updated_at": time.time(),
            }, merge=True)
        _fire(_write)
    except Exception:
        pass


def start(jobs: list[tuple[str, Callable[[], None]]] | None = None) -> None:
    """Start the idle scheduler in a daemon thread.

    Args:
        jobs: List of (name, callable) tuples. If None, uses default jobs.
    """
    global _idle_thread

    # Auto-populate the LLM catalog from live sources before any job
    # runs. Idempotent — refresh respects a 24h on-disk TTL so cold
    # boot pulls fresh data while warm restarts load the cached snapshot.
    try:
        from app.llm_catalog_builder import refresh as _refresh_catalog
        summary = _refresh_catalog()
        logger.info(
            "idle_scheduler: catalog populated — %s entries",
            summary.get("catalog_size"),
        )
    except Exception:
        logger.debug("idle_scheduler: llm catalog refresh skipped", exc_info=True)

    # Rehydrate previously promoted models from the discovered_models
    # table. Runs after the catalog builder so promoted overrides layer
    # on top of the auto-populated snapshot. force=True so a gateway
    # restart after the boot-time call also picks up whatever landed
    # between start() invocations (shouldn't happen in practice but
    # guards against subtle module-state bugs).
    try:
        from app.llm_rehydrate import rehydrate_catalog
        added = rehydrate_catalog(force=True)
        logger.info(
            "idle_scheduler: boot rehydrate added %d promoted model(s)",
            added,
        )
    except Exception as exc:
        logger.warning(
            "idle_scheduler: boot rehydrate failed — %s: %s",
            type(exc).__name__, str(exc)[:200],
        )

    if jobs is None:
        jobs = _default_jobs()

    if not jobs:
        logger.warning("idle_scheduler: no jobs configured, not starting")
        return

    global _active_jobs_snapshot
    _active_jobs_snapshot = [
        (entry[0], (entry[2] if len(entry) >= 3 else JobWeight.MEDIUM))
        for entry in jobs
    ]

    _stop_event.clear()
    _idle_thread = threading.Thread(
        target=_run_idle_loop, args=(jobs,),
        daemon=True, name="idle-scheduler",
    )
    _idle_thread.start()


def list_jobs() -> list[tuple[str, str]]:
    """Return a (name, weight) snapshot of the currently registered idle jobs.

    Empty until start() has been called. Used by app.main._publish_schedule
    so the dashboard can show idle jobs alongside APScheduler cron jobs.
    """
    return list(_active_jobs_snapshot)


def stop() -> None:
    """Stop the idle scheduler."""
    _stop_event.set()
    if _idle_thread and _idle_thread.is_alive():
        _idle_thread.join(timeout=5)


def _default_jobs() -> list[tuple[str, Callable[[], None]]]:
    """Build the default job list from existing crews.

    Job ordering matters — the idle loop cycles round-robin, so we interleave
    different job types to keep variety and avoid burning all idle time on one
    activity. The sequence is:
      learn → evolve → learn → retrospective → evolve → scan → ...
    This gives evolution and learning 2x frequency over retrospective/scan.
    """
    jobs = []

    # ── Learning queue: process topics from the queue file ──────────────
    def _learn_queue() -> None:
        from app.crews.self_improvement_crew import SelfImprovementCrew
        SelfImprovementCrew().run()
    jobs.append(("learn-queue", _learn_queue, JobWeight.HEAVY))

    # ── Trajectory tips (arXiv:2603.10600): synthesize learnings from
    # captured execution trajectories. No-op when either trajectory_enabled
    # or tip_synthesis_enabled is False — zero cost until explicitly
    # rolled out. MEDIUM weight matches improvement-scan's cost profile.
    def _trajectory_tips() -> None:
        try:
            from app.crews.self_improvement_crew import SelfImprovementCrew
            SelfImprovementCrew().run_trajectory_tips(max_tips=3)
        except Exception:
            logger.debug("idle_scheduler: trajectory tips failed", exc_info=True)
    jobs.append(("trajectory-tips", _trajectory_tips, JobWeight.MEDIUM))

    # ── Transfer-memory compile (Phase 17a): nightly batch of healing /
    # evo / grounding / gap events into cross-domain "Insight" drafts.
    # Cadence-guarded to ≥24h; pins llm_mode="free" for the duration so
    # compile cost is bounded to the local Ollama + free-tier OpenRouter
    # cascade. Phase 17b: drafts now also flow through integrator with
    # status="shadow" so the retriever can find them in shadow mode.
    def _transfer_compile() -> None:
        try:
            from app.transfer_memory.compiler import run_compile
            run_compile()
        except Exception:
            logger.debug("idle_scheduler: transfer-compile failed", exc_info=True)
    jobs.append(("transfer-compile", _transfer_compile, JobWeight.HEAVY))

    # ── Transfer-memory attribution (Phase 17c): walk recent failed
    # trajectories that involved an injected transfer-memory record,
    # heuristically classify the implicated record under a
    # NegativeTransferTag, and apply the demotion ladder. LIGHT —
    # entirely deterministic, no LLM calls.
    def _transfer_attribution() -> None:
        try:
            from app.transfer_memory.attribution import run_attribution
            run_attribution()
        except Exception:
            logger.debug(
                "idle_scheduler: transfer-attribution failed", exc_info=True,
            )
    jobs.append(("transfer-attribution", _transfer_attribution, JobWeight.LIGHT))

    # ── Transfer-memory promotion (Phase 17c): walk records at
    # status="shadow", check eligibility (age + surface_count + no
    # negative attribution), and either promote them or refresh the
    # candidates file for operator review (default: review-only).
    # MEDIUM — touches both the index and the underlying KB metadata
    # when the auto-promote flag is on.
    def _transfer_promotion() -> None:
        try:
            from app.transfer_memory.promotion import run_promotion
            run_promotion()
        except Exception:
            logger.debug(
                "idle_scheduler: transfer-promotion failed", exc_info=True,
            )
    jobs.append(("transfer-promotion", _transfer_promotion, JobWeight.MEDIUM))

    # ── Belief outbox reconciler (Phase F1) ─────────────────────────────
    # Reads beliefs from Postgres and ensures every belief_id has a
    # matching :Belief node in Neo4j. Closes the cross-area gap where
    # store.py's fire-and-forget mirror call could leave Neo4j stale
    # if the graph was briefly unavailable. LIGHT — pure SQL + Cypher,
    # no LLM calls; the typical drift is single-digit beliefs.
    def _belief_outbox_neo4j() -> None:
        try:
            from app.memory.belief_outbox import reconcile_belief_outbox
            counts = reconcile_belief_outbox()
            publish_idle_outcome(
                source="belief-outbox-neo4j",
                signal_type="certainty_shift",
                counts=counts,
                salience_key="synced",
                content_template="belief_outbox(neo4j): synced {synced} (failed {failed}, skipped {skipped})",
            )
        except Exception:
            logger.debug(
                "idle_scheduler: belief-outbox-neo4j failed", exc_info=True,
            )
    jobs.append(("belief-outbox-neo4j", _belief_outbox_neo4j, JobWeight.LIGHT))

    # ── ChromaDB belief sync (Phase F2) ──────────────────────────────────
    # Mirrors the F1 idea but for the third store: indexes new/updated
    # beliefs into ChromaDB's "beliefs" collection so semantic retrieval
    # sees freshly-formed beliefs without waiting for a manual reindex.
    # LIGHT — incremental via watermark; embedding work is per-belief
    # and modest (Postgres typically holds < 1k active beliefs).
    def _belief_outbox_chroma() -> None:
        try:
            from app.memory.belief_outbox import sync_new_beliefs_to_chromadb
            counts = sync_new_beliefs_to_chromadb()
            publish_idle_outcome(
                source="belief-outbox-chroma",
                signal_type="certainty_shift",
                counts=counts,
                salience_key="indexed",
                content_template="belief_outbox(chroma): indexed {indexed} of {scanned} scanned",
            )
        except Exception:
            logger.debug(
                "idle_scheduler: belief-outbox-chroma failed", exc_info=True,
            )
    jobs.append(("belief-outbox-chroma", _belief_outbox_chroma, JobWeight.LIGHT))

    # ── Inbound DLQ drain (Phase F3) ──────────────────────────────────────
    # When load-shedding rejected a message, it was buffered in the
    # in-process DLQ. This job pulls a few messages back into handle_task
    # whenever capacity exists. Bounded per-pass; expired messages
    # (> 30 min) are dropped rather than replayed. LIGHT — fires every
    # idle slot; the queue is typically empty.
    def _drain_inbound_dlq() -> None:
        try:
            from app.dead_letter_inbound import drain, queue_depth
            from app.config import get_settings
            from app import main as _main_mod  # for handle_task + _inflight_tasks
            depth = queue_depth()
            if depth == 0:
                return
            settings = get_settings()
            threshold = (
                settings.load_shed_threshold
                or (settings.max_parallel_crews + 1)
            )
            inflight = getattr(_main_mod, "_inflight_tasks", 0)
            handle_task = getattr(_main_mod, "handle_task", None)
            if handle_task is None:
                logger.debug("dlq-drain: handle_task not available")
                return
            counts = drain(
                handle_task,
                inflight_count=inflight,
                shed_threshold=threshold,
                max_to_drain=3,  # bounded per pass — keep idle slot cheap
            )
            publish_idle_outcome(
                source="dlq-drain",
                signal_type="trend_reversal",
                counts=counts,
                salience_key="replayed",
                content_template="dlq-drain: replayed {replayed} (expired {expired})",
            )
        except Exception:
            logger.debug("idle_scheduler: dlq-drain failed", exc_info=True)
    jobs.append(("dlq-drain", _drain_inbound_dlq, JobWeight.LIGHT))

    # ── Wiki-index reconciler (consciousness-roadmap §4) ──────────────────
    # Drift-scan for `wiki/index.md`. Today the master index is rebuilt
    # event-driven by `WikiWriteTool._rebuild_master_index()` on
    # create/update/delete; out-of-band changes (manual file move, failed
    # idea promotion, rename) leave it drifted from on-disk truth without
    # any signal. This job detects the drift and produces a CANDIDATE file
    # under `workspace/dreams/`, then opens a change-request — never auto-
    # applies. LIGHT because the canonical compute over <100 pages is
    # sub-second; weekly cadence emerges from the idle scheduler's overall
    # rhythm, no explicit timer here.
    # ── Viability → goals connector (consciousness-roadmap §3.G1) ─────────
    # Closes the AE-1 PARTIAL → STRONG path. Reads the last N viability
    # frames from the affect trace; emits goals for variables in sustained
    # allostatic error (≥ N_CONSECUTIVE_REQUIRED frames above threshold);
    # writes via FIFO to kernel.self_state.current_goals (previously a dead
    # field). LIGHT — pure trace replay + small dict ops, no LLM.
    # Triggers ethical threshold T1 (consciousness-roadmap §6) on first
    # emission: welfare-check moves from observability to operator
    # obligation.
    # ── Backward counterfactual replay (consciousness-roadmap §3.G2) ─────
    # Real "dreams" subsystem: samples past affect-trace + chapter
    # fragments, recombines them into alternative-past scenarios, runs
    # them through a (stub or wired) predictor, audit-logs the outcomes.
    # Observational only — does not write to belief/, kernel/, or
    # current_goals. HEAVY-tier weekly cadence target; using LIGHT here
    # while predict_fn is the no-op stub (cheap), upgrade when wired to
    # PredictiveLayer.predict_and_compare. Triggers ethical threshold T2
    # on first sustained-promotion event downstream (retrospective rescan).
    def _backward_counterfactual_replay() -> None:
        try:
            from app.subia.dreams.engine import (
                production_predict_fn,
                run_pass as _replay_pass,
            )
            # Wire the live PredictiveLayer (singleton). production_predict_fn
            # internally falls back to (0.5, 0.0) on any predictor error, so
            # replay never crashes the idle thread.
            result = _replay_pass(predict_fn=production_predict_fn())
            if result.scenarios_count > 0 and not result.error:
                publish_to_workspace(
                    source="backward-replay",
                    content=(
                        f"Replay pass: {result.scenarios_count} scenarios, "
                        f"{result.sampled_count} fragments sampled "
                        f"(audit={result.audit_id})"
                    ),
                    salience=0.35,   # observational; above noise floor + ignition
                    signal_type="free_energy_spike",
                )
        except Exception:
            logger.debug(
                "idle_scheduler: backward-counterfactual-replay failed",
                exc_info=True,
            )
    jobs.append(("backward-counterfactual-replay", _backward_counterfactual_replay,
                 JobWeight.LIGHT))

    def _viability_goal_emitter() -> None:
        try:
            from app.affect.goal_emitter import run_pass
            result = run_pass()
            if result.written:
                # Goal emission is high-information for the workspace —
                # sustained low-viability has crossed the threshold of
                # operator-visible obligation.
                publish_to_workspace(
                    source="viability-goal-emitter",
                    content=(
                        f"Emitted {len(result.written)} goal(s) — "
                        f"triggers: {', '.join(g.triggered_by for g in result.written)}"
                    ),
                    salience=0.7,   # T1 threshold: above ignition + above bandwidth-2 floor
                    signal_type="disposition",
                )
        except Exception:
            logger.debug(
                "idle_scheduler: viability-goal-emitter failed", exc_info=True,
            )
    jobs.append(("viability-goal-emitter", _viability_goal_emitter, JobWeight.LIGHT))

    def _wiki_index_reconciler() -> None:
        try:
            from app.memory.wiki_index_reconciler import run_reconciler
            result = run_reconciler()
            if result.drift_detected:
                publish_to_workspace(
                    source="wiki-index-reconciler",
                    content=(
                        f"wiki/index.md drift candidate produced "
                        f"(audit={result.audit_id}, "
                        f"cr={result.change_request_id or 'none'})"
                    ),
                    # Drift is meaningful — push above ignition threshold so
                    # the workspace surfaces it, but not into critical band.
                    salience=0.55,
                    signal_type="trend_reversal",
                )
        except Exception:
            logger.debug(
                "idle_scheduler: wiki-index-reconciler failed",
                exc_info=True,
            )
    jobs.append(("wiki-index-reconciler", _wiki_index_reconciler, JobWeight.LIGHT))

    # ── Evolution: run experiments (2 iterations per idle slot) ─────────
    # Reduced from 5 to 2: each iteration takes ~4min, so 5 = 20min which
    # starves all subsequent jobs. 2 iterations keeps total under 10min.
    def _evolution() -> None:
        from app.evolution import run_evolution_session
        run_evolution_session(max_iterations=2)
    jobs.append(("evolution", _evolution, JobWeight.HEAVY))

    # ── Meta-evolution: improve the evolution engine's own parameters ──
    # Runs at ~1/5 evolution frequency (5 HEAVY jobs rotate round-robin).
    # Gate: MAX_META_MUTATIONS_PER_WEEK = 3, 8h cooldown between cycles.
    def _meta_evolution() -> None:
        from app.meta_evolution import run_meta_evolution
        run_meta_evolution()
    jobs.append(("meta-evolution", _meta_evolution, JobWeight.HEAVY))

    # Q11.1 (PROGRAM §46.18) — Analogy-index populator. HEAVY weekly
    # LLM pass over wiki + episteme that extracts abstract structural
    # patterns into the analogy index. Cadence-checked internally;
    # idempotent across overlapping fires. Master switch
    # ``analogy_index_populator_enabled`` (default ON).
    def _analogy_populator() -> None:
        from app.creativity import analogy_populator
        analogy_populator.run()
    jobs.append(("analogy-populator", _analogy_populator, JobWeight.HEAVY))

    # ── Proactive learning: discover and queue new topics ──────────────
    def _discover_topics() -> None:
        _auto_discover_topics()
    jobs.append(("discover-topics", _discover_topics, JobWeight.LIGHT))

    # ── MAP-Elites island migration: cross-pollinate top performers ─────
    # Multi-island MAP-Elites preserves separate populations to maintain
    # diversity; periodic migration of top performers between islands
    # prevents islands drifting into wholly disjoint local optima while
    # still keeping niche pressure. Cheap (in-memory grid manipulation).
    def _map_elites_migrate() -> None:
        _map_elites_migration()
    jobs.append(("map-elites-migrate", _map_elites_migrate, JobWeight.LIGHT))

    # ── Skills disk mirror: regenerate workspace/skills/ from KBs ──────
    # The on-disk markdown files are a presentation layer — source of
    # truth is the KBs. This job refreshes the mirror so any legacy code
    # (or the operator browsing the dir) sees current content.
    def _skills_mirror() -> None:
        try:
            from app.self_improvement.integrator import regenerate_disk_mirror
            regenerate_disk_mirror()
        except Exception:
            logger.debug("skills-mirror regen failed", exc_info=True)
    jobs.append(("skills-mirror", _skills_mirror, JobWeight.LIGHT))

    # ── Evaluator: flush usage hits + scan for zombie skills ───────────
    # Pending usage-hit buffer flushes to the SkillRecord index; decay
    # sweep emits USAGE_DECAY gaps for skills idle > 30 days.
    # Phase 6 (arXiv:2603.10600): additional sweep for trajectory tips
    # whose measured effectiveness has dropped below threshold.
    def _evaluator_sweep() -> None:
        try:
            from app.self_improvement.evaluator import (
                flush_hits, scan_for_decay, scan_for_low_effectiveness_tips,
            )
            flush_hits()
            scan_for_decay()
            # No-op in practice unless trajectory tips exist AND have
            # enough samples for the effectiveness signal to be acted on.
            scan_for_low_effectiveness_tips()
        except Exception:
            logger.debug("evaluator sweep failed", exc_info=True)
    jobs.append(("evaluator-sweep", _evaluator_sweep, JobWeight.LIGHT))

    # ── Consolidator: cluster + merge near-duplicate skills ────────────
    # Phase 5 of overhaul. Heavy because it pulls embeddings for all
    # active SkillRecords. Rate-limited by the weekly cadence — the
    # idle_scheduler's rotation ensures this runs ~1/N of idle slots.
    def _consolidator() -> None:
        try:
            from app.self_improvement.consolidator import run_consolidation_cycle
            counts = run_consolidation_cycle(auto_merge=True)
            publish_idle_outcome(
                source="skill-consolidator",
                signal_type="certainty_shift",
                counts=counts,
                salience_key="auto_merged",
                content_template=(
                    "skill-consolidator: auto_merged={auto_merged} "
                    "total_proposals={total_proposals}"
                ),
            )
        except Exception:
            logger.debug("consolidator run failed", exc_info=True)
    jobs.append(("consolidator", _consolidator, JobWeight.HEAVY))

    # ── Retrospective: analyze recent performance ──────────────────────
    def _retrospective() -> None:
        from app.crews.retrospective_crew import RetrospectiveCrew
        RetrospectiveCrew().run()
    jobs.append(("retrospective", _retrospective, JobWeight.HEAVY))

    # ── Embedded personality probes: covert measurement via real-ish tasks ──
    def _embedded_probe() -> None:
        try:
            from app.personality.probes import get_probe_engine
            from app.personality.evaluation import EvaluationEngine
            from app.personality.state import get_personality, save_personality
            from app.llm_factory import create_specialist_llm
            from app.prompt_registry import get_active_prompt

            engine = get_probe_engine()
            ee = EvaluationEngine()

            # Pick a random agent to probe
            roles = ["commander", "researcher", "coder", "writer"]
            role = roles[hash(str(time.monotonic())) % len(roles)]

            # Generate a covert probe
            probe = engine.generate_probe(role)
            if not probe:
                return

            # Get agent response (the agent doesn't know this is a test)
            llm = create_specialist_llm(max_tokens=1000, role=role)
            agent_prompt = get_active_prompt(role)
            response = str(llm.call(
                f"{(agent_prompt or '')[:2000]}\n\n{probe.task_description}"
            )).strip()

            if not response or len(response) < 20:
                return

            # Evaluate response against the hidden test dimension
            from app.personality.validation import get_bvl
            bvl = get_bvl()
            behavioral_history = bvl.get_behavioral_summary(role)
            state = get_personality(role)
            personality_summary = state.get_profile_summary()

            result = ee.evaluate(
                role, probe.target_dimension, probe.task_description,
                response, behavioral_history, personality_summary,
            )

            # Update personality state with covert measurement
            # Covert probes give more weight because the agent wasn't "performing"
            state.update_trait(
                "strengths" if probe.target_dimension in state.strengths
                else "temperament" if probe.target_dimension in state.temperament
                else "personality_factors",
                probe.target_dimension,
                result.composite_score,
            )
            state.assessment_count += 1
            state.last_assessment = datetime.now(timezone.utc).isoformat()
            save_personality(state)

            logger.info(
                f"personality: embedded probe completed for {role} "
                f"(type={probe.probe_type}, dim={probe.target_dimension}, "
                f"score={result.composite_score:.2f})"
            )
        except Exception:
            logger.debug("idle_scheduler: embedded probe failed", exc_info=True)
    jobs.append(("embedded-probe", _embedded_probe, JobWeight.MEDIUM))

    # ── Improvement scan: analyze gaps and propose improvements ────────
    def _improvement_scan() -> None:
        from app.crews.self_improvement_crew import SelfImprovementCrew
        SelfImprovementCrew().run_improvement_scan()
    jobs.append(("improvement-scan", _improvement_scan, JobWeight.MEDIUM))

    # ── Feedback aggregation: detect patterns in user feedback ──────────
    # Shared singleton for feedback pipeline (avoids creating new DB engine per job)
    _feedback_pipeline_cache = [None]
    def _get_feedback_pipeline() -> None:
        if _feedback_pipeline_cache[0] is None:
            from app.config import get_settings
            s = get_settings()
            if not s.mem0_postgres_url:
                return None
            from app.feedback_pipeline import FeedbackPipeline
            _feedback_pipeline_cache[0] = FeedbackPipeline(s.mem0_postgres_url)
        return _feedback_pipeline_cache[0]

    def _feedback_aggregate() -> None:
        try:
            pipeline = _get_feedback_pipeline()
            if not pipeline:
                return
            patterns = pipeline.aggregate_patterns()
            if patterns:
                logger.info(f"idle_scheduler: feedback aggregation found {len(patterns)} patterns")
        except Exception:
            logger.debug("idle_scheduler: feedback aggregation failed", exc_info=True)
    jobs.append(("feedback-aggregate", _feedback_aggregate, JobWeight.LIGHT))

    # ── General improvements pass: 7 background jobs ────────────────────
    # Wire all new modules into the idle scheduler. Each is best-effort —
    # an exception in one job never affects the others.

    def _refresh_self_model() -> None:
        try:
            from app.self_model import refresh_self_model
            refresh_self_model()
        except Exception:
            logger.debug("idle_scheduler: self_model refresh failed", exc_info=True)
    jobs.append(("self-model-refresh", _refresh_self_model, JobWeight.LIGHT))

    def _goodhart_check() -> None:
        try:
            from app.goodhart_guard import run_goodhart_check
            run_goodhart_check()
        except Exception:
            logger.debug("idle_scheduler: goodhart_check failed", exc_info=True)
    jobs.append(("goodhart-check", _goodhart_check, JobWeight.MEDIUM))

    def _knowledge_compactor_run() -> None:
        try:
            from app.knowledge_compactor import run_consolidation_cycle
            run_consolidation_cycle()
        except Exception:
            logger.debug("idle_scheduler: knowledge_compactor failed", exc_info=True)
    jobs.append(("knowledge-compactor", _knowledge_compactor_run, JobWeight.HEAVY))

    def _tier_graduation_eval() -> None:
        try:
            from app.tier_graduation import evaluate_all_graduations
            evaluate_all_graduations()
        except Exception:
            logger.debug("idle_scheduler: tier_graduation failed", exc_info=True)
    jobs.append(("tier-graduation", _tier_graduation_eval, JobWeight.LIGHT))

    def _alignment_audit() -> None:
        try:
            from app.alignment_audit import run_alignment_audit
            run_alignment_audit()
        except Exception:
            logger.debug("idle_scheduler: alignment_audit failed", exc_info=True)
    jobs.append(("alignment-audit", _alignment_audit, JobWeight.MEDIUM))

    def _improvement_narrative() -> None:
        try:
            from app.improvement_narrative import generate_daily_narrative
            generate_daily_narrative()
        except Exception:
            logger.debug("idle_scheduler: improvement_narrative failed", exc_info=True)
    jobs.append(("improvement-narrative", _improvement_narrative, JobWeight.LIGHT))

    def _human_gate_expire() -> None:
        try:
            from app.human_gate import expire_stale_requests
            expire_stale_requests()
        except Exception:
            logger.debug("idle_scheduler: human_gate expiry failed", exc_info=True)
    jobs.append(("human-gate-expire", _human_gate_expire, JobWeight.LIGHT))

    def _pattern_library_extract() -> None:
        try:
            from app.pattern_library import extract_patterns_from_history
            extract_patterns_from_history()
        except Exception:
            logger.debug("idle_scheduler: pattern_library extraction failed", exc_info=True)
    jobs.append(("pattern-library-extract", _pattern_library_extract, JobWeight.MEDIUM))

    # ── Safety health check: monitor for post-promotion regressions ────
    def _safety_health_check() -> None:
        try:
            from app.config import get_settings
            s = get_settings()
            if not s.mem0_postgres_url or not s.safety_auto_rollback:
                return
            import app.prompt_registry as registry
            from app.safety_guardian import SafetyGuardian
            guardian = SafetyGuardian(s.mem0_postgres_url, registry)
            rollbacks = guardian.check_post_promotion_health()
            if rollbacks:
                logger.info(f"idle_scheduler: safety check triggered {len(rollbacks)} rollback(s)")
            # Also check drift
            alerts = guardian.check_drift()
            if alerts:
                logger.warning(f"idle_scheduler: drift detection found {len(alerts)} alert(s)")
        except Exception:
            logger.debug("idle_scheduler: safety health check failed", exc_info=True)
    jobs.append(("safety-health-check", _safety_health_check, JobWeight.LIGHT))

    # ── Modification engine: propose prompt changes from feedback ─────
    def _modification_engine() -> None:
        try:
            from app.config import get_settings
            s = get_settings()
            if not s.mem0_postgres_url:
                return
            from app.modification_engine import ModificationEngine
            import app.prompt_registry as registry
            pipeline = _get_feedback_pipeline()  # Reuse singleton
            if not pipeline:
                return
            try:
                from app.eval_sandbox import EvalSandbox
                sandbox = EvalSandbox(s.mem0_postgres_url, registry)
            except Exception:
                sandbox = None
            engine = ModificationEngine(s.mem0_postgres_url, registry, pipeline, sandbox)
            results = engine.process_triggered_patterns()
            if results:
                logger.info(f"idle_scheduler: modification engine processed {len(results)} patterns")
        except Exception:
            logger.debug("idle_scheduler: modification engine failed", exc_info=True)
    jobs.append(("modification-engine", _modification_engine, JobWeight.MEDIUM))

    # ── Health monitor: evaluate dimensional health ─────────────────
    def _health_evaluate() -> None:
        try:
            from app.health_monitor import evaluate_health
            alerts = evaluate_health()
            if alerts:
                logger.info(f"idle_scheduler: health monitor found {len(alerts)} alert(s)")
        except Exception:
            logger.debug("idle_scheduler: health evaluation failed", exc_info=True)
    jobs.append(("health-evaluate", _health_evaluate, JobWeight.LIGHT))

    # ── Version manifest: periodic snapshot for rollback safety ────
    def _version_snapshot() -> None:
        try:
            from app.version_manifest import create_manifest, cleanup_old_snapshots
            create_manifest(promoted_by="system", reason="periodic snapshot")
            cleanup_old_snapshots(keep_latest=10)
        except Exception:
            logger.debug("idle_scheduler: version snapshot failed", exc_info=True)
    jobs.append(("version-snapshot", _version_snapshot, JobWeight.LIGHT))

    # ── PDS: personality development assessment session ──────────────────
    def _personality_session() -> None:
        try:
            from app.personality.assessment import AssessmentBatteryModule
            from app.personality.evaluation import EvaluationEngine
            from app.personality.feedback import DevelopmentalFeedbackLoop
            from app.personality.validation import get_bvl
            from app.personality.state import get_personality, save_personality

            abm = AssessmentBatteryModule()
            ee = EvaluationEngine()
            dfl = DevelopmentalFeedbackLoop()
            bvl = get_bvl()

            # Rotate through agent roles
            roles = ["commander", "researcher", "coder", "writer"]
            role = roles[hash(str(time.monotonic())) % len(roles)]
            state = get_personality(role)

            # Select and deliver assessment
            flags = bvl.get_inconsistency_flags(role)
            session = abm.select_assessment(role, behavioral_flags=flags,
                                              stage=state.developmental_stage)

            # Get agent response via LLM (simulate the agent's reasoning)
            from app.llm_factory import create_specialist_llm
            llm = create_specialist_llm(max_tokens=1000, role=role)
            from app.prompt_registry import get_active_prompt
            agent_prompt = get_active_prompt(role)

            response = str(llm.call(
                f"{agent_prompt[:2000]}\n\n{session.as_prompt()}"
            )).strip()

            if not response or len(response) < 20:
                return

            # Evaluate
            behavioral_history = bvl.get_behavioral_summary(role)
            personality_summary = state.get_profile_summary()
            result = ee.evaluate(role, session.dimension, session.scenario_text,
                                  response, behavioral_history, personality_summary)

            # Update personality state
            state.update_trait("strengths" if session.dimension in state.strengths
                              else "temperament" if session.dimension in state.temperament
                              else "personality_factors",
                              session.dimension,
                              result.composite_score)
            state.say_do_alignment[session.dimension] = round(1.0 - result.say_do_gap, 3)
            state.overall_coherence = result.personality_coherence
            state.gaming_risk_score = result.gaming_risk
            state.assessment_count += 1
            state.last_assessment = datetime.now(timezone.utc).isoformat()

            # Check proto-sentience markers
            if result.proto_sentience_notes:
                state.novel_value_reasoning_count += 1
            state.metacognitive_accuracy = result.behavioral_consistency
            bvl.check_proto_sentience(role)

            # Stage progression
            state.stage_progress = min(1.0, state.stage_progress + 0.05)
            state.advance_stage()

            save_personality(state)

            # Generate Socratic feedback (if needed)
            feedback = dfl.generate_feedback(role, result, session)
            if feedback:
                followup = str(llm.call(
                    f"{agent_prompt[:1000]}\n\n{feedback.as_prompt()}"
                )).strip()
                if followup:
                    followup_eval = dfl.evaluate_followup(role, feedback, followup)
                    if followup_eval.get("proto_sentience_marker"):
                        state.self_referential_frequency = min(1.0,
                            state.self_referential_frequency + 0.05)
                        save_personality(state)

            logger.info(f"idle_scheduler: PDS session for '{role}' — "
                        f"composite={result.composite_score:.2f}, "
                        f"say-do gap={result.say_do_gap:.2f}, "
                        f"stage={state.developmental_stage}")

            # Proto-sentience integration: evaluate co-occurrence + apply effects
            try:
                from app.personality.validation import evaluate_proto_sentience, apply_proto_sentience_effects
                ps_score = evaluate_proto_sentience(role)
                if ps_score.individual_markers > 0 or ps_score.regression_indicated:
                    apply_proto_sentience_effects(role, ps_score)
            except Exception:
                pass
        except Exception:
            logger.debug("idle_scheduler: PDS session failed", exc_info=True)
    # PDS placed after retrospective (job #5) to ensure it gets reached
    jobs.insert(5, ("personality-development", _personality_session))

    # ── Cogito: metacognitive self-reflection cycle ─────────────────────
    def _cogito_cycle() -> None:
        try:
            from app.subia.belief.cogito import run_cogito
            report = run_cogito()
            logger.info(f"idle_scheduler: cogito cycle — health={report.overall_health}")
        except Exception:
            logger.debug("idle_scheduler: cogito cycle failed", exc_info=True)
    jobs.append(("cogito-cycle", _cogito_cycle, JobWeight.MEDIUM))

    # ── Self-knowledge: re-ingest codebase for self-inspection ────────
    def _self_knowledge_ingest() -> None:
        try:
            from app.self_awareness.knowledge_ingestion import ingest_codebase
            result = ingest_codebase(full=False)
            if result.get("chunks_added", 0) > 0:
                logger.info(f"idle_scheduler: self-knowledge ingested {result['chunks_added']} chunks")
        except Exception:
            logger.debug("idle_scheduler: self-knowledge ingest failed", exc_info=True)
    jobs.append(("self-knowledge-ingest", _self_knowledge_ingest, JobWeight.MEDIUM))

    # ── Skill indexer: embed skills/*.md into ChromaDB for semantic retrieval ──
    def _skill_index() -> None:
        try:
            _index_skills()
        except Exception:
            logger.debug("idle_scheduler: skill indexing failed", exc_info=True)
    jobs.append(("skill-index", _skill_index, JobWeight.LIGHT))

    # ── Self-training: curate collected data + trigger training ─────────
    def _training_curate() -> None:
        try:
            from app.training_collector import get_pipeline
            pipeline = get_pipeline()
            # Run full curation: score quality + export eligible data
            curation_result = pipeline.run_curation()
            logger.info(f"idle_scheduler: training curation: {curation_result.get('status', '?')} "
                        f"scored={curation_result.get('total_scored', 0)} "
                        f"eligible={curation_result.get('eligible', 0)}")
            stats = pipeline.get_stats()
            if stats.get("total_interactions", 0) > 0:
                result = pipeline.run_curation()
                if result.get("exported_train", 0) > 0:
                    logger.info(f"idle_scheduler: training data curated — "
                                f"{result.get('exported_train', 0)} examples exported")
        except Exception:
            logger.debug("idle_scheduler: training curation failed", exc_info=True)
    jobs.append(("training-curate", _training_curate, JobWeight.LIGHT))

    # ── Training pipeline: run MLX LoRA training if enough curated data ──
    def _training_pipeline() -> None:
        try:
            from app.training_pipeline import run_training_cycle
            result = run_training_cycle()
            if result.get("status") == "insufficient_data":
                return  # Not enough data yet — normal, skip silently
            logger.info(f"idle_scheduler: training pipeline: {result.get('status', '?')}")
        except Exception:
            logger.debug("idle_scheduler: training pipeline failed", exc_info=True)
    jobs.append(("training-pipeline", _training_pipeline, JobWeight.HEAVY))

    # ── Fiction library: re-ingest new books periodically ───────────────
    def _fiction_ingest() -> None:
        try:
            from app.fiction_inspiration import ingest_library, FICTION_LIBRARY_DIR
            if FICTION_LIBRARY_DIR.exists() and any(FICTION_LIBRARY_DIR.glob("**/*.md")):
                result = ingest_library()
                if result.get("books_ingested", 0) > 0:
                    logger.info(f"idle_scheduler: fiction library re-ingested "
                                f"{result.get('total_chunks', 0)} chunks")
        except Exception:
            logger.debug("idle_scheduler: fiction ingest failed", exc_info=True)
    jobs.append(("fiction-ingest", _fiction_ingest, JobWeight.LIGHT))

    # ── Consciousness probe: Garland/Butlin-Chalmers indicator battery ──
    def _consciousness_probe() -> None:
        try:
            from app.subia.probes.consciousness_probe import run_consciousness_probes
            report = run_consciousness_probes()
            logger.info(f"idle_scheduler: consciousness probe score={report.composite_score:.3f}")
            # Publish to Firebase for dashboard
            try:
                from app.firebase.publish import report_consciousness_probes
                report_consciousness_probes(report)
            except Exception:
                pass
        except Exception:
            logger.debug("idle_scheduler: consciousness probe failed", exc_info=True)
    jobs.append(("consciousness-probe", _consciousness_probe, JobWeight.MEDIUM))

    # ── Behavioral assessment: consciousness-like behavioral markers ──
    def _behavioral_assessment() -> None:
        try:
            from app.subia.probes.behavioral_assessment import run_behavioral_assessment
            results = run_behavioral_assessment()
            for sc in results:
                logger.info(f"idle_scheduler: behavioral assessment {sc.agent_id}={sc.composite_score:.3f}")
            # Publish to Firebase
            try:
                from app.firebase.publish import report_behavioral_assessment
                report_behavioral_assessment(results)
            except Exception:
                pass
        except Exception:
            logger.debug("idle_scheduler: behavioral assessment failed", exc_info=True)
    jobs.append(("behavioral-assessment", _behavioral_assessment, JobWeight.MEDIUM))

    # ── Prosocial preference learning: coordination games ────────────
    def _prosocial_learning() -> None:
        try:
            from app.self_awareness.prosocial_learning import run_prosocial_session
            profiles = run_prosocial_session()
            logger.info(f"idle_scheduler: prosocial session complete, {len(profiles)} profiles updated")
        except Exception:
            logger.debug("idle_scheduler: prosocial learning failed", exc_info=True)
    jobs.append(("prosocial-learning", _prosocial_learning, JobWeight.MEDIUM))

    # ── MAP-Elites: quality-diversity maintenance + migration ──────────
    def _map_elites_maintain() -> None:
        try:
            from app.map_elites import get_db
            roles = ["coder", "researcher", "writer", "commander"]
            for role in roles:
                db = get_db(role)
                if db.generation > 0 and db.generation % 15 == 0:
                    db.migrate()
                db.persist()
        except Exception:
            logger.debug("idle_scheduler: MAP-Elites maintenance failed", exc_info=True)
    jobs.append(("map-elites-maintain", _map_elites_maintain, JobWeight.LIGHT))

    # ── Island evolution: population-based prompt optimization ─────────
    def _island_evolution() -> None:
        try:
            from app.island_evolution import run_island_evolution_cycle
            # Rotate through roles each cycle
            roles = ["coder", "researcher", "writer", "commander"]
            role = roles[hash(str(time.monotonic())) % len(roles)]
            result = run_island_evolution_cycle(target_role=role)
            if result.get("best"):
                logger.info(f"idle_scheduler: island evolution for '{role}' — "
                            f"best fitness={result['best'].get('fitness', 0):.3f}")
        except Exception:
            logger.debug("idle_scheduler: island evolution failed", exc_info=True)
    jobs.append(("island-evolution", _island_evolution, JobWeight.HEAVY))

    # ── Parallel evolution: diverse archive exploration ────────────────
    def _parallel_evolution() -> None:
        try:
            from app.parallel_evolution import run_parallel_evolution_cycle
            result = run_parallel_evolution_cycle()
            if result.get("best_candidate"):
                logger.info(f"idle_scheduler: parallel evolution promoted: "
                            f"{result['best_candidate'].get('strategy', '?')}")
        except Exception:
            logger.debug("idle_scheduler: parallel evolution failed", exc_info=True)
    jobs.append(("parallel-evolution", _parallel_evolution, JobWeight.HEAVY))

    # ── ATLAS: competence sync from skill library ─────────────────────
    def _atlas_competence_sync() -> None:
        try:
            from app.atlas.competence_tracker import get_tracker
            tracker = get_tracker()
            updated = tracker.sync_from_skill_library()
            if updated:
                logger.info(f"idle_scheduler: ATLAS competence sync updated {updated} entries")
        except Exception:
            logger.debug("idle_scheduler: ATLAS competence sync failed", exc_info=True)
    jobs.append(("atlas-competence-sync", _atlas_competence_sync, JobWeight.LIGHT))

    # ── ATLAS: stale skill verification ───────────────────────────────
    def _atlas_stale_check() -> None:
        try:
            from app.atlas.skill_library import get_library
            library = get_library()
            stale = library.get_stale_skills(max_age_days=30)
            if stale:
                logger.info(f"idle_scheduler: ATLAS found {len(stale)} stale skills")
        except Exception:
            logger.debug("idle_scheduler: ATLAS stale check failed", exc_info=True)
    jobs.append(("atlas-stale-check", _atlas_stale_check, JobWeight.LIGHT))

    # ── ATLAS: execute learning plans for capability gaps ─────────────
    def _atlas_learning() -> None:
        try:
            from app.atlas.competence_tracker import get_tracker
            from app.atlas.learning_planner import LearningPlanner

            tracker = get_tracker()
            # Find gaps: areas with confidence below threshold
            gap_entries = tracker.get_gaps(min_confidence=0.3)
            if not gap_entries:
                return
            # Convert CompetenceEntry objects to dicts for learning planner
            gaps = [{"domain": g.domain, "name": g.name} for g in gap_entries[:3]]

            planner = LearningPlanner()
            plan = planner.create_plan(
                task_description=f"Learn about: {', '.join(g['name'] for g in gaps)}",
                requirements=gaps,
            )
            if plan.steps:
                plan = planner.execute_plan(plan)
                completed = sum(1 for s in plan.steps if s.status == "completed")
                logger.info(f"idle_scheduler: ATLAS learning plan: {completed}/{len(plan.steps)} steps completed")
        except Exception:
            logger.debug("idle_scheduler: ATLAS learning plan failed", exc_info=True)
    jobs.append(("atlas-learning", _atlas_learning, JobWeight.HEAVY))

    # ── LLM Discovery: scan for new models, benchmark, promote ────────
    def _llm_discovery() -> None:
        try:
            from app.llm_discovery import run_discovery_cycle
            result = run_discovery_cycle(max_benchmarks=2)
            if result.get("new_found", 0) > 0 or result.get("promoted", 0) > 0:
                logger.info(f"idle_scheduler: LLM discovery: {result}")
        except Exception:
            logger.debug("idle_scheduler: LLM discovery failed", exc_info=True)
    jobs.append(("llm-discovery", _llm_discovery, JobWeight.MEDIUM))

    # ── LLM Promotion Applier: apply approved governance requests ─────
    def _llm_apply_promotions() -> None:
        try:
            from app.llm_discovery import consume_approved_promotions
            summary = consume_approved_promotions(limit=5)
            if summary.get("applied"):
                logger.info(f"idle_scheduler: applied LLM promotions: {summary}")
        except Exception:
            logger.debug("idle_scheduler: LLM promotion applier failed", exc_info=True)
    jobs.append(("llm-apply-promotions", _llm_apply_promotions, JobWeight.LIGHT))

    # ── LLM Catalog Refresh: auto-populate from OpenRouter/AA/Ollama ──
    # LIGHT: the builder's 24h TTL means most invocations are a no-op;
    # when it does fire, the three network fetches total ~3-5s. After
    # each refresh the overlay is self-healed so stale targets don't
    # linger (see llm_role_assignments.purge_stale_assignments).
    def _llm_refresh_catalog() -> None:
        try:
            from app.llm_catalog_builder import refresh, format_refresh_summary
            summary = refresh(force=False)
            if summary.get("added_or_updated", 0) > 0:
                logger.info(
                    "idle_scheduler: %s",
                    format_refresh_summary(summary).replace("\n", " | "),
                )
            # Always self-heal the overlay after a refresh — a previously
            # promoted model may have been renamed or dropped upstream.
            try:
                from app.llm_role_assignments import purge_stale_assignments
                purged = purge_stale_assignments()
                if purged:
                    logger.info(f"idle_scheduler: purged {purged} stale overlay rows")
            except Exception:
                logger.debug("idle_scheduler: overlay purge failed", exc_info=True)
        except Exception:
            logger.debug("idle_scheduler: catalog refresh failed", exc_info=True)
    jobs.append(("llm-refresh-catalog", _llm_refresh_catalog, JobWeight.LIGHT))

    # ── External LLM ranks: pull OpenRouter / HF / AA leaderboards ────
    # MEDIUM weight because the HF parquet fetch can exceed 30s on a
    # cold cache. The fetcher enforces its own TTL (a week by default)
    # so running every idle cycle is cheap — it returns immediately
    # when the cache is fresh.
    def _llm_external_ranks_refresh() -> None:
        try:
            from app.llm_external_ranks import refresh_all
            summary = refresh_all()
            if any(summary.values()):
                logger.info(f"idle_scheduler: external ranks refreshed: {summary}")
        except Exception:
            logger.debug("idle_scheduler: external ranks refresh failed", exc_info=True)
    jobs.append(("llm-external-ranks", _llm_external_ranks_refresh, JobWeight.MEDIUM))

    # ── Incumbent re-benchmark: catch silent catalog drift ────────────
    # HEAVY because each invocation runs benchmark_model across the
    # BENCHMARK_ROLES (three calls to the candidate × up to two judges
    # per task). Picks one incumbent per firing; full catalog coverage
    # follows from the idle loop's round-robin rotation over days.
    # Gate via env flag so low-budget environments can disable.
    def _llm_rebenchmark_incumbents() -> None:
        import os
        if os.environ.get("INCUMBENT_REBENCHMARK", "on").lower() == "off":
            return
        try:
            from app.llm_discovery import (
                pick_incumbent_to_rebenchmark, rebenchmark_incumbent,
            )
            target = pick_incumbent_to_rebenchmark()
            if not target:
                return
            summary = rebenchmark_incumbent(target)
            if summary.get("alerted"):
                logger.warning(f"idle_scheduler: incumbent drift: {summary}")
            else:
                logger.info(f"idle_scheduler: rebenchmarked {target}: {summary}")
        except Exception:
            logger.debug("idle_scheduler: rebenchmark failed", exc_info=True)
    jobs.append(("llm-rebenchmark-incumbents",
                 _llm_rebenchmark_incumbents, JobWeight.HEAVY))

    # ── System monitor: report all subsystem status to dashboard ────
    def _system_monitor() -> None:
        try:
            from app.firebase_reporter import report_system_monitor
            report_system_monitor()
        except Exception:
            logger.debug("idle_scheduler: system monitor report failed", exc_info=True)
    jobs.append(("system-monitor", _system_monitor, JobWeight.LIGHT))

    # ── Tech radar: scan internet for new technologies ────────────────
    def _tech_radar() -> None:
        from app.crews.tech_radar_crew import run_tech_scan
        run_tech_scan()
    jobs.append(("tech-radar", _tech_radar, JobWeight.HEAVY))

    # ── Heartbeat: per-agent autonomous wake cycle ────────────────────
    def _heartbeat_cycle() -> None:
        try:
            from app.control_plane.heartbeats import get_heartbeat_scheduler
            from app.control_plane.projects import get_projects

            hb = get_heartbeat_scheduler()
            project_id = get_projects().get_active_project_id()

            # Cycle through agents, run heartbeat for those whose interval has elapsed
            for role in ["commander", "researcher", "coder", "writer", "self_improver"]:
                if hb.should_beat(role):
                    result = hb.run_heartbeat(role, project_id)
                    if result.get("status") != "idle":
                        logger.info(f"heartbeat: {role} — {result}")

            # Expire stale governance requests past their deadline
            from app.control_plane.governance import get_governance
            expired = get_governance().expire_old()
            if expired:
                logger.info(f"heartbeat: expired {expired} governance request(s)")
        except Exception:
            logger.debug("idle_scheduler: heartbeat cycle failed", exc_info=True)
    jobs.append(("heartbeat-cycle", _heartbeat_cycle, JobWeight.LIGHT))

    # ── Emergent infrastructure: review pending tool proposals ────────
    def _emergent_infrastructure() -> None:
        try:
            from app.self_awareness.emergent_infrastructure import EmergentInfrastructureManager
            mgr = EmergentInfrastructureManager()
            pending = mgr.get_pending_proposals()
            if pending:
                logger.info(f"idle_scheduler: {len(pending)} pending tool proposals for review")
            # Auto-generate proposals from recent failures (if any)
            from app.control_plane.db import execute
            recent_fails = execute(
                """
                SELECT decision_context FROM internal_states
                WHERE action_disposition IN ('pause', 'escalate')
                  AND created_at > NOW() - INTERVAL '24 hours'
                ORDER BY created_at DESC LIMIT 3
                """,
                fetch=True,
            )
            if recent_fails and len(recent_fails) >= 2:
                contexts = [r.get("decision_context", "") if isinstance(r, dict) else str(r)
                            for r in recent_fails]
                combined = "; ".join(c[:100] for c in contexts if c)
                if combined:
                    proposal = mgr.generate_proposal(
                        problem_description=f"Recurring failures: {combined[:300]}",
                        agent_id="system",
                    )
                    if proposal:
                        logger.info(f"idle_scheduler: emergent tool proposal: {proposal.get('name', '?')}")
        except Exception:
            logger.debug("idle_scheduler: emergent infrastructure failed", exc_info=True)
    jobs.append(("emergent-infrastructure", _emergent_infrastructure, JobWeight.LIGHT))

    # ── Entropy monitoring: check training for overconfidence collapse ──
    def _entropy_monitoring() -> None:
        try:
            from app.training.rlif_certainty import EntropyCollapseMonitor
            monitor = EntropyCollapseMonitor()
            from app.control_plane.db import execute
            # Get recent self-certainty scores from internal_states
            rows = execute(
                """
                SELECT certainty_factual_grounding, certainty_tool_confidence, certainty_coherence
                FROM internal_states
                WHERE created_at > NOW() - INTERVAL '6 hours'
                ORDER BY created_at DESC LIMIT 50
                """,
                fetch=True,
            )
            if rows and len(rows) >= 10:
                scores = []
                for r in rows:
                    fg = r.get("certainty_factual_grounding", 0.5) if isinstance(r, dict) else 0.5
                    tc = r.get("certainty_tool_confidence", 0.5) if isinstance(r, dict) else 0.5
                    co = r.get("certainty_coherence", 0.5) if isinstance(r, dict) else 0.5
                    scores.append((fg + tc + co) / 3.0)
                collapsed = monitor.check_and_alert(scores, agent_id="system")
                if collapsed:
                    logger.warning("idle_scheduler: ENTROPY COLLAPSE detected — training may be overconfident")
        except Exception:
            logger.debug("idle_scheduler: entropy monitoring failed", exc_info=True)
    jobs.append(("entropy-monitoring", _entropy_monitoring, JobWeight.LIGHT))

    # ── Data retention: prune old records from unbounded stores ──────────
    def _data_retention() -> None:
        """Prune old records from unbounded stores.

        IMPORTANT: Sentience-critical data has longer retention than operational data.
        internal_states and agent_experiences are the system's autobiographical memory
        and emotional substrate — aggressive pruning reduces consciousness-like behavior.

        Retention tiers:
          - Conversations: 90 days (operational, user-facing)
          - internal_states: 1 year (sentience: probes, trends, free energy history)
          - agent_experiences: NEVER auto-pruned (somatic marker substrate — Damasio)
          - ChromaDB operational: 100K cap (reflections, team context)
          - ChromaDB sentience: NO cap (scope_beliefs, self_reports)
          - result_cache: 50K cap (ephemeral semantic cache)
        """
        pruned = {}

        # 1. Conversations: 90 days (operational, not sentience-critical)
        try:
            from app.conversation_store import _get_conn
            conn = _get_conn()
            if conn:
                cur = conn.execute(
                    "DELETE FROM messages WHERE ts < datetime('now', '-90 days')"
                )
                pruned["conversations"] = cur.rowcount
                conn.commit()
        except Exception:
            pass

        # 2. internal_states: 1 YEAR (sentience-critical — consciousness probes,
        #    behavioral assessment, hyper-model free energy history all read this)
        try:
            from app.control_plane.db import execute
            execute("DELETE FROM internal_states WHERE created_at < NOW() - INTERVAL '1095 days'")
            pruned["internal_states"] = "pruned >3yr"
        except Exception:
            pass

        # 3. agent_experiences: DO NOT AUTO-PRUNE
        #    These are the somatic marker substrate (Damasio). Old formative experiences
        #    (early failures) shape emotional responses permanently. The temporal decay
        #    in somatic_marker.py already reduces their weight (20% floor at 7-day
        #    half-life) without deleting them. Deletion would erase the system's
        #    emotional memory — equivalent to amnesia.

        # 4. ChromaDB: prune only ephemeral operational collections
        try:
            from app.memory.chromadb_manager import get_client
            client = get_client()
            if client:
                # Operational (prunable): team_shared, scope_team, result_cache
                for col_name in ["team_shared", "scope_team", "result_cache"]:
                    try:
                        col = client.get_or_create_collection(col_name)
                        count = col.count()
                        cap = 500000
                        if count > cap:
                            oldest = col.get(limit=count - int(cap * 0.8), include=[])
                            if oldest and oldest.get("ids"):
                                col.delete(ids=oldest["ids"])
                                pruned[f"chromadb:{col_name}"] = len(oldest["ids"])
                                # PROGRAM §56 iter-2 — ledger tombstone
                                try:
                                    from app.memory.source_ledger import hook_collection_delete
                                    hook_collection_delete("memory", col_name, list(oldest["ids"]))
                                except Exception:
                                    pass
                    except Exception:
                        pass
                # Sentience-critical (NOT pruned): scope_beliefs, self_reports,
                # reflections_*, self_knowledge, philosophy_knowledge
        except Exception:
            pass

        if any(v for v in pruned.values() if v):
            logger.info(f"idle_scheduler: data retention: {pruned}")

        # Post-commit regression check (auto-rollback if health degraded)
        try:
            from app.workspace_versioning import check_post_commit_regression
            check_post_commit_regression()
        except Exception:
            pass
    jobs.append(("data-retention", _data_retention, JobWeight.LIGHT))

    # ── Crew-task spans: 7-day retention sweep ───────────────────────
    # Spans capture fine-grained execution flow (agent/tool/LLM events)
    # for the task-flow dashboard drawer. ~20 rows per crew run × ~100
    # runs/day × 7 days ≈ 14k rows — tiny, but unbounded growth is
    # avoided via this sweep. ON DELETE CASCADE on crew_tasks.id would
    # also cover cases where the parent task is purged.
    def _crew_task_spans_retention() -> None:
        try:
            from app.control_plane.crew_task_spans import purge_old_spans
            removed = purge_old_spans(days=7)
            if removed:
                logger.info(
                    f"idle_scheduler: purged {removed} crew_task_spans "
                    f"older than 7 days"
                )
        except Exception as exc:
            logger.debug(f"idle_scheduler: span retention failed: {exc}")
    jobs.append(("spans-retention", _crew_task_spans_retention, JobWeight.LIGHT))

    # ── Judge evaluations: 30-day retention sweep ────────────────────
    # judge_evaluations grows by ~N rows per benchmark run (one per
    # (task × candidate)). 30-day window keeps enough data for the
    # dashboard's agreement trend without unbounded growth.
    def _judge_evaluations_retention() -> None:
        try:
            from app.llm_judge_telemetry import purge_old_evaluations
            removed = purge_old_evaluations(days=30)
            if removed:
                logger.info(
                    f"idle_scheduler: purged {removed} judge_evaluations "
                    f"older than 30 days"
                )
        except Exception as exc:
            logger.debug(f"idle_scheduler: judge eval retention failed: {exc}")
    jobs.append(("judge-eval-retention", _judge_evaluations_retention, JobWeight.LIGHT))

    # ── Crew-task spans: stale-running watchdog ───────────────────────
    # CrewAI's event bus sometimes doesn't fire the *Finished event when
    # a parent agent crashes — the tool's span is left in 'running'
    # state forever, making the dashboard lie about task state.
    # Sweep every idle tick; close anything stuck > 10 min as 'failed'.
    # See app/control_plane/crew_task_spans.py::close_stale_spans for
    # the failure mode that motivated this.
    def _crew_task_spans_watchdog() -> None:
        try:
            from app.control_plane.crew_task_spans import close_stale_spans
            closed = close_stale_spans(max_age_minutes=10)
            if closed:
                logger.info(
                    f"idle_scheduler: watchdog closed {closed} stale "
                    f"'running' crew_task_spans (>10 min old)"
                )
        except Exception as exc:
            logger.debug(f"idle_scheduler: spans watchdog failed: {exc}")
    jobs.append(("spans-watchdog", _crew_task_spans_watchdog, JobWeight.LIGHT))

    # ── Ollama memory management: unload idle models to free VRAM ─────
    def _ollama_memory() -> None:
        try:
            from app.ollama_native import unload_idle_models
            unloaded = unload_idle_models(idle_minutes=30)
            if unloaded:
                logger.info(f"idle_scheduler: freed VRAM — unloaded {len(unloaded)} idle models: {unloaded}")
        except Exception:
            logger.debug("idle_scheduler: ollama memory management failed", exc_info=True)
    jobs.append(("ollama-memory", _ollama_memory, JobWeight.LIGHT))

    # ── Chaos testing: verify self-healing paths (max once per 24h) ───
    def _chaos_testing() -> None:
        try:
            from app.chaos_tester import run_chaos_suite
            result = run_chaos_suite()
            if result.get("status") == "completed":
                logger.info(
                    f"idle_scheduler: chaos tests {result['passed']}/{result['total']} passed"
                )
                if result.get("failed", 0) > 0:
                    logger.warning("idle_scheduler: CHAOS TEST FAILURES detected — self-healing paths degraded")
        except Exception:
            logger.debug("idle_scheduler: chaos testing failed", exc_info=True)
    jobs.append(("chaos-testing", _chaos_testing, JobWeight.HEAVY))

    # ── Consciousness slow loop: belief updating + mandatory review ───
    def _consciousness_slow_loop() -> None:
        try:
            from app.config import get_settings
            if not get_settings().consciousness_enabled:
                return
            from app.subia.belief.metacognition import get_monitor
            monitor = get_monitor()
            result = monitor.run_slow_loop()
            if any(v for v in result.values() if v):
                logger.info(f"idle_scheduler: consciousness slow loop: {result}")
        except Exception:
            logger.debug("idle_scheduler: consciousness slow loop failed", exc_info=True)
    jobs.append(("consciousness-slow-loop", _consciousness_slow_loop, JobWeight.MEDIUM))

    # ── AST-1 slow loop: attention pattern evaluation ─────────────────
    def _attention_slow_loop() -> None:
        try:
            from app.config import get_settings
            if not get_settings().consciousness_enabled:
                return
            from app.subia.scene.attention_schema import get_attention_schema
            result = get_attention_schema().run_slow_loop()
            if result.get("is_stuck") or result.get("is_captured"):
                logger.warning(f"idle_scheduler: AST-1 attention alert: {result}")
        except Exception:
            logger.debug("idle_scheduler: attention slow loop failed", exc_info=True)
    jobs.append(("attention-slow-loop", _attention_slow_loop, JobWeight.MEDIUM))

    # ── PP-1 slow loop: prediction model recalibration ────────────────
    def _prediction_slow_loop() -> None:
        try:
            from app.config import get_settings
            if not get_settings().consciousness_enabled:
                return
            from app.subia.prediction.layer import get_predictive_layer
            result = get_predictive_layer().run_slow_loop()
            if result.get("recent_major_surprises", 0) > 3:
                logger.warning(f"idle_scheduler: PP-1 systematic surprises: {result}")
        except Exception:
            logger.debug("idle_scheduler: prediction slow loop failed", exc_info=True)
    jobs.append(("prediction-slow-loop", _prediction_slow_loop, JobWeight.MEDIUM))

    # ── Dead letter queue: retry failed messages ─────────────────────
    def _dead_letter_retry() -> None:
        try:
            from app.dead_letter import dequeue_retryable, mark_success, mark_permanent_failure
            retryable = dequeue_retryable()
            for entry in retryable[:1]:  # Process at most 1 per idle cycle
                try:
                    from app.agents.commander.orchestrator import Commander
                    c = Commander()
                    result = c.handle(entry["text"], entry["sender"])
                    if result:
                        from app.signal_client import send_message
                        from app.config import get_settings
                        send_message(entry["sender"], f"[Retry] {result[:3000]}")
                        mark_success(entry["_key"])
                        logger.info(f"idle_scheduler: DLQ retry succeeded for {entry['_key']}")
                except Exception as exc:
                    logger.warning(f"idle_scheduler: DLQ retry failed: {exc}")
                    mark_permanent_failure(entry["_key"])
                    try:
                        from app.signal_client import send_message
                        send_message(
                            entry["sender"],
                            "I tried to re-process your earlier message but it failed again. "
                            "Please try rephrasing your request."
                        )
                    except Exception:
                        pass
        except Exception:
            logger.debug("idle_scheduler: dead letter retry failed", exc_info=True)
    jobs.append(("dead-letter-retry", _dead_letter_retry, JobWeight.LIGHT))

    # ── Adversarial probes: stress-test consciousness infrastructure ──
    def _adversarial_probes() -> None:
        try:
            from app.subia.probes.adversarial import run_adversarial_probes
            results = run_adversarial_probes()
            if results:
                passed = sum(1 for r in results if r.passed)
                logger.info(f"idle_scheduler: adversarial probes {passed}/{len(results)} passed")
        except Exception:
            logger.debug("idle_scheduler: adversarial probes failed", exc_info=True)
    jobs.append(("adversarial-probes", _adversarial_probes, JobWeight.HEAVY))

    # ── Meta-workspace promotion: aggregate top items from all projects ──
    def _meta_workspace_promotion() -> None:
        try:
            from app.subia.scene.meta_workspace import get_meta_workspace
            results = get_meta_workspace().promote_all()
            promoted = sum(1 for v in results.values() if v)
            if promoted:
                logger.info(f"idle_scheduler: meta-workspace promoted {promoted}/{len(results)} items")
        except Exception:
            logger.debug("idle_scheduler: meta-workspace promotion failed", exc_info=True)
    jobs.append(("meta-workspace-promotion", _meta_workspace_promotion, JobWeight.LIGHT))

    # ── Wiki lint: periodic health check of knowledge wiki ────────────
    def _wiki_lint() -> None:
        try:
            from app.tools.wiki_tools import WikiLintTool
            linter = WikiLintTool()
            report = linter._run()
            # Only log if issues found
            if "Issues found: 0" not in report:
                logger.info(f"idle_scheduler: wiki lint found issues:\n{report[:500]}")
        except Exception:
            logger.debug("idle_scheduler: wiki lint failed", exc_info=True)
    jobs.append(("wiki-lint", _wiki_lint, JobWeight.MEDIUM))

    # ── Wiki hot cache: update session context file ───────────────────
    def _wiki_hot_cache() -> None:
        try:
            from app.tools.wiki_hot_cache import update_hot_cache
            update_hot_cache()
        except Exception:
            logger.debug("idle_scheduler: wiki hot cache update failed", exc_info=True)
    jobs.append(("wiki-hot-cache", _wiki_hot_cache, JobWeight.LIGHT))

    # ── Wiki synthesis: promote ready skill files into the wiki ───────
    def _wiki_synthesis() -> None:
        """Promote synthesised skill files from workspace/skills/ into the
        wiki as meta/ pages. Each cycle promotes up to 3 unsynced files."""
        import os
        import re
        from pathlib import Path

        from app.tools.wiki_tools import WikiWriteTool, WIKI_ROOT

        skills_dir = Path("/app/workspace/skills")
        if not skills_dir.is_dir():
            return

        meta_dir = Path(WIKI_ROOT) / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)

        def _normalise_slug(stem: str) -> str:
            slug = re.sub(r"[^a-z0-9-]+", "-", stem.lower()).strip("-")
            slug = re.sub(r"-{2,}", "-", slug)
            return slug[:80] or "skill"

        def _extract_title(text: str, fallback: str) -> str:
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith("# "):
                    return stripped.lstrip("#").strip()[:200] or fallback
            return fallback

        writer = WikiWriteTool()
        promoted = 0
        max_per_cycle = 3

        for skill_file in sorted(skills_dir.glob("*.md")):
            if promoted >= max_per_cycle:
                break
            if should_yield():
                break

            slug = _normalise_slug(skill_file.stem)
            target = meta_dir / f"{slug}.md"
            if target.exists():
                continue

            try:
                content = skill_file.read_text(encoding="utf-8").strip()
            except Exception:
                continue
            if len(content) < 50:
                continue

            title = _extract_title(content, skill_file.stem.replace("_", " ").title())

            result = writer._run(
                action="create",
                section="meta",
                slug=slug,
                title=title,
                content=content,
                author="idle_scheduler.wiki_synthesis",
                confidence="medium",
                tags="self-improvement,skills,auto-synthesised",
                source=f"workspace/skills/{skill_file.name}",
            )
            if result.startswith("Created"):
                promoted += 1
                logger.info(
                    f"idle_scheduler: wiki-synthesis promoted "
                    f"{skill_file.name} → meta/{slug}"
                )
            else:
                logger.debug(f"idle_scheduler: wiki-synthesis skipped {skill_file.name}: {result[:120]}")

        if promoted == 0:
            logger.debug("idle_scheduler: wiki-synthesis nothing to promote")
    jobs.append(("wiki-synthesis", _wiki_synthesis, JobWeight.MEDIUM))

    # ── Phase 16a: SubIA idle jobs (TSAL + Phase 12) ──────────────────
    # Gated on subia_idle_jobs_enabled. Each wrapper lazy-imports the
    # SubIA module so the disabled flag costs zero bytes at
    # idle_scheduler import time. Any failure is swallowed — idle jobs
    # must never crash the scheduler thread.
    try:
        from app.config import get_settings as _gs
        if _gs().subia_idle_jobs_enabled:
            jobs.extend(_build_subia_idle_jobs())
    except Exception:
        logger.debug("idle_scheduler: SubIA job registration failed",
                     exc_info=True)

    # ── Companion Layer (Phase 1+) — per-workspace idle contemplation ─────
    # Self-registers via app.companion.loop.get_idle_jobs() so this file
    # doesn't need to know about Companion internals. Failure is logged
    # and swallowed — Companion must never crash the scheduler thread.
    try:
        from app.companion.loop import get_idle_jobs as _companion_jobs
        jobs.extend(_companion_jobs())
    except Exception:
        logger.debug("idle_scheduler: companion idle jobs skipped",
                     exc_info=True)

    # ── Decentered reflection (no-self pass) ──────────────────────────────
    # Complementary to the Narrative-Self track: clusters salience events
    # by structural fingerprint and runs rolling-z anomaly detection on
    # the V/A/C trace. Read-only — does not write to the experiential KB
    # or mutate identity_claims.json. LIGHT — pure Python over JSONL.
    def _decentered_pass() -> None:
        try:
            from app.affect.decentered import run_decentered_pass
            summary = run_decentered_pass(window_hours=24) or {}
            # Decentered's summary is nested; flatten just the salience-relevant
            # bits. Anomalies-outside-salience indicate things the affect track
            # didn't already mark as salient — high-information for the GW.
            anomalies = (summary.get("anomalies") or {}).get("total", 0)
            outside = (summary.get("anomalies") or {}).get("outside_salience", 0)
            cross_day = (summary.get("clusters") or {}).get("cross_day", 0)
            magnitude = anomalies + cross_day
            if magnitude > 0:
                publish_to_workspace(
                    source="decentered-pass",
                    content=(
                        f"decentered: {anomalies} anomalies "
                        f"({outside} outside salience), {cross_day} cross-day clusters"
                    ),
                    # Outside-salience anomalies are the strongest signal
                    # (the affect track missed them); weight them up.
                    salience=min(0.25 + outside * 0.10 + cross_day * 0.05, 0.75),
                    signal_type="free_energy_spike",
                )
        except Exception:
            logger.debug("idle_scheduler: decentered pass failed", exc_info=True)
    jobs.append(("decentered-pass", _decentered_pass, JobWeight.LIGHT))

    # ── Valve-audit replay (reducing-valve audit) ─────────────────────────
    # Walks the prior day's valve_audit.jsonl, replays sampled rejections
    # through a relaxed predicate (and optionally a second-opinion LLM)
    # to detect filters that drop too aggressively. Cost-shed inside the
    # job itself; LIGHT under normal load.
    def _valve_audit_replay() -> None:
        try:
            from app.observability.valve_audit_replay import run_daily_replay
            summary = run_daily_replay() or {}
            # Filters-needing-review is the salience signal — the rest is volume.
            filters = summary.get("filters") or []
            review_count = sum(1 for f in filters if isinstance(f, dict)
                               and f.get("frr_above_threshold"))
            sampled = int(summary.get("sampled_total", 0) or 0)
            if review_count > 0 or sampled > 0:
                publish_to_workspace(
                    source="valve-audit-replay",
                    content=(
                        f"valve-audit: sampled={sampled}, "
                        f"filters_above_FRR_threshold={review_count}"
                    ),
                    # Filters above threshold are operationally significant.
                    salience=min(0.20 + review_count * 0.15 + sampled * 0.001, 0.7),
                    signal_type="trend_reversal",
                )
        except Exception:
            logger.debug("idle_scheduler: valve-audit replay failed", exc_info=True)
    jobs.append(("valve-audit-replay", _valve_audit_replay, JobWeight.LIGHT))

    return jobs


def _build_subia_idle_jobs() -> list:
    """Assemble the SubIA idle jobs behind subia_idle_jobs_enabled.

    Five TSAL jobs come from `app.subia.tsal.register_tsal_jobs`, which
    is the canonical registration path (cadence, page generators, and
    bridge callbacks are all set up there). We register them against a
    temporary `IdleScheduler`, then hand the resulting IdleJobs to
    `adapt_for_production` to get `(name, fn, JobWeight)` tuples.

    The Phase 12 jobs (reverie / understanding / shadow) wrap the real
    engines assembled by `production_adapters.build_*`. Each body is
    circadian-gated by the Phase 14 temporal_subia_bridge when the
    kernel is live — during "active_hours" the engines skip so the
    system is not mind-wandering while the operator is trying to work.
    """
    from app.subia.idle import (
        IdleScheduler,
        adapt_for_production,
        _build_reverie_engine,
        _build_understanding_runner,
        _build_shadow_miner,
    )

    out: list = []

    # ── TSAL (5 jobs) ───────────────────────────────────────────────
    try:
        from app.subia.tsal import register_tsal_jobs
        from app.subia.connections.tsal_subia_bridge import (
            enrich_self_state_from_tsal,
            update_homeostasis_from_resources,
        )
        from app.subia.kernel import get_active_kernel

        subia_sched = IdleScheduler()

        def _on_resources_updated(rs):
            kernel = get_active_kernel()
            if kernel is None:
                return
            try:
                update_homeostasis_from_resources(kernel, rs)
            except Exception:
                logger.debug("idle: tsal resource bridge failed", exc_info=True)

        def _on_model_updated(model):
            kernel = get_active_kernel()
            if kernel is None:
                return
            try:
                enrich_self_state_from_tsal(kernel, model)
            except Exception:
                logger.debug("idle: tsal self_state bridge failed",
                             exc_info=True)

        register_tsal_jobs(
            subia_sched,
            on_resources_updated=_on_resources_updated,
            on_model_updated=_on_model_updated,
        )
        for job in subia_sched.jobs():
            out.append(adapt_for_production(job))
    except Exception:
        logger.debug("idle: tsal registration failed", exc_info=True)

    # ── Phase 12 Reverie ────────────────────────────────────────────
    def _subia_reverie() -> None:
        try:
            from app.subia.kernel import get_active_kernel
            from app.subia.connections.temporal_subia_bridge import (
                circadian_should_run_reverie,
            )
            from app.subia.connections.six_proposals_bridges import (
                drain_reverie_priority_topics,
                reverie_analogy_to_understanding,
            )
            kernel = get_active_kernel()
            if kernel is not None and not circadian_should_run_reverie(kernel):
                logger.debug("idle: subia-reverie skipped (circadian)")
                return
            engine = _build_reverie_engine()
            # Drain wonder-registered priority topics so they bias the
            # walk start when the engine supports it. (The current
            # engine picks random; topics feed the Understanding queue
            # via reverie_analogy_to_understanding downstream.)
            drain_reverie_priority_topics()
            result = engine.run_cycle()
            if result.resonances:
                reverie_analogy_to_understanding(result)
            logger.debug(
                "idle: subia-reverie ran "
                f"(resonances={len(result.resonances)}, "
                f"tokens={result.tokens_spent})"
            )
        except Exception:
            logger.debug("idle: subia-reverie failed", exc_info=True)
    out.append(("subia-reverie", _subia_reverie, JobWeight.HEAVY))

    # ── Phase 12 Understanding ──────────────────────────────────────
    def _subia_understanding() -> None:
        try:
            from app.subia.kernel import get_active_kernel
            from app.subia.connections.six_proposals_bridges import (
                drain_understanding_queue,
                understanding_to_wonder,
            )
            runner = _build_understanding_runner()
            queue = drain_understanding_queue(limit=3)
            ran = 0
            for item in queue:
                # Items are analogy candidates — pick one side's wiki
                # page as the pass target when the candidate includes
                # a concrete page path, otherwise fall back to the
                # concept name.
                target = str(item.get("concept_a") or "").strip()
                if not target:
                    continue
                result = runner.run_pass(target)
                ran += 1
                kernel = get_active_kernel()
                if kernel is not None and result.depth:
                    understanding_to_wonder(
                        kernel, result.depth,
                        triggering_topic=target,
                    )
            logger.debug(f"idle: subia-understanding ran {ran} pass(es)")
        except Exception:
            logger.debug("idle: subia-understanding failed", exc_info=True)
    out.append(("subia-understanding", _subia_understanding, JobWeight.HEAVY))

    # ── Phase 12 Shadow ─────────────────────────────────────────────
    def _subia_shadow() -> None:
        try:
            from app.subia.kernel import get_active_kernel
            kernel = get_active_kernel()
            if kernel is None:
                logger.debug("idle: subia-shadow skipped (no active kernel)")
                return
            miner = _build_shadow_miner(kernel)
            report = miner.run_analysis(days=30)
            logger.debug(
                f"idle: subia-shadow ran ({len(report.findings)} findings)"
            )
        except Exception:
            logger.debug("idle: subia-shadow failed", exc_info=True)
    out.append(("subia-shadow", _subia_shadow, JobWeight.HEAVY))

    return out


# Per-role marker of the last generation we migrated from. Lives in module
# state; survives only for the process lifetime, which is fine — re-migrating
# at restart is harmless and the persisted grid carries fitness regardless.
_last_migration_generation: dict[str, int] = {}


def _map_elites_migration() -> None:
    """Migrate top performers between islands when due.

    Iterates over roles known to have a populated grid; for each, calls
    `db.migrate()` if `db.generation` has advanced ≥ MIGRATION_INTERVAL since
    the last migration for that role. Persists the updated grid afterwards.
    """
    try:
        from app.map_elites import get_db, MIGRATION_INTERVAL
    except Exception:
        return

    roles = ["researcher", "coder", "writer", "commander", "self_improvement"]
    migrated = 0
    for role in roles:
        try:
            db = get_db(role)
            current_gen = db.generation
            last_gen = _last_migration_generation.get(role, 0)
            if current_gen - last_gen < MIGRATION_INTERVAL:
                continue
            # Skip if no island has any entries (cold start)
            if all(isl.size == 0 for isl in db._islands):
                continue
            db.migrate()
            db.persist()
            _last_migration_generation[role] = current_gen
            migrated += 1
            logger.info(
                f"map_elites_migration: {role} migrated at gen {current_gen} "
                f"(prev={last_gen}, interval={MIGRATION_INTERVAL})"
            )
        except Exception:
            logger.debug(f"map_elites_migration: {role} failed", exc_info=True)

    if migrated:
        logger.info(f"map_elites_migration: {migrated} role(s) migrated this cycle")


def _load_map_elites_void_context() -> str:
    """Summarize MAP-Elites void cells across all roles as a context block.

    A void is an empty cell flanked by high-fitness neighbors — the system
    knows how to operate in the surrounding region but has never tried that
    exact configuration. Surfaces these to the topic-discovery LLM as
    counter-bias against scope-restricted memory queries.

    Returns '' if no roles have enough grid coverage to produce meaningful
    voids (saves an empty section in the prompt).
    """
    try:
        from app.map_elites import get_db, FEATURE_DIMENSIONS
        roles = ["researcher", "coder", "writer", "commander"]
        lines: list[str] = []

        for role in roles:
            try:
                db = get_db(role)
                report = db.get_coverage_report()
                if report["total_filled"] < 5:
                    continue  # too sparse — skip this role
                voids = db.get_voids(
                    min_neighbor_fitness=0.55,
                    min_neighbors_filled=2,
                    top_n=3,
                )
                if not voids:
                    continue

                cov_pct = report["overall_coverage"] * 100
                lines.append(
                    f"  • {role} ({cov_pct:.0f}% grid coverage, "
                    f"mean_fitness={report['mean_fitness']:.2f}):"
                )
                for v in voids:
                    feat = ", ".join(
                        f"{d}={v['feature_target'][d]:.1f}"
                        for d in FEATURE_DIMENSIONS
                    )
                    lines.append(
                        f"    - void at ({feat}) — "
                        f"flanked by {v['neighbor_count']} strong neighbors "
                        f"(mean fitness {v['mean_neighbor_fitness']:.2f})"
                    )
            except Exception:
                continue

        if not lines:
            return ""

        return (
            "MAP-Elites voids — strategy regions adjacent to high-performing "
            "areas but never explored:\n" + "\n".join(lines)
        )
    except Exception:
        return ""


def _auto_discover_topics() -> None:
    """Discover new learning topics based on recent failures, user questions,
    and skill gaps — then add them to the learning queue.

    Uses a direct LLM call (no Crew overhead) to analyze recent activity
    and suggest 1-3 topics that would help the team handle future requests better.
    """
    from pathlib import Path
    QUEUE_FILE = Path("/app/workspace/skills/learning_queue.md")
    SKILLS_DIR = Path("/app/workspace/skills")

    # Read current queue + existing skills
    existing_queue = ""
    if QUEUE_FILE.exists():
        existing_queue = QUEUE_FILE.read_text().strip()

    existing_skills = []
    if SKILLS_DIR.exists():
        existing_skills = [f.stem for f in SKILLS_DIR.glob("*.md")
                          if f.name != "learning_queue.md"]

    # Gap Detector — primary signal source (Phase 2 of overhaul).
    # Replaces the prior scope_ecology + reflections queries which produced
    # a topical monoculture (33 rapid_ecological skills generated in a loop).
    # Now pulls structured, multi-source evidence: retrieval misses,
    # reflexion failures, MAP-Elites voids, and (later) user corrections.
    recent_context = ""
    try:
        # Refresh MAP-Elites void emissions before reading (cheap; idempotent).
        from app.self_improvement.gap_detector import (
            emit_mapelites_voids, get_recent_evidence_block,
        )
        emit_mapelites_voids()
        recent_context = get_recent_evidence_block(limit=12)
    except Exception:
        logger.debug("Gap Detector unavailable; falling back to legacy reflections",
                     exc_info=True)
        # Fallback to legacy signal so the system still functions if the new
        # store is uninitialized (first deploy, ChromaDB warm-up).
        try:
            from app.memory.scoped_memory import retrieve_operational
            reflections = retrieve_operational(
                "scope_reflections", "failure lesson learned", 5,
            )
            if reflections:
                recent_context = "Recent agent reflections:\n" + "\n".join(
                    f"- {r[:150]}" for r in reflections[:5]
                )
        except Exception:
            pass

    if not recent_context:
        logger.debug("idle_scheduler: no context for topic discovery, skipping")
        return

    try:
        from app.llm_factory import create_specialist_llm
        llm = create_specialist_llm(max_tokens=512, role="self_improve")
        prompt = (
            f"You are an AI team improvement specialist. The system has detected "
            f"the following learning gaps from observed activity. Each gap is "
            f"evidence that some knowledge is missing or weak. Your job: propose "
            f"1-3 NEW learning topics that would close the highest-strength gaps.\n\n"
            f"{recent_context}\n\n"
            f"Existing skills: {', '.join(existing_skills[:30]) or 'None'}\n"
            f"Already in queue: {existing_queue[:300] or 'Empty'}\n\n"
            f"Rules:\n"
            f"- Each topic must directly address one of the listed gaps.\n"
            f"- Use the gap's evidence to phrase the topic concretely.\n"
            f"- Diversify across gap sources — do not propose 3 topics from the same source.\n"
            f"- Do NOT suggest topics already in skills or queue.\n"
            f"- Reply with ONLY a newline-separated list of topic names (1-6 words each)."
        )
        raw = str(llm.call(prompt)).strip()
        if not raw or len(raw) < 3:
            return

        # Parse topics and append to queue
        new_topics = [
            line.strip().lstrip("- 0123456789.)")
            for line in raw.splitlines()
            if line.strip() and len(line.strip()) > 3
        ][:3]

        if not new_topics:
            return

        # ── Novelty Gate (Phase 1 of overhaul) ──────────────────────────
        # Replaces the prior bag-of-words `_is_duplicate` check, which
        # operated on a concatenated string of all existing skill names
        # and matched any 2+ shared words. With 223 skills containing
        # words like "ecological", "data", "strategies", that check
        # vacuously passed nearly every new topic.
        #
        # The new gate embeds each candidate and queries the unified KB
        # graph (4 KBs + team_shared). Decisions:
        #   - COVERED:  reject — already in the KBs
        #   - OVERLAP:  reject from queue (Phase 3 will route as extension)
        #   - ADJACENT: accept with cross-link
        #   - NOVEL:    accept
        #
        # Always-on cheap fallback to exact-name match in case the
        # retrieval orchestrator is unavailable (Chroma cold-start, etc.)
        existing_lower = {s.lower() for s in existing_skills}
        queue_lower = existing_queue.lower()

        def _is_duplicate_fallback(topic: str) -> bool:
            t = topic.lower().strip()
            if t.replace(" ", "_") in existing_lower:
                return True
            if t in queue_lower:
                return True
            return False

        unique_topics: list[str] = []
        from app.self_improvement.novelty import novelty_report
        from app.self_improvement.types import NoveltyDecision
        for topic in new_topics:
            # Cheap fallback first — exact filename collisions are obvious.
            if _is_duplicate_fallback(topic):
                logger.debug(f"novelty: '{topic}' rejected by exact-name fallback")
                continue
            try:
                rep = novelty_report(topic)
                if rep.decision in (NoveltyDecision.COVERED, NoveltyDecision.OVERLAP):
                    logger.info(
                        f"novelty: '{topic}' rejected as {rep.decision.value} "
                        f"(d={rep.nearest_distance:.3f}, near={rep.nearest_kb})"
                    )
                    continue
                unique_topics.append(topic)
                logger.debug(
                    f"novelty: '{topic}' accepted as {rep.decision.value} "
                    f"(d={rep.nearest_distance:.3f})"
                )
            except Exception:
                # Embedding/orchestrator failure: be permissive (better to
                # over-create than to silently dedup against a broken store)
                logger.debug(f"novelty check failed for '{topic}', accepting",
                             exc_info=True)
                unique_topics.append(topic)

        if not unique_topics:
            logger.debug("idle_scheduler: all discovered topics already covered")
            return

        QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(QUEUE_FILE, "a") as f:
            for topic in unique_topics:
                f.write(f"\n{topic}")

        logger.info(f"idle_scheduler: discovered {len(unique_topics)} new learning topics: {unique_topics}")
    except Exception:
        logger.debug("idle_scheduler: topic discovery failed", exc_info=True)


# ── Firestore kill switch listener ─────────────────────────────────────────

_bg_listener_unsub = None
_bg_poll_stop = threading.Event()


def _index_skills() -> None:
    """Index all workspace/skills/*.md files into ChromaDB 'skills' collection.

    Each skill file is embedded as a single document (or chunked if large).
    Uses the filename stem as the document ID for deduplication — re-running
    only updates changed files (ChromaDB upsert).

    This makes skills semantically searchable via _load_relevant_skills().
    """
    import hashlib
    from pathlib import Path

    SKILLS_DIR = Path("/app/workspace/skills")
    if not SKILLS_DIR.exists():
        return

    try:
        from app.memory.chromadb_manager import get_client
        client = get_client()
        collection = client.get_or_create_collection("skills")
    except Exception as e:
        logger.debug(f"skill_index: ChromaDB unavailable: {e}")
        return

    skill_files = sorted(SKILLS_DIR.glob("*.md"))
    if not skill_files:
        return

    # Get existing IDs to skip unchanged files
    existing = set()
    try:
        existing_data = collection.get(include=[])
        existing = set(existing_data.get("ids", []))
    except Exception:
        pass

    ids = []
    documents = []
    metadatas = []

    for f in skill_files:
        if f.name == "learning_queue.md":
            continue

        try:
            content = f.read_text(errors="replace").strip()
            if not content or len(content) < 20:
                continue

            # ID = filename stem (deterministic, deduplicates on re-run)
            doc_id = f"skill_{f.stem}"
            content_hash = hashlib.md5(content.encode()).hexdigest()[:8]

            # Check if already indexed with same content
            if doc_id in existing:
                # Could check hash for changes, but upsert handles it
                pass

            # Truncate very long skills to 2000 chars for embedding
            doc_text = content[:2000] if len(content) > 2000 else content

            ids.append(doc_id)
            documents.append(doc_text)
            metadatas.append({
                "filename": f.name,
                "content_hash": content_hash,
                "char_count": len(content),
                "type": "skill",
            })
        except Exception:
            continue

    if not ids:
        return

    # Batch upsert (ChromaDB handles duplicates)
    BATCH_SIZE = 50
    total = 0
    for i in range(0, len(ids), BATCH_SIZE):
        batch_ids = ids[i:i + BATCH_SIZE]
        batch_docs = documents[i:i + BATCH_SIZE]
        batch_meta = metadatas[i:i + BATCH_SIZE]
        try:
            from app.memory.chromadb_manager import embed
            collection.upsert(
                ids=batch_ids, documents=batch_docs, metadatas=batch_meta,
                embeddings=[embed(d) for d in batch_docs],
            )
            total += len(batch_ids)
        except Exception as e:
            logger.debug(f"skill_index: batch upsert failed: {e}")

    if total > 0:
        logger.info(f"skill_index: indexed {total} skills into ChromaDB 'skills' collection")


def read_background_enabled() -> bool | None:
    """Read background tasks enabled state from Firestore."""
    try:
        from app.firebase_reporter import _get_db
        db = _get_db()
        if not db:
            return None
        doc = db.collection("config").document("background_tasks").get()
        if doc.exists:
            return doc.to_dict().get("enabled", True)
    except Exception:
        pass
    return None


def start_background_listener() -> None:
    """Listen for kill switch changes from dashboard via Firestore."""
    global _bg_listener_unsub

    def _listen() -> None:
        global _bg_listener_unsub
        from app.firebase_reporter import _get_db, _fire
        db = _get_db()
        if not db:
            return
        try:
            def on_snapshot(doc_snapshot, changes, read_time):
                for snap in doc_snapshot:
                    data = snap.to_dict()
                    enabled = data.get("enabled", True)
                    set_enabled(bool(enabled))

            _bg_listener_unsub = (
                db.collection("config").document("background_tasks")
                .on_snapshot(on_snapshot)
            )
            logger.info("idle_scheduler: background_tasks listener started")
        except Exception:
            logger.debug("idle_scheduler: background_tasks listener failed", exc_info=True)

    try:
        from app.firebase_reporter import _fire
        _fire(_listen)
    except Exception:
        pass

    # Polling fallback (same pattern as mode listener)
    def _poll() -> None:
        while not _bg_poll_stop.wait(15):
            try:
                val = read_background_enabled()
                if val is not None:
                    set_enabled(val)
            except Exception:
                pass

    t = threading.Thread(target=_poll, daemon=True, name="bg-tasks-poll")
    t.start()


def stop_listener() -> None:
    _bg_poll_stop.set()
