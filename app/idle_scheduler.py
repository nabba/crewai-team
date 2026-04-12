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
from typing import Callable

logger = logging.getLogger(__name__)

# How long to wait after last user task before starting background work
IDLE_DELAY_SECONDS = 30

# Pause between background job iterations (brief cooldown, then next job)
INTER_JOB_PAUSE_SECONDS = 2  # Reduced from 5 — lightweight jobs don't need long pauses


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

# Global state
_last_task_end: float = 0.0  # monotonic timestamp of last user task completion
_active_tasks: int = 0
_lock = threading.Lock()
_stop_event = threading.Event()
_enabled = True  # kill switch — toggled from Firestore
_enabled_lock = threading.Lock()
_idle_thread: threading.Thread | None = None


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


# Load on module init
_load_job_state()


def _run_single_job(name: str, fn: Callable, timeout_s: int = 60) -> bool:
    """Run a single job with time cap, retry on failure, and skip-after-3-failures.

    Returns True if job succeeded, False if failed.
    Jobs that fail 3 consecutive times are skipped for 1 hour.
    """
    # Check if job is in skip cooldown (wall clock — survives restarts)
    skip_until = _job_skip_until.get(name, 0)
    if skip_until and time.time() < skip_until:
        return False

    _job_timeout.clear()
    timer = threading.Timer(timeout_s, _job_timeout.set)
    timer.daemon = True
    timer.start()
    try:
        _report_background_activity(name, "running")
        fn()
        logger.info(f"idle_scheduler: '{name}' completed")
        _report_background_activity(name, "completed")
        _job_failure_counts[name] = 0  # Reset on success
        _persist_job_failure(name, 0)
        return True
    except Exception as exc:
        _job_failure_counts[name] = _job_failure_counts.get(name, 0) + 1
        consec = _job_failure_counts[name]
        _persist_job_failure(name, consec)
        logger.warning(f"idle_scheduler: '{name}' failed ({consec} consecutive): {exc}")
        _report_background_activity(name, "failed")

        # After 3 consecutive failures, skip job for 1 hour
        if consec >= 3:
            skip_ts = time.time() + 3600
            _job_skip_until[name] = skip_ts
            _persist_job_skip(name, skip_ts)
            logger.warning(f"idle_scheduler: '{name}' skipped for 1h after {consec} consecutive failures")

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
        _training_interval = 3600
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
            # Wait for all lightweight jobs (bounded by their time caps)
            from concurrent.futures import as_completed
            for future in as_completed(futures, timeout=TIME_CAPS[JobWeight.LIGHT] + 10):
                try:
                    future.result()
                except Exception:
                    pass

        if _stop_event.is_set() or not is_idle():
            continue

        # ── Phase 2: Run ONE medium job ──────────────────────────────────
        if medium_jobs:
            name, fn = medium_jobs[medium_idx % len(medium_jobs)]
            medium_idx += 1
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
        def _write():
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

    if jobs is None:
        jobs = _default_jobs()

    if not jobs:
        logger.warning("idle_scheduler: no jobs configured, not starting")
        return

    _stop_event.clear()
    _idle_thread = threading.Thread(
        target=_run_idle_loop, args=(jobs,),
        daemon=True, name="idle-scheduler",
    )
    _idle_thread.start()


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
    def _learn_queue():
        from app.crews.self_improvement_crew import SelfImprovementCrew
        SelfImprovementCrew().run()
    jobs.append(("learn-queue", _learn_queue, JobWeight.HEAVY))

    # ── Evolution: run experiments (2 iterations per idle slot) ─────────
    # Reduced from 5 to 2: each iteration takes ~4min, so 5 = 20min which
    # starves all subsequent jobs. 2 iterations keeps total under 10min.
    def _evolution():
        from app.evolution import run_evolution_session
        run_evolution_session(max_iterations=2)
    jobs.append(("evolution", _evolution, JobWeight.HEAVY))

    # ── Proactive learning: discover and queue new topics ──────────────
    def _discover_topics():
        _auto_discover_topics()
    jobs.append(("discover-topics", _discover_topics, JobWeight.LIGHT))

    # ── Retrospective: analyze recent performance ──────────────────────
    def _retrospective():
        from app.crews.retrospective_crew import RetrospectiveCrew
        RetrospectiveCrew().run()
    jobs.append(("retrospective", _retrospective, JobWeight.HEAVY))

    # ── Embedded personality probes: covert measurement via real-ish tasks ──
    def _embedded_probe():
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
    def _improvement_scan():
        from app.crews.self_improvement_crew import SelfImprovementCrew
        SelfImprovementCrew().run_improvement_scan()
    jobs.append(("improvement-scan", _improvement_scan, JobWeight.MEDIUM))

    # ── Feedback aggregation: detect patterns in user feedback ──────────
    # Shared singleton for feedback pipeline (avoids creating new DB engine per job)
    _feedback_pipeline_cache = [None]
    def _get_feedback_pipeline():
        if _feedback_pipeline_cache[0] is None:
            from app.config import get_settings
            s = get_settings()
            if not s.mem0_postgres_url:
                return None
            from app.feedback_pipeline import FeedbackPipeline
            _feedback_pipeline_cache[0] = FeedbackPipeline(s.mem0_postgres_url)
        return _feedback_pipeline_cache[0]

    def _feedback_aggregate():
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

    # ── Safety health check: monitor for post-promotion regressions ────
    def _safety_health_check():
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
    def _modification_engine():
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
    def _health_evaluate():
        try:
            from app.health_monitor import evaluate_health
            alerts = evaluate_health()
            if alerts:
                logger.info(f"idle_scheduler: health monitor found {len(alerts)} alert(s)")
        except Exception:
            logger.debug("idle_scheduler: health evaluation failed", exc_info=True)
    jobs.append(("health-evaluate", _health_evaluate, JobWeight.LIGHT))

    # ── Version manifest: periodic snapshot for rollback safety ────
    def _version_snapshot():
        try:
            from app.version_manifest import create_manifest, cleanup_old_snapshots
            create_manifest(promoted_by="system", reason="periodic snapshot")
            cleanup_old_snapshots(keep_latest=10)
        except Exception:
            logger.debug("idle_scheduler: version snapshot failed", exc_info=True)
    jobs.append(("version-snapshot", _version_snapshot, JobWeight.LIGHT))

    # ── PDS: personality development assessment session ──────────────────
    def _personality_session():
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
    def _cogito_cycle():
        try:
            from app.self_awareness.cogito import run_cogito
            report = run_cogito()
            logger.info(f"idle_scheduler: cogito cycle — health={report.overall_health}")
        except Exception:
            logger.debug("idle_scheduler: cogito cycle failed", exc_info=True)
    jobs.append(("cogito-cycle", _cogito_cycle, JobWeight.MEDIUM))

    # ── Self-knowledge: re-ingest codebase for self-inspection ────────
    def _self_knowledge_ingest():
        try:
            from app.self_awareness.knowledge_ingestion import ingest_codebase
            result = ingest_codebase(full=False)
            if result.get("chunks_added", 0) > 0:
                logger.info(f"idle_scheduler: self-knowledge ingested {result['chunks_added']} chunks")
        except Exception:
            logger.debug("idle_scheduler: self-knowledge ingest failed", exc_info=True)
    jobs.append(("self-knowledge-ingest", _self_knowledge_ingest, JobWeight.MEDIUM))

    # ── Skill indexer: embed skills/*.md into ChromaDB for semantic retrieval ──
    def _skill_index():
        try:
            _index_skills()
        except Exception:
            logger.debug("idle_scheduler: skill indexing failed", exc_info=True)
    jobs.append(("skill-index", _skill_index, JobWeight.LIGHT))

    # ── Self-training: curate collected data + trigger training ─────────
    def _training_curate():
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
    def _training_pipeline():
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
    def _fiction_ingest():
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
    def _consciousness_probe():
        try:
            from app.self_awareness.consciousness_probe import run_consciousness_probes
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
    def _behavioral_assessment():
        try:
            from app.self_awareness.behavioral_assessment import run_behavioral_assessment
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
    def _prosocial_learning():
        try:
            from app.self_awareness.prosocial_learning import run_prosocial_session
            profiles = run_prosocial_session()
            logger.info(f"idle_scheduler: prosocial session complete, {len(profiles)} profiles updated")
        except Exception:
            logger.debug("idle_scheduler: prosocial learning failed", exc_info=True)
    jobs.append(("prosocial-learning", _prosocial_learning, JobWeight.MEDIUM))

    # ── MAP-Elites: quality-diversity maintenance + migration ──────────
    def _map_elites_maintain():
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
    def _island_evolution():
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
    def _parallel_evolution():
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
    def _atlas_competence_sync():
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
    def _atlas_stale_check():
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
    def _atlas_learning():
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
    def _llm_discovery():
        try:
            from app.llm_discovery import run_discovery_cycle
            result = run_discovery_cycle(max_benchmarks=2)
            if result.get("new_found", 0) > 0 or result.get("promoted", 0) > 0:
                logger.info(f"idle_scheduler: LLM discovery: {result}")
        except Exception:
            logger.debug("idle_scheduler: LLM discovery failed", exc_info=True)
    jobs.append(("llm-discovery", _llm_discovery, JobWeight.MEDIUM))

    # ── System monitor: report all subsystem status to dashboard ────
    def _system_monitor():
        try:
            from app.firebase_reporter import report_system_monitor
            report_system_monitor()
        except Exception:
            logger.debug("idle_scheduler: system monitor report failed", exc_info=True)
    jobs.append(("system-monitor", _system_monitor, JobWeight.LIGHT))

    # ── Tech radar: scan internet for new technologies ────────────────
    def _tech_radar():
        from app.crews.tech_radar_crew import run_tech_scan
        run_tech_scan()
    jobs.append(("tech-radar", _tech_radar, JobWeight.HEAVY))

    # ── Heartbeat: per-agent autonomous wake cycle ────────────────────
    def _heartbeat_cycle():
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
        except Exception:
            logger.debug("idle_scheduler: heartbeat cycle failed", exc_info=True)
    jobs.append(("heartbeat-cycle", _heartbeat_cycle, JobWeight.LIGHT))

    # ── Emergent infrastructure: review pending tool proposals ────────
    def _emergent_infrastructure():
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
    def _entropy_monitoring():
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
    def _data_retention():
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

    # ── Ollama memory management: unload idle models to free VRAM ─────
    def _ollama_memory():
        try:
            from app.ollama_native import unload_idle_models
            unloaded = unload_idle_models(idle_minutes=30)
            if unloaded:
                logger.info(f"idle_scheduler: freed VRAM — unloaded {len(unloaded)} idle models: {unloaded}")
        except Exception:
            logger.debug("idle_scheduler: ollama memory management failed", exc_info=True)
    jobs.append(("ollama-memory", _ollama_memory, JobWeight.LIGHT))

    # ── Chaos testing: verify self-healing paths (max once per 24h) ───
    def _chaos_testing():
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
    def _consciousness_slow_loop():
        try:
            from app.config import get_settings
            if not get_settings().consciousness_enabled:
                return
            from app.consciousness.metacognitive_monitor import get_monitor
            monitor = get_monitor()
            result = monitor.run_slow_loop()
            if any(v for v in result.values() if v):
                logger.info(f"idle_scheduler: consciousness slow loop: {result}")
        except Exception:
            logger.debug("idle_scheduler: consciousness slow loop failed", exc_info=True)
    jobs.append(("consciousness-slow-loop", _consciousness_slow_loop, JobWeight.MEDIUM))

    # ── AST-1 slow loop: attention pattern evaluation ─────────────────
    def _attention_slow_loop():
        try:
            from app.config import get_settings
            if not get_settings().consciousness_enabled:
                return
            from app.consciousness.attention_schema import get_attention_schema
            result = get_attention_schema().run_slow_loop()
            if result.get("is_stuck") or result.get("is_captured"):
                logger.warning(f"idle_scheduler: AST-1 attention alert: {result}")
        except Exception:
            logger.debug("idle_scheduler: attention slow loop failed", exc_info=True)
    jobs.append(("attention-slow-loop", _attention_slow_loop, JobWeight.MEDIUM))

    # ── PP-1 slow loop: prediction model recalibration ────────────────
    def _prediction_slow_loop():
        try:
            from app.config import get_settings
            if not get_settings().consciousness_enabled:
                return
            from app.consciousness.predictive_layer import get_predictive_layer
            result = get_predictive_layer().run_slow_loop()
            if result.get("recent_major_surprises", 0) > 3:
                logger.warning(f"idle_scheduler: PP-1 systematic surprises: {result}")
        except Exception:
            logger.debug("idle_scheduler: prediction slow loop failed", exc_info=True)
    jobs.append(("prediction-slow-loop", _prediction_slow_loop, JobWeight.MEDIUM))

    # ── Dead letter queue: retry failed messages ─────────────────────
    def _dead_letter_retry():
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
    def _adversarial_probes():
        try:
            from app.consciousness.adversarial_probes import run_adversarial_probes
            results = run_adversarial_probes()
            if results:
                passed = sum(1 for r in results if r.passed)
                logger.info(f"idle_scheduler: adversarial probes {passed}/{len(results)} passed")
        except Exception:
            logger.debug("idle_scheduler: adversarial probes failed", exc_info=True)
    jobs.append(("adversarial-probes", _adversarial_probes, JobWeight.HEAVY))

    # ── Meta-workspace promotion: aggregate top items from all projects ──
    def _meta_workspace_promotion():
        try:
            from app.consciousness.meta_workspace import get_meta_workspace
            results = get_meta_workspace().promote_all()
            promoted = sum(1 for v in results.values() if v)
            if promoted:
                logger.info(f"idle_scheduler: meta-workspace promoted {promoted}/{len(results)} items")
        except Exception:
            logger.debug("idle_scheduler: meta-workspace promotion failed", exc_info=True)
    jobs.append(("meta-workspace-promotion", _meta_workspace_promotion, JobWeight.LIGHT))

    # ── Wiki lint: periodic health check of knowledge wiki ────────────
    def _wiki_lint():
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

    return jobs


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

    # Gather recent failure/reflection data from memory
    recent_context = ""
    try:
        from app.memory.scoped_memory import retrieve_operational
        reflections = retrieve_operational("scope_reflections", "failure lesson learned", 5)
        if reflections:
            recent_context = "Recent agent reflections:\n" + "\n".join(
                f"- {r[:150]}" for r in reflections[:5]
            )
    except Exception:
        pass

    # Gather recent user questions that were hard (difficulty >= 6)
    try:
        from app.memory.scoped_memory import retrieve_operational
        hard_tasks = retrieve_operational("scope_ecology", "difficulty duration", 5)
        if hard_tasks:
            recent_context += "\n\nRecent hard tasks:\n" + "\n".join(
                f"- {t[:150]}" for t in hard_tasks[:5]
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
            f"You are an AI team improvement specialist. Based on recent agent activity, "
            f"suggest 1-3 NEW topics the team should learn to handle future requests better.\n\n"
            f"{recent_context}\n\n"
            f"Existing skills: {', '.join(existing_skills[:30]) or 'None'}\n"
            f"Already in queue: {existing_queue[:300] or 'Empty'}\n\n"
            f"Reply with ONLY a newline-separated list of topic names (1-6 words each). "
            f"Do NOT suggest topics already in skills or queue. "
            f"Focus on practical, actionable topics that would help with real user requests."
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

        # Filter out duplicates (exact + fuzzy substring matching)
        existing_lower = {s.lower() for s in existing_skills}
        queue_lower = existing_queue.lower()
        all_known = " ".join(existing_lower) + " " + queue_lower

        def _is_duplicate(topic: str) -> bool:
            t = topic.lower().strip()
            # Exact match
            if t.replace(" ", "_") in existing_lower:
                return True
            if t in queue_lower:
                return True
            # Fuzzy: if 2+ words from the topic appear in existing skills, likely duplicate
            words = [w for w in t.split() if len(w) > 3]
            if words and sum(1 for w in words if w in all_known) >= min(2, len(words)):
                return True
            return False

        unique_topics = [t for t in new_topics if not _is_duplicate(t)]

        if not unique_topics:
            logger.debug("idle_scheduler: all discovered topics already known")
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
            collection.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_meta)
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

    def _listen():
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
    def _poll():
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
