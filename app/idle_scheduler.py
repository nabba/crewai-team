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
INTER_JOB_PAUSE_SECONDS = 5

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


def should_yield() -> bool:
    """Check if a background job should abort because a user task arrived.

    Long-running background functions (evolution iterations, self-improvement)
    should call this between units of work and return early if True.
    """
    with _lock:
        return _active_tasks > 0


def set_enabled(enabled: bool) -> None:
    """Kill switch — disable/enable background work."""
    global _enabled
    with _enabled_lock:
        _enabled = enabled
    logger.info(f"idle_scheduler: background tasks {'enabled' if enabled else 'disabled'}")


def is_enabled() -> bool:
    with _enabled_lock:
        return _enabled


def _run_idle_loop(jobs: list[tuple[str, Callable[[], None]]]) -> None:
    """Main idle loop — cycles through background jobs when system is idle.

    Each job is a (name, callable) tuple. The callable should do one unit of work
    and return. The loop cycles through jobs round-robin, pausing between each.

    Background threads are marked as low-priority for rate limiting — user-facing
    LLM calls always get priority over background work.
    """
    # Mark this thread as a background caller for rate_throttle priority
    from app.rate_throttle import set_background_caller
    set_background_caller(True)

    logger.info(f"idle_scheduler: started with {len(jobs)} jobs")
    job_idx = 0

    while not _stop_event.is_set():
        # Wait until idle
        while not _stop_event.is_set():
            if is_enabled() and is_idle():
                break
            _stop_event.wait(5)  # check every 5 seconds

        if _stop_event.is_set():
            break

        # Pick next job
        name, fn = jobs[job_idx % len(jobs)]
        job_idx += 1

        # Double-check still idle + enabled before running
        if not is_enabled() or not is_idle():
            continue

        logger.info(f"idle_scheduler: running '{name}' (system idle)")
        _report_background_activity(name, "running")
        try:
            fn()
            logger.info(f"idle_scheduler: '{name}' completed")
            _report_background_activity(name, "completed")
        except Exception:
            logger.warning(f"idle_scheduler: '{name}' failed", exc_info=True)
            _report_background_activity(name, "failed")

        # Pause between jobs — also check for user activity
        for _ in range(INTER_JOB_PAUSE_SECONDS):
            if _stop_event.is_set() or not is_idle():
                break
            time.sleep(1)

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
    jobs.append(("learn-queue", _learn_queue))

    # ── Evolution: run experiments (5 iterations per idle slot) ─────────
    def _evolution():
        from app.evolution import run_evolution_session
        run_evolution_session(max_iterations=5)
    jobs.append(("evolution", _evolution))

    # ── Proactive learning: discover and queue new topics ──────────────
    def _discover_topics():
        _auto_discover_topics()
    jobs.append(("discover-topics", _discover_topics))

    # ── Retrospective: analyze recent performance ──────────────────────
    def _retrospective():
        from app.crews.retrospective_crew import RetrospectiveCrew
        RetrospectiveCrew().run()
    jobs.append(("retrospective", _retrospective))

    # ── Evolution again (doubled frequency) ────────────────────────────
    def _evolution2():
        from app.evolution import run_evolution_session
        run_evolution_session(max_iterations=5)
    jobs.append(("evolution-2", _evolution2))

    # ── Improvement scan: analyze gaps and propose improvements ────────
    def _improvement_scan():
        from app.crews.self_improvement_crew import SelfImprovementCrew
        SelfImprovementCrew().run_improvement_scan()
    jobs.append(("improvement-scan", _improvement_scan))

    # ── Feedback aggregation: detect patterns in user feedback ──────────
    def _feedback_aggregate():
        try:
            from app.config import get_settings
            s = get_settings()
            if not s.mem0_postgres_url:
                return
            from app.feedback_pipeline import FeedbackPipeline
            pipeline = FeedbackPipeline(s.mem0_postgres_url)
            patterns = pipeline.aggregate_patterns()
            if patterns:
                logger.info(f"idle_scheduler: feedback aggregation found {len(patterns)} patterns")
        except Exception:
            logger.debug("idle_scheduler: feedback aggregation failed", exc_info=True)
    jobs.append(("feedback-aggregate", _feedback_aggregate))

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
    jobs.append(("safety-health-check", _safety_health_check))

    # ── Modification engine: propose prompt changes from feedback ─────
    def _modification_engine():
        try:
            from app.config import get_settings
            s = get_settings()
            if not s.mem0_postgres_url:
                return
            from app.feedback_pipeline import FeedbackPipeline
            from app.modification_engine import ModificationEngine
            import app.prompt_registry as registry
            pipeline = FeedbackPipeline(s.mem0_postgres_url)
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
    jobs.append(("modification-engine", _modification_engine))

    # ── Health monitor: evaluate dimensional health ─────────────────
    def _health_evaluate():
        try:
            from app.health_monitor import evaluate_health
            alerts = evaluate_health()
            if alerts:
                logger.info(f"idle_scheduler: health monitor found {len(alerts)} alert(s)")
        except Exception:
            logger.debug("idle_scheduler: health evaluation failed", exc_info=True)
    jobs.append(("health-evaluate", _health_evaluate))

    # ── Version manifest: periodic snapshot for rollback safety ────
    def _version_snapshot():
        try:
            from app.version_manifest import create_manifest, cleanup_old_snapshots
            create_manifest(promoted_by="system", reason="periodic snapshot")
            cleanup_old_snapshots(keep_latest=10)
        except Exception:
            logger.debug("idle_scheduler: version snapshot failed", exc_info=True)
    jobs.append(("version-snapshot", _version_snapshot))

    # ── Cogito: metacognitive self-reflection cycle ─────────────────────
    def _cogito_cycle():
        try:
            from app.self_awareness.cogito import run_cogito
            report = run_cogito()
            logger.info(f"idle_scheduler: cogito cycle — health={report.overall_health}")
        except Exception:
            logger.debug("idle_scheduler: cogito cycle failed", exc_info=True)
    jobs.append(("cogito-cycle", _cogito_cycle))

    # ── Self-knowledge: re-ingest codebase for self-inspection ────────
    def _self_knowledge_ingest():
        try:
            from app.self_awareness.knowledge_ingestion import ingest_codebase
            result = ingest_codebase(full=False)
            if result.get("chunks_added", 0) > 0:
                logger.info(f"idle_scheduler: self-knowledge ingested {result['chunks_added']} chunks")
        except Exception:
            logger.debug("idle_scheduler: self-knowledge ingest failed", exc_info=True)
    jobs.append(("self-knowledge-ingest", _self_knowledge_ingest))

    # ── Self-training: curate collected data + trigger training ─────────
    def _training_curate():
        try:
            from app.training_collector import get_pipeline
            pipeline = get_pipeline()
            stats = pipeline.get_stats()
            if stats.get("total_interactions", 0) > 0:
                result = pipeline.run_curation()
                if result.get("exported_train", 0) > 0:
                    logger.info(f"idle_scheduler: training data curated — "
                                f"{result.get('exported_train', 0)} examples exported")
        except Exception:
            logger.debug("idle_scheduler: training curation failed", exc_info=True)
    jobs.append(("training-curate", _training_curate))

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
    jobs.append(("fiction-ingest", _fiction_ingest))

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
    jobs.append(("map-elites-maintain", _map_elites_maintain))

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
    jobs.append(("island-evolution", _island_evolution))

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
    jobs.append(("parallel-evolution", _parallel_evolution))

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
    jobs.append(("atlas-competence-sync", _atlas_competence_sync))

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
    jobs.append(("atlas-stale-check", _atlas_stale_check))

    # ── Tech radar: scan internet for new technologies ────────────────
    def _tech_radar():
        from app.crews.tech_radar_crew import run_tech_scan
        run_tech_scan()
    jobs.append(("tech-radar", _tech_radar))

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

        # Filter out duplicates
        existing_lower = {s.lower() for s in existing_skills}
        queue_lower = existing_queue.lower()
        unique_topics = [
            t for t in new_topics
            if t.lower().replace(" ", "_") not in existing_lower
            and t.lower() not in queue_lower
        ]

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
