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
        except Exception as exc:
            logger.warning(f"idle_scheduler: '{name}' failed", exc_info=True)
            _report_background_activity(name, "failed")
            # Detect credit exhaustion from any background job failure
            try:
                from app.firebase_reporter import detect_credit_error, report_credit_alert
                provider = detect_credit_error(exc)
                if provider:
                    report_credit_alert(provider, str(exc)[:300])
            except Exception:
                pass

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

    # ── Evolution: run experiments (2 iterations per idle slot) ─────────
    # Reduced from 5 to 2: each iteration takes ~4min, so 5 = 20min which
    # starves all subsequent jobs. 2 iterations keeps total under 10min.
    def _evolution():
        from app.evolution import run_evolution_session
        run_evolution_session(max_iterations=2)
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
    jobs.append(("embedded-probe", _embedded_probe))

    # ── Improvement scan: analyze gaps and propose improvements ────────
    def _improvement_scan():
        from app.crews.self_improvement_crew import SelfImprovementCrew
        SelfImprovementCrew().run_improvement_scan()
    jobs.append(("improvement-scan", _improvement_scan))

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

    # ── Skill indexer: embed skills/*.md into ChromaDB for semantic retrieval ──
    def _skill_index():
        try:
            _index_skills()
        except Exception:
            logger.debug("idle_scheduler: skill indexing failed", exc_info=True)
    jobs.append(("skill-index", _skill_index))

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
    jobs.append(("training-curate", _training_curate))

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
    jobs.append(("training-pipeline", _training_pipeline))

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
    jobs.append(("consciousness-probe", _consciousness_probe))

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
    jobs.append(("behavioral-assessment", _behavioral_assessment))

    # ── Prosocial preference learning: coordination games ────────────
    def _prosocial_learning():
        try:
            from app.self_awareness.prosocial_learning import run_prosocial_session
            profiles = run_prosocial_session()
            logger.info(f"idle_scheduler: prosocial session complete, {len(profiles)} profiles updated")
        except Exception:
            logger.debug("idle_scheduler: prosocial learning failed", exc_info=True)
    jobs.append(("prosocial-learning", _prosocial_learning))

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
    jobs.append(("atlas-learning", _atlas_learning))

    # ── LLM Discovery: scan for new models, benchmark, promote ────────
    def _llm_discovery():
        try:
            from app.llm_discovery import run_discovery_cycle
            result = run_discovery_cycle(max_benchmarks=2)
            if result.get("new_found", 0) > 0 or result.get("promoted", 0) > 0:
                logger.info(f"idle_scheduler: LLM discovery: {result}")
        except Exception:
            logger.debug("idle_scheduler: LLM discovery failed", exc_info=True)
    jobs.append(("llm-discovery", _llm_discovery))

    # ── System monitor: report all subsystem status to dashboard ────
    def _system_monitor():
        try:
            from app.firebase_reporter import report_system_monitor
            report_system_monitor()
        except Exception:
            logger.debug("idle_scheduler: system monitor report failed", exc_info=True)
    jobs.append(("system-monitor", _system_monitor))

    # ── Tech radar: scan internet for new technologies ────────────────
    def _tech_radar():
        from app.crews.tech_radar_crew import run_tech_scan
        run_tech_scan()
    jobs.append(("tech-radar", _tech_radar))

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
    jobs.append(("heartbeat-cycle", _heartbeat_cycle))

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
