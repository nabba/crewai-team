import hmac
import logging
import logging.handlers
import asyncio
import os
import re
import threading
from datetime import datetime, timezone, timedelta

# Ensure all loggers output to stdout so docker logs captures tracebacks
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# Install API rate throttle BEFORE any litellm/crewai imports to monkey-patch early
from app.rate_throttle import install_throttle
install_throttle()

from fastapi import FastAPI, Request, HTTPException
from contextlib import asynccontextmanager
from app.config import get_settings, get_gateway_secret
from app.security import is_authorized_sender, is_within_rate_limit, _redact_number
from app.signal_client import SignalClient
from app.agents.commander import Commander
from app.self_heal import diagnose_and_fix
from app.audit import (
    log_request_received, log_response_sent, log_security_event
)
from app.conversation_store import add_message, start_task, complete_task
from app.workspace_sync import setup_workspace_repo, sync_workspace
from app.firebase_reporter import (
    report_system_online, report_system_offline, heartbeat, report_schedule,
    cleanup_stale_tasks, report_llm_mode, start_mode_listener,
    start_kb_queue_poller, start_phil_queue_poller, start_fiction_queue_poller,
    report_chat_message, start_chat_inbox_poller,
)
from app import idle_scheduler
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

settings = get_settings()
logger = logging.getLogger(__name__)
scheduler = AsyncIOScheduler()

# Dedicated thread pool for commander.handle() calls — ensures multiple
# messages can be processed concurrently without saturating the default
# asyncio executor.  Each message gets its own thread.
from concurrent.futures import ThreadPoolExecutor
_commander_pool = ThreadPoolExecutor(
    max_workers=settings.max_parallel_crews + 2,
    thread_name_prefix="commander",
)

_WORKSPACE_ROOT = "/app/workspace"


def _configure_audit_log() -> None:
    """Set up a rotating file handler for the structured audit log."""
    audit_logger = logging.getLogger("crewai.audit")
    audit_logger.setLevel(logging.INFO)
    audit_logger.propagate = False  # Don't mix with app logs

    # Validate that AUDIT_LOG_PATH stays within the workspace — prevents an
    # attacker with env-var access from redirecting logs to /dev/null or elsewhere
    from pathlib import Path
    raw_path = os.environ.get("AUDIT_LOG_PATH", f"{_WORKSPACE_ROOT}/audit.log")
    resolved = Path(raw_path).resolve()
    try:
        resolved.relative_to(_WORKSPACE_ROOT)
    except ValueError:
        raise ValueError(f"AUDIT_LOG_PATH must be inside {_WORKSPACE_ROOT}, got: {raw_path}")
    log_path = str(resolved)

    handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=10 * 1024 * 1024,  # 10 MB per file
        backupCount=5,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    audit_logger.addHandler(handler)

    # Also emit to stdout so Docker log drivers can capture it
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter("AUDIT %(message)s"))
    audit_logger.addHandler(stdout_handler)

MAX_MESSAGE_LENGTH = 4000  # Prevent abuse / token bombing


# ── Extracted to app/response_utils.py ────────────────────────────────────────
from app.response_utils import write_response_md as _write_response_md_impl, extract_to_mem0 as _extract_to_mem0

def _write_response_md(full_text, user_question):
    return _write_response_md_impl(full_text, user_question, settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.crews.self_improvement_crew import SelfImprovementCrew

    # Fail fast if gateway is misconfigured to bind on a public interface
    if settings.gateway_bind != "127.0.0.1":
        raise RuntimeError(
            f"GATEWAY_BIND must be 127.0.0.1, got {settings.gateway_bind!r}. "
            "Refusing to start — binding to a public interface is unsafe."
        )

    _configure_audit_log()

    # Validate cron expressions early so misconfiguration fails fast
    try:
        trigger = CronTrigger.from_crontab(settings.self_improve_cron)
    except ValueError as exc:
        raise RuntimeError(
            f"Invalid SELF_IMPROVE_CRON expression {settings.self_improve_cron!r}: {exc}"
        ) from exc

    try:
        sync_trigger = CronTrigger.from_crontab(settings.workspace_sync_cron)
    except ValueError as exc:
        raise RuntimeError(
            f"Invalid WORKSPACE_SYNC_CRON expression {settings.workspace_sync_cron!r}: {exc}"
        ) from exc

    # Restore workspace state from cloud backup (non-fatal if remote is empty/absent)
    if settings.workspace_backup_repo:
        await asyncio.to_thread(setup_workspace_repo, settings.workspace_backup_repo)

    scheduler.add_job(SelfImprovementCrew().run, trigger, id="self_improve")

    # Native Ollama — verify availability (Metal GPU acceleration)
    from app.ollama_native import _is_running as ollama_is_running
    if await asyncio.to_thread(ollama_is_running):
        logger.info("Native Ollama detected — Metal GPU acceleration enabled")
    else:
        logger.warning(
            "Native Ollama not detected at %s. "
            "Local models will fall back to Claude API. "
            "Start Ollama with: ollama serve",
            settings.ollama_base_url,
        )

    # Auditor — continuous code quality + error resolution
    from app.auditor import run_code_audit, run_error_resolution
    auditor_cron = os.environ.get("AUDITOR_CRON", "0 */4 * * *")
    error_fix_cron = os.environ.get("ERROR_FIX_CRON", "*/30 * * * *")
    try:
        scheduler.add_job(run_code_audit, CronTrigger.from_crontab(auditor_cron), id="code_audit")
        logger.info(f"Code auditor scheduled: {auditor_cron}")
    except ValueError:
        logger.warning(f"Invalid AUDITOR_CRON: {auditor_cron}")
    try:
        scheduler.add_job(run_error_resolution, CronTrigger.from_crontab(error_fix_cron), id="error_resolution")
        logger.info(f"Error resolution loop scheduled: {error_fix_cron}")
    except ValueError:
        logger.warning(f"Invalid ERROR_FIX_CRON: {error_fix_cron}")

    # Evolution loop — autoresearch-style continuous improvement (every 6 hours)
    from app.evolution import run_evolution_session
    evolution_cron = os.environ.get("EVOLUTION_CRON", "0 */6 * * *")
    try:
        evo_trigger = CronTrigger.from_crontab(evolution_cron)
        scheduler.add_job(
            run_evolution_session,
            evo_trigger,
            id="evolution",
            kwargs={"max_iterations": settings.evolution_iterations},
        )
        logger.info(f"Evolution loop scheduled: {evolution_cron} ({settings.evolution_iterations} iterations/session)")
    except ValueError:
        logger.warning(f"Invalid EVOLUTION_CRON: {evolution_cron}, evolution loop disabled")

    # Retrospective crew — meta-cognitive self-improvement (daily at 4 AM by default)
    from app.crews.retrospective_crew import RetrospectiveCrew
    try:
        retro_trigger = CronTrigger.from_crontab(settings.retrospective_cron)
        scheduler.add_job(RetrospectiveCrew().run, retro_trigger, id="retrospective")
        logger.info(f"Retrospective crew scheduled: {settings.retrospective_cron}")
    except ValueError:
        logger.warning(f"Invalid RETROSPECTIVE_CRON: {settings.retrospective_cron}")

    # Benchmark snapshot — daily summary of performance metrics
    from app.benchmarks import get_benchmark_summary
    try:
        bench_trigger = CronTrigger.from_crontab(settings.benchmark_cron)
        scheduler.add_job(
            lambda: logger.info(f"Benchmark snapshot: {get_benchmark_summary()}"),
            bench_trigger, id="benchmark_snapshot",
        )
        logger.info(f"Benchmark snapshot scheduled: {settings.benchmark_cron}")
    except ValueError:
        logger.warning(f"Invalid BENCHMARK_CRON: {settings.benchmark_cron}")

    # Workspace sync: pass backup_repo as a kwarg so the job is a no-op when unset
    scheduler.add_job(
        sync_workspace,
        sync_trigger,
        kwargs={"backup_repo": settings.workspace_backup_repo},
    )
    # Heartbeat — keep monitoring dashboard "last_seen" fresh + anomaly detection + dashboard data
    _hb_counter = [0]
    def _heartbeat_with_anomaly():
        heartbeat()
        _hb_counter[0] += 1
        try:
            from app.anomaly_detector import collect_and_check, handle_alerts
            alerts = collect_and_check()
            if alerts:
                handle_alerts(alerts)
        except Exception:
            pass
        # Push self-healing/evolving data to dashboard every 5 minutes (every 5th heartbeat)
        if _hb_counter[0] % 5 == 0:
            try:
                from app.firebase_reporter import (
                    report_anomalies, report_variants, report_tech_radar,
                    report_deploys, report_proposal_actions, report_proposals,
                )
                report_anomalies()
                report_variants()
                report_tech_radar()
                report_deploys()
                report_proposals()  # push pending proposals to dashboard
                report_proposal_actions()  # process dashboard approve/reject clicks
                from app.firebase_reporter import report_philosophy_kb, report_evolution_stats
                report_philosophy_kb()
                report_evolution_stats()
            except Exception:
                pass
    scheduler.add_job(_heartbeat_with_anomaly, "interval", seconds=60, id="heartbeat")
    scheduler.start()

    # ── PARALLELIZED: cleanup + mode read are independent I/O operations ──
    from app.llm_mode import set_mode
    from app.firebase_reporter import read_llm_mode_from_firestore
    _cleanup_task = asyncio.to_thread(cleanup_stale_tasks)
    _mode_task = asyncio.to_thread(read_llm_mode_from_firestore)
    _, firestore_mode = await asyncio.gather(_cleanup_task, _mode_task)
    report_system_online()
    _publish_schedule()
    initial_mode = firestore_mode or settings.llm_mode
    set_mode(initial_mode)
    if firestore_mode:
        logger.info(f"LLM mode restored from dashboard: {firestore_mode}")
    else:
        report_llm_mode(settings.llm_mode)
    start_mode_listener()
    start_kb_queue_poller()
    start_phil_queue_poller()
    start_fiction_queue_poller()

    # Chat inbox poller — processes messages sent from the dashboard
    # and delivers them through the same Commander pipeline as Signal
    def _handle_chat_message(text: str) -> str:
        """Process a dashboard chat message through Commander, same as Signal.

        This is a SYNC function — called from the Firebase polling thread.
        Dashboard input is UNTRUSTED — apply same sanitization as Signal messages.
        """
        from app.sanitize import sanitize_input
        text = sanitize_input(text)
        if not text.strip():
            return "Empty or blocked message."
        # Run commander synchronously (we're already in a thread)
        result = commander.handle(text, settings.signal_owner_number, [])
        # Mirror to Signal so the user sees it there too (sync call)
        try:
            from app.signal_client import _chunk_at_sentences, MAX_SIGNAL_LENGTH
            signal_client._send_sync(settings.signal_owner_number, f"[Dashboard] {text}"[:MAX_SIGNAL_LENGTH])
            from app.agents.commander import _MAX_RESPONSE_LENGTH, truncate_for_signal
            resp_text = truncate_for_signal(result) if len(result) > _MAX_RESPONSE_LENGTH else result
            for chunk in _chunk_at_sentences(resp_text, MAX_SIGNAL_LENGTH):
                signal_client._send_sync(settings.signal_owner_number, chunk)
        except Exception:
            logger.debug("Failed to mirror chat response to Signal", exc_info=True)
        return result

    start_chat_inbox_poller(_handle_chat_message)

    # Idle scheduler — run background work (self-improvement, retrospective, evolution)
    # during idle time. Kill switch read from Firestore config/background_tasks.
    bg_enabled = await asyncio.to_thread(idle_scheduler.read_background_enabled)
    if bg_enabled is not None:
        idle_scheduler.set_enabled(bg_enabled)
        logger.info(f"Background tasks: {'enabled' if bg_enabled else 'disabled'} (from dashboard)")
    idle_scheduler.start_background_listener()
    idle_scheduler.start()
    logger.info("Idle scheduler started — background work runs when no user tasks active")

    # Initialize versioned prompt registry — extracts souls/*.md on first boot
    try:
        from app.prompt_registry import init_registry
        await asyncio.to_thread(init_registry)
    except Exception:
        logger.warning("Prompt registry initialization failed (non-fatal)", exc_info=True)

    # Initialize version manifest — create initial manifest if none exists
    try:
        from app.version_manifest import get_current_manifest, create_manifest, MANIFESTS_DIR
        MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
        if not get_current_manifest():
            await asyncio.to_thread(create_manifest, "system", "initial boot manifest")
            logger.info("Version manifest: created initial manifest")
    except Exception:
        logger.warning("Version manifest initialization failed (non-fatal)", exc_info=True)

    # Health monitor — wire self-healer alerts
    try:
        from app.health_monitor import get_monitor
        from app.self_healer import SelfHealer
        monitor = get_monitor()
        healer = SelfHealer()

        async def _on_health_alert(alerts):
            try:
                await healer.handle_alerts(alerts)
            except Exception:
                logger.debug("Self-healer alert handling failed", exc_info=True)

        def _sync_alert_handler(alerts):
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(_on_health_alert(alerts))
            except Exception:
                pass

        monitor.on_alert(_sync_alert_handler)
        logger.info("Health monitor + self-healer initialized")
    except Exception:
        logger.warning("Health monitor initialization failed (non-fatal)", exc_info=True)

    # ── Agent Zero amendments initialization ────────────────────────────
    try:
        if settings.lifecycle_hooks_enabled:
            from app.lifecycle_hooks import get_registry
            hook_reg = get_registry()
            logger.info(f"Lifecycle hooks initialized ({len(hook_reg.list_hooks())} hooks)")
    except Exception:
        logger.warning("Lifecycle hooks init failed (non-fatal)", exc_info=True)

    try:
        if settings.project_isolation_enabled:
            from app.project_isolation import get_manager
            pm = get_manager()
            logger.info(f"Project isolation initialized ({len(pm.list_projects())} projects)")
    except Exception:
        logger.warning("Project isolation init failed (non-fatal)", exc_info=True)

    # Create philosophy KB directories
    os.makedirs("/app/workspace/philosophy/texts", exist_ok=True)

    # ── PARALLELIZED STARTUP: run independent I/O tasks concurrently ──
    # These three operations are independent and previously ran sequentially
    # (~2-3s total). Running in parallel saves ~1-2s on cold boot.
    async def _report_phil():
        try:
            from app.firebase_reporter import report_philosophy_kb
            await asyncio.to_thread(report_philosophy_kb)
        except Exception:
            pass

    async def _gen_chronicle():
        try:
            from app.memory.system_chronicle import generate_and_save
            await asyncio.to_thread(generate_and_save)
            logger.info("System chronicle generated.")
        except Exception:
            logger.warning("System chronicle generation failed (non-fatal)", exc_info=True)

    async def _report_monitor():
        try:
            from app.firebase_reporter import report_system_monitor
            await asyncio.to_thread(report_system_monitor)
            logger.info("System monitor reported to dashboard")
        except Exception:
            logger.debug("System monitor report failed (non-fatal)", exc_info=True)

    await asyncio.gather(_report_phil(), _gen_chronicle(), _report_monitor())

    logger.info("CrewAI Agent Team started")
    yield
    # Final sync on clean shutdown
    idle_scheduler.stop()
    idle_scheduler.stop_listener()
    if settings.workspace_backup_repo:
        await asyncio.to_thread(sync_workspace, settings.workspace_backup_repo)
    report_system_offline()
    scheduler.shutdown()


def _publish_schedule() -> None:
    """Push current scheduler job list to Firestore."""
    try:
        jobs = []
        for job in scheduler.get_jobs():
            next_run = job.next_run_time
            jobs.append({
                "id": job.id,
                "name": job.name or job.id,
                "next_run": next_run.isoformat() if next_run else None,
                "cron": str(job.trigger),
            })
        report_schedule(jobs)
    except Exception:
        logger.debug("Failed to publish schedule", exc_info=True)


app = FastAPI(title="CrewAI Agent Gateway", lifespan=lifespan)

# ── CORS — allow control plane dashboard (port 3100) to call API (port 8765) ──
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3100", "http://127.0.0.1:3100",
                   "http://100.85.195.121:3100",
                   "http://plgs-macbook-pro---andrus:3100",
                   "http://plgs-macbook-pro---andrus.tail5b289b.ts.net:3100"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount philosophy API routes
from app.philosophy.api import philosophy_router
app.include_router(philosophy_router)

# ── Middleware (extracted to app/middleware.py) ────────────────────────────────
from app.middleware import add_middleware
add_middleware(app, settings)

signal_client = SignalClient()
commander = Commander()
_signal_msg_count = 0
_inflight_tasks = 0  # count of currently running commander.handle() calls
_inflight_lock = threading.Lock()


def _verify_gateway_secret(request: Request) -> bool:
    """Verify the forwarder is authenticated with the gateway secret."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return False
    token = auth[7:]
    return hmac.compare_digest(token, get_gateway_secret())


@app.post("/signal/inbound")
async def receive_signal(request: Request):
    # Authenticate the request source
    if not _verify_gateway_secret(request):
        raise HTTPException(status_code=401, detail="Unauthorized")

    payload = await request.json()

    # ── Handle reaction feedback (from forwarder) ──────────────────────
    if payload.get("type") == "reaction_feedback":
        sender = payload.get("sender", "")
        if not is_authorized_sender(sender):
            raise HTTPException(status_code=403, detail="Forbidden")
        try:
            from app.feedback_pipeline import FeedbackPipeline
            from app.config import get_settings
            _s = get_settings()
            if _s.mem0_postgres_url:
                pipeline = _get_feedback_pipeline()
                if pipeline:
                    from app.security import _sender_hash
                    sender_id = _sender_hash(sender)
                    asyncio.get_running_loop().run_in_executor(
                        None,
                        pipeline.process_reaction,
                        sender_id,
                        payload.get("emoji", ""),
                        payload.get("target_timestamp", 0),
                        payload.get("is_remove", False),
                    )
            # Acknowledge with 👀 on the message the user reacted to
            target_ts = payload.get("target_timestamp", 0)
            if target_ts and not payload.get("is_remove", False):
                try:
                    client = SignalClient()
                    await client.react(
                        recipient=sender,
                        emoji="👀",
                        target_author=_s.signal_bot_number,
                        target_timestamp=target_ts,
                    )
                except Exception:
                    logger.debug("Failed to send 👀 ack for feedback", exc_info=True)
        except Exception:
            logger.debug("Feedback reaction processing failed", exc_info=True)
        return {"status": "accepted"}

    # ── Handle regular messages ────────────────────────────────────────
    sender = payload.get("sender", "")
    text = payload.get("message", "").strip()
    timestamp = payload.get("timestamp", 0)
    attachments = payload.get("attachments", [])

    if not is_authorized_sender(sender):
        log_security_event("unauthorized_sender", _redact_number(sender))
        raise HTTPException(status_code=403, detail="Forbidden")
    if not is_within_rate_limit(sender):
        log_security_event("rate_limit_exceeded", _redact_number(sender))
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    if not text and not attachments:
        return {"status": "ignored"}

    # Enforce message length limit + sanitize input
    if len(text) > MAX_MESSAGE_LENGTH:
        text = text[:MAX_MESSAGE_LENGTH]
    from app.sanitize import sanitize_input
    text = sanitize_input(text)

    # Cap attachments (prevent abuse)
    attachments = attachments[:5]

    log_request_received(_redact_number(sender), len(text))

    # Send 👀 reaction immediately — before any other processing.
    # Fire-and-forget: don't await the Signal API roundtrip (~3s).
    if timestamp:
        asyncio.ensure_future(_safe_react(sender, timestamp))

    # Track Signal connection health for dashboard (fire-and-forget)
    global _signal_msg_count
    _signal_msg_count += 1
    from app.firebase_reporter import report_signal_status
    report_signal_status(
        connected=True,
        last_message_at=datetime.now(timezone.utc).isoformat(),
        message_count=_signal_msg_count,
    )

    asyncio.create_task(handle_task(sender, text, attachments, timestamp))
    return {"status": "accepted"}


# Lazy-initialized feedback pipeline singleton
_feedback_pipeline_instance = None

def _get_feedback_pipeline():
    """Get or create the feedback pipeline singleton."""
    global _feedback_pipeline_instance
    if _feedback_pipeline_instance is None:
        try:
            from app.feedback_pipeline import FeedbackPipeline
            s = get_settings()
            if s.mem0_postgres_url:
                _feedback_pipeline_instance = FeedbackPipeline(s.mem0_postgres_url)
        except Exception:
            logger.debug("Feedback pipeline initialization failed", exc_info=True)
    return _feedback_pipeline_instance


async def _safe_react(sender: str, msg_timestamp: int):
    """Send 👀 reaction in background — never block the handler."""
    try:
        await signal_client.react(sender, "👀", sender, msg_timestamp)
    except Exception:
        logger.debug("Failed to send 👀 reaction", exc_info=True)


async def handle_task(sender: str, text: str, attachments: list = None,
                      msg_timestamp: int = 0):
    global _inflight_tasks
    import time as _time
    _task_start = _time.monotonic()
    # Start task tracking for metrics
    task_row_id = start_task(sender)

    # 👀 reaction is already sent in receive_signal() — no need to send again here

    # Track activity for idle scheduler
    idle_scheduler.notify_task_start()

    # Notify user if other tasks are already in progress
    with _inflight_lock:
        currently_running = _inflight_tasks
        _inflight_tasks += 1
    if currently_running > 0:
        try:
            await signal_client.send(
                sender,
                f"Queued — {currently_running} task(s) in progress. I'll get to this next."
            )
        except Exception:
            pass

    try:
        # Persist the incoming message before processing so history is available
        # even if the response fails
        att_note = ""
        if attachments:
            att_note = f" [+{len(attachments)} attachment(s)]"
        add_message(sender, "user", text + att_note)

        # ── Agent Zero amendments: history + project detection ────────
        try:
            from app.config import get_settings as _gs
            _s = _gs()

            # History compression: track message in compressed history
            if _s.history_compression_enabled:
                from app.history_compression import get_history, Message as HMsg
                from app.security import _sender_hash
                h = get_history(_sender_hash(sender))
                h.start_new_topic()
                h.add_message(HMsg(role="user", content=text + att_note))

            # Project isolation: auto-detect venture from task text
            if _s.project_isolation_enabled:
                from app.project_isolation import get_manager as _get_pm
                _pm = _get_pm()
                detected = _pm.detect_project(text)
                if detected:
                    _pm.activate(detected)
                    logger.debug(f"Project detected: {detected}")
        except Exception:
            logger.debug("Amendment hooks (history/project) failed", exc_info=True)

        # Mirror to dashboard chat (so dashboard users see Signal messages)
        report_chat_message("user", text + att_note, source="signal")

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            _commander_pool, commander.handle, text, sender, attachments or []
        )
        log_response_sent(_redact_number(sender), len(result))

        # Record which crew handled the task (for per-crew analytics)
        try:
            from app.conversation_store import update_task_crew
            crew_used = getattr(commander, '_last_crew', '') or ''
            if crew_used:
                update_task_crew(task_row_id, crew_used)
        except Exception:
            pass

        # Persist the assistant reply (full text for conversation history)
        add_message(sender, "assistant", result)

        # Mirror to dashboard chat (so dashboard users see Signal responses)
        report_chat_message("assistant", result, source="signal")

        # ── Agent Zero: record response in compressed history + async compress ──
        try:
            from app.config import get_settings as _gs2
            _s2 = _gs2()
            if _s2.history_compression_enabled:
                from app.history_compression import get_history, Message as HMsg
                from app.security import _sender_hash
                h = get_history(_sender_hash(sender))
                h.add_message(HMsg(role="assistant", content=result[:4000]))
                if h.needs_compression:
                    h.compress_async()
        except Exception:
            logger.debug("History compression post-response failed", exc_info=True)

        # Extract facts into Mem0 persistent memory (fire-and-forget background task)
        asyncio.get_running_loop().run_in_executor(None, _extract_to_mem0, text, result)

        # Record response metadata for feedback correlation (fire-and-forget)
        try:
            pipeline = _get_feedback_pipeline()
            if pipeline:
                from app.prompt_registry import get_prompt_versions_map
                from app.security import _sender_hash
                # We'll get the actual Signal send timestamp after sending;
                # for now use msg_timestamp as a correlation key
                asyncio.get_running_loop().run_in_executor(
                    None,
                    pipeline.record_response_metadata,
                    msg_timestamp,  # placeholder — updated after send
                    _sender_hash(sender),
                    text[:2000],
                    result[:2000],
                    commander.last_crew_used if hasattr(commander, 'last_crew_used') else "",
                    get_prompt_versions_map(),
                    commander.last_model_used if hasattr(commander, 'last_model_used') else "",
                    task_row_id,
                )
        except Exception:
            logger.debug("Response metadata recording failed", exc_info=True)

        # Check if user's message is a correction of the previous response
        try:
            pipeline = _get_feedback_pipeline()
            if pipeline:
                from app.conversation_store import get_history
                from app.security import _sender_hash
                history = get_history(sender, n=2)
                if history:
                    # Get the previous assistant response for context
                    asyncio.get_running_loop().run_in_executor(
                        None,
                        pipeline.process_correction,
                        _sender_hash(sender),
                        text,
                        text,  # current task as context
                        result[:500],  # recent response
                        commander.last_crew_used if hasattr(commander, 'last_crew_used') else "",
                        0,  # prompt version (will be looked up)
                    )
        except Exception:
            logger.debug("Correction detection failed", exc_info=True)

        # Record successful task completion with timing
        complete_task(task_row_id, success=True)

        # Record health metrics for this interaction
        try:
            from app.health_monitor import InteractionMetrics, record_interaction
            import time as _time
            record_interaction(InteractionMetrics(
                timestamp=_time.time(),
                task_id=str(task_row_id),
                success=True,
                latency_ms=((_time.monotonic() - _task_start) * 1000) if '_task_start' in dir() else 0,
                crew_used=commander.last_crew_used if hasattr(commander, 'last_crew_used') else "",
            ))
        except Exception:
            pass

        # ── Prepare for Signal delivery ─────────────────────────────────
        from app.agents.commander import _MAX_RESPONSE_LENGTH, truncate_for_signal

        if len(result) > _MAX_RESPONSE_LENGTH:
            signal_text = truncate_for_signal(result)
            # Write .md file in background — don't block Signal delivery
            md_future = asyncio.get_running_loop().run_in_executor(
                None, _write_response_md, result, text
            )
            await signal_client.send(sender, signal_text)
            # Attach file in a follow-up if write succeeded
            try:
                md_path = await asyncio.wait_for(md_future, timeout=5)
                if md_path:
                    await signal_client.send(sender, "", attachments=[md_path])
            except Exception:
                pass
        else:
            await signal_client.send(sender, result)
    except Exception as exc:
        logger.exception("Error handling task")
        log_security_event("task_error", "unhandled exception in handle_task")

        # Record failed task for metrics
        complete_task(task_row_id, success=False, error_type=type(exc).__name__)

        # Record health metrics for failed interaction
        try:
            from app.health_monitor import InteractionMetrics, record_interaction
            record_interaction(InteractionMetrics(
                timestamp=_time.time(),
                task_id=str(task_row_id),
                success=False,
                latency_ms=(_time.monotonic() - _task_start) * 1000,
                error_type=type(exc).__name__,
                crew_used=commander.last_crew_used if hasattr(commander, 'last_crew_used') else "",
            ))
        except Exception:
            pass

        # Trigger self-healing: diagnose the error in the background
        diagnose_and_fix(
            crew="handle_task",
            user_input=text,
            error=exc,
            context=f"attachments={len(attachments or [])}",
        )
        # Generic error — do not leak internals to Signal
        await signal_client.send(
            sender,
            "Something went wrong processing your request. "
            "The self-healing system is analyzing the error and will attempt a fix. "
            "Please try again shortly."
        )
    finally:
        with _inflight_lock:
            _inflight_tasks -= 1
        idle_scheduler.notify_task_end()


# ── API routers (extracted from main.py to app/api/) ──────────────────────────
from app.api.config_api import router as config_router
app.include_router(config_router, prefix="/config")


# ── Knowledge Base router (extracted to app/api/kb.py) ────────────────────────
from app.api.kb import router as kb_router
app.include_router(kb_router, prefix="/kb")


# ── Health + Dashboard router (extracted to app/api/health.py) ─────────────────
from app.api.health import router as health_router
app.include_router(health_router)

# ── Control Plane API routes ─────────────────────────────────────────────
try:
    from app.control_plane.dashboard_api import router as cp_router
    app.include_router(cp_router)
    logger.info("Control Plane API mounted at /api/cp/")
except Exception:
    logger.debug("Control Plane API not available", exc_info=True)

# ── React Dashboard (self-hosted, replaces Firebase) ─────────────────────
# Mount LAST so API routes take precedence. Serves the React SPA build.
from pathlib import Path as _Path
_dashboard_build = _Path("/app/dashboard/build")
if _dashboard_build.exists() and (_dashboard_build / "index.html").exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/cp", StaticFiles(directory=str(_dashboard_build), html=True), name="dashboard-cp")
    logger.info(f"React dashboard mounted at /cp/ from {_dashboard_build}")
