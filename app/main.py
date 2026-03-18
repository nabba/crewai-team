import hmac
import logging
import logging.handlers
import asyncio
import os

# Ensure all loggers output to stdout so docker logs captures tracebacks
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# Install API rate throttle BEFORE any litellm/crewai imports to monkey-patch early
from app.rate_throttle import install_throttle
install_throttle()

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
    cleanup_stale_tasks,
)
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

settings = get_settings()
logger = logging.getLogger(__name__)
scheduler = AsyncIOScheduler()


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
    # Heartbeat — keep monitoring dashboard "last_seen" fresh
    scheduler.add_job(heartbeat, "interval", seconds=60, id="heartbeat")
    scheduler.start()

    # Clean up zombie tasks from previous container, then report online
    cleanup_stale_tasks()
    report_system_online()
    _publish_schedule()

    logger.info("CrewAI Agent Team started")
    yield
    # Final sync on clean shutdown
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1"],
    allow_methods=["POST"],
    allow_headers=["Content-Type", "Authorization"],
)

signal_client = SignalClient()
commander = Commander()


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
    sender = payload.get("sender", "")
    text = payload.get("message", "").strip()
    attachments = payload.get("attachments", [])

    if not is_authorized_sender(sender):
        log_security_event("unauthorized_sender", _redact_number(sender))
        raise HTTPException(status_code=403, detail="Forbidden")
    if not is_within_rate_limit(sender):
        log_security_event("rate_limit_exceeded", _redact_number(sender))
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    if not text and not attachments:
        return {"status": "ignored"}

    # Enforce message length limit
    if len(text) > MAX_MESSAGE_LENGTH:
        text = text[:MAX_MESSAGE_LENGTH]

    # Cap attachments (prevent abuse)
    attachments = attachments[:5]

    log_request_received(_redact_number(sender), len(text))
    asyncio.create_task(handle_task(sender, text, attachments))
    return {"status": "accepted"}


async def handle_task(sender: str, text: str, attachments: list = None):
    # Start task tracking for metrics
    task_row_id = start_task(sender)

    try:
        # Persist the incoming message before processing so history is available
        # even if the response fails
        att_note = ""
        if attachments:
            att_note = f" [+{len(attachments)} attachment(s)]"
        add_message(sender, "user", text + att_note)

        result = await asyncio.to_thread(
            commander.handle, text, sender, attachments or []
        )
        log_response_sent(_redact_number(sender), len(result))

        # Persist the assistant reply
        add_message(sender, "assistant", result)

        # Record successful task completion with timing
        complete_task(task_row_id, success=True)

        await signal_client.send(sender, result)
    except Exception as exc:
        logger.exception("Error handling task")
        log_security_event("task_error", "unhandled exception in handle_task")

        # Record failed task for metrics
        complete_task(task_row_id, success=False, error_type=type(exc).__name__)

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


@app.get("/health")
async def health():
    return {"status": "ok"}
