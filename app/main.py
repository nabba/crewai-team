import hmac
import logging
import logging.handlers
import asyncio
import os

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.config import get_settings, get_gateway_secret
from app.security import is_authorized_sender, is_within_rate_limit, _redact_number
from app.signal_client import SignalClient
from app.agents.commander import Commander
from app.audit import (
    log_request_received, log_response_sent, log_security_event
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
    try:
        trigger = CronTrigger.from_crontab(settings.self_improve_cron)
    except ValueError as exc:
        raise RuntimeError(
            f"Invalid SELF_IMPROVE_CRON expression {settings.self_improve_cron!r}: {exc}"
        ) from exc
    scheduler.add_job(SelfImprovementCrew().run, trigger)
    scheduler.start()
    logger.info("CrewAI Agent Team started")
    yield
    scheduler.shutdown()


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

    if not is_authorized_sender(sender):
        log_security_event("unauthorized_sender", _redact_number(sender))
        raise HTTPException(status_code=403, detail="Forbidden")
    if not is_within_rate_limit(sender):
        log_security_event("rate_limit_exceeded", _redact_number(sender))
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    if not text:
        return {"status": "ignored"}

    # Enforce message length limit
    if len(text) > MAX_MESSAGE_LENGTH:
        text = text[:MAX_MESSAGE_LENGTH]

    log_request_received(_redact_number(sender), len(text))
    asyncio.create_task(handle_task(sender, text))
    return {"status": "accepted"}


async def handle_task(sender: str, text: str):
    try:
        result = await asyncio.to_thread(commander.handle, text)
        log_response_sent(_redact_number(sender), len(result))
        await signal_client.send(sender, result)
    except Exception:
        logger.exception("Error handling task")
        log_security_event("task_error", "unhandled exception in handle_task")
        # Generic error — do not leak internals to Signal
        await signal_client.send(sender, "Sorry, something went wrong processing your request. Please try again.")


@app.get("/health")
async def health():
    return {"status": "ok"}
