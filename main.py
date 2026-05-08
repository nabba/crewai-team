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
from app.conversation_store import add_message
from app.workspace_sync import setup_workspace_repo, sync_workspace
from app.firebase_reporter import (
    report_system_online, report_system_offline, heartbeat, report_schedule
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

    # Phase 7: scheduled jobs ping Signal + Web Push when they finish.
    # Self-improvement: notify on every run (daily; success matters as a
    # heartbeat). Workspace sync: failure-only — hourly success would spam.
    # Heartbeat: silent (60s cadence; no notifications wanted).
    from app.notify import notify_on_complete

    scheduler.add_job(
        notify_on_complete(label="Self-improvement")(SelfImprovementCrew().run),
        trigger,
    )
    scheduler.add_job(
        notify_on_complete(
            label="Workspace sync", notify_on_failure_only=True,
        )(sync_workspace),
        sync_trigger,
        kwargs={"backup_repo": settings.workspace_backup_repo},
    )
    # Heartbeat — keep monitoring dashboard "last_seen" fresh.
    # No notify wrapper: 60s cadence would flood Signal + Web Push.
    scheduler.add_job(heartbeat, "interval", seconds=60, id="heartbeat")
    scheduler.start()

    # Report online status and publish schedule to Firebase
    report_system_online()
    _publish_schedule()

    # Discord connector (opt-in via DISCORD_ENABLED).
    try:
        from app.discord_client import start_bot as _discord_start
        await _discord_start()
    except Exception:
        logger.exception("Discord bot startup failed (non-fatal)")

    logger.info("CrewAI Agent Team started")
    yield
    # Final sync on clean shutdown
    if settings.workspace_backup_repo:
        await asyncio.to_thread(sync_workspace, settings.workspace_backup_repo)
    try:
        from app.discord_client import stop_bot as _discord_stop
        await _discord_stop()
    except Exception:
        logger.debug("Discord bot shutdown raised", exc_info=True)
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
    attachments = payload.get("attachments") or []

    if not is_authorized_sender(sender):
        log_security_event("unauthorized_sender", _redact_number(sender))
        raise HTTPException(status_code=403, detail="Forbidden")
    if not is_within_rate_limit(sender):
        log_security_event("rate_limit_exceeded", _redact_number(sender))
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    # Voice-note path — if an audio attachment came in and voice mode is
    # active, transcribe it and treat the transcript as the user's message.
    # Sets an "active" flag on the sender so the reply path knows to TTS.
    if not text and attachments:
        text = await asyncio.to_thread(_maybe_transcribe_voice, sender, attachments)

    if not text:
        return {"status": "ignored"}

    # Enforce message length limit
    if len(text) > MAX_MESSAGE_LENGTH:
        text = text[:MAX_MESSAGE_LENGTH]

    log_request_received(_redact_number(sender), len(text))
    asyncio.create_task(handle_task(sender, text))
    return {"status": "accepted"}


def _maybe_transcribe_voice(sender: str, attachments: list) -> str:
    """Inspect inbound attachments; if any is audio, transcribe and return
    the transcript. Marks the sender as "voice-active" so the reply path
    answers with TTS. Returns "" if no audio attachment or transcription
    failed (caller treats the inbound as text-less)."""
    try:
        from app.voice import (
            AUDIO_MIME_PREFIXES, transcribe, mark_voice_inbound,
        )
        from app.runtime_settings import get_voice_mode
    except Exception:
        logger.debug("voice subsystem unavailable", exc_info=True)
        return ""

    if get_voice_mode() == "off":
        return ""

    audio_att = next(
        (a for a in attachments
         if (a.get("contentType") or "").startswith(AUDIO_MIME_PREFIXES)),
        None,
    )
    if not audio_att:
        return ""

    # Resolve the file via the same path resolver attachment_reader uses.
    try:
        from app.tools.attachment_reader import _safe_path
    except Exception:
        return ""
    fn = audio_att.get("filename") or audio_att.get("id") or ""
    if not fn:
        return ""
    path = _safe_path(fn, audio_att.get("contentType", ""))
    if path is None or not path.exists():
        logger.info(f"voice inbound: audio file not found: {fn!r}")
        return ""

    try:
        audio_bytes = path.read_bytes()
    except OSError as exc:
        logger.warning(f"voice inbound: read failed: {exc}")
        return ""

    fmt = (audio_att.get("contentType") or "audio/m4a").split("/")[-1]
    transcript = transcribe(audio_bytes, audio_format=fmt)
    if not transcript:
        logger.info("voice inbound: transcription returned empty")
        return ""

    mark_voice_inbound(sender)
    logger.info(
        f"voice inbound from {_redact_number(sender)}: {len(transcript)} chars transcribed"
    )
    return transcript


async def handle_task(sender: str, text: str):
    try:
        # Persist the incoming message before processing so history is available
        # even if the response fails
        add_message(sender, "user", text)

        result = await asyncio.to_thread(commander.handle, text, sender)

        # Phase 8: optionally reword the reply in the concierge voice for
        # conversational warmth on Signal DMs. Skipped automatically for
        # structured outputs (slash-command help, JSON, code blocks, etc.).
        # The wrapper is a no-op when the runtime toggle is off.
        try:
            from app.personality.concierge_wrapper import apply_concierge
            result = await asyncio.to_thread(apply_concierge, result)
        except Exception:
            logger.debug("concierge wrap failed (non-fatal)", exc_info=True)

        log_response_sent(_redact_number(sender), len(result))

        # Persist the assistant reply
        add_message(sender, "assistant", result)

        # Route the reply to whichever surface the message came in on.
        # ``discord:<user_id>`` senders go through the Discord bot;
        # everything else falls back to Signal (the original surface).
        if sender.startswith("discord:"):
            await asyncio.to_thread(_send_discord_reply, sender, result)
        else:
            # If the inbound was voice and voice mode is on, deliver the reply
            # as a TTS attachment. Falls back to plain text on synthesis failure.
            voice_path = await asyncio.to_thread(_maybe_synthesize_reply, sender, result)
            if voice_path is not None:
                await signal_client.send(sender, result, attachments=[voice_path])
            else:
                await signal_client.send(sender, result)
    except Exception:
        logger.exception("Error handling task")
        log_security_event("task_error", "unhandled exception in handle_task")
        # Generic error — do not leak internals to the user surface
        msg = "Sorry, something went wrong processing your request. Please try again."
        if sender.startswith("discord:"):
            await asyncio.to_thread(_send_discord_reply, sender, msg)
        else:
            await signal_client.send(sender, msg)


def _send_discord_reply(sender: str, text: str) -> None:
    """Translate a 'discord:<user_id>' sender into a DM via the bot."""
    try:
        from app.discord_client import send_via_discord
        user_id = sender[len("discord:"):]
        ok, detail = send_via_discord(user_id, body=text)
        if not ok:
            logger.warning(f"discord reply failed: {detail}")
    except Exception:
        logger.exception("discord reply dispatch failed")


def _maybe_synthesize_reply(sender: str, text: str) -> str | None:
    """Synthesize ``text`` to a host-side audio file when the sender is
    voice-active. Returns the absolute host path of the audio attachment
    (suitable for signal-cli) or None to send a text-only reply."""
    try:
        from app.voice import is_voice_active, synthesize, clear_voice_state
        from app.runtime_settings import get_voice_mode
        from app.voice.tts import TTS_OUTPUT_FORMAT
        from app.paths import WORKSPACE_ROOT
    except Exception:
        return None

    if get_voice_mode() == "off":
        return None
    if not is_voice_active(sender):
        return None

    audio = synthesize(text)
    if audio is None:
        return None

    fmt = TTS_OUTPUT_FORMAT.get("latest", "ogg") or "ogg"
    out_dir = WORKSPACE_ROOT / "voice_tmp"
    out_dir.mkdir(parents=True, exist_ok=True)
    docker_path = out_dir / f"reply_{int(__import__('time').time())}.{fmt}"
    try:
        docker_path.write_bytes(audio)
    except OSError as exc:
        logger.warning(f"voice outbound: write failed: {exc}")
        return None

    # signal-cli runs on the host and reads attachment paths from the host
    # filesystem — translate the Docker workspace path accordingly.
    s = settings
    host_root = (s.workspace_host_path or "").rstrip("/")
    if host_root:
        host_path = host_root + str(docker_path).removeprefix("/app/workspace")
    else:
        host_path = str(docker_path)

    # Forget the voice flag now that we've delivered one voice reply — the
    # next reply lands as text unless the user sends another voice note.
    clear_voice_state(sender)
    logger.info(f"voice outbound to {_redact_number(sender)}: {len(audio)} bytes ({fmt})")
    return host_path


@app.get("/health")
async def health():
    return {"status": "ok"}
