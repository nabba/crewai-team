import hmac
import logging
import logging.handlers
import asyncio
import os
import re
import threading
from datetime import datetime, timezone, timedelta

# Install uvloop as the asyncio event loop policy BEFORE FastAPI/uvicorn pick it up.
# 3-4x faster than the default asyncio loop on I/O-heavy workloads (retrieval,
# Mem0 search, webhook concurrency). Safe no-op if uvloop isn't installed.
try:
    import uvloop
    uvloop.install()
except ImportError:
    pass

# Ensure all loggers output to stdout so docker logs captures tracebacks
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# Structured error logging (JSON file for aggregation/debugging)
try:
    from app.error_handler import setup_structured_logging
    setup_structured_logging()
except Exception:
    pass  # Non-fatal — structured logging is additive

# Install API rate throttle BEFORE any litellm/crewai imports to monkey-patch early
from app.rate_throttle import install_throttle
install_throttle()

# Install Anthropic prompt-caching hook — must be before any litellm.completion call
from app.prompt_cache_hook import install_cache_hook
install_cache_hook()

# Install process-wide wall-clock timeout on every CrewAI BaseTool.run.
# Must be before agents/tools are constructed so the patched method
# lands on every tool instance. See app/tools_timeout.py for the
# motivation (2026-04-22 PSP task had recall_facts hung for 2h 11m
# because no tool-level timeout existed).
from app.tools_timeout import install as install_tool_timeouts
install_tool_timeouts()

# Subscribe the single LLM-activity heartbeat to CrewAI's event bus.  Covers
# every CrewAI-routed provider (native Anthropic/OpenAI/Gemini/Azure/Bedrock
# AND the LiteLLM-mediated generic ``crewai.LLM``) through one subscriber
# instead of per-path instrumentation.  See app/observability/llm_events.py.
from app.observability.llm_events import install as install_llm_event_subscriber
install_llm_event_subscriber()

# Populate the crew-dispatch registry so the commander resolves crew names
# via a registered table rather than an if/elif chain.  See
# app/crews/registry.py for the full list + adapters.
from app.crews.registry import install_defaults as _install_crew_registry
_install_crew_registry()

# Ensure the dossier subsystem is wired up at boot:
#   * importing the package fires @register_tool for `build_company_dossier`
#     so it shows up in `tool_search` and the /api/cp/tools panel
#   * the dossier collector's adapters install lazily on first call,
#     so this import is just for tool-registry visibility
# The crew itself is registered via _install_crew_registry above; this
# import is purely for the tool-registry side-effect.  See
# app/dossier/__init__.py for the package init that pulls
# app.dossier.tools.
try:
    import app.dossier  # noqa: F401
except Exception as _exc:
    logger = logging.getLogger(__name__)
    logger.warning("dossier subsystem import failed at boot: %s — "
                   "build_company_dossier will not be discoverable until "
                   "the issue is resolved", _exc)

# Subscribe the default lifecycle event handlers (belief state, Firebase,
# metric, journal, auto-skill distillation) to the crew event bus.
# Adding a new sink (e.g. SubIA pre/post hooks once that layer goes live)
# is a single @crew_events.on_crew_completed() registration somewhere —
# no more editing app/crews/lifecycle.py to add another inline call.
from app.crews.event_handlers import install_defaults as _install_crew_event_handlers
_install_crew_event_handlers()

# Subscribe CrewAI event-bus listeners that persist agent/tool/LLM
# spans into control_plane.crew_task_spans. Powers the dashboard's
# task-flow drawer. Fail-soft on older CrewAI versions that don't
# expose the event types; see app/crews/span_events.py.
from app.crews.span_events import install_listeners as _install_span_listeners
_install_span_listeners()

from fastapi import FastAPI, Request, HTTPException
from contextlib import asynccontextmanager
from app.config import get_settings, get_gateway_secret
from app.security import is_authorized_sender, is_within_rate_limit, _redact_number
from app.signal_client import SignalClient
from app.agents.commander import Commander

# Load-bearing side-effect import — DO NOT remove or convert to lazy.
# Importing app.healing executes app/healing/__init__.py, which eager-
# starts daemon threads for the 22-monitor driver, the auditor bridge,
# the watchdog reaper, and three observational subsystems
# (capability_gap_analyzer, library_radar, proposal_bridge). Each
# submodule has a module-level start() at import-time. Previously this
# wiring was reached only transitively via the diagnose_and_fix import
# below — making the entire self-healing surface depend on a single
# from-import staying eager. See app/healing/__init__.py for the
# eager-wiring inventory and the noted history of subsystems whose
# daemons "never ran in production" before being anchored here.
import app.healing  # noqa: F401
from app.healing.error_diagnosis import diagnose_and_fix
from app.audit import (
    log_request_received, log_response_sent, log_security_event
)
from app.conversation_store import (
    add_message, start_task, complete_task,
    enqueue_inbound, mark_inbound_processing,
    mark_inbound_done, mark_inbound_failed,
    get_pending_inbound, prune_old_inbound,
)
from app.workspace_sync import setup_workspace_repo, sync_workspace
from app.firebase_reporter import (
    report_system_online, report_system_offline, heartbeat, report_schedule,
    cleanup_stale_tasks, report_llm_mode, start_mode_listener,
    start_kb_queue_poller, start_phil_queue_poller, start_fiction_queue_poller,
    start_episteme_queue_poller, start_experiential_queue_poller,
    start_aesthetics_queue_poller, start_tensions_queue_poller,
    report_chat_message, start_chat_inbox_poller,
)
from app import idle_scheduler
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

settings = get_settings()
logger = logging.getLogger(__name__)
# Generous misfire window: APScheduler's default ``misfire_grace_time``
# is 1 s, so any cron job delayed by even 3-4 s during normal load
# triggers ``Run time of job ... was missed by 0:00:03`` WARN spam in
# errors.jsonl (613 occurrences/week as of 2026-05-09 pattern_learner
# scan). Most of our jobs run on minute-or-coarser cadences, so a 60 s
# grace tolerates routine scheduling jitter without losing visibility
# into actual stalls (>60 s overruns still log). ``coalesce=True``
# means when the scheduler catches up after a pause, multiple missed
# runs collapse into one execution rather than firing in sequence.
scheduler = AsyncIOScheduler(
    job_defaults={"misfire_grace_time": 60, "coalesce": True},
)

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

def _write_response_md(full_text: str, user_question: str) -> str | None:
    return _write_response_md_impl(full_text, user_question, settings)


# ── Q3.2 (PROGRAM §40.2) — restart-claim self-check ──────────────────────


def _process_post_amendment_restart_claims() -> None:
    """At every boot: walk the post-amendment-restart-claim queue.
    For each claim, decide if the current process satisfies it. If
    yes, clear it. If no (and any remain), log a loud warning, send
    a Signal, and let the operator know they're running stale.

    The mechanism is generic so future Tier-3 amendments with hot-
    reload-incompatible effects can use it (just write a checker
    that looks at the live module state vs the claim's expected
    state).
    """
    try:
        from app.runtime_settings import (
            get_post_amendment_restart_claims,
            clear_post_amendment_restart_claims,
        )
    except Exception:
        return

    claims = get_post_amendment_restart_claims()
    if not claims:
        return

    satisfied_ids: list[str] = []
    unsatisfied: list[dict] = []

    for claim in claims:
        kind = claim.get("claim_kind", "")
        source = claim.get("source", "")
        if (
            kind == "restart_required"
            and source == "embedding_migration.cutover"
        ):
            # Satisfied when the live ``_EMBED_DIM`` matches the
            # plan's target dim — i.e. the running process is
            # already reading the post-cutover source.
            expected = claim.get("expected_embed_dim")
            try:
                from app.memory.chromadb_manager import get_embed_dim
                live = get_embed_dim()
                if expected is not None and int(expected) == int(live):
                    satisfied_ids.append(claim["id"])
                    continue
            except Exception:
                pass
            unsatisfied.append(claim)
        else:
            # Unknown claim kinds are kept (operator gets visibility);
            # this lets the mechanism be forward-compatible without a
            # release-coupled enum.
            unsatisfied.append(claim)

    if satisfied_ids:
        cleared = clear_post_amendment_restart_claims(ids=satisfied_ids)
        logger.info(
            "post-amendment restart claims: %d satisfied + cleared", cleared,
        )

    if unsatisfied:
        msg_lines = [
            f"⚠️ {len(unsatisfied)} post-amendment restart claim(s) "
            f"outstanding after boot — system is running STALE relative "
            f"to a recently-applied Tier-3 amendment:",
        ]
        for c in unsatisfied[:5]:
            msg_lines.append(
                f"  • [{c.get('claim_kind', '?')}] {c.get('source', '?')}: "
                f"{c.get('reason', '?')[:160]}"
            )
        if len(unsatisfied) > 5:
            msg_lines.append(f"  • …and {len(unsatisfied) - 5} more")
        msg = "\n".join(msg_lines)
        logger.warning(msg)
        try:
            from app.life_companion._common import send_signal_alert
            send_signal_alert(msg, tag="post_amendment_restart")
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.crews.self_improvement_crew import SelfImprovementCrew

    # ── Q3.3 (PROGRAM §40.3 Item 1) — Post-amendment restart claims FIRST.
    # Per FastAPI/uvicorn semantics, lifespan-startup runs BEFORE port-
    # binding, so the claim surface here is already before any route
    # serves. We moved it to the VERY FIRST step of lifespan (above the
    # gateway-bind check) for defense-in-depth: if a Tier-3 amendment
    # changed binding behavior AND the new code wasn't reloaded, the
    # operator wants the restart-needed alert even if the bind check is
    # about to crash. The check is failure-isolated; it never raises.
    try:
        _process_post_amendment_restart_claims()
    except Exception:
        logger.exception(
            "main: post-amendment restart-claim self-check raised; "
            "claims left in place for operator visibility",
        )

    # Fail fast if gateway is misconfigured to bind on a public interface.
    # Exception: when running inside Kubernetes (KUBERNETES_SERVICE_HOST is set
    # automatically by kubelet for every pod), 0.0.0.0 is correct — the pod
    # network namespace is private, k8s Services + NetworkPolicy provide the
    # actual access boundary. Detect k8s and allow 0.0.0.0 in that case.
    in_kubernetes = bool(os.environ.get("KUBERNETES_SERVICE_HOST"))
    if settings.gateway_bind != "127.0.0.1" and not in_kubernetes:
        raise RuntimeError(
            f"GATEWAY_BIND must be 127.0.0.1 outside Kubernetes, got "
            f"{settings.gateway_bind!r}. Refusing to start — binding to a "
            "public interface is unsafe on a host with no other access "
            "control. (In k8s, NetworkPolicy + Service do the gating.)"
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
    _ollama_up = await asyncio.to_thread(ollama_is_running)
    if _ollama_up:
        logger.info("Native Ollama detected — Metal GPU acceleration enabled")
    else:
        logger.warning(
            "Native Ollama not detected at %s. "
            "Local models will fall back to Claude API. "
            "Start Ollama with: ollama serve",
            settings.ollama_base_url,
        )

    # ── Stage 3.3: Ollama model preload ───────────────────────────────────
    # Fires off warm-up requests for the role models + the embedding model so
    # the first user request doesn't pay the 2-5 s Metal GPU cold-start cost.
    # Fire-and-forget — doesn't block gateway startup.
    # Gate: PRELOAD_OLLAMA=0 disables (default ON when Ollama is up).
    if _ollama_up and os.environ.get("PRELOAD_OLLAMA", "1") == "1":
        async def _preload_ollama_models() -> None:
            import httpx as _httpx
            models_to_warm = set()
            # Role models + default
            for attr in ("local_model_default", "local_model_coding",
                         "local_model_architecture", "local_model_research",
                         "local_model_writing"):
                m = getattr(settings, attr, None)
                if m:
                    models_to_warm.add(m)
            # Embedding model (for nomic-embed-text etc. used by ChromaDB)
            embed_model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
            models_to_warm.add(embed_model)

            base_url = settings.local_llm_base_url.rstrip("/")
            async with _httpx.AsyncClient(timeout=60.0) as client:
                for model in models_to_warm:
                    try:
                        # Warm-up: tiny request with long keep_alive
                        await client.post(
                            f"{base_url}/api/generate",
                            json={"model": model, "prompt": "",
                                  "keep_alive": "30m", "stream": False},
                        )
                        logger.info(f"Ollama preload: {model} warmed (keep_alive 30m)")
                    except Exception as exc:
                        logger.debug(f"Ollama preload skipped for {model}: {exc}")
        asyncio.create_task(_preload_ollama_models())

    # ── Stage 2: pgvector HNSW + Neo4j indexes (idempotent) ──────────────
    async def _apply_startup_migrations() -> None:
        try:
            from app.memory.startup_migrations import apply_all
            await asyncio.to_thread(apply_all)
        except Exception as exc:
            logger.debug(f"startup_migrations dispatch failed (non-fatal): {exc}")
    asyncio.create_task(_apply_startup_migrations())

    # ── Stage 3.4: Pre-open ChromaDB collections ──────────────────────────
    # First access to a collection pays a peek(1) dimension-check penalty.
    # Pay it once at startup, offline, so every user request sees a cached
    # collection handle. Gate: PRELOAD_CHROMA=0 disables.
    if os.environ.get("PRELOAD_CHROMA", "1") == "1":
        async def _preopen_chroma() -> None:
            try:
                from app.memory.chromadb_manager import _get_col
            except Exception as exc:
                logger.debug(f"ChromaDB preopen skipped (import failed): {exc}")
                return
            # Core collections used on the hot path.
            collections = [
                "team_shared", "commander",
                "episteme", "experiential", "aesthetics", "tensions",
            ]
            for name in collections:
                try:
                    await asyncio.to_thread(_get_col, name)
                    logger.info(f"ChromaDB preopen: {name}")
                except Exception as exc:
                    logger.debug(f"ChromaDB preopen skipped for {name}: {exc}")
        asyncio.create_task(_preopen_chroma())

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
    # ── Heartbeat publishers (every 60s) ─────────────────────────────────
    # The set of publishers and their cadences live in
    # app/observability/publishers.py so that adding a new dashboard
    # report is a single register() call, not an edit of the heartbeat
    # body here.  Each publisher runs under its own try/except so one
    # failing sink doesn't cancel the rest.
    from app.observability.publishers import install_defaults as _install_publishers
    from app.observability import publishers as _observability_publishers
    _install_publishers()
    logger.info(
        "observability.publishers: active publishers: %s",
        ", ".join(_observability_publishers.registered_names()),
    )

    _hb_counter = [0]
    def _heartbeat_tick() -> None:
        _hb_counter[0] += 1
        _observability_publishers.run_all(tick=_hb_counter[0])
    scheduler.add_job(_heartbeat_tick, "interval", seconds=60, id="heartbeat")

    # ── Stuck-ticket janitor ────────────────────────────────────────────
    # Background sweeper for control-plane tickets that are "in_progress"
    # with $0 cost for longer than 15 min.  This is the last line of
    # defense against orchestrator paths that hang past the 900s
    # handle_task timeout (Python can't cancel threads, so when the
    # auditor retry loop or a crew thread spins forever, the outer
    # safety net in handle_task() never fires and the ticket stays
    # in_progress indefinitely).  Runs every 5 min; no-ops when the
    # board is clean.
    def _run_stuck_ticket_janitor() -> list:
        try:
            from app.control_plane.tickets import get_tickets
            failed = get_tickets().fail_stuck_in_progress(
                max_age_minutes=15, only_zero_cost=True,
            )
            # Intentionally no-op-quiet when nothing to fail — the
            # fail_stuck_in_progress() method already logs a WARNING
            # with ids when it acts.
            return failed
        except Exception:
            logger.debug("Stuck-ticket janitor failed (non-fatal)",
                         exc_info=True)
            return []
    scheduler.add_job(
        _run_stuck_ticket_janitor,
        "interval",
        minutes=5,
        id="stuck_ticket_janitor",
    )
    logger.info("Stuck-ticket janitor scheduled: every 5 min "
                "(fails in_progress > 15min with $0 cost)")

    # ── Permanent error monitor ─────────────────────────────────────────
    # Tails workspace/logs/errors.jsonl every 5 min, groups errors by
    # signature, and writes detected anomalies (new patterns, rate spikes,
    # total-rate σ deviations) to control_plane.error_anomalies for the
    # React /cp/ops dashboard. See app/observability/error_monitor.py.
    def _run_error_monitor_scan() -> dict | None:
        try:
            from app.observability.error_monitor import scan
            return scan()
        except Exception:
            logger.debug("Error monitor scan failed (non-fatal)", exc_info=True)
            return None
    # Run the first scan immediately so the dashboard has data right
    # after boot (otherwise the /cp/ops Monitor tab shows zeros for the
    # first 5 minutes of every restart, which looks broken).
    from datetime import datetime as _dt
    scheduler.add_job(
        _run_error_monitor_scan,
        "interval",
        minutes=5,
        id="error_monitor_scan",
        next_run_time=_dt.now(),
    )
    logger.info("Error monitor scheduled: every 5 min "
                "(tails errors.jsonl, surfaces anomalies on /cp/ops)")

    # ── User-configurable schedules (loaded from workspace/schedules.json) ──
    try:
        from app.tools.schedule_manager_tools import register_user_schedules
        n_user_schedules = register_user_schedules(scheduler)
        if n_user_schedules:
            logger.info(f"Registered {n_user_schedules} user schedule(s)")
    except Exception:
        logger.debug("User schedule registration skipped", exc_info=True)

    # ── NL cron jobs (T2-6) — persisted across restarts ─────────────────
    try:
        from app.agents.commander.commands import restore_nl_jobs
        restored = restore_nl_jobs(scheduler, commander)
        if restored:
            logger.info(f"Restored {restored} natural-language scheduled job(s)")
    except Exception:
        logger.debug("NL cron restoration failed", exc_info=True)

    # ── Forge maintenance jobs (periodic re-audit, anomaly detection,
    #     hash-chain integrity verification) ──
    try:
        from app.forge.cron import register_periodic_jobs
        register_periodic_jobs(scheduler)
    except Exception:
        logger.debug("Forge cron registration failed", exc_info=True)

    # ── Affective layer: register POST_LLM_CALL/ON_COMPLETE hooks and
    #     schedule the daily 04:30 Helsinki reflection cycle. ──
    try:
        from app.affect.hooks import install as install_affect_hooks
        install_affect_hooks()
    except Exception:
        logger.debug("Affect layer install failed", exc_info=True)

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
    start_episteme_queue_poller()
    start_experiential_queue_poller()
    start_aesthetics_queue_poller()
    start_tensions_queue_poller()

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
        # Mirror to Signal so the user sees it there too (sync call).
        # BOTH the question AND the response are prefixed with [Dashboard]
        # so the user can distinguish dashboard-originated messages from
        # their own Signal queries.
        try:
            from app.signal_client import _chunk_at_sentences, MAX_SIGNAL_LENGTH
            signal_client._send_sync(settings.signal_owner_number, f"[Dashboard] {text}"[:MAX_SIGNAL_LENGTH])
            from app.agents.commander import _MAX_RESPONSE_LENGTH, truncate_for_signal
            resp_text = truncate_for_signal(result) if len(result) > _MAX_RESPONSE_LENGTH else result
            for chunk in _chunk_at_sentences(resp_text, MAX_SIGNAL_LENGTH):
                signal_client._send_sync(settings.signal_owner_number, f"[Dashboard] {chunk}")
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

    # Re-publish the schedule now that idle jobs are registered. The first
    # call at boot only saw APScheduler cron jobs (~14); this call adds the
    # ~95 idle-scheduler jobs so the dashboard shows the full picture.
    _publish_schedule()

    # Initialize versioned prompt registry — extracts souls/*.md on first boot
    try:
        from app.prompt_registry import init_registry
        await asyncio.to_thread(init_registry)
    except Exception:
        logger.warning("Prompt registry initialization failed (non-fatal)", exc_info=True)

    # Seed eval sets for evolution benchmarks (coder_v1, researcher_v1, writer_v1)
    try:
        from app.evolution_db.eval_sets import seed_default_eval_sets
        created = await asyncio.to_thread(seed_default_eval_sets)
        if created:
            logger.info(f"Seeded {created} evolution eval sets")
    except Exception:
        logger.debug("Eval set seeding failed (non-fatal)", exc_info=True)

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
        from app.healing.health_remediator import SelfHealer
        monitor = get_monitor()
        healer = SelfHealer()

        async def _on_health_alert(alerts: list) -> None:
            try:
                await healer.handle_alerts(alerts)
            except Exception:
                logger.debug("Self-healer alert handling failed", exc_info=True)

        def _sync_alert_handler(alerts: list) -> None:
            try:
                loop = asyncio.get_running_loop()
                asyncio.ensure_future(_on_health_alert(alerts))
            except RuntimeError:
                pass  # No running event loop — alert dropped (non-fatal)

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

    # ── MCP client (T1-1) — consume external MCP servers ────────────────
    if settings.mcp_client_enabled:
        try:
            from app.mcp.registry import connect_all as mcp_connect
            n = await asyncio.to_thread(mcp_connect)
            if n:
                logger.info(f"MCP client: connected to {n} external server(s)")
            else:
                logger.info("MCP client enabled but no servers configured")
        except Exception:
            logger.debug("MCP client startup failed (non-fatal)", exc_info=True)

    # ── Tool plugin registry (T1-2) — MCP, browser, session search ──────
    try:
        from app.crews.base_crew import _register_default_plugins
        _register_default_plugins()
        logger.info("Tool plugin registry initialized (MCP + browser + session search)")
    except Exception:
        logger.debug("Tool plugin registration failed (non-fatal)", exc_info=True)

    # ── Training adapter registry (T4-14) — hydrate promoted adapters ───
    # Without this, _get_promoted_adapter() in llm_factory sees an empty dict
    # on cold start, and MLX-adapter inference never fires until a training
    # cycle runs. Instantiating the orchestrator triggers registry load.
    try:
        from app.training_pipeline import get_orchestrator
        _ = get_orchestrator()
    except Exception:
        logger.debug("Training orchestrator init failed (non-fatal)", exc_info=True)

    # ── Phase 16a: SubIA consciousness wire-in ──────────────────────────
    # Opt-in via SUBIA_FEATURE_FLAG_LIVE=1. When disabled, the entire
    # SubIA stack stays unimported (no latency, no memory, no risk).
    # enable_subia_hooks() already wraps internal failures and returns a
    # structured state — it never raises. The outer try/except is
    # defence-in-depth in case an import itself fails.
    try:
        if settings.subia_live_enabled:
            from app.subia.live_integration import enable_subia_hooks
            subia_state = enable_subia_hooks(feature_flag=True)
            if subia_state.registered:
                logger.info(
                    "SubIA CIL hooks registered "
                    f"(kernel loop_count={getattr(subia_state.kernel, 'loop_count', 0)})"
                )
            else:
                logger.warning(f"SubIA wire-in incomplete: {subia_state.reason}")
        else:
            logger.info("SubIA live integration disabled (SUBIA_FEATURE_FLAG_LIVE=0)")
    except Exception:
        logger.warning("SubIA live integration failed (non-fatal)", exc_info=True)

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
    async def _report_phil() -> None:
        try:
            from app.firebase_reporter import report_philosophy_kb
            await asyncio.to_thread(report_philosophy_kb)
        except Exception:
            pass

    async def _gen_chronicle() -> None:
        try:
            from app.memory.system_chronicle import generate_and_save
            await asyncio.to_thread(generate_and_save)
            logger.info("System chronicle generated.")
        except Exception:
            logger.warning("System chronicle generation failed (non-fatal)", exc_info=True)

    async def _report_monitor() -> None:
        try:
            from app.firebase_reporter import report_system_monitor
            await asyncio.to_thread(report_system_monitor)
            logger.info("System monitor reported to dashboard")
        except Exception:
            logger.debug("System monitor report failed (non-fatal)", exc_info=True)

    await asyncio.gather(_report_phil(), _gen_chronicle(), _report_monitor())

    # ── LLM catalog early refresh ─────────────────────────────────────
    # The bootstrap CATALOG only holds 3 survival entries.  Without a
    # refresh, resolve_role_default() can only choose from those 3 — so
    # the selector's smart pick of Grok-4.20 / other frontier models
    # never reaches the runtime even though the snapshot on disk has
    # 347 entries.  Do a non-forced refresh here (respects the 24h TTL
    # so it just loads the cached snapshot if still fresh, or does one
    # network scan if not).  This runs BEFORE the first agent is built
    # so all role resolutions see the full catalog.
    try:
        from app.llm_catalog_builder import refresh as _refresh_catalog
        _cat_before = 3  # bootstrap size
        summary = await asyncio.to_thread(_refresh_catalog)
        _cat_after = summary.get("catalog_size", _cat_before)
        logger.info(
            f"LLM catalog loaded at startup: {_cat_before} → {_cat_after} entries "
            f"(sources: {summary.get('source_counts', {})})"
        )
    except Exception:
        logger.warning(
            "LLM catalog startup refresh failed — role defaults will fall "
            "back to bootstrap (sonnet/deepseek/qwen3)",
            exc_info=True,
        )

    # ── Inbound queue replay ─────────────────────────────────────────
    # If a previous container instance died or was restarted while
    # processing user messages, those messages are still in the
    # inbound_queue table with status 'queued' or 'processing'.  Pick
    # them up and dispatch them now.  Dedup guard in handle_task uses
    # (sender, signal_ts) so if the previous instance actually finished
    # (and just failed to mark the queue as 'done' before dying), the
    # message won't be sent twice because the assistant's reply is also
    # stored in conversations.db with the same sender+timestamp.
    try:
        pending = get_pending_inbound(max_attempts=3)
        if pending:
            logger.warning(
                f"Replaying {len(pending)} unfinished inbound messages from "
                f"previous container instance"
            )
            for row in pending:
                try:
                    asyncio.create_task(handle_task(
                        sender=row["sender"],
                        text=row["message"],
                        attachments=row["attachments"] or [],
                        msg_timestamp=row["signal_ts"],
                        queue_id=row["id"],
                    ))
                except Exception:
                    logger.exception("inbound queue replay failed for row %s", row.get("id"))
        # Prune entries older than 7 days regardless of status
        try:
            pruned = prune_old_inbound(days=7)
            if pruned:
                logger.info(f"Pruned {pruned} old inbound_queue rows (>7d)")
        except Exception:
            pass
    except Exception:
        logger.exception("Inbound queue replay failed at startup")

    # ── Outbound queue replay ─────────────────────────────────────────
    # Counterpart to the inbound replay: if a reply (especially an .md
    # attachment) was interrupted mid-send by a restart, redeliver it.
    # Rows that fail 3 replays stay in 'failed' with last_error for
    # inspection so a poison payload can't loop forever.
    try:
        from app.signal_client import replay_pending_outbound_sync
        from app.conversation_store import prune_old_outbound
        replayed = await asyncio.to_thread(replay_pending_outbound_sync)
        if replayed:
            logger.warning(f"Replayed {replayed} unfinished outbound sends from previous instance")
        try:
            pruned = prune_old_outbound(days=7)
            if pruned:
                logger.info(f"Pruned {pruned} old outbound_queue rows (>7d)")
        except Exception:
            pass
    except Exception:
        logger.exception("Outbound queue replay failed at startup")

    # ── Theory-of-Mind state hygiene ─────────────────────────────────
    # Purge phantom crews (test fixtures, typos, retired names) from
    # agent_state.json so get_best_crew_for_difficulty can't recommend
    # a crew that doesn't actually exist.  The real bug behind the
    # 2026-04-20 'Estonian deforestation research echoed the task
    # description' failure: agent_state.json had 'tom_test_research'
    # with a planted 5/0 record, ToM picked it, Commander dispatched
    # to that phantom crew, execution silently fell through.
    try:
        from app.subia.self.agent_state import prune_phantom_crews
        phantom = await asyncio.to_thread(prune_phantom_crews)
        if phantom:
            logger.warning(
                f"agent_state: pruned {phantom} phantom crew(s) from ToM state"
            )
    except Exception:
        logger.debug("Phantom crew prune failed at startup", exc_info=True)

    # ── Tool Registry: Forge bridge reconciliation loop (Phase 3) ─────
    # Picks up Forge-tool tier transitions (SHADOW → CANARY → ACTIVE,
    # demotions, KILLED removals) every 5 minutes. No-op when Forge is
    # disabled; non-fatal on individual iteration failures.
    try:
        from app.tool_registry.forge_bridge import reconciliation_loop
        asyncio.create_task(reconciliation_loop())
    except Exception as exc:
        logger.debug("Forge bridge reconciliation loop not started: %s", exc)

    # ── Discord connector (Phase 6 — May 2026) ───────────────────────
    # Opt-in via DISCORD_ENABLED + DISCORD_BOT_TOKEN. The bot runs as a
    # background asyncio task on the gateway's loop and forwards owner
    # DMs to handle_task with sender prefix `discord:<user_id>`.
    try:
        from app.discord_client import start_bot as _discord_start
        await _discord_start()
    except Exception:
        logger.exception("Discord bot startup failed (non-fatal)")

    logger.info("CrewAI Agent Team started")
    yield
    # Discord clean shutdown (closes the gateway WS before APScheduler dies).
    try:
        from app.discord_client import stop_bot as _discord_stop
        await _discord_stop()
    except Exception:
        logger.debug("Discord bot shutdown raised", exc_info=True)
    # ── Graceful shutdown: drain in-flight tasks before letting the
    # container die.  Without this, SIGTERM → immediate exit strands any
    # handle_task() threads that were in the middle of processing a
    # user message or sending a reply.  The durable inbound+outbound
    # queues protect against abrupt loss now, but giving active threads
    # up to 30s to finish cleanly is still better UX (user doesn't see
    # "duplicate reply from replay" for something that was about to
    # succeed anyway).  Configurable via GRACEFUL_SHUTDOWN_S.
    shutdown_budget = float(os.environ.get("GRACEFUL_SHUTDOWN_S", "30"))
    if shutdown_budget > 0:
        import time as _time
        deadline = _time.monotonic() + shutdown_budget
        drained_cleanly = False
        while True:
            with _inflight_lock:
                n = _inflight_tasks
            if n <= 0:
                drained_cleanly = True
                break
            remaining = deadline - _time.monotonic()
            if remaining <= 0:
                logger.warning(
                    f"Graceful shutdown: {n} task(s) still in-flight after "
                    f"{shutdown_budget:.0f}s — exiting anyway (queues will replay)"
                )
                break
            logger.info(f"Graceful shutdown: waiting for {n} task(s) to drain ({remaining:.0f}s left)")
            await asyncio.sleep(2)
        if drained_cleanly:
            logger.info("Graceful shutdown: all tasks drained cleanly")
    # Final sync on clean shutdown
    idle_scheduler.stop()
    idle_scheduler.stop_listener()
    if settings.workspace_backup_repo:
        await asyncio.to_thread(sync_workspace, settings.workspace_backup_repo)
    # Disconnect any MCP client sessions
    try:
        from app.mcp.registry import disconnect_all as mcp_disconnect
        mcp_disconnect()
    except Exception:
        pass
    report_system_offline()
    scheduler.shutdown()


def _publish_schedule() -> None:
    """Push current scheduler job list to Firestore.

    Merges APScheduler cron jobs with the idle-scheduler job registry so
    the dashboard sees both surfaces. Idle jobs do not have a discrete
    next_run (they fire whenever the gateway is idle), so we mark them
    with cron="idle:<weight>" and next_run=None.
    """
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
        try:
            for name, weight in idle_scheduler.list_jobs():
                jobs.append({
                    "id": f"idle:{name}",
                    "name": name,
                    "next_run": None,
                    "cron": f"idle:{weight}",
                })
        except Exception:
            logger.debug("Failed to enumerate idle jobs", exc_info=True)
        report_schedule(jobs)
    except Exception:
        logger.debug("Failed to publish schedule", exc_info=True)


app = FastAPI(title="CrewAI Agent Gateway", lifespan=lifespan)

# Expose Prometheus /metrics — auto-emits http_requests_total + duration
# histogram via prometheus-fastapi-instrumentator, and surfaces the
# application-level metrics defined in app.observability.metrics
# (llm_requests_total, llm_request_duration_seconds, etc.). Safe no-op
# when prometheus-fastapi-instrumentator isn't installed.
try:
    from app.observability.metrics import register_metrics
    register_metrics(app)
except Exception:
    logger.warning("metrics: registration failed", exc_info=True)

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

# Mount fiction inspiration API routes
from app.api.fiction import fiction_router
app.include_router(fiction_router)

# Mount new knowledge base API routes (Phase 2)
try:
    from app.episteme.api import episteme_router
    app.include_router(episteme_router)
except ImportError:
    pass
try:
    from app.experiential.api import experiential_router
    app.include_router(experiential_router)
except ImportError:
    pass
try:
    from app.aesthetics.api import aesthetics_router
    app.include_router(aesthetics_router)
except ImportError:
    pass
try:
    from app.tensions.api import tensions_router
    app.include_router(tensions_router)
except ImportError:
    pass

# ── Middleware (extracted to app/middleware.py) ────────────────────────────────
from app.middleware import add_middleware
add_middleware(app, settings)

signal_client = SignalClient()
from app.history_compression import CompressionMiddleware
commander = CompressionMiddleware(Commander())
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


@app.get("/self_improvement/health")
async def self_improvement_health():
    """Dashboard endpoint: Self-Improvement Overhaul observability.

    Returns a single snapshot with:
      - pipeline funnel (gaps → drafts → skills → usage)
      - topic diversity (Shannon entropy over KB topic clusters)
      - competence summary (total skills + open gaps + curiosity signal)
      - MAP-Elites per-role latency baselines

    Cheap enough for polling. All aggregations are read-only over existing
    stores; no recomputation of embeddings.
    """
    try:
        from app.self_improvement import health_summary
        return health_summary()
    except Exception as e:
        return {"error": str(e)}


@app.get("/self_improvement/describe")
async def self_improvement_describe():
    """First-person competence description for the chronicle / introspection."""
    try:
        from app.subia.self.competence_map import describe_self
        return {"description": describe_self()}
    except Exception as e:
        return {"description": "", "error": str(e)}


@app.get("/location")
async def get_location():
    """Current resolved location (for dashboard and debugging)."""
    try:
        from app.spatial_context import get_location as resolve_location
        from app.temporal_context import get_temporal_context
        loc = resolve_location()
        tc = get_temporal_context()
        return {
            "location": loc,
            "temporal": {
                "date": tc.get("date_str", ""),
                "time": tc.get("time_str", ""),
                "season": tc.get("season", ""),
                "daylight_hours": tc.get("daylight_hours", 0),
            },
        }
    except Exception as e:
        return {"error": str(e)}


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

        emoji = payload.get("emoji", "")
        target_ts = payload.get("target_timestamp", 0)
        is_remove = payload.get("is_remove", False)

        # ── Proposal approval via reaction ─────────────────────────────
        # 👍 on a proposal notification approves it; 👎 rejects it.
        # Check this BEFORE feedback pipeline so approval side-effects fire.
        # Removals (is_remove=True) are no-ops — we never want accidental
        # reversals from just un-reacting.
        if target_ts and not is_remove and emoji in ("👍", "👎", "+1", "-1"):
            try:
                from app.proposals import (
                    find_proposal_by_signal_timestamp,
                    approve_proposal, reject_proposal,
                )
                pid = find_proposal_by_signal_timestamp(target_ts)
                if pid is not None:
                    is_approve = emoji in ("👍", "+1")
                    action_fn = approve_proposal if is_approve else reject_proposal
                    action_name = "approved" if is_approve else "rejected"
                    # Run the action in a thread — approve_proposal is sync
                    # and may do I/O (file copy, auto-deploy trigger).
                    result = await asyncio.get_running_loop().run_in_executor(
                        None, action_fn, pid,
                    )
                    logger.info(f"Reaction {emoji} on proposal #{pid} → {action_name}: {result}")
                    # Acknowledge to the user so they see the action landed.
                    try:
                        client = SignalClient()
                        await client.send(
                            sender,
                            f"✅ Proposal #{pid} {action_name} via reaction.\n{result}",
                        )
                    except Exception:
                        logger.debug("Failed to send reaction-approval ack", exc_info=True)
                    return {"status": "accepted", "proposal_action": action_name, "pid": pid}
            except Exception:
                logger.debug("Reaction-based proposal handling failed", exc_info=True)
                # Fall through to human_gate / feedback pipeline below

        # ── Workspace-switch proposal approval via reaction ────────────
        # When auto-detection sees a likely workspace mismatch, it asks
        # the user via Signal (see app/workspace_switch_proposals.py).
        # 👍 confirms the switch (with source="user" so it's sticky);
        # 👎 records a decline that suppresses re-asking the same
        # detection for 24 h.
        if target_ts and not is_remove and emoji in ("👍", "👎", "+1", "-1"):
            try:
                from app.workspace_switch_proposals import (
                    find_by_signal_ts as _ws_find,
                    accept as _ws_accept,
                    decline as _ws_decline,
                )
                ws_proposal_id = _ws_find(target_ts)
                if ws_proposal_id is not None:
                    is_approve = emoji in ("👍", "+1")
                    fn = _ws_accept if is_approve else _ws_decline
                    result = await asyncio.get_running_loop().run_in_executor(
                        None, fn, ws_proposal_id,
                    )
                    action_name = "switch confirmed" if is_approve else "switch declined"
                    logger.info(
                        f"Reaction {emoji} on workspace proposal "
                        f"{ws_proposal_id} → {action_name}: {result}"
                    )
                    try:
                        client = SignalClient()
                        await client.send(sender, f"✅ {result}")
                    except Exception:
                        logger.debug(
                            "Failed to send workspace-proposal ack",
                            exc_info=True,
                        )
                    return {
                        "status": "accepted",
                        "workspace_proposal_action": action_name,
                        "proposal_id": ws_proposal_id,
                    }
            except Exception:
                logger.debug(
                    "Reaction-based workspace proposal handling failed",
                    exc_info=True,
                )
                # Fall through to human_gate / feedback pipeline below

        # ── Human-gate borderline-mutation approval via reaction ───────
        # Same pattern as proposals: 👍 routes to human_gate.approve_request,
        # 👎 to reject_request. Mirrors the message text "React 👍 to approve
        # or 👎 to reject" produced by human_gate._send_approval_notification.
        if target_ts and not is_remove and emoji in ("👍", "👎", "+1", "-1"):
            try:
                from app.human_gate import (
                    find_request_by_signal_timestamp,
                    approve_request, reject_request,
                )
                request_id = find_request_by_signal_timestamp(target_ts)
                if request_id is not None:
                    is_approve = emoji in ("👍", "+1")
                    if is_approve:
                        ok = await asyncio.get_running_loop().run_in_executor(
                            None, approve_request, request_id, sender,
                        )
                        action_name = "approved"
                    else:
                        ok = await asyncio.get_running_loop().run_in_executor(
                            None, reject_request, request_id, sender, "rejected via 👎",
                        )
                        action_name = "rejected"

                    logger.info(
                        f"Reaction {emoji} on human_gate {request_id} → {action_name} (ok={ok})"
                    )
                    try:
                        client = SignalClient()
                        await client.send(
                            sender,
                            f"✅ Mutation {action_name} via reaction.\nID: {request_id}",
                        )
                    except Exception:
                        logger.debug("Failed to send human_gate ack", exc_info=True)
                    return {
                        "status": "accepted",
                        "human_gate_action": action_name,
                        "request_id": request_id,
                    }
            except Exception:
                logger.debug("Reaction-based human_gate handling failed", exc_info=True)
                # Fall through to normal feedback pipeline below

        # ── Change-request approval via reaction (Phase 5.3) ───────────
        # 👍 on a CHANGE REQUEST message approves + applies (hot-write +
        # auto-PR); 👎 rejects. Mirrors the approval-via-reaction
        # pattern used for proposals + human_gate. The change-request
        # store correlates by signal_message_ts.
        if target_ts and not is_remove and emoji in ("👍", "👎", "+1", "-1"):
            try:
                from app.change_requests import (
                    find_request_by_signal_ts as _cr_find,
                    DecisionSource as _CR_DS,
                    Status as _CR_Status,
                    apply_change as _cr_apply,
                    approve as _cr_approve,
                    get as _cr_get,
                    reject as _cr_reject,
                )
                cr_id = _cr_find(target_ts)
                if cr_id is not None:
                    is_approve = emoji in ("👍", "+1")
                    cr_now = _cr_get(cr_id)
                    if cr_now is not None and cr_now.status == _CR_Status.PENDING:
                        loop = asyncio.get_running_loop()
                        if is_approve:
                            await loop.run_in_executor(
                                None,
                                lambda: _cr_approve(
                                    cr_id, source=_CR_DS.SIGNAL_THUMBS_UP,
                                ),
                            )
                            apply_result = await loop.run_in_executor(
                                None, _cr_apply, cr_id,
                            )
                            # Honest ack: pre-2026-05-09 we always wrote
                            # "✅ approved + applied" regardless of
                            # whether the apply actually succeeded —
                            # contradicting the "ok: False" / "ERROR: …"
                            # lines below. Now the headline reflects
                            # the outcome.
                            if apply_result.ok:
                                ack_msg = (
                                    f"✅ Change request {cr_id} approved + applied.\n"
                                    f"  branch: {apply_result.git_branch or '?'}\n"
                                    f"  PR: {apply_result.pr_url or '(failed to open)'}\n"
                                    f"  module reload: {apply_result.module_reload_note}"
                                )
                            else:
                                ack_msg = (
                                    f"⚠️ Change request {cr_id} approved, "
                                    f"but apply FAILED.\n"
                                    f"  ERROR: {apply_result.error}\n"
                                    f"  Status is now APPLY_FAILED — use the "
                                    f"'Retry apply' button in /cp/changes once "
                                    f"the underlying issue is resolved.\n"
                                    f"  branch: {apply_result.git_branch or '(none)'}\n"
                                    f"  PR: {apply_result.pr_url or '(not opened)'}"
                                )
                        else:
                            await loop.run_in_executor(
                                None,
                                lambda: _cr_reject(
                                    cr_id,
                                    source=_CR_DS.SIGNAL_THUMBS_DOWN,
                                    decision_reason="rejected via 👎 in Signal",
                                ),
                            )
                            ack_msg = f"❌ Change request {cr_id} rejected."

                        logger.info(
                            "Reaction %s on change_request %s → %s",
                            emoji, cr_id,
                            "approved+applied" if is_approve else "rejected",
                        )
                        try:
                            client = SignalClient()
                            await client.send(sender, ack_msg)
                        except Exception:
                            logger.debug(
                                "Failed to send change-request ack", exc_info=True,
                            )
                        return {
                            "status": "accepted",
                            "change_request_action": (
                                "approved+applied" if is_approve else "rejected"
                            ),
                            "request_id": cr_id,
                        }
            except Exception:
                logger.debug(
                    "Reaction-based change_request handling failed",
                    exc_info=True,
                )
                # Fall through

        # ── Architecture-request approval via reaction (Piece 2) ───────
        # 👍 on an ARCHITECTURE REQUEST message approves the design;
        # 👎 rejects. The lifecycle is package-granularity — approval
        # only moves PROPOSED → APPROVED. Scaffolding (writing stubs to
        # the staging dir) is a separate operator step in the React UI;
        # per-file landing flows through the existing change_request
        # gate. Mirrors the change_request reaction-handler pattern.
        if target_ts and not is_remove and emoji in ("👍", "👎", "+1", "-1"):
            try:
                from app.architecture_requests import (
                    ArchStatus as _AR_Status,
                    DecisionSource as _AR_DS,
                    approve as _ar_approve,
                    find_request_by_signal_ts as _ar_find,
                    get as _ar_get,
                    reject as _ar_reject,
                )
                ar_id = _ar_find(target_ts)
                if ar_id is not None:
                    is_approve = emoji in ("👍", "+1")
                    ar_now = _ar_get(ar_id)
                    if ar_now is not None and ar_now.status == _AR_Status.PROPOSED:
                        loop = asyncio.get_running_loop()
                        if is_approve:
                            await loop.run_in_executor(
                                None,
                                lambda: _ar_approve(
                                    ar_id, source=_AR_DS.SIGNAL_THUMBS_UP,
                                ),
                            )
                            ack_msg = (
                                f"✅ Architecture request {ar_id} approved.\n"
                                f"  package: {ar_now.package_path}\n"
                                f"  next: scaffold the stubs in /cp/architecture-requests, "
                                f"then per-file CRs land via /cp/changes."
                            )
                        else:
                            await loop.run_in_executor(
                                None,
                                lambda: _ar_reject(
                                    ar_id,
                                    source=_AR_DS.SIGNAL_THUMBS_DOWN,
                                    decision_reason="rejected via 👎 in Signal",
                                ),
                            )
                            ack_msg = f"❌ Architecture request {ar_id} rejected."

                        logger.info(
                            "Reaction %s on architecture_request %s → %s",
                            emoji, ar_id,
                            "approved" if is_approve else "rejected",
                        )
                        try:
                            client = SignalClient()
                            await client.send(sender, ack_msg)
                        except Exception:
                            logger.debug(
                                "Failed to send arch-request ack", exc_info=True,
                            )
                        return {
                            "status": "accepted",
                            "architecture_request_action": (
                                "approved" if is_approve else "rejected"
                            ),
                            "request_id": ar_id,
                        }
            except Exception:
                logger.debug(
                    "Reaction-based architecture_request handling failed",
                    exc_info=True,
                )
                # Fall through

        # ── Action-request approval via reaction (§5.5) ───────────────
        # 👍 on an ACTION REQUEST message approves + applies (sends the
        # email / creates the calendar event / etc); 👎 rejects.
        # APPLY_FAILED is recoverable: a second 👍 retries the apply.
        if target_ts and not is_remove and emoji in ("👍", "👎", "+1", "-1"):
            try:
                from app.action_requests import (
                    ActionStatus as _AC_Status,
                    DecisionSource as _AC_DS,
                    apply as _ac_apply,
                    approve as _ac_approve,
                    find_by_signal_ts as _ac_find,
                    get as _ac_get,
                    reject as _ac_reject,
                )
                ac_id = _ac_find(target_ts)
                if ac_id is not None:
                    is_approve = emoji in ("👍", "+1")
                    ac_now = _ac_get(ac_id)
                    if ac_now is not None and ac_now.status in (
                        _AC_Status.PENDING, _AC_Status.APPLY_FAILED,
                    ):
                        loop = asyncio.get_running_loop()
                        if is_approve:
                            await loop.run_in_executor(
                                None,
                                lambda: _ac_approve(
                                    ac_id, source=_AC_DS.SIGNAL_THUMBS_UP,
                                ),
                            )
                            applied = await loop.run_in_executor(
                                None, _ac_apply, ac_id,
                            )
                            if applied.status is _AC_Status.APPLIED:
                                ack_msg = (
                                    f"✅ Action {ac_id} applied "
                                    f"({applied.action_type.value}).\n"
                                    f"  artifact: {applied.apply_artifact}"
                                )
                            else:
                                ack_msg = (
                                    f"⚠️ Action {ac_id} approved, but apply "
                                    f"FAILED.\n"
                                    f"  ERROR: {applied.apply_error}\n"
                                    f"  Status now APPLY_FAILED — react 👍 "
                                    f"again to retry, or use "
                                    f"/cp/action-requests."
                                )
                        elif ac_now.status is _AC_Status.PENDING:
                            await loop.run_in_executor(
                                None,
                                lambda: _ac_reject(
                                    ac_id,
                                    source=_AC_DS.SIGNAL_THUMBS_DOWN,
                                    decision_reason="rejected via 👎 in Signal",
                                ),
                            )
                            ack_msg = f"❌ Action {ac_id} rejected."
                        else:
                            # 👎 on APPLY_FAILED is a no-op (action
                            # already didn't run); fall through.
                            ack_msg = None

                        if ack_msg is not None:
                            logger.info(
                                "Reaction %s on action_request %s → %s",
                                emoji, ac_id,
                                "applied/failed" if is_approve else "rejected",
                            )
                            try:
                                client = SignalClient()
                                await client.send(sender, ack_msg)
                            except Exception:
                                logger.debug(
                                    "Failed to send action-request ack",
                                    exc_info=True,
                                )
                            return {
                                "status": "accepted",
                                "action_request_action": (
                                    "applied/failed" if is_approve else "rejected"
                                ),
                                "request_id": ac_id,
                            }
            except Exception:
                logger.debug(
                    "Reaction-based action_request handling failed",
                    exc_info=True,
                )
                # Fall through

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
                        emoji,
                        target_ts,
                        is_remove,
                    )
            # Acknowledge with 👀 on the message the user reacted to
            if target_ts and not is_remove:
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

    # Persist the message to the durable inbound queue BEFORE returning
    # 200 OK.  If the container crashes or is restarted mid-processing,
    # replay_pending_inbound() on the next startup will re-dispatch any
    # unfinished queue entries so the user's message isn't silently lost.
    queue_id = enqueue_inbound(sender, text, timestamp, attachments)

    asyncio.create_task(handle_task(sender, text, attachments, timestamp, queue_id))
    return {"status": "accepted", "queue_id": queue_id}


# Lazy-initialized feedback pipeline singleton
_feedback_pipeline_instance = None

def _get_feedback_pipeline() -> "FeedbackPipeline | None":
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


async def _safe_react(sender: str, msg_timestamp: int) -> None:
    """Send 👀 reaction in background — never block the handler."""
    try:
        await signal_client.react(sender, "👀", sender, msg_timestamp)
    except Exception:
        logger.debug("Failed to send 👀 reaction", exc_info=True)


# ── Message idempotency guard ────────────────────────────────────────────────
class _MessageDedup:
    """Bounded LRU deduplication by sender+timestamp. Thread-safe."""
    def __init__(self, max_size: int = 500):
        from collections import OrderedDict
        self._seen: OrderedDict = OrderedDict()
        self._max = max_size
        self._lock = threading.Lock()

    def is_dup(self, key: str) -> bool:
        with self._lock:
            if key in self._seen:
                return True
            self._seen[key] = True
            if len(self._seen) > self._max:
                self._seen.popitem(last=False)
            return False

_msg_dedup = _MessageDedup()


# ── Watchdog thresholds (request-path stall detection) ─────────────────
#
# See ``_evaluate_stall`` below and ``docs/CONTROL_PLANES.md`` §"Watchdog
# (request-path timeouts)" for the full flow. Module-level so they can
# be patched in tests without monkey-patching the closure.

_SOFT_TIMEOUT_SECS = 900             # 15 min — soft checkpoint
_HARD_TIMEOUT_SECS = 2700            # 45 min — absolute ceiling
_STALL_THRESHOLD_SECS = 240          # 4 min w/o LLM return (loose)
_OUTPUT_STALL_THRESHOLD_SECS = 300   # 5 min w/o partial result (tight)
# Past this elapsed wall-clock with ZERO output-progress events ever
# recorded, we kill even if the LLM is still cycling. Backstop for the
# slow-but-cycling case (tools firing every minute, no deliverable yet).
_ZERO_OUTPUT_KILL_SECS = 1200        # 20 min elapsed + zero partials → kill
# Crew-zero-progress: tighter than zero-output, gated on the
# tool-activity heartbeat. Fires when the system has been "thinking"
# for a long time without any external action — the canonical
# "stuck-in-LLM-retry-loop" or "MCP-tool-hung" signature. Strictly
# tighter than the 1200s zero-output tier; leaves the 1200s backstop
# in place for the slow-cycling case.
_CREW_ZERO_PROGRESS_KILL_SECS = 600  # 10 min elapsed + zero partials + tool quiet
_CREW_TOOL_QUIET_SECS = 240          # tool entry/exit must be ≥ this stale (or never)
_PROGRESS_CHECK_EVERY = 30           # how often we re-check after soft


# Per-failure-kind apology templates (Cure C, 2026-05-10). When a
# stall fires AFTER a specific failure was recorded by vetting /
# artifact verification / completion guard, the watchdog message
# leads with the actual reason — the user gets actionable feedback
# instead of generic "narrow your question".
_FAILURE_CONTEXT_TEMPLATES: dict[str, str] = {
    "vetting_fail": (
        "\n\nLast detected issue: the response was rejected by quality "
        "review with the following reasons:\n  {detail}\n"
        "Try re-sending with the gaps explicitly addressed (e.g. "
        "missing data, incorrect identifiers, format mismatch)."
    ),
    "artifact_missing": (
        "\n\nLast detected issue: the task was classified as artifact-"
        "producing ({detail}) but no actual file was produced. The "
        "agent returned code that would, if run, generate the file — "
        "but the file itself never appeared. Try re-sending with an "
        "explicit instruction to execute the script and confirm the "
        "output path exists."
    ),
    "completion_truncated": (
        "\n\nLast detected issue: an LLM response was cut off mid-"
        "output by the token budget ({detail}). Re-sending with a "
        "narrower scope OR an explicit higher max-tokens hint usually "
        "recovers."
    ),
    "vetting_timeout": (
        "\n\nLast detected issue: the quality-review (vetting) layer "
        "exceeded its 90s budget ({detail}). The response was delivered "
        "unvetted, so the watchdog couldn't surface specific issues. "
        "Re-sending often helps; if the request involves heavy "
        "synthesis, narrowing the scope reduces vetting load."
    ),
    "exception": (
        "\n\nLast detected issue: an internal exception ({detail}) "
        "interrupted the task. Logs at /api/cp/ops or errors.jsonl "
        "have the full traceback."
    ),
}


def _format_failure_context_suffix(ctx: dict | None) -> str:
    """Render the failure-context dict as an apology suffix.

    ctx schema: ``{"kind": str, "detail": str, "age_s": float}`` —
    see :func:`app.observability.task_progress.get_failure_context`.

    Returns an empty string when ctx is None or unrecognized; never
    raises (the watchdog path is already handling an error and we
    don't want a bug here to swallow the underlying failure).
    """
    if not ctx:
        return ""
    try:
        kind = ctx.get("kind") or ""
        detail = ctx.get("detail") or ""
        # Truncate the detail to keep the apology bounded.
        detail_short = detail[:300] + ("…" if len(detail) > 300 else "")
        template = _FAILURE_CONTEXT_TEMPLATES.get(kind)
        if template is None:
            # Unknown kind — render generically so even a future
            # failure type surfaces SOMETHING actionable.
            return (
                f"\n\nLast detected issue ({kind}): {detail_short}"
            )
        return template.format(detail=detail_short)
    except Exception:
        return ""


def _evaluate_stall(task_id: str, elapsed_secs: float) -> tuple[str, float] | None:
    """Tiered stall check used by ``handle_task``.

    Pure function over the global heartbeat timestamps; module-level so
    tests can drive it without spinning up the full request path.

    Returns ``(kind, seconds)`` if the task should be killed, else None.
    Tiers are checked in order of strictness — first match wins:

      * **output-stall**       — a tool recorded a partial recently,
                                 then stopped. Strict 5-min threshold.
      * **crew-zero-progress** — > 10 min elapsed, never produced a
                                 partial, AND the tool-activity heartbeat
                                 is stale (no tool entry/exit for 4 min,
                                 or never). Catches retry-loops where the
                                 LLM keeps cycling but no work reaches
                                 tools and no output appears.
      * **zero-output**        — > 20 min elapsed, never produced a
                                 partial. Backstop for the slow-cycling
                                 case (tools firing, but no deliverable).
      * **llm-stall**          — LLM heartbeat stale for 4+ min. Loose
                                 fallback for hung threads.
    """
    from app.observability.task_progress import (
        output_progress_count,
        seconds_since_last_output_progress,
    )
    from app.rate_throttle import seconds_since_last_llm_activity
    from app.tools_timeout import (
        get_tool_timeout_count,
        seconds_since_last_tool_activity,
        seconds_since_last_tool_timeout,
    )

    out_stall = seconds_since_last_output_progress(task_id)

    if out_stall is not None:
        # Instrumented path: a tool recorded a partial at some point.
        # Ignore LLM-activity AND tool-activity — output progress is
        # the strictest signal and supersedes the looser tiers.
        if out_stall > _OUTPUT_STALL_THRESHOLD_SECS:
            return ("output-stall", out_stall)
        return None

    progress_count = output_progress_count(task_id)

    # Crew-zero-progress: tighter than zero-output. The tool-activity
    # heartbeat distinguishes "LLM is genuinely cycling through tools"
    # (heartbeat fresh — give it more time) from "LLM is stuck without
    # any external action" (heartbeat stale or never — kill earlier).
    # The 2026-05-02 production failure was the second case: 20 min of
    # Anthropic credit-exhausted retries kept the LLM-activity stamp
    # warm, but no tools fired and no partial appeared. This tier kills
    # that pattern at 10 min instead of 20.
    if elapsed_secs > _CREW_ZERO_PROGRESS_KILL_SECS and progress_count == 0:
        tool_idle = seconds_since_last_tool_activity()
        if tool_idle is None or tool_idle > _CREW_TOOL_QUIET_SECS:
            logger.info(
                "stall: crew-zero-progress %.1fs elapsed (tool_idle=%s "
                "task_id=%s)",
                elapsed_secs,
                f"{tool_idle:.1f}s" if tool_idle is not None else "never",
                task_id,
            )
            return ("crew-zero-progress", elapsed_secs)

    # Zero-output backstop: catches the slow-cycling case — tools are
    # firing (so crew-zero-progress stays inert), but no deliverable
    # has ever appeared. 20 min is "any legitimate task would have said
    # SOMETHING by now" and kills hallucination loops that ship nothing.
    if elapsed_secs > _ZERO_OUTPUT_KILL_SECS and progress_count == 0:
        return ("zero-output", elapsed_secs)

    # Loosest fallback: LLM heartbeat. Catches hung threads when the
    # task isn't output-instrumented.
    llm_stall = seconds_since_last_llm_activity()
    if llm_stall is not None and llm_stall > _STALL_THRESHOLD_SECS:
        # Phase E2: enrich the kill diagnostic with tool-activity state.
        # Helps distinguish "tool wedged mid-call" from "agent looped
        # through tools without progress".
        tool_idle = seconds_since_last_tool_activity()
        tool_timeout_age = seconds_since_last_tool_timeout()
        tool_timeouts = get_tool_timeout_count()
        logger.info(
            "stall: llm-stall %.1fs (tool_idle=%s tool_timeout_age=%s "
            "tool_timeouts=%d task_id=%s)",
            llm_stall,
            f"{tool_idle:.1f}s" if tool_idle is not None else "n/a",
            f"{tool_timeout_age:.1f}s" if tool_timeout_age is not None else "n/a",
            tool_timeouts,
            task_id,
        )
        return ("llm-stall", llm_stall)
    return None


async def handle_task(sender: str, text: str, attachments: list = None,
                      msg_timestamp: int = 0, queue_id: int | None = None):
    """Process a Signal message.  queue_id references the inbound_queue
    row created in receive_signal (or by replay_pending_inbound at
    startup); it's used to mark the durable queue as 'processing',
    'done', or 'failed' as we progress, so a restart mid-processing can
    find and replay the work.
    """
    global _inflight_tasks
    import time as _time
    if queue_id:
        mark_inbound_processing(queue_id)

    # Message idempotency: skip exact duplicate (same sender + Signal timestamp)
    if msg_timestamp:
        _dedup_key = f"{sender}:{msg_timestamp}"
        if _msg_dedup.is_dup(_dedup_key):
            logger.info(f"Duplicate message ignored: ts={msg_timestamp}")
            return

    _task_start = _time.monotonic()

    # Request tracing: generate correlation ID for this request lifecycle
    from app.trace import new_trace_id, get_trace_id
    trace_id = new_trace_id()

    # Start task tracking for metrics
    task_row_id = start_task(sender)

    # 👀 reaction is already sent in receive_signal() — no need to send again here

    # ── Load shedding: reject if at capacity ─────────────────────────────────
    _settings = get_settings()
    _shed_threshold = _settings.load_shed_threshold or (_settings.max_parallel_crews + 1)
    with _inflight_lock:
        currently_running = _inflight_tasks
        if currently_running >= _shed_threshold:
            logger.warning(f"Load shedding: rejecting request ({currently_running} inflight, threshold={_shed_threshold})")
            # Phase F3: instead of dropping, buffer to the in-process DLQ
            # so a follow-up idle scheduler pass can replay when capacity
            # returns. The user is still informed; the difference is the
            # message is no longer lost on rejection.
            from app.dead_letter_inbound import enqueue as _dlq_enqueue, queue_depth as _dlq_depth
            buffered = _dlq_enqueue(sender, text, attachments)
            try:
                if buffered:
                    await signal_client.send(
                        sender,
                        f"I'm currently handling {currently_running} tasks and at capacity. "
                        f"Your message has been queued (depth {_dlq_depth()}); "
                        f"I'll process it when capacity returns.",
                    )
                else:
                    await signal_client.send(
                        sender,
                        f"I'm currently handling {currently_running} tasks and the retry "
                        f"queue is also full. Please try again in a few minutes.",
                    )
            except Exception:
                pass
            return
        _inflight_tasks += 1

    # Track activity for idle scheduler (only if accepted)
    idle_scheduler.notify_task_start()

    # Notify user if other tasks are already in progress
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

        # ── Phase 15 grounding: ingress correction capture ─────────────
        # Detects user corrections ("actually it's X", "use source Y") and
        # synchronously persists them to the belief store + source
        # registry. No-ops when SUBIA_GROUNDING_ENABLED=0. Never raises.
        try:
            if settings.subia_grounding_enabled:
                from app.subia.connections.grounding_chat_bridge import (
                    observe_user_correction,
                )
                from app.conversation_store import get_last_assistant_message
                prior = get_last_assistant_message(sender)
                observe_user_correction(text, prior_response=prior)
        except Exception:
            logger.debug("Grounding ingress hook failed (non-fatal)",
                         exc_info=True)

        # ── Phase 17 introspection routing ──────────────────────────────
        # When the user asks AndrusAI about its own state ("what
        # frustrates you?", "are you tired?", "what's your mood?"),
        # inject a system-prompt prefix with the live homeostasis
        # snapshot so the LLM can ground its answer in actual data
        # instead of falling back to "I have no feelings". No-ops when
        # SUBIA_INTROSPECTION_ENABLED=0. Never raises — any failure
        # falls through with the original text.
        try:
            if settings.subia_introspection_enabled:
                from app.subia.connections.introspection_chat_bridge import (
                    inject_introspection,
                )
                augmented = inject_introspection(text)
                if augmented and augmented != text:
                    logger.info(
                        "Introspection: detected — augmented user message "
                        "with live self-state context (%d chars added)",
                        len(augmented) - len(text),
                    )
                    text = augmented
        except Exception:
            logger.debug("Introspection ingress hook failed (non-fatal)",
                         exc_info=True)

        # ── Project isolation: auto-detect venture from task text ───────
        # History compression is now handled by CompressionMiddleware in the
        # Commander wrapper (app.history_compression.CompressionMiddleware).
        try:
            from app.config import get_settings as _gs
            _s = _gs()
            if _s.project_isolation_enabled:
                # Two-mode workspace auto-detection (revised 2026-05-09):
                #
                #   1. Detected project differs from current →
                #      ASK via Signal (👍 to switch, 👎 to stay).
                #      Always propose, even when no explicit user pick
                #      exists yet — operators want consistent UX
                #      ("never silently switch").
                #   2. Detection matches current OR no detection →
                #      no action.
                #
                # History (one revision per failure mode):
                #
                #   Pre-2026-05-02 — every Signal message blew away the
                #   user's `switch workspace to eesti mets` whenever
                #   text contained "estonia" / "event" / "ticket" because
                #   those words triggered PLG's keyword list. Tickets
                #   ended up filed under PLG with no log explaining why.
                #   Fix: introduce sticky-user-pick + Mode 2 (ask) for
                #   keyword-detected mismatches.
                #
                #   2026-05-02 → 2026-05-09 — kept Mode 1 (auto-switch
                #   when no explicit pick) on the rationale "don't ask
                #   the user a question on every fresh session." But
                #   that meant the very first task in a session could
                #   silently land in the wrong workspace if the user
                #   hadn't typed `switch workspace` first. Operator
                #   explicit feedback: "always ask, never auto-switch."
                #   Fix: collapse to two modes — propose or no-op.
                from app.project_isolation import get_manager as _get_pm
                from app.control_plane.projects import get_projects as _gp
                _pm = _get_pm()
                detected = _pm.detect_project(text)
                if detected:
                    cp = _gp()
                    # Get the current workspace's display name for the ask
                    current_name = "default"
                    try:
                        cur_id = cp.get_active_project_id()
                        cur_row = cp.get_by_id(cur_id) if cur_id else None
                        if cur_row:
                            current_name = cur_row.get("name") or "default"
                    except Exception:
                        pass

                    if detected.lower() == current_name.lower():
                        pass  # already on it — no action
                    else:
                        # ALWAYS propose (don't silently switch, even
                        # when no explicit user pick exists yet).
                        try:
                            from app.workspace_switch_proposals import (
                                has_recent_decision, propose,
                            )
                            if not has_recent_decision(detected, sender):
                                propose(
                                    detected_name=detected,
                                    current_name=current_name,
                                    sender=sender,
                                )
                        except Exception:
                            logger.debug(
                                "workspace switch proposal failed",
                                exc_info=True,
                            )
        except Exception:
            logger.debug("Project detection failed", exc_info=True)

        # Mirror to dashboard chat (so dashboard users see Signal messages)
        report_chat_message("user", text + att_note, source="signal")

        loop = asyncio.get_running_loop()
        # ── Progressive timeout ─────────────────────────────────────────────
        #
        # Deep-research tasks at difficulty 9+ can legitimately take 15–30 min
        # (commander routing ~90s + crew_exec 6–7 min + vetting ~60s + optional
        # reflexion retry 6 min).  A single hard wall-clock timeout couldn't
        # distinguish "still making progress" from "stalled", so we used to
        # either kill healthy work early (900s) or let stuck threads burn
        # budget indefinitely.
        #
        # This progressive scheme keeps the task alive as long as it's
        # demonstrably doing LLM work:
        #
        #   1. Soft checkpoint @ 900s:
        #        If finished → return.  If not → notify the user that we're
        #        still working and enter progress-gated extension mode.
        #   2. Progress-gated extension (up to hard cap):
        #        Every 30s, check both (a) did the task finish? and (b) has
        #        ANY LLM call completed in the last `_STALL_THRESHOLD_SECS`?
        #        If no LLM activity for that long → stalled, give up cleanly.
        #        Otherwise keep waiting.
        #   3. Hard cap @ 2700s (45 min):
        #        Absolute ceiling.  Nothing legitimate in this system takes
        #        longer than this; past the cap the thread is abandoned and
        #        the user gets a "hit hard cap" message.
        #
        # Progress signal is a layered check (see ``_evaluate_stall``
        # for the full tier list):
        #
        #   (1) Output-progress (PREFERRED, tighter) — "the task produced
        #       a user-visible partial result in the last N seconds".
        #       Recorded by tools via
        #       :func:`app.observability.task_progress.record_output_progress`.
        #       This is strict: a retry loop that spews LLM calls but
        #       produces no deliverable does **not** advance it, so we
        #       kill the task much faster than the LLM-activity signal
        #       would.  Tasks using the ``research_orchestrator`` tool
        #       (or any tool that streams partials) get this.
        #
        #   (2) Tool-activity (mid-tier) — "any tool entered or exited
        #       in the last N seconds".  Distinguishes "LLM is genuinely
        #       cycling through tools" (heartbeat fresh) from "LLM is
        #       stuck without external action" (heartbeat stale or
        #       never).  Drives the ``crew-zero-progress`` tier — kills
        #       at 10 min when partials never appeared AND no tool
        #       ran recently.  See ``app/tools_timeout.py`` for the
        #       monkey-patch that ticks this on every BaseTool.run.
        #
        #   (3) LLM-activity fallback — "any LLM call returned
        #       (success or failure) in the last N seconds".  Used when
        #       the task isn't instrumented for partial output yet
        #       (backward compat: every existing tool still works, just
        #       with the looser stall threshold).  Cannot detect retry
        #       loops (failures keep the heartbeat warm) — tier (2) is
        #       the answer for that pattern.
        #
        # The context-var ``current_task_id`` is set here so tools can
        # record progress without having ``sender`` threaded through
        # every call signature.
        from app.observability.task_progress import current_task_id, reset_task

        _task_started_at = _time.monotonic()
        _ctx_token = current_task_id.set(str(sender or ""))
        _handle_fut = loop.run_in_executor(
            _commander_pool, commander.handle, text, sender, attachments or [],
        )

        def _stall_check() -> tuple[str, float] | None:
            """Delegate to the module-level :func:`_evaluate_stall`.

            See ``_evaluate_stall`` for the tier list and thresholds.
            """
            return _evaluate_stall(
                task_id=str(sender or ""),
                elapsed_secs=_time.monotonic() - _task_started_at,
            )

        try:
            # ── Phase 1: wait up to soft timeout ──
            try:
                result = await asyncio.wait_for(
                    asyncio.shield(_handle_fut),
                    timeout=_SOFT_TIMEOUT_SECS,
                )
            except asyncio.TimeoutError:
                # Not done in 15 min — is the crew still doing useful work?
                # Output-progress is preferred when available (strict); LLM
                # activity is the backstop (loose — catches hung threads).
                stalled = _stall_check()
                if stalled is not None:
                    kind, secs = stalled
                    raise asyncio.TimeoutError(
                        f"soft-timeout with {kind} (no progress for {secs:.0f}s)"
                    )

                # Alive — let the user know we're continuing.
                logger.info(
                    "handle_task: soft timeout (%ds) reached, task still"
                    " making progress — extending toward hard cap",
                    _SOFT_TIMEOUT_SECS,
                )
                try:
                    await signal_client.send(
                        sender,
                        "Still working on this — the request is deep enough that "
                        "it's taking longer than usual. I'll keep going as long as "
                        "there's progress, up to ~30 more minutes.",
                    )
                except Exception:
                    pass  # don't let a Signal blip abort the extension

                # ── Phase 2: progress-gated extension ──
                deadline_monotonic = _task_started_at + _HARD_TIMEOUT_SECS
                result = None  # pyright: ignore[reportGeneralTypeIssues]
                while True:
                    remaining_hard = deadline_monotonic - _time.monotonic()
                    if remaining_hard <= 0:
                        raise asyncio.TimeoutError("hard-cap reached")
                    try:
                        result = await asyncio.wait_for(
                            asyncio.shield(_handle_fut),
                            timeout=min(_PROGRESS_CHECK_EVERY, remaining_hard),
                        )
                        break  # task finished within the check window
                    except asyncio.TimeoutError:
                        # Check-window elapsed — still making progress?
                        # Same two-tier rule as Phase 1.
                        stalled = _stall_check()
                        if stalled is not None:
                            kind, secs = stalled
                            raise asyncio.TimeoutError(
                                f"extension stalled ({kind} for {secs:.0f}s)"
                            )
                        # Still cycling — wait another window.
                        continue
        except asyncio.TimeoutError as _to_exc:
            elapsed_min = (_time.monotonic() - _task_started_at) / 60.0
            reason = str(_to_exc) or "soft+extension timeout"
            logger.error(
                "TIMEOUT (%.1f min elapsed): handle_task for %s: %s (reason: %s)",
                elapsed_min, _redact_number(sender), text[:80], reason,
            )
            # Purge any cache entries that may have been stored mid-flight
            # for this task.  Run in a DAEMON THREAD with a 3s timeout so
            # a slow/unreachable Ollama embedder (the call path goes
            # ``invalidate_by_task`` → ChromaDB query → embed via Ollama)
            # can NEVER block the asyncio event loop or extend a
            # timeout-error handler.  The 2026-04-24 hypothesis: in-loop
            # ChromaDB calls during Ollama wobbles may have been part of
            # the restart cascade on tasks 76/77/78.
            try:
                import threading as _th
                _text_for_invalidate = text
                def _bg_invalidate():
                    try:
                        from app.result_cache import invalidate_by_task
                        invalidate_by_task(_text_for_invalidate)
                    except Exception:
                        pass  # daemon — dying is fine
                _th.Thread(
                    target=_bg_invalidate, daemon=True,
                    name="cache-invalidate-timeout",
                ).start()
            except Exception:
                logger.debug("result_cache.invalidate_by_task on TIMEOUT failed",
                             exc_info=True)
            # Cure C (2026-05-10): pull the last-known failure context
            # if any. When a task stalls AFTER hitting an explicit
            # failure (vetting reject / artifact missing / completion
            # truncated), the user gets the actual reason instead of
            # the watchdog's generic "narrow your question" advice.
            # Pre-fix the apology messages were misleading: a 30-min
            # stall caused by repeated vetting rejections looked
            # identical to a stall caused by no progress, even though
            # the actionable feedback is completely different.
            _failure_ctx = None
            try:
                from app.observability.task_progress import get_failure_context
                _failure_ctx = get_failure_context(str(sender or ""))
            except Exception:
                pass
            _ctx_suffix = _format_failure_context_suffix(_failure_ctx)

            if "crew-zero-progress" in reason.lower():
                # Stricter sibling of zero-output: tool-activity heartbeat
                # was stale alongside zero partials, so we killed at 10 min
                # instead of 20. The signature is "stuck thinking, not
                # acting" — provider retry-loop or a hung external call.
                result = (
                    "Sorry — your request ran for 10+ minutes without "
                    "producing any partial result and without any tool "
                    "activity in the last few minutes. This usually means "
                    "a stuck retry loop (provider outage or credit "
                    "exhausted) or a hung external call. Please try "
                    "again — if it recurs, narrow the scope or break the "
                    "request into smaller parts."
                ) + _ctx_suffix
            elif "zero-output" in reason.lower():
                result = (
                    "Sorry — your request ran for 20+ minutes without "
                    "producing any partial result. This usually means the "
                    "researcher agent is trying to answer from memory / "
                    "looping on a blocked source instead of streaming "
                    "findings. Please re-send with a more specific shape "
                    "— e.g. for a table of providers, say explicitly "
                    "'make a table of these N companies with these M "
                    "columns' so the researcher reaches for its structured "
                    "research tool."
                ) + _ctx_suffix
            elif "output-stall" in reason.lower():
                # If we have a specific failure context, the apology
                # leads with THAT (the user's actionable signal).
                # Otherwise fall back to the original generic message.
                if _failure_ctx:
                    result = (
                        f"Sorry — the task stopped after hitting the "
                        f"failure shown below. The retry path didn't "
                        f"recover before the output-stall threshold.\n"
                        f"{_ctx_suffix.lstrip()}"
                    )
                else:
                    result = (
                        "Sorry — the task stopped producing partial results (no "
                        "new rows / findings for several minutes).  I'll deliver "
                        "what's been streamed so far; please re-send a narrower "
                        "question to fill the gaps."
                    )
            elif "stall" in reason.lower():
                result = (
                    "Sorry — your request stalled (no LLM activity for several "
                    "minutes). This usually means a provider outage or a stuck "
                    "retry loop.  Please try again in a moment."
                ) + _ctx_suffix
            else:
                result = (
                    "Sorry — your request hit the absolute 45-minute ceiling. "
                    "I'll deliver what's been assembled where I can; otherwise "
                    "please break the question into smaller parts."
                ) + _ctx_suffix
        except Exception as _handle_exc:
            result = "Sorry, an error occurred processing your request. Please try again."
            logger.error(
                f"HANDLE_ERROR: {type(_handle_exc).__name__}: {_handle_exc} "
                f"for {_redact_number(sender)}: {text[:80]}",
                exc_info=True,
            )
            # Mirror the timeout path's cache purge — also moved to a
            # daemon thread so a slow Ollama embedder can't cascade into
            # blocking the error handler itself.
            try:
                import threading as _th
                _text_for_invalidate = text
                def _bg_invalidate():
                    try:
                        from app.result_cache import invalidate_by_task
                        invalidate_by_task(_text_for_invalidate)
                    except Exception:
                        pass
                _th.Thread(
                    target=_bg_invalidate, daemon=True,
                    name="cache-invalidate-error",
                ).start()
            except Exception:
                logger.debug("result_cache.invalidate_by_task on HANDLE_ERROR failed",
                             exc_info=True)
            # Safety net: if handle() raised before it finalized its ticket,
            # mark the ticket as failed so it doesn't hang in_progress
            # forever.  Commander stashes the ticket id on self before any
            # code that might raise.
            try:
                _stuck_tid = getattr(commander, "_last_ticket_id", None)
                _was_final = getattr(commander, "_last_ticket_finalized", True)
                if _stuck_tid and not _was_final:
                    from app.control_plane.tickets import get_tickets
                    get_tickets().fail(
                        _stuck_tid,
                        f"Uncaught {type(_handle_exc).__name__}: {str(_handle_exc)[:200]}",
                    )
                    logger.info(f"Safety-net: marked ticket {_stuck_tid} as failed")
            except Exception:
                logger.debug("Safety-net ticket fail failed", exc_info=True)

        # ── Phase 15 grounding: egress fact-checking ──────────────────
        # Inspects factual claims in the draft response against the
        # verified belief store. ALLOW keeps the draft. ESCALATE rewrites
        # to an honest "let me fetch from X" reply. BLOCK cites the
        # verified contradicting value. No-ops when grounding disabled.
        try:
            if settings.subia_grounding_enabled and result:
                from app.subia.connections.grounding_chat_bridge import (
                    ground_response,
                )
                result = ground_response(result, user_message=text) or result
        except Exception:
            logger.debug("Grounding egress hook failed (non-fatal)",
                         exc_info=True)

        log_response_sent(_redact_number(sender), len(result))

        # Record which crew handled the task (for per-crew analytics)
        try:
            from app.conversation_store import update_task_crew
            # Commander sets _last_crew in handle() — it's a singleton so the attr persists
            crew_used = getattr(commander, '_last_crew', '') or ''
            if crew_used and crew_used != 'direct':
                update_task_crew(task_row_id, crew_used)
        except Exception:
            pass

        # Persist the assistant reply (full text for conversation history)
        add_message(sender, "assistant", result)

        # Mirror to dashboard chat (so dashboard users see Signal responses)
        report_chat_message("assistant", result, source="signal")

        # History compression post-response is now handled by
        # CompressionMiddleware wrapping the Commander. No block here.

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
        # Both replies (truncated text + .md attachment) go through the
        # durable outbound queue so a gateway restart mid-send can't drop
        # them — the next startup replays any unsent rows.
        from app.agents.commander import _MAX_RESPONSE_LENGTH, truncate_for_signal
        from app.signal_client import send_durable

        if len(result) > _MAX_RESPONSE_LENGTH:
            signal_text = truncate_for_signal(result)
            # Write .md file synchronously now — the write is ~ms, and a
            # durable path is needed BEFORE we enqueue the attachment send.
            md_path = await asyncio.get_running_loop().run_in_executor(
                None, _write_response_md, result, text
            )
            # Reply #1 — truncated summary
            await send_durable(sender, signal_text, reply_to_id=queue_id)
            # Reply #2 — full .md attachment (if write succeeded)
            if md_path:
                try:
                    await send_durable(
                        sender, "", attachments=[md_path],
                        reply_to_id=queue_id,
                    )
                except Exception:
                    # send_durable already recorded the failure in the
                    # outbound_queue; startup replay will retry it.
                    logger.debug("Durable attachment send raised", exc_info=True)
        else:
            await send_durable(sender, result, reply_to_id=queue_id)
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

        # Dead letter queue: persist for retry after self-heal has time to fix
        try:
            from app.dead_letter import enqueue as dlq_enqueue
            dlq_enqueue(sender, text[:2000], type(exc).__name__, trace_id)
        except Exception:
            pass
        # Generic error — do not leak internals to Signal
        await signal_client.send(
            sender,
            "Something went wrong processing your request. "
            "The self-healing system is analyzing the error and will attempt a fix. "
            "Please try again shortly."
        )
        if queue_id:
            mark_inbound_failed(queue_id, f"{type(exc).__name__}: {exc}")
    else:
        # Only reached on fully successful completion of the try-block
        if queue_id:
            mark_inbound_done(queue_id)
    finally:
        with _inflight_lock:
            _inflight_tasks -= 1
        idle_scheduler.notify_task_end()
        # Drop output-progress counters for this task.  The context-var
        # token is reset implicitly when the asyncio task completes, but
        # the per-task dict entry is process-global and needs explicit
        # cleanup so crashed threads don't leak.
        try:
            reset_task(str(sender or ""))
        except Exception:
            pass


# ── API routers (extracted from main.py to app/api/) ──────────────────────────
from app.api.config_api import router as config_router
app.include_router(config_router, prefix="/config")

# ── Architecture-request control plane (Piece 2 follow-up) ──────────────────
try:
    from app.control_plane.architecture_requests_api import (
        router as architecture_requests_router,
    )
    app.include_router(architecture_requests_router)
except Exception:
    logger.debug(
        "Architecture-requests router registration failed", exc_info=True,
    )

# ── Inquiry essays (Piece 4 follow-up; read-only listing) ──────────────────
try:
    from app.control_plane.inquiries_api import router as inquiries_router
    app.include_router(inquiries_router)
except Exception:
    logger.debug("Inquiries router registration failed", exc_info=True)

# ── Action-request control plane (§5.5 follow-up) ──────────────────────────
try:
    from app.control_plane.action_requests_api import (
        router as action_requests_router,
    )
    app.include_router(action_requests_router)
except Exception:
    logger.debug("Action-requests router registration failed", exc_info=True)

# ── Long-horizon thread control plane (§4.1 follow-up) ─────────────────────
try:
    from app.control_plane.threads_api import router as threads_router
    app.include_router(threads_router)
except Exception:
    logger.debug("Threads router registration failed", exc_info=True)

# ── Browser-history ingestion (PROGRAM §50 — Q15) ─────────────────────────
try:
    from app.control_plane.browse_api import router as browse_router
    app.include_router(browse_router)
except Exception:
    logger.debug("Browse router registration failed", exc_info=True)

# ── Workflow_templates (PROGRAM §46.3, Q8.3) ───────────────────────────────
try:
    from app.control_plane.workflows_api import router as workflows_router
    app.include_router(workflows_router)
except Exception:
    logger.debug("Workflows router registration failed", exc_info=True)

# ── Long-term goal review (PROGRAM §46.9, Q9.6) ────────────────────────────
try:
    from app.control_plane.goals_api import router as goals_router
    app.include_router(goals_router)
except Exception:
    logger.debug("Goals router registration failed", exc_info=True)

# ── /api/cp/settings alias (PROGRAM §46.12) ─────────────────────────────────
# Latent-bug closure: every React settings card was calling
# /api/cp/settings, which had no server-side handler. Add a thin alias
# router that forwards to the canonical /config/runtime_settings
# setter so existing React code starts working and new cards (Travel,
# etc.) inherit the same path.
try:
    from app.control_plane.settings_alias_api import router as settings_alias_router
    app.include_router(settings_alias_router)
except Exception:
    logger.debug("Settings alias router registration failed", exc_info=True)

# ── Proposals aggregator (capability_gap + library_radar + recipe) ─────────
try:
    from app.control_plane.proposals_api import router as proposals_router
    app.include_router(proposals_router)
except Exception:
    logger.debug("Proposals router registration failed", exc_info=True)

# ── Skill registry (Phase 5 — May 2026) ──────────────────────────────────────
try:
    from app.api.skills_api import router as skills_router
    app.include_router(skills_router)
except Exception:
    logger.debug("Skills API router registration failed", exc_info=True)

# ── Files / artifacts (May 2026) — list + download + multi-channel send ──────
try:
    from app.api.files_api import router as files_router
    app.include_router(files_router)
except Exception:
    logger.debug("Files API router registration failed", exc_info=True)


# ── Knowledge Base router (extracted to app/api/kb.py) ────────────────────────
from app.api.kb import router as kb_router
app.include_router(kb_router, prefix="/kb")


# ── Notes viewer (Obsidian-style browser for markdown files) ─────────────────
try:
    from app.api.notes_api import router as notes_router
    app.include_router(notes_router)
except Exception:
    logger.debug("Notes API not available", exc_info=True)


# ── Health + Dashboard router (extracted to app/api/health.py) ─────────────────
from app.api.health import router as health_router
app.include_router(health_router)

# ── Affective layer router (Phase 1: /affect/now, /welfare-audit,
#     /reference-panel, /calibration, POST /override-reset) ─────────────
try:
    from app.affect.api import router as affect_router
    app.include_router(affect_router)
except Exception:
    logger.debug("Affect API router registration failed", exc_info=True)

# ── Epistemic Integrity router (/epistemic/now, /feed, /biases, /verifiers,
#     /claim/{id}, /pushback/*, /incidents/*) — see EPISTEMIC_INTEGRITY.md ─
try:
    from app.epistemic.api import router as epistemic_router
    app.include_router(epistemic_router)
    # Phase 5: wire affect ↔ epistemic bridge so the
    # register_confidence_mismatch detector sees the live grounding
    # signal and high-severity bias firings emit cognitive_failure
    # SalienceEvents into the narrative-self pipeline.
    from app.epistemic.affect_bridge import bootstrap as epistemic_affect_bootstrap
    epistemic_affect_bootstrap()
except Exception:
    logger.debug("Epistemic API router registration failed", exc_info=True)

# ── Workspace API (consciousness workspaces for React dashboard) ──────────────
try:
    from app.api.workspace_api import router as workspace_router
    app.include_router(workspace_router, prefix="/api")
except Exception:
    logger.debug("Workspace API not available", exc_info=True)

# ── Workspace Companion API (idle ideation, surfacing, wiki, etc.) ───────────
try:
    from app.api.companion_api import router as companion_router
    app.include_router(companion_router)
    logger.info("Workspace Companion API mounted at /api/cp/companion/")
except Exception:
    logger.debug("Workspace Companion API not available", exc_info=True)

# ── Vacation Mode API (PROGRAM §51 Q16 Theme 3) ─────────────────────────────
try:
    from app.api.vacation_api import router as vacation_router
    app.include_router(vacation_router)
    logger.info("Vacation Mode API mounted at /api/cp/vacation/")
except Exception:
    logger.debug("Vacation Mode API not available", exc_info=True)

# ── Brainstorm API (multi-agent + solo Q/A sessions) ─────────────────────────
try:
    from app.brainstorm.api import router as brainstorm_router
    app.include_router(brainstorm_router)
    logger.info("Brainstorm API mounted at /api/cp/brainstorm/")
except Exception:
    logger.debug("Brainstorm API not available", exc_info=True)

# ── Tool Registry (Phase 1a) ─────────────────────────────────────────────
# Boot the registry exactly once: imports every module under
# tool_registry.boot.TOOL_MODULE_ROOTS so all @register_tool decorators
# fire, then snapshots to Postgres + runs drift detection. Failures here
# are non-fatal — the gateway continues without the registry; existing
# agent factories don't depend on it (Phase 1a is purely additive).
try:
    from app.tool_registry.boot import boot_registry
    _registry = boot_registry(snapshot_to_postgres=True)
    logger.info(
        "Tool Registry booted: %d tools registered.",
        len(_registry.all()),
    )
except Exception:
    logger.warning("Tool Registry boot failed — continuing without it", exc_info=True)

# ── Control Plane API routes ─────────────────────────────────────────────
try:
    from app.control_plane.dashboard_api import router as cp_router
    app.include_router(cp_router)
    # Tool registry browser — read-only catalog view.
    from app.control_plane.tools_api import router as tools_cp_router
    app.include_router(tools_cp_router)
    # System state — git head + gateway uptime + recent crew runs (Phase 5.1).
    # Foundation for the routing fix (5.2) and change-request UI (5.3).
    from app.control_plane.system_state_api import router as system_state_cp_router
    app.include_router(system_state_cp_router)
    # Change requests — agent-proposed code modifications via human gate (Phase 5.3a).
    # GET / POST endpoints under /api/cp/changes — paired with the Signal
    # 👍/👎 voting flow + the React control plane UI (5.3b).
    from app.control_plane.changes_api import router as changes_cp_router
    app.include_router(changes_cp_router)
    # Coding sessions — read-only operator view of agent worktrees
    # (Phase 5.4-f). Lifecycle owned by agent + reconciler; operator
    # has visibility but no approve/reject — submission decisions
    # happen in the change-request UI, not here.
    from app.control_plane.coding_sessions_api import router as coding_sessions_cp_router
    app.include_router(coding_sessions_cp_router)
    logger.info("Control Plane API mounted at /api/cp/")
    # Ensure every project has default budget rows for the current period
    from app.control_plane.budgets import get_budget_enforcer as _get_be
    from app.control_plane.projects import get_projects as _get_proj
    _be = _get_be()
    for _proj in (_get_proj().list_all() or []):
        _be.ensure_default_budgets(str(_proj["id"]))
    logger.info("Control Plane: default budgets ensured for all projects")
except Exception:
    logger.debug("Control Plane API not available", exc_info=True)

# ── Evolution Monitoring API routes ─────────────────────────────────────────
try:
    from app.api.evolution_api import router as evolution_router
    app.include_router(evolution_router)
    logger.info("Evolution API mounted at /api/cp/evolution/")
except Exception:
    logger.debug("Evolution API not available", exc_info=True)

# ── Forge API (staged tool generation) ─────────────────────────────────────
# Default OFF: TOOL_FORGE_ENABLED env var must be true and runtime override
# must allow before any forged tool can run. Registry + audits work either
# way so the UI can show what would have been created.
try:
    from app.forge.api import router as forge_router
    app.include_router(forge_router)
    logger.info("Forge API mounted at /api/forge/")
except Exception:
    logger.debug("Forge API not available", exc_info=True)

# ── MCP Server (Model Context Protocol — P6) ──────────────────────────────
# Exposes philosophical RAG, MCSV, blackboard, personality, and Mem0
# as MCP resources + tools via SSE at /mcp/sse. Gracefully disabled if
# the mcp SDK is not installed.
try:
    from app.mcp.server import mount_mcp_routes
    if mount_mcp_routes(app):
        logger.info("MCP server mounted at /mcp/sse")
    else:
        logger.info("MCP server not available (mcp SDK not installed)")
except Exception:
    logger.debug("MCP server mount failed", exc_info=True)

# ── React Dashboard (self-hosted, replaces Firebase) ─────────────────────
# Mount LAST so API routes take precedence. Serves the React SPA build.
# Check Docker path first, then local dev path relative to this file.
from pathlib import Path as _Path
_dashboard_candidates = [
    _Path("/app/dashboard/build"),                          # Docker container
    _Path(__file__).resolve().parent.parent / "dashboard" / "build",  # local dev
]
_dashboard_build = next((p for p in _dashboard_candidates if (p / "index.html").exists()), None)
if _dashboard_build:
    from fastapi.staticfiles import StaticFiles
    app.mount("/cp", StaticFiles(directory=str(_dashboard_build), html=True), name="dashboard-cp")
    logger.info(f"React dashboard mounted at /cp/ from {_dashboard_build}")
