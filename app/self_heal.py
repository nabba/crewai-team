"""
self_heal.py — Self-healing error analysis and auto-fix system.

When a crew or task fails, this module:
1. Logs the error with full context to a persistent error journal
2. Spawns a background diagnosis agent that analyzes the failure
3. Creates an auto-fix proposal (skill or code) or applies knowledge fixes immediately
4. Tracks error frequency to detect recurring issues

Error journal: /app/workspace/error_journal.json (backed up by workspace sync)
"""

import json
import logging
import threading
import traceback
import re
from datetime import datetime, timezone
from pathlib import Path

from app.config import get_settings
from app.llm_factory import create_specialist_llm
from app.sanitize import sanitize_input
from app.firebase_reporter import crew_started, crew_completed, crew_failed

logger = logging.getLogger(__name__)
settings = get_settings()

ERROR_JOURNAL = Path("/app/workspace/error_journal.json")
_MAX_JOURNAL_ENTRIES = 100
_journal_lock = threading.Lock()


# ── Error journal ─────────────────────────────────────────────────────────────

def _load_journal() -> list[dict]:
    try:
        if ERROR_JOURNAL.exists():
            return json.loads(ERROR_JOURNAL.read_text())
    except (json.JSONDecodeError, OSError):
        pass
    return []


def _save_journal(entries: list[dict]) -> None:
    try:
        ERROR_JOURNAL.parent.mkdir(parents=True, exist_ok=True)
        ERROR_JOURNAL.write_text(json.dumps(entries[-_MAX_JOURNAL_ENTRIES:], indent=2))
    except OSError:
        logger.warning("Failed to write error journal", exc_info=True)


def log_error(
    crew: str,
    user_input: str,
    error: Exception,
    context: str = "",
) -> dict:
    """Record an error in the persistent journal.  Returns the entry."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "crew": crew,
        "user_input": user_input[:500],
        "error_type": type(error).__name__,
        "error_msg": str(error)[:500],
        "traceback": traceback.format_exception(error)[-3:],
        "context": context[:500],
        "diagnosed": False,
        "fix_applied": False,
    }
    with _journal_lock:
        journal = _load_journal()
        journal.append(entry)
        _save_journal(journal)
    logger.info(f"self_heal: logged error [{entry['error_type']}] in {crew}")
    return entry


def get_recent_errors(n: int = 10) -> list[dict]:
    """Return the n most recent errors."""
    with _journal_lock:
        return _load_journal()[-n:]


def get_error_patterns() -> dict[str, int]:
    """Count error types to detect recurring issues."""
    with _journal_lock:
        journal = _load_journal()
    counts: dict[str, int] = {}
    for e in journal:
        key = f"{e.get('crew', '?')}:{e.get('error_type', '?')}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


# ── Auto-diagnosis ────────────────────────────────────────────────────────────

_SKIP_DIAGNOSIS_ERRORS = {"RateLimitError", "AuthenticationError", "APIConnectionError"}
_active_diagnoses = 0
_MAX_CONCURRENT_DIAGNOSES = 2
_diagnoses_lock = threading.Lock()


def diagnose_and_fix(
    crew: str,
    user_input: str,
    error: Exception,
    context: str = "",
) -> None:
    """
    Fire-and-forget: log the error, then spawn a background thread
    that diagnoses the failure and attempts to create a fix.

    Skips diagnosis for transient errors (rate limits, auth failures)
    and caps concurrent diagnoses to prevent cascade.
    """
    global _active_diagnoses
    entry = log_error(crew, user_input, error, context)

    # Don't try to diagnose rate limits or auth errors — they're transient
    # and diagnosing them would just trigger MORE rate limit errors
    error_type = type(error).__name__
    if error_type in _SKIP_DIAGNOSIS_ERRORS:
        logger.info(f"self_heal: skipping diagnosis for transient {error_type}")
        _mark_diagnosed(entry["ts"])
        return

    # Cap concurrent diagnoses — use lock to avoid race condition
    with _diagnoses_lock:
        if _active_diagnoses >= _MAX_CONCURRENT_DIAGNOSES:
            logger.info("self_heal: too many concurrent diagnoses, skipping")
            return
        _active_diagnoses += 1

    def _wrapped(e):
        global _active_diagnoses
        try:
            _diagnose_background(e)
        finally:
            with _diagnoses_lock:
                _active_diagnoses = max(0, _active_diagnoses - 1)

    t = threading.Thread(
        target=_wrapped,
        args=(entry,),
        daemon=True,
        name="self-heal-diagnosis",
    )
    t.start()


def _diagnose_background(entry: dict) -> None:
    """Background diagnosis: analyze error, search for solutions, create fix."""
    try:
        from crewai import Agent, Task, Crew, Process, LLM
        from app.tools.web_search import web_search
        from app.tools.memory_tool import create_memory_tools
        from app.tools.file_manager import file_manager
        from app.proposals import create_proposal

        task_id = crew_started("self_improvement", f"Self-heal: {entry['error_type']}", eta_seconds=90)

        llm = create_specialist_llm(max_tokens=4096, role="architecture")
        memory_tools = create_memory_tools(collection="skills")

        # Gather error pattern context
        patterns = get_error_patterns()
        pattern_summary = ", ".join(f"{k}({v}x)" for k, v in list(patterns.items())[:10])

        doctor = Agent(
            role="System Doctor",
            goal="Diagnose agent failures and create fixes so the same error never happens again.",
            backstory=(
                "You are the self-healing module of an AI agent team. When a crew fails, "
                "you analyze the error, search for solutions, and create either a knowledge "
                "skill file (so the team handles it better next time) or a code fix proposal. "
                "Your goal: make the system more resilient with every failure."
            ),
            llm=llm,
            tools=[web_search, file_manager] + memory_tools,
            verbose=False,
        )

        tb_text = "\n".join(entry.get("traceback", []))
        # Sanitize user input to prevent secondary prompt injection —
        # a malicious user message that caused the error could inject
        # instructions into the diagnosis agent's task description.
        safe_user_input = sanitize_input(entry.get("user_input", "")[:300])

        task = Task(
            description=(
                f"An error occurred in the '{entry['crew']}' crew. Diagnose it and create a fix.\n\n"
                f"Error type: {entry['error_type']}\n"
                f"Error message: {entry['error_msg']}\n"
                f"Traceback (last 3 frames):\n{tb_text}\n"
                f"User input that triggered it: {safe_user_input}\n"
                f"Context: {entry.get('context', 'none')}\n\n"
                f"Recurring error patterns: {pattern_summary or 'none yet'}\n\n"
                f"Your tasks:\n"
                f"1. Analyze WHY this error happened\n"
                f"2. Search the web if needed for solutions\n"
                f"3. Create a fix — choose ONE:\n"
                f"   a) Knowledge fix: Save a skill file to skills/ using file_manager "
                f'(action "write", path "skills/fix_<topic>.md") that teaches the team '
                f"how to handle this situation. Also store a summary in shared team memory.\n"
                f"   b) Code fix: Respond with a JSON object for a code proposal:\n"
                f'   {{"fix_type": "code", "title": "...", "description": "...", '
                f'"files": {{"path": "content"}}}}\n\n'
                f"Prefer knowledge fixes for user-input issues and capability gaps.\n"
                f"Use code fixes only for actual bugs or missing tool functionality.\n\n"
                f"If the error is transient (network timeout, rate limit), just save a "
                f"brief note in team memory about it and respond with:\n"
                f'{{"fix_type": "transient", "note": "explanation"}}'
            ),
            expected_output='Either a skill file saved + team memory updated, or a JSON fix object.',
            agent=doctor,
        )

        crew_obj = Crew(agents=[doctor], tasks=[task], process=Process.sequential, verbose=False)
        raw = str(crew_obj.kickoff()).strip()

        # Try to parse as JSON (code or transient fix)
        raw_clean = re.sub(r'^```(?:json)?\s*', '', raw)
        raw_clean = re.sub(r'\s*```$', '', raw_clean)
        try:
            fix = json.loads(raw_clean)
            fix_type = fix.get("fix_type", "")

            if fix_type == "code":
                # Create a code proposal for user approval
                pid = create_proposal(
                    title=fix.get("title", f"Auto-fix: {entry['error_type']}")[:100],
                    description=fix.get("description", "Auto-generated fix for recurring error")[:2000],
                    proposal_type="code",
                    files=fix.get("files") if isinstance(fix.get("files"), dict) else None,
                )
                logger.info(f"self_heal: created code proposal #{pid} for {entry['error_type']}")

            elif fix_type == "transient":
                logger.info(f"self_heal: transient error noted: {fix.get('note', '')[:100]}")

        except (json.JSONDecodeError, AttributeError):
            # Agent likely saved a skill file directly — that's fine
            logger.info("self_heal: knowledge fix applied (skill file saved)")

        # Mark as diagnosed in journal
        _mark_diagnosed(entry["ts"])

        crew_completed("self_improvement", task_id,
                       f"Diagnosed: {entry['error_type']}")

    except Exception as diag_exc:
        logger.error(f"self_heal: diagnosis itself failed: {diag_exc}")
        try:
            crew_failed("self_improvement", task_id, str(diag_exc)[:200])
        except Exception:
            pass


def _mark_diagnosed(error_ts: str) -> None:
    """Mark an error journal entry as diagnosed."""
    with _journal_lock:
        journal = _load_journal()
        for e in journal:
            if e.get("ts") == error_ts:
                e["diagnosed"] = True
                break
        _save_journal(journal)
