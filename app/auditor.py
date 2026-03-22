"""
auditor.py — Continuous code quality auditor and error resolution loop.

Two modes of operation:

1. CODE AUDIT (runs on cron, e.g. every 4 hours):
   - Reads all Python source files in /app/
   - Uses deepseek-r1 (architecture role) to analyze code quality
   - Identifies bugs, anti-patterns, security issues, dead code
   - Creates auto-apply patches for safe fixes, proposals for risky ones
   - Triggers auto-deploy for safe fixes

2. ERROR RESOLUTION LOOP (runs on cron, e.g. every 30 minutes):
   - Scans error journal for unresolved recurring errors
   - Groups errors by pattern (same error_type + crew)
   - For each unresolved pattern with >= 2 occurrences:
     a) Generates a targeted fix (code or skill)
     b) Applies the fix
     c) Monitors if the error recurs after the fix
     d) If it does: generates a deeper fix and loops again
     e) If it doesn't: marks the pattern as resolved
   - Loops up to MAX_FIX_ATTEMPTS per error pattern

Audit journal: /app/workspace/audit_journal.json
"""

import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

from crewai import Agent, Task, Crew, Process
from app.config import get_settings
from app.llm_factory import create_specialist_llm
from app.tools.file_manager import file_manager
from app.tools.memory_tool import create_memory_tools
from app.firebase_reporter import crew_started, crew_completed, crew_failed
from app.self_heal import get_error_patterns, get_recent_errors, _journal_lock, _load_journal, _save_journal

logger = logging.getLogger(__name__)
settings = get_settings()

AUDIT_JOURNAL = Path("/app/workspace/audit_journal.json")
APP_DIR = Path("/app/app")
MAX_FIX_ATTEMPTS = 3
_audit_lock = threading.Lock()


# ── Audit journal ────────────────────────────────────────────────────────────

def _load_audit_journal() -> list[dict]:
    try:
        if AUDIT_JOURNAL.exists():
            return json.loads(AUDIT_JOURNAL.read_text())
    except (json.JSONDecodeError, OSError):
        pass
    return []


def _save_audit_journal(entries: list[dict]) -> None:
    try:
        AUDIT_JOURNAL.parent.mkdir(parents=True, exist_ok=True)
        AUDIT_JOURNAL.write_text(json.dumps(entries[-200:], indent=2))
    except OSError:
        logger.warning("Failed to write audit journal", exc_info=True)


def _log_audit(event: str, detail: str, files_changed: list = None) -> None:
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "detail": detail[:500],
        "files_changed": files_changed or [],
    }
    journal = _load_audit_journal()
    journal.append(entry)
    _save_audit_journal(journal)


# ── Error resolution tracking ────────────────────────────────────────────────

ERROR_TRACKER = Path("/app/workspace/error_tracker.json")


def _load_tracker() -> dict:
    try:
        if ERROR_TRACKER.exists():
            return json.loads(ERROR_TRACKER.read_text())
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def _save_tracker(tracker: dict) -> None:
    try:
        ERROR_TRACKER.parent.mkdir(parents=True, exist_ok=True)
        ERROR_TRACKER.write_text(json.dumps(tracker, indent=2))
    except OSError:
        pass


# ── Code audit ───────────────────────────────────────────────────────────────

def _collect_source_files() -> dict[str, str]:
    """Read all Python source files in /app/app/."""
    sources = {}
    if not APP_DIR.exists():
        return sources
    for f in sorted(APP_DIR.rglob("*.py")):
        rel = str(f.relative_to(APP_DIR.parent))  # e.g. "app/main.py"
        try:
            content = f.read_text()
            if len(content) > 20000:
                content = content[:20000] + "\n# ... (truncated)"
            sources[rel] = content
        except OSError:
            continue
    return sources


def run_code_audit() -> str:
    """
    Full code audit cycle. Reads source, asks the architecture model
    to find issues, and auto-applies safe fixes.
    """
    with _audit_lock:
        return _run_code_audit_locked()


def _run_code_audit_locked() -> str:
    task_id = crew_started("self_improvement", "Code audit", eta_seconds=300)

    try:
        sources = _collect_source_files()
        if not sources:
            crew_completed("self_improvement", task_id, "No source files found")
            return "No source files found to audit."

        # Build a compact summary of the codebase for the auditor
        file_list = "\n".join(f"- {f} ({len(c)} chars)" for f, c in sources.items())

        # Pick up to 6 files to audit per cycle (rotate through the codebase)
        journal = _load_audit_journal()
        recently_audited = set()
        for e in journal[-20:]:
            if e.get("event") == "code_audit":
                recently_audited.update(e.get("files_changed", []))

        # Prioritize files not recently audited
        candidates = [f for f in sources if f not in recently_audited]
        if not candidates:
            candidates = list(sources.keys())
        audit_files = candidates[:6]

        # Build source content for the auditor
        source_block = ""
        for f in audit_files:
            source_block += f"\n\n--- FILE: {f} ---\n{sources[f]}\n"

        llm = create_specialist_llm(max_tokens=8192, role="architecture")
        memory_tools = create_memory_tools(collection="skills")

        auditor = Agent(
            role="Code Auditor",
            goal="Find bugs, security issues, and non-optimal code. Produce exact fixes.",
            backstory=(
                "You are a senior software auditor reviewing an AI agent system's Python code. "
                "You find real bugs, security vulnerabilities, performance issues, and anti-patterns. "
                "You produce EXACT file patches that can be applied directly. "
                "You do NOT flag style issues or minor formatting — only real problems. "
                "For each issue, provide the exact fix as a file_manager write operation."
            ),
            llm=llm,
            tools=[file_manager] + memory_tools,
            verbose=False,
        )

        task = Task(
            description=(
                f"Audit these Python source files for bugs, security issues, and non-optimal code.\n\n"
                f"Files in the codebase:\n{file_list}\n\n"
                f"Source code to audit:\n{source_block[:30000]}\n\n"
                f"For each real issue found:\n"
                f"1. Describe the bug/issue (one line)\n"
                f"2. Explain the impact\n"
                f"3. Write the fixed file using file_manager tool "
                f'(action "write", path relative to workspace like "applied_code/app/filename.py")\n\n'
                f"IMPORTANT RULES:\n"
                f"- Only flag REAL bugs that cause incorrect behavior or security risk\n"
                f"- Do NOT flag style issues, missing docstrings, or minor formatting\n"
                f"- Do NOT rewrite code that works correctly\n"
                f"- If code is clean, respond with: {{\"issues\": 0, \"summary\": \"No issues found\"}}\n"
                f"- If you find issues, respond with:\n"
                f'  {{"issues": N, "summary": "what you fixed", '
                f'"fixes": [{{"file": "path", "description": "what was wrong"}}]}}\n'
            ),
            expected_output="JSON summary of issues found and fixes applied.",
            agent=auditor,
        )

        crew = Crew(agents=[auditor], tasks=[task], process=Process.sequential, verbose=False)
        raw = str(crew.kickoff()).strip()

        # Parse result
        from app.utils import safe_json_parse
        result, err = safe_json_parse(raw)
        if result is not None:
            n_issues = result.get("issues", 0)
            summary = result.get("summary", "Audit complete")
        else:
            n_issues = 0
            summary = raw[:200]

        _log_audit("code_audit", f"{n_issues} issues in {len(audit_files)} files: {summary}", audit_files)

        if n_issues > 0:
            # Create a proposal for user approval — NEVER auto-deploy LLM code
            from app.proposals import create_proposal
            create_proposal(
                title=f"Auditor: {n_issues} code issues",
                description=summary[:2000],
                proposal_type="code",
            )

        crew_completed("self_improvement", task_id, f"Audit: {n_issues} issues")
        return f"Code audit complete: {n_issues} issues found in {len(audit_files)} files. {summary}"

    except Exception as exc:
        crew_failed("self_improvement", task_id, str(exc)[:200])
        logger.error(f"Code audit failed: {exc}")
        return f"Code audit failed: {str(exc)[:200]}"


# ── Error resolution loop ────────────────────────────────────────────────────

def run_error_resolution() -> str:
    """
    Scan for recurring unresolved errors. For each pattern, generate a fix,
    apply it, and track whether it resolves the error. Loop until resolved
    or max attempts reached.
    """
    with _audit_lock:
        return _run_error_resolution_locked()


def _run_error_resolution_locked() -> str:
    patterns = get_error_patterns()
    if not patterns:
        return "No error patterns to resolve."

    tracker = _load_tracker()
    errors = get_recent_errors(50)
    resolved_count = 0
    attempted_count = 0

    for pattern_key, count in patterns.items():
        if count < 2:
            continue  # Only target recurring errors

        # Check tracker status
        track = tracker.get(pattern_key, {
            "attempts": 0,
            "resolved": False,
            "last_attempt": None,
            "fixes_applied": [],
        })

        if track["resolved"]:
            resolved_count += 1
            continue

        if track["attempts"] >= MAX_FIX_ATTEMPTS:
            continue  # Give up after max attempts

        # H3: Check if last fix worked — require 24h cooldown with no recurrence.
        # Previously any absence of errors was treated as "resolved", which
        # incorrectly credited transient error cessation to the diagnosis agent.
        if track["last_attempt"]:
            new_errors_since = [
                e for e in errors
                if f"{e.get('crew', '?')}:{e.get('error_type', '?')}" == pattern_key
                and e.get("ts", "") > track["last_attempt"]
            ]
            if not new_errors_since:
                # No new errors — but require 24h cooldown before declaring resolved
                try:
                    last_attempt_time = datetime.fromisoformat(track["last_attempt"])
                    hours_since = (datetime.now(timezone.utc) - last_attempt_time).total_seconds() / 3600
                    if hours_since >= 24:
                        track["resolved"] = True
                        tracker[pattern_key] = track
                        _save_tracker(tracker)
                        _log_audit("error_resolved", f"Pattern {pattern_key} resolved after {track['attempts']} attempts (24h clear)")
                        resolved_count += 1
                        logger.info(f"auditor: error pattern {pattern_key} RESOLVED after {track['attempts']} attempts")
                    else:
                        logger.debug(f"auditor: {pattern_key} — no new errors but only {hours_since:.0f}h since last fix (need 24h)")
                except (ValueError, TypeError):
                    pass
                continue

        # Need to attempt a fix
        attempted_count += 1
        logger.info(f"auditor: attempting fix #{track['attempts']+1} for {pattern_key}")

        fix_result = _attempt_error_fix(pattern_key, errors, track)

        track["attempts"] += 1
        track["last_attempt"] = datetime.now(timezone.utc).isoformat()
        if fix_result:
            track["fixes_applied"].append(fix_result)
        tracker[pattern_key] = track
        _save_tracker(tracker)

        # Only fix one pattern per cycle to avoid resource overload
        break

    summary = f"Error resolution: {resolved_count} resolved, {attempted_count} attempted, {len(patterns)} total patterns"
    if attempted_count > 0:
        _log_audit("error_resolution", summary)
    return summary


def _attempt_error_fix(pattern_key: str, errors: list, track: dict) -> str | None:
    """Generate a targeted fix for an error pattern.

    H5/H7: Uses direct LLM call instead of CrewAI Crew.
    H4: Includes actual source code from traceback.
    """
    task_id = crew_started("self_improvement", f"Fix: {pattern_key}", eta_seconds=60)

    try:
        matching = [
            e for e in errors
            if f"{e.get('crew', '?')}:{e.get('error_type', '?')}" == pattern_key
        ]
        if not matching:
            crew_completed("self_improvement", task_id, "No matching errors")
            return None

        latest = matching[-1]
        previous_fixes = "\n".join(f"- Attempt {i+1}: {f}" for i, f in enumerate(track.get("fixes_applied", [])))
        tb_text = "\n".join(latest.get("traceback", []))

        # H4: Read actual source code from traceback
        from app.self_heal import _read_source_from_traceback
        source_context = _read_source_from_traceback(latest)

        # H5/H7: Direct LLM call — no CrewAI overhead
        llm = create_specialist_llm(max_tokens=2048, role="architecture")

        prompt = (
            f"Fix this recurring error (attempt #{track['attempts']+1}/{MAX_FIX_ATTEMPTS}):\n\n"
            f"Error: {pattern_key} ({len(matching)} occurrences)\n"
            f"Message: {latest['error_msg']}\n"
            f"Traceback:\n{tb_text}\n"
            f"Crew: {latest['crew']}\n"
        )
        if source_context:
            prompt += f"\nSource code:\n{source_context[:6000]}\n"
        if previous_fixes:
            prompt += f"\nPrevious FAILED attempts:\n{previous_fixes}\n"
        prompt += (
            f"\nRespond with ONLY JSON:\n"
            f'{{"fix": "description of the root cause and exact code change needed", '
            f'"fixable": true|false}}\n\n'
            f"If you cannot determine the fix from the traceback, set fixable=false.\n"
            f"Make the MINIMUM change needed. Do NOT refactor unrelated code."
        )

        raw = str(llm.call(prompt)).strip()

        from app.utils import safe_json_parse
        result, err = safe_json_parse(raw)
        if result and isinstance(result, dict):
            fix_desc = result.get("fix", "unknown")
            fixable = result.get("fixable", False)

            if fixable and fix_desc and fix_desc != "unable to fix":
                from app.proposals import create_proposal
                pid = create_proposal(
                    title=f"Fix: {pattern_key}"[:100],
                    description=fix_desc[:2000],
                    proposal_type="code",
                )
                if pid > 0:
                    _log_audit("error_fix_proposed",
                               f"Pattern {pattern_key} attempt #{track['attempts']+1}: {fix_desc}",
                               [])
                    crew_completed("self_improvement", task_id, f"Fix proposed: {fix_desc[:100]}")
                    return fix_desc[:200]

        crew_completed("self_improvement", task_id, "Fix attempted — not fixable from traceback")
        return None

    except Exception as exc:
        try:
            crew_failed("self_improvement", task_id, str(exc)[:200])
        except Exception:
            pass
        logger.error(f"auditor: error fix attempt failed: {exc}")
        return None


# ── Status reporting ─────────────────────────────────────────────────────────

def get_audit_summary(n: int = 10) -> str:
    """Return recent audit activity."""
    journal = _load_audit_journal()[-n:]
    if not journal:
        return "No audit activity yet."
    lines = []
    for e in journal:
        lines.append(f"[{e['ts'][:16]}] {e['event']}: {e['detail'][:80]}")
    return "\n".join(lines)


def get_error_resolution_status() -> str:
    """Return error pattern tracking status."""
    tracker = _load_tracker()
    if not tracker:
        return "No error patterns tracked yet."
    lines = ["Error Resolution Status:\n"]
    for pattern, track in sorted(tracker.items()):
        status = "RESOLVED" if track["resolved"] else f"attempt {track['attempts']}/{MAX_FIX_ATTEMPTS}"
        lines.append(f"  {pattern}: {status}")
    return "\n".join(lines)
