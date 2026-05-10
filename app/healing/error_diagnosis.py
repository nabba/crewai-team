"""
error_diagnosis.py — Per-exception error analysis and auto-fix system.

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
    from app.utils import load_json_file
    return load_json_file(ERROR_JOURNAL, default=[])


def _save_journal(entries: list[dict]) -> None:
    from app.utils import save_json_file
    save_json_file(ERROR_JOURNAL, entries, max_entries=_MAX_JOURNAL_ENTRIES)


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

# H2/H6: Track skill files created per error pattern to prevent spam.
# Key = "crew:error_type", value = count of skill files already created.
_skill_files_per_pattern: dict[str, int] = {}
_MAX_SKILLS_PER_PATTERN = 2  # max skill files per unique error pattern


def diagnose_and_fix(
    crew: str,
    user_input: str,
    error: Exception,
    context: str = "",
    task_id: str = "",
) -> None:
    """
    Fire-and-forget: log the error, then spawn a background thread
    that diagnoses the failure and attempts to create a fix.

    Skips diagnosis for transient errors (rate limits, auth failures)
    and caps concurrent diagnoses to prevent cascade.
    """
    global _active_diagnoses
    entry = log_error(crew, user_input, error, context)
    entry["task_id"] = task_id  # link back to the failed task for dashboard tracing

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


def _update_task_healing(entry: dict, action: str, detail: str, proposal_id: int = 0) -> None:
    """Write healing outcome back to the failed task's Firestore doc for dashboard tracing."""
    try:
        from app.firebase_reporter import _get_db, _fire, _now_iso
        failed_task_id = entry.get("task_id", "")
        if not failed_task_id:
            return

        # Control Plane mirror — authoritative for the dashboard.
        try:
            from app.control_plane.crew_tasks import mark_healed
            mark_healed(task_id=failed_task_id, heal_detail=f"{action}: {detail}")
        except Exception:
            pass

        def _write():
            db = _get_db()
            if not db:
                return
            try:
                db.collection("tasks").document(failed_task_id).update({
                    "heal_action": action,
                    "heal_detail": detail[:300],
                    "heal_proposal_id": proposal_id,
                    "heal_at": _now_iso(),
                })
            except Exception:
                pass
        _fire(_write)
    except Exception:
        pass


def _read_source_from_traceback(entry: dict) -> str:
    """H4: Extract source file content referenced in traceback frames.

    Reads the actual source code so the diagnosis can see what went wrong.
    """
    source_blocks = []
    for frame in entry.get("traceback", []):
        # Extract file path from traceback frame: '  File "/app/app/foo.py", line 42, in bar'
        match = re.search(r'File "(/app/app/[^"]+)"', frame)
        if not match:
            continue
        filepath = Path(match.group(1))
        if not filepath.exists() or filepath.suffix != ".py":
            continue
        try:
            content = filepath.read_text()
            if len(content) > 8000:
                content = content[:8000] + "\n# ... (truncated)"
            rel = str(filepath).replace("/app/", "")
            source_blocks.append(f"--- {rel} ---\n{content}")
        except OSError:
            continue
    return "\n\n".join(source_blocks[:3])  # max 3 files


def _diagnose_background(entry: dict) -> None:
    """Background diagnosis: analyze error and create fix.

    H5/H7: Uses direct LLM call instead of full CrewAI Crew (saves 3-5 LLM
    round-trips). H4: Includes actual source code from traceback.
    H2/H6: Caps skill file creation per error pattern.
    """
    task_id = None
    try:
        from app.proposals import create_proposal

        task_id = crew_started("self_improvement", f"Self-heal: {entry['error_type']}", eta_seconds=30)

        # H4: Read actual source code referenced in traceback
        source_context = _read_source_from_traceback(entry)

        # Gather error pattern context
        patterns = get_error_patterns()
        pattern_key = f"{entry['crew']}:{entry['error_type']}"
        pattern_summary = ", ".join(f"{k}({v}x)" for k, v in list(patterns.items())[:10])

        # H2/H6: Check if we've already created enough skill files for this pattern
        existing_skills = _skill_files_per_pattern.get(pattern_key, 0)
        skill_creation_allowed = existing_skills < _MAX_SKILLS_PER_PATTERN

        tb_text = "\n".join(entry.get("traceback", []))
        safe_user_input = sanitize_input(entry.get("user_input", "")[:300])

        # Healing knowledge lookup — reuse proven fixes instead of LLM diagnosis
        try:
            from app.healing_knowledge import get_best_known_fix
            known_fix = get_best_known_fix(
                f"{entry['error_type']}: {entry['error_msg']}", entry.get("crew", "")
            )
            if known_fix:
                logger.info(
                    f"self_heal: reusing known fix (applied {known_fix.times_applied}x) — "
                    f"skipping LLM diagnosis"
                )
                fix = {
                    "diagnosis": f"Known fix (applied {known_fix.times_applied}x previously)",
                    "fix_type": known_fix.fix_type,
                    "title": f"Known fix: {known_fix.fix_applied[:60]}",
                    "description": known_fix.fix_applied,
                }
                # Jump to the fix application logic (skip LLM call)
                # The existing code below handles fix_type routing
                if fix and isinstance(fix, dict):
                    fix_type = fix.get("fix_type", "")
                    if fix_type == "transient":
                        if task_id:
                            crew_completed(task_id, "Transient error — self-healed (known fix)")
                        return
                    # For code/skill fixes, fall through to the existing handler below
        except Exception:
            pass  # Graceful: if healing KB fails, proceed with LLM diagnosis

        # H5/H7: Direct LLM call — no CrewAI overhead (saves 60-90s)
        llm = create_specialist_llm(max_tokens=2048, role="architecture")

        prompt = (
            f"You are diagnosing an error in an AI agent system.\n\n"
            f"Error type: {entry['error_type']}\n"
            f"Error message: {entry['error_msg']}\n"
            f"Traceback:\n{tb_text}\n"
            f"Crew: {entry['crew']}\n"
            f"User input: {safe_user_input}\n"
            f"Context: {entry.get('context', 'none')}\n"
            f"Recurring patterns: {pattern_summary or 'none'}\n\n"
        )

        if source_context:
            prompt += f"Source code from traceback:\n{source_context[:6000]}\n\n"

        prompt += (
            f"Respond with ONLY a JSON object:\n"
            f'{{"diagnosis": "root cause in 1-2 sentences", '
            f'"fix_type": "code"|"transient"|"skill", '
            f'"title": "short title", '
            f'"description": "what to change and why"}}\n\n'
            f"Use fix_type=\"transient\" for network/rate/timeout errors.\n"
            f"Use fix_type=\"code\" for actual bugs (describe the exact fix).\n"
            f"Use fix_type=\"skill\" ONLY for capability gaps (NOT for code bugs).\n"
        )

        if not skill_creation_allowed:
            prompt += f"\nNOTE: {existing_skills} skill files already exist for this error pattern. Do NOT suggest fix_type=\"skill\".\n"

        raw = str(llm.call(prompt)).strip()

        # Parse JSON response
        from app.utils import safe_json_parse
        fix, err = safe_json_parse(raw)

        if fix and isinstance(fix, dict):
            fix_type = fix.get("fix_type", "")
            diagnosis = fix.get("diagnosis", "")
            title = fix.get("title", f"Fix: {entry['error_type']}")[:100]
            description = fix.get("description", diagnosis)[:2000]

            if fix_type == "code" and description:
                # Q2 §39: try the structured-diagnosis path FIRST — if
                # the LLM produces a (path, new_content) fix above the
                # auto-tuned confidence threshold, file it as a CR
                # through the standard operator gate. This bypasses
                # the prose-only proposal path that drove the May 2026
                # "0 resolved, 1 attempted" gap.
                if _try_structured_path(entry, diagnosis):
                    _mark_diagnosed(entry["ts"])
                    return

                pid = create_proposal(
                    title=title,
                    description=f"Diagnosis: {diagnosis}\n\nFix: {description}",
                    proposal_type="code",
                )
                if pid > 0:
                    logger.info(f"self_heal: created code proposal #{pid} for {entry['error_type']}")
                    # Link diagnosis back to the failed task for dashboard tracing
                    _update_task_healing(entry, "proposal_created", f"Proposal #{pid}: {title}", pid)
                    # Notify user via Signal about the proposal — attach the
                    # proposal.md so the user sees rationale/changes/risks.
                    # Capture the Signal timestamp so 👍/👎 reactions to this
                    # notification can be mapped back to the proposal.
                    try:
                        from app.signal_client import send_message_blocking
                        from app.config import get_settings
                        from app.proposals import (
                            get_proposal_md_path, set_proposal_signal_timestamp,
                        )
                        s = get_settings()
                        md_path = get_proposal_md_path(pid)
                        attachments = [str(md_path)] if md_path else None
                        signal_ts = send_message_blocking(
                            s.signal_owner_number,
                            f"🔧 SELF-HEAL PROPOSAL #{pid}: {title}\n"
                            f"Error: {entry['error_type']}\n"
                            f"Full writeup attached (rationale, changes, risks).\n"
                            f"React 👍 to approve or 👎 to reject, or reply "
                            f"'approve {pid}' / 'diff {pid}'.",
                            attachments=attachments,
                        )
                        if signal_ts and pid > 0:
                            set_proposal_signal_timestamp(pid, signal_ts)
                    except Exception:
                        pass

            elif fix_type == "skill" and skill_creation_allowed:
                # Store a concise skill note in team memory (not a file)
                try:
                    from app.memory.chromadb_manager import store_team
                    store_team(
                        f"ERROR FIX [{pattern_key}]: {diagnosis}. Resolution: {description}",
                        {"type": "error_fix", "pattern": pattern_key},
                    )
                    _skill_files_per_pattern[pattern_key] = existing_skills + 1
                    logger.info(f"self_heal: stored knowledge fix for {pattern_key}")
                    _update_task_healing(entry, "knowledge_fix", f"Learned: {diagnosis[:100]}")
                except Exception:
                    pass

            elif fix_type == "transient":
                logger.info(f"self_heal: transient error: {diagnosis[:100]}")
                _update_task_healing(entry, "transient", f"Transient: {diagnosis[:100]}")

        else:
            # F5: Do NOT mark as diagnosed if we couldn't parse the response.
            # This allows the error to be re-diagnosed on next cycle.
            logger.warning(f"self_heal: couldn't parse diagnosis: {err}")
            crew_completed("self_improvement", task_id, f"Diagnosis parse failed: {entry['error_type']}")
            return  # <-- early return: don't mark diagnosed

        # Store causal belief in world model for future context
        if fix and isinstance(fix, dict) and fix.get("diagnosis"):
            try:
                from app.subia.belief.world_model import store_causal_belief
                store_causal_belief(
                    cause=f"{entry.get('crew', 'unknown')}:{entry.get('error_type', 'unknown')}",
                    effect=fix.get("diagnosis", "")[:200],
                    confidence="high" if fix.get("fix_type") == "code" else "medium",
                    source="self_heal",
                )
            except Exception:
                pass

        _mark_diagnosed(entry["ts"])
        crew_completed("self_improvement", task_id, f"Diagnosed: {entry['error_type']}")

    except Exception as diag_exc:
        logger.error(f"self_heal: diagnosis failed: {diag_exc}")
        if task_id:
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


# ── Structured-diagnosis path (Q2 §39) ─────────────────────────────────


def _try_structured_path(entry: dict, diagnosis: str) -> bool:
    """Attempt the structured-diagnosis path. Returns True iff a CR
    was filed (caller skips the prose fallback). Returns False on
    any failure or decline (caller falls back to prose).

    Failures inside structured_diagnosis already emit telemetry
    (declined events with reasons). We only emit the ``filed`` event
    here, after a successful CR creation.
    """
    try:
        from app.healing.structured_diagnosis import (
            generate_structured_fix, current_threshold,
        )
    except Exception:
        logger.debug("structured_diagnosis unavailable", exc_info=True)
        return False

    file_path, file_content = _pick_target_file(entry)
    if not file_path or not file_content:
        return False

    fix = generate_structured_fix(
        error_message=entry.get("error_message") or entry.get("error_type") or "",
        error_traceback="\n".join(entry.get("traceback") or []),
        file_path=file_path,
        file_content=file_content,
        pattern_signature=entry.get("pattern_signature") or entry.get("error_type") or "",
        error_class=entry.get("error_type") or "",
    )
    if fix is None or not fix.is_actionable:
        # Declined / unavailable / below threshold — prose fallback.
        return False

    try:
        from app.change_requests import create_request, send_ask, Status
    except Exception:
        logger.debug(
            "structured_diagnosis: change_requests unavailable", exc_info=True,
        )
        return False

    reason = (
        f"Auto-diagnosis (structured): {fix.reasoning[:300]}\n\n"
        f"Originating error pattern: "
        f"{entry.get('pattern_signature') or entry.get('error_type') or '?'}\n"
        f"LLM self-assessed confidence: {fix.confidence:.2f}\n"
        f"Active threshold at decision: {current_threshold():.2f}\n\n"
        f"Operator review: the diff below shows EXACTLY what changed. "
        f"Approve via Signal 👍 / `/cp/changes` to hot-apply + open PR."
    )
    try:
        cr = create_request(
            requestor="error_diagnosis",
            path=fix.path,
            new_content=fix.new_content,
            old_content=fix.old_content,
            reason=reason,
        )
    except Exception:
        logger.warning(
            "structured_diagnosis: create_request raised", exc_info=True,
        )
        return False

    if cr.status != Status.PENDING:
        # Validator-side rejection (TIER_IMMUTABLE / blocked path /
        # outside roots / etc). The CR system already audited the
        # refusal — we just fall back to prose.
        logger.info(
            "structured_diagnosis: CR not pending — status=%s for %s",
            cr.status.value, fix.path,
        )
        return False

    # Filed-eligible: emit telemetry + Signal ASK.
    try:
        from app.healing.diagnosis_telemetry import record_filed
        added, removed = _delta_for_telemetry(fix.old_content, fix.new_content)
        record_filed(
            cr_id=cr.id,
            pattern_signature=entry.get("pattern_signature") or "",
            file_path=fix.path,
            error_class=entry.get("error_type") or "",
            confidence=fix.confidence,
            threshold=current_threshold(),
            delta_added=added,
            delta_removed=removed,
        )
    except Exception:
        logger.debug("structured_diagnosis: telemetry filed-emit failed",
                     exc_info=True)

    try:
        send_ask(cr.id)
    except Exception:
        logger.debug("structured_diagnosis: send_ask failed", exc_info=True)

    logger.info(
        "structured_diagnosis: filed CR %s for %s (confidence=%.2f)",
        cr.id, fix.path, fix.confidence,
    )
    return True


def _pick_target_file(entry: dict) -> tuple[str, str]:
    """Pick the most-likely-broken file from the traceback. Reads the
    file content and returns (repo_relative_path, content). Returns
    ("", "") when no usable file is found."""
    for frame in entry.get("traceback") or []:
        match = re.search(r'File "(/app/app/[^"]+)"', frame)
        if not match:
            continue
        absolute = Path(match.group(1))
        if not absolute.exists() or absolute.suffix != ".py":
            continue
        try:
            content = absolute.read_text()
        except OSError:
            continue
        # Repo-relative path the change-request validator expects.
        rel = str(absolute).removeprefix("/app/")
        return rel, content
    return "", ""


def _delta_for_telemetry(old_content: str, new_content: str) -> tuple[int, int]:
    """Same shape as ``app.change_requests.validator._net_line_delta``."""
    try:
        import difflib
        diff = list(difflib.unified_diff(
            (old_content or "").splitlines(),
            (new_content or "").splitlines(),
            lineterm="",
        ))
        added = sum(
            1 for line in diff
            if line.startswith("+") and not line.startswith("+++")
        )
        removed = sum(
            1 for line in diff
            if line.startswith("-") and not line.startswith("---")
        )
        return added, removed
    except Exception:
        return 0, 0
