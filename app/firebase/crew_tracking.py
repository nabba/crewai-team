"""
firebase.crew_tracking — Crew lifecycle events.

Tracks crew start/complete/fail, ETA updates, sub-agent progress,
task delegation, and stale task cleanup.
"""

import logging
import threading
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional

from app.firebase.infra import _get_db, _fire, _now_iso, _add_activity

logger = logging.getLogger(__name__)

# Track task start times so crew_completed can query llm_benchmarks for tokens
_task_start_times: dict[str, str] = {}
_task_start_lock = threading.Lock()


# ── Crew / agent status ───────────────────────────────────────────────────────

def crew_started(crew: str, task_summary: str, eta_seconds: Optional[int] = None,
                 parent_task_id: Optional[str] = None,
                 model: Optional[str] = None) -> str:
    """Mark a crew as active.  Returns a task_id for later updates.

    If parent_task_id is set, this task is a sub-agent spawned by a parent task.
    model: the LLM model name used for this task (e.g. "qwen3:30b-a3b").
    """
    task_id = uuid.uuid4().hex
    # Record start time for token attribution fallback in crew_completed
    with _task_start_lock:
        _task_start_times[task_id] = datetime.now(timezone.utc).isoformat()
    eta_iso = None
    if eta_seconds:
        eta_iso = (datetime.now(timezone.utc) + timedelta(seconds=eta_seconds)).isoformat()

    def _write():
        db = _get_db()
        if not db:
            return
        now = _now_iso()
        try:
            # Only update crew-level status for top-level tasks (not sub-agents)
            if not parent_task_id:
                db.collection("crews").document(crew).set({
                    "name": crew,
                    "state": "active",
                    "current_task": task_summary[:300],
                    "task_id": task_id,
                    "started_at": now,
                    "eta": eta_iso,
                    "model": model or "",
                    "last_updated": now,
                })
            # Write a task record
            db.collection("tasks").document(task_id).set({
                "id": task_id,
                "crew": crew,
                "summary": task_summary[:4000],
                "state": "running",
                "started_at": now,
                "eta": eta_iso,
                "completed_at": None,
                "result_preview": None,
                "parent_task_id": parent_task_id,
                "is_sub_agent": parent_task_id is not None,
                "model": model or "",
                "delegated_to": None,
                "delegated_from": None,
            })
            # Append to activity feed
            _add_activity(db, "task_started", crew, task_summary[:200], task_id)
        except Exception:
            logger.debug("firebase.crew_tracking: crew_started write failed", exc_info=True)
    _fire(_write)
    return task_id


def _get_tracker_data() -> tuple[int, str, float]:
    """Get token usage from the active request cost tracker (if any)."""
    try:
        from app.rate_throttle import get_active_tracker
        t = get_active_tracker()
        if t and t.call_count > 0:
            model_str = ", ".join(sorted(t.models_used)) if t.models_used else ""
            return t.total_tokens, model_str, t.total_cost_usd
    except Exception:
        pass
    return 0, "", 0.0


def _get_tokens_since(since_iso: str) -> tuple[int, str, float]:
    """Fallback: read tokens from llm_benchmarks recorded after since_iso.

    More reliable than the ContextVar tracker when CrewAI uses thread pools
    that don't inherit the calling thread's context variable.
    """
    try:
        from app.llm_benchmarks import get_tokens_since
        data = get_tokens_since(since_iso)
        return data["total_tokens"], data["models"], data["cost_usd"]
    except Exception:
        return 0, "", 0.0


def crew_completed(crew: str, task_id: str, result_preview: str = "",
                   tokens_used: int = 0, model: str = "",
                   cost_usd: float = 0.0) -> None:
    """Mark a crew as idle and record task completion with token/model data.

    If tokens_used/model are not provided, attempts to read from the active
    RequestCostTracker automatically.
    """
    # Auto-fill from tracker if not explicitly provided
    if tokens_used == 0:
        _t, _m, _c = _get_tracker_data()
        if _t > 0:
            tokens_used = _t
            if not model: model = _m
            if cost_usd == 0: cost_usd = _c

    # Fallback: query llm_benchmarks for tokens recorded during this task.
    # This is more reliable than the ContextVar tracker when CrewAI uses
    # a thread pool that doesn't inherit the calling context.
    if tokens_used == 0:
        with _task_start_lock:
            since = _task_start_times.pop(task_id, None)
        if since:
            _t, _m, _c = _get_tokens_since(since)
            if _t > 0:
                tokens_used = _t
                if not model: model = _m
                if cost_usd == 0: cost_usd = _c
    else:
        # Clean up start time even if we got data from tracker
        with _task_start_lock:
            _task_start_times.pop(task_id, None)

    # Clean model name: strip provider prefixes and deduplicate
    # e.g. "deepseek/deepseek-chat-v3, openrouter/deepseek/deepseek-chat" -> "deepseek-chat-v3"
    if model:
        parts: set[str] = set()
        for m in model.split(","):
            m = m.strip()
            if m:
                # strip provider prefix (e.g. "openrouter/deepseek/deepseek-chat" -> "deepseek-chat")
                clean = m.split("/")[-1]
                if clean and clean not in ("unknown",):
                    parts.add(clean)
        model = ", ".join(sorted(parts)) if parts else model

    def _write():
        db = _get_db()
        if not db:
            return
        now = _now_iso()
        try:
            db.collection("crews").document(crew).update({
                "state": "idle",
                "current_task": None,
                "task_id": None,
                "eta": None,
                "last_updated": now,
            })
            update_data = {
                "state": "completed",
                "completed_at": now,
                "result_preview": result_preview[:4000],
            }
            if tokens_used > 0:
                update_data["tokens_used"] = tokens_used
            if model:
                update_data["model"] = model
            if cost_usd > 0:
                update_data["cost_usd"] = round(cost_usd, 6)
            db.collection("tasks").document(task_id).update(update_data)
            _add_activity(db, "task_completed", crew, result_preview[:200], task_id)
        except Exception:
            logger.debug("firebase.crew_tracking: crew_completed write failed", exc_info=True)
    _fire(_write)


def crew_failed(crew: str, task_id: str, error: str = "") -> None:
    """Mark a task as failed."""
    # Clean up start time tracking on failure
    with _task_start_lock:
        _task_start_times.pop(task_id, None)
    def _write():
        db = _get_db()
        if not db:
            return
        now = _now_iso()
        try:
            db.collection("crews").document(crew).update({
                "state": "idle",
                "current_task": None,
                "task_id": None,
                "eta": None,
                "last_updated": now,
            })
            db.collection("tasks").document(task_id).update({
                "state": "failed",
                "completed_at": now,
                "error": error[:300],
            })
            _add_activity(db, "task_failed", crew, error[:200], task_id)
        except Exception:
            logger.debug("firebase.crew_tracking: crew_failed write failed", exc_info=True)
    _fire(_write)


def update_eta(crew: str, task_id: str, eta_seconds: int) -> None:
    """Revise the ETA estimate for a running task."""
    eta_iso = (datetime.now(timezone.utc) + timedelta(seconds=eta_seconds)).isoformat()

    def _write():
        db = _get_db()
        if not db:
            return
        try:
            db.collection("crews").document(crew).update({"eta": eta_iso})
            db.collection("tasks").document(task_id).update({"eta": eta_iso})
        except Exception:
            logger.debug("firebase.crew_tracking: update_eta write failed", exc_info=True)
    _fire(_write)


def task_delegated(task_id: str, from_crew: str, to_crew: str, reason: str = "") -> None:
    """Record that a task was delegated from one crew/agent to another."""
    def _write():
        db = _get_db()
        if not db:
            return
        now = _now_iso()
        try:
            db.collection("tasks").document(task_id).update({
                "delegated_to": to_crew,
                "delegated_from": from_crew,
                "delegation_reason": reason[:200],
                "delegation_ts": now,
            })
            _add_activity(db, "task_delegated", from_crew,
                          f"-> {to_crew}: {reason[:100]}", task_id)
        except Exception:
            logger.debug("firebase.crew_tracking: task_delegated write failed", exc_info=True)
    _fire(_write)


def update_sub_agent_progress(crew: str, parent_task_id: str,
                               completed: int, total: int) -> None:
    """Update the parent task with sub-agent completion progress."""
    def _write():
        db = _get_db()
        if not db:
            return
        try:
            db.collection("tasks").document(parent_task_id).update({
                "sub_agent_progress": f"{completed}/{total}",
                "last_updated": _now_iso(),
            })
        except Exception:
            logger.debug("firebase.crew_tracking: sub_agent_progress write failed", exc_info=True)
    _fire(_write)


def cleanup_stale_tasks() -> None:
    """
    On startup: mark any 'running' tasks as failed (they're zombies from a
    previous container that was restarted). Also reset all crews to idle.
    """
    def _cleanup():
        db = _get_db()
        if not db:
            return
        try:
            from google.cloud.firestore_v1.base_query import FieldFilter
            tasks = db.collection("tasks").where(
                filter=FieldFilter("state", "==", "running")
            ).get()
            now = _now_iso()
            cleaned = 0
            for t in tasks:
                t.reference.update({
                    "state": "failed",
                    "error": "Task was running when system restarted. Marked as failed.",
                    "completed_at": now,
                })
                cleaned += 1

            # Reset all crews to idle
            for crew in ["commander", "research", "coding", "writing", "self_improvement"]:
                db.collection("crews").document(crew).set({
                    "state": "idle",
                    "current_task": None,
                    "eta": None,
                    "started_at": None,
                }, merge=True)

            # Clear stale credit alerts from previous session — they'll be
            # re-raised if the problem persists on the first API call.
            db.collection("status").document("credit_alerts").set({
                "alerts": {},
                "updated_at": _now_iso(),
            })

            if cleaned:
                logger.info(f"firebase.crew_tracking: cleaned up {cleaned} stale running tasks")
        except Exception:
            logger.debug("firebase.crew_tracking: stale task cleanup failed", exc_info=True)
    _fire(_cleanup)


