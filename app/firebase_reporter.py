"""
firebase_reporter.py — Real-time status reporting to Firestore.

Writes agent activity, task state, and system health to Firebase so the
monitoring dashboard can display live status without polling the server.

Firestore schema:
  status/system          — overall health, uptime, last-seen
  crews/{name}           — per-crew status, current task, eta
  tasks/{id}             — individual task records with ETA tracking
  activities/{id}        — rolling activity feed (last 50 entries)
  schedule/jobs          — upcoming cron-scheduled work

All writes are fire-and-forget (non-blocking) so a Firestore outage
never degrades agent performance.
"""

import logging
import os
import threading
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

_db = None
_db_lock = threading.Lock()
_PROJECT_ID = "botarmy-ba0c9"


def _get_db():
    """Lazy-initialise the Firestore client once per process."""
    global _db
    if _db is not None:
        return _db
    with _db_lock:
        if _db is not None:
            return _db
        try:
            import firebase_admin
            from firebase_admin import credentials, firestore

            # Prefer a service-account JSON file; fall back to Application Default Credentials
            sa_path = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON", "")
            if sa_path and os.path.exists(sa_path):
                cred = credentials.Certificate(sa_path)
            else:
                cred = credentials.ApplicationDefault()

            if not firebase_admin._apps:
                firebase_admin.initialize_app(cred, {"projectId": _PROJECT_ID})

            _db = firestore.client()
            logger.info("firebase_reporter: Firestore client initialised")
        except Exception:
            logger.warning("firebase_reporter: Firestore unavailable — monitoring disabled", exc_info=True)
            _db = False   # sentinel: skip all future attempts
    return _db


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fire(fn):
    """Run fn in a daemon thread so Firestore latency never blocks agents."""
    t = threading.Thread(target=fn, daemon=True)
    t.start()


# ── System status ─────────────────────────────────────────────────────────────

def report_system_online(version: str = "1.0") -> None:
    """Called once at startup to mark the system as online."""
    def _write():
        db = _get_db()
        if not db:
            return
        try:
            db.collection("status").document("system").set({
                "state": "online",
                "version": version,
                "started_at": _now_iso(),
                "last_seen": _now_iso(),
                "crews": {
                    "commander": "idle",
                    "research":  "idle",
                    "coding":    "idle",
                    "writing":   "idle",
                    "self_improvement": "idle",
                },
            })
        except Exception:
            logger.debug("firebase_reporter: system online write failed", exc_info=True)
    _fire(_write)


def report_system_offline() -> None:
    """Called on graceful shutdown."""
    def _write():
        db = _get_db()
        if not db:
            return
        try:
            db.collection("status").document("system").update({
                "state": "offline",
                "last_seen": _now_iso(),
            })
        except Exception:
            logger.debug("firebase_reporter: system offline write failed", exc_info=True)
    _fire(_write)


def heartbeat() -> None:
    """Update last_seen timestamp — call every 60 s from a background task."""
    def _write():
        db = _get_db()
        if not db:
            return
        try:
            db.collection("status").document("system").update({
                "last_seen": _now_iso(),
                "state": "online",
            })
        except Exception:
            logger.debug("firebase_reporter: heartbeat write failed", exc_info=True)
    _fire(_write)


# ── Crew / agent status ───────────────────────────────────────────────────────

def crew_started(crew: str, task_summary: str, eta_seconds: Optional[int] = None) -> str:
    """Mark a crew as active.  Returns a task_id for later updates."""
    task_id = uuid.uuid4().hex
    eta_iso = None
    if eta_seconds:
        eta_iso = (datetime.now(timezone.utc) + timedelta(seconds=eta_seconds)).isoformat()

    def _write():
        db = _get_db()
        if not db:
            return
        now = _now_iso()
        try:
            # Update per-crew status
            db.collection("crews").document(crew).set({
                "name": crew,
                "state": "active",
                "current_task": task_summary[:200],
                "task_id": task_id,
                "started_at": now,
                "eta": eta_iso,
                "last_updated": now,
            })
            # Write a task record
            db.collection("tasks").document(task_id).set({
                "id": task_id,
                "crew": crew,
                "summary": task_summary[:200],
                "state": "running",
                "started_at": now,
                "eta": eta_iso,
                "completed_at": None,
                "result_preview": None,
            })
            # Append to activity feed
            _add_activity(db, "task_started", crew, task_summary[:200], task_id)
        except Exception:
            logger.debug("firebase_reporter: crew_started write failed", exc_info=True)
    _fire(_write)
    return task_id


def crew_completed(crew: str, task_id: str, result_preview: str = "") -> None:
    """Mark a crew as idle and record task completion."""
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
                "state": "completed",
                "completed_at": now,
                "result_preview": result_preview[:300],
            })
            _add_activity(db, "task_completed", crew, result_preview[:200], task_id)
        except Exception:
            logger.debug("firebase_reporter: crew_completed write failed", exc_info=True)
    _fire(_write)


def crew_failed(crew: str, task_id: str, error: str = "") -> None:
    """Mark a task as failed."""
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
            logger.debug("firebase_reporter: crew_failed write failed", exc_info=True)
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
            logger.debug("firebase_reporter: update_eta write failed", exc_info=True)
    _fire(_write)


# ── Scheduled jobs ────────────────────────────────────────────────────────────

def report_schedule(jobs: list[dict]) -> None:
    """Publish upcoming scheduled jobs.

    Each job dict: {"id": str, "name": str, "next_run": ISO str, "cron": str}
    """
    def _write():
        db = _get_db()
        if not db:
            return
        try:
            db.collection("schedule").document("jobs").set({"jobs": jobs, "updated_at": _now_iso()})
        except Exception:
            logger.debug("firebase_reporter: schedule write failed", exc_info=True)
    _fire(_write)


# ── Activity feed helper ──────────────────────────────────────────────────────

def _add_activity(db, event_type: str, crew: str, detail: str, task_id: str = "") -> None:
    """Append an entry to the activities collection (fire-and-forget, called inside _write threads)."""
    try:
        act_id = uuid.uuid4().hex
        db.collection("activities").document(act_id).set({
            "id": act_id,
            "ts": _now_iso(),
            "event": event_type,
            "crew": crew,
            "detail": detail,
            "task_id": task_id,
        })
        # Prune oldest entries beyond 50 — best-effort, non-blocking
        _prune_activities(db)
    except Exception:
        pass


def _prune_activities(db) -> None:
    """Keep only the 50 most recent activity entries."""
    try:
        docs = list(
            db.collection("activities")
            .order_by("ts", direction="DESCENDING")
            .offset(50)
            .limit(50)
            .stream()
        )
        for doc in docs:
            doc.reference.delete()
    except Exception:
        pass
