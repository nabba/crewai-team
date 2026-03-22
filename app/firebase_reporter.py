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
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_db = None
_db_lock = threading.Lock()
_PROJECT_ID = "botarmy-ba0c9"

# Bounded thread pool prevents unbounded thread accumulation when Firestore is slow.
# 4 workers is enough for fire-and-forget writes; excess tasks queue instead of spawning threads.
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="firebase")


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
    """Run fn in the bounded thread pool so Firestore latency never blocks agents.

    Uses a ThreadPoolExecutor(max_workers=4) instead of spawning unbounded
    daemon threads — prevents thread accumulation when Firestore is slow.
    """
    try:
        _executor.submit(fn)
    except RuntimeError:
        # Pool is shut down (e.g., during interpreter teardown)
        pass


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

            if cleaned:
                logger.info(f"firebase_reporter: cleaned up {cleaned} stale running tasks")
        except Exception:
            logger.debug("firebase_reporter: stale task cleanup failed", exc_info=True)
    _fire(_cleanup)


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
    """Update last_seen timestamp + all status data — call every 60 s."""
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

        # Push fleet status + benchmarks
        try:
            from app.ollama_native import get_fleet_status
            fleet = get_fleet_status()
            benchmarks = []
            try:
                from app.llm_benchmarks import get_scores
                for task_type in ["coding", "architecture", "research", "writing"]:
                    scores = get_scores(task_type)
                    for model, score in scores.items():
                        benchmarks.append({"model": model, "task": task_type, "score": round(score, 2)})
            except Exception:
                pass
            report_fleet_status(fleet, benchmarks)
        except Exception:
            pass

        # Push token usage stats
        try:
            report_token_stats()
        except Exception:
            pass

        # Push composite metrics
        try:
            report_metrics()
        except Exception:
            pass

        # Push circuit breaker states
        try:
            report_circuit_breakers()
        except Exception:
            pass

        # Push error journal summary
        try:
            report_errors()
        except Exception:
            pass

        # Push evolution/experiment data
        try:
            report_evolution()
        except Exception:
            pass

        # Push request cost stats
        try:
            report_request_costs()
        except Exception:
            pass

        # Push model catalog (only once or on changes, but cheap enough per heartbeat)
        try:
            report_catalog()
        except Exception:
            pass

        # Push knowledge base stats
        try:
            report_knowledge_base()
        except Exception:
            pass

        # L5: Push ecological awareness stats
        try:
            report_ecological_stats()
        except Exception:
            pass

        # Push learned skills inventory
        try:
            report_skills()
        except Exception:
            pass
    _fire(_write)


# ── Skills inventory ─────────────────────────────────────────────────────────

_SKILLS_DIR = Path("/app/workspace/skills")


def report_skills() -> None:
    """Push learned skills inventory to Firestore at status/skills."""
    db = _get_db()
    if not db:
        return
    try:
        skills = []
        if _SKILLS_DIR.exists():
            for f in sorted(_SKILLS_DIR.glob("*.md")):
                if f.name == "learning_queue.md":
                    continue
                name = f.stem
                stat = f.stat()
                # Extract first line as description
                description = ""
                try:
                    first_line = f.read_text(errors="replace").split("\n", 1)[0].strip()
                    if first_line.startswith("#"):
                        first_line = first_line.lstrip("#").strip()
                    description = first_line
                except Exception:
                    pass
                skills.append({
                    "name": name,
                    "description": description[:200],
                    "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                    "size_bytes": stat.st_size,
                })
        db.collection("status").document("skills").set({
            "skills": skills,
            "total": len(skills),
            "updated_at": _now_iso(),
        })
    except Exception:
        logger.debug("firebase_reporter: skills write failed", exc_info=True)


# ── Signal connection health ─────────────────────────────────────────────────

def report_signal_status(connected: bool, last_message_at: Optional[str] = None,
                         message_count: int = 0) -> None:
    """Push Signal connection health to Firestore at status/signal."""
    def _write():
        db = _get_db()
        if not db:
            return
        try:
            db.collection("status").document("signal").set({
                "connected": connected,
                "last_message_at": last_message_at,
                "message_count": message_count,
                "updated_at": _now_iso(),
            })
        except Exception:
            logger.debug("firebase_reporter: signal status write failed", exc_info=True)
    _fire(_write)


# ── Crew / agent status ───────────────────────────────────────────────────────

def crew_started(crew: str, task_summary: str, eta_seconds: Optional[int] = None,
                 parent_task_id: Optional[str] = None,
                 model: Optional[str] = None) -> str:
    """Mark a crew as active.  Returns a task_id for later updates.

    If parent_task_id is set, this task is a sub-agent spawned by a parent task.
    model: the LLM model name used for this task (e.g. "qwen3:30b-a3b").
    """
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
            # Only update crew-level status for top-level tasks (not sub-agents)
            if not parent_task_id:
                db.collection("crews").document(crew).set({
                    "name": crew,
                    "state": "active",
                    "current_task": task_summary[:200],
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
                "summary": task_summary[:200],
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
                          f"→ {to_crew}: {reason[:100]}", task_id)
        except Exception:
            logger.debug("firebase_reporter: task_delegated write failed", exc_info=True)
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
            logger.debug("firebase_reporter: sub_agent_progress write failed", exc_info=True)
    _fire(_write)


# ── Proposals ─────────────────────────────────────────────────────────────────

def report_proposals(proposals: list[dict]) -> None:
    """Push current proposal list to Firestore for the dashboard."""
    def _write():
        db = _get_db()
        if not db:
            return
        try:
            db.collection("status").document("proposals").set({
                "proposals": proposals[:20],
                "updated_at": _now_iso(),
            })
        except Exception:
            logger.debug("firebase_reporter: proposals write failed", exc_info=True)
    _fire(_write)


# ── Fleet status ─────────────────────────────────────────────────────────

def report_fleet_status(fleet_data: list[dict], benchmarks: list[dict] = None) -> None:
    """Push LLM fleet container status to Firestore for the dashboard."""
    def _write():
        db = _get_db()
        if not db:
            return
        try:
            db.collection("status").document("fleet").set({
                "containers": fleet_data[:10],
                "benchmarks": (benchmarks or [])[:20],
                "updated_at": _now_iso(),
            })
        except Exception:
            logger.debug("firebase_reporter: fleet write failed", exc_info=True)
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
        # Prune oldest entries beyond 50 — only every 10th write to reduce Firestore load
        global _activity_write_count
        _activity_write_count = getattr(_add_activity, '_count', 0) + 1
        _add_activity._count = _activity_write_count
        if _activity_write_count % 10 == 0:
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


# ── LLM mode control ────────────────────────────────────────────────────────

def report_llm_mode(mode: str) -> None:
    """Push current LLM mode to Firestore for the dashboard."""
    def _write():
        db = _get_db()
        if not db:
            return
        try:
            db.collection("config").document("llm").set({
                "mode": mode,
                "updated_at": _now_iso(),
            })
        except Exception:
            logger.debug("firebase_reporter: llm mode write failed", exc_info=True)
    _fire(_write)


def read_llm_mode_from_firestore() -> str | None:
    """Read LLM mode from Firestore (dashboard-set value). Returns None if unavailable."""
    db = _get_db()
    if not db:
        return None
    try:
        doc = db.collection("config").document("llm").get()
        if doc.exists:
            mode = doc.to_dict().get("mode")
            if mode in ("local", "cloud", "hybrid", "insane"):
                return mode
    except Exception:
        logger.debug("firebase_reporter: failed to read llm mode from Firestore", exc_info=True)
    return None


_mode_listener_unsub = None  # Must be kept alive to prevent GC of the listener
_mode_poll_stop = threading.Event()  # Signal to stop the polling thread


def _apply_mode_if_changed(new_mode: str) -> bool:
    """Apply a mode change if it differs from current. Returns True if changed."""
    if new_mode not in ("local", "cloud", "hybrid", "insane"):
        return False
    from app.llm_mode import get_mode, set_mode
    if new_mode != get_mode():
        set_mode(new_mode)
        logger.info(f"firebase_reporter: mode changed from dashboard → {new_mode}")
        return True
    return False


def start_mode_listener() -> None:
    """Listen for dashboard mode changes via Firestore on_snapshot + polling fallback.

    The on_snapshot gRPC stream can silently drop in Docker containers,
    so we also poll every 15 seconds as a reliable backup.
    """
    global _mode_listener_unsub

    def _listen():
        global _mode_listener_unsub
        db = _get_db()
        if not db:
            return
        try:
            def on_snapshot(doc_snapshot, changes, read_time):
                for snap in doc_snapshot:
                    data = snap.to_dict()
                    new_mode = data.get("mode")
                    if new_mode:
                        _apply_mode_if_changed(new_mode)

            _mode_listener_unsub = (
                db.collection("config").document("llm").on_snapshot(on_snapshot)
            )
            logger.info("firebase_reporter: mode listener started (on_snapshot)")
        except Exception:
            logger.debug("firebase_reporter: mode listener failed", exc_info=True)
    _fire(_listen)

    # Start polling fallback in a daemon thread
    def _poll_mode():
        """Poll Firestore every 15s for mode changes — backup for flaky gRPC streams."""
        while not _mode_poll_stop.wait(15):
            try:
                mode = read_llm_mode_from_firestore()
                if mode:
                    _apply_mode_if_changed(mode)
            except Exception:
                pass  # never crash the poll loop
        logger.debug("firebase_reporter: mode poll stopped")

    t = threading.Thread(target=_poll_mode, daemon=True, name="firebase-mode-poll")
    t.start()
    logger.info("firebase_reporter: mode poll started (15s interval)")


# ── Metrics ──────────────────────────────────────────────────────────────────

def report_metrics() -> None:
    """Push composite metrics to Firestore for the dashboard."""
    db = _get_db()
    if not db:
        return
    try:
        from app.metrics import compute_metrics
        metrics = compute_metrics()
        db.collection("status").document("metrics").set({
            **metrics,
            "updated_at": _now_iso(),
        })
    except Exception:
        logger.debug("firebase_reporter: metrics write failed", exc_info=True)


# ── Circuit breakers ─────────────────────────────────────────────────────────

def report_circuit_breakers() -> None:
    """Push circuit breaker states to Firestore."""
    db = _get_db()
    if not db:
        return
    try:
        from app.circuit_breaker import get_all_states
        states = get_all_states()
        db.collection("status").document("circuit_breakers").set({
            "providers": states,
            "updated_at": _now_iso(),
        })
    except Exception:
        logger.debug("firebase_reporter: circuit breaker write failed", exc_info=True)


# ── Error journal ────────────────────────────────────────────────────────────

def report_errors() -> None:
    """Push recent errors to Firestore for dashboard display."""
    db = _get_db()
    if not db:
        return
    try:
        from app.self_heal import get_recent_errors, get_error_patterns
        errors = get_recent_errors(20)
        patterns = get_error_patterns()
        db.collection("status").document("errors").set({
            "recent": errors[:20],
            "patterns": patterns,
            "total_recent": len(errors),
            "updated_at": _now_iso(),
        })
    except Exception:
        logger.debug("firebase_reporter: errors write failed", exc_info=True)


# ── Evolution / experiments ──────────────────────────────────────────────────

def report_evolution() -> None:
    """Push evolution experiment history to Firestore."""
    db = _get_db()
    if not db:
        return
    try:
        from app.results_ledger import get_recent_results, get_best_score, get_improvement_trend
        results = get_recent_results(20)
        best = get_best_score()
        trend = get_improvement_trend(20)
        db.collection("status").document("evolution").set({
            "recent_experiments": results[:20],
            "best_score": best,
            "trend": trend[:20],
            "updated_at": _now_iso(),
        })
    except Exception:
        logger.debug("firebase_reporter: evolution write failed", exc_info=True)


# ── Request cost stats ───────────────────────────────────────────────────────

def report_request_costs() -> None:
    """Push per-request cost aggregates and per-crew breakdown to Firestore."""
    db = _get_db()
    if not db:
        return
    try:
        from app.llm_benchmarks import get_request_cost_stats, get_crew_cost_stats
        costs = {}
        crew_costs = {}
        for period in ("hour", "day", "week", "month"):
            costs[period] = get_request_cost_stats(period)
            crew_costs[period] = get_crew_cost_stats(period)
        db.collection("status").document("request_costs").set({
            "stats": costs,
            "by_crew": crew_costs,
            "updated_at": _now_iso(),
        })
    except Exception:
        logger.debug("firebase_reporter: request costs write failed", exc_info=True)


# ── Model catalog ────────────────────────────────────────────────────────────

def report_catalog() -> None:
    """Push model catalog and role assignments to Firestore."""
    db = _get_db()
    if not db:
        return
    try:
        from app.llm_catalog import CATALOG, ROLE_DEFAULTS
        from app.config import get_settings
        settings = get_settings()
        # Compact catalog for dashboard
        models = []
        for name, info in CATALOG.items():
            models.append({
                "name": name,
                "tier": info["tier"],
                "provider": info["provider"],
                "cost_input": info.get("cost_input_per_m", 0),
                "cost_output": info.get("cost_output_per_m", 0),
                "context": info.get("context", 0),
                "multimodal": info.get("multimodal", False),
                "tool_reliability": info.get("tool_use_reliability", 0),
                "description": info.get("description", ""),
            })
        # Current role assignments
        cost_mode = settings.cost_mode
        assignments = ROLE_DEFAULTS.get(cost_mode, ROLE_DEFAULTS.get("balanced", {}))
        db.collection("status").document("catalog").set({
            "models": models,
            "role_assignments": assignments,
            "cost_mode": cost_mode,
            "updated_at": _now_iso(),
        })
    except Exception:
        logger.debug("firebase_reporter: catalog write failed", exc_info=True)


def report_knowledge_base() -> None:
    """Push knowledge base stats to Firestore for the dashboard."""
    db = _get_db()
    if not db:
        return
    try:
        from app.knowledge_base.vectorstore import KnowledgeStore
        store = KnowledgeStore()
        stats = store.stats()
        # Compact document list for dashboard
        docs = []
        for d in stats.get("documents", [])[:50]:
            docs.append({
                "source": d.get("source", "unknown"),
                "format": d.get("format", "?"),
                "category": d.get("category", "general"),
                "chunks": d.get("total_chunks", 0),
                "ingested_at": d.get("ingested_at", ""),
            })
        db.collection("status").document("knowledge_base").set({
            "total_documents": stats.get("total_documents", 0),
            "total_chunks": stats.get("total_chunks", 0),
            "total_characters": stats.get("total_characters", 0),
            "estimated_tokens": stats.get("estimated_tokens", 0),
            "categories": stats.get("categories", {}),
            "documents": docs,
            "updated_at": _now_iso(),
        })
    except Exception:
        logger.debug("firebase_reporter: knowledge_base write failed", exc_info=True)


# ── L5: Ecological awareness stats ────────────────────────────────────────────

def report_ecological_stats() -> None:
    """Push ecological awareness stats to Firestore for the dashboard (L5)."""
    db = _get_db()
    if not db:
        return
    try:
        from app.memory.scoped_memory import retrieve_operational
        recent = retrieve_operational("scope_ecology", "crew execution", n=20)
        if not recent:
            return

        # Parse ecological reports for aggregate stats
        crew_stats: dict[str, list[float]] = {}
        for entry in recent:
            # Extract crew name and duration from "ECOLOGICAL: crew=X, ..."
            parts = {}
            for segment in entry.split(","):
                segment = segment.strip()
                if "=" in segment:
                    key, val = segment.split("=", 1)
                    key = key.replace("ECOLOGICAL: ", "").strip()
                    parts[key] = val.strip()
            crew_name = parts.get("crew", "unknown")
            try:
                duration = float(parts.get("duration", "0").rstrip("s"))
            except (ValueError, TypeError):
                duration = 0
            crew_stats.setdefault(crew_name, []).append(duration)

        # Build summary
        summary = {}
        for crew, durations in crew_stats.items():
            summary[crew] = {
                "recent_executions": len(durations),
                "avg_duration_s": round(sum(durations) / len(durations), 1) if durations else 0,
                "max_duration_s": round(max(durations), 1) if durations else 0,
            }

        db.collection("status").document("ecology").set({
            "crew_stats": summary,
            "total_recent_executions": sum(len(d) for d in crew_stats.values()),
            "updated_at": _now_iso(),
        })
    except Exception:
        logger.debug("firebase_reporter: ecology write failed", exc_info=True)


# ── Token usage stats ────────────────────────────────────────────────────────

def report_token_stats() -> None:
    """Push aggregated token usage to Firestore for the dashboard."""
    def _write():
        db = _get_db()
        if not db:
            return
        try:
            from app.llm_benchmarks import get_token_stats
            stats = {}
            for period in ("hour", "day", "week", "month", "quarter", "year"):
                stats[period] = get_token_stats(period)
            db.collection("status").document("tokens").set({
                "stats": stats,
                "updated_at": _now_iso(),
            })
        except Exception:
            logger.debug("firebase_reporter: token stats write failed", exc_info=True)
    _fire(_write)


# ── Self-Healing/Evolving Dashboard Data ─────────────────────────────────────

def report_anomalies() -> None:
    """Push recent anomaly alerts to Firestore for the dashboard."""
    def _write():
        db = _get_db()
        if not db:
            return
        try:
            from app.anomaly_detector import get_recent_alerts
            alerts = get_recent_alerts(20)
            db.collection("status").document("anomalies").set({
                "recent_alerts": alerts,
                "updated_at": _now_iso(),
            })
        except Exception:
            logger.debug("firebase_reporter: anomalies write failed", exc_info=True)
    _fire(_write)


def report_variants() -> None:
    """Push variant archive summary to Firestore for the dashboard."""
    def _write():
        db = _get_db()
        if not db:
            return
        try:
            from app.variant_archive import get_recent_variants, get_drift_score
            recent = get_recent_variants(20)
            drift = get_drift_score()
            max_gen = max((v.get("generation", 0) for v in recent), default=0) if recent else 0
            db.collection("status").document("variants").set({
                "recent": recent,
                "drift_score": drift,
                "max_generation": max_gen,
                "updated_at": _now_iso(),
            })
        except Exception:
            logger.debug("firebase_reporter: variants write failed", exc_info=True)
    _fire(_write)


def report_tech_radar() -> None:
    """Push tech radar discoveries to Firestore for the dashboard."""
    def _write():
        db = _get_db()
        if not db:
            return
        try:
            from app.memory.scoped_memory import retrieve_operational
            items = retrieve_operational("scope_tech_radar", "technology discovery", n=20)
            discoveries = []
            for item in (items or []):
                # Parse stored format: [category] title: summary. Action: ...
                import re as _re
                m = _re.match(r'\[(\w+)\]\s*(.+?):\s*(.+?)(?:\.\s*Action:\s*(.+))?$', item, _re.DOTALL)
                if m:
                    discoveries.append({
                        "category": m.group(1),
                        "title": m.group(2).strip(),
                        "summary": m.group(3).strip(),
                        "action": (m.group(4) or "").strip(),
                    })
                else:
                    discoveries.append({"category": "unknown", "title": item[:80], "summary": item[:200]})
            db.collection("status").document("tech_radar").set({
                "discoveries": discoveries[:15],
                "updated_at": _now_iso(),
            })
        except Exception:
            logger.debug("firebase_reporter: tech_radar write failed", exc_info=True)
    _fire(_write)


def report_deploys() -> None:
    """Push recent deploy log to Firestore for the dashboard."""
    def _write():
        db = _get_db()
        if not db:
            return
        try:
            import json as _json
            from pathlib import Path as _Path
            deploy_log = _Path("/app/workspace/deploy_log.json")
            if deploy_log.exists():
                entries = _json.loads(deploy_log.read_text())[-10:]
            else:
                entries = []
            db.collection("status").document("deploys").set({
                "recent": entries,
                "updated_at": _now_iso(),
            })
        except Exception:
            logger.debug("firebase_reporter: deploys write failed", exc_info=True)
    _fire(_write)


def report_proposal_actions() -> None:
    """Poll proposal_actions collection for dashboard-initiated approve/reject."""
    db = _get_db()
    if not db:
        return
    try:
        docs = db.collection("proposal_actions").where("status", "==", "pending").limit(5).get()
        for snap in docs:
            data = snap.to_dict()
            pid = data.get("proposal_id")
            action = data.get("action")
            if not pid or not action:
                snap.reference.update({"status": "invalid"})
                continue
            try:
                if action == "approve":
                    from app.proposals import approve_proposal
                    result = approve_proposal(pid)
                elif action == "reject":
                    from app.proposals import reject_proposal
                    result = reject_proposal(pid)
                elif action == "rollback":
                    result = f"Rollback #{pid} — use Signal for rollback"
                else:
                    result = f"Unknown action: {action}"
                snap.reference.update({"status": "done", "result": result[:200]})
                logger.info(f"firebase_reporter: proposal action {action} #{pid}: {result[:100]}")
            except Exception as e:
                snap.reference.update({"status": "error", "error": str(e)[:200]})
    except Exception:
        logger.debug("firebase_reporter: proposal actions poll failed", exc_info=True)


# ── KB Queue Poller ──────────────────────────────────────────────────────────

_kb_poll_stop = threading.Event()


def start_kb_queue_poller() -> None:
    """Poll Firestore kb_queue for pending uploads and ingest them."""

    def _poll_kb():
        import base64
        import tempfile

        while not _kb_poll_stop.wait(10):
            db = _get_db()
            if not db:
                continue
            try:
                docs = (
                    db.collection("kb_queue")
                    .where("status", "==", "pending")
                    .limit(5)
                    .get()
                )
                for snap in docs:
                    data = snap.to_dict()
                    try:
                        content_b64 = data.get("content_b64", "")
                        fname = data.get("filename", "upload.txt")
                        category = data.get("category", "general")

                        raw = base64.b64decode(content_b64)

                        suffix = "." + fname.rsplit(".", 1)[-1] if "." in fname else ".txt"
                        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                            tmp.write(raw)
                            tmp_path = tmp.name

                        try:
                            from app.knowledge_base.vectorstore import KnowledgeStore
                            ks = KnowledgeStore()
                            result = ks.add_document(tmp_path, category=category)

                            snap.reference.update({
                                "status": "done",
                                "chunks_created": result.chunks_created,
                                "processed_at": _now_iso(),
                            })
                            logger.info(f"firebase_reporter: KB ingested '{fname}' -> {result.chunks_created} chunks")
                        finally:
                            import os as _os
                            _os.unlink(tmp_path)

                    except Exception as e:
                        snap.reference.update({
                            "status": "error",
                            "error": str(e)[:200],
                            "processed_at": _now_iso(),
                        })
                        logger.warning(f"firebase_reporter: KB ingest failed for '{data.get('filename')}': {e}")

            except Exception:
                pass
        logger.debug("firebase_reporter: KB poll stopped")

    t = threading.Thread(target=_poll_kb, daemon=True, name="firebase-kb-poll")
    t.start()
    logger.info("firebase_reporter: KB queue poller started (10s interval)")
