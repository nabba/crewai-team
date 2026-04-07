"""
Persistent conversation history using SQLite.

Each Signal sender gets a rolling window of their recent exchanges stored
in /app/workspace/conversations.db.  This file is included in workspace
git backups so history survives power outages and redeployments.

Sender phone numbers are stored as a truncated HMAC-SHA256 so the raw
number is never written to disk.

Task tracking: the tasks table records timing and success/failure for
each user request, providing data for the metrics system.
"""
import hashlib
import hmac
import logging
import sqlite3
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path

from app.config import get_gateway_secret

logger = logging.getLogger(__name__)

DB_PATH = Path("/app/workspace/conversations.db")

# Global connection pool (one connection per thread via threading.local)
_local = threading.local()
_init_lock = threading.Lock()


def _get_conn() -> sqlite3.Connection:
    """Return a thread-local SQLite connection, creating it if needed."""
    if not hasattr(_local, "conn") or _local.conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")   # concurrent reads during writes
        conn.execute("PRAGMA synchronous=NORMAL") # durable without full fsync overhead
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                sender_id TEXT    NOT NULL,
                role      TEXT    NOT NULL,  -- 'user' or 'assistant'
                content   TEXT    NOT NULL,
                ts        TEXT    NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sender_ts "
            "ON messages(sender_id, ts)"
        )
        # Task tracking table — records timing and success for metrics
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                sender_id   TEXT    NOT NULL,
                crew        TEXT    NOT NULL DEFAULT '',
                started_at  TEXT    NOT NULL,
                completed_at TEXT,
                success     INTEGER NOT NULL DEFAULT 1,
                duration_s  REAL,
                error_type  TEXT    DEFAULT ''
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tasks_started "
            "ON tasks(started_at)"
        )
        conn.commit()
        _local.conn = conn
    return _local.conn


def _sender_id(sender: str) -> str:
    """Return a stable, non-reversible 16-char token for a sender number."""
    # Use gateway secret as HMAC key so IDs are unpredictable even if DB leaks.
    # No fallback key — if the secret is unavailable the system should not
    # silently degrade to a brutable hash.
    try:
        key = get_gateway_secret().encode()
        if len(key) < 8:
            raise ValueError("gateway secret too short for secure HMAC")
    except Exception:
        logger.error("conversation_store: gateway secret unavailable — cannot hash sender ID securely")
        # Use a per-process random salt so at least IDs are unpredictable
        # within this process lifetime (won't be stable across restarts)
        import secrets
        key = getattr(_sender_id, "_ephemeral_key", None)
        if key is None:
            key = secrets.token_bytes(32)
            _sender_id._ephemeral_key = key  # type: ignore[attr-defined]
    return hmac.new(key, sender.encode(), hashlib.sha256).hexdigest()[:16]


def add_message(sender: str, role: str, content: str) -> None:
    """Append a message (role='user' or 'assistant') to the conversation log."""
    ts = datetime.now(timezone.utc).isoformat()
    try:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO messages (sender_id, role, content, ts) VALUES (?, ?, ?, ?)",
            (_sender_id(sender), role, content, ts),
        )
        conn.commit()
    except Exception:
        logger.exception("conversation_store: failed to persist message")


def get_history(sender: str, n: int = 10) -> str:
    """
    Return the last *n* user+assistant exchanges as a formatted string,
    oldest-first, suitable for injecting into an LLM prompt.
    Returns "" if no history exists.
    """
    try:
        conn = _get_conn()
        rows = conn.execute(
            """
            SELECT role, content FROM (
                SELECT role, content, ts
                FROM messages
                WHERE sender_id = ?
                ORDER BY ts DESC
                LIMIT ?
            ) ORDER BY ts ASC
            """,
            (_sender_id(sender), n * 2),  # n exchanges = up to 2n rows
        ).fetchall()
    except Exception:
        logger.exception("conversation_store: failed to retrieve history")
        return ""

    if not rows:
        return ""

    lines = []
    for role, content in rows:
        label = "User" if role == "user" else "Assistant"
        # Truncate very long individual messages to keep prompt size bounded
        snippet = content[:600] + ("…" if len(content) > 600 else "")
        lines.append(f"{label}: {snippet}")
    return "\n".join(lines)


# ── Task tracking (for metrics) ─────────────────────────────────────────────

def start_task(sender: str, crew: str = "") -> int:
    """Record the start of a task. Returns the task row ID."""
    ts = datetime.now(timezone.utc).isoformat()
    try:
        conn = _get_conn()
        cur = conn.execute(
            "INSERT INTO tasks (sender_id, crew, started_at) VALUES (?, ?, ?)",
            (_sender_id(sender), crew, ts),
        )
        conn.commit()
        return cur.lastrowid
    except Exception:
        logger.exception("conversation_store: failed to start task")
        return -1


def complete_task(task_id: int, success: bool = True, error_type: str = "") -> None:
    """Record the completion of a task with timing."""
    ts = datetime.now(timezone.utc).isoformat()
    try:
        conn = _get_conn()
        # Compute duration from started_at
        row = conn.execute(
            "SELECT started_at FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        duration = 0.0
        if row:
            try:
                started = datetime.fromisoformat(row[0])
                duration = (datetime.now(timezone.utc) - started).total_seconds()
            except (ValueError, TypeError):
                pass
        conn.execute(
            "UPDATE tasks SET completed_at = ?, success = ?, duration_s = ?, error_type = ? WHERE id = ?",
            (ts, 1 if success else 0, duration, error_type, task_id),
        )
        conn.commit()
    except Exception:
        logger.exception("conversation_store: failed to complete task")


def update_task_crew(task_id: int, crew: str) -> None:
    """Update the crew name on a task (set after Commander routing)."""
    try:
        conn = _get_conn()
        conn.execute("UPDATE tasks SET crew = ? WHERE id = ?", (crew, task_id))
        conn.commit()
    except Exception:
        logger.debug("conversation_store: failed to update task crew", exc_info=True)


def count_recent_tasks(hours: int = 24) -> tuple[int, int]:
    """Count (total, successful) tasks in the last N hours."""
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    try:
        conn = _get_conn()
        total = conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE started_at > ? AND completed_at IS NOT NULL",
            (cutoff,),
        ).fetchone()[0]
        successful = conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE started_at > ? AND completed_at IS NOT NULL AND success = 1",
            (cutoff,),
        ).fetchone()[0]
        return (total, successful)
    except Exception:
        logger.exception("conversation_store: failed to count tasks")
        return (0, 0)


def avg_response_time(hours: int = 24) -> float:
    """Average task duration in seconds over the last N hours."""
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    try:
        conn = _get_conn()
        row = conn.execute(
            "SELECT AVG(duration_s) FROM tasks WHERE started_at > ? AND completed_at IS NOT NULL AND duration_s > 0",
            (cutoff,),
        ).fetchone()
        return row[0] if row and row[0] else 0.0
    except Exception:
        logger.exception("conversation_store: failed to compute avg response time")
        return 0.0


# Default ETAs (seconds) — used when no historical data exists yet
_DEFAULT_ETA: dict[str, int] = {
    "commander": 30,
    "research": 120,
    "coding": 180,
    "writing": 90,
    "self_improvement": 300,
    "retrospective": 180,
}


def get_crew_avg_duration(crew: str) -> float:
    """Average task duration for a specific crew (last 7 days, successful tasks).

    Returns seconds. Falls back to defaults if no historical data.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    try:
        conn = _get_conn()
        row = conn.execute(
            "SELECT AVG(duration_s), COUNT(*) FROM tasks "
            "WHERE crew = ? AND started_at > ? AND completed_at IS NOT NULL "
            "AND success = 1 AND duration_s > 0",
            (crew, cutoff),
        ).fetchone()
        avg, count = row if row else (None, 0)
        if avg and count >= 3:
            return round(avg, 1)
    except Exception:
        logger.debug("conversation_store: failed to get crew avg duration", exc_info=True)
    return float(_DEFAULT_ETA.get(crew, 120))


def estimate_eta(crew: str) -> int:
    """Return estimated seconds for a task on this crew, based on history."""
    return int(get_crew_avg_duration(crew))
