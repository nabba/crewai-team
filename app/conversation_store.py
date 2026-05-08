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


# ── Schema migrations (T3-10) ────────────────────────────────────────────────

_MIGRATIONS: list[tuple[str, str]] = [
    ("v1_messages", """
        CREATE TABLE IF NOT EXISTS messages (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            sender_id TEXT    NOT NULL,
            role      TEXT    NOT NULL,
            content   TEXT    NOT NULL,
            ts        TEXT    NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_sender_ts ON messages(sender_id, ts);
    """),
    ("v2_tasks", """
        CREATE TABLE IF NOT EXISTS tasks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            sender_id   TEXT    NOT NULL,
            crew        TEXT    NOT NULL DEFAULT '',
            started_at  TEXT    NOT NULL,
            completed_at TEXT,
            success     INTEGER NOT NULL DEFAULT 1,
            duration_s  REAL,
            error_type  TEXT    DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_tasks_started ON tasks(started_at);
    """),
    ("v3_fts5", """
        CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
        USING fts5(content, sender_id UNINDEXED, role UNINDEXED, ts UNINDEXED,
                   content='messages', content_rowid='id');
    """),
    ("v3_fts5_triggers", """
        CREATE TRIGGER IF NOT EXISTS messages_fts_ai AFTER INSERT ON messages BEGIN
            INSERT INTO messages_fts(rowid, content, sender_id, role, ts)
            VALUES (new.id, new.content, new.sender_id, new.role, new.ts);
        END;
        CREATE TRIGGER IF NOT EXISTS messages_fts_ad AFTER DELETE ON messages BEGIN
            INSERT INTO messages_fts(messages_fts, rowid, content, sender_id, role, ts)
            VALUES ('delete', old.id, old.content, old.sender_id, old.role, old.ts);
        END;
    """),
    # Durable inbound-message queue.  Every accepted webhook is persisted
    # BEFORE the 200 OK returns, so a mid-processing container restart can
    # replay unfinished work on the next startup instead of silently losing
    # the user's message.  UNIQUE(sender, signal_ts) makes enqueue
    # idempotent (the forwarder may retry on transient errors without
    # creating duplicate rows).
    ("v4_inbound_queue", """
        CREATE TABLE IF NOT EXISTS inbound_queue (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            sender                TEXT    NOT NULL,
            message               TEXT    NOT NULL,
            signal_ts             INTEGER NOT NULL,
            attachments_json      TEXT    NOT NULL DEFAULT '[]',
            status                TEXT    NOT NULL DEFAULT 'queued',
            attempts              INTEGER NOT NULL DEFAULT 0,
            last_error            TEXT    DEFAULT '',
            received_at           TEXT    NOT NULL,
            processing_started_at TEXT,
            processed_at          TEXT
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_inbound_dedup
            ON inbound_queue(sender, signal_ts);
        CREATE INDEX IF NOT EXISTS idx_inbound_status
            ON inbound_queue(status, received_at);
    """),
    # Durable outbound-reply queue.  Counterpart to inbound_queue on the
    # reply side: every Signal send-with-attachment is persisted BEFORE the
    # actual signal-cli call, so a container restart mid-send doesn't lose
    # the attachment.  Sent rows record the signal-cli timestamp so a
    # subsequent replay can tell "already delivered" from "never sent".
    ("v5_outbound_queue", """
        CREATE TABLE IF NOT EXISTS outbound_queue (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            recipient         TEXT    NOT NULL,
            message           TEXT    NOT NULL DEFAULT '',
            attachments_json  TEXT    NOT NULL DEFAULT '[]',
            reply_to_id       INTEGER,
            status            TEXT    NOT NULL DEFAULT 'queued',
            attempts          INTEGER NOT NULL DEFAULT 0,
            last_error        TEXT    DEFAULT '',
            signal_timestamp  INTEGER,
            queued_at         TEXT    NOT NULL,
            sent_at           TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_outbound_status
            ON outbound_queue(status, queued_at);
    """),
]


def _run_migrations(conn: sqlite3.Connection) -> None:
    """Run pending schema migrations. Idempotent."""
    conn.execute("CREATE TABLE IF NOT EXISTS _schema_version (name TEXT PRIMARY KEY, applied_at TEXT)")
    applied = {r[0] for r in conn.execute("SELECT name FROM _schema_version").fetchall()}
    for name, sql in _MIGRATIONS:
        if name not in applied:
            try:
                conn.executescript(sql)
                conn.execute(
                    "INSERT INTO _schema_version VALUES (?, ?)",
                    (name, datetime.now(timezone.utc).isoformat()),
                )
                logger.info(f"conversation_store: applied migration '{name}'")
            except sqlite3.OperationalError as exc:
                # FTS5 may not be compiled in on some SQLite builds — log and skip
                logger.warning(f"conversation_store: migration '{name}' skipped: {exc}")
    conn.commit()


def _get_conn() -> sqlite3.Connection:
    """Return a thread-local SQLite connection, creating it if needed."""
    if not hasattr(_local, "conn") or _local.conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        _run_migrations(conn)
        _local.conn = conn
    return _local.conn


_SENDER_KEY_FILE = Path("/app/workspace/.sender_key")


def _get_stable_sender_key() -> bytes:
    """Get a stable HMAC key for sender ID hashing.

    Priority: gateway_secret > persisted file > generate and persist new.
    The persisted file ensures sender IDs survive process restarts.
    """
    # Priority 1: gateway secret (best — derived from env config)
    try:
        secret = get_gateway_secret()
        if secret and len(secret) >= 8:
            return secret.encode()
    except Exception:
        pass
    # Priority 2: persisted key file (survives restarts)
    try:
        if _SENDER_KEY_FILE.exists():
            key = _SENDER_KEY_FILE.read_bytes()
            if len(key) >= 16:
                return key
    except Exception:
        pass
    # Priority 3: generate and persist new key
    import secrets
    key = secrets.token_bytes(32)
    try:
        _SENDER_KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
        _SENDER_KEY_FILE.write_bytes(key)
        logger.info("conversation_store: generated persistent sender key")
    except Exception:
        logger.warning("conversation_store: could not persist sender key — IDs may change on restart")
    return key


def _sender_id(sender: str) -> str:
    """Return a stable, non-reversible 16-char token for a sender number."""
    key = _get_stable_sender_key()
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

    # Filter out system/internal responses that could contaminate crew context
    _INTERNAL_PREFIXES = (
        "LLM Discovery:", "Evolution session", "Retrospective analysis",
        "Self-heal:", "Improvement scan", "Tech Radar", "Code audit",
        "Training pipeline", "Consciousness probe", "Behavioral assessment",
        "Prosocial session", "Fiction ingest", "Knowledge base ingestion",
    )

    lines = []
    for role, content in rows:
        label = "User" if role == "user" else "Assistant"
        # Skip internal system responses from conversation history
        if role == "assistant" and any(content.strip().startswith(p) for p in _INTERNAL_PREFIXES):
            continue
        # Truncate assistant responses to prevent context pollution
        # but keep enough for follow-up context (was 300, raised to 500)
        max_len = 500 if role == "assistant" else 600
        snippet = content[:max_len] + ("…" if len(content) > max_len else "")
        lines.append(f"{label}: {snippet}")
    return "\n".join(lines)


def get_recent_messages(sender: str, limit: int = 10) -> list[dict]:
    """Return raw recent messages as dicts, newest-first.

    Each entry has ``{"role": str, "content": str, "ts": float}``.
    Used by callers that need to walk history programmatically (vs.
    ``get_history`` which returns a formatted prompt string).
    """
    try:
        conn = _get_conn()
        rows = conn.execute(
            """
            SELECT role, content, ts
              FROM messages
             WHERE sender_id = ?
             ORDER BY ts DESC
             LIMIT ?
            """,
            (_sender_id(sender), max(1, int(limit))),
        ).fetchall()
    except Exception:
        logger.exception("conversation_store: failed to retrieve recent messages")
        return []
    return [{"role": r[0], "content": r[1], "ts": float(r[2])} for r in rows]


def get_last_assistant_message(sender: str) -> str:
    """Return the raw content of the most recent assistant reply to this
    sender, or empty string. Used by Phase 15 grounding to supply
    prior-response context to correction detection."""
    try:
        conn = _get_conn()
        row = conn.execute(
            """
            SELECT content FROM messages
            WHERE sender_id = ? AND role = 'assistant'
            ORDER BY ts DESC
            LIMIT 1
            """,
            (_sender_id(sender),),
        ).fetchone()
    except Exception:
        logger.debug("conversation_store: get_last_assistant_message failed",
                     exc_info=True)
        return ""
    return str(row[0]) if row else ""


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


# ── Durable inbound message queue (crash-safe replay) ────────────────────────
# Before every /signal/inbound response returns 200, the message is written
# to the inbound_queue table.  If the container dies mid-processing, the
# next startup calls replay_pending_inbound() which re-dispatches any rows
# that are still 'queued' or 'processing'.  UNIQUE(sender, signal_ts) keeps
# enqueue idempotent under forwarder retries.

import json as _json


def enqueue_inbound(
    sender: str, message: str, signal_ts: int, attachments: list | None = None,
) -> int | None:
    """Persist an incoming Signal message.  Returns the queue row id, or
    None if the (sender, signal_ts) pair was already queued (idempotent).

    Called from /signal/inbound BEFORE returning 200 OK, so that every
    accepted message has a durable record even if handle_task() crashes
    or the container is restarted mid-processing.
    """
    try:
        conn = _get_conn()
        cur = conn.execute(
            """
            INSERT INTO inbound_queue
                (sender, message, signal_ts, attachments_json, status, received_at)
            VALUES (?, ?, ?, ?, 'queued', ?)
            ON CONFLICT(sender, signal_ts) DO NOTHING
            """,
            (
                sender, message, int(signal_ts),
                _json.dumps(attachments or []),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        if cur.rowcount == 0:
            return None  # already enqueued
        return cur.lastrowid
    except Exception:
        logger.exception("conversation_store: enqueue_inbound failed")
        return None


def mark_inbound_processing(queue_id: int) -> None:
    try:
        conn = _get_conn()
        conn.execute(
            "UPDATE inbound_queue SET status='processing', "
            "processing_started_at=?, attempts=attempts+1 WHERE id=?",
            (datetime.now(timezone.utc).isoformat(), queue_id),
        )
        conn.commit()
    except Exception:
        logger.exception("conversation_store: mark_inbound_processing failed")


def mark_inbound_done(queue_id: int) -> None:
    try:
        conn = _get_conn()
        conn.execute(
            "UPDATE inbound_queue SET status='done', processed_at=? WHERE id=?",
            (datetime.now(timezone.utc).isoformat(), queue_id),
        )
        conn.commit()
    except Exception:
        logger.exception("conversation_store: mark_inbound_done failed")


def mark_inbound_failed(queue_id: int, error: str) -> None:
    try:
        conn = _get_conn()
        conn.execute(
            "UPDATE inbound_queue SET status='failed', processed_at=?, "
            "last_error=? WHERE id=?",
            (datetime.now(timezone.utc).isoformat(), str(error)[:500], queue_id),
        )
        conn.commit()
    except Exception:
        logger.exception("conversation_store: mark_inbound_failed failed")


def get_pending_inbound(max_attempts: int = 3) -> list[dict]:
    """Return inbound-queue rows that need replay at startup.

    Policy
    ------
    * ``'queued'`` rows → always replay (never ran in a prior instance).
    * ``'processing'`` rows → replay only when ``attempts == 0`` AND the
      row was picked up within the last 5 minutes.  Older processing
      rows are assumed handled by a previous instance that shipped
      partial output before dying; replaying them spawns a parallel
      handle_task that hits the 15-min soft timeout and emits a
      duplicate "Still working" heartbeat to the user. That was the
      observed failure mode when rapid gateway restarts (during
      development) compounded with a long-running research task.

    Callers who want the ``'failed'`` rows for operator inspection
    should query ``inbound_queue`` directly.
    """
    try:
        conn = _get_conn()
        # Queued rows: always replay.
        queued = conn.execute(
            """
            SELECT id, sender, message, signal_ts, attachments_json,
                   status, attempts, received_at
              FROM inbound_queue
             WHERE status = 'queued'
               AND attempts < ?
             ORDER BY received_at ASC
            """,
            (max_attempts,),
        ).fetchall()
        # Processing rows: only very recent first-attempt ones.  Older
        # processing rows get marked 'failed' and skipped — see below.
        processing_recent = conn.execute(
            """
            SELECT id, sender, message, signal_ts, attachments_json,
                   status, attempts, received_at
              FROM inbound_queue
             WHERE status = 'processing'
               AND attempts = 0
               AND processing_started_at IS NOT NULL
               AND processing_started_at > datetime('now', '-5 minutes')
             ORDER BY received_at ASC
            """,
        ).fetchall()
        # Stuck processing rows: mark failed so they don't keep
        # matching on future startups either.  BEFORE marking them
        # failed, collect their message text so we can invalidate any
        # cache entries that were stored mid-flight.  This prevents a
        # partially-cached (and possibly broken) result from resurfacing
        # as a similarity=1.0 HIT on the user's next identical request.
        _abandoned_messages = [
            row[0] for row in conn.execute(
                """
                SELECT message FROM inbound_queue
                 WHERE status = 'processing'
                   AND (
                        attempts >= 1
                     OR processing_started_at IS NULL
                     OR processing_started_at <= datetime('now', '-5 minutes')
                   )
                """,
            ).fetchall()
        ]
        conn.execute(
            """
            UPDATE inbound_queue
               SET status = 'failed',
                   last_error = 'abandoned: crossed container restart boundary'
             WHERE status = 'processing'
               AND (
                    attempts >= 1
                 OR processing_started_at IS NULL
                 OR processing_started_at <= datetime('now', '-5 minutes')
               )
            """,
        )
        conn.commit()

        # Fire-and-forget cache invalidation for each abandoned task.
        # Runs in a DAEMON THREAD so a slow/unreachable Ollama embedder
        # (ChromaDB → embed → Ollama call) can't block startup replay.
        # If invalidation itself fails, the worst case is that a stale
        # cache entry survives until the next identical task — then
        # its own cache_store gate catches the failure-shape.
        if _abandoned_messages:
            import threading as _th
            _msgs_to_invalidate = list(_abandoned_messages)
            def _bg_invalidate_many():
                try:
                    from app.result_cache import invalidate_by_task as _inv
                    for msg in _msgs_to_invalidate:
                        if msg:
                            _inv(msg)
                    logger.info(
                        "conversation_store: invalidated cache for %d "
                        "abandoned task(s) on startup replay "
                        "(background thread)",
                        len(_msgs_to_invalidate),
                    )
                except Exception:
                    logger.debug(
                        "conversation_store: post-replay cache invalidation "
                        "failed in background thread",
                        exc_info=True,
                    )
            _th.Thread(
                target=_bg_invalidate_many, daemon=True,
                name="startup-cache-invalidate",
            ).start()

        rows = list(queued) + list(processing_recent)
    except Exception:
        logger.exception("conversation_store: get_pending_inbound failed")
        return []
    out = []
    for r in rows or []:
        try:
            attachments = _json.loads(r[4] or "[]")
        except Exception:
            attachments = []
        out.append({
            "id": r[0], "sender": r[1], "message": r[2],
            "signal_ts": r[3], "attachments": attachments,
            "status": r[5], "attempts": r[6], "received_at": r[7],
        })
    return out


def prune_old_inbound(days: int = 7) -> int:
    """Delete done/failed inbound rows older than N days.  Returns count."""
    try:
        conn = _get_conn()
        cur = conn.execute(
            "DELETE FROM inbound_queue "
            "WHERE status IN ('done', 'failed') "
            "  AND received_at < datetime('now', '-' || ? || ' days')",
            (str(days),),
        )
        conn.commit()
        return cur.rowcount
    except Exception:
        logger.exception("conversation_store: prune_old_inbound failed")
        return 0


# ── Durable outbound send queue (crash-safe Signal delivery) ─────────────────
# Counterpart to inbound_queue, for outbound Signal sends.  A row is written
# BEFORE the signal-cli call; on success the row is marked 'sent' with the
# returned Signal timestamp; on crash or failure it stays 'queued' and the
# startup replay re-sends it.  Especially important for attachment sends
# which happen in a background executor and can be interrupted mid-flight.

def enqueue_outbound(
    recipient: str, message: str = "",
    attachments: list | None = None,
    reply_to_id: int | None = None,
) -> int | None:
    """Persist an outbound Signal send BEFORE actually sending.  Returns
    the queue row id.  Call mark_outbound_sent() or mark_outbound_failed()
    based on the signal-cli response."""
    try:
        conn = _get_conn()
        cur = conn.execute(
            """
            INSERT INTO outbound_queue
                (recipient, message, attachments_json, reply_to_id,
                 status, queued_at)
            VALUES (?, ?, ?, ?, 'queued', ?)
            """,
            (
                recipient, message or "",
                _json.dumps(attachments or []),
                reply_to_id,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        return cur.lastrowid
    except Exception:
        logger.exception("conversation_store: enqueue_outbound failed")
        return None


def mark_outbound_sent(queue_id: int, signal_timestamp: int | None = None) -> None:
    try:
        conn = _get_conn()
        conn.execute(
            "UPDATE outbound_queue SET status='sent', sent_at=?, "
            "signal_timestamp=?, attempts=attempts+1 WHERE id=?",
            (datetime.now(timezone.utc).isoformat(),
             int(signal_timestamp) if signal_timestamp else None, queue_id),
        )
        conn.commit()
    except Exception:
        logger.exception("conversation_store: mark_outbound_sent failed")


def mark_outbound_failed(queue_id: int, error: str) -> None:
    try:
        conn = _get_conn()
        conn.execute(
            "UPDATE outbound_queue SET status='failed', attempts=attempts+1, "
            "last_error=? WHERE id=?",
            (str(error)[:500], queue_id),
        )
        conn.commit()
    except Exception:
        logger.exception("conversation_store: mark_outbound_failed failed")


def mark_outbound_requeued(queue_id: int) -> None:
    """Reset a failed row back to 'queued' so the next replay cycle retries it."""
    try:
        conn = _get_conn()
        conn.execute(
            "UPDATE outbound_queue SET status='queued' WHERE id=?",
            (queue_id,),
        )
        conn.commit()
    except Exception:
        logger.exception("conversation_store: mark_outbound_requeued failed")


def get_pending_outbound(max_attempts: int = 3) -> list[dict]:
    """Return queued rows in oldest-first order.  Rows whose attempts have
    hit the cap are excluded so a poisoned payload can't replay forever —
    they stay in the DB with last_error populated for inspection."""
    try:
        conn = _get_conn()
        rows = conn.execute(
            """
            SELECT id, recipient, message, attachments_json, reply_to_id,
                   status, attempts, queued_at
              FROM outbound_queue
             WHERE status = 'queued' AND attempts < ?
             ORDER BY queued_at ASC
            """,
            (max_attempts,),
        ).fetchall()
    except Exception:
        logger.exception("conversation_store: get_pending_outbound failed")
        return []
    out = []
    for r in rows or []:
        try:
            attachments = _json.loads(r[3] or "[]")
        except Exception:
            attachments = []
        out.append({
            "id": r[0], "recipient": r[1], "message": r[2],
            "attachments": attachments, "reply_to_id": r[4],
            "status": r[5], "attempts": r[6], "queued_at": r[7],
        })
    return out


def prune_old_outbound(days: int = 7) -> int:
    """Delete sent/failed outbound rows older than N days.  Returns count."""
    try:
        conn = _get_conn()
        cur = conn.execute(
            "DELETE FROM outbound_queue "
            "WHERE status IN ('sent', 'failed') "
            "  AND queued_at < datetime('now', '-' || ? || ' days')",
            (str(days),),
        )
        conn.commit()
        return cur.rowcount
    except Exception:
        logger.exception("conversation_store: prune_old_outbound failed")
        return 0


# ── FTS5 full-text search (T3-10) ────────────────────────────────────────────

def search_messages(query: str, sender: str | None = None, limit: int = 10) -> list[dict]:
    """Full-text search across conversations using FTS5.

    Returns a list of dicts: {role, content_snippet, ts}. Empty list on no
    results or if FTS5 is unavailable on this SQLite build.
    """
    if not query or not query.strip():
        return []
    import re
    clean = re.sub(r'[^\w\s]', ' ', query).strip()
    if not clean:
        return []
    try:
        conn = _get_conn()
        if sender:
            sid = _sender_id(sender)
            rows = conn.execute(
                """SELECT m.role, m.content, m.ts,
                          snippet(messages_fts, 0, '>>>', '<<<', '...', 40)
                   FROM messages_fts
                   JOIN messages m ON m.id = messages_fts.rowid
                   WHERE messages_fts MATCH ? AND m.sender_id = ?
                   ORDER BY m.ts DESC LIMIT ?""",
                (clean, sid, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT m.role, m.content, m.ts,
                          snippet(messages_fts, 0, '>>>', '<<<', '...', 40)
                   FROM messages_fts
                   JOIN messages m ON m.id = messages_fts.rowid
                   WHERE messages_fts MATCH ?
                   ORDER BY m.ts DESC LIMIT ?""",
                (clean, limit),
            ).fetchall()
        return [{"role": r[0], "content_snippet": r[3] or r[1][:200], "ts": r[2]} for r in rows]
    except Exception:
        logger.debug("search_messages failed", exc_info=True)
        return []


def rebuild_fts_index() -> int:
    """Rebuild FTS5 index from existing data. Idempotent."""
    try:
        conn = _get_conn()
        conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        logger.info(f"FTS5 rebuilt: {count} messages")
        return count
    except Exception:
        logger.debug("FTS5 rebuild failed", exc_info=True)
        return 0
