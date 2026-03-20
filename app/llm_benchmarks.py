"""
llm_benchmarks.py — Track LLM model performance and token usage per task type.

Stores outcomes (success/failure, latency, tokens) in SQLite.
Used by llm_selector to prefer models that historically perform well.
Also tracks per-model token usage with cost estimation for the dashboard.
"""

import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = Path("/app/workspace/llm_benchmarks.db")
_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    if not hasattr(_local, "conn") or _local.conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS benchmarks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                model       TEXT NOT NULL,
                task_type   TEXT NOT NULL,
                success     INTEGER NOT NULL,  -- 1=success, 0=failure
                latency_ms  INTEGER,
                tokens      INTEGER,
                ts          TEXT NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_model_task "
            "ON benchmarks(model, task_type)"
        )
        conn.execute("""
            CREATE TABLE IF NOT EXISTS token_usage (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                model             TEXT NOT NULL,
                prompt_tokens     INTEGER NOT NULL,
                completion_tokens INTEGER NOT NULL,
                total_tokens      INTEGER NOT NULL,
                cost_usd          REAL NOT NULL DEFAULT 0.0,
                ts                TEXT NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tu_model_ts "
            "ON token_usage(model, ts)"
        )
        conn.execute("""
            CREATE TABLE IF NOT EXISTS request_costs (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id        TEXT NOT NULL,
                crew_name         TEXT NOT NULL DEFAULT '',
                total_prompt      INTEGER NOT NULL,
                total_completion  INTEGER NOT NULL,
                total_cost_usd    REAL NOT NULL,
                call_count        INTEGER NOT NULL,
                models_used       TEXT NOT NULL,
                ts                TEXT NOT NULL
            )
        """)
        # Add crew_name column if upgrading from older schema
        try:
            conn.execute("ALTER TABLE request_costs ADD COLUMN crew_name TEXT NOT NULL DEFAULT ''")
        except sqlite3.OperationalError:
            pass  # column already exists
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rc_ts "
            "ON request_costs(ts)"
        )
        conn.commit()
        _local.conn = conn
    return _local.conn


def record(
    model: str,
    task_type: str,
    success: bool,
    latency_ms: int = 0,
    tokens: int = 0,
) -> None:
    """Record a model invocation outcome."""
    try:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO benchmarks (model, task_type, success, latency_ms, tokens, ts) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (model, task_type, int(success), latency_ms, tokens,
             datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    except Exception:
        logger.debug("llm_benchmarks: failed to record", exc_info=True)


def get_scores(task_type: str) -> dict[str, float]:
    """
    Return model→score for a task type.
    Score = success_rate * speed_factor (higher is better).
    Only considers last 50 runs per model.
    """
    try:
        conn = _get_conn()
        rows = conn.execute(
            """
            SELECT model,
                   AVG(success) as success_rate,
                   AVG(latency_ms) as avg_latency,
                   COUNT(*) as runs
            FROM (
                SELECT model, success, latency_ms
                FROM benchmarks
                WHERE task_type = ?
                ORDER BY ts DESC
                LIMIT 200
            )
            GROUP BY model
            HAVING runs >= 2
            """,
            (task_type,),
        ).fetchall()
    except Exception:
        return {}

    scores = {}
    for model, success_rate, avg_latency, runs in rows:
        # Speed factor: penalize slow models (normalize to 0.5-1.0 range)
        speed = max(0.5, 1.0 - (avg_latency or 0) / 120000)  # 120s → 0.5
        # Confidence: more runs → more weight (caps at 1.0 after 10 runs)
        confidence = min(1.0, runs / 10)
        scores[model] = (success_rate or 0) * speed * (0.5 + 0.5 * confidence)
    return scores


def get_summary(n: int = 10) -> str:
    """Format benchmark summary for display."""
    try:
        conn = _get_conn()
        rows = conn.execute(
            """
            SELECT model, task_type,
                   ROUND(AVG(success) * 100, 0) as pct,
                   ROUND(AVG(latency_ms) / 1000.0, 1) as avg_sec,
                   COUNT(*) as runs
            FROM benchmarks
            GROUP BY model, task_type
            ORDER BY runs DESC
            LIMIT ?
            """,
            (n,),
        ).fetchall()
    except Exception:
        return "No benchmark data yet."

    if not rows:
        return "No benchmark data yet."

    lines = ["Model Benchmarks:\n"]
    for model, task, pct, avg_sec, runs in rows:
        lines.append(f"  {model} [{task}]: {pct:.0f}% success, {avg_sec}s avg, {runs} runs")
    return "\n".join(lines)


# ── Token usage tracking ─────────────────────────────────────────────────

def record_tokens(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cost_usd: float = 0.0,
) -> None:
    """Record token usage from a single LLM call."""
    try:
        conn = _get_conn()
        total = prompt_tokens + completion_tokens
        conn.execute(
            "INSERT INTO token_usage (model, prompt_tokens, completion_tokens, total_tokens, cost_usd, ts) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (model, prompt_tokens, completion_tokens, total,
             cost_usd, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    except Exception:
        logger.debug("llm_benchmarks: failed to record tokens", exc_info=True)


def get_token_stats(period: str = "day") -> list[dict]:
    """
    Aggregate token usage by model for a time period.
    period: 'hour', 'day', 'week', 'month', 'quarter', 'year'
    Returns: [{"model": str, "prompt_tokens": int, "completion_tokens": int,
               "total": int, "cost_usd": float, "calls": int}]
    """
    # Map period to hours — avoids f-string interpolation in SQL (injection risk)
    _PERIOD_HOURS = {
        "hour": 1, "day": 24, "week": 168,
        "month": 720, "quarter": 2160, "year": 8760,
    }
    hours = _PERIOD_HOURS.get(period, 24)
    try:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT model, "
            "       SUM(prompt_tokens) as prompt, "
            "       SUM(completion_tokens) as completion, "
            "       SUM(total_tokens) as total, "
            "       SUM(cost_usd) as cost, "
            "       COUNT(*) as calls "
            "FROM token_usage "
            "WHERE ts >= datetime('now', '-' || ? || ' hours') "
            "GROUP BY model "
            "ORDER BY total DESC",
            (str(hours),),
        ).fetchall()
    except Exception:
        return []

    return [
        {"model": m, "prompt_tokens": p or 0, "completion_tokens": c or 0,
         "total": t or 0, "cost_usd": round(cost or 0, 6), "calls": n}
        for m, p, c, t, cost, n in rows
    ]


def format_token_stats(period: str = "day") -> str:
    """Human-readable token usage for Signal display."""
    stats = get_token_stats(period)
    if not stats:
        return f"No token usage recorded ({period})."

    period_labels = {
        "hour": "Last Hour", "day": "Today", "week": "This Week",
        "month": "This Month", "quarter": "This Quarter", "year": "This Year",
    }
    label = period_labels.get(period, period)
    lines = [f"Token Usage ({label}):\n"]
    total_all = 0
    total_cost = 0.0
    for s in stats:
        cost_str = f" (${s['cost_usd']:.4f})" if s["cost_usd"] > 0 else ""
        lines.append(f"  {s['model']}: {s['total']:,} tokens, {s['calls']} calls{cost_str}")
        total_all += s["total"]
        total_cost += s["cost_usd"]
    cost_line = f" (${total_cost:.4f})" if total_cost > 0 else ""
    lines.append(f"\nTotal: {total_all:,} tokens{cost_line}")
    return "\n".join(lines)


# ── Request-level cost tracking ──────────────────────────────────────────────

def record_request_cost(tracker) -> None:
    """Persist aggregated request-level cost data."""
    try:
        conn = _get_conn()
        models = ",".join(sorted(tracker.models_used)) if tracker.models_used else "none"
        crew = getattr(tracker, "crew_name", "") or ""
        conn.execute(
            "INSERT INTO request_costs "
            "(request_id, crew_name, total_prompt, total_completion, total_cost_usd, call_count, models_used, ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (tracker.request_id, crew, tracker.total_prompt_tokens, tracker.total_completion_tokens,
             tracker.total_cost_usd, tracker.call_count, models,
             datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    except Exception:
        logger.debug("llm_benchmarks: failed to record request cost", exc_info=True)


def get_request_cost_stats(period: str = "day") -> dict:
    """Aggregate request-level cost stats for a time period.

    Returns: {"requests": int, "total_cost_usd": float, "avg_cost_usd": float,
              "avg_calls": float, "avg_tokens": float}
    """
    _PERIOD_HOURS_RC = {
        "hour": 1, "day": 24, "week": 168,
        "month": 720, "quarter": 2160, "year": 8760,
    }
    hours = _PERIOD_HOURS_RC.get(period, 24)
    try:
        conn = _get_conn()
        row = conn.execute(
            "SELECT COUNT(*) as requests, "
            "       COALESCE(SUM(total_cost_usd), 0) as total_cost, "
            "       COALESCE(AVG(total_cost_usd), 0) as avg_cost, "
            "       COALESCE(AVG(call_count), 0) as avg_calls, "
            "       COALESCE(AVG(total_prompt + total_completion), 0) as avg_tokens "
            "FROM request_costs "
            "WHERE ts >= datetime('now', '-' || ? || ' hours')",
            (str(hours),),
        ).fetchone()
        if row:
            return {
                "requests": row[0],
                "total_cost_usd": round(row[1], 4),
                "avg_cost_usd": round(row[2], 6),
                "avg_calls": round(row[3], 1),
                "avg_tokens": round(row[4], 0),
            }
    except Exception:
        pass
    return {"requests": 0, "total_cost_usd": 0, "avg_cost_usd": 0, "avg_calls": 0, "avg_tokens": 0}


def get_crew_cost_stats(period: str = "day") -> list[dict]:
    """Aggregate cost stats per crew for a time period.

    Returns: [{"crew": str, "requests": int, "total_cost_usd": float,
               "avg_cost_usd": float, "avg_tokens": float}]
    """
    _PERIOD_HOURS_CC = {
        "hour": 1, "day": 24, "week": 168,
        "month": 720, "quarter": 2160, "year": 8760,
    }
    hours = _PERIOD_HOURS_CC.get(period, 24)
    try:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT crew_name, "
            "       COUNT(*) as requests, "
            "       COALESCE(SUM(total_cost_usd), 0) as total_cost, "
            "       COALESCE(AVG(total_cost_usd), 0) as avg_cost, "
            "       COALESCE(AVG(total_prompt + total_completion), 0) as avg_tokens "
            "FROM request_costs "
            "WHERE ts >= datetime('now', '-' || ? || ' hours') "
            "AND crew_name != '' "
            "GROUP BY crew_name "
            "ORDER BY total_cost DESC",
            (str(hours),),
        ).fetchall()
        return [
            {"crew": r[0], "requests": r[1], "total_cost_usd": round(r[2], 4),
             "avg_cost_usd": round(r[3], 6), "avg_tokens": round(r[4], 0)}
            for r in rows
        ]
    except Exception:
        return []
