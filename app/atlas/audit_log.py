"""
audit_log.py — Structured audit logging for ATLAS external interactions.

Every external API call, code execution, and credential access is logged.
Provides accountability trail for autonomous operations.

IMMUTABLE — infrastructure-level module.
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


def log_external_call(
    agent: str,
    action: str,
    target: str,
    method: str = "",
    credential_used: str = "",
    sandbox_id: str = "",
    result: str = "success",
    response_code: int = 0,
    execution_time_ms: float = 0,
    tokens_consumed: dict | None = None,
    cost_usd: float = 0.0,
    approval: str = "auto",
) -> None:
    """Log an external API call or code execution to PostgreSQL.

    Falls back to structured logging if DB unavailable.
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent": agent,
        "action": action,
        "target": target,
        "method": method,
        "credential_used": credential_used,
        "sandbox_id": sandbox_id,
        "result": result,
        "response_code": response_code,
        "execution_time_ms": execution_time_ms,
        "tokens_consumed": tokens_consumed or {},
        "cost_usd": cost_usd,
        "approval": approval,
    }

    # Try PostgreSQL first
    try:
        _store_to_db(record)
    except Exception:
        pass

    # Always log structurally
    logger.info(f"ATLAS_AUDIT: {agent}/{action} → {target} "
                f"[{result}] {execution_time_ms:.0f}ms")


def _store_to_db(record: dict) -> None:
    """Store audit record in PostgreSQL."""
    try:
        from app.config import get_settings
        import psycopg2
        s = get_settings()
        if not s.mem0_postgres_url:
            return

        conn = psycopg2.connect(s.mem0_postgres_url)
        conn.autocommit = True
        with conn.cursor() as cur:
            # Ensure table exists
            cur.execute("""
                CREATE TABLE IF NOT EXISTS atlas.audit_log (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    agent TEXT NOT NULL,
                    action TEXT NOT NULL,
                    target TEXT NOT NULL,
                    method TEXT DEFAULT '',
                    credential_used TEXT DEFAULT '',
                    sandbox_id TEXT DEFAULT '',
                    result TEXT DEFAULT 'success',
                    response_code INTEGER DEFAULT 0,
                    execution_time_ms REAL DEFAULT 0,
                    tokens_consumed JSONB DEFAULT '{}',
                    cost_usd REAL DEFAULT 0,
                    approval TEXT DEFAULT 'auto'
                )
            """)
            cur.execute("""
                INSERT INTO atlas.audit_log
                (timestamp, agent, action, target, method, credential_used,
                 sandbox_id, result, response_code, execution_time_ms,
                 tokens_consumed, cost_usd, approval)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                record["timestamp"], record["agent"], record["action"],
                record["target"], record["method"], record["credential_used"],
                record["sandbox_id"], record["result"], record["response_code"],
                record["execution_time_ms"], json.dumps(record["tokens_consumed"]),
                record["cost_usd"], record["approval"],
            ))
        conn.close()
    except Exception:
        pass
