"""Database connection pool for the control_plane schema.

Reuses the existing Mem0 PostgreSQL instance. Thread-safe singleton pool.
"""
import logging
import threading
from typing import Optional

import psycopg2
from psycopg2 import pool as pg_pool

logger = logging.getLogger(__name__)

_pool: Optional[pg_pool.ThreadedConnectionPool] = None
_pool_lock = threading.Lock()


def get_pool() -> Optional[pg_pool.ThreadedConnectionPool]:
    """Get or create the connection pool. Thread-safe singleton."""
    global _pool
    if _pool is not None:
        return _pool
    with _pool_lock:
        if _pool is not None:
            return _pool
        try:
            from app.config import get_settings
            s = get_settings()
            if not s.mem0_postgres_url:
                logger.warning("control_plane: no postgres URL configured")
                return None
            _pool = pg_pool.ThreadedConnectionPool(
                minconn=1, maxconn=4,
                dsn=s.mem0_postgres_url,
            )
            logger.info("control_plane: connection pool created")
            return _pool
        except Exception as e:
            logger.warning(f"control_plane: pool creation failed: {e}")
            return None


def execute(query: str, params: tuple = (), fetch: bool = False) -> list | None:
    """Execute a query using the pool. Returns rows if fetch=True."""
    p = get_pool()
    if not p:
        return None
    conn = p.getconn()
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(query, params)
            if fetch:
                cols = [d[0] for d in cur.description] if cur.description else []
                return [dict(zip(cols, row)) for row in cur.fetchall()]
            return []
    except Exception as e:
        logger.error(f"control_plane SQL error: {e}")
        return None
    finally:
        p.putconn(conn)


def execute_one(query: str, params: tuple = ()) -> dict | None:
    """Execute and return a single row."""
    rows = execute(query, params, fetch=True)
    return rows[0] if rows else None


def execute_scalar(query: str, params: tuple = ()):
    """Execute and return a single value."""
    p = get_pool()
    if not p:
        return None
    conn = p.getconn()
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(query, params)
            row = cur.fetchone()
            return row[0] if row else None
    except Exception as e:
        logger.error(f"control_plane SQL error: {e}")
        return None
    finally:
        p.putconn(conn)
