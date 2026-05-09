"""Database connection pool for the control_plane schema.

Reuses the existing Mem0 PostgreSQL instance. Thread-safe singleton pool.

Phase D #1 (2026-05-09) — startup-path circuit breaker. The original
``get_pool()`` retried on every call without backoff; with a network-
unreachable PG (DNS failure on laptop dev, slow boot in K8s, transient
connect storm under load) every cold-path call would spend the connect
timeout each time. Now the first-creation block has a bounded
exponential backoff (3 attempts, 1/3/9 s); after the third consecutive
failure the circuit opens for ``_STARTUP_BREAKER_OPEN_S`` (default
60 s), during which ``get_pool()`` returns None immediately. A single
Signal alert fires on first open. The breaker auto-resets on the next
successful pool creation.

Connect-timeout: ``CONTROL_PLANE_CONNECT_TIMEOUT_S`` (default 8 s) —
appended to the DSN if not already specified, so a stuck DNS/network
lookup can't block the gateway boot indefinitely.
"""
import logging
import os
import threading
import time
from typing import Optional

import psycopg2
from psycopg2 import pool as pg_pool

logger = logging.getLogger(__name__)

_pool: pg_pool.ThreadedConnectionPool | None = None
_pool_lock = threading.Lock()

# ── Startup circuit breaker (Phase D #1) ─────────────────────────────────
_STARTUP_RETRY_ATTEMPTS = 3
_STARTUP_RETRY_BACKOFF_S = (1, 3, 9)
_STARTUP_BREAKER_OPEN_S = 60.0

_breaker_open_until: float = 0.0
_breaker_alert_sent: bool = False


def _circuit_open() -> bool:
    """True when the breaker is currently OPEN (stop retrying)."""
    return time.monotonic() < _breaker_open_until


def _open_breaker(reason: str) -> None:
    """Mark the breaker open + Signal alert (best-effort, once per open)."""
    global _breaker_open_until, _breaker_alert_sent
    _breaker_open_until = time.monotonic() + _STARTUP_BREAKER_OPEN_S
    if _breaker_alert_sent:
        return
    _breaker_alert_sent = True
    try:
        from app.healing.handlers._common import send_signal_alert
        send_signal_alert(
            f"🛑 control_plane: PG pool creation failed {_STARTUP_RETRY_ATTEMPTS}× — "
            f"breaker open for {int(_STARTUP_BREAKER_OPEN_S)}s. Last error: "
            f"{reason[:200]}",
            tag="control_plane_pg_breaker",
        )
    except Exception:
        # No Signal in laptop dev — fail silently.
        pass


def _close_breaker() -> None:
    """Reset breaker after a successful pool creation."""
    global _breaker_open_until, _breaker_alert_sent
    _breaker_open_until = 0.0
    _breaker_alert_sent = False


def _connect_timeout_s() -> int:
    raw = os.environ.get("CONTROL_PLANE_CONNECT_TIMEOUT_S", "8").strip()
    try:
        return max(1, min(60, int(raw)))
    except ValueError:
        return 8


def _dsn_with_timeout(dsn: str) -> str:
    """Ensure ``connect_timeout`` is set in the DSN. Idempotent."""
    timeout = _connect_timeout_s()
    if "connect_timeout=" in dsn:
        return dsn
    sep = "&" if "?" in dsn else "?"
    return f"{dsn}{sep}connect_timeout={timeout}"

def get_pool() -> pg_pool.ThreadedConnectionPool | None:
    """Get or create the connection pool. Thread-safe singleton.

    Phase D #1: bounded retry with backoff on first creation; circuit
    breaker after consecutive failures.
    """
    global _pool
    if _pool is not None:
        return _pool
    if _circuit_open():
        # Don't even try — let callers fall through to degraded mode.
        return None
    with _pool_lock:
        if _pool is not None:
            return _pool
        if _circuit_open():
            return None
        return _create_pool_with_retry()


def _create_pool_with_retry() -> Optional[pg_pool.ThreadedConnectionPool]:
    """Try up to ``_STARTUP_RETRY_ATTEMPTS`` times with exponential backoff.
    Caller holds ``_pool_lock``."""
    global _pool
    try:
        from app.config import get_settings
        s = get_settings()
    except Exception as exc:
        logger.warning(f"control_plane: settings unavailable: {exc}")
        return None
    if not s.mem0_postgres_url:
        logger.warning("control_plane: no postgres URL configured")
        return None

    dsn = _dsn_with_timeout(s.mem0_postgres_url)
    # Pool sizing notes (2026-04-24): research_orchestrator fan-out
    # requires headroom; bumped from 4→24 after PSP-tender outage.
    _maxconn = int(os.environ.get("CONTROL_PLANE_POOL_MAX", "24"))

    last_err: Exception | None = None
    for attempt in range(_STARTUP_RETRY_ATTEMPTS):
        try:
            _pool = pg_pool.ThreadedConnectionPool(
                minconn=2, maxconn=_maxconn, dsn=dsn,
            )
            logger.info(
                "control_plane: connection pool created "
                "(minconn=2, maxconn=%d, attempt=%d)", _maxconn, attempt + 1,
            )
            _close_breaker()
            return _pool
        except Exception as exc:
            last_err = exc
            logger.warning(
                "control_plane: pool creation failed (attempt %d/%d): %s",
                attempt + 1, _STARTUP_RETRY_ATTEMPTS, exc,
            )
            if attempt < _STARTUP_RETRY_ATTEMPTS - 1:
                time.sleep(_STARTUP_RETRY_BACKOFF_S[attempt])

    # All attempts failed — open the breaker.
    err_str = str(last_err) if last_err else "unknown"
    _open_breaker(err_str)
    return None

def _reset_pool() -> None:
    """Destroy and recreate the pool on persistent connection failures."""
    global _pool
    with _pool_lock:
        if _pool:
            try:
                _pool.closeall()
            except Exception:
                pass
            _pool = None

def execute(query: str, params: tuple = (), fetch: bool = False) -> list | None:
    """Execute a query using the pool. Returns rows if fetch=True.

    Validates connection health before use. Resets pool on persistent failures.
    """
    p = get_pool()
    if not p:
        return None
    conn = None
    try:
        conn = p.getconn()
        # Validate connection is alive before use
        try:
            conn.autocommit = True
        except (psycopg2.InterfaceError, psycopg2.OperationalError):
            # Stale connection — close and get fresh one
            try:
                p.putconn(conn, close=True)
            except Exception:
                pass
            conn = p.getconn()
            conn.autocommit = True

        with conn.cursor() as cur:
            cur.execute(query, params)
            if fetch:
                cols = [d[0] for d in cur.description] if cur.description else []
                return [dict(zip(cols, row)) for row in cur.fetchall()]
            return []
    except (psycopg2.InterfaceError, psycopg2.OperationalError) as e:
        logger.warning(f"control_plane: connection error, resetting pool: {e}")
        _reset_pool()
        return None
    except Exception as e:
        logger.error(f"control_plane SQL error: {e}")
        return None
    finally:
        if conn:
            try:
                p.putconn(conn)
            except Exception:
                pass

def execute_one(query: str, params: tuple = ()) -> dict | None:
    """Execute and return a single row."""
    rows = execute(query, params, fetch=True)
    return rows[0] if rows else None

def execute_scalar(query: str, params: tuple = ()):
    """Execute and return a single value."""
    p = get_pool()
    if not p:
        return None
    conn = None
    try:
        conn = p.getconn()
        try:
            conn.autocommit = True
        except (psycopg2.InterfaceError, psycopg2.OperationalError):
            try:
                p.putconn(conn, close=True)
            except Exception:
                pass
            conn = p.getconn()
            conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(query, params)
            row = cur.fetchone()
            return row[0] if row else None
    except (psycopg2.InterfaceError, psycopg2.OperationalError) as e:
        logger.warning(f"control_plane: connection error in scalar, resetting pool: {e}")
        _reset_pool()
        return None
    except Exception as e:
        logger.error(f"control_plane SQL error: {e}")
        return None
    finally:
        if conn:
            try:
                p.putconn(conn)
            except Exception:
                pass
