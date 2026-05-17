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

PR 1 (2026-05-16) — required vs optional helper families. Until now,
every helper (``execute`` / ``execute_one`` / ``execute_scalar``)
swallowed all failures and returned ``None`` / ``[]``, which meant a
missing table or a dropped pool looked exactly like "query returned
no rows". The new ``execute_required`` / ``execute_one_required`` /
``execute_scalar_required`` family raises on any failure — use for
correctness-critical operations (schema CREATE, audit log writes,
governance state transitions). The existing optional helpers are
refactored to share the same private inner loop, so the two families
cannot drift apart. A new ``DBUnavailable`` exception covers the
specific case of ``get_pool() → None`` (not configured, breaker open).

Pool diagnostics: see ``get_pool_diagnostics()``. Thread-safe counters
that surface concurrent borrows, peak borrows, and per-failure-kind
counts. Hooked into both families so we can finally see what's behind
the "connection pool exhausted" volume in errors.jsonl — observational
only, never changes pool behavior.
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


class DBUnavailable(Exception):
    """Control-plane DB pool is unavailable.

    Raised by the ``*_required`` family when ``get_pool()`` returns
    ``None`` — the pool is not configured, the circuit breaker is open,
    or the pool was never successfully created. Distinct from
    ``psycopg2.pool.PoolError`` (which is raised when the pool exists
    but a connection cannot currently be borrowed).
    """


# ── Pool diagnostics ──────────────────────────────────────────────────
#
# Lightweight thread-safe counters. The goal is to finally answer the
# question "what is happening when 'connection pool exhausted' fires
# tens of thousands of times" without changing pool behavior. We track:
#   * total acquires, slow acquires (> _SLOW_ACQUIRE_THRESHOLD_S)
#   * concurrent borrows (current + peak) — surfaces leak signatures
#   * failures by kind — distinguishes pool_exhausted from
#     connection_error from pool_unavailable
#
# Counters are intentionally cheap (single lock, integer increments) so
# the hot path overhead is negligible. Diagnostics are observational —
# they NEVER change query routing or retry behavior.

_diag_lock = threading.Lock()
_diag: dict[str, float | int] = {
    "acquires_total": 0,
    "acquires_slow": 0,
    "failures_pool_unavailable": 0,
    "failures_pool_exhausted": 0,
    "failures_pool_other": 0,
    "failures_connection_error": 0,
    "current_borrows": 0,
    "peak_borrows": 0,
    "last_exhaust_ts": 0.0,
    "last_slow_acquire_ms": 0,
}
_SLOW_ACQUIRE_THRESHOLD_S = 0.5


def get_pool_diagnostics() -> dict:
    """Read pool diagnostics. Thread-safe snapshot."""
    with _diag_lock:
        return dict(_diag)


def reset_pool_diagnostics() -> None:
    """Test-only: zero the counters."""
    with _diag_lock:
        for k in _diag:
            _diag[k] = 0


def _diag_record_acquire(duration_s: float) -> None:
    with _diag_lock:
        _diag["acquires_total"] += 1
        if duration_s > _SLOW_ACQUIRE_THRESHOLD_S:
            _diag["acquires_slow"] += 1
            _diag["last_slow_acquire_ms"] = int(duration_s * 1000)
        _diag["current_borrows"] += 1
        if _diag["current_borrows"] > _diag["peak_borrows"]:
            _diag["peak_borrows"] = _diag["current_borrows"]


def _diag_record_release() -> None:
    with _diag_lock:
        if _diag["current_borrows"] > 0:
            _diag["current_borrows"] -= 1


def _diag_record_failure(kind: str) -> None:
    """kind: 'pool_unavailable' | 'pool_exhausted' | 'pool_other' | 'connection_error'"""
    key = f"failures_{kind}"
    with _diag_lock:
        if key in _diag:
            _diag[key] += 1
        if kind == "pool_exhausted":
            _diag["last_exhaust_ts"] = time.time()

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

# ── Private execution helpers ────────────────────────────────────────


def _checkout(p: pg_pool.ThreadedConnectionPool):
    """Borrow a connection, retrying once on stale-connection InterfaceError.

    Records acquire timing into the diagnostics counters. Caller is
    responsible for calling ``p.putconn()`` AND ``_diag_record_release()``.

    Raises ``psycopg2.pool.PoolError`` if the pool is exhausted; records
    the failure kind into diagnostics before re-raising.
    """
    t0 = time.monotonic()
    try:
        conn = p.getconn()
    except pg_pool.PoolError as exc:
        kind = "pool_exhausted" if "exhausted" in str(exc).lower() else "pool_other"
        _diag_record_failure(kind)
        raise
    try:
        conn.autocommit = True
    except (psycopg2.InterfaceError, psycopg2.OperationalError):
        # Stale connection — close and get fresh one
        try:
            p.putconn(conn, close=True)
        except Exception:
            pass
        try:
            conn = p.getconn()
        except pg_pool.PoolError as exc:
            kind = "pool_exhausted" if "exhausted" in str(exc).lower() else "pool_other"
            _diag_record_failure(kind)
            raise
        conn.autocommit = True
    _diag_record_acquire(time.monotonic() - t0)
    return conn


def _run_query(conn, query: str, params: tuple, fetch: bool) -> list:
    """Run the query on an already-borrowed connection.

    Returns dict-row results when fetch=True; empty list when fetch=False.
    Lets psycopg2 errors propagate to the caller.
    """
    with conn.cursor() as cur:
        cur.execute(query, params)
        if fetch:
            cols = [d[0] for d in cur.description] if cur.description else []
            return [dict(zip(cols, row)) for row in cur.fetchall()]
        return []


# ── Required helpers (raise on failure) ──────────────────────────────


def execute_required(
    query: str, params: tuple = (), fetch: bool = False,
) -> list:
    """Execute a query that MUST succeed. Raises on any failure.

    Use for operations where silent failure would corrupt state:
    schema CREATE TABLE, audit log writes, governance request
    transitions, error_anomalies inserts, etc.

    Returns:
        list of dict rows when fetch=True (possibly empty)
        empty list when fetch=False

    Raises:
        DBUnavailable: ``get_pool()`` returned None (pool not configured
            or circuit breaker open)
        psycopg2.pool.PoolError: pool exhausted or otherwise unusable
        psycopg2.InterfaceError / OperationalError: connection failure
            (pool is reset as a side effect)
        psycopg2.Error: SQL error (missing table/column, syntax, etc.)
    """
    p = get_pool()
    if not p:
        _diag_record_failure("pool_unavailable")
        raise DBUnavailable("control_plane pool is not available")
    conn = None
    try:
        conn = _checkout(p)
        return _run_query(conn, query, params, fetch)
    except (psycopg2.InterfaceError, psycopg2.OperationalError):
        _diag_record_failure("connection_error")
        _reset_pool()
        raise
    finally:
        if conn:
            try:
                p.putconn(conn)
                _diag_record_release()
            except Exception:
                pass


def execute_one_required(query: str, params: tuple = ()) -> dict | None:
    """Required variant of ``execute_one``. Returns first row or None on empty result."""
    rows = execute_required(query, params, fetch=True)
    return rows[0] if rows else None


def execute_scalar_required(query: str, params: tuple = ()):
    """Required variant of ``execute_scalar``. Returns first column of first row, or None on empty result."""
    rows = execute_required(query, params, fetch=True)
    if not rows:
        return None
    row = rows[0]
    return next(iter(row.values())) if row else None


# ── Optional helpers (return None on failure) ────────────────────────


def execute(query: str, params: tuple = (), fetch: bool = False) -> list | None:
    """Execute a query with OPTIONAL semantics. Returns None on failure.

    Use for observational / non-critical writes where silent failure is
    acceptable. For correctness-critical operations use ``execute_required()``.
    """
    try:
        return execute_required(query, params, fetch)
    except DBUnavailable:
        return None
    except pg_pool.PoolError as e:
        logger.warning(f"control_plane: pool error: {e}")
        return None
    except (psycopg2.InterfaceError, psycopg2.OperationalError) as e:
        logger.warning(f"control_plane: connection error, resetting pool: {e}")
        return None
    except Exception as e:
        logger.error(f"control_plane SQL error: {e}")
        return None


def execute_one(query: str, params: tuple = ()) -> dict | None:
    """Optional variant. Execute and return a single row, or None on failure/empty."""
    rows = execute(query, params, fetch=True)
    return rows[0] if rows else None


def execute_scalar(query: str, params: tuple = ()):
    """Optional variant. Execute and return a single value, or None on failure/empty."""
    try:
        return execute_scalar_required(query, params)
    except DBUnavailable:
        return None
    except pg_pool.PoolError as e:
        logger.warning(f"control_plane: pool error in scalar: {e}")
        return None
    except (psycopg2.InterfaceError, psycopg2.OperationalError) as e:
        logger.warning(f"control_plane: connection error in scalar, resetting pool: {e}")
        return None
    except Exception as e:
        logger.error(f"control_plane SQL error: {e}")
        return None
