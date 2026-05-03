"""Postgres snapshot of the in-memory ToolRegistry.

The registry is in-memory primary — it works without Postgres. The
snapshot is for cross-process visibility (the React control plane
reads from Postgres so it doesn't need to share the gateway's Python
state).

Schema::

    CREATE TABLE tool_registry (
        name             TEXT PRIMARY KEY,
        capabilities     TEXT[] NOT NULL,
        tier             TEXT NOT NULL,
        lifecycle        TEXT NOT NULL,
        description      TEXT NOT NULL,
        description_hash TEXT NOT NULL,
        workspace_scope  TEXT[] NOT NULL,
        source_module    TEXT NOT NULL,
        is_loadable      BOOLEAN NOT NULL,
        snapshot_ts      TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

Snapshots are full overwrites — drop+rewrite the entire table on each
boot. We don't track historical versions here; the description_hash
+ git history are the audit trail. Drift detection (drift.py) compares
the current decorator-derived hash against the prior snapshot to
spot un-PRed description changes.

Failures here are non-fatal — registry still works without Postgres.
"""
from __future__ import annotations

import logging
from typing import Iterable

from app.tool_registry.types import ToolSpec

logger = logging.getLogger(__name__)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS tool_registry (
    name             TEXT PRIMARY KEY,
    capabilities     TEXT[] NOT NULL,
    tier             TEXT NOT NULL,
    lifecycle        TEXT NOT NULL,
    description      TEXT NOT NULL,
    description_hash TEXT NOT NULL,
    workspace_scope  TEXT[] NOT NULL,
    source_module    TEXT NOT NULL,
    is_loadable      BOOLEAN NOT NULL,
    snapshot_ts      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_tool_registry_tier ON tool_registry(tier);
CREATE INDEX IF NOT EXISTS idx_tool_registry_loadable ON tool_registry(is_loadable);
"""


def _connect():
    """Return (conn, error_str). conn is None on failure."""
    try:
        import psycopg2  # type: ignore
        from app.config import get_settings
    except Exception as exc:
        return None, f"psycopg2/config unavailable: {exc}"

    pg_url = get_settings().mem0_postgres_url
    if not pg_url:
        return None, "mem0_postgres_url not set"

    try:
        return psycopg2.connect(pg_url), None
    except Exception as exc:
        return None, f"connect failed: {exc}"


def ensure_schema() -> bool:
    """Create the table if it doesn't exist. Returns True on success."""
    conn, err = _connect()
    if conn is None:
        logger.info("tool_registry: skipping schema setup (%s)", err)
        return False
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(_SCHEMA)
        return True
    except Exception as exc:
        logger.warning("tool_registry: schema setup failed: %s", exc)
        return False
    finally:
        conn.close()


def snapshot(specs: Iterable[ToolSpec]) -> bool:
    """Overwrite the tool_registry table with the current spec list.

    Returns True on success, False otherwise. Always non-fatal.
    """
    if not ensure_schema():
        return False

    conn, err = _connect()
    if conn is None:
        logger.warning("tool_registry: snapshot skipped (%s)", err)
        return False

    rows = [
        (
            s.name,
            list(s.capabilities),
            s.tier.value,
            s.lifecycle.value,
            s.description,
            s.description_hash,
            list(s.workspace_scope),
            s.source_module,
            s.is_loadable,
        )
        for s in specs
    ]

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM tool_registry")
                if rows:
                    cur.executemany(
                        """
                        INSERT INTO tool_registry
                            (name, capabilities, tier, lifecycle,
                             description, description_hash,
                             workspace_scope, source_module, is_loadable)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        rows,
                    )
        logger.info("tool_registry: snapshot wrote %d rows", len(rows))
        return True
    except Exception as exc:
        logger.warning("tool_registry: snapshot failed: %s", exc)
        return False
    finally:
        conn.close()


def load_snapshot() -> list[dict] | None:
    """Read the current snapshot — used by /api/cp/tools when the
    gateway's in-memory registry isn't accessible (e.g. cross-pod).
    Returns None on connection failure (caller can fall back to
    in-process registry).
    """
    conn, err = _connect()
    if conn is None:
        logger.debug("tool_registry: load_snapshot skipped (%s)", err)
        return None

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT name, capabilities, tier, lifecycle,
                           description, description_hash,
                           workspace_scope, source_module, is_loadable,
                           snapshot_ts
                    FROM tool_registry
                    ORDER BY name
                    """
                )
                rows = cur.fetchall()
        return [
            {
                "name": r[0],
                "capabilities": list(r[1]),
                "tier": r[2],
                "lifecycle": r[3],
                "description": r[4],
                "description_hash": r[5],
                "workspace_scope": list(r[6]),
                "source_module": r[7],
                "is_loadable": r[8],
                "snapshot_ts": r[9].isoformat() if r[9] else None,
            }
            for r in rows
        ]
    except Exception as exc:
        logger.warning("tool_registry: load_snapshot failed: %s", exc)
        return None
    finally:
        conn.close()
