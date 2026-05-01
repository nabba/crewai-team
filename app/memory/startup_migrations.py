"""startup_migrations.py — idempotent speed-upgrade indexes applied at startup.

HNSW indexes on pgvector tables (Mem0 infrastructure).

All operations use IF NOT EXISTS / CREATE INDEX IF NOT EXISTS so re-running
is a no-op. Failures are logged but never raise — a startup hiccup must not
take down the gateway.

Call from app/main.py lifespan(). Runs in a background thread so it doesn't
block gateway startup.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# ── pgvector HNSW indexes ──────────────────────────────────────────────────

# SQL is inlined so the gateway image doesn't need to ship migrations/.
# Mirror of migrations/018_speed_indexes.sql. Keep them in sync if either changes.
_PGVECTOR_HNSW_SQL = """
SET maintenance_work_mem = '256MB';

CREATE INDEX IF NOT EXISTS agent_experiences_emb_hnsw
  ON agent_experiences USING hnsw (context_embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS beliefs_emb_hnsw
  ON beliefs USING hnsw (content_embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'workspace_items') THEN
    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_name = 'workspace_items' AND column_name = 'content_embedding') THEN
      EXECUTE 'CREATE INDEX IF NOT EXISTS workspace_items_emb_hnsw
               ON workspace_items USING hnsw (content_embedding vector_cosine_ops)
               WITH (m = 16, ef_construction = 64)';
    END IF;
  END IF;
END $$;
"""


def _apply_pgvector_indexes() -> None:
    """Apply HNSW indexes to pgvector tables. Safe to call multiple times.

    Runs inlined _PGVECTOR_HNSW_SQL (mirror of migrations/018_speed_indexes.sql).
    Failures are logged but never raise — startup must continue.
    """
    try:
        from app.config import get_settings
        s = get_settings()
        if not s.mem0_enabled:
            logger.debug("startup_migrations: Mem0 disabled, skipping pgvector indexes")
            return
        url = s.mem0_postgres_url
        if not url:
            logger.debug("startup_migrations: no mem0_postgres_url, skipping")
            return
    except Exception as exc:
        logger.debug(f"startup_migrations: settings unavailable ({exc}), skipping pgvector")
        return

    try:
        import psycopg2
        conn = psycopg2.connect(url, connect_timeout=10)
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                cur.execute(_PGVECTOR_HNSW_SQL)
            logger.info("startup_migrations: pgvector HNSW indexes applied (018)")
        finally:
            conn.close()
    except Exception as exc:
        # Common reasons: Postgres not yet healthy, tables not yet created
        # (migrations 012/015 haven't run yet). Log and move on — the next
        # gateway restart will retry.
        logger.warning(f"startup_migrations: pgvector HNSW apply skipped/failed: {exc}")


def apply_all() -> None:
    """Apply migrations. Safe to call multiple times."""
    if os.environ.get("SKIP_STARTUP_MIGRATIONS", "0") == "1":
        logger.info("startup_migrations: SKIP_STARTUP_MIGRATIONS=1 — skipping")
        return
    _apply_pgvector_indexes()
