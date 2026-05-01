"""
memory.belief_outbox — Postgres → Neo4j belief reconciler.

Phase F1 of the remediation plan.

Today, ``app/subia/belief/store.py`` writes every belief to Postgres
(system of record) and then fires-and-forgets a ``mirror_belief()`` to
Neo4j. If Neo4j is down, restarting, or the network blips, Postgres has
the belief but the graph does not. Subsequent Cypher queries return
incomplete chains; downstream ``HOT-3`` belief-gated dispatch sees the
graph as more sparse than it really is.

This module ships a periodic reconciler that:

  1. Fetches every belief_id from Postgres.
  2. Asks Neo4j which ``:Belief`` nodes exist.
  3. For the missing ones, calls ``neo4j_mirror.mirror_belief(...)``
     with the canonical fields read from Postgres.
  4. Logs counts (``synced``, ``already_mirrored``, ``failed``).

It is read-only against Postgres and additive against Neo4j (MERGE-only;
never deletes). Safe to run any time, idempotent, eventually-consistent.

Wiring: Phase F1 registers this as an idle-scheduler MEDIUM job. It
runs only when no user task is active. The reconcile budget is small
(typical Postgres → Neo4j drift is single-digit beliefs per day) so
the job completes well within the MEDIUM time cap.

Subsystem boundary: this module does NOT modify ``subia/belief/store.py``
(Tier-3 protected). It treats Postgres + Neo4j as the two halves of the
canonical state and reconciles between them; the application code can
keep its current fire-and-forget semantics, and we close the gap from
the outside.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _fetch_all_postgres_beliefs() -> list[dict[str, Any]]:
    """Read (belief_id, domain, confidence, belief_status, formed_at)
    for every active row in ``beliefs``. Returns [] on connection error.
    """
    try:
        import psycopg2  # type: ignore
        from app.config import get_settings
    except Exception as exc:
        logger.debug("belief_outbox: psycopg2/config unavailable: %s", exc)
        return []

    pg_url = get_settings().mem0_postgres_url
    if not pg_url:
        return []

    try:
        conn = psycopg2.connect(pg_url)
    except Exception as exc:
        logger.warning("belief_outbox: postgres connect failed: %s", exc)
        return []

    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT belief_id::text, domain, confidence, belief_status, formed_at "
                "FROM beliefs"
            )
            rows = cur.fetchall()
    except Exception as exc:
        logger.warning("belief_outbox: belief read failed: %s", exc)
        rows = []
    finally:
        try:
            conn.close()
        except Exception:
            pass

    return [
        {
            "belief_id": r[0],
            "domain": r[1],
            "confidence": float(r[2]) if r[2] is not None else None,
            "belief_status": r[3],
            "formed_at": r[4],
        }
        for r in rows
    ]


def _fetch_existing_neo4j_belief_ids() -> set[str] | None:
    """Return every ``:Belief`` node's ``belief_id`` from Neo4j.

    Returns None when Neo4j is unavailable (caller should skip the run).
    """
    try:
        from app.subia.belief import neo4j_mirror
    except Exception as exc:
        logger.debug("belief_outbox: neo4j_mirror import failed: %s", exc)
        return None

    if not neo4j_mirror.is_available():
        return None

    driver = neo4j_mirror._get_driver()  # noqa: SLF001 — reuse the cached driver
    if driver is None:
        return None

    try:
        with driver.session() as session:
            result = session.run("MATCH (b:Belief) RETURN b.belief_id AS id")
            return {record["id"] for record in result if record["id"]}
    except Exception as exc:
        logger.warning("belief_outbox: neo4j read failed: %s", exc)
        return None


def reconcile_belief_outbox() -> dict[str, int]:
    """Run one reconciliation pass. Returns counts.

    Result format::

        {"postgres_total": int, "neo4j_existing": int,
         "synced": int, "failed": int, "skipped": int}

    Use ``skipped`` to mean "Neo4j unavailable, deferred to next pass".
    Idempotent: re-running on a converged pair is a no-op.
    """
    counts = {
        "postgres_total": 0,
        "neo4j_existing": 0,
        "synced": 0,
        "failed": 0,
        "skipped": 0,
    }

    pg_rows = _fetch_all_postgres_beliefs()
    counts["postgres_total"] = len(pg_rows)
    if not pg_rows:
        return counts

    neo4j_ids = _fetch_existing_neo4j_belief_ids()
    if neo4j_ids is None:
        # Neo4j down — skip rather than error. Next idle pass will retry.
        counts["skipped"] = len(pg_rows)
        logger.info("belief_outbox: neo4j unavailable; deferred %d beliefs", len(pg_rows))
        return counts

    counts["neo4j_existing"] = len(neo4j_ids)

    from app.subia.belief import neo4j_mirror

    for row in pg_rows:
        bid = row["belief_id"]
        if bid in neo4j_ids:
            continue
        ok = neo4j_mirror.mirror_belief(
            bid,
            domain=row.get("domain"),
            confidence=row.get("confidence"),
            belief_status=row.get("belief_status"),
            formed_at=row.get("formed_at"),
        )
        if ok:
            counts["synced"] += 1
        else:
            counts["failed"] += 1

    if counts["synced"] or counts["failed"]:
        logger.info(
            "belief_outbox: reconcile done — pg=%d neo4j=%d synced=%d failed=%d",
            counts["postgres_total"], counts["neo4j_existing"],
            counts["synced"], counts["failed"],
        )
    return counts


# ── Phase F2: ChromaDB belief sync ─────────────────────────────────────
# Mirrors the F1 pattern but for the third store (ChromaDB) so semantic
# queries over the beliefs collection see freshly-formed beliefs.
# Postgres remains the system of record; ChromaDB carries an embedded
# index used by retrieval. Today, beliefs land in Postgres + (best-effort)
# Neo4j but NOT in ChromaDB — the retrieval layer therefore sees stale
# data until a manual refresh runs. This sync closes that gap.
#
# Watermark file: ``workspace/memory/belief_chroma_watermark.json``
# stores the last-synced ``last_updated`` timestamp so each run is
# incremental. On schema drift / parse error the watermark is reset
# to 0 (full re-sync), which is safe because ChromaDB's add() is
# idempotent against the chosen ID.

import json
from pathlib import Path

_WATERMARK_PATH = Path("/app/workspace/memory/belief_chroma_watermark.json")
_BELIEF_COLLECTION = "beliefs"


def _load_watermark() -> float:
    try:
        text = _WATERMARK_PATH.read_text(encoding="utf-8")
        data = json.loads(text)
        return float(data.get("last_synced_ts", 0.0))
    except Exception:
        return 0.0


def _save_watermark(ts: float) -> None:
    try:
        _WATERMARK_PATH.parent.mkdir(parents=True, exist_ok=True)
        _WATERMARK_PATH.write_text(
            json.dumps({"last_synced_ts": float(ts)}, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("belief_outbox: watermark save failed: %s", exc)


def _fetch_unsynced_beliefs(since_ts: float) -> list[dict[str, Any]]:
    """Read beliefs whose ``last_updated`` is strictly newer than since_ts.

    Returns rows with content + metadata for the ChromaDB ``add()``.
    """
    try:
        import psycopg2  # type: ignore
        from app.config import get_settings
    except Exception:
        return []

    pg_url = get_settings().mem0_postgres_url
    if not pg_url:
        return []

    try:
        conn = psycopg2.connect(pg_url)
    except Exception as exc:
        logger.warning("belief_outbox(chroma): postgres connect failed: %s", exc)
        return []

    try:
        with conn.cursor() as cur:
            # ``EXTRACT(EPOCH FROM last_updated)`` gives a float we can
            # compare against the watermark and store back as float.
            cur.execute(
                "SELECT belief_id::text, content, domain, confidence, "
                "       belief_status, EXTRACT(EPOCH FROM last_updated)::float "
                "FROM beliefs "
                "WHERE EXTRACT(EPOCH FROM last_updated)::float > %s "
                "ORDER BY last_updated ASC",
                (since_ts,),
            )
            rows = cur.fetchall()
    except Exception as exc:
        logger.warning("belief_outbox(chroma): belief read failed: %s", exc)
        rows = []
    finally:
        try:
            conn.close()
        except Exception:
            pass

    return [
        {
            "belief_id": r[0],
            "content": r[1] or "",
            "domain": r[2],
            "confidence": float(r[3]) if r[3] is not None else None,
            "belief_status": r[4],
            "last_updated_ts": float(r[5]),
        }
        for r in rows
    ]


def sync_new_beliefs_to_chromadb() -> dict[str, int]:
    """Index every new/updated belief in ChromaDB's ``beliefs`` collection.

    Returns ``{"scanned": int, "indexed": int, "skipped_empty": int,
                "watermark": float}``.

    Idempotent: re-running on a converged pair indexes nothing. The
    watermark advances only when the run completes successfully so a
    crash in the middle leaves the run resumable.
    """
    counts: dict[str, int] = {"scanned": 0, "indexed": 0, "skipped_empty": 0}
    since = _load_watermark()
    rows = _fetch_unsynced_beliefs(since)
    counts["scanned"] = len(rows)
    if not rows:
        counts["watermark"] = since  # type: ignore[assignment]
        return counts

    # Lazy-import the chromadb manager so this module remains importable
    # in environments where ChromaDB is not installed (tests, smoke runs).
    try:
        from app.memory import chromadb_manager
    except Exception as exc:
        logger.debug("belief_outbox(chroma): chromadb_manager import failed: %s", exc)
        return counts

    high_water = since
    for row in rows:
        if not row["content"].strip():
            counts["skipped_empty"] += 1
            continue
        metadata = {
            "belief_id": row["belief_id"],
            "domain": row["domain"] or "",
            "confidence": row["confidence"] if row["confidence"] is not None else -1.0,
            "belief_status": row["belief_status"] or "",
            "source": "belief_outbox",
        }
        try:
            chromadb_manager.store(_BELIEF_COLLECTION, row["content"], metadata)
            counts["indexed"] += 1
        except Exception as exc:
            logger.warning(
                "belief_outbox(chroma): index failed for %s: %s",
                row["belief_id"], exc,
            )
            # Stop advancing the watermark past a failed row so the next
            # run retries it.
            break
        if row["last_updated_ts"] > high_water:
            high_water = row["last_updated_ts"]

    if high_water > since:
        _save_watermark(high_water)

    counts["watermark"] = high_water  # type: ignore[assignment]
    if counts["indexed"]:
        logger.info(
            "belief_outbox(chroma): indexed=%d (scanned=%d skipped_empty=%d watermark=%.1f)",
            counts["indexed"], counts["scanned"], counts["skipped_empty"], high_water,
        )
    return counts
