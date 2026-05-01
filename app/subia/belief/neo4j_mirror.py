"""Belief → Neo4j projection.

Mirrors PostgreSQL beliefs to a graph store for queries that SQL handles
awkwardly: supersession chains, cross-domain belief relationships.

PostgreSQL is the source of truth. Neo4j is a forward-only, read-optimized
projection — every write is best-effort, every failure is logged but never
surfaced to callers.

Reuses the existing `mem0_neo4j_*` settings for the connection. After the
Mem0 v2 migration, Neo4j is no longer Mem0's; renaming the env vars is a
separate cleanup that would touch Philosophy/Dialectics too.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_driver: Any | None = None
_driver_failed: bool = False
_driver_lock = threading.Lock()


def _get_driver():
    """Lazily build a Neo4j driver. Returns None if Neo4j is unavailable.

    On first successful connect, also creates :Belief indexes.
    Repeated failure is sticky (`_driver_failed=True`) — a process restart
    is required to retry, mirroring the Philosophy/Dialectics pattern.
    """
    global _driver, _driver_failed
    if _driver is not None:
        return _driver
    if _driver_failed:
        return None
    with _driver_lock:
        if _driver is not None:
            return _driver
        if _driver_failed:
            return None
        try:
            from app.config import get_settings
            s = get_settings()
            url = s.mem0_neo4j_url
            user = s.mem0_neo4j_user
            pw = s.mem0_neo4j_password.get_secret_value()
            if not (url and pw):
                _driver_failed = True
                logger.info(
                    "belief.neo4j_mirror: Neo4j not configured — :Belief projection disabled"
                )
                return None
            from neo4j import GraphDatabase
            _driver = GraphDatabase.driver(url, auth=(user, pw))
            _driver.verify_connectivity()
            _ensure_indexes(_driver)
            logger.info("belief.neo4j_mirror: connected; :Belief indexes ensured")
            return _driver
        except Exception as exc:
            _driver_failed = True
            logger.info(
                "belief.neo4j_mirror: Neo4j unavailable — :Belief projection disabled: %s",
                exc,
            )
            return None


def _ensure_indexes(driver) -> None:
    """Create :Belief indexes idempotently. Called once on first driver init."""
    statements = [
        "CREATE INDEX belief_id_idx IF NOT EXISTS FOR (b:Belief) ON (b.belief_id)",
        "CREATE INDEX belief_domain_idx IF NOT EXISTS FOR (b:Belief) ON (b.domain)",
        "CREATE INDEX belief_status_idx IF NOT EXISTS FOR (b:Belief) ON (b.belief_status)",
    ]
    try:
        with driver.session() as session:
            for cypher in statements:
                try:
                    session.run(cypher)
                except Exception as exc:
                    logger.debug(
                        "belief.neo4j_mirror: index '%s…' failed: %s",
                        cypher[:50], exc,
                    )
    except Exception as exc:
        logger.warning("belief.neo4j_mirror: index creation skipped: %s", exc)


def is_available() -> bool:
    """True if Neo4j is reachable for :Belief mirroring."""
    return _get_driver() is not None


def mirror_belief(
    belief_id: str,
    *,
    domain: str | None = None,
    confidence: float | None = None,
    belief_status: str | None = None,
    formed_at: datetime | None = None,
) -> bool:
    """Upsert a :Belief node with the supplied fields. Best-effort.

    Only fields explicitly passed are written. `belief_id` is the MERGE
    key; everything else is optional. This lets callers issue partial
    updates (e.g., status-only after retract) without overwriting fields
    they don't have loaded.

    Returns True on success, False if Neo4j is unavailable or the write
    failed. Callers should not branch on the return value — it's for
    metrics and tests; SQL is always the source of truth.
    """
    if not belief_id:
        return False
    driver = _get_driver()
    if driver is None:
        return False

    set_clauses = ["b.last_synced_at = datetime()"]
    params: dict[str, Any] = {"belief_id": belief_id}

    if domain is not None:
        set_clauses.append("b.domain = $domain")
        params["domain"] = domain
    if confidence is not None:
        set_clauses.append("b.confidence = $confidence")
        params["confidence"] = float(confidence)
    if belief_status is not None:
        set_clauses.append("b.belief_status = $belief_status")
        params["belief_status"] = belief_status

    on_create_clause = ""
    if formed_at is not None:
        on_create_clause = "ON CREATE SET b.formed_at = $formed_at"
        params["formed_at"] = (
            formed_at.isoformat() if hasattr(formed_at, "isoformat")
            else str(formed_at)
        )

    cypher = f"""
    MERGE (b:Belief {{belief_id: $belief_id}})
    {on_create_clause}
    SET {", ".join(set_clauses)}
    """
    try:
        with driver.session() as session:
            session.run(cypher, params)
        return True
    except Exception as exc:
        logger.warning(
            "belief.neo4j_mirror: mirror_belief failed for %s: %s", belief_id, exc
        )
        return False


def mirror_supersession(old_belief_id: str, new_belief_id: str) -> bool:
    """Write a (:Belief)-[:SUPERSEDED_BY]->(:Belief) edge. Best-effort.

    Both endpoint nodes are MERGEd, so this works even if `mirror_belief`
    hasn't run for one or both yet (they'll exist as stub :Belief nodes
    that the next `mirror_belief` call will fill in).
    """
    if not (old_belief_id and new_belief_id):
        return False
    driver = _get_driver()
    if driver is None:
        return False
    cypher = """
    MERGE (old:Belief {belief_id: $old_id})
    MERGE (new:Belief {belief_id: $new_id})
    MERGE (old)-[r:SUPERSEDED_BY]->(new)
    ON CREATE SET r.created_at = datetime()
    """
    try:
        with driver.session() as session:
            session.run(cypher, {"old_id": old_belief_id, "new_id": new_belief_id})
        return True
    except Exception as exc:
        logger.warning(
            "belief.neo4j_mirror: mirror_supersession %s→%s failed: %s",
            old_belief_id, new_belief_id, exc,
        )
        return False


def get_supersession_chain(belief_id: str, max_depth: int = 20) -> list[dict]:
    """Walk the supersession chain forward from `belief_id`.

    Returns the chain oldest-to-newest as a list of node dicts. The starting
    belief is the first element; the terminal (current) belief is the last.
    Returns [] if Neo4j is unavailable or the belief has no successors and
    isn't itself in the graph.

    `max_depth` bounds the traversal — chains are typically short (1–3 hops).
    """
    if not belief_id:
        return []
    driver = _get_driver()
    if driver is None:
        return []
    cypher = f"""
    MATCH path = (start:Belief {{belief_id: $belief_id}})-[:SUPERSEDED_BY*0..{int(max_depth)}]->(end:Belief)
    WHERE NOT (end)-[:SUPERSEDED_BY]->(:Belief)
    RETURN [n IN nodes(path) | {{
        belief_id: n.belief_id,
        domain: n.domain,
        confidence: n.confidence,
        belief_status: n.belief_status
    }}] AS chain
    LIMIT 1
    """
    try:
        with driver.session() as session:
            result = session.run(cypher, {"belief_id": belief_id})
            row = result.single()
            if not row:
                return []
            chain = row.get("chain")
            return list(chain) if chain else []
    except Exception as exc:
        logger.warning(
            "belief.neo4j_mirror: get_supersession_chain failed for %s: %s",
            belief_id, exc,
        )
        return []


def _reset_for_tests() -> None:
    """Test hook to clear module-level driver state."""
    global _driver, _driver_failed
    _driver = None
    _driver_failed = False
