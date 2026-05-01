"""
philosophy/dialectics.py — Graph-based dialectical argument structure.

Encodes philosophical arguments as a directed graph in Neo4j:

    (Claim) -[:COUNTERED_BY]-> (CounterClaim) -[:SYNTHESIZED_INTO]-> (Synthesis)

This enables retrieval patterns that vector search alone cannot:
  - "Find the counter-argument to X"
  - "Show the dialectical chain for topic Y"
  - "What tensions exist between Stoic and Utilitarian views on Z?"

Uses Neo4j via the existing Mem0 connection config.  Gracefully degrades
to empty results if Neo4j is unavailable.

IMMUTABLE — infrastructure-level module.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ── Neo4j connection ────────────────────────────────────────────────────────

_driver: Any | None = None
_driver_failed: bool = False


def _get_driver():
    """Get a Neo4j driver, reusing the Mem0 connection config."""
    global _driver, _driver_failed
    if _driver is not None:
        return _driver
    if _driver_failed:
        return None

    try:
        from app.config import get_settings
        s = get_settings()

        url = s.mem0_neo4j_url or ""
        user = s.mem0_neo4j_user or "neo4j"
        password = s.mem0_neo4j_password.get_secret_value()

        if not (url and password):
            _driver_failed = True
            logger.info("philosophy.dialectics: Neo4j URL/password not configured — dialectics disabled")
            return None

        import neo4j
        _driver = neo4j.GraphDatabase.driver(url, auth=(user, password))
        _driver.verify_connectivity()
        logger.info("philosophy.dialectics: connected to Neo4j at %s", url)
        return _driver
    except Exception as exc:
        _driver_failed = True
        logger.info("philosophy.dialectics: Neo4j unavailable — dialectics disabled: %s", exc)
        return None


def _node_id(text: str) -> str:
    """Deterministic short hash for a text passage."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ── Public API ──────────────────────────────────────────────────────────────

class DialecticalGraph:
    """Manages philosophical argument structures in Neo4j."""

    def store_argument(
        self,
        claim_text: str,
        counter_claim_text: str,
        synthesis_text: str | None = None,
        source: str = "",
        tradition_a: str = "",
        tradition_b: str = "",
    ) -> bool:
        """Store a claim → counter-claim (→ synthesis) relationship.

        Returns True if stored, False if Neo4j unavailable.
        """
        driver = _get_driver()
        if driver is None:
            return False

        claim_id = _node_id(claim_text)
        counter_id = _node_id(counter_claim_text)

        cypher = """
        MERGE (c:PhilClaim {id: $claim_id})
        SET c.text = $claim_text, c.source = $source, c.tradition = $tradition_a
        MERGE (cc:PhilCounterClaim {id: $counter_id})
        SET cc.text = $counter_text, cc.source = $source, cc.tradition = $tradition_b
        MERGE (c)-[:COUNTERED_BY]->(cc)
        """
        params: dict[str, Any] = {
            "claim_id": claim_id,
            "claim_text": claim_text[:2000],
            "counter_id": counter_id,
            "counter_text": counter_claim_text[:2000],
            "source": source,
            "tradition_a": tradition_a,
            "tradition_b": tradition_b,
        }

        if synthesis_text:
            synth_id = _node_id(synthesis_text)
            cypher += """
            MERGE (s:PhilSynthesis {id: $synth_id})
            SET s.text = $synth_text, s.source = $source
            MERGE (cc)-[:SYNTHESIZED_INTO]->(s)
            """
            params["synth_id"] = synth_id
            params["synth_text"] = synthesis_text[:2000]

        try:
            with driver.session() as session:
                session.run(cypher, params)
            return True
        except Exception as exc:
            logger.warning("dialectics.store_argument failed: %s", exc)
            return False

    def find_counter_arguments(
        self, claim_query: str, n: int = 3
    ) -> list[dict]:
        """Find counter-arguments to a claim.

        Strategy: vector-search the philosophy collection for the claim,
        then follow COUNTERED_BY edges in Neo4j to find counter-claims.
        """
        driver = _get_driver()
        if driver is None:
            return []

        # Step 1: Find matching claims via vector search.
        try:
            from app.philosophy.vectorstore import get_store
            store = get_store()
            matches = store.query(query_text=claim_query, n_results=5)
        except Exception:
            return []

        if not matches:
            return []

        # Build node IDs from matched texts.
        claim_ids = [_node_id(m["text"]) for m in matches]

        # Step 2: Follow graph edges.
        cypher = """
        MATCH (c:PhilClaim)-[:COUNTERED_BY]->(cc:PhilCounterClaim)
        WHERE c.id IN $claim_ids
        OPTIONAL MATCH (cc)-[:SYNTHESIZED_INTO]->(s:PhilSynthesis)
        RETURN cc.text AS counter_text, cc.tradition AS tradition,
               cc.source AS source, s.text AS synthesis_text
        LIMIT $limit
        """
        try:
            with driver.session() as session:
                result = session.run(cypher, {"claim_ids": claim_ids, "limit": n})
                return [
                    {
                        "counter_claim": record["counter_text"],
                        "tradition": record.get("tradition", ""),
                        "source": record.get("source", ""),
                        "synthesis": record.get("synthesis_text"),
                    }
                    for record in result
                ]
        except Exception as exc:
            logger.warning("dialectics.find_counter_arguments failed: %s", exc)
            return []

    def find_dialectical_chain(self, topic: str, n: int = 5) -> list[dict]:
        """Find claim → counter → synthesis chains related to a topic.

        Returns chains as dicts with claim, counter_claim, synthesis.
        """
        driver = _get_driver()
        if driver is None:
            return []

        try:
            from app.philosophy.vectorstore import get_store
            store = get_store()
            matches = store.query(query_text=topic, n_results=5)
        except Exception:
            return []

        if not matches:
            return []

        claim_ids = [_node_id(m["text"]) for m in matches]

        cypher = """
        MATCH (c:PhilClaim)-[:COUNTERED_BY]->(cc:PhilCounterClaim)
        WHERE c.id IN $claim_ids
        OPTIONAL MATCH (cc)-[:SYNTHESIZED_INTO]->(s:PhilSynthesis)
        RETURN c.text AS claim, c.tradition AS claim_tradition,
               cc.text AS counter_claim, cc.tradition AS counter_tradition,
               s.text AS synthesis
        LIMIT $limit
        """
        try:
            with driver.session() as session:
                result = session.run(cypher, {"claim_ids": claim_ids, "limit": n})
                return [
                    {
                        "claim": record["claim"],
                        "claim_tradition": record.get("claim_tradition", ""),
                        "counter_claim": record["counter_claim"],
                        "counter_tradition": record.get("counter_tradition", ""),
                        "synthesis": record.get("synthesis"),
                    }
                    for record in result
                ]
        except Exception as exc:
            logger.warning("dialectics.find_dialectical_chain failed: %s", exc)
            return []


# ── Singleton ───────────────────────────────────────────────────────────────
_graph: DialecticalGraph | None = None


def get_graph() -> DialecticalGraph:
    """Lazy-singleton accessor."""
    global _graph
    if _graph is None:
        _graph = DialecticalGraph()
    return _graph
