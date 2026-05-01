"""
belief_store.py — HOT-3: Formal belief store with metacognitive update.

Implements Butlin et al. (2025) HOT-3: a general belief-formation system with
action selection guided by beliefs and metacognitive updating when monitoring
flags issues.

Beliefs are PostgreSQL-backed with pgvector for semantic search. Unlike the
existing ChromaDB belief_state.py (which tracks agent working states), this
stores inspectable, mutable EPISTEMIC beliefs about the world, task strategies,
user preferences, and agent capabilities.

DGM Safety: HUMANIST_CONSTITUTION constraints are checked during action selection
and cannot be overridden by belief state. The constitution sits above beliefs.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ── Domain constants ────────────────────────────────────────────────────────

VALID_DOMAINS = frozenset({
    "task_strategy", "user_model", "self_model",
    "world_model", "agent_capability", "environment",
})

VALID_STATUSES = frozenset({"ACTIVE", "SUSPENDED", "RETRACTED", "SUPERSEDED"})

@dataclass
class Belief:
    """An inspectable, mutable epistemic belief."""
    belief_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    content_embedding: list[float] = field(default_factory=list)
    domain: str = "world_model"
    confidence: float = 0.5
    evidence_sources: list[dict] = field(default_factory=list)
    formed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_validated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metacognitive_flags: list[dict] = field(default_factory=list)
    update_history: list[dict] = field(default_factory=list)
    belief_status: str = "ACTIVE"
    superseded_by: str | None = None

    def to_dict(self) -> dict:
        return {
            "belief_id": self.belief_id,
            "content": self.content[:300],
            "domain": self.domain,
            "confidence": round(self.confidence, 3),
            "belief_status": self.belief_status,
            "evidence_count": len(self.evidence_sources),
            "formed_at": self.formed_at.isoformat() if self.formed_at else "",
            "last_validated": self.last_validated.isoformat() if self.last_validated else "",
            "updates": len(self.update_history),
        }

@dataclass
class MetacognitiveUpdate:
    """Record of a belief update from metacognitive monitoring."""
    update_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_belief_id: str = ""
    trigger: str = "COGITO_CYCLE"      # COGITO_CYCLE | PREDICTION_ERROR | BEHAVIORAL_MISMATCH | BROADCAST_REACTION | EXTERNAL_EVIDENCE
    observation: str = ""
    action_taken: str = "NO_CHANGE"    # CONFIDENCE_ADJUSTED | BELIEF_SUSPENDED | BELIEF_RETRACTED | BELIEF_REVISED | NO_CHANGE
    old_confidence: float = 0.0
    new_confidence: float = 0.0
    reasoning: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "update_id": self.update_id,
            "source_belief_id": self.source_belief_id,
            "trigger": self.trigger,
            "action_taken": self.action_taken,
            "old_confidence": round(self.old_confidence, 3),
            "new_confidence": round(self.new_confidence, 3),
            "reasoning": self.reasoning[:200],
        }

class BeliefStore:
    """PostgreSQL-backed belief store with semantic search and metacognitive updating."""

    def __init__(self):
        from app.consciousness.config import load_config
        self._config = load_config()

    def form_belief(self, content: str, domain: str, confidence: float = 0.5,
                    evidence: list[dict] = None,
                    min_evidence: int = 1, dedup_threshold: float = 0.85,
                    ) -> Belief | None:
        """Create a new belief from evidence, with dedup + evidence threshold.

        Anti-proliferation measures:
        1. Similarity dedup: if existing belief in same domain has cosine > dedup_threshold,
           merge evidence and update confidence instead of creating new belief.
        2. Evidence threshold: require at least min_evidence sources OR confidence >= 0.5.

        Returns Belief (new or updated) or None on failure.
        """
        if domain not in VALID_DOMAINS:
            logger.warning(f"belief_store: invalid domain '{domain}'")
            return None

        evidence = evidence or []

        # Evidence threshold: reject beliefs with insufficient grounding
        if len(evidence) < min_evidence and confidence < 0.5:
            logger.debug(f"belief_store: rejected belief [{domain}] — "
                         f"evidence={len(evidence)} < {min_evidence} and conf={confidence:.2f} < 0.5")
            return None

        try:
            from app.memory.chromadb_manager import embed
            embedding = embed(content[:500])
        except Exception:
            embedding = []

        # Similarity dedup: check for existing similar belief in same domain
        if embedding:
            try:
                existing = self.query_relevant(content[:500], domain=domain, n=1,
                                               min_confidence=0.0)
                if existing:
                    from app.subia.scene.buffer import _cosine_sim
                    sim = _cosine_sim(embedding, existing[0].content_embedding)
                    if sim >= dedup_threshold:
                        # Merge: update existing belief instead of creating new
                        merged = existing[0]
                        # Combine evidence (dedup by source)
                        existing_sources = {str(e): e for e in (merged.evidence_sources or [])}
                        for e in evidence:
                            existing_sources[str(e)] = e
                        merged.evidence_sources = list(existing_sources.values())
                        # Nudge confidence toward new value (weighted average)
                        merged.confidence = round(
                            min(1.0, merged.confidence * 0.7 + confidence * 0.3), 3
                        )
                        self._persist_belief(merged)
                        logger.info(
                            f"belief_store: merged into existing [{domain}] "
                            f"sim={sim:.2f} conf={merged.confidence:.2f}: {content[:40]}"
                        )
                        return merged
            except Exception:
                pass  # Dedup check failed — proceed with new belief

        belief = Belief(
            content=content,
            content_embedding=embedding,
            domain=domain,
            confidence=max(0.0, min(1.0, confidence)),
            evidence_sources=evidence,
        )

        self._persist_belief(belief)
        logger.info(f"belief_store: formed belief [{domain}] conf={confidence:.2f}: {content[:60]}")
        return belief

    def query_relevant(self, query: str, domain: str = None, n: int = 5,
                       min_confidence: float = 0.0) -> list[Belief]:
        """Semantic search for beliefs relevant to a query."""
        try:
            from app.memory.chromadb_manager import embed
            from app.control_plane.db import execute

            q_emb = embed(query[:500])

            sql = """
                SELECT belief_id, content, domain, confidence, evidence_sources,
                       formed_at, last_validated, last_updated, belief_status, update_history,
                       1 - (content_embedding <=> %s::vector) AS similarity
                FROM beliefs
                WHERE belief_status = 'ACTIVE'
                  AND confidence >= %s
            """
            params = [q_emb, min_confidence]

            if domain:
                sql += " AND domain = %s"
                params.append(domain)

            sql += " ORDER BY content_embedding <=> %s::vector LIMIT %s"
            params.extend([q_emb, n])

            rows = execute(sql, tuple(params), fetch=True)
            if not rows:
                return []

            beliefs = []
            for r in (rows or []):
                # Apply confidence decay for unvalidated beliefs
                raw_conf = r.get("confidence", 0.5) if isinstance(r, dict) else 0.5
                last_val = r.get("last_validated") if isinstance(r, dict) else None
                effective_conf = self._apply_confidence_decay(raw_conf, last_val)

                beliefs.append(Belief(
                    belief_id=str(r.get("belief_id", "")) if isinstance(r, dict) else "",
                    content=r.get("content", "") if isinstance(r, dict) else "",
                    domain=r.get("domain", "") if isinstance(r, dict) else "",
                    confidence=effective_conf,
                    evidence_sources=r.get("evidence_sources", []) if isinstance(r, dict) else [],
                    belief_status=r.get("belief_status", "ACTIVE") if isinstance(r, dict) else "ACTIVE",
                ))
            return beliefs

        except Exception as e:
            logger.debug(f"belief_store: query failed: {e}")
            return []

    def update_confidence(self, belief_id: str, delta: float,
                          trigger: str, reasoning: str) -> MetacognitiveUpdate | None:
        """Adjust belief confidence. Asymmetric: decay faster than growth."""
        try:
            from app.control_plane.db import execute

            rows = execute(
                "SELECT confidence FROM beliefs WHERE belief_id = %s",
                (belief_id,), fetch=True,
            )
            if not rows:
                return None

            old_conf = rows[0].get("confidence", 0.5) if isinstance(rows[0], dict) else 0.5

            # Asymmetric: disconfirmation at full rate, confirmation at reduced rate
            if delta < 0:
                effective_delta = delta  # Full penalty
            else:
                effective_delta = delta * (self._config.confirmation_rate / self._config.disconfirmation_rate)

            new_conf = max(0.0, min(1.0, old_conf + effective_delta))

            # Determine action
            action = "CONFIDENCE_ADJUSTED"
            new_status = "ACTIVE"
            if new_conf < self._config.belief_suspension_threshold:
                action = "BELIEF_SUSPENDED"
                new_status = "SUSPENDED"

            execute(
                """
                UPDATE beliefs
                SET confidence = %s, last_updated = NOW(), belief_status = %s
                WHERE belief_id = %s
                """,
                (new_conf, new_status, belief_id),
            )

            # Mirror confidence + status changes to Neo4j (best-effort).
            try:
                from app.subia.belief import neo4j_mirror
                neo4j_mirror.mirror_belief(
                    belief_id,
                    confidence=new_conf,
                    belief_status=new_status,
                )
            except Exception:
                logger.debug("belief_store: neo4j mirror failed", exc_info=True)

            update = MetacognitiveUpdate(
                source_belief_id=belief_id,
                trigger=trigger,
                observation=reasoning[:500],
                action_taken=action,
                old_confidence=old_conf,
                new_confidence=new_conf,
                reasoning=reasoning,
            )
            self._persist_update(update)
            return update

        except Exception as e:
            logger.debug(f"belief_store: update_confidence failed: {e}")
            return None

    def validate_belief(self, belief_id: str) -> None:
        """Mark a belief as recently validated (resets confidence decay clock)."""
        try:
            from app.control_plane.db import execute
            execute(
                "UPDATE beliefs SET last_validated = NOW() WHERE belief_id = %s",
                (belief_id,),
            )
        except Exception:
            pass

    def get_oldest_unvalidated(self, n: int = 3) -> list[Belief]:
        """Get N oldest unvalidated ACTIVE beliefs for mandatory review."""
        try:
            from app.control_plane.db import execute
            rows = execute(
                """
                SELECT belief_id, content, domain, confidence, last_validated
                FROM beliefs
                WHERE belief_status = 'ACTIVE'
                ORDER BY last_validated ASC
                LIMIT %s
                """,
                (n,), fetch=True,
            )
            return [
                Belief(
                    belief_id=str(r.get("belief_id", "")),
                    content=r.get("content", ""),
                    domain=r.get("domain", ""),
                    confidence=r.get("confidence", 0.5),
                    last_validated=r.get("last_validated"),
                ) for r in (rows or []) if isinstance(r, dict)
            ]
        except Exception:
            return []

    def suspend_belief(self, belief_id: str, reason: str) -> MetacognitiveUpdate | None:
        """Suspend a belief (confidence below threshold)."""
        return self.update_confidence(
            belief_id, delta=-1.0, trigger="COGITO_CYCLE",
            reasoning=f"Suspended: {reason}",
        )

    def retract_belief(self, belief_id: str, reason: str,
                       replacement_id: str = None) -> MetacognitiveUpdate | None:
        """Retract a belief (strong contradicting evidence)."""
        try:
            from app.control_plane.db import execute
            execute(
                """
                UPDATE beliefs
                SET belief_status = 'RETRACTED', superseded_by = %s, last_updated = NOW()
                WHERE belief_id = %s
                """,
                (replacement_id, belief_id),
            )

            # Mirror retraction + supersession edge to Neo4j (best-effort).
            try:
                from app.subia.belief import neo4j_mirror
                neo4j_mirror.mirror_belief(belief_id, belief_status="RETRACTED")
                if replacement_id:
                    neo4j_mirror.mirror_supersession(belief_id, replacement_id)
            except Exception:
                logger.debug("belief_store: neo4j mirror failed", exc_info=True)

            update = MetacognitiveUpdate(
                source_belief_id=belief_id,
                trigger="EXTERNAL_EVIDENCE",
                observation=reason[:500],
                action_taken="BELIEF_RETRACTED",
                old_confidence=0.0,
                new_confidence=0.0,
                reasoning=reason,
            )
            self._persist_update(update)
            return update
        except Exception:
            return None

    def get_supersession_chain(self, belief_id: str, max_depth: int = 20) -> list[dict]:
        """Walk the supersession chain forward from `belief_id` via Neo4j.

        Returns the chain oldest-to-newest. Each element has belief_id,
        domain, confidence, belief_status. The starting belief is index 0;
        the terminal (current) belief is the last element.

        Returns [] if Neo4j is unavailable (PostgreSQL `superseded_by`
        column remains the source of truth — callers needing a chain when
        Neo4j is down should walk that recursively).
        """
        try:
            from app.subia.belief import neo4j_mirror
            return neo4j_mirror.get_supersession_chain(belief_id, max_depth=max_depth)
        except Exception as e:
            logger.debug(f"belief_store: get_supersession_chain failed: {e}")
            return []

    def get_stats(self) -> dict:
        """Dashboard stats: belief counts by status and domain."""
        try:
            from app.control_plane.db import execute
            rows = execute(
                """
                SELECT belief_status, domain, COUNT(*) as cnt,
                       AVG(confidence) as avg_conf
                FROM beliefs
                GROUP BY belief_status, domain
                """,
                fetch=True,
            )
            return {"belief_stats": rows or []}
        except Exception:
            return {"belief_stats": []}

    def _apply_confidence_decay(self, stored_confidence: float,
                                 last_validated) -> float:
        """Apply time-based confidence decay for unvalidated beliefs."""
        if not last_validated:
            return stored_confidence
        try:
            now = datetime.now(timezone.utc)
            if hasattr(last_validated, "timestamp"):
                age_hours = (now - last_validated).total_seconds() / 3600
            else:
                return stored_confidence
            # Decay: ~0.5% per cycle (1 cycle ≈ 1 task ≈ minutes)
            # Convert to hours: very slow decay
            cycles_approx = age_hours / 0.5  # Assume ~30 min per cycle
            decay = self._config.confidence_decay_factor ** cycles_approx
            return round(stored_confidence * decay, 4)
        except Exception:
            return stored_confidence

    def _persist_belief(self, belief: Belief) -> None:
        """Store belief to PostgreSQL, then mirror to Neo4j (best-effort)."""
        try:
            from app.control_plane.db import execute
            execute(
                """
                INSERT INTO beliefs
                    (belief_id, content, content_embedding, domain, confidence,
                     evidence_sources, formed_at, last_validated, last_updated,
                     belief_status)
                VALUES (%s, %s, %s::vector, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    belief.belief_id,
                    belief.content[:2000],
                    belief.content_embedding or None,
                    belief.domain,
                    belief.confidence,
                    json.dumps(belief.evidence_sources),
                    belief.formed_at,
                    belief.last_validated,
                    belief.last_updated,
                    belief.belief_status,
                ),
            )
        except Exception:
            logger.debug("belief_store: persist failed", exc_info=True)
            return

        # Mirror to Neo4j as a :Belief node. Best-effort; SQL is authoritative.
        try:
            from app.subia.belief import neo4j_mirror
            neo4j_mirror.mirror_belief(
                belief.belief_id,
                domain=belief.domain,
                confidence=belief.confidence,
                belief_status=belief.belief_status,
                formed_at=belief.formed_at,
            )
        except Exception:
            logger.debug("belief_store: neo4j mirror failed", exc_info=True)

    def _persist_update(self, update: MetacognitiveUpdate) -> None:
        """Store metacognitive update to PostgreSQL."""
        try:
            from app.control_plane.db import execute
            execute(
                """
                INSERT INTO metacognitive_updates
                    (update_id, source_belief_id, trigger, observation,
                     action_taken, old_confidence, new_confidence, reasoning)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    update.update_id,
                    update.source_belief_id,
                    update.trigger,
                    update.observation[:2000],
                    update.action_taken,
                    update.old_confidence,
                    update.new_confidence,
                    update.reasoning[:2000],
                ),
            )
        except Exception:
            logger.debug("belief_store: update persist failed", exc_info=True)

# ── Module-level singleton ──────────────────────────────────────────────────

_store: BeliefStore | None = None

def get_belief_store() -> BeliefStore:
    global _store
    if _store is None:
        _store = BeliefStore()
    return _store
