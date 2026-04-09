"""
somatic_marker.py — Somatic Marker computation for AndrusAI agents.

Functional approximation of Damasio's Somatic Marker Hypothesis:
past experiences are tagged with outcome valence, and when a similar
decision context is encountered, the valence biases action selection.

NOT claiming phenomenal experience — this is a functional approximation
that achieves the decisional effect of valence-tagged experience.

IMMUTABLE — infrastructure-level module.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.self_awareness.internal_state import SomaticMarker

logger = logging.getLogger(__name__)


class SomaticMarkerComputer:
    """Computes somatic markers via pgvector similarity search on agent_experiences."""

    def __init__(self, top_k: int = 5, decay_factor: float = 0.95, min_similarity: float = 0.3):
        self.top_k = top_k
        self.decay_factor = decay_factor
        self.min_similarity = min_similarity

    def compute(
        self,
        agent_id: str,
        decision_context: str,
        context_embedding: Optional[list[float]] = None,
    ) -> SomaticMarker:
        """Compute somatic marker for current decision context. ~10ms with indexed pgvector."""
        if context_embedding is None:
            try:
                from app.memory.chromadb_manager import embed
                context_embedding = embed(decision_context[:1000])
            except Exception:
                return SomaticMarker()

        try:
            from app.control_plane.db import execute
            rows = execute(
                """
                SELECT
                    outcome_score,
                    context_summary,
                    created_at,
                    1 - (context_embedding <=> %s::vector) AS similarity
                FROM agent_experiences
                WHERE agent_id = %s
                  AND context_embedding IS NOT NULL
                ORDER BY context_embedding <=> %s::vector
                LIMIT %s
                """,
                (context_embedding, agent_id, context_embedding, self.top_k),
                fetch=True,
            )

            if not rows:
                return SomaticMarker(valence=0.0, intensity=0.0, source="no_prior_experience", match_count=0)

            # Filter by minimum similarity
            relevant = [r for r in rows if r[3] >= self.min_similarity]
            if not relevant:
                return SomaticMarker(valence=0.0, intensity=0.0, source="no_relevant_experience", match_count=0)

            # Weighted average: recency x similarity
            weighted_sum = 0.0
            weight_total = 0.0
            for i, row in enumerate(relevant):
                outcome_score, context_summary, created_at, similarity = row
                recency_weight = self.decay_factor ** i
                weight = recency_weight * similarity
                weighted_sum += outcome_score * weight
                weight_total += weight

            valence = weighted_sum / weight_total if weight_total > 0 else 0.0
            intensity = relevant[0][3]  # Strongest match similarity
            source = str(relevant[0][1])[:200]

            return SomaticMarker(
                valence=round(valence, 3),
                intensity=round(intensity, 3),
                source=source,
                match_count=len(relevant),
            )

        except Exception as e:
            logger.debug(f"Somatic marker computation failed for {agent_id}: {e}")
            return SomaticMarker()


def record_experience_sync(
    agent_id: str,
    context_summary: str,
    outcome_score: float,
    outcome_description: str = "",
    task_type: str = "",
    tools_used: Optional[list[str]] = None,
    venture: str = "system",
) -> None:
    """Record a completed experience for future somatic marker lookups.

    Call from post-crew telemetry with the task outcome.
    Non-fatal — never crashes the agent.
    """
    try:
        from app.memory.chromadb_manager import embed
        from app.control_plane.db import execute

        embedding = embed(context_summary[:1000])
        execute(
            """
            INSERT INTO agent_experiences (
                agent_id, venture, context_summary, context_embedding,
                outcome_score, outcome_description, task_type, tools_used
            ) VALUES (%s, %s, %s, %s::vector, %s, %s, %s, %s)
            """,
            (
                agent_id,
                venture,
                context_summary[:2000],
                embedding,
                max(-1.0, min(1.0, outcome_score)),
                outcome_description[:500],
                task_type,
                tools_used or [],
            ),
        )
    except Exception as e:
        logger.debug(f"Failed to record experience for {agent_id}: {e}")
