"""
state_logger.py — Persists InternalState to PostgreSQL after each reasoning step.

Uses sync psycopg2 via control_plane/db.py (matching codebase pattern).
All operations are non-fatal — logging never crashes the agent.

IMMUTABLE — infrastructure-level module.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from app.self_awareness.internal_state import InternalState

logger = logging.getLogger(__name__)


class InternalStateLogger:
    """Persists InternalState objects to PostgreSQL."""

    INSERT_SQL = """
        INSERT INTO internal_states (
            state_id, agent_id, crew_id, venture, step_number, decision_context,
            certainty_factual_grounding, certainty_tool_confidence, certainty_coherence,
            certainty_task_understanding, certainty_value_alignment, certainty_meta,
            somatic_valence, somatic_intensity, somatic_source, somatic_match_count,
            meta_strategy_assessment, meta_modification_proposed, meta_modification_description,
            meta_compute_phase, meta_compute_budget_remaining, meta_reassessment_triggered,
            certainty_trend, action_disposition, risk_tier, full_state, created_at
        ) VALUES (
            %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s
        )
    """

    def log(self, state: InternalState) -> None:
        """Persist an InternalState to the database. Non-fatal."""
        try:
            from app.control_plane.db import execute
            execute(
                self.INSERT_SQL,
                (
                    state.state_id,
                    state.agent_id,
                    state.crew_id,
                    state.venture or "system",
                    state.step_number,
                    (state.decision_context or "")[:2000],
                    state.certainty.factual_grounding,
                    state.certainty.tool_confidence,
                    state.certainty.coherence,
                    state.certainty.task_understanding,
                    state.certainty.value_alignment,
                    state.certainty.meta_certainty,
                    state.somatic.valence,
                    state.somatic.intensity,
                    state.somatic.source,
                    state.somatic.match_count,
                    state.meta.strategy_assessment,
                    state.meta.modification_proposed,
                    state.meta.modification_description,
                    state.meta.compute_phase,
                    state.meta.compute_budget_remaining_pct,
                    state.meta.reassessment_triggered,
                    state.certainty_trend,
                    state.action_disposition,
                    state.risk_tier,
                    state.to_json(),
                    state.created_at,
                ),
            )
        except Exception as e:
            logger.debug(f"Failed to log InternalState {state.state_id}: {e}")

    def get_recent_states(self, agent_id: str, limit: int = 10) -> list[dict]:
        """Retrieve recent states for trend computation."""
        try:
            from app.control_plane.db import execute
            rows = execute(
                """
                SELECT full_state
                FROM internal_states
                WHERE agent_id = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (agent_id, limit),
                fetch=True,
            )
            return [row[0] if isinstance(row[0], dict) else json.loads(row[0]) for row in (rows or [])]
        except Exception:
            return []

    def compute_trend(self, agent_id: str, window: int = 5) -> str:
        """Compute certainty trend over the last N states."""
        try:
            from app.control_plane.db import execute
            rows = execute(
                """
                SELECT
                    certainty_factual_grounding,
                    certainty_tool_confidence,
                    certainty_coherence
                FROM internal_states
                WHERE agent_id = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (agent_id, window),
                fetch=True,
            )
            if not rows or len(rows) < 3:
                return "stable"

            means = []
            for row in rows:
                m = (row[0] + row[1] + row[2]) / 3.0
                means.append(m)

            half = len(means) // 2
            recent_avg = sum(means[:half]) / half if half > 0 else 0
            older_avg = sum(means[half:]) / (len(means) - half) if (len(means) - half) > 0 else 0

            delta = recent_avg - older_avg
            if delta > 0.05:
                return "rising"
            if delta < -0.05:
                return "falling"
            return "stable"
        except Exception:
            return "stable"


# Module-level singleton
_logger: Optional[InternalStateLogger] = None


def get_state_logger() -> InternalStateLogger:
    global _logger
    if _logger is None:
        _logger = InternalStateLogger()
    return _logger
