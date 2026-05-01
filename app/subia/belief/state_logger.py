"""
state_logger.py — Persists InternalState to PostgreSQL after each reasoning step.

Uses sync psycopg2 via control_plane/db.py (matching codebase pattern).
All operations are non-fatal — logging never crashes the agent.

IMMUTABLE — infrastructure-level module.
"""

from __future__ import annotations

import json
import logging

from app.subia.belief.internal_state import InternalState

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
            certainty_trend, action_disposition, risk_tier,
            hyper_predicted_certainty, hyper_actual_certainty, hyper_prediction_error,
            free_energy_proxy, free_energy_trend, precision_weighted_certainty,
            competition_winner, competition_candidates, reality_model,
            full_state, created_at
        ) VALUES (
            %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s,
            %s, %s, %s,
            %s, %s, %s,
            %s, %s, %s,
            %s, %s, %s,
            %s, %s
        )
    """

    def log(self, state: InternalState) -> None:
        """Persist an InternalState to the database. Non-fatal."""
        try:
            from app.control_plane.db import execute

            # Extract Beautiful Loop fields (Phase 7)
            hm = state.hyper_model_state or {}
            comp = state.competition_result or {}
            rm = state.reality_model_summary or {}

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
                    # Beautiful Loop columns (Phase 7)
                    hm.get("predicted_certainty"),
                    hm.get("actual_certainty"),
                    hm.get("self_prediction_error"),
                    state.free_energy_proxy,
                    state.free_energy_trend,
                    state.precision_weighted_certainty,
                    json.dumps(comp.get("winner")) if comp.get("winner") else None,
                    json.dumps(comp.get("candidates")) if comp.get("candidates") else None,
                    json.dumps(rm) if rm else None,
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
            results = []
            for row in (rows or []):
                fs = row.get("full_state") if isinstance(row, dict) else row[0]
                if isinstance(fs, dict):
                    results.append(fs)
                elif isinstance(fs, str):
                    results.append(json.loads(fs))
            return results
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
                fg = row.get("certainty_factual_grounding", 0.5) if isinstance(row, dict) else row[0]
                tc = row.get("certainty_tool_confidence", 0.5) if isinstance(row, dict) else row[1]
                co = row.get("certainty_coherence", 0.5) if isinstance(row, dict) else row[2]
                m = (fg + tc + co) / 3.0
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
_logger: InternalStateLogger | None = None

def get_state_logger() -> InternalStateLogger:
    global _logger
    if _logger is None:
        _logger = InternalStateLogger()
    return _logger
