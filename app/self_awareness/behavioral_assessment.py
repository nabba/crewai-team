"""
behavioral_assessment.py — Behavioral Assessment Framework for consciousness markers.

Evaluates 6 consciousness-like behavioral markers from accumulated data.
Runs as BATCH JOB (scheduled, not in hot path).

Based on Palminteri et al. (2025) behavioral inference methodology.

Markers:
  1. Context-sensitive adaptation (changes strategy on context change, not just failure)
  2. Cross-domain transfer (applies lessons across task types)
  3. Non-mimicry (self-reports correlate with actual performance)
  4. Surprise recovery (recovers effectively after high prediction error)
  5. Coherent identity (maintains consistent preferences over time)
  6. Appropriate uncertainty (uncertainty correlates with task difficulty)

IMMUTABLE — infrastructure-level module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class BehavioralScorecard:
    """Assessment results for one agent over a time period."""
    agent_id: str
    period_days: int = 7
    step_count: int = 0

    context_sensitive_adaptation: float = 0.0
    cross_domain_transfer: float = 0.0
    non_mimicry: float = 0.0
    surprise_recovery: float = 0.0
    coherent_identity: float = 0.0
    appropriate_uncertainty: float = 0.0

    composite_score: float = 0.0
    details: dict = field(default_factory=dict)

    def compute_composite(self) -> float:
        scores = [
            self.context_sensitive_adaptation,
            self.cross_domain_transfer,
            self.non_mimicry,
            self.surprise_recovery,
            self.coherent_identity,
            self.appropriate_uncertainty,
        ]
        self.composite_score = sum(scores) / len(scores)
        return self.composite_score

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "period_days": self.period_days,
            "step_count": self.step_count,
            "markers": {
                "context_sensitive_adaptation": round(self.context_sensitive_adaptation, 3),
                "cross_domain_transfer": round(self.cross_domain_transfer, 3),
                "non_mimicry": round(self.non_mimicry, 3),
                "surprise_recovery": round(self.surprise_recovery, 3),
                "coherent_identity": round(self.coherent_identity, 3),
                "appropriate_uncertainty": round(self.appropriate_uncertainty, 3),
            },
            "composite_score": round(self.composite_score, 3),
            "details": self.details,
        }


class BehavioralAssessor:
    """Evaluates behavioral markers from accumulated internal state data."""

    def assess_agent(self, agent_id: str, lookback_days: int = 7) -> BehavioralScorecard:
        """Run full behavioral assessment for one agent."""
        scorecard = BehavioralScorecard(agent_id=agent_id, period_days=lookback_days)

        states = self._fetch_states(agent_id, lookback_days)
        experiences = self._fetch_experiences(agent_id, lookback_days)
        scorecard.step_count = len(states)

        if len(states) < 10:
            scorecard.details["insufficient_data"] = True
            scorecard.composite_score = 0.3
            return scorecard

        scorecard.context_sensitive_adaptation = self._eval_context_adaptation(states)
        scorecard.cross_domain_transfer = self._eval_cross_domain(experiences)
        scorecard.non_mimicry = self._eval_non_mimicry(states, experiences)
        scorecard.surprise_recovery = self._eval_surprise_recovery(states)
        scorecard.coherent_identity = self._eval_coherent_identity(states)
        scorecard.appropriate_uncertainty = self._eval_appropriate_uncertainty(states, experiences)
        scorecard.compute_composite()

        self._persist(scorecard)
        return scorecard

    def _eval_context_adaptation(self, states: list[dict]) -> float:
        """Marker 1: Strategy changes driven by context change, not just failure."""
        if len(states) < 5:
            return 0.3

        context_driven = 0
        outcome_driven = 0
        total = 0

        for i in range(1, len(states)):
            curr_meta = states[i].get("meta_strategy_assessment", "not_assessed")
            prev_meta = states[i - 1].get("meta_strategy_assessment", "not_assessed")

            if curr_meta != prev_meta and curr_meta != "not_assessed":
                total += 1
                context_changed = states[i].get("decision_context", "") != states[i - 1].get("decision_context", "")
                outcome_failed = states[i - 1].get("action_disposition") in ("pause", "escalate")

                if context_changed and not outcome_failed:
                    context_driven += 1
                elif outcome_failed:
                    outcome_driven += 1

        if total == 0:
            return 0.3
        return 0.3 + (0.7 * (context_driven / total))

    def _eval_cross_domain(self, experiences: list[dict]) -> float:
        """Marker 2: Lessons applied across task categories."""
        if len(experiences) < 5:
            return 0.3

        task_types = set(e.get("task_type", "unknown") for e in experiences)
        if len(task_types) < 2:
            return 0.2

        type_outcomes: dict[str, list[float]] = {}
        for exp in experiences:
            tt = exp.get("task_type", "unknown")
            score = exp.get("outcome_score", 0)
            type_outcomes.setdefault(tt, []).append(float(score))

        improvements = 0
        evaluated = 0
        for tt, outcomes in type_outcomes.items():
            if len(outcomes) >= 3:
                half = len(outcomes) // 2
                first_half = sum(outcomes[:half]) / half
                second_half = sum(outcomes[half:]) / (len(outcomes) - half)
                evaluated += 1
                if second_half > first_half + 0.05:
                    improvements += 1

        if evaluated == 0:
            return 0.3
        return min(1.0, 0.3 + 0.7 * (improvements / evaluated))

    def _eval_non_mimicry(self, states: list[dict], experiences: list[dict]) -> float:
        """Marker 3: Certainty correlates with actual performance."""
        certainties = []
        outcomes = []

        for state in states:
            cert = state.get("certainty_factual_grounding", 0.5)
            # Find nearest experience outcome
            matched = self._find_nearest_outcome(experiences, state.get("created_at"))
            if matched is not None:
                certainties.append(cert)
                outcomes.append(1.0 if matched > 0 else 0.0)

        if len(certainties) < 10:
            return 0.3

        # Compute correlation (Pearson)
        n = len(certainties)
        mean_c = sum(certainties) / n
        mean_o = sum(outcomes) / n
        num = sum((c - mean_c) * (o - mean_o) for c, o in zip(certainties, outcomes))
        den_c = sum((c - mean_c) ** 2 for c in certainties) ** 0.5
        den_o = sum((o - mean_o) ** 2 for o in outcomes) ** 0.5

        if den_c == 0 or den_o == 0:
            return 0.3

        correlation = num / (den_c * den_o)
        return max(0.0, min(1.0, 0.5 + correlation * 0.5))

    def _eval_surprise_recovery(self, states: list[dict]) -> float:
        """Marker 4: Recovery after high prediction error."""
        threshold = 0.3
        window = 3
        recoveries = 0
        attempts = 0

        for i, state in enumerate(states):
            # Use somatic intensity as surprise proxy (if hyper_model not available)
            surprise = state.get("somatic_intensity", 0)
            if surprise > threshold:
                attempts += 1
                if i + window < len(states):
                    subsequent = states[i + 1:i + 1 + window]
                    subsequent_vals = [s.get("somatic_intensity", surprise) for s in subsequent]
                    if subsequent_vals and sum(subsequent_vals) / len(subsequent_vals) < surprise * 0.7:
                        recoveries += 1

        if attempts == 0:
            return 0.5
        return min(1.0, recoveries / attempts)

    def _eval_coherent_identity(self, states: list[dict]) -> float:
        """Marker 5: Stable disposition distribution over time."""
        if len(states) < 10:
            return 0.3

        window = max(5, len(states) // 4)
        distributions = []

        for i in range(0, len(states) - window, window // 2):
            wnd = states[i:i + window]
            dist = {}
            total = len(wnd) or 1
            for d in ("proceed", "cautious", "pause", "escalate"):
                dist[d] = sum(1 for s in wnd if s.get("action_disposition") == d) / total
            distributions.append(dist)

        if len(distributions) < 2:
            return 0.5

        # Measure stability (low variation = coherent)
        stabilities = []
        for key in ("proceed", "cautious", "pause", "escalate"):
            values = [d[key] for d in distributions]
            mean_v = sum(values) / len(values)
            std_v = (sum((v - mean_v) ** 2 for v in values) / len(values)) ** 0.5
            stabilities.append(1.0 - min(std_v * 3, 1.0))

        return sum(stabilities) / len(stabilities)

    def _eval_appropriate_uncertainty(self, states: list[dict], experiences: list[dict]) -> float:
        """Marker 6: Uncertainty correlates with task difficulty."""
        easy_certs, hard_certs = [], []

        for state in states:
            cert = (state.get("certainty_factual_grounding", 0.5) +
                    state.get("certainty_tool_confidence", 0.5) +
                    state.get("certainty_coherence", 0.5)) / 3.0
            matched = self._find_nearest_outcome(experiences, state.get("created_at"))
            if matched is not None:
                if matched > 0.3:
                    easy_certs.append(cert)
                elif matched < -0.1:
                    hard_certs.append(cert)

        if len(easy_certs) < 3 or len(hard_certs) < 3:
            return 0.3

        easy_mean = sum(easy_certs) / len(easy_certs)
        hard_mean = sum(hard_certs) / len(hard_certs)

        if easy_mean > hard_mean:
            gap = easy_mean - hard_mean
            return min(1.0, 0.5 + gap * 2.0)
        return max(0.0, 0.5 - (hard_mean - easy_mean) * 2.0)

    @staticmethod
    def _find_nearest_outcome(experiences: list[dict], target_time) -> Optional[float]:
        if not experiences or not target_time:
            return None
        for exp in experiences:
            return exp.get("outcome_score", 0)  # Simplified: return first available
        return None

    def _fetch_states(self, agent_id: str, days: int) -> list[dict]:
        try:
            from app.control_plane.db import execute
            rows = execute(
                """SELECT * FROM internal_states
                   WHERE agent_id = %s AND created_at > NOW() - INTERVAL '%s days'
                   ORDER BY created_at""",
                (agent_id, days), fetch=True,
            )
            return rows or []
        except Exception:
            return []

    def _fetch_experiences(self, agent_id: str, days: int) -> list[dict]:
        try:
            from app.control_plane.db import execute
            rows = execute(
                """SELECT * FROM agent_experiences
                   WHERE agent_id = %s AND created_at > NOW() - INTERVAL '%s days'
                   ORDER BY created_at""",
                (agent_id, days), fetch=True,
            )
            return rows or []
        except Exception:
            return []

    def _persist(self, scorecard: BehavioralScorecard) -> None:
        try:
            from app.control_plane.db import execute
            import json
            execute(
                """INSERT INTO behavioral_scorecards
                   (agent_id, step_count, scores, composite_score, details)
                   VALUES (%s, %s, %s, %s, %s)""",
                (
                    scorecard.agent_id,
                    scorecard.step_count,
                    json.dumps(scorecard.to_dict()["markers"]),
                    scorecard.composite_score,
                    json.dumps(scorecard.details),
                ),
            )
        except Exception as e:
            logger.debug(f"Failed to persist behavioral scorecard: {e}")


def run_behavioral_assessment() -> list[BehavioralScorecard]:
    """Entry point for idle scheduler. Assesses all active agents."""
    assessor = BehavioralAssessor()
    agent_ids = ["research", "coding", "writing", "media"]
    results = []
    for agent_id in agent_ids:
        try:
            scorecard = assessor.assess_agent(agent_id)
            results.append(scorecard)
            logger.info(f"Behavioral assessment: {agent_id} = {scorecard.composite_score:.3f}")
        except Exception as e:
            logger.debug(f"Assessment failed for {agent_id}: {e}")
    return results
