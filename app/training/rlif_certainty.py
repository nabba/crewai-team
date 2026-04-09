"""
rlif_certainty.py — RLIF Self-Certainty scoring for MLX QLoRA training.

Computes self-certainty (KL divergence from uniform) for candidate interactions
during curation. Used as a weight in training data selection.

EntropyCollapseMonitor detects when model becomes overconfident and pauses training.

References:
  - Zhao et al. (2025) "Learning to Reason without External Rewards" (INTUITOR)
  - Zhang et al. (2025) "No Free Lunch: Rethinking Internal Feedback"

IMMUTABLE — infrastructure-level module.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)


class SelfCertaintyScorer:
    """Computes self-certainty for training data curation.

    Self-certainty = average KL(Uniform || P) across response tokens.
    Higher = model more certain about its response.

    Note: Actual MLX forward pass runs on host via Host Bridge.
    This module provides the scoring logic and curation weights.
    """

    @staticmethod
    def compute_curation_weight(quality_score: float, self_certainty_score: float) -> float:
        """Combined curation weight for training data selection.

        Logic:
          High quality + high certainty = strong positive (model knows what it's doing)
          High quality + low certainty  = moderate (model got lucky)
          Low quality + high certainty  = NEGATIVE (overconfident failure — train AGAINST)
          Low quality + low certainty   = neutral (model correctly doubted itself)
        """
        weight = (
            quality_score * 0.6
            + self_certainty_score * 0.2
            + (quality_score * self_certainty_score) * 0.2
        )
        return max(0.0, min(1.0, weight))

    @staticmethod
    def score_from_logprobs(logprobs: list[dict]) -> float:
        """Compute self-certainty from token logprobs (if available from API).

        Args:
            logprobs: List of {token, logprob, top_logprobs} dicts from API response.

        Returns:
            Float self-certainty score. Higher = more certain.
        """
        if not logprobs:
            return 0.5

        import math
        certainties = []
        for lp in logprobs:
            prob = math.exp(lp.get("logprob", -1.0))
            certainties.append(prob)

        if certainties:
            return sum(certainties) / len(certainties)
        return 0.5


class EntropyCollapseMonitor:
    """Monitors for entropy collapse during RLIF-weighted training.

    Entropy collapse = model becomes overconfident (self-certainty scores
    converge to high values with low variance). Pauses training when detected.
    """

    def __init__(
        self,
        window_size: int = 50,
        variance_threshold: float = 0.05,
        mean_ceiling: float = 0.85,
    ):
        self.sc_history: deque[float] = deque(maxlen=window_size)
        self.variance_threshold = variance_threshold
        self.mean_ceiling = mean_ceiling
        self.window_size = window_size

    def check_batch(self, batch_sc_scores: list[float]) -> Optional[str]:
        """Check a batch for entropy collapse. Returns warning string or None."""
        if not batch_sc_scores:
            return None

        batch_mean = sum(batch_sc_scores) / len(batch_sc_scores)
        self.sc_history.append(batch_mean)

        if len(self.sc_history) < 10:
            return None

        window = list(self.sc_history)
        overall_mean = sum(window) / len(window)
        variance = sum((x - overall_mean) ** 2 for x in window) / len(window)

        if variance < self.variance_threshold and overall_mean > self.mean_ceiling:
            warning = (
                f"ENTROPY_COLLAPSE_WARNING: "
                f"Mean self-certainty={overall_mean:.3f} (ceiling={self.mean_ceiling}), "
                f"Variance={variance:.5f} (threshold={self.variance_threshold}). "
                f"Training should be paused."
            )
            logger.warning(warning)
            return warning

        return None

    def check_and_alert(
        self,
        batch_sc_scores: list[float],
        agent_id: str,
    ) -> bool:
        """Check for collapse and alert control plane. Returns True if should pause."""
        warning = self.check_batch(batch_sc_scores)
        if warning:
            try:
                from app.control_plane.audit import log_event
                log_event(
                    actor="training_pipeline",
                    action="entropy_collapse_detected",
                    resource_type="training",
                    resource_id=agent_id,
                    detail=warning[:500],
                )
            except Exception as e:
                logger.debug(f"Failed to alert control plane: {e}")
            return True  # Pause training
        return False

    def reset(self) -> None:
        self.sc_history.clear()


class TrajectoryEntropyScorer:
    """Complementary RLIF signal: trajectory-level entropy (Zhang et al. 2025).

    While self-certainty measures per-token confidence, trajectory-level entropy
    measures diversity of complete response trajectories. Low trajectory entropy =
    model consistently produces similar responses = high confidence.

    Combining both signals is more robust than either alone.
    """

    @staticmethod
    def compute_trajectory_entropy_from_embeddings(
        response_embeddings: list[list[float]],
    ) -> float:
        """Compute trajectory entropy from pre-computed response embeddings.

        Low average similarity = high entropy (diverse/uncertain).
        High average similarity = low entropy (consistent/confident).

        Returns float trajectory entropy in [0.0, 1.0].
        """
        if len(response_embeddings) < 2:
            return 0.5

        import math
        similarities = []
        for i in range(len(response_embeddings)):
            for j in range(i + 1, len(response_embeddings)):
                a, b = response_embeddings[i], response_embeddings[j]
                dot = sum(x * y for x, y in zip(a, b))
                norm_a = math.sqrt(sum(x * x for x in a))
                norm_b = math.sqrt(sum(x * x for x in b))
                if norm_a > 0 and norm_b > 0:
                    sim = (dot / (norm_a * norm_b) + 1.0) / 2.0
                    similarities.append(sim)

        if not similarities:
            return 0.5

        mean_sim = sum(similarities) / len(similarities)
        return 1.0 - mean_sim  # High similarity = low entropy

    @staticmethod
    def combine_signals(
        self_certainty: float,
        trajectory_entropy: float,
        sc_weight: float = 0.6,
        te_weight: float = 0.4,
    ) -> float:
        """Combine self-certainty and trajectory entropy.

        Self-certainty: higher = more certain.
        Trajectory entropy: higher = less certain.
        Combined: higher = better training candidate.
        """
        te_signal = 1.0 - trajectory_entropy
        return sc_weight * self_certainty + te_weight * te_signal

    @staticmethod
    def compute_curation_weight_combined(
        quality_score: float,
        self_certainty: float,
        trajectory_entropy: float,
    ) -> float:
        """Combined curation weight with both RLIF signals."""
        combined = TrajectoryEntropyScorer.combine_signals(self_certainty, trajectory_entropy)
        return max(0.0, min(1.0,
            quality_score * 0.5 + combined * 0.25 + (quality_score * combined) * 0.25
        ))
