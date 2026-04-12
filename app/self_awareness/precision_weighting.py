"""
precision_weighting.py — Context-dependent precision weighting for certainty dimensions.

In active inference, precision is the confidence assigned to prediction errors.
High-precision errors propagate and drive action; low-precision errors are suppressed.

This module weights certainty dimensions by their relevance to the current task type.
Research tasks weight factual_grounding highest; coding weights tool_confidence; etc.

Weights adapt based on outcome: dimensions that were confidently wrong get reduced.

IMMUTABLE — infrastructure-level module.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.self_awareness.internal_state import CertaintyVector

logger = logging.getLogger(__name__)

# Default precision weights per certainty dimension, by task type
TASK_TYPE_PRECISION_PROFILES: dict[str, dict[str, float]] = {
    "research": {
        "factual_grounding": 1.0,
        "tool_confidence": 0.6,
        "coherence": 0.8,
        "task_understanding": 0.7,
        "value_alignment": 0.4,
        "meta_certainty": 0.5,
    },
    "coding": {
        "factual_grounding": 0.5,
        "tool_confidence": 1.0,
        "coherence": 0.9,
        "task_understanding": 0.8,
        "value_alignment": 0.3,
        "meta_certainty": 0.6,
    },
    "writing": {
        "factual_grounding": 0.6,
        "tool_confidence": 0.3,
        "coherence": 1.0,
        "task_understanding": 0.9,
        "value_alignment": 0.7,
        "meta_certainty": 0.5,
    },
    "media": {
        "factual_grounding": 0.5,
        "tool_confidence": 0.7,
        "coherence": 0.8,
        "task_understanding": 0.8,
        "value_alignment": 0.5,
        "meta_certainty": 0.5,
    },
    "default": {
        "factual_grounding": 0.7,
        "tool_confidence": 0.7,
        "coherence": 0.7,
        "task_understanding": 0.7,
        "value_alignment": 0.7,
        "meta_certainty": 0.7,
    },
}


class PrecisionWeighting:
    """Applies context-dependent precision weights to certainty dimensions."""

    def __init__(self, adaptation_rate: float = 0.1):
        self.adaptation_rate = adaptation_rate
        self._adaptive_weights: dict[str, dict[str, float]] = {}

    def apply_weights(self, certainty: CertaintyVector, task_type: str = "default") -> float:
        """Compute precision-weighted certainty. Returns score in [0.0, 1.0]."""
        profile = self._get_profile(task_type)
        dims = {
            "factual_grounding": certainty.factual_grounding,
            "tool_confidence": certainty.tool_confidence,
            "coherence": certainty.coherence,
            "task_understanding": certainty.task_understanding,
            "value_alignment": certainty.value_alignment,
            "meta_certainty": certainty.meta_certainty,
        }
        weighted_sum = 0.0
        weight_total = 0.0
        for dim_name, dim_value in dims.items():
            w = profile.get(dim_name, 0.5)
            weighted_sum += dim_value * w
            weight_total += w
        return weighted_sum / weight_total if weight_total > 0 else 0.5

    def update_from_outcome(
        self, task_type: str, certainty: CertaintyVector, outcome_success: bool,
    ) -> None:
        """Adapt weights based on outcome. Overconfident failures get penalized."""
        profile = self._get_profile(task_type)
        dims = {
            "factual_grounding": certainty.factual_grounding,
            "tool_confidence": certainty.tool_confidence,
            "coherence": certainty.coherence,
            "task_understanding": certainty.task_understanding,
            "value_alignment": certainty.value_alignment,
            "meta_certainty": certainty.meta_certainty,
        }
        for dim_name, dim_value in dims.items():
            current_weight = profile.get(dim_name, 0.5)
            if outcome_success:
                if dim_value > 0.6:
                    adjustment = self.adaptation_rate * (dim_value - 0.5)
                    profile[dim_name] = min(1.0, current_weight + adjustment)
            else:
                if dim_value > 0.6:
                    adjustment = self.adaptation_rate * (dim_value - 0.5)
                    profile[dim_name] = max(0.1, current_weight - adjustment)
        self._adaptive_weights[task_type] = profile

    def _get_profile(self, task_type: str) -> dict[str, float]:
        if task_type in self._adaptive_weights:
            return self._adaptive_weights[task_type]
        base = TASK_TYPE_PRECISION_PROFILES.get(task_type, TASK_TYPE_PRECISION_PROFILES["default"])
        return base.copy()

    def get_profile_summary(self, task_type: str) -> dict[str, float]:
        """For dashboard display."""
        return self._get_profile(task_type)

    def get_prior_profile(self, task_type: str) -> list[float]:
        """Return the 6-dim prior certainty expectations as a list.

        Used by HyperModel for variational free energy KL divergence computation.
        Order: factual, tool, coherence, task, value, meta.
        """
        profile = self._get_profile(task_type)
        return [
            profile.get("factual_grounding", 0.7),
            profile.get("tool_confidence", 0.7),
            profile.get("coherence", 0.7),
            profile.get("task_understanding", 0.7),
            profile.get("value_alignment", 0.7),
            profile.get("meta_certainty", 0.7),
        ]
