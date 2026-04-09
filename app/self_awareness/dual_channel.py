"""
dual_channel.py — Dual-Channel Feedback Composition for AndrusAI agents.

Composes epistemic certainty + experiential valence into action disposition.
Maps to Host Bridge 4-tier risk model.

CRITICAL SAFETY PROPERTY:
  This function can only INCREASE caution, never decrease it.
  There is no path from escalated → proceed within the same decision.

IMMUTABLE — infrastructure-level module.
"""

from __future__ import annotations

import logging
from app.self_awareness.internal_state import (
    InternalState,
    DISPOSITION_TO_RISK_TIER,
)

logger = logging.getLogger(__name__)


# Disposition matrix
# Rows: certainty level (high / mid / low)
# Columns: valence level (positive / neutral / negative)
DISPOSITION_MATRIX: dict[tuple[str, str], str] = {
    # High certainty
    ("high", "positive"): "proceed",
    ("high", "neutral"):  "proceed",
    ("high", "negative"): "cautious",   # Confident but bad somatic signal

    # Mid certainty
    ("mid", "positive"):  "proceed",
    ("mid", "neutral"):   "cautious",
    ("mid", "negative"):  "pause",

    # Low certainty
    ("low", "positive"):  "cautious",   # Uncertain but positive intuition
    ("low", "neutral"):   "pause",
    ("low", "negative"):  "escalate",
}


class DualChannelComposer:
    """Composes epistemic certainty and somatic valence into action disposition."""

    def __init__(
        self,
        certainty_high_threshold: float = 0.7,
        certainty_low_threshold: float = 0.4,
        valence_positive_threshold: float = 0.2,
        valence_negative_threshold: float = -0.2,
        critical_budget_threshold: float = 0.1,
    ):
        self.certainty_high = certainty_high_threshold
        self.certainty_low = certainty_low_threshold
        self.valence_positive = valence_positive_threshold
        self.valence_negative = valence_negative_threshold
        self.critical_budget = critical_budget_threshold

    def compose(self, state: InternalState) -> InternalState:
        """Compute action_disposition and risk_tier. Mutates and returns state."""
        cert_level = self._discretize_certainty(state)
        val_level = self._discretize_valence(state)

        disposition = DISPOSITION_MATRIX.get(
            (cert_level, val_level), "cautious"
        )
        risk_tier = DISPOSITION_TO_RISK_TIER[disposition]

        # Override: critical compute budget → force at least tier 3
        if state.meta.compute_budget_remaining_pct < self.critical_budget:
            if risk_tier < 3:
                risk_tier = 3
                disposition = "pause"

        state.action_disposition = disposition
        state.risk_tier = risk_tier

        return state

    def _discretize_certainty(self, state: InternalState) -> str:
        adjusted = state.certainty.adjusted_certainty
        if adjusted > self.certainty_high:
            return "high"
        if adjusted > self.certainty_low:
            return "mid"
        return "low"

    def _discretize_valence(self, state: InternalState) -> str:
        v = state.somatic.valence
        if v > self.valence_positive:
            return "positive"
        if v > self.valence_negative:
            return "neutral"
        return "negative"
