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
from dataclasses import dataclass
from app.self_awareness.internal_state import (
    InternalState,
    DISPOSITION_TO_RISK_TIER,
)

logger = logging.getLogger(__name__)


@dataclass
class AttentionSchema:
    """Continuous attention modulation model (Graziano 2013).

    The system maintains an internal model of its own attention allocation —
    not just what it decides, but a continuous representation of HOW it's
    attending. This is the attention schema that AST identifies as a
    prerequisite for consciousness.

    5 continuous dimensions replace the 4 discrete disposition bins:
    """
    focus_intensity: float = 0.5      # [0,1] — diffuse to laser-focused
    caution_level: float = 0.25       # [0,1] — continuous caution (0=proceed, 1=escalate)
    exploration_drive: float = 0.5    # [0,1] — exploit to explore (from free energy)
    metacognitive_load: float = 0.3   # [0,1] — monitoring overhead
    somatic_salience: float = 0.0     # [0,1] — emotional influence strength

    def to_dict(self) -> dict:
        return {
            "focus_intensity": round(self.focus_intensity, 3),
            "caution_level": round(self.caution_level, 3),
            "exploration_drive": round(self.exploration_drive, 3),
            "metacognitive_load": round(self.metacognitive_load, 3),
            "somatic_salience": round(self.somatic_salience, 3),
        }


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

    def __init__(self, critical_budget_threshold: float = 0.1):
        # Load thresholds from sentience config (adjustable by cogito feedback loop)
        try:
            from app.self_awareness.sentience_config import load_config
            cfg = load_config()
        except Exception:
            cfg = {}
        self.certainty_high = cfg.get("certainty_high_threshold", 0.7)
        self.certainty_low = cfg.get("certainty_low_threshold", 0.4)
        self.valence_positive = cfg.get("valence_positive_threshold", 0.2)
        self.valence_negative = cfg.get("valence_negative_threshold", -0.2)
        self.critical_budget = critical_budget_threshold

    def compose(self, state: InternalState, task_context: dict = None) -> InternalState:
        """Compute action_disposition and risk_tier. Mutates and returns state.

        Args:
            state: The InternalState to compose.
            task_context: Optional task context dict containing pre-reasoning
                somatic advisories (Phase 3R somatic floor).
        """
        cert_level = self._discretize_certainty(state)
        val_level = self._discretize_valence(state)

        disposition = DISPOSITION_MATRIX.get(
            (cert_level, val_level), "cautious"
        )
        risk_tier = DISPOSITION_TO_RISK_TIER[disposition]

        # Phase 3R: Enforce pre-reasoning somatic disposition floor
        if task_context:
            try:
                from app.self_awareness.somatic_bias import SomaticBiasInjector
                floor = SomaticBiasInjector.get_disposition_floor(task_context)
                if floor:
                    floor_tier = DISPOSITION_TO_RISK_TIER.get(floor, 1)
                    if floor_tier > risk_tier:
                        risk_tier = floor_tier
                        disposition = floor
            except Exception:
                pass

        # Override: critical compute budget → force at least tier 3
        if state.meta.compute_budget_remaining_pct < self.critical_budget:
            if risk_tier < 3:
                risk_tier = 3
                disposition = "pause"

        state.action_disposition = disposition
        state.risk_tier = risk_tier

        # Continuous attention schema (Graziano AST)
        try:
            schema = self._compute_attention_schema(state)
            state.attention_schema = schema.to_dict()
        except Exception:
            pass

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

    def _compute_attention_schema(self, state: InternalState) -> AttentionSchema:
        """Compute continuous attention model from internal state signals.

        Replaces the coarse 4-bin disposition with a nuanced 5D representation
        of how the system is allocating its attentional resources.
        """
        cert = state.certainty.adjusted_certainty
        val = state.somatic.valence
        intensity = state.somatic.intensity
        variance = state.certainty.variance

        # Focus intensity: high certainty + low variance = sharply focused
        focus = cert * (1.0 - min(1.0, variance * 5.0))
        focus = max(0.0, min(1.0, focus))

        # Caution level: continuous version of disposition
        # Low certainty → more caution, negative valence → more caution
        caution = (1.0 - cert) * 0.5 + max(0.0, -val) * 0.3 + (1.0 - focus) * 0.2
        caution = max(0.0, min(1.0, caution))

        # Exploration drive: from free energy (if available in hyper_model_state)
        exploration = 0.5
        hm = state.hyper_model_state
        if hm and isinstance(hm, dict):
            fe = hm.get("variational_fe") or hm.get("free_energy_proxy") or 0
            exploration = min(1.0, float(fe) * 1.5)

        # Metacognitive load: high variance = more monitoring needed
        meta_load = min(1.0, variance * 3.0)
        if state.meta.reassessment_triggered:
            meta_load = min(1.0, meta_load + 0.3)

        # Somatic salience: how strongly emotions influence this step
        somatic_sal = intensity * abs(val)

        return AttentionSchema(
            focus_intensity=round(focus, 3),
            caution_level=round(caution, 3),
            exploration_drive=round(exploration, 3),
            metacognitive_load=round(meta_load, 3),
            somatic_salience=round(somatic_sal, 3),
        )
