"""
somatic_bias.py — Pre-reasoning somatic bias injection.

Translates somatic valence from past experiences into concrete context
modifications that bias the agent's reasoning BEFORE it begins.

Implements Damasio's insight: emotions pre-filter the option space,
not just evaluate outcomes after the fact. A firefighter's anxiety
narrows possible actions before deliberation starts.

The bias is expressed as natural-language guidance, not hard constraints.
The agent can override the bias — it's a "gut feeling", not a rule.

IMMUTABLE — infrastructure-level module.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.self_awareness.internal_state import SomaticMarker, DISPOSITION_TO_RISK_TIER

logger = logging.getLogger(__name__)


class SomaticBiasInjector:
    """Injects somatic bias into task context before reasoning begins."""

    STRONG_NEGATIVE_THRESHOLD = -0.5
    MILD_NEGATIVE_THRESHOLD = -0.2
    MILD_POSITIVE_THRESHOLD = 0.2
    STRONG_POSITIVE_THRESHOLD = 0.5
    MIN_INTENSITY = 0.3

    def inject(self, task_context: dict, somatic: SomaticMarker) -> dict:
        """Modify task context based on somatic valence. Additive only."""
        if somatic.intensity < self.MIN_INTENSITY:
            return task_context

        bias = self._compute_bias(somatic)
        if bias is None:
            return task_context

        task_context.setdefault("_somatic_advisories", []).append(bias)

        if bias.get("context_note"):
            current_desc = task_context.get("description", "")
            note = bias["context_note"]
            task_context["description"] = f"[Somatic note: {note}]\n\n{current_desc}"

        if bias.get("approach_hint"):
            task_context.setdefault("strategy_hints", []).append(bias["approach_hint"])

        return task_context

    def _compute_bias(self, somatic: SomaticMarker) -> Optional[dict]:
        v = somatic.valence

        if v <= self.STRONG_NEGATIVE_THRESHOLD:
            return {
                "level": "strong_negative",
                "context_note": (
                    f"Past experience with similar contexts was strongly negative "
                    f"(source: {somatic.source[:80]}). Exercise heightened caution."
                ),
                "approach_hint": (
                    "Consider alternative approaches before proceeding. "
                    "Verify assumptions explicitly."
                ),
                "disposition_floor": "cautious",
            }

        elif v <= self.MILD_NEGATIVE_THRESHOLD:
            return {
                "level": "mild_negative",
                "context_note": (
                    f"Past experience with similar contexts was mixed-to-negative "
                    f"(source: {somatic.source[:80]}). Proceed with awareness."
                ),
                "approach_hint": "Double-check intermediate results before finalizing.",
                "disposition_floor": None,
            }

        elif v >= self.STRONG_POSITIVE_THRESHOLD:
            return {
                "level": "strong_positive",
                "context_note": (
                    f"Past experience with similar contexts was strongly positive "
                    f"(source: {somatic.source[:80]})."
                ),
                "approach_hint": None,
                "disposition_floor": None,
            }

        elif v >= self.MILD_POSITIVE_THRESHOLD:
            return {
                "level": "mild_positive",
                "context_note": None,
                "approach_hint": None,
                "disposition_floor": None,
            }

        return None

    @staticmethod
    def get_disposition_floor(task_context: dict) -> Optional[str]:
        """Extract the highest disposition floor from all somatic advisories."""
        advisories = task_context.get("_somatic_advisories", [])
        floors = [a["disposition_floor"] for a in advisories if a.get("disposition_floor")]
        if not floors:
            return None
        order = {"proceed": 0, "cautious": 1, "pause": 2, "escalate": 3}
        return max(floors, key=lambda f: order.get(f, 0))
