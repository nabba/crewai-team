"""
Adaptation path — improvement systems that evolve the agent over time.

This package provides a clean boundary around the adaptation control plane.
Multiple improvement systems operate in parallel, each with different
trigger mechanisms but unified promotion governance.

Systems:
  - Evolution: population-based prompt optimization (island, parallel, MAP-Elites)
  - Feedback/Modification: user reaction → pattern detection → prompt changes
  - Training: self-training via MLX LoRA on collected interaction data
  - ATLAS: autonomous tool-learning and skill acquisition

All systems submit promotions through app.governance for unified gates,
consistent safety thresholds, and a shared audit trail.
"""

from app.governance import (
    evaluate_promotion,
    PromotionRequest,
    PromotionResult,
    get_recent_promotions,
    format_governance_report,
    SAFETY_MINIMUM,
    QUALITY_MINIMUM,
    MAX_REGRESSION,
)

__all__ = [
    "evaluate_promotion",
    "PromotionRequest",
    "PromotionResult",
    "get_recent_promotions",
    "format_governance_report",
    "SAFETY_MINIMUM",
    "QUALITY_MINIMUM",
    "MAX_REGRESSION",
]
