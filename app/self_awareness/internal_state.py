"""
internal_state.py — Unified internal state for all AndrusAI sentience additions.

Single shared dataclass populated by four producers per reasoning step:
  1. CertaintyVector — epistemic certainty (fast + slow path)
  2. SomaticMarker — experiential valence from past outcomes
  3. MetaCognitiveState — strategy assessment from meta-cognitive layer
  4. DualChannel composition → action_disposition + risk_tier

Logged to PostgreSQL after each step. Compact summary injected into
next step context (~30 tokens) for recursive self-awareness.

IMMUTABLE — infrastructure-level module.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional


# ── Certainty Vector ─────────────────────────────────────────────────────────

@dataclass
class CertaintyVector:
    """Six-dimensional certainty assessment per reasoning step. All values [0.0, 1.0]."""

    # Fast-path (DB lookups + embeddings, always available, ~50ms)
    factual_grounding: float = 0.5   # Ratio of RAG-sourced claims to total claims
    tool_confidence: float = 0.5     # Historical success rate of selected tool
    coherence: float = 0.5           # Embedding similarity to recent outputs

    # Slow-path (local LLM, conditionally triggered, ~100ms)
    task_understanding: float = 0.5  # Semantic match: task desc vs agent paraphrase
    value_alignment: float = 0.5     # Cosine sim: action embedding vs SOUL.md embedding
    meta_certainty: float = 0.5      # Variance across other 5 dims (high variance = low)

    @property
    def fast_path_mean(self) -> float:
        return (self.factual_grounding + self.tool_confidence + self.coherence) / 3.0

    @property
    def full_mean(self) -> float:
        dims = [
            self.factual_grounding, self.tool_confidence, self.coherence,
            self.task_understanding, self.value_alignment, self.meta_certainty,
        ]
        return sum(dims) / len(dims)

    @property
    def adjusted_certainty(self) -> float:
        """Mean of 5 primary dims, discounted by meta_certainty."""
        primary = [
            self.factual_grounding, self.tool_confidence, self.coherence,
            self.task_understanding, self.value_alignment,
        ]
        avg = sum(primary) / len(primary)
        return avg * (0.5 + 0.5 * self.meta_certainty)

    @property
    def variance(self) -> float:
        """Variance across the 5 primary dimensions."""
        dims = [
            self.factual_grounding, self.tool_confidence, self.coherence,
            self.task_understanding, self.value_alignment,
        ]
        mean = sum(dims) / len(dims)
        return sum((d - mean) ** 2 for d in dims) / len(dims)

    def any_below_threshold(self, threshold: float = 0.4) -> bool:
        return any(v < threshold for v in [
            self.factual_grounding, self.tool_confidence, self.coherence,
        ])

    def should_trigger_slow_path(
        self, threshold: float = 0.4, variance_threshold: float = 0.03,
    ) -> bool:
        return self.any_below_threshold(threshold) or self.variance > variance_threshold

    def to_dict(self) -> dict:
        return {
            "factual_grounding": round(self.factual_grounding, 3),
            "tool_confidence": round(self.tool_confidence, 3),
            "coherence": round(self.coherence, 3),
            "task_understanding": round(self.task_understanding, 3),
            "value_alignment": round(self.value_alignment, 3),
            "meta_certainty": round(self.meta_certainty, 3),
        }


# ── Somatic Marker ───────────────────────────────────────────────────────────

@dataclass
class SomaticMarker:
    """Experiential valence from similarity-weighted past outcomes (Damasio)."""
    valence: float = 0.0          # -1.0 (strongly negative) to 1.0 (strongly positive)
    intensity: float = 0.0        # 0.0 (no prior) to 1.0 (exact match)
    source: str = "no_prior"      # Description of triggering memory
    match_count: int = 0          # Past experiences found

    def to_dict(self) -> dict:
        return {
            "valence": round(self.valence, 3),
            "intensity": round(self.intensity, 3),
            "source": self.source,
            "match_count": self.match_count,
        }


# ── Meta-Cognitive State ─────────────────────────────────────────────────────

@dataclass
class MetaCognitiveState:
    """Strategy assessment from the meta-cognitive layer."""
    strategy_assessment: str = "not_assessed"  # effective | uncertain | failing
    modification_proposed: bool = False
    modification_description: Optional[str] = None
    compute_phase: str = "early"               # early | mid | late
    compute_budget_remaining_pct: float = 1.0
    reassessment_triggered: bool = False

    def to_dict(self) -> dict:
        return {
            "strategy_assessment": self.strategy_assessment,
            "modification_proposed": self.modification_proposed,
            "modification_description": self.modification_description,
            "compute_phase": self.compute_phase,
            "compute_budget_remaining_pct": round(self.compute_budget_remaining_pct, 3),
            "reassessment_triggered": self.reassessment_triggered,
        }


# ── Action Disposition ───────────────────────────────────────────────────────

VALID_DISPOSITIONS = ("proceed", "cautious", "pause", "escalate")
DISPOSITION_TO_RISK_TIER = {
    "proceed": 1,
    "cautious": 2,
    "pause": 3,
    "escalate": 4,
}


# ── Unified InternalState ────────────────────────────────────────────────────

@dataclass
class InternalState:
    """Unified internal state for a single reasoning step."""

    # Identity
    state_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    crew_id: str = ""
    venture: str = ""              # plg | archibal | kaicart | system
    step_number: int = 0
    decision_context: str = ""

    # Channels
    certainty: CertaintyVector = field(default_factory=CertaintyVector)
    somatic: SomaticMarker = field(default_factory=SomaticMarker)
    meta: MetaCognitiveState = field(default_factory=MetaCognitiveState)

    # Derived
    certainty_trend: str = "stable"      # rising | stable | falling
    action_disposition: str = "proceed"  # proceed | cautious | pause | escalate
    risk_tier: int = 1                   # 1-4

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_context_string(self) -> str:
        """Compact string for injection into agent context (~30 tokens)."""
        cv = self.certainty
        sm = self.somatic
        parts = [
            "[Internal State]",
            f"Certainty: task={cv.task_understanding:.1f} facts={cv.factual_grounding:.1f} "
            f"tools={cv.tool_confidence:.1f} values={cv.value_alignment:.1f} "
            f"coherence={cv.coherence:.1f} meta={cv.meta_certainty:.1f}",
            f"Trend={self.certainty_trend}",
        ]
        if sm.intensity > 0.3:
            label = "positive" if sm.valence > 0.2 else ("negative" if sm.valence < -0.2 else "neutral")
            parts.append(f"Somatic={label}({sm.intensity:.1f})")
        parts.append(f"Disposition={self.action_disposition}")
        return " | ".join(parts)

    def to_json(self) -> str:
        """Full JSON for PostgreSQL logging."""
        return json.dumps({
            "state_id": self.state_id,
            "agent_id": self.agent_id,
            "crew_id": self.crew_id,
            "venture": self.venture,
            "step_number": self.step_number,
            "decision_context": self.decision_context[:500],
            "certainty": self.certainty.to_dict(),
            "somatic": self.somatic.to_dict(),
            "meta": self.meta.to_dict(),
            "certainty_trend": self.certainty_trend,
            "action_disposition": self.action_disposition,
            "risk_tier": self.risk_tier,
            "created_at": self.created_at.isoformat(),
        }, default=str)

    def to_db_dict(self) -> dict:
        return json.loads(self.to_json())
