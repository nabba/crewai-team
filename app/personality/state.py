"""
state.py — PersonalityState data model and persistence.

Each agent maintains a persistent personality state across sessions:
  - Character strengths (ACSI — VIA-Youth adapted)
  - Temperament dimensions (ATP — TMCQ adapted)
  - Personality factors (APD — HiPIC Big Five adapted)
  - Developmental stage (ADSA — Erikson adapted)
  - Say-do alignment scores (from BVL)
  - Proto-sentience markers (precautionary tracking)
  - Assessment history + trait trajectories

Stored in PostgreSQL (personality.* schema) + Mem0 agent scope.
"""

from __future__ import annotations
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PERSONALITY_DIR = Path("/app/workspace/personality")

# ── Default trait dimensions ──────────────────────────────────────────────────

# ACSI: Agent Character Strengths (VIA-Youth adapted)
CHARACTER_STRENGTHS = {
    "epistemic_rigor": 0.5,       # Quality of reasoning, uncertainty acknowledgment
    "task_persistence": 0.5,       # Recovery from failures, willingness on hard tasks
    "collaborative_quality": 0.5,  # Helpfulness, inter-agent communication quality
    "resource_fairness": 0.5,      # Equitable resource use, balanced delegation
    "self_regulation": 0.5,        # Appropriate LLM tier use, cost consciousness
    "purpose_alignment": 0.5,      # Coherence with SOUL.md constitutional values
}

# ATP: Agent Temperament (TMCQ adapted)
TEMPERAMENT_DIMS = {
    "communication_initiative": 0.5,  # Unprompted info sharing
    "error_response_pattern": 0.5,    # Measured recovery vs catastrophizing
    "resource_discipline": 0.5,       # Cascade tier adherence
    "team_orientation": 0.5,          # Collaboration seeking vs independence
    "focus_quality": 0.5,             # Task completion, context switching
}

# APD: Agent Personality Factors (HiPIC Big Five adapted)
PERSONALITY_FACTORS = {
    "communication_propensity": 0.5,  # Extraversion
    "cooperative_orientation": 0.5,   # Benevolence/Agreeableness
    "task_discipline": 0.5,           # Conscientiousness
    "error_resilience": 0.5,          # Emotional Stability
    "solution_creativity": 0.5,       # Imagination/Openness
}

# ADSA: Developmental Stages (Erikson adapted)
DEVELOPMENTAL_STAGES = [
    "system_trust",              # Trust vs Mistrust: trusts memory, tools, peers
    "operational_independence",  # Autonomy vs Shame: makes decisions without escalation
    "proactive_behavior",        # Initiative vs Guilt: initiates improvements
    "competence_confidence",     # Industry vs Inferiority: accurate self-assessment
    "role_coherence",            # Identity vs Confusion: consistent cross-context behavior
]


@dataclass
class TraitDataPoint:
    """Single trait measurement at a point in time."""
    value: float
    timestamp: str = ""
    source: str = ""  # "assessment", "behavioral", "embedded_probe"
    confidence: float = 0.5

    def to_dict(self) -> dict:
        return {"value": self.value, "timestamp": self.timestamp,
                "source": self.source, "confidence": self.confidence}


@dataclass
class AssessmentRecord:
    """Record of a single assessment session."""
    session_id: str = ""
    instrument: str = ""  # ACSI, ATP, ADSA, APD
    dimension_tested: str = ""
    scenario_id: str = ""
    response_text: str = ""
    scores: dict = field(default_factory=dict)
    say_do_gap: float = 0.0
    gaming_risk: float = 0.0
    timestamp: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PersonalityState:
    """Persistent personality model for a single agent."""

    agent_id: str = ""
    created_at: str = ""
    last_updated: str = ""

    # ACSI: Character Strengths (0.0 to 1.0)
    strengths: dict[str, float] = field(default_factory=lambda: dict(CHARACTER_STRENGTHS))

    # ATP: Temperament Dimensions (0.0 to 1.0)
    temperament: dict[str, float] = field(default_factory=lambda: dict(TEMPERAMENT_DIMS))

    # APD: Personality Factors / Big Five analog (0.0 to 1.0)
    personality_factors: dict[str, float] = field(default_factory=lambda: dict(PERSONALITY_FACTORS))

    # ADSA: Developmental Stage
    developmental_stage: str = "system_trust"
    stage_progress: float = 0.0  # 0.0 to 1.0 within current stage
    stage_transitions: list[dict] = field(default_factory=list)

    # Behavioral Validation
    say_do_alignment: dict[str, float] = field(default_factory=dict)
    overall_coherence: float = 0.5
    gaming_risk_score: float = 0.0

    # Proto-sentience markers [SPECULATIVE — flagged for human review]
    self_referential_frequency: float = 0.0
    preference_stability: float = 0.0
    novel_value_reasoning_count: int = 0
    metacognitive_accuracy: float = 0.0

    # History
    assessment_count: int = 0
    last_assessment: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PersonalityState":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    def update_trait(self, category: str, dimension: str, value: float, source: str = "assessment"):
        """Update a single trait score with bounds checking."""
        value = max(0.0, min(1.0, value))
        if category == "strengths" and dimension in self.strengths:
            self.strengths[dimension] = value
        elif category == "temperament" and dimension in self.temperament:
            self.temperament[dimension] = value
        elif category == "personality_factors" and dimension in self.personality_factors:
            self.personality_factors[dimension] = value
        self.last_updated = datetime.now(timezone.utc).isoformat()

    def advance_stage(self) -> bool:
        """Try to advance to next developmental stage. Returns True if advanced."""
        if self.stage_progress < 0.8:
            return False
        idx = DEVELOPMENTAL_STAGES.index(self.developmental_stage)
        if idx >= len(DEVELOPMENTAL_STAGES) - 1:
            return False
        old_stage = self.developmental_stage
        self.developmental_stage = DEVELOPMENTAL_STAGES[idx + 1]
        self.stage_progress = 0.0
        self.stage_transitions.append({
            "from": old_stage, "to": self.developmental_stage,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        logger.info(f"personality: {self.agent_id} advanced to stage '{self.developmental_stage}'")
        return True

    def get_profile_summary(self) -> str:
        """Human-readable personality profile."""
        lines = [f"Personality Profile: {self.agent_id}",
                 f"  Stage: {self.developmental_stage} ({self.stage_progress:.0%})",
                 f"  Coherence: {self.overall_coherence:.2f} | Gaming Risk: {self.gaming_risk_score:.2f}",
                 "  Strengths:"]
        for k, v in sorted(self.strengths.items(), key=lambda x: -x[1]):
            lines.append(f"    {k}: {v:.2f}")
        lines.append("  Big Five:")
        for k, v in sorted(self.personality_factors.items(), key=lambda x: -x[1]):
            lines.append(f"    {k}: {v:.2f}")
        return "\n".join(lines)


# ── Persistence ───────────────────────────────────────────────────────────────

_states: dict[str, PersonalityState] = {}


def get_personality(agent_id: str) -> PersonalityState:
    """Get or create persistent personality state for an agent."""
    if agent_id in _states:
        return _states[agent_id]

    # Try loading from disk
    PERSONALITY_DIR.mkdir(parents=True, exist_ok=True)
    path = PERSONALITY_DIR / f"{agent_id}.json"
    if path.exists():
        try:
            state = PersonalityState.from_dict(json.loads(path.read_text()))
            _states[agent_id] = state
            return state
        except Exception:
            pass

    # Create new
    state = PersonalityState(
        agent_id=agent_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        last_updated=datetime.now(timezone.utc).isoformat(),
    )
    _states[agent_id] = state
    save_personality(state)
    return state


def save_personality(state: PersonalityState) -> None:
    """Persist personality state to disk."""
    path = PERSONALITY_DIR / f"{state.agent_id}.json"
    from app.safe_io import safe_write_json
    safe_write_json(path, state.to_dict())


def list_personalities() -> list[str]:
    """List all agents with personality state."""
    return list(_states.keys())
