"""
personality_workspace.py — PDS-driven workspace parameter adaptation.

Maps personality traits to workspace parameters so each agent's cognitive
style influences how information competes for attention:
  - focus_quality → workspace capacity (narrow vs broad attention)
  - solution_creativity → novelty floor (exploration vs exploitation)
  - error_resilience → consumption decay (revisit vs move on)
  - developmental stage → capacity bonus (experience broadens capacity)

Homeostasis temporarily overrides personality-derived capacity:
  - High frustration → broaden search (capacity+2)
  - Low energy → narrow focus (capacity-2)

DGM Safety: personality traits are READ-ONLY in this path. No modification.
All computation is pure arithmetic — no LLM calls, <1ms latency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Capacity safety bounds (DGM immutable)
MIN_CAPACITY = 2
MAX_CAPACITY = 9
DEFAULT_CAPACITY = 5

# Developmental stage → capacity adjustment
_STAGE_BONUS = {
    "system_trust": -1,             # Stage 1: conservative, narrow focus
    "operational_independence": 0,   # Stage 2: default
    "proactive_behavior": 0,        # Stage 3: default
    "competence_confidence": 1,     # Stage 4: experienced, broader
    "role_coherence": 1,            # Stage 5: mature, broader
}


@dataclass
class WorkspaceProfile:
    """Personality-derived workspace parameters for one agent."""
    capacity: int = DEFAULT_CAPACITY
    novelty_floor_pct: float = 0.20
    consumption_decay: float = 0.50
    source_traits: dict = field(default_factory=dict)


def compute_workspace_profile(agent_id: str = "commander") -> WorkspaceProfile:
    """Derive workspace parameters from personality state + homeostasis.

    Pure arithmetic on existing state — no LLM calls, <1ms.
    Falls back to defaults on any failure.
    """
    profile = WorkspaceProfile()

    # ── Step 1: Load personality traits ──────────────────────────────────
    try:
        from app.personality.state import get_personality
        personality = get_personality(agent_id)

        focus = personality.temperament.get("focus_quality", 0.5)
        creativity = personality.personality_factors.get("solution_creativity", 0.5)
        resilience = personality.personality_factors.get("error_resilience", 0.5)
        stage = personality.developmental_stage

        profile.source_traits = {
            "focus_quality": round(focus, 3),
            "solution_creativity": round(creativity, 3),
            "error_resilience": round(resilience, 3),
            "developmental_stage": stage,
        }

        # ── Step 2: Map traits → workspace parameters ───────────────────
        # focus_quality → capacity (high focus = narrow, low focus = broad)
        if focus > 0.7:
            profile.capacity = 3
        elif focus < 0.4:
            profile.capacity = 7
        else:
            profile.capacity = DEFAULT_CAPACITY

        # solution_creativity → novelty floor
        if creativity > 0.7:
            profile.novelty_floor_pct = 0.30
        elif creativity < 0.4:
            profile.novelty_floor_pct = 0.10
        else:
            profile.novelty_floor_pct = 0.20

        # error_resilience → consumption decay
        if resilience > 0.7:
            profile.consumption_decay = 0.30  # Revisits old items
        elif resilience < 0.4:
            profile.consumption_decay = 0.70  # Moves on quickly
        else:
            profile.consumption_decay = 0.50

        # ── Step 3: Developmental stage bonus ────────────────────────────
        stage_adj = _STAGE_BONUS.get(stage, 0)
        profile.capacity += stage_adj
        profile.source_traits["stage_adjustment"] = stage_adj

    except Exception:
        logger.debug("personality_workspace: personality load failed, using defaults",
                     exc_info=True)

    # ── Step 4: Homeostasis override ─────────────────────────────────────
    try:
        from app.subia.homeostasis.state import get_state
        homeo = get_state()
        frustration = homeo.get("frustration", 0.1)
        energy = homeo.get("cognitive_energy", 0.7)

        homeo_adj = 0
        if frustration > 0.7:
            homeo_adj += 2  # Broaden search when stuck
        if energy < 0.3:
            homeo_adj -= 2  # Narrow when tired

        if homeo_adj != 0:
            profile.capacity += homeo_adj
            profile.source_traits["homeostasis_adjustment"] = homeo_adj
    except Exception:
        pass

    # ── Step 5: Clamp to safety bounds ───────────────────────────────────
    profile.capacity = max(MIN_CAPACITY, min(MAX_CAPACITY, profile.capacity))

    logger.debug(
        f"personality_workspace: {agent_id} → capacity={profile.capacity} "
        f"novelty={profile.novelty_floor_pct} decay={profile.consumption_decay} "
        f"traits={profile.source_traits}"
    )
    return profile
