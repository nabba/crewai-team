"""
homeostasis.py — System-level homeostatic self-regulation (Layer 6).

Tracks internal state variables that function as proto-emotions — not
subjective feelings, but functional signals that influence decision-making.
Analogous to Damasio's somatic markers.

NOTE ON CONFIDENCE: The `confidence` value here (0.0-1.0) is a SYSTEM-WIDE
proto-emotional signal reflecting overall operational confidence. It is
DIFFERENT from the per-agent confidence in agent_state.py:
  - homeostasis.confidence: system-wide, affects behavior (triggers critic
    review when >0.9, boosts exploration when <0.3). Updated per task.
  - agent_state.avg_confidence: per-agent rolling average derived from
    crew self-report confidence levels (low=0.3, medium=0.5, high=0.7).
Both are valid metrics at different scopes — they are NOT expected to match.

State variables drift toward immutable set-points after each update
(exponential decay toward target, like a biological thermostat).

Updated by the existing _post_crew_telemetry() hook — no new threads.
Read at task start and injected as a brief context note (~20 tokens).

Safety: TARGETS are code constants (immutable trust layer).
Agents cannot modify them. The self-improver cannot access this module's
internals — only the behavioral modifiers it produces.
"""

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_STATE_PATH = Path("/app/workspace/homeostasis.json")

# ── Immutable set-points (DGM safety: outside agent reach) ──────────────────
TARGETS = {
    "cognitive_energy": 0.7,    # Ideal: well-resourced
    "frustration": 0.1,         # Ideal: low
    "confidence": 0.65,         # Ideal: cautiously confident
    "curiosity": 0.5,           # Ideal: moderately exploratory
}

# Decay rate toward targets (0.0 = no regulation, 1.0 = instant snap)
_DECAY_RATE = 0.05

# ── Competing drives (Layer 9 lightweight) ──────────────────────────────────
# Named drives that the homeostatic state influences.
# These compete for behavioral control via the modifiers.
DRIVES = {
    "THOROUGHNESS": "Wants deeper research, more sources, higher quality",
    "EFFICIENCY": "Wants faster responses, fewer tokens, simpler approaches",
    "CAUTION": "Wants more verification, lower risk, critic review",
    "GROWTH": "Wants to learn, explore novel approaches, expand capabilities",
    "INTEGRITY": "Wants consistency with constitutional principles",
}


def _load() -> dict:
    """Load homeostatic state from disk."""
    try:
        if _STATE_PATH.exists():
            return json.loads(_STATE_PATH.read_text())
    except Exception:
        logger.debug("homeostasis: load failed", exc_info=True)
    return {
        "cognitive_energy": 0.7,
        "frustration": 0.1,
        "confidence": 0.5,
        "curiosity": 0.5,
        "tasks_since_rest": 0,
        "consecutive_failures": 0,
        "last_updated": "",
    }


def _save(state: dict) -> None:
    """Atomic write."""
    try:
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(_STATE_PATH.parent), suffix=".tmp")
        try:
            os.write(fd, json.dumps(state, indent=2).encode())
            os.close(fd)
            os.replace(tmp, str(_STATE_PATH))
        except Exception:
            try:
                os.close(fd)
            except Exception:
                pass
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
    except Exception:
        logger.debug("homeostasis: save failed", exc_info=True)


def _regulate(state: dict) -> dict:
    """Apply homeostatic regulation — drift toward set-points."""
    for key, target in TARGETS.items():
        current = state.get(key, target)
        # Exponential decay toward target
        state[key] = round(current + _DECAY_RATE * (target - current), 4)
    return state


def update_state(
    event_type: str,
    crew_name: str = "",
    success: bool = True,
    difficulty: int = 5,
) -> None:
    """Update homeostatic state based on a system event.

    Called from _post_crew_telemetry() after every crew execution.
    """
    state = _load()

    if event_type == "task_complete":
        state["tasks_since_rest"] = state.get("tasks_since_rest", 0) + 1

        if success:
            # Success: restore energy, reduce frustration, boost confidence
            state["cognitive_energy"] = min(1.0, state.get("cognitive_energy", 0.7) + 0.05)
            state["frustration"] = max(0.0, state.get("frustration", 0.1) - 0.05)
            state["confidence"] = min(1.0, state.get("confidence", 0.5) + 0.02)
            state["consecutive_failures"] = 0
        else:
            # Failure: deplete energy, increase frustration, reduce confidence
            state["cognitive_energy"] = max(0.0, state.get("cognitive_energy", 0.7) - 0.1)
            state["frustration"] = min(1.0, state.get("frustration", 0.1) + 0.15)
            state["confidence"] = max(0.0, state.get("confidence", 0.5) - 0.05)
            state["consecutive_failures"] = state.get("consecutive_failures", 0) + 1

        # High difficulty tasks increase curiosity (novel challenge)
        if difficulty >= 7:
            state["curiosity"] = min(1.0, state.get("curiosity", 0.5) + 0.03)

        # Many tasks without rest → energy depletion
        if state["tasks_since_rest"] > 20:
            state["cognitive_energy"] = max(0.0, state.get("cognitive_energy", 0.7) - 0.02)

    elif event_type == "rest":
        # Triggered by idle periods or explicit rest command
        state["cognitive_energy"] = min(1.0, state.get("cognitive_energy", 0.7) + 0.2)
        state["tasks_since_rest"] = 0

    # Apply homeostatic regulation (drift toward set-points)
    state = _regulate(state)
    state["last_updated"] = datetime.now(timezone.utc).isoformat()
    _save(state)


def get_state() -> dict:
    """Read current homeostatic state."""
    return _load()


def get_behavioral_modifiers() -> dict:
    """Return routing/behavior adjustments based on current state.

    These implement competing drives (Layer 9):
    - CAUTION drive: triggered by high frustration or overconfidence
    - EFFICIENCY drive: triggered by low energy
    - THOROUGHNESS drive: triggered by consecutive failures
    - INTEGRITY drive: always active via constitution (not computed here)
    """
    state = _load()
    modifiers = {}

    frustration = state.get("frustration", 0.1)
    energy = state.get("cognitive_energy", 0.7)
    confidence = state.get("confidence", 0.5)
    consecutive_failures = state.get("consecutive_failures", 0)

    # High frustration → CAUTION drive: switch approach, escalate model tier
    if frustration > 0.7:
        modifiers["strategy"] = "switch_approach"
        modifiers["tier_boost"] = 2
    elif frustration > 0.4:
        modifiers["tier_boost"] = 1

    # Low energy → EFFICIENCY drive: prefer simpler strategies
    if energy < 0.3:
        modifiers["prefer_simple"] = True
        modifiers["skip_parallel"] = True

    # Overconfidence → CAUTION drive: force critic review
    if confidence > 0.9:
        modifiers["force_critic_review"] = True

    # Consecutive failures → THOROUGHNESS drive: increase difficulty rating
    if consecutive_failures >= 3:
        modifiers["tier_boost"] = max(modifiers.get("tier_boost", 0), 2)
        modifiers["strategy"] = "switch_approach"

    return modifiers


def get_state_summary() -> str:
    """One-line state summary for context injection (~20 tokens)."""
    state = _load()
    energy = state.get("cognitive_energy", 0.7)
    conf = state.get("confidence", 0.5)
    frust = state.get("frustration", 0.1)
    curiosity = state.get("curiosity", 0.5)

    modifiers = get_behavioral_modifiers()
    mod_str = ", ".join(f"{k}={v}" for k, v in modifiers.items()) if modifiers else "none"

    return (
        f"SYSTEM STATE: energy={energy:.2f} confidence={conf:.2f} "
        f"frustration={frust:.2f} curiosity={curiosity:.2f} | "
        f"Active drives: {mod_str}\n"
    )
