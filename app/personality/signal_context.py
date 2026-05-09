"""Build a small per-message context for the concierge rewriter.

Phase B #5 (2026-05-09). Until this module landed, the concierge LLM
saw only the message-to-rewrite — it had no idea whether the user was
energized at 9am or winding down at 11pm. This module synthesizes a
lightweight ``ToneContext`` from three free signals:

  * **Affect state**       — current valence/arousal from the affect trace
  * **Circadian segment**  — local hour of day → morning/afternoon/evening/night
  * **Top interests**      — top-3 topics from the interest_model profile

The contract is minimal — three string-valued fields, easy for the
concierge prompt to absorb without bloating the system prompt or
making the LLM think harder than it needs to.

Each lookup is best-effort. If affect is unavailable, mood defaults
to "steady"; if interest_model has never run, the topics list is
empty. Failure to enrich never blocks the rewrite — the concierge
just runs without a context hint, same as before this module existed.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToneContext:
    mood: str = "steady"          # tired | steady | energized | agitated
    time_of_day: str = "day"      # morning | afternoon | evening | night
    top_interests: list[str] = field(default_factory=list)

    def to_prompt_line(self) -> str:
        """One short line embedded in the concierge user prompt."""
        bits = [self.time_of_day, self.mood]
        if self.top_interests:
            interest_str = ", ".join(self.top_interests[:3])
            bits.append(f"interested in {interest_str}")
        return " · ".join(bits)


# ── Mood from affect ──────────────────────────────────────────────────────


def _mood_from_affect() -> str:
    """Map AffectState V/A → coarse mood bucket. "steady" on any failure."""
    try:
        from app.affect.core import latest_affect
    except Exception:
        return "steady"
    try:
        state = latest_affect()
    except Exception:
        return "steady"
    if state is None:
        return "steady"

    valence = float(getattr(state, "valence", 0.0) or 0.0)
    arousal = float(getattr(state, "arousal", 0.0) or 0.0)

    # Quadrants (Russell circumplex, simplified):
    #   V > 0.2, A > 0.5  → energized
    #   V > 0.2, A ≤ 0.5  → steady (positive but calm)
    #   V < -0.2, A > 0.5 → agitated
    #   V < -0.2, A ≤ 0.5 → tired (low energy, low valence)
    #   else              → steady
    if valence > 0.2 and arousal > 0.5:
        return "energized"
    if valence < -0.2 and arousal > 0.5:
        return "agitated"
    if valence < -0.2 and arousal <= 0.5:
        return "tired"
    return "steady"


# ── Circadian segment ─────────────────────────────────────────────────────


_CIRCADIAN_TO_TIME_OF_DAY: dict[str, str] = {
    # SubIA circadian modes → coarse time-of-day strings the concierge
    # prompt understands. Phase F #4 (2026-05-09): the prior code
    # imported a non-existent ``current_segment`` and even when caught
    # by the broad except, the function name's intent was misleading.
    # Now uses the actual ``current_circadian_mode`` and maps its
    # output ("active_hours", "deep_work_hours", …) into our four
    # coarse buckets.
    "dawn_transition":     "morning",
    "active_hours":        "afternoon",
    "deep_work_hours":     "evening",
    "consolidation_hours": "night",
}


def _time_of_day() -> str:
    """Local hour → coarse segment. Tries SubIA's circadian helper first."""
    try:
        from app.subia.temporal.circadian import current_circadian_mode
        mode = (current_circadian_mode() or "").strip().lower()
        mapped = _CIRCADIAN_TO_TIME_OF_DAY.get(mode)
        if mapped:
            return mapped
    except Exception:
        pass
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 22:
        return "evening"
    return "night"


# ── Top interests ─────────────────────────────────────────────────────────


def _top_interests(n: int = 3) -> list[str]:
    """Read interest_model profile; return top-N topic names. [] on miss."""
    try:
        from app.companion.interest_model import current_profile
    except Exception:
        return []
    try:
        profile = current_profile()
    except Exception:
        return []
    topics = profile.get("topics", []) or []
    out: list[str] = []
    for t in topics[:n]:
        name = (t.get("name") or "").strip() if isinstance(t, dict) else str(t)
        if name:
            out.append(name)
    return out


# ── Public API ────────────────────────────────────────────────────────────


def build_context() -> ToneContext:
    """Return the current ToneContext. Always succeeds; degrades to defaults."""
    try:
        return ToneContext(
            mood=_mood_from_affect(),
            time_of_day=_time_of_day(),
            top_interests=_top_interests(),
        )
    except Exception:
        logger.debug("signal_context: build failed; using defaults", exc_info=True)
        return ToneContext()


def context_dict() -> dict[str, Any]:
    return asdict(build_context())
