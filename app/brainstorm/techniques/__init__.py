"""Technique registry — name → :class:`Technique` instance."""

from __future__ import annotations

from app.brainstorm.techniques.base import (
    LinearTechnique,
    Step,
    Technique,
    TechniqueState,
)
from app.brainstorm.techniques.concept_blend import ConceptBlendTechnique
from app.brainstorm.techniques.crazy_8s import CrazyEightsTechnique
from app.brainstorm.techniques.how_might_we import HowMightWeTechnique
from app.brainstorm.techniques.rapid_ideation import RapidIdeationTechnique
from app.brainstorm.techniques.reverse import ReverseBrainstormingTechnique
from app.brainstorm.techniques.scamper import ScamperTechnique
from app.brainstorm.techniques.six_hats import SixHatsTechnique
from app.brainstorm.techniques.starbursting import StarburstingTechnique

_TECHNIQUES: dict[str, Technique] = {
    t.name: t
    for t in (
        ScamperTechnique(),
        SixHatsTechnique(),
        HowMightWeTechnique(),
        ReverseBrainstormingTechnique(),
        CrazyEightsTechnique(),
        RapidIdeationTechnique(),
        StarburstingTechnique(),
        # Q11.2 (PROGRAM §46.19) — 8th technique
        ConceptBlendTechnique(),
    )
}


def registry() -> dict[str, Technique]:
    """Return the immutable technique registry (name → instance)."""
    return dict(_TECHNIQUES)


def get(name: str) -> Technique | None:
    """Look up a technique by short name. Returns None if unknown."""
    return _TECHNIQUES.get(name)


def names() -> list[str]:
    return sorted(_TECHNIQUES.keys())


def menu() -> str:
    """Human-readable list of techniques for menu display."""
    lines = ["Available techniques:"]
    for n in names():
        t = _TECHNIQUES[n]
        lines.append(f"  • {n} — {t.title}: {t.description}")
    return "\n".join(lines)


__all__ = [
    "LinearTechnique",
    "Step",
    "Technique",
    "TechniqueState",
    "registry",
    "get",
    "names",
    "menu",
]
