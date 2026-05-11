"""Person centrality — Level 2 of person-correlation.

PROGRAM §42 (2026-05-11) — Q4.2 Level 2.

Computes per-person scalar in [0, 1] using an OPERATOR-CHOSEN formula.
Never auto-tuned, never learned. Three formulas:

  * ``frequency``         — normalized appearance count, 30d window
  * ``recency-weighted``  — exp-decay weighted, 30d half-life
  * ``cross-modal``       — modality_count × log10(total+1), capped

Critical Goodhart guard: the React surface displays scores but
**never sorts by them**. Sorting by score is the gateway to
"optimize against centrality" Goodhart trap. List sorts by last_seen.

Master switches:
  * ``person_correlation_enabled`` (L1 master, also required for L2)
  * ``person_centrality_enabled`` (L2 master)

The formula choice is operator-pickable via
``person_centrality_formula`` runtime setting (string enum).
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


FORMULA_FREQUENCY = "frequency"
FORMULA_RECENCY = "recency_weighted"
FORMULA_CROSS_MODAL = "cross_modal"

VALID_FORMULAS = {FORMULA_FREQUENCY, FORMULA_RECENCY, FORMULA_CROSS_MODAL}


def _enabled() -> bool:
    """Both L1 + L2 must be on."""
    try:
        from app.runtime_settings import (
            get_person_correlation_enabled,
            get_person_centrality_enabled,
        )
        return get_person_correlation_enabled() and get_person_centrality_enabled()
    except Exception:
        return False


def _formula() -> str:
    try:
        from app.runtime_settings import get_person_centrality_formula
        f = get_person_centrality_formula()
        return f if f in VALID_FORMULAS else FORMULA_FREQUENCY
    except Exception:
        return FORMULA_FREQUENCY


def _score_frequency(person: dict[str, Any], max_total: int) -> float:
    """Just normalized appearance count."""
    total = int(person.get("total_occurrences") or 0)
    if max_total <= 0:
        return 0.0
    return min(1.0, total / max_total)


def _score_recency_weighted(person: dict[str, Any], now: datetime) -> float:
    """Exp-decay weighted by last_seen. 30d half-life."""
    last = person.get("last_seen") or ""
    if not last:
        return 0.0
    try:
        ts = datetime.fromisoformat(last.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    age_days = max(0.0, (now - ts).total_seconds() / 86400.0)
    decay = math.exp(-math.log(2) * age_days / 30.0)
    total = int(person.get("total_occurrences") or 0)
    # Scale by a soft saturation curve: log(total+1) / log(20) caps at 1
    volume = min(1.0, math.log(total + 1) / math.log(20))
    return round(decay * volume, 4)


def _score_cross_modal(person: dict[str, Any]) -> float:
    """modality_factor (saturates at 4) × volume_factor."""
    modality_count = int(person.get("modality_count") or 0)
    total = int(person.get("total_occurrences") or 0)
    if modality_count <= 0 or total <= 0:
        return 0.0
    modality_factor = min(1.0, modality_count / 4.0)
    volume_factor = min(1.0, math.log(total + 1) / math.log(20))
    return round(modality_factor * volume_factor, 4)


def compute_centrality() -> dict[str, Any]:
    """For each non-muted person, compute the score using the
    operator-chosen formula. Returns dict suitable for the REST
    surface. No-op when L2 master switch is OFF."""
    if not _enabled():
        return {"scores": [], "enabled": False, "formula": _formula()}

    try:
        from app.companion.person_model import current_profile
        prof = current_profile() or {}
    except Exception:
        return {"scores": [], "enabled": True, "formula": _formula(), "error": "profile unavailable"}

    people = prof.get("people") or []
    if not people:
        return {"scores": [], "enabled": True, "formula": _formula()}

    formula = _formula()
    now = datetime.now(timezone.utc)

    # Pre-compute max_total for frequency normalization.
    max_total = max((int(p.get("total_occurrences") or 0) for p in people), default=1)

    scores: list[dict[str, Any]] = []
    for p in people:
        if formula == FORMULA_FREQUENCY:
            s = _score_frequency(p, max_total)
        elif formula == FORMULA_RECENCY:
            s = _score_recency_weighted(p, now)
        elif formula == FORMULA_CROSS_MODAL:
            s = _score_cross_modal(p)
        else:
            s = 0.0
        scores.append({
            "person_id": p.get("person_id"),
            "display_names": p.get("display_names") or [],
            "last_seen": p.get("last_seen"),
            "score": s,
            "total_occurrences": p.get("total_occurrences"),
            "modality_count": p.get("modality_count"),
        })

    # CRITICAL Goodhart guard: sort by last_seen, NOT by score.
    # The operator should see "this is what the math says" not
    # "these are your most important people."
    scores.sort(key=lambda s: s.get("last_seen") or "", reverse=True)

    return {
        "scores": scores,
        "enabled": True,
        "formula": formula,
        "caveat": (
            "Scores are computed from the formula you chose. They are "
            "what the math says, not what you should do. List is sorted "
            "by last_seen, not by score — deliberate Goodhart guard."
        ),
        "generated_at": now.isoformat(),
    }


def centrality_for(person_id: str) -> float:
    """Single-person lookup. Used by arbiter for salience boost.
    Returns 0.0 when L2 off or person not found."""
    if not _enabled():
        return 0.0
    pid = (person_id or "").strip().lower()
    if not pid:
        return 0.0
    try:
        from app.companion.person_model import current_profile
        prof = current_profile() or {}
    except Exception:
        return 0.0
    formula = _formula()
    now = datetime.now(timezone.utc)
    people = prof.get("people") or []
    target = next((p for p in people if p.get("person_id") == pid), None)
    if target is None:
        return 0.0
    max_total = max((int(p.get("total_occurrences") or 0) for p in people), default=1)
    if formula == FORMULA_FREQUENCY:
        return _score_frequency(target, max_total)
    if formula == FORMULA_RECENCY:
        return _score_recency_weighted(target, now)
    if formula == FORMULA_CROSS_MODAL:
        return _score_cross_modal(target)
    return 0.0
