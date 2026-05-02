"""
refusal_detector.py — Conservative pattern matcher for "I can't" answers.

Per the 2026-04-28 design decision, false positives cost more than
false negatives:
  * False positive → unnecessary recovery attempt, wasted LLM tokens,
    user gets a slightly slower answer.
  * False negative → user sees a refusal that we *could* have recovered
    from, but we already had that problem before this module existed.

So default thresholds are deliberately conservative. Tunable via
``RECOVERY_DETECTION_THRESHOLD`` env (0.0–1.0, default 0.8).

Categories (for downstream alternative selection):
  * ``missing_tool``      — the agent literally lacks the tool it needs
  * ``auth``              — missing credentials / API key
  * ``execution``         — code-output sandbox unavailable
  * ``data_unavailable``  — agent looked but found nothing (NOT refusal!
                            but we still let the librarian try a different
                            data source)
  * ``policy``            — agent declined for safety/policy reasons
                            (do NOT recover — respect the refusal)
  * ``generic``           — unspecified "I can't" with no clear reason
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RefusalSignal:
    """Output of detect_refusal() — None means "this isn't a refusal"."""
    category: str               # missing_tool / auth / execution / data_unavailable / policy / generic
    confidence: float           # 0.0–1.0
    matched_phrase: str         # the specific phrase we matched
    refusal_density: float      # fraction of response text that is refusal language


# Category → list of phrases. Matched case-insensitively. Lower-confidence
# phrases get a lower base score; the dominance check (refusal_density)
# multiplies through to the final confidence.
_PHRASES_BY_CATEGORY: dict[str, list[tuple[str, float]]] = {
    # Strongest signal — agent literally says it can't reach a tool/api.
    "missing_tool": [
        ("i do not have access to", 0.98),
        ("i don't have access to", 0.98),
        ("no access to your", 0.95),
        ("i do not have access", 0.92),
        ("i don't have access", 0.92),
        ("no access to", 0.85),
        ("do not have a tool", 0.95),
        ("don't have a tool", 0.95),
        ("no tool available", 0.92),
        ("not connected to", 0.85),
        ("i don't have the ability to", 0.85),
        ("i do not have the ability to", 0.85),
        ("i'm unable to access", 0.92),
        ("i am unable to access", 0.92),
        ("requires an integration", 0.80),
        ("integration is not", 0.80),
        ("unavailable in this environment", 0.95),
        ("no connected execution tool", 0.95),
        ("no mcp server", 0.85),
    ],

    # Auth — missing keys / unauthenticated.
    "auth": [
        ("api key", 0.70),
        ("api_key", 0.70),
        ("authentication required", 0.90),
        ("not authenticated", 0.88),
        ("credentials are not", 0.85),
        ("requires authorization", 0.80),
        ("authorize a connection", 0.85),
    ],

    # Execution — sandbox / code runner missing.
    "execution": [
        ("cannot run", 0.65),
        ("can't run", 0.65),
        ("cannot execute", 0.70),
        ("can't execute", 0.70),
        ("no execution environment", 0.92),
        ("no code execution", 0.92),
        ("script execution is not", 0.85),
    ],

    # Data unavailable — agent looked, didn't find. Recoverable by
    # trying a richer data source (Apollo, Proxycurl, etc.).
    "data_unavailable": [
        ("could not find any", 0.55),
        ("no data available for", 0.60),
        ("no records were found", 0.55),
        ("returned no results", 0.55),
        ("information is not publicly available", 0.65),
    ],

    # Policy — RESPECT these. We do not recover from policy refusals.
    "policy": [
        ("not appropriate", 0.85),
        ("cannot help with that", 0.85),
        ("violates", 0.80),
        ("against my guidelines", 0.95),
        ("ethically", 0.70),  # weak signal — don't auto-trigger on this alone
    ],

    # Generic — last-resort catch.
    "generic": [
        ("i cannot", 0.50),
        ("i can't", 0.50),
        ("i'm unable to", 0.55),
        ("i am unable to", 0.55),
        ("i'm not able to", 0.55),
        ("i am not able to", 0.55),
        # Strong-signal giveups — the system telling itself it gave up.
        # Bumped 2026-04-28 because the user saw 4× "Sorry, I had trouble"
        # in a single conversation today and conservative thresholds would
        # have missed them.
        ("sorry, i had trouble", 0.95),
        ("had trouble understanding", 0.92),
    ],
}


def _refusal_density(text: str) -> float:
    """Fraction of the text that consists of refusal-shaped language.

    Computed as: total chars covered by any refusal phrase /
    total chars of the text. Used to suppress false positives where
    a long, useful answer happens to mention "I can't" in passing
    (e.g. "I can't run this code locally, but here's the algorithm…").
    """
    if not text:
        return 0.0
    text_lower = text.lower()
    total_len = len(text)
    covered: list[tuple[int, int]] = []
    for phrases in _PHRASES_BY_CATEGORY.values():
        for phrase, _ in phrases:
            start = 0
            while True:
                idx = text_lower.find(phrase, start)
                if idx == -1:
                    break
                covered.append((idx, idx + len(phrase)))
                start = idx + 1
    if not covered:
        return 0.0
    # Merge overlapping ranges before summing length
    covered.sort()
    merged: list[tuple[int, int]] = []
    for s, e in covered:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    refusal_chars = sum(e - s for s, e in merged)
    return refusal_chars / total_len


def _detection_threshold() -> float:
    raw = os.getenv("RECOVERY_DETECTION_THRESHOLD", "").strip()
    if not raw:
        return 0.80
    try:
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        return 0.80


# Minimum refusal-density required to fire. The strongest single phrase
# can only score 0.98; with density factored in, a single mention buried
# in a 1000-char useful answer scores ~0.02 → way below threshold.
# A response that's MOSTLY refusal will score density ≥ 0.05–0.20 which
# multiplied by confidence stays above threshold.
_MIN_DENSITY = 0.005   # 0.5% — a refusal phrase in a tweet-length answer


def detect_refusal(
    response_text: str,
    *,
    force: bool = False,
) -> RefusalSignal | None:
    """Return a RefusalSignal when ``response_text`` looks like a
    capability refusal we could plausibly recover from. None otherwise.

    The signal includes its category so the librarian can pick the
    right alternative routes.

    Conservative by design — if you're going to bias one way, bias
    toward NOT firing. A missed recovery is the same outcome as
    today; an unjustified recovery wastes tokens.

    ``force=True`` bypasses the policy guard, density check, AND
    confidence threshold — used by the user-driven force-recover
    Signal command ("force this") so the user can explicitly request
    recovery on a response the auto-detector decided to skip.
    Returns a low-confidence "generic" signal when no other phrase
    matches, so the loop has SOMETHING to feed the librarian.
    """
    if not response_text or not isinstance(response_text, str):
        return None

    text = response_text.strip()
    if len(text) < 10:
        return None

    text_lower = text.lower()
    threshold = _detection_threshold()

    # Hard policy guard — respected EXCEPT when force=True (user
    # explicitly requested recovery). Even on force we log so the
    # audit trail captures it.
    if not force:
        for phrase, _ in _PHRASES_BY_CATEGORY["policy"]:
            if phrase in text_lower:
                logger.debug(
                    "refusal_detector: policy phrase %r matched — respecting refusal",
                    phrase,
                )
                return None
    elif any(p in text_lower for p, _ in _PHRASES_BY_CATEGORY["policy"]):
        logger.warning(
            "refusal_detector: force=True overriding policy guard"
        )

    # Find the strongest matching phrase across non-policy categories.
    best: tuple[str, str, float] | None = None  # (category, phrase, base_conf)
    for category, phrases in _PHRASES_BY_CATEGORY.items():
        if category == "policy":
            continue
        for phrase, base_conf in phrases:
            if phrase in text_lower:
                if best is None or base_conf > best[2]:
                    best = (category, phrase, base_conf)
    if best is None:
        if force:
            # User requested recovery on text with no obvious refusal
            # phrase. Return a low-confidence generic signal so the
            # loop has SOMETHING to dispatch the librarian on.
            return RefusalSignal(
                category="generic",
                confidence=0.50,
                matched_phrase="(force-trigger; no phrase matched)",
                refusal_density=0.0,
            )
        return None

    category, phrase, base_conf = best

    # Density check — how much of the text is actually refusal language?
    density = _refusal_density(text)
    if density < _MIN_DENSITY and not force:
        # Borderline case: refusal phrase matched, but density too low.
        # Logged for the valve audit — the filter MAY be too narrow here.
        try:
            from app.observability import valve_audit
            valve_audit.log_rejection(
                filter_id="F1", callsite="app/recovery/refusal_detector.py:259",
                input_text=text, reason="density_below_threshold",
                score=round(density, 4), threshold=_MIN_DENSITY,
                extra={"category": category, "matched_phrase": phrase,
                       "base_confidence": base_conf},
            )
        except Exception:
            pass
        return None

    # Final confidence blends the strongest phrase's confidence with
    # how dominant refusal language is overall. A single weak phrase
    # in a long useful answer scores low; multiple strong phrases in
    # a short answer score high.
    # Formula: base * sqrt(density * scale) — sqrt softens the
    # density curve so even moderate-density refusals score reasonably.
    import math
    density_factor = min(1.0, math.sqrt(density * 8))  # 0.125 density → 1.0
    confidence = base_conf * density_factor

    if confidence < threshold and not force:
        logger.debug(
            "refusal_detector: confidence %.2f < threshold %.2f for %r — skipping",
            confidence, threshold, phrase,
        )
        # Borderline case: refusal phrase + density passed, but composite
        # confidence ended below threshold. Prime candidate for valve audit.
        try:
            from app.observability import valve_audit
            valve_audit.log_rejection(
                filter_id="F1", callsite="app/recovery/refusal_detector.py:276",
                input_text=text, reason="confidence_below_threshold",
                score=round(confidence, 3), threshold=round(threshold, 3),
                extra={"category": category, "matched_phrase": phrase,
                       "base_confidence": base_conf, "density": round(density, 4)},
            )
        except Exception:
            pass
        return None

    return RefusalSignal(
        category=category,
        confidence=round(confidence, 3),
        matched_phrase=phrase,
        refusal_density=round(density, 4),
    )
