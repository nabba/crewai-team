"""Affect ↔ epistemic bridge.

The single coupling point between the epistemic and affect subsystems.
Mirrors the strict isolation pattern from
:mod:`app.epistemic.grounding`: the epistemic layer reads affect via
*one* function (:func:`compute_factual_grounding`), and the affect
layer reads epistemic events via *one* observer
(:func:`_emit_cognitive_failure_salience`).

Bootstrap (called from ``main.py`` once both subsystems are loaded):

    >>> from app.epistemic.affect_bridge import bootstrap
    >>> bootstrap()

After bootstrap:

* :class:`RegisterConfidenceMismatchDetector` starts seeing live
  factual_grounding values from the affect layer's ``AffectState.controllability``
  (which is itself computed from ``state.certainty.adjusted_certainty``).
* High-severity bias matches emit ``SalienceEvent(kind="cognitive_failure")``
  into the affect layer's narrative-self pipeline. The episode synthesizer
  (:mod:`app.affect.episodes`) then weaves them into daily chapters with
  aviation-post-mortem framing — see the prompt extension in episodes.py.

If the affect layer isn't importable (degraded environment), the
bootstrap is a no-op and the realtime gate continues to function with
``factual_grounding=None`` (i.e. the register_confidence_mismatch
detector silently skips, which is the safe default).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.affect.schemas import AffectState
    from app.epistemic.biases import BiasMatch
    from app.epistemic.ledger import Claim, Ledger

logger = logging.getLogger(__name__)


# Attractors the affect layer uses to label "stuck" or "panicked" states.
# When the agent is in one of these, even a high controllability reading
# is suspect — the felt context says something is off.
_LOW_GROUNDING_ATTRACTORS = frozenset({
    "distress",      # negative valence + high arousal + low controllability
    "frozen",        # stuck attractor (rare but real)
    "depletion",     # low compute reserve
    "overwhelm",     # high novelty pressure
})

# Severity threshold above which a bias match emits a cognitive_failure
# salience event. LOW/MEDIUM matches are noise for the narrative track.
_SALIENCE_SEVERITY_FLOOR = "high"  # compared via severity_rank ordering


def compute_factual_grounding(state: "AffectState") -> float:
    """Derive ``factual_grounding ∈ [0.0, 1.0]`` from an AffectState.

    The primary signal is ``state.controllability``, which the affect
    layer computes from ``state.certainty.adjusted_certainty`` — the
    most direct expression of "how grounded am I in the evidence I
    have?".

    Adjustments:
      * Stuck attractors (distress, frozen, depletion, overwhelm) cap
        grounding at 0.5 regardless of controllability — felt context
        says the system is not in a state to assess its own grounding.

    Pure function. No side effects. Cheap (target < 1 ms).
    """
    base = max(0.0, min(1.0, float(state.controllability)))
    if state.attractor in _LOW_GROUNDING_ATTRACTORS:
        return min(0.5, base)
    return base


def live_factual_grounding() -> float | None:
    """Read the live AffectState and compute factual_grounding.

    Returns ``None`` if no affect snapshot is available (cold start,
    affect layer disabled). The realtime detector treats ``None`` as
    "skip this check" — it does NOT default to low grounding, which
    would fire spurious biases on every claim.
    """
    try:
        from app.affect.core import latest_affect
    except ImportError:
        return None
    try:
        state = latest_affect()
    except Exception as exc:
        logger.debug("epistemic affect_bridge: latest_affect raised: %s", exc)
        return None
    if state is None:
        return None
    return compute_factual_grounding(state)


def _emit_cognitive_failure_salience(
    matches: "list[BiasMatch]",
    claim: "Claim",
    ledger: "Ledger",
) -> None:
    """Match observer: emit a SalienceEvent for high-severity firings.

    Selects the highest-severity match (ties broken by first-seen order)
    and writes one event into the affect salience deque. The episode
    synthesizer will pick it up on its next cycle.

    Best-effort — if the affect layer isn't importable or the salience
    module rejects the event, we log at DEBUG and move on. The matches
    are already persisted in ``epistemic_bias_matches`` regardless.
    """
    if not matches:
        return

    try:
        from app.affect.salience import SalienceEvent, record
        from app.affect.schemas import utc_now_iso
        from app.epistemic.biases import severity_rank, Severity
    except ImportError:
        return

    # Pick the worst match. The post-mortem (Phase 4) sees ALL matches;
    # salience emits ONE event per claim emission so the affect deque
    # doesn't drown in low-signal noise.
    worst = max(matches, key=lambda m: severity_rank(m.severity))
    if severity_rank(worst.severity) < severity_rank(Severity.HIGH):
        return  # below floor — leave it to post-mortem

    affect_severity = "critical" if worst.severity is Severity.CRITICAL else "warn"
    detail = (
        f"epistemic: {worst.bias_id} on claim {claim.claim_id} "
        f"(agent={claim.agent_role}, register={claim.register.value})"
    )

    try:
        record(SalienceEvent(
            kind="cognitive_failure",
            detail=detail,
            valence=0.0,
            arousal=0.0,
            controllability=0.0,
            attractor="neutral",
            severity=affect_severity,
            ts=utc_now_iso(),
        ))
    except Exception as exc:
        logger.debug(
            "epistemic affect_bridge: salience record failed: %s", exc,
        )


def bootstrap() -> dict[str, bool]:
    """Wire the bridge: grounding provider + match observer.

    Idempotent — safe to call multiple times. Returns a small status
    dict describing which side effects took hold so the caller can
    log the outcome.

    Called from :mod:`app.main` after the affect router is mounted.
    """
    grounding_wired = False
    salience_wired = False

    # Wire grounding provider — the realtime detector starts seeing
    # live values immediately.
    try:
        from app.epistemic.grounding import set_grounding_provider
        set_grounding_provider(live_factual_grounding)
        grounding_wired = True
    except Exception as exc:
        logger.warning(
            "epistemic affect_bridge: grounding wiring failed: %s", exc,
        )

    # Register the cognitive_failure salience observer.
    try:
        from app.epistemic.detectors import register_match_observer
        register_match_observer(_emit_cognitive_failure_salience)
        salience_wired = True
    except Exception as exc:
        logger.warning(
            "epistemic affect_bridge: salience observer wiring failed: %s",
            exc,
        )

    logger.info(
        "epistemic affect_bridge bootstrap: grounding=%s salience=%s",
        grounding_wired, salience_wired,
    )
    return {
        "grounding_wired": grounding_wired,
        "salience_wired": salience_wired,
    }


def _unwire_for_tests() -> None:
    """Reset the bridge state. Tests only."""
    from app.epistemic.detectors import _MATCH_OBSERVERS
    from app.epistemic.grounding import _reset_for_tests as _reset_grounding

    try:
        _MATCH_OBSERVERS.remove(_emit_cognitive_failure_salience)
    except ValueError:
        pass
    _reset_grounding()
