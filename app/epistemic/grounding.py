"""Pluggable grounding-signal provider.

The :class:`~app.epistemic.detectors.realtime.RegisterConfidenceMismatchDetector`
needs to know the agent's *felt* certainty (``factual_grounding`` ∈
[0.0, 1.0]) at the moment a claim is emitted. The default provider
returns ``None`` — meaning the affect layer isn't wired and the
detector silently doesn't fire. When the affect layer is wired
(Phase 5+ in the rollout plan), it calls
:func:`set_grounding_provider` with a function that reads the live
affect state.

This module is the *only* coupling point between the epistemic and
affect layers. Both can be developed and tested independently.
"""
from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)


GroundingProvider = Callable[[], float | None]


def _default_provider() -> float | None:
    """Default: no grounding signal available.

    The detector treats ``None`` as "skip this check" — it does NOT
    treat it as low grounding (which would fire the bias on every
    declarative load-bearing claim).
    """
    return None


_provider: GroundingProvider = _default_provider


def set_grounding_provider(provider: GroundingProvider) -> None:
    """Replace the current grounding provider.

    Called by ``app.affect`` (Phase 5) to wire the live signal. The
    function must be cheap (target: < 1 ms) — it runs on every claim
    emission via the realtime meta-hook.
    """
    global _provider
    _provider = provider


def factual_grounding() -> float | None:
    """Current factual_grounding signal, or ``None`` if not available.

    Swallows exceptions: if the provider raises, the realtime gate
    must not break. Logs at DEBUG so missing or buggy providers are
    visible without spamming the console.
    """
    try:
        return _provider()
    except Exception as exc:
        logger.debug("epistemic grounding provider raised: %s", exc)
        return None


def _reset_for_tests() -> None:
    """Restore the default provider. Tests only."""
    global _provider
    _provider = _default_provider
