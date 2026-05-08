"""Phase 14 hot-path hooks for the CIL loop.

Two narrow entry points the loop calls. Together they cost ~10 ms and
zero LLM tokens (all temporal computation is deterministic).

Loop integration:
  - Pre-task   (before Step 1):  refresh_temporal_state(kernel, ...)
  - Post-step6 (after Monitor):  bind_just_computed_signals(...)

The hooks are safe to call before any subpackage is fully wired —
they no-op gracefully if dependencies are missing.
"""
from __future__ import annotations

import logging
from typing import Optional

from app.subia.kernel import SubjectivityKernel

logger = logging.getLogger(__name__)


def refresh_temporal_state(
    kernel: SubjectivityKernel,
    *,
    previous_focal_ids: Optional[set] = None,
    previous_homeostasis: Optional[dict] = None,
    protention_forecast: Optional[dict] = None,
    density_sample=None,
    rhythms: Optional[list] = None,
    clock_provider=None,
) -> dict:
    """Pre-task hook. Refresh specious_present + temporal_context +
    homeostatic momentum in one call.

    Returns a small report dict for logging.
    """
    out = {"specious_present_updated": False,
           "temporal_context_updated": False,
           "momentum_updated": False}

    # 1. Specious present (Husserl/James felt-now)
    try:
        from app.subia.temporal.specious_present import update_specious_present
        update_specious_present(
            kernel,
            previous_focal_ids=previous_focal_ids,
            previous_homeostasis=previous_homeostasis,
            protention_forecast=protention_forecast,
        )
        out["specious_present_updated"] = True
    except Exception as exc:
        logger.debug("phase14: specious_present refresh failed: %s", exc)

    # 2. Homeostatic momentum (rising/falling/stable)
    try:
        from app.subia.temporal.momentum import update_momentum
        update_momentum(kernel.homeostasis, previous_values=previous_homeostasis)
        out["momentum_updated"] = True
    except Exception as exc:
        logger.debug("phase14: momentum refresh failed: %s", exc)

    # 3. Temporal context (clock + circadian + density + rhythms)
    try:
        from app.subia.temporal.context import refresh_temporal_context
        refresh_temporal_context(
            kernel,
            density_sample=density_sample,
            rhythms=rhythms,
            clock_provider=clock_provider,
        )
        out["temporal_context_updated"] = True
    except Exception as exc:
        logger.debug("phase14: temporal_context refresh failed: %s", exc)

    # 4. Close the loop: circadian mode drives homeostatic set-points
    #    and discovered rhythms populate self_state.capabilities. These
    #    bridges are the Phase 14 closed-loop payoff — signals computed
    #    above have to consume some behaviour, or they're dead data.
    try:
        from app.subia.connections.temporal_subia_bridge import (
            circadian_to_setpoints,
            rhythms_to_self_state,
        )
        diff = circadian_to_setpoints(kernel)
        out["circadian_setpoint_diff"] = diff
        if rhythms:
            out["rhythms_ingested"] = rhythms_to_self_state(kernel, rhythms)
    except Exception as exc:
        logger.debug("phase14: temporal bridge close failed: %s", exc)

    return out


def bind_just_computed_signals(
    *,
    feel: Optional[dict] = None,
    attend: Optional[dict] = None,
    own: Optional[dict] = None,
    predict: Optional[dict] = None,
    monitor: Optional[dict] = None,
    kernel: Optional[SubjectivityKernel] = None,
):
    """Post-Step-6 hook. Reduce the just-computed signals into a
    BoundMoment using the SpeciousPresent's retention as the
    stability bias.
    """
    try:
        from app.subia.temporal.binding import temporal_bind
        retention = []
        if kernel is not None and getattr(kernel, "specious_present", None):
            retention = kernel.specious_present.retention
        return temporal_bind(
            feel=feel, attend=attend, own=own,
            predict=predict, monitor=monitor,
            retention=retention,
        )
    except Exception as exc:
        logger.debug("phase14: temporal_bind failed: %s", exc)
        return None


def quick_bind_compressed_signals(
    *,
    feel: Optional[dict] = None,
    attend: Optional[dict] = None,
):
    """Post-Step-3 hook for the *compressed* CIL path.

    Consciousness-roadmap §3.G4 (compressed-loop binding cadence). The
    compressed loop early-returns after Step 3 ATTEND, so the full
    `bind_just_computed_signals` would receive only feel/attend (the
    other inputs haven't run). This wrapper calls the cheap
    `temporal_quick_bind` reducer that's explicit about what it can and
    cannot derive from those two layers alone.

    Returned BoundMoment has `dominant_affect` and `salient_focus` populated;
    `confidence_unified` stays at the dataclass default 0.5 because PREDICT
    + MONITOR haven't run yet.
    """
    try:
        from app.subia.temporal.binding import temporal_quick_bind
        return temporal_quick_bind(feel=feel, attend=attend)
    except Exception as exc:
        logger.debug("phase14: temporal_quick_bind failed: %s", exc)
        return None
