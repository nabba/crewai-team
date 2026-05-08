"""subia.temporal — Phase 14: Temporal Synchronization.

Closes three temporal gaps the proposal identifies:
  1. NO SPECIOUS PRESENT — the felt "now" with retention/primal/protention
     (Husserl/James). Implemented as a sliding window on the kernel.
  2. NO SIMULTANEOUS BINDING — currently sequential CIL steps. Phase 14
     adds a `temporal_bind()` integration over the just-computed signals
     (the binding is the unity, not parallelism for its own sake).
  3. NO EXTERNAL TEMPORAL RHYTHM — clock-time exists via
     `app.temporal_context` but no circadian processing modes, no
     discovered Andrus/Firecrawl rhythms, no felt processing density.

Architecture honours the existing SubIA convention:
  * Pure-data dataclasses + pure-function reducers (no side effects).
  * Adapters for any external dependency (logs, predict_fn).
  * Closed-loop discipline (Phase 2 invariant): every computed signal
    has a behavioural consequence — momentum modulates context blocks,
    circadian mode shifts setpoints, density drops the wonder threshold,
    consolidation mode preferences IdleScheduler jobs.
  * Tier-3 protection (mode tables, weights, thresholds).
  * Built ON `app.temporal_context` for clock/season/timezone — does not
    duplicate.
"""
from .specious_present import (
    KernelMoment, SpeciousPresent, update_specious_present,
)
from .momentum import (
    update_momentum, render_momentum_arrows, MomentumEntry,
)
from .circadian import (
    CIRCADIAN_MODES, current_circadian_mode, apply_circadian_setpoints,
    circadian_allows_reverie, circadian_cascade_preference,
)
from .density import (
    compute_processing_density, DensitySample,
)
from .binding import (
    temporal_bind, temporal_quick_bind, BoundMoment,
)
from .rhythm_discovery import (
    discover_rhythms, Rhythm,
)
from .context import TemporalContext, refresh_temporal_context

__all__ = [
    "KernelMoment", "SpeciousPresent", "update_specious_present",
    "update_momentum", "render_momentum_arrows", "MomentumEntry",
    "CIRCADIAN_MODES", "current_circadian_mode", "apply_circadian_setpoints",
    "circadian_allows_reverie", "circadian_cascade_preference",
    "compute_processing_density", "DensitySample",
    "temporal_bind", "temporal_quick_bind", "BoundMoment",
    "discover_rhythms", "Rhythm",
    "TemporalContext", "refresh_temporal_context",
]
