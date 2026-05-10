"""
SubIA — Subjective Integration Architecture.

Infrastructure layer that binds discrete agent operations into a
continuous, subject-centered, affectively-modulated, predictively-
structured experience loop.

This package is the migration target for the unified consciousness
program. Existing consciousness/* and self_awareness/* modules will
be migrated here in phased subpackages:

    scene/          GWT-2 workspace + AST-1 attention schema
    self/           Persistent subject token, self-model, temporal identity
    homeostasis/    Affective regulation with PDS-derived set-points
    belief/         HOT-3 belief store, metacognition, certainty
    prediction/     PP-1 predictive layer, hierarchy, predictor, cascade
    social/         Self/other distinction, Theory-of-Mind
    memory/         Dual-tier consolidation (curated + full)
    safety/         DGM extensions: setpoint + audit immutability
    probes/         Evaluation: Butlin, RSM, SK scorecards
    wiki_surface/   Wiki integration + strange loop

No behavior is wired yet. See PROGRAM.md for the phased migration plan.
"""

from app.subia.kernel import get_active_kernel, set_active_kernel

# Eager-import the inquiry scheduler so its daemon thread comes up on
# boot wherever the kernel loads. The scheduler is env-gated
# (INQUIRY_PASS_ENABLED, default true) and silently no-ops in test
# environments that disable it.
from app.subia.inquiry import scheduler as _inquiry_scheduler  # noqa: F401

__all__ = [
    "config",
    "kernel",
    "get_active_kernel",
    "set_active_kernel",
]
