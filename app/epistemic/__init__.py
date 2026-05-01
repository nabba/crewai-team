"""
Epistemic Integrity Layer — provenance, calibration, and post-mortem
analysis of agent reasoning.

Real-time gate that distinguishes verified from inferred claims, plus
post-hoc analysis that feeds the Self-Improver's existing 6-stage
pipeline. Full design: see ``crewai-team/docs/EPISTEMIC_INTEGRITY.md``.

Phase 0 (this commit) ships only the foundational data model:
  * the Claim Ledger and its three emission paths (path 1 wired,
    paths 2 and 3 reserved for Phase 1)
  * PostgreSQL persistence into ``control_plane.epistemic_claims``
  * a hook registry so detector subsystems can self-register
    without the Ledger importing them

Off by default — toggle with EPISTEMIC_ENABLED=true.
"""
from __future__ import annotations

import os

# ── Public API ─────────────────────────────────────────────────────────
# Re-export the small surface that callers should depend on. Anything
# not exported here is internal.
from app.epistemic.ledger import (
    LEDGER_MAX_CLAIMS_PER_TASK,
    Claim,
    Evidence,
    Ledger,
    Register,
    VerificationStatus,
    VerifyingAction,
)
from app.epistemic.registry import (
    ClaimHook,
    claim_hooks,
    register as register_claim_hook,
)

__all__ = [
    "Claim",
    "ClaimHook",
    "Evidence",
    "LEDGER_MAX_CLAIMS_PER_TASK",
    "Ledger",
    "Register",
    "VerificationStatus",
    "VerifyingAction",
    "claim_hooks",
    "is_enabled",
    "register_claim_hook",
]


def is_enabled() -> bool:
    """Off by default. Flip EPISTEMIC_ENABLED=true to activate.

    Mirrors the pattern from ``app.recovery.loop.is_enabled``: a pure
    environment-variable gate, not a Settings field. The reason is that
    additive subsystems with a kill switch want a single deployment-level
    knob, not a Settings round-trip — and they want the gate readable
    from contexts (tests, scripts) that don't construct full Settings.
    """
    val = os.getenv("EPISTEMIC_ENABLED", "").strip().lower()
    return val in ("1", "true", "yes", "on")


# ── Bootstrap: register detectors via import side-effect ────────────
# Importing this module attaches the realtime meta-hook to the claim
# ledger and registers the post-hoc detectors. Idempotent — re-import
# is a no-op (the registries dedup).
from app.epistemic.detectors import realtime as _realtime  # noqa: E402,F401
from app.epistemic.detectors import posthoc as _posthoc  # noqa: E402,F401
