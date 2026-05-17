"""
app.substrate — thin coordination facade for AndrusAI's living subsystems.

This package is intentionally tiny. Two modules:

  status.py   gather_substrate_status() — aggregate snapshot for the
              dashboard, the CLI, and the doctor command. Pure read,
              never raises, errors collected per-probe.
  policy.py   should_defer_heavy_work() — single decision used by the
              idle scheduler to honor host resource posture without
              silently dropping work.

The facade does NOT introduce a new event bus, supervisor, or policy
matrix. Coordination happens through existing primitives (continuity
ledger, healing monitors, runtime settings, proposal bridge, governance
amendment). The substrate package is just the place to read them all
in one breath.

See: docs in the productization plan, Work Package K (revised T2.1).
"""

from app.substrate.status import SubstrateStatus, gather_substrate_status
from app.substrate.policy import (
    ResourcePolicy,
    should_defer_heavy_work,
)

__all__ = [
    "SubstrateStatus",
    "gather_substrate_status",
    "ResourcePolicy",
    "should_defer_heavy_work",
    # Productization WP D Phase 2 + 3 (cloud migration).
    "MigrationRun",
    "MigrationStep",
    "GateResult",
    "GateFailure",
    "run_migration_dry_run",
    "run_migration_live",
    "evaluate_live_gates",
    "format_run",
]


_LAZY_MIGRATION_NAMES = frozenset({
    "MigrationRun", "MigrationStep", "GateResult", "GateFailure",
    "run_migration_dry_run", "run_migration_live",
    "evaluate_live_gates", "format_run",
})

# Productization WP D Phase 4 (cutover). Same lazy-load pattern.
_LAZY_CUTOVER_NAMES = frozenset({
    "CutoverRun", "CutoverStep", "CutoverGateResult", "CutoverGateFailure",
    "run_cutover", "evaluate_cutover_gates", "format_cutover_run",
})


def __getattr__(name: str):
    """Lazy imports for the cloud-related submodules to avoid pulling
    subprocess + tarfile machinery into the substrate facade unless
    actually used.
    """
    if name in _LAZY_MIGRATION_NAMES:
        from app.substrate import migration as _m
        return getattr(_m, name)
    if name in _LAZY_CUTOVER_NAMES:
        from app.substrate import cutover as _c
        return getattr(_c, name)
    raise AttributeError(f"module 'app.substrate' has no attribute {name!r}")
