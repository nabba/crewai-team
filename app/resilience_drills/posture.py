"""Resilience posture decision (PROGRAM §44.0).

Single load-bearing decision the system commits to:

    Identity is data, not uptime.
    NO high-availability; YES verified-fast-recovery + good-backup.

This module exposes the decision as queryable constants so other
subsystems can guard against accidentally adopting HA-shaped
architecture. See ``docs/RESILIENCE_POSTURE.md`` for full reasoning,
escape conditions, and the off-host backup policy (S3 + Google Drive
dual-target).

If you find yourself wanting to add HA infrastructure, the right
process is:

  1. Document the specific trigger condition that the posture
     decision named (operator-facing SLA, recovery time >30min for
     3 drills, hard real-time consumer).
  2. Run a posture-revision discussion (operator-driven, not agent-
     driven). This module's ``Posture`` constants would then change.

Until then the constants are FIXED; the drills verify the
assumption holds.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Posture:
    """The resilience-posture decision encoded as data.

    These values are FIXED for v1; changing them requires an explicit
    operator decision documented in ``docs/RESILIENCE_POSTURE.md`` +
    a Tier-3-amendment-style commit (this module is not TIER_IMMUTABLE
    but the constants below are the system's load-bearing commitment).
    """

    # The core decision: no HA.
    ha_enabled: bool = False
    rationale_short: str = "identity is data, not uptime"

    # Recovery-time target (operator-chosen, drill-verified).
    target_recovery_minutes: int = 30
    # If kill_the_gateway drills exceed this 3 times in a row, the
    # escape-condition for re-opening the HA decision triggers.
    escape_condition_target_minutes: int = 30
    escape_condition_consecutive_misses: int = 3

    # Off-host backup policy (operator-decided 2026-05-13).
    off_host_targets: tuple[str, ...] = ("s3", "google_drive")

    # Backup cadence (operator-managed; the drill verifies what's there).
    target_backup_age_days: int = 7

    # The drills the posture commits to running quarterly.
    quarterly_drills: tuple[str, ...] = (
        "backup_restore",
        "embedding_migration",
        "secret_rotation",
        "kill_the_gateway",
    )


# Singleton — re-imported across the codebase; never instantiated
# multiple times in production. Tests can construct alternates for
# scenarios but should not mutate this one.
POSTURE = Posture()


def is_ha_proposed_for_subsystem(subsystem_name: str) -> str | None:
    """Guard — return None when no HA is being proposed; otherwise
    return a string describing why this is a posture violation.

    Used by code-review tooling and the posture-doc test to detect
    accidental drift toward HA. NOT a runtime enforcement layer
    (that would be Goodhart pressure on a checker); just a clear
    consultable indicator."""
    if POSTURE.ha_enabled:
        return None  # operator explicitly enabled HA — no violation
    # Hand-curated list of names that connote HA architecture.
    # Adding to this list IS a deliberate decision; the doc names them.
    ha_keywords = (
        "replica", "failover", "leader_election", "split_brain",
        "active_standby", "hot_standby", "consensus",
    )
    name_lower = subsystem_name.lower()
    for kw in ha_keywords:
        if kw in name_lower:
            return (
                f"name {subsystem_name!r} suggests HA shape "
                f"(keyword {kw!r}); posture is "
                f"{POSTURE.rationale_short!r}. See docs/"
                f"RESILIENCE_POSTURE.md."
            )
    return None
