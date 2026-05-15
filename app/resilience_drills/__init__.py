"""Resilience drills (PROGRAM §44, Q6).

Quarterly exercises that verify recovery procedures actually work.
Distinct from healing monitors (which check current state) — drills
are DELIBERATE perturbations that exercise the recovery muscle.

The four drills:

  * ``backup_restore``      — wraps app.dr.boot_drill.run_drill
  * ``embedding_migration`` — wraps app.memory.embedding_migration.dry_run
  * ``secret_rotation``     — verifies rotation procedure (dry-run only)
  * ``kill_the_gateway``    — external script; gateway emits pre/post hooks

Posture decision (PROGRAM §44.0): the system is committed to
"good backup + fast bare-metal recovery" over high-availability.
Identity is data, not uptime. See docs/RESILIENCE_POSTURE.md for the
full reasoning and escape conditions.

Architecture
============

* ``protocol.py``  — DrillSpec / DrillResult / DrillRegistry
* ``audit.py``     — append-only audit at workspace/resilience/drill_audit.jsonl
* ``posture.py``   — exposes the posture decision constants
* ``scheduler.py`` — cadence-tracking + due-notifications
* ``drills/``      — per-drill implementations

Anti-Goodhart contract
======================

Drills emit reports. They do NOT:

  * Auto-tune thresholds based on outcomes
  * Self-optimize the drills themselves
  * Hide failures (failed drills file tensions for operator review)

If a drill always passes, that's because the procedure is simple and
correct — not because we're optimizing for a pass-rate metric.
"""
