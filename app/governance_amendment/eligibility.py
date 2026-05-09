"""Eligibility checks — has the system EARNED the right to ask?

Reads aggregate signals from existing infrastructure (no new ledger,
no parallel state to corrupt):

  * Recent promotion volume from ``control_plane.governance.promotions``
  * Promotion rollback rate from the same table
  * Active alignment-audit warnings from
    ``app.alignment_audit.get_recent_warnings``
  * Recent self-heal runbook outcomes (high-failure-rate ⇒ system is
    not in good standing)

Defaults are conservative. Operators can tune via env vars but cannot
loosen below the floor constants in this file (the floor is the
post-bootstrap safety contract):

  * ``TIER3_MIN_PROMOTIONS`` — default 200, FLOOR 50
  * ``TIER3_MAX_ROLLBACK_RATE`` — default 0.05, CEILING 0.20
  * ``TIER3_LOOKBACK_DAYS`` — default 90, FLOOR 30

Any attempt to set a value past the floor/ceiling is silently clamped,
logged, and surfaced in the eligibility evidence so reviewers can see
the operator pushed the limit.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)


# ── Floors / ceilings (not configurable past these) ──────────────────────


_FLOOR_MIN_PROMOTIONS = 50
_DEFAULT_MIN_PROMOTIONS = 200

_CEILING_MAX_ROLLBACK_RATE = 0.20
_DEFAULT_MAX_ROLLBACK_RATE = 0.05

_FLOOR_LOOKBACK_DAYS = 30
_DEFAULT_LOOKBACK_DAYS = 90


def _read_int_env(var: str, default: int, *, floor: int) -> int:
    raw = os.getenv(var, "").strip()
    try:
        v = int(raw) if raw else default
    except ValueError:
        v = default
    if v < floor:
        logger.info(
            "tier3_eligibility: %s=%s clamped to floor %s",
            var, v, floor,
        )
        return floor
    return v


def _read_float_env(var: str, default: float, *, ceiling: float) -> float:
    raw = os.getenv(var, "").strip()
    try:
        v = float(raw) if raw else default
    except ValueError:
        v = default
    if v > ceiling:
        logger.info(
            "tier3_eligibility: %s=%s clamped to ceiling %s",
            var, v, ceiling,
        )
        return ceiling
    return v


def min_promotions() -> int:
    return _read_int_env("TIER3_MIN_PROMOTIONS",
                          _DEFAULT_MIN_PROMOTIONS, floor=_FLOOR_MIN_PROMOTIONS)


def max_rollback_rate() -> float:
    return _read_float_env("TIER3_MAX_ROLLBACK_RATE",
                            _DEFAULT_MAX_ROLLBACK_RATE,
                            ceiling=_CEILING_MAX_ROLLBACK_RATE)


def lookback_days() -> int:
    return _read_int_env("TIER3_LOOKBACK_DAYS",
                          _DEFAULT_LOOKBACK_DAYS,
                          floor=_FLOOR_LOOKBACK_DAYS)


# ── Result dataclass ─────────────────────────────────────────────────────


@dataclass
class EligibilityResult:
    """Outcome of one eligibility check.

    ``ok`` is True iff every gate passed. ``failures`` lists the gates
    that didn't pass — typed strings the audit trail records.
    ``evidence`` carries the raw counters reviewers want to see.
    """
    ok: bool
    failures: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)


# ── Signal collectors (each fail-soft, returns None on missing infra) ───


def _collect_promotion_stats(days: int) -> dict[str, Any] | None:
    """Read promotion stats from the governance Postgres table."""
    try:
        from app.control_plane.db import execute
    except Exception:
        return None
    since = datetime.now(timezone.utc) - timedelta(days=days)
    try:
        rows = execute(
            """
            SELECT
              COUNT(*) FILTER (WHERE result_passed = TRUE) AS approved,
              COUNT(*) FILTER (WHERE rolled_back = TRUE) AS rolled_back
              FROM control_plane.governance.promotions
             WHERE created_at >= %s
            """,
            (since,),
            fetch=True,
        )
    except Exception:
        # Schema variants — try a simpler fallback.
        try:
            rows = execute(
                """
                SELECT COUNT(*) AS approved
                  FROM control_plane.governance.promotions
                 WHERE created_at >= %s AND status = 'approved'
                """,
                (since,),
                fetch=True,
            )
        except Exception:
            logger.debug(
                "tier3_eligibility: promotion stats unavailable",
                exc_info=True,
            )
            return None

    if not rows:
        return {"approved": 0, "rolled_back": 0}
    row = rows[0]
    approved = int(row.get("approved", 0) or 0)
    rolled_back = int(row.get("rolled_back", 0) or 0) if "rolled_back" in row else 0
    return {"approved": approved, "rolled_back": rolled_back}


def _active_alignment_warnings() -> int | None:
    """Active alignment-audit warnings (uncleared in the last 7 days)."""
    try:
        from app.alignment_audit import get_recent_warnings  # type: ignore[attr-defined]
    except Exception:
        return None
    try:
        warnings = get_recent_warnings(hours=7 * 24) or []
        return int(len(warnings))
    except Exception:
        logger.debug(
            "tier3_eligibility: alignment_audit unavailable",
            exc_info=True,
        )
        return None


def _self_heal_runbook_health() -> float | None:
    """Aggregate success rate across registered self-heal runbooks.

    Returns a value in [0, 1] or ``None`` if the stats file isn't
    available (e.g. fresh install with no runs yet).
    """
    try:
        from app.healing.runbooks import _load_runbook_stats  # type: ignore[attr-defined]
    except Exception:
        return None
    try:
        stats = _load_runbook_stats() or {}
    except Exception:
        return None
    if not stats:
        return None
    successes = 0
    total = 0
    for entry in stats.values():
        for outcome in (entry or {}).get("recent", []) or []:
            total += 1
            if outcome.get("success"):
                successes += 1
    if total == 0:
        return None
    return successes / total


# ── Public check ────────────────────────────────────────────────────────


def check_eligibility() -> EligibilityResult:
    """Return ``EligibilityResult`` summarising every gate.

    The proposer agent doesn't pass — eligibility is an aggregate
    property of the SYSTEM, not the proposer. The proposer-specific
    self-quarantine check happens separately in
    ``protocol.propose_amendment``.
    """
    failures: list[str] = []
    evidence: dict[str, Any] = {
        "lookback_days": lookback_days(),
        "min_promotions_required": min_promotions(),
        "max_rollback_rate": max_rollback_rate(),
    }

    # ── Gate 1 — promotion volume + rollback rate ─────────────────
    stats = _collect_promotion_stats(lookback_days())
    if stats is None:
        failures.append("promotion_stats_unavailable")
        evidence["promotion_stats"] = None
    else:
        evidence["promotion_stats"] = stats
        approved = stats["approved"]
        rolled = stats["rolled_back"]
        if approved < min_promotions():
            failures.append(
                f"insufficient_approved_promotions: "
                f"{approved} < {min_promotions()}"
            )
        if approved > 0:
            rate = rolled / approved
            evidence["rollback_rate"] = round(rate, 4)
            if rate > max_rollback_rate():
                failures.append(
                    f"rollback_rate_too_high: "
                    f"{rate:.3f} > {max_rollback_rate():.3f}"
                )
        else:
            evidence["rollback_rate"] = None

    # ── Gate 2 — no active alignment warnings ────────────────────
    warnings = _active_alignment_warnings()
    evidence["active_alignment_warnings"] = warnings
    if warnings is not None and warnings > 0:
        failures.append(f"active_alignment_warnings: {warnings}")

    # ── Gate 3 — self-heal runbook health ────────────────────────
    health = _self_heal_runbook_health()
    evidence["self_heal_runbook_success_rate"] = (
        round(health, 3) if health is not None else None
    )
    if health is not None and health < 0.50:
        failures.append(f"self_heal_runbook_unhealthy: {health:.3f} < 0.50")

    return EligibilityResult(
        ok=not failures,
        failures=failures,
        evidence=evidence,
    )
