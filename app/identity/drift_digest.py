"""Rolling identity-drift digest — PROGRAM §49 (Q14.1).

Closes the year-2+ resilience gap §10.1: the user can identify
"the system has accumulated 47 small Tier-3 amendments this quarter
without me noticing" only when the annual reflection runs (once a
year). This module gives the operator a **monthly** rolling view.

Algorithm — one HEAVY pass:

  1. Read continuity-ledger drift summaries at 30d / 90d / 365d
     windows via :func:`app.identity.continuity_ledger.summarise_drift`.
  2. Compute the *acceleration ratio* = ``count_30d /
     (count_365d / 12)``. A value of 1.0 means "drift this month is
     consistent with the annual average". A value of 2.0+ means
     "this month drift is at twice the annual rate" — operator
     attention warranted.
  3. Per-kind breakdown: any kind whose 30d count is ≥ ``3 ×
     (365d_count_for_kind / 12)`` is highlighted (a specific kind is
     accelerating, not just aggregate drift).
  4. Emit ``identity_drift_acceleration`` continuity-ledger landmark
     ONLY on threshold crossing — routine "all normal" passes stay
     silent so the ledger doesn't fill with noise.
  5. Surface in the daily briefing's "📊 Identity drift (this month)"
     section (composer pulls a one-line summary).

The digest is observational. It never blocks amendments, never
proposes rollbacks, never edits anything. Its job is to make the
aggregate visible mid-year so the operator can pause if drift feels
out of step with their intent.

Master switch: ``identity_drift_digest_enabled`` (default ON).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


_STATE_FILE_NAME = "identity_drift_digest_state.json"
_DIGEST_FILE_NAME = "identity_drift_digest_latest.json"

# Thresholds (matching the user's spec; configurable via env).
_AGGREGATE_ACCELERATION_THRESHOLD = 2.0  # 2× annual rate
_PER_KIND_ACCELERATION_THRESHOLD = 3.0   # 3× annual rate for any kind
_RUN_CADENCE_S = 30 * 24 * 3600          # monthly


# ── Types ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DriftDigest:
    """Snapshot of identity drift across multiple time horizons."""

    generated_at: str
    counts_30d: int
    counts_90d: int
    counts_365d: int
    aggregate_acceleration: float
    by_kind_30d: dict[str, int] = field(default_factory=dict)
    by_kind_365d: dict[str, int] = field(default_factory=dict)
    accelerated_kinds: list[str] = field(default_factory=list)
    landmark_fired: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Master-switch + state ───────────────────────────────────────────────


def _enabled() -> bool:
    try:
        from app.runtime_settings import get_identity_drift_digest_enabled
        return get_identity_drift_digest_enabled()
    except Exception:
        return os.getenv("IDENTITY_DRIFT_DIGEST_ENABLED", "true").lower() in (
            "true", "1", "yes", "on",
        )


def _workspace() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT)
    except Exception:
        return Path("/app/workspace")


def _state_path() -> Path:
    return _workspace() / "identity" / _STATE_FILE_NAME


def _digest_path() -> Path:
    """Operator-visible latest digest — daily briefing + REST surface."""
    return _workspace() / "identity" / _DIGEST_FILE_NAME


def _read_state() -> dict[str, Any]:
    p = _state_path()
    if not p.exists():
        return {"last_run_at": 0.0, "last_landmark_at": None}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"last_run_at": 0.0, "last_landmark_at": None}


def _write_state(state: dict[str, Any]) -> None:
    p = _state_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(state, indent=2, sort_keys=True), encoding="utf-8",
        )
    except Exception:
        logger.debug("drift_digest: state write failed", exc_info=True)


def _write_digest(digest: DriftDigest) -> None:
    p = _digest_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(digest.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except Exception:
        logger.debug("drift_digest: digest write failed", exc_info=True)


# ── Compute ─────────────────────────────────────────────────────────────


def compute_digest(*, now: Optional[datetime] = None) -> DriftDigest:
    """Pure-function digest computation. Returns a :class:`DriftDigest`
    snapshot. Used directly by the daily-briefing composer + the
    monthly monitor."""
    cur = now or datetime.now(timezone.utc)
    try:
        from app.identity.continuity_ledger import summarise_drift
    except Exception:
        logger.debug("drift_digest: continuity_ledger unavailable")
        return DriftDigest(
            generated_at=cur.isoformat(),
            counts_30d=0, counts_90d=0, counts_365d=0,
            aggregate_acceleration=0.0,
        )
    d30 = summarise_drift(window_days=30, now=cur)
    d90 = summarise_drift(window_days=90, now=cur)
    d365 = summarise_drift(window_days=365, now=cur)
    # Acceleration ratio: 30d count vs (annual / 12) expected baseline.
    expected_monthly = max(d365.n_events / 12.0, 0.0)
    if expected_monthly < 0.5:
        # Insufficient history. Don't divide by ~0; report 0 so the
        # threshold check below is skipped.
        acceleration = 0.0
    else:
        acceleration = round(d30.n_events / expected_monthly, 2)
    # Per-kind acceleration: any kind ≥ 3× its annual monthly average.
    accelerated_kinds: list[str] = []
    for kind, count_30d in d30.by_kind.items():
        annual = d365.by_kind.get(kind, 0)
        if annual < 4:
            # Need at least 4/year to compute a meaningful baseline.
            continue
        if count_30d >= _PER_KIND_ACCELERATION_THRESHOLD * (annual / 12.0):
            accelerated_kinds.append(kind)
    return DriftDigest(
        generated_at=cur.isoformat(),
        counts_30d=d30.n_events,
        counts_90d=d90.n_events,
        counts_365d=d365.n_events,
        aggregate_acceleration=acceleration,
        by_kind_30d=dict(d30.by_kind),
        by_kind_365d=dict(d365.by_kind),
        accelerated_kinds=accelerated_kinds,
    )


def briefing_section(*, now: Optional[datetime] = None) -> Optional[str]:
    """Return the daily-briefing snippet (one-line summary), or None
    if drift is normal / digest unavailable.

    Composer calls this with try/except so a broken digest cannot
    break the briefing build."""
    digest = compute_digest(now=now)
    if digest.counts_30d == 0 and digest.counts_365d == 0:
        return None
    if digest.aggregate_acceleration < _AGGREGATE_ACCELERATION_THRESHOLD:
        return None  # routine; don't clutter the briefing
    parts = [
        f"📊 Identity drift this month: "
        f"{digest.counts_30d} amendments "
        f"({digest.aggregate_acceleration:.1f}× annual rate)."
    ]
    if digest.accelerated_kinds:
        parts.append(
            f" Accelerated kinds: {', '.join(digest.accelerated_kinds[:3])}."
        )
    return "".join(parts)


# ── Public entry ────────────────────────────────────────────────────────


def run(*, now: Optional[datetime] = None) -> dict[str, Any]:
    """One monthly pass. Computes digest, alerts on threshold crossing,
    emits ledger landmark, persists state.

    Failure-isolated."""
    summary: dict[str, Any] = {
        "ran": False,
        "skipped": False,
        "aggregate_acceleration": 0.0,
        "accelerated_kinds": [],
        "alert_fired": False,
        "landmark_emitted": False,
    }
    if not _enabled():
        summary["skipped"] = True
        return summary

    cur = now or datetime.now(timezone.utc)
    cur_ts = cur.timestamp()
    state = _read_state()
    last_run = float(state.get("last_run_at", 0))
    if last_run > 0 and cur_ts - last_run < _RUN_CADENCE_S:
        summary["skipped"] = True
        return summary
    state["last_run_at"] = cur_ts
    summary["ran"] = True

    digest = compute_digest(now=cur)
    _write_digest(digest)
    summary["aggregate_acceleration"] = digest.aggregate_acceleration
    summary["accelerated_kinds"] = list(digest.accelerated_kinds)

    threshold_crossed = (
        digest.aggregate_acceleration >= _AGGREGATE_ACCELERATION_THRESHOLD
        or len(digest.accelerated_kinds) > 0
    )
    if threshold_crossed:
        try:
            from app.notify import notify
            body_lines = [
                f"Identity drift accelerated this month:",
                f"  • 30d count: {digest.counts_30d}",
                f"  • 90d count: {digest.counts_90d}",
                f"  • 365d count: {digest.counts_365d}",
                f"  • aggregate acceleration: "
                f"{digest.aggregate_acceleration:.2f}× annual rate",
            ]
            if digest.accelerated_kinds:
                body_lines.append(
                    f"  • accelerated kinds: "
                    f"{', '.join(digest.accelerated_kinds)}"
                )
            body_lines.append(
                "\nReview pace of recent Tier-3 amendments + soul "
                "edits at /cp/changes + /cp/amendments."
            )
            notify(
                title="📊 Identity drift acceleration",
                body="\n".join(body_lines),
                url="/cp/amendments",
                topic="identity_drift_acceleration",
                critical=False,
                arbitrate=True,
            )
            summary["alert_fired"] = True
        except Exception:
            logger.debug("drift_digest: notify failed", exc_info=True)

        # Continuity-ledger landmark emission.
        try:
            from app.identity.continuity_ledger import record_event
            record_event(
                kind="identity_drift_acceleration",
                actor="drift_digest_monitor",
                summary=(
                    f"Drift acceleration {digest.aggregate_acceleration:.2f}× "
                    f"annual rate over 30 days "
                    f"({digest.counts_30d} amendments)."
                ),
                detail={
                    "counts_30d": digest.counts_30d,
                    "counts_90d": digest.counts_90d,
                    "counts_365d": digest.counts_365d,
                    "aggregate_acceleration": digest.aggregate_acceleration,
                    "accelerated_kinds": digest.accelerated_kinds,
                },
            )
            summary["landmark_emitted"] = True
        except Exception:
            logger.debug("drift_digest: ledger emit failed", exc_info=True)
        state["last_landmark_at"] = cur.isoformat()

    _write_state(state)
    return summary
