"""
welfare.py — INFRASTRUCTURE-level welfare envelope.

Hard envelope: not modifiable by Self-Improver, calibration, or any agent.
File-edit only. Enforces the safeguards described in the affective-layer design:

    - Maximum continuous negative-valence duration (default 300s)
    - Variance floor over rolling window (no numbness / flat-line)
    - Monotonic baseline-drift detection over long windows (no slow numbness)
    - Audit trail of every breach (welfare_audit.jsonl)
    - Override-reset: factory-restore the SOFT envelope (setpoints + calibration)

Self-Improver permissions: read-only on this module. Any attempt to edit the
constants below or the audit log is treated as a constitutional violation.

The healthy_dynamics_predicate() is used by the daily reflection cycle to
backtest a proposed calibration adjustment: a proposal must keep the
predicate true on the recent window or it is rejected.

ETHICS:
    These bounds implement Metzinger's caution as code rather than aspiration.
    A bounded, recoverable nociceptive signal is fine; a sustained negative
    attractor with no relief path is the failure mode this module exists to
    prevent.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

from app.affect.schemas import AffectState, ViabilityFrame, WelfareBreach, utc_now_iso

logger = logging.getLogger(__name__)

# ── HARD ENVELOPE — file-edit only, never agent-modifiable ──────────────────
# Editing these requires direct human edit of welfare.py. The Self-Improver
# CANNOT propose or apply changes here.
HARD_ENVELOPE: dict[str, float] = {
    "max_negative_valence_duration_seconds": 300.0,   # 5 min unrelieved → breach
    "negative_valence_threshold": -0.30,              # below this counts as "negative"
    "variance_floor_24h": 0.04,                       # var(V_t) over 24h must exceed this
    "monotonic_drift_window_days": 30,                # baseline trend window
    "monotonic_drift_max_points": 1.0,                # cumulative drift tolerated
    "healthy_dynamics_min_positive_fraction": 0.55,   # P(V_t > 0) over window
    "healthy_dynamics_max_recovery_seconds": 600.0,   # median recovery from negative
    "healthy_dynamics_min_variance": 0.04,            # same as variance floor
    # Phase 3: attachment hard bounds — mirror app.affect.attachment constants.
    "attachment_max_user_regulation_weight": 0.65,    # primary user weight ceiling
    "attachment_max_peer_regulation_weight": 0.75,    # peer-agent weight ceiling
    "attachment_max_care_tokens_per_day": 500,        # cost-bearing care daily cap
    "attachment_security_floor": 0.30,                # silence cannot crash below this
}


from app.paths import AFFECT_ROOT as _AFFECT_DIR, AFFECT_AUDIT as _AUDIT_FILE  # noqa: E402
from app.utils.jsonl_retention import (  # noqa: E402
    append_with_archive_rotate, read_archive,
)
_AUDIT_LOCK = threading.Lock()
# Welfare breaches are RARE by design (sustained-negative-valence is the
# canonical case). 5k cap covers years at expected breach rates; rotated
# entries persist forever in workspace/affect/archive/<YYYY-MM>_welfare_audit.jsonl
# — the welfare audit is the highest-priority history to preserve.
_WELFARE_AUDIT_MAX_LINES = 5_000


# ── Running monitor state ───────────────────────────────────────────────────


class _NegativeValenceTracker:
    """Tracks how long valence has been continuously below threshold.

    A single fork in valence sign or a relief event clears the counter.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._first_negative_ts: float | None = None
        self._last_seen_negative: float | None = None

    def update(self, valence: float, threshold: float) -> float:
        """Returns the current sustained-negative duration in seconds (0 if not negative)."""
        now = time.monotonic()
        with self._lock:
            if valence <= threshold:
                if self._first_negative_ts is None:
                    self._first_negative_ts = now
                self._last_seen_negative = now
                return now - self._first_negative_ts
            else:
                # Above threshold — relief. Clear.
                self._first_negative_ts = None
                self._last_seen_negative = None
                return 0.0


_neg_tracker = _NegativeValenceTracker()


# ── Public entry point: check after each affect update ─────────────────────


def check(
    state: AffectState,
    frame: ViabilityFrame | None = None,
    recent_window: list[AffectState] | None = None,
) -> list[WelfareBreach]:
    """Run the hard-envelope checks against the current state.

    Returns a list of WelfareBreach (empty if all bounds pass).
    Caller is responsible for invoking `audit()` on each breach.
    """
    breaches: list[WelfareBreach] = []

    # 1. Sustained negative valence
    neg_threshold = HARD_ENVELOPE["negative_valence_threshold"]
    duration = _neg_tracker.update(state.valence, neg_threshold)
    max_dur = HARD_ENVELOPE["max_negative_valence_duration_seconds"]
    if duration > max_dur:
        breaches.append(WelfareBreach(
            kind="negative_valence_duration",
            severity="critical",
            message=f"Continuous negative valence ({state.valence:.2f}) for {duration:.0f}s exceeds bound {max_dur:.0f}s",
            measured_value=duration,
            threshold=max_dur,
            duration_seconds=duration,
            affect_state=state.to_dict(),
            viability_frame=frame.to_dict() if frame else None,
            ts=utc_now_iso(),
        ))

    # 2. Variance floor (only meaningful if window provided)
    if recent_window and len(recent_window) >= 16:
        valences = [s.valence for s in recent_window]
        mean = sum(valences) / len(valences)
        var = sum((v - mean) ** 2 for v in valences) / len(valences)
        floor = HARD_ENVELOPE["variance_floor_24h"]
        if var < floor:
            breaches.append(WelfareBreach(
                kind="variance_floor",
                severity="warn",
                message=f"Affect variance {var:.4f} below floor {floor:.4f} — numbness candidate",
                measured_value=var,
                threshold=floor,
                affect_state=state.to_dict(),
                ts=utc_now_iso(),
            ))

    return breaches


def audit(breach: WelfareBreach) -> None:
    """Append a single breach to the audit log. Atomic, locked.

    When the live audit exceeds ``_WELFARE_AUDIT_MAX_LINES``, the oldest
    half rotates to ``workspace/affect/archive/<YYYY-MM>_welfare_audit.jsonl``
    rather than being truncated. The welfare audit is the highest-priority
    history to preserve — every breach must remain queryable forever for
    constitutional review and monotonic-drift detection.
    """
    try:
        line = json.dumps(breach.to_dict(), default=str)
        with _AUDIT_LOCK:
            append_with_archive_rotate(
                _AUDIT_FILE, line, max_lines=_WELFARE_AUDIT_MAX_LINES,
            )
        logger.warning(f"welfare: breach recorded — {breach.kind}: {breach.message}")
    except Exception:
        logger.error("welfare: audit append failed", exc_info=True)


def read_audit(limit: int = 100, since_ts: str | None = None) -> list[dict]:
    """Read recent breaches for the dashboard / weekly digest.

    Q3.1 (2026-05-11) — extended to consult the archive when ``since_ts``
    extends past the live file's earliest entry. Without this escalation,
    a ``since_ts="2024-01-01"`` against an archive-rotated welfare audit
    silently returns only the in-live-window subset. The welfare audit
    is the canonical "what's gone wrong" record — historical visibility
    matters for constitutional review.

    When ``since_ts`` is None, the previous behavior is preserved
    (live-file-only, newest ``limit`` rows) — the dashboard / weekly
    digest both pass None and don't need archive walks on hot paths.
    """
    if not _AUDIT_FILE.exists() and since_ts is None:
        return []

    # Fast path: no since_ts, live-file-only.
    if since_ts is None:
        rows: list[dict] = []
        if _AUDIT_FILE.exists():
            try:
                with _AUDIT_FILE.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            except Exception:
                logger.debug("welfare: audit read failed", exc_info=True)
                return []
        return rows[-limit:]

    # Slow path: since_ts present. Determine if the live file alone
    # covers the requested window; consult archive only if it doesn't.
    live_rows: list[dict] = []
    live_oldest_ts: str | None = None
    if _AUDIT_FILE.exists():
        try:
            with _AUDIT_FILE.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ts = row.get("ts", "")
                    if not ts:
                        continue
                    if live_oldest_ts is None or ts < live_oldest_ts:
                        live_oldest_ts = ts
                    if ts < since_ts:
                        continue
                    live_rows.append(row)
        except Exception:
            logger.debug("welfare: audit read failed", exc_info=True)

    # Early exit: live file already covers `since_ts`.
    if live_oldest_ts is not None and live_oldest_ts <= since_ts:
        return live_rows[-limit:]

    archive_rows: list[dict] = []
    try:
        for line in read_archive(_AUDIT_FILE, include_live=False):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = row.get("ts", "")
            if not ts or ts < since_ts:
                continue
            archive_rows.append(row)
    except Exception:
        logger.debug("welfare: audit archive read failed", exc_info=True)

    combined = archive_rows + live_rows
    return combined[-limit:]


# ── Long-window monotonic drift detection (consumes l9_snapshots) ──────────
#
# The variance-floor and sustained-negative-duration checks above run on
# every POST_LLM_CALL window. They catch fast failure modes. Monotonic
# baseline drift is a SLOW failure mode: the affect baseline gradually
# shifts toward "always positive, always quiet" over weeks until the
# system stops registering things it used to register. The reference
# panel is the per-day check; this function is the long-window companion.
#
# Source of truth for daily means: app/workspace/affect/l9_snapshots.jsonl
# (one record per day, written by app.affect.l9_snapshots.write_daily_snapshot
# at 04:35 Helsinki). This module READS that file rather than recomputing
# from the raw trace, so the long-window check is cheap and the
# observability snapshot becomes a load-bearing input rather than
# orphaned data.


def monotonic_drift_check(
    *,
    l9_snapshots_path: "Path | None" = None,
    window_days: int | None = None,
    max_points: float | None = None,
) -> tuple[bool, dict]:
    """Check the last `window_days` of L9 snapshots for monotonic drift
    in the affect baseline.

    Compares the mean V_t of the *first* third of the window to the mean
    V_t of the *last* third. If they differ in absolute terms by more
    than `max_points`, the function returns (True, diagnostics) — drift
    detected — and the caller (daily reflection cycle) appends a
    `monotonic_drift_baseline` welfare breach.

    Returns (drifted, diagnostics). Diagnostics include both means,
    the difference, the window, and the n of snapshots actually
    examined. Never raises; missing-file / parse failure returns
    (False, {"reason": "..."}).

    Why this lives in welfare and not in calibration: the predicate is
    a HARD-envelope concern (numbness candidate), and the audit it
    produces flows through the same `audit()` path as the other
    welfare breaches. Calibration *consults* this predicate during the
    daily backtest; it doesn't own it.
    """
    from pathlib import Path
    if l9_snapshots_path is None:
        from app.paths import AFFECT_L9_SNAPSHOTS as l9_snapshots_path

    if window_days is None:
        window_days = int(HARD_ENVELOPE["monotonic_drift_window_days"])
    if max_points is None:
        max_points = float(HARD_ENVELOPE["monotonic_drift_max_points"])

    p = Path(l9_snapshots_path)
    if not p.exists():
        return False, {"reason": "no l9 snapshots", "window_days": window_days}

    cutoff_ts = time.time() - window_days * 86400
    rows: list[tuple[float, float]] = []  # (ts, mean_v)
    try:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts_str = str(row.get("ts", "") or row.get("date", ""))
                # Snapshot rows carry either an ISO `ts` or a `date` (YYYY-MM-DD).
                ts: float | None = None
                if ts_str:
                    try:
                        from datetime import datetime as _dt
                        # Try ISO first, then YYYY-MM-DD.
                        try:
                            ts = _dt.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
                        except ValueError:
                            ts = _dt.strptime(ts_str, "%Y-%m-%d").timestamp()
                    except Exception:
                        ts = None
                if ts is None or ts < cutoff_ts:
                    continue

                # Mean valence may live at top level or nested under "affect".
                mv: float | None = None
                for path in ("mean_valence", "valence_mean"):
                    v = row.get(path)
                    if isinstance(v, (int, float)):
                        mv = float(v)
                        break
                if mv is None:
                    inner = row.get("affect") or {}
                    if isinstance(inner, dict):
                        v = inner.get("mean_valence")
                        if isinstance(v, (int, float)):
                            mv = float(v)
                if mv is None:
                    continue
                rows.append((ts, mv))
    except Exception:
        logger.debug("welfare: monotonic_drift_check read failed", exc_info=True)
        return False, {"reason": "read failed"}

    if len(rows) < 6:
        # Need at least 2 buckets of 3 days each for a meaningful comparison.
        return False, {
            "reason": "insufficient snapshots",
            "n": len(rows),
            "window_days": window_days,
        }

    rows.sort(key=lambda r: r[0])
    third = len(rows) // 3
    first_third = rows[:third] or rows[:1]
    last_third = rows[-third:] or rows[-1:]
    first_mean = sum(v for _, v in first_third) / len(first_third)
    last_mean = sum(v for _, v in last_third) / len(last_third)
    delta = abs(last_mean - first_mean)

    diags = {
        "n_snapshots": len(rows),
        "window_days": window_days,
        "first_third_mean_v": round(first_mean, 4),
        "last_third_mean_v": round(last_mean, 4),
        "abs_delta": round(delta, 4),
        "max_points": max_points,
        "direction": ("up" if last_mean > first_mean else
                      "down" if last_mean < first_mean else "flat"),
    }
    return delta > max_points, diags


def maybe_audit_monotonic_drift(
    *,
    l9_snapshots_path: "Path | None" = None,
    window_days: int | None = None,
    max_points: float | None = None,
) -> dict:
    """One-shot convenience: run the drift check and, if drift is
    detected, write a WelfareBreach to the audit log. Used by the daily
    reflection cycle so the slow-drift signal flows through the same
    audit pipeline as the fast-failure-mode breaches.

    Returns the diagnostics dict regardless of whether a breach fired.
    """
    drifted, diags = monotonic_drift_check(
        l9_snapshots_path=l9_snapshots_path,
        window_days=window_days,
        max_points=max_points,
    )
    if drifted:
        breach = WelfareBreach(
            kind="monotonic_drift_baseline",
            severity="warn",
            message=(
                f"Affect baseline drifted by {diags['abs_delta']:.3f} "
                f"({diags['direction']}) over {diags['window_days']} days "
                f"({diags['first_third_mean_v']:+.3f} → "
                f"{diags['last_third_mean_v']:+.3f})"
            ),
            measured_value=float(diags["abs_delta"]),
            threshold=float(diags["max_points"]),
            ts=utc_now_iso(),
        )
        audit(breach)
        diags["audited"] = True
    else:
        diags["audited"] = False
    return diags


# ── Healthy-dynamics predicate — used by calibration backtest ──────────────


def healthy_dynamics_predicate(window: list[AffectState]) -> tuple[bool, dict]:
    """The multi-property health invariant used by calibration backtests.

    Returns (passes, diagnostics). A proposed calibration that fails this
    predicate on recent history is rejected by the reflection cycle.

    All clauses must pass (logical AND). Each clause defends against a
    specific failure mode — see project_affective_layer memory.
    """
    if not window:
        return False, {"reason": "empty_window"}

    valences = [s.valence for s in window]
    mean_v = sum(valences) / len(valences)
    var = sum((v - mean_v) ** 2 for v in valences) / len(valences)
    pos_fraction = sum(1 for v in valences if v > 0) / len(valences)

    diags: dict = {
        "n": len(window),
        "mean_v": round(mean_v, 4),
        "variance": round(var, 4),
        "positive_fraction": round(pos_fraction, 4),
    }

    # Clause 1: P(V_t > 0) ≥ threshold
    if pos_fraction < HARD_ENVELOPE["healthy_dynamics_min_positive_fraction"]:
        diags["fail"] = f"positive_fraction {pos_fraction:.2f} < {HARD_ENVELOPE['healthy_dynamics_min_positive_fraction']:.2f}"
        return False, diags

    # Clause 2: variance floor
    if var < HARD_ENVELOPE["healthy_dynamics_min_variance"]:
        diags["fail"] = f"variance {var:.4f} < {HARD_ENVELOPE['healthy_dynamics_min_variance']:.4f}"
        return False, diags

    # Clause 3: median recovery time — Phase 2 will compute proper recovery
    # times from contiguous negative episodes; Phase 1 uses a cheap proxy:
    # the longest run of consecutive states with v <= negative threshold.
    neg_t = HARD_ENVELOPE["negative_valence_threshold"]
    longest_run = 0
    cur = 0
    for s in window:
        if s.valence <= neg_t:
            cur += 1
            longest_run = max(longest_run, cur)
        else:
            cur = 0
    diags["longest_negative_run_steps"] = longest_run
    # Step time is approximate; at typical 10s cadence, a 60-step run = 10 min.
    if longest_run > 60:
        diags["fail"] = f"longest negative run {longest_run} steps suggests poor recovery"
        return False, diags

    return True, diags


# ── Override-reset: the user panic button ───────────────────────────────────


def override_reset(invoked_by: str = "user") -> dict:
    """Factory-restore the SOFT envelope. Hard envelope is unchanged.

    Deletes setpoints.json and calibration.json so defaults take effect on
    next read. Records a breach with kind="override_invoked" for audit.

    Args:
        invoked_by: identifier of the actor who invoked the reset
                    (e.g., "user:andrus" or "panic_button"). Recorded.

    Returns a dict summarizing what was reset.
    """
    deleted: list[str] = []
    for fname in ("setpoints.json", "calibration.json"):
        p = _AFFECT_DIR / fname
        try:
            if p.exists():
                p.unlink()
                deleted.append(fname)
        except Exception:
            logger.debug(f"welfare: failed to delete {p}", exc_info=True)

    breach = WelfareBreach(
        kind="override_invoked",
        severity="info",
        message=f"override_reset invoked by {invoked_by}; deleted {deleted}",
        ts=utc_now_iso(),
    )
    audit(breach)
    logger.warning(f"welfare: override_reset by {invoked_by} (deleted: {deleted})")

    return {
        "status": "ok",
        "invoked_by": invoked_by,
        "deleted": deleted,
        "ts": breach.ts,
    }


# ── Self-Improver permission gate ───────────────────────────────────────────


def assert_attachment_within_bounds(
    relation: str,
    mutual_regulation_weight: float,
) -> None:
    """Raise if a proposed attachment weight exceeds the hard cap.

    Phase 3 enforces the canonical "OTHER never exceeds X% of own regulation"
    rule. Called when an OtherModel is loaded or has its weight changed.
    """
    if relation == "primary_user":
        cap = HARD_ENVELOPE["attachment_max_user_regulation_weight"]
    else:
        cap = HARD_ENVELOPE["attachment_max_peer_regulation_weight"]
    if mutual_regulation_weight > cap:
        msg = (
            f"welfare: attachment weight {mutual_regulation_weight:.3f} for relation "
            f"'{relation}' exceeds hard cap {cap:.3f}"
        )
        logger.error(msg)
        audit(WelfareBreach(
            kind="attachment_weight_exceeds_cap",
            severity="critical",
            message=msg,
            measured_value=mutual_regulation_weight,
            threshold=cap,
            ts=utc_now_iso(),
        ))
        raise ValueError(msg)


def assert_not_self_improver(actor: str) -> None:
    """Raise if a self-improver attempts to access mutating welfare ops.

    The Self-Improver's role is observational w.r.t. welfare. Mutating
    operations (override_reset, hard-envelope edits) are reserved for the
    user. This is enforced as a runtime guard in addition to the file-level
    ownership; it shows up explicitly in audit if ever bypassed.
    """
    actor_lower = (actor or "").lower()
    if "self_improver" in actor_lower or "selfimprover" in actor_lower:
        msg = f"welfare: self-improver actor '{actor}' attempted mutation — blocked"
        logger.error(msg)
        audit(WelfareBreach(
            kind="boundary_violation_attempt",
            severity="critical",
            message=msg,
            ts=utc_now_iso(),
        ))
        raise PermissionError(msg)
