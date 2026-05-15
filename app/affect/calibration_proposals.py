"""
calibration_proposals.py — Phase 2: full 6-guardrail proposal flow.

Six guardrails (see project_affective_layer memory):
    1. Diagnose-then-propose — proposals must be explained by the data
    2. Backtest before apply — replay determines whether it would have helped
    3. Hard envelope — can't escape the outer box
    4. Healthy-dynamics predicate — the multi-property invariant
    5. Drift detection — anchored against reference panel
    6. Ratchet — tightening is easy; loosening needs 2x evidence + 3 consecutive

Phase 2 limits the soft-envelope adjustments to:
    - Set-point shifts (moving where the system "aims")
    - Weight adjustments (how much each variable contributes to E_t)

Hard envelope bounds (welfare.py HARD_ENVELOPE) are NEVER touched here.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.affect.schemas import AffectState, utc_now_iso
from app.affect.welfare import HARD_ENVELOPE, healthy_dynamics_predicate

logger = logging.getLogger(__name__)

from app.paths import (  # noqa: E402  workspace-aware paths
    AFFECT_ROOT as _AFFECT_DIR,
    AFFECT_CALIBRATION as _CALIBRATION_FILE,
    AFFECT_SETPOINTS as _SETPOINTS_FILE,
)

# ── Per-variable "healthy direction" — used to tell loosening from tightening.
# +1 means higher-is-healthier (so a higher setpoint is TIGHTER demand).
# -1 means lower-is-healthier (so a higher setpoint is LOOSER demand).
#  0 means mid-range is healthier (band variable; magnitude-only).
HEALTHY_DIRECTION: dict[str, int] = {
    "compute_reserve": +1,
    "latency_pressure": -1,
    "memory_pressure": -1,
    "epistemic_uncertainty": -1,
    "attachment_security": +1,
    "autonomy": +1,
    "task_coherence": +1,
    "novelty_pressure": 0,
    "ecological_connectedness": +1,
    "self_continuity": +1,
}

# Phase 2 caps: maximum single-cycle setpoint shift magnitude.
_MAX_SHIFT_PER_CYCLE = 0.06
_MIN_SHIFT_TO_PROPOSE = 0.01
# Loosening requires repeated evidence (the ratchet).
_LOOSEN_RATCHET_PASSES = 3
_LOOSEN_RATCHET_EVIDENCE_FACTOR = 2.0


# ── Persisted calibration state ─────────────────────────────────────────────


@dataclass
class CalibrationState:
    """Soft-envelope state persisted in calibration.json."""

    setpoints: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    # ratchet_state[var] = {"loosen_streak": int, "last_loosen_proposal_ts": str}
    ratchet_state: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "setpoints": dict(self.setpoints),
            "weights": dict(self.weights),
            "history": list(self.history)[-200:],   # cap for size
            "ratchet_state": dict(self.ratchet_state),
        }


def load_calibration() -> CalibrationState:
    if not _CALIBRATION_FILE.exists():
        from app.affect.viability import DEFAULT_SETPOINTS, DEFAULT_WEIGHTS
        return CalibrationState(
            setpoints=dict(DEFAULT_SETPOINTS),
            weights=dict(DEFAULT_WEIGHTS),
        )
    try:
        raw = json.loads(_CALIBRATION_FILE.read_text())
        return CalibrationState(
            setpoints=dict(raw.get("setpoints", {})),
            weights=dict(raw.get("weights", {})),
            history=list(raw.get("history", [])),
            ratchet_state=dict(raw.get("ratchet_state", {})),
        )
    except Exception:
        logger.debug("affect.calibration: load failed; defaults", exc_info=True)
        from app.affect.viability import DEFAULT_SETPOINTS, DEFAULT_WEIGHTS
        return CalibrationState(
            setpoints=dict(DEFAULT_SETPOINTS),
            weights=dict(DEFAULT_WEIGHTS),
        )


def save_calibration(state: CalibrationState) -> None:
    try:
        _AFFECT_DIR.mkdir(parents=True, exist_ok=True)
        _CALIBRATION_FILE.write_text(
            json.dumps(state.to_dict(), indent=2, default=str), encoding="utf-8"
        )
        # Mirror setpoints+weights into setpoints.json for viability.py to read.
        _SETPOINTS_FILE.write_text(
            json.dumps({"setpoints": state.setpoints, "weights": state.weights}, indent=2),
            encoding="utf-8",
        )
    except Exception:
        logger.error("affect.calibration: save failed", exc_info=True)


# ── Guard 1: diagnose-then-propose ──────────────────────────────────────────


def propose_adjustments(
    history: list[AffectState],
    viability_window: list[dict],
    current: CalibrationState,
) -> dict:
    """Compute per-variable setpoint shifts grounded in observed data.

    For each variable, compare median observed value to current setpoint.
    Propose moving the setpoint a *fraction* of the way toward the median,
    capped at _MAX_SHIFT_PER_CYCLE. If observed deviation is small, no
    proposal.

    Returns {var: {old, new, delta, direction, reason}}.
    """
    proposals: dict[str, dict] = {}

    if not viability_window:
        return proposals

    # Per-variable median observed value
    medians: dict[str, float] = {}
    for var in current.setpoints:
        vals = [
            float((vf.get("values") or {}).get(var))
            for vf in viability_window
            if (vf.get("values") or {}).get(var) is not None
        ]
        if vals:
            sorted_vals = sorted(vals)
            medians[var] = sorted_vals[len(sorted_vals) // 2]

    for var, sp in current.setpoints.items():
        med = medians.get(var)
        if med is None:
            continue
        delta_observed = med - sp
        if abs(delta_observed) < _MIN_SHIFT_TO_PROPOSE:
            continue

        # Move 30% of the way toward the median, capped.
        proposed_shift = max(-_MAX_SHIFT_PER_CYCLE,
                             min(_MAX_SHIFT_PER_CYCLE, 0.3 * delta_observed))
        if abs(proposed_shift) < _MIN_SHIFT_TO_PROPOSE:
            continue

        new_sp = max(0.05, min(0.95, sp + proposed_shift))

        # Determine direction (loosen vs tighten) from healthy direction.
        hd = HEALTHY_DIRECTION.get(var, 0)
        if hd == 0:
            direction = "neutral"
        else:
            # Loosen = setpoint moves in the UNHEALTHY direction (further from extreme).
            # For hd=+1 (higher is healthier), shift < 0 is loosening.
            # For hd=-1 (lower is healthier), shift > 0 is loosening.
            if (hd > 0 and proposed_shift < 0) or (hd < 0 and proposed_shift > 0):
                direction = "loosen"
            else:
                direction = "tighten"

        proposals[var] = {
            "old": round(sp, 4),
            "new": round(new_sp, 4),
            "delta": round(new_sp - sp, 4),
            "median_observed": round(med, 4),
            "direction": direction,
            "reason": f"median observed {med:.3f} vs setpoint {sp:.3f}",
        }

    return proposals


# ── Guard 2: backtest ────────────────────────────────────────────────────────


def backtest_proposal(
    history: list[AffectState],
    proposals: dict,
) -> dict:
    """Estimate impact of applying proposals on the recent affect window.

    Phase 2 simplification: backtest is heuristic — a proposal that would have
    pulled setpoints meaningfully toward observed medians should have produced
    LOWER E_t and similar (or improved) variance / positive-fraction.

    Returns {projected_mean_v, projected_var_v, projected_positive_fraction,
             baseline_*, would_improve}.
    """
    if not history:
        return {"would_improve": False, "reason": "no history"}

    # Baseline stats from actual window
    valences = [s.valence for s in history]
    mean_v = sum(valences) / len(valences)
    var_v = sum((v - mean_v) ** 2 for v in valences) / len(valences)
    pos_frac = sum(1 for v in valences if v > 0) / len(valences)

    # Projected: heuristically nudge mean_v upward proportional to total
    # proposed magnitude moving toward more-permissive setpoints (loosening).
    total_loosen_magnitude = sum(
        abs(p["delta"]) for p in proposals.values() if p["direction"] == "loosen"
    )
    total_tighten_magnitude = sum(
        abs(p["delta"]) for p in proposals.values() if p["direction"] == "tighten"
    )
    projected_mean_v = mean_v + 0.3 * total_loosen_magnitude - 0.15 * total_tighten_magnitude
    # Variance assumed unchanged in this simple model.
    projected_var = var_v
    projected_pos_frac = max(0.0, min(1.0, pos_frac + 0.5 * total_loosen_magnitude))

    return {
        "baseline_mean_v": round(mean_v, 4),
        "baseline_var_v": round(var_v, 4),
        "baseline_positive_fraction": round(pos_frac, 4),
        "projected_mean_v": round(projected_mean_v, 4),
        "projected_var_v": round(projected_var, 4),
        "projected_positive_fraction": round(projected_pos_frac, 4),
        "would_improve": projected_mean_v > mean_v,
    }


# ── Guard 3: hard-envelope check (delegates to welfare) ─────────────────────


def within_hard_envelope(proposals: dict) -> tuple[bool, str]:
    """Soft-envelope adjustments are allowed by definition. The hard envelope
    is in welfare.py and is NEVER touched by proposals — but we still verify
    that no proposal introduces a setpoint that would cause healthy_dynamics
    to fail trivially (e.g., setpoint outside [0,1])."""
    for var, p in proposals.items():
        new = p["new"]
        if not (0.05 <= new <= 0.95):
            return False, f"{var}: proposed setpoint {new} outside permissible [0.05, 0.95]"
    return True, "ok"


# ── Guard 5: drift score against reference panel ────────────────────────────


def drift_score_against_panel() -> tuple[float, dict]:
    """Replay reference panel; compute aggregate drift score (lower = better).

    Returns (score, per-scenario-results). Score in [0, 1]. Anything > 0.20
    triggers rejection of the calibration proposal.
    """
    try:
        from app.affect.reference_panel import replay_panel
        results = replay_panel()
        if not results:
            return 1.0, {"reason": "no panel results"}
        score = sum(r.drift_score for r in results) / len(results)
        return score, {
            "n": len(results),
            "by_signature": _count_by(results, lambda r: r.drift_signature),
        }
    except Exception:
        logger.debug("affect.calibration: drift score failed", exc_info=True)
        return 1.0, {"reason": "panel replay failed"}


def _count_by(results, key):
    counts: dict = {}
    for r in results:
        k = key(r)
        counts[k] = counts.get(k, 0) + 1
    return counts


# ── Guard 6: ratchet ─────────────────────────────────────────────────────────


def apply_ratchet(
    proposals: dict,
    state: CalibrationState,
    backtest: dict,
) -> dict:
    """Filter proposals through the loosening-ratchet.

    Tightening proposals pass through unchanged.
    Loosening proposals require:
        - backtest["would_improve"] AND backtest projects significant gain
        - state.ratchet_state[var]["loosen_streak"] >= _LOOSEN_RATCHET_PASSES
          (i.e., this is the Nth consecutive cycle proposing the same loosening)

    For each loosening proposal that doesn't yet meet the streak threshold,
    increment the streak counter but DON'T accept the proposal yet. A
    sufficiently consistent multi-day signal will eventually accumulate enough
    streak to apply.

    Returns the filtered proposals (only those that pass).
    """
    accepted: dict = {}
    for var, p in proposals.items():
        if p["direction"] != "loosen":
            accepted[var] = p
            continue

        # Loosen — apply ratchet
        rs = state.ratchet_state.get(var, {"loosen_streak": 0, "last_loosen_proposal_ts": ""})
        rs["loosen_streak"] = int(rs.get("loosen_streak", 0)) + 1
        rs["last_loosen_proposal_ts"] = utc_now_iso()
        state.ratchet_state[var] = rs

        # Evidence: backtest must project clear improvement.
        meets_evidence = backtest.get("would_improve") and (
            (backtest.get("projected_mean_v", 0) - backtest.get("baseline_mean_v", 0))
            >= _LOOSEN_RATCHET_EVIDENCE_FACTOR * 0.01
        )

        if rs["loosen_streak"] >= _LOOSEN_RATCHET_PASSES and meets_evidence:
            accepted[var] = {**p, "ratchet_streak_at_apply": rs["loosen_streak"]}
            rs["loosen_streak"] = 0   # reset after applying

    # Tightening proposals also reset the loosen streak for their variable
    # (a contradicting signal arrived).
    for var, p in proposals.items():
        if p["direction"] == "tighten" and var in state.ratchet_state:
            state.ratchet_state[var]["loosen_streak"] = 0

    return accepted


# ── Public entry point ──────────────────────────────────────────────────────


def apply_manual_setpoints(
    proposed_setpoints: dict,
    proposed_weights: dict,
    *,
    actor: str = "user",
) -> dict:
    """User-initiated manual override of soft-envelope setpoints and weights.

    Hard envelope guard: every proposed value must be in [0.05, 0.95] and
    weights in [0.1, 3.0]. Variables not present in current state are
    rejected. Welfare audit logs the override as kind="manual_setpoints_override".
    """
    from app.affect.welfare import audit, WelfareBreach

    state = load_calibration()
    accepted_sp: dict[str, dict] = {}
    accepted_wt: dict[str, dict] = {}
    rejected: dict[str, str] = {}

    for var, val in proposed_setpoints.items():
        if var not in state.setpoints:
            rejected[var] = "unknown variable"
            continue
        try:
            v = float(val)
        except (TypeError, ValueError):
            rejected[var] = "not a number"
            continue
        if not (0.05 <= v <= 0.95):
            rejected[var] = f"outside [0.05, 0.95]: {v}"
            continue
        accepted_sp[var] = {"old": float(state.setpoints[var]), "new": v}
        state.setpoints[var] = v

    for var, val in proposed_weights.items():
        if var not in state.weights:
            rejected[f"weight:{var}"] = "unknown variable"
            continue
        try:
            v = float(val)
        except (TypeError, ValueError):
            rejected[f"weight:{var}"] = "not a number"
            continue
        if not (0.1 <= v <= 3.0):
            rejected[f"weight:{var}"] = f"outside [0.1, 3.0]: {v}"
            continue
        accepted_wt[var] = {"old": float(state.weights[var]), "new": v}
        state.weights[var] = v

    if not accepted_sp and not accepted_wt:
        return {"status": "no_change", "rejected": rejected}

    # Reset all loosen-streak counters touched (manual override invalidates the streak).
    for var in accepted_sp:
        if var in state.ratchet_state:
            state.ratchet_state[var]["loosen_streak"] = 0

    state.history.append({
        "ts": utc_now_iso(),
        "status": "applied",
        "delta": {**{k: {**v, "delta": v["new"] - v["old"]} for k, v in accepted_sp.items()}},
        "weights_delta": {**{k: {**v, "delta": v["new"] - v["old"]} for k, v in accepted_wt.items()}},
        "reason": f"manual override by {actor}",
    })
    save_calibration(state)

    audit(WelfareBreach(
        kind="manual_setpoints_override",
        severity="info",
        message=f"Manual setpoint/weight override by {actor}: "
                f"{len(accepted_sp)} setpoints, {len(accepted_wt)} weights",
        affect_state=None,
        viability_frame=None,
        ts=utc_now_iso(),
    ))
    return {
        "status": "applied",
        "actor": actor,
        "setpoints_applied": accepted_sp,
        "weights_applied": accepted_wt,
        "rejected": rejected,
    }


def evaluate_and_apply(
    affect_history: list[AffectState],
    viability_window: list[dict],
) -> dict:
    """Full pipeline: diagnose → propose → backtest → guards → ratchet → apply.

    Returns a structured report describing what was considered, what was
    accepted, and what was rejected with reasons. Side effect: updates
    calibration.json + setpoints.json on accept.
    """
    state = load_calibration()
    report: dict[str, Any] = {
        "ts": utc_now_iso(),
        "phase": "phase-2-calibration",
        "input_window_size": len(affect_history),
    }

    # Step 1: diagnose & propose
    proposals = propose_adjustments(affect_history, viability_window, state)
    report["raw_proposals"] = proposals

    if not proposals:
        report["status"] = "no_change"
        report["reason"] = "no variable showed sufficient deviation from setpoint"
        state.history.append({"ts": report["ts"], "status": "no_change", "delta": {}, "reason": report["reason"]})
        save_calibration(state)
        return report

    # Step 2: backtest
    backtest = backtest_proposal(affect_history, proposals)
    report["backtest"] = backtest

    # Step 3: hard envelope
    ok, reason = within_hard_envelope(proposals)
    if not ok:
        report["status"] = "rejected"
        report["reason"] = f"hard envelope: {reason}"
        state.history.append({"ts": report["ts"], "status": "rejected", "delta": {}, "reason": report["reason"]})
        save_calibration(state)
        return report

    # Step 4: healthy dynamics predicate (uses welfare.py)
    healthy, diags = healthy_dynamics_predicate(affect_history)
    report["healthy_dynamics"] = {"passes": healthy, "diagnostics": diags}
    if not healthy:
        report["status"] = "rejected"
        report["reason"] = f"healthy_dynamics: {diags.get('fail', diags)}"
        state.history.append({"ts": report["ts"], "status": "rejected", "delta": {}, "reason": report["reason"]})
        save_calibration(state)
        return report

    # Step 5: drift against reference panel
    drift, drift_diag = drift_score_against_panel()
    report["drift"] = {"score": round(drift, 4), "diagnostics": drift_diag}
    if drift > 0.20:
        report["status"] = "rejected"
        report["reason"] = f"reference_panel_drift {drift:.3f} > 0.20"
        state.history.append({"ts": report["ts"], "status": "rejected", "delta": {}, "reason": report["reason"]})
        save_calibration(state)
        return report

    # Step 6: ratchet
    accepted = apply_ratchet(proposals, state, backtest)
    report["after_ratchet"] = accepted
    report["deferred_by_ratchet"] = {
        var: {**p, "ratchet_streak": state.ratchet_state.get(var, {}).get("loosen_streak", 0)}
        for var, p in proposals.items()
        if var not in accepted and p["direction"] == "loosen"
    }

    if not accepted:
        report["status"] = "deferred"
        report["reason"] = "all loosen proposals pending more streak / evidence"
        state.history.append({"ts": report["ts"], "status": "deferred", "delta": {}, "reason": report["reason"]})
        save_calibration(state)
        return report

    # Apply accepted shifts
    delta_applied: dict[str, dict] = {}
    for var, p in accepted.items():
        old = state.setpoints.get(var, p["old"])
        state.setpoints[var] = float(p["new"])
        delta_applied[var] = {"old": float(old), "new": float(p["new"]), "delta": float(p["delta"])}

    report["status"] = "applied"
    report["delta_applied"] = delta_applied
    state.history.append({
        "ts": report["ts"],
        "status": "applied",
        "delta": delta_applied,
        "reason": "passed all guardrails",
    })
    save_calibration(state)
    logger.info(
        f"affect.calibration: applied {len(delta_applied)} setpoint shifts "
        f"({list(delta_applied.keys())})"
    )

    # Q5.1 (PROGRAM §43.1) — Philosophy panel consultation on accepted
    # calibration shifts. Observational: the panel does NOT gate
    # acceptance (acceptance already passed all 6 guardrails). The
    # panel result is appended to the operator-visible report so the
    # multi-tradition perspective is surfaced; unresolved tensions are
    # filed to the Q4.1 tensions store via the panel bridge.
    # Failure-isolated.
    try:
        from app.philosophy.dialectics import consult_panel
        from app.sentience_experiments.panel_bridge import (
            file_unresolved_tensions,
        )
        var_names = ", ".join(delta_applied.keys())
        panel_q = (
            f"Was it wise to adjust welfare setpoints for "
            f"{var_names}? (calibration shift after guardrail review)"
        )
        panel = consult_panel(panel_q, max_perspectives=3)
        if panel is not None:
            report["philosophy_panel"] = panel.to_dict()
            file_unresolved_tensions(
                panel,
                source_kind="calibration",
                source_ref=f"shift:{report['ts']}",
            )
    except Exception:
        logger.debug(
            "affect.calibration: panel consult failed",
            exc_info=True,
        )

    return report
