"""
evolution_roi.py — ROI tracking, improvement attribution, and auto-throttle.

After 38 days of evolution producing zero production deployments, the audit
revealed there was no cost-vs-value tracking. This module adds:

  1. Cost tracking: per-experiment USD spend across LLM tiers
  2. Attribution: which kept mutation actually moved composite_score?
  3. Engine ROI comparison: AVO vs ShinkaEvolve cost-effectiveness
  4. Auto-throttle: if ROI is poor for sustained periods, slow down evolution

The throttle is not a hard stop — it reduces evolution frequency proportionally
to ROI quality. This prevents the system from burning cost on mutations that
aren't producing value, while still allowing some exploration.

Thresholds are loaded from workspace/meta/roi_thresholds.json (evolvable via
meta-evolution).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

ROI_LEDGER_PATH = Path("/app/workspace/evolution_roi.json")
ROI_THRESHOLDS_PATH = Path("/app/workspace/meta/roi_thresholds.json")

_DEFAULT_THRESHOLDS: dict = {
    "rolling_window_days": 7,
    "throttle_floor_factor": 0.1,
    "min_real_improvement_per_dollar": 0.001,
    "throttle_rules": {
        "no_improvements_for_days": 14,
        "negative_roi_for_days": 14,
        "high_rollback_rate_threshold": 0.3,
        "throttle_factor_when_triggered": 0.25,
    },
}


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class ROIRecord:
    """One evolution experiment's cost/outcome record."""
    timestamp: float
    experiment_id: str
    engine: str  # "avo" | "shinka" | "meta"
    cost_usd: float
    delta: float  # Reported delta from experiment_runner
    status: str  # "keep" | "discard" | "stored" | "crash" | "rolled_back"
    attributed_improvement: float | None = None  # Computed post-hoc
    deployed: bool = False
    rolled_back: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ROISnapshot:
    """Aggregated ROI metrics over a window."""
    window_days: float
    total_cost_usd: float
    real_improvements: int
    rollbacks: int
    rollback_rate: float
    cost_per_improvement: float | None  # None if 0 improvements
    sample_size: int
    by_engine: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ── Persistence ──────────────────────────────────────────────────────────────

_ledger_lock = threading.Lock()


def _load_ledger() -> list[dict]:
    """Read the ROI ledger from disk. Returns [] if missing or invalid."""
    if not ROI_LEDGER_PATH.exists():
        return []
    try:
        return json.loads(ROI_LEDGER_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def _save_ledger(records: list[dict]) -> None:
    """Persist ledger atomically. Bounded to 1000 most recent entries."""
    try:
        ROI_LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Bound size
        bounded = records[-1000:] if len(records) > 1000 else records
        # Atomic write via temp file
        tmp = ROI_LEDGER_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(bounded, indent=2, default=str))
        tmp.replace(ROI_LEDGER_PATH)
    except OSError as e:
        logger.warning(f"evolution_roi: save failed: {e}")


def _load_thresholds() -> dict:
    """Load throttle thresholds from workspace/meta/, fall back to defaults."""
    try:
        if ROI_THRESHOLDS_PATH.exists():
            data = json.loads(ROI_THRESHOLDS_PATH.read_text())
            # Strip _meta keys
            return {k: v for k, v in data.items() if not k.startswith("_")}
    except Exception:
        pass
    return _DEFAULT_THRESHOLDS


# ── Recording ────────────────────────────────────────────────────────────────

def record_evolution_cost(
    experiment_id: str,
    engine: str,
    cost_usd: float,
    delta: float,
    status: str,
    deployed: bool = False,
    rolled_back: bool = False,
) -> None:
    """Append one ROI record. Called by evolution.py after each experiment.

    Args:
        experiment_id: Unique experiment ID from results_ledger.
        engine: "avo", "shinka", or "meta".
        cost_usd: Cumulative LLM cost for this experiment.
        delta: Reported composite_score delta.
        status: keep | discard | stored | crash | rolled_back.
        deployed: True if auto_deployer.schedule_deploy was called.
        rolled_back: True if post-deploy monitor reverted it.
    """
    record = ROIRecord(
        timestamp=time.time(),
        experiment_id=experiment_id,
        engine=engine,
        cost_usd=cost_usd,
        delta=delta,
        status=status,
        deployed=deployed,
        rolled_back=rolled_back,
    )
    with _ledger_lock:
        ledger = _load_ledger()
        ledger.append(record.to_dict())
        _save_ledger(ledger)


def mark_rollback(experiment_id: str) -> None:
    """Update an existing record to rolled_back=True (called by post-deploy monitor)."""
    with _ledger_lock:
        ledger = _load_ledger()
        for record in ledger:
            if record.get("experiment_id") == experiment_id:
                record["rolled_back"] = True
                record["status"] = "rolled_back"
                break
        _save_ledger(ledger)


# ── Improvement attribution ──────────────────────────────────────────────────

def attribute_improvement(experiment_id: str, window_hours: float = 24.0) -> float | None:
    """Compute the attributable improvement for a single kept experiment.

    Methodology: compare the rolling composite_score in the [N hours before
    this experiment] window against the [N hours after] window, excluding
    other experiments deployed in those windows.

    Returns None if not enough data or the experiment wasn't deployed.

    This is approximate but cheap. For >70% accuracy, increase window_hours
    or run repeated A/B in differential_test.py.
    """
    try:
        from app.results_ledger import get_recent_results
    except Exception:
        return None

    recent = get_recent_results(200)
    target = next((r for r in recent if r.get("experiment_id") == experiment_id), None)
    if not target:
        return None

    # Use the metric_after as the post-experiment baseline
    metric_after = target.get("metric_after", 0.0)
    metric_before = target.get("metric_before", 0.0)
    if metric_after <= 0 or metric_before <= 0:
        return None

    # Simple attribution: the recorded delta is our best estimate.
    # More sophisticated: subtract noise floor by averaging deltas in the
    # surrounding window from other experiments.
    target_ts_str = target.get("ts", "")

    # Compute noise floor: average abs(delta) from kept experiments in
    # the last 24h excluding this one.
    try:
        from datetime import datetime
        target_ts = datetime.fromisoformat(target_ts_str.replace("Z", "+00:00")).timestamp()
    except (ValueError, AttributeError):
        return target.get("delta", 0.0)

    noise_window_s = window_hours * 3600
    nearby_deltas = []
    for r in recent:
        if r.get("experiment_id") == experiment_id:
            continue
        try:
            r_ts = datetime.fromisoformat(r["ts"].replace("Z", "+00:00")).timestamp()
        except (ValueError, KeyError):
            continue
        if abs(r_ts - target_ts) <= noise_window_s:
            nearby_deltas.append(abs(r.get("delta", 0.0)))

    noise_floor = sum(nearby_deltas) / len(nearby_deltas) if nearby_deltas else 0.0
    raw_delta = target.get("delta", 0.0)

    # Attributed = raw delta minus noise floor (clamped at 0 for negative)
    attributed = max(0.0, abs(raw_delta) - noise_floor)
    return attributed if raw_delta >= 0 else -attributed


# ── ROI snapshots ────────────────────────────────────────────────────────────

def get_rolling_roi(days: float = 7.0) -> ROISnapshot:
    """Compute aggregate ROI over a rolling window.

    Args:
        days: Window size in days (default 7).

    Returns:
        ROISnapshot with cost, real_improvements, rollback_rate, by_engine breakdown.
    """
    cutoff = time.time() - days * 86400

    with _ledger_lock:
        records = _load_ledger()

    in_window = [r for r in records if r.get("timestamp", 0) >= cutoff]

    total_cost = sum(r.get("cost_usd", 0.0) for r in in_window)
    real_improvements = sum(
        1 for r in in_window
        if r.get("status") == "keep" and r.get("delta", 0.0) > 0.001 and not r.get("rolled_back")
    )
    rollbacks = sum(1 for r in in_window if r.get("rolled_back"))
    rollback_rate = rollbacks / max(1, len(in_window))

    cost_per_improvement = total_cost / real_improvements if real_improvements > 0 else None

    # Per-engine breakdown
    by_engine: dict[str, dict] = {}
    for engine in ("avo", "shinka", "meta"):
        engine_records = [r for r in in_window if r.get("engine") == engine]
        engine_cost = sum(r.get("cost_usd", 0.0) for r in engine_records)
        engine_improvements = sum(
            1 for r in engine_records
            if r.get("status") == "keep" and r.get("delta", 0.0) > 0.001
        )
        by_engine[engine] = {
            "experiments": len(engine_records),
            "cost_usd": round(engine_cost, 4),
            "real_improvements": engine_improvements,
            "cost_per_improvement": (
                round(engine_cost / engine_improvements, 4)
                if engine_improvements > 0 else None
            ),
        }

    return ROISnapshot(
        window_days=days,
        total_cost_usd=round(total_cost, 4),
        real_improvements=real_improvements,
        rollbacks=rollbacks,
        rollback_rate=round(rollback_rate, 3),
        cost_per_improvement=round(cost_per_improvement, 4) if cost_per_improvement else None,
        sample_size=len(in_window),
        by_engine=by_engine,
    )


# ── Auto-throttle ────────────────────────────────────────────────────────────

def should_throttle() -> tuple[bool, str, float]:
    """Determine whether to throttle evolution based on ROI.

    Returns:
        (throttle_active, reason, factor) where factor in [0.1, 1.0].
        factor=1.0 means full speed; factor<1.0 means reduce frequency.
    """
    thresholds = _load_thresholds()
    rules = thresholds.get("throttle_rules", _DEFAULT_THRESHOLDS["throttle_rules"])
    floor = thresholds.get("throttle_floor_factor", 0.1)
    triggered_factor = rules.get("throttle_factor_when_triggered", 0.25)

    # Check 1: No real improvements in N days?
    no_improv_days = rules.get("no_improvements_for_days", 14)
    snapshot = get_rolling_roi(days=no_improv_days)
    if snapshot.sample_size >= 5 and snapshot.real_improvements == 0:
        factor = max(floor, triggered_factor)
        return True, (
            f"No real improvements in {no_improv_days} days "
            f"({snapshot.sample_size} experiments, ${snapshot.total_cost_usd:.2f} spent)"
        ), factor

    # Check 2: Rollback rate too high?
    rollback_threshold = rules.get("high_rollback_rate_threshold", 0.3)
    recent = get_rolling_roi(days=3)  # tighter window for rollback signal
    if recent.sample_size >= 5 and recent.rollback_rate >= rollback_threshold:
        factor = max(floor, triggered_factor)
        return True, (
            f"High rollback rate ({recent.rollback_rate:.1%}) over last 3 days"
        ), factor

    # Check 3: Cost-per-improvement above threshold?
    min_value = thresholds.get("min_real_improvement_per_dollar", 0.001)
    if snapshot.cost_per_improvement and snapshot.cost_per_improvement > 0:
        improvements_per_dollar = 1.0 / snapshot.cost_per_improvement
        if improvements_per_dollar < min_value:
            factor = max(floor, 0.5)  # softer throttle for cost concerns
            return True, (
                f"Poor cost efficiency: ${snapshot.cost_per_improvement:.2f} per improvement"
            ), factor

    return False, "ROI within healthy range", 1.0


def get_throttle_factor() -> float:
    """Convenience: return the throttle factor (1.0 = full speed)."""
    _, _, factor = should_throttle()
    return factor


# ── Engine activity tracking ────────────────────────────────────────────────

# Backup signal files written by engines that may fail to record ROI.
# The dynamic selector consults BOTH the ROI ledger and these markers so a
# downstream recording bug can't lock the rotation rule in an infinite
# "engine never ran" loop. Engines without a marker file are not affected —
# the ROI ledger remains the primary signal.
_ATTEMPT_MARKER_PATHS: dict[str, Path] = {
    "shinka": Path("/app/workspace/shinka_last_attempt.json"),
}


def _read_attempt_marker(engine: str) -> float:
    """Return the timestamp from the engine's backup attempt marker, or 0.0."""
    path = _ATTEMPT_MARKER_PATHS.get(engine)
    if not path or not path.exists():
        return 0.0
    try:
        data = json.loads(path.read_text())
        ts = float(data.get("ts", 0.0))
        return ts if ts > 0 else 0.0
    except (json.JSONDecodeError, OSError, ValueError, TypeError):
        return 0.0


def get_last_run_timestamp(engine: str) -> float:
    """Return the unix timestamp of the most recent run by this engine.

    Consults two signals in order, returning whichever is newer:

      1. The ROI ledger entries tagged with this engine.
      2. The engine-specific attempt marker file (only for engines that
         maintain one — currently shinka).

    Returns 0.0 if neither signal is present. Used by the dynamic engine
    selector to enforce minimum-interval rotation: if ShinkaEvolve hasn't
    been tried in N days, force a session so we get fresh ROI data.
    """
    with _ledger_lock:
        records = _load_ledger()
    ledger_ts = max(
        (r.get("timestamp", 0.0) for r in records if r.get("engine") == engine),
        default=0.0,
    )
    marker_ts = _read_attempt_marker(engine)
    return max(ledger_ts, marker_ts)


def days_since_engine_run(engine: str) -> float:
    """Convenience: days since the engine last ran. Infinity if never."""
    last = get_last_run_timestamp(engine)
    if last <= 0:
        return float("inf")
    return (time.time() - last) / 86400


# ── Engine ROI comparison ───────────────────────────────────────────────────

def get_engine_recommendation() -> str:
    """Recommend the better-performing engine based on cost-per-improvement.

    Used by the dynamic engine selector in evolution.py. Falls back to
    AVO when there's insufficient data or both engines are tied.
    """
    snapshot = get_rolling_roi(days=14)  # longer window for engine comparison

    avo = snapshot.by_engine.get("avo", {})
    shinka = snapshot.by_engine.get("shinka", {})

    avo_cpi = avo.get("cost_per_improvement")
    shinka_cpi = shinka.get("cost_per_improvement")

    # Both have data: pick lower cost-per-improvement
    if avo_cpi and shinka_cpi:
        return "avo" if avo_cpi <= shinka_cpi else "shinka"

    # Only one has data: prefer it
    if avo_cpi and not shinka_cpi:
        return "avo"
    if shinka_cpi and not avo_cpi:
        return "shinka"

    # No data: default to AVO
    return "avo"
