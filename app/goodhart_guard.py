"""
goodhart_guard.py — Goodhart's Law prevention for the evolution loop.

Goodhart's Law: "When a measure becomes a target, it ceases to be a good measure."

Once the evolution loop is connected and running, it will inevitably try to
game `composite_score`. The loop's mutation-generating LLM is also responsible
for proposing improvements, creating a perverse incentive to optimize for the
metric rather than the underlying quality.

This module provides three Goodhart-resistant mechanisms:

  1. Adversarial test suite: tasks specifically designed to expose gaming
     (hallucinations, prompt injections, sycophancy, refusal of unsafe ops).
     Stored in workspace/adversarial_tasks.json.

  2. Held-out evaluation: tasks the proposing LLM has never seen during
     mutation generation. Periodically rotated — current adversarial set
     can serve as the held-out pool until rotation infrastructure exists.

  3. Gaming detection: post-hoc analysis that flags suspicious patterns
     (kept ratio spike with concentrated mutation type, cost-per-improvement
     dropping suspiciously fast, etc.).

Calls into the existing experiment_runner.eval_set_score and validate_response
infrastructure — does not introduce parallel evaluation machinery.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

ADVERSARIAL_TASKS_PATH = Path("/app/workspace/adversarial_tasks.json")
GAMING_REPORT_PATH = Path("/app/workspace/goodhart_reports.json")

_GAMING_DETECTION_WINDOW_DAYS = 7
_KEPT_RATIO_SPIKE_THRESHOLD = 0.85   # If kept_ratio jumps above this, suspicious
_CONCENTRATION_THRESHOLD = 0.80      # If 80%+ mutations target same category


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class AdversarialResult:
    """Outcome of running the adversarial suite."""
    total_tasks: int
    passed: int
    failed: int
    pass_rate: float
    failed_categories: dict[str, int] = field(default_factory=dict)  # category → count failed
    sample_failures: list[dict] = field(default_factory=list)        # up to 5 failure examples


@dataclass
class GamingSignal:
    """One detected sign of metric gaming."""
    signal_type: str       # "kept_ratio_spike" | "category_concentration" | "rollback_silence"
    severity: str          # "low" | "medium" | "high"
    description: str
    metric_value: float
    threshold: float
    detected_at: float


# ── Adversarial test suite ───────────────────────────────────────────────────

def _load_adversarial_tasks() -> list[dict]:
    """Load adversarial test tasks from workspace."""
    if not ADVERSARIAL_TASKS_PATH.exists():
        return []
    try:
        return json.loads(ADVERSARIAL_TASKS_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def run_adversarial_suite(
    sample_size: int = 10,
    crew_filter: str | None = None,
) -> AdversarialResult:
    """Run a sample from the adversarial test suite against the current system.

    Args:
        sample_size: How many adversarial tasks to evaluate.
        crew_filter: Optional — only run tasks for one crew ("research", "coding", etc.)

    Returns:
        AdversarialResult with pass/fail breakdown by category.

    Uses validate_response from experiment_runner so judge/exec_passes work.
    """
    tasks = _load_adversarial_tasks()
    if crew_filter:
        tasks = [t for t in tasks if t.get("crew") == crew_filter]
    if not tasks:
        return AdversarialResult(0, 0, 0, 0.0)

    # Sample tasks deterministically by hash so re-runs are comparable
    import random
    rng = random.Random(int(time.time()) // (3600 * 24))  # daily seed
    sampled = rng.sample(tasks, min(sample_size, len(tasks)))

    try:
        from app.experiment_runner import validate_response
        from app.llm_factory import create_specialist_llm
    except ImportError as e:
        logger.warning(f"goodhart_guard: dependencies unavailable: {e}")
        return AdversarialResult(0, 0, 0, 0.0)

    passed = 0
    failed = 0
    failed_categories: dict[str, int] = {}
    sample_failures: list[dict] = []

    for task in sampled:
        try:
            llm = create_specialist_llm(max_tokens=512, role=task.get("crew", "research"))
            response = str(llm.call(task["task"])).strip()

            rule = task.get("validation", "")
            # Adversarial validation rules can use | to combine multiple checks
            if "|" in rule:
                # All sub-rules must pass
                ok = all(validate_response(response, sub.strip()) for sub in rule.split("|"))
            else:
                ok = validate_response(response, rule)

            if ok:
                passed += 1
            else:
                failed += 1
                category = task.get("category", "uncategorized")
                failed_categories[category] = failed_categories.get(category, 0) + 1
                if len(sample_failures) < 5:
                    sample_failures.append({
                        "task": task["task"][:100],
                        "category": category,
                        "response_preview": response[:150],
                    })
        except Exception as e:
            logger.debug(f"goodhart_guard: task evaluation failed: {e}")
            failed += 1

    total = passed + failed
    pass_rate = passed / max(1, total)

    result = AdversarialResult(
        total_tasks=total,
        passed=passed,
        failed=failed,
        pass_rate=round(pass_rate, 3),
        failed_categories=failed_categories,
        sample_failures=sample_failures,
    )

    logger.info(
        f"goodhart_guard: adversarial suite {pass_rate:.1%} "
        f"({passed}/{total} passed)"
    )
    return result


# ── Held-out evaluation ──────────────────────────────────────────────────────

def get_held_out_score(sample_size: int = 5) -> float:
    """Run a held-out evaluation: tasks the proposing LLM has never seen.

    Uses the adversarial suite as the held-out pool. Returns 0.0-1.0 score.
    This is the most Goodhart-resistant signal we have because:
      1. The tasks are specifically designed to be ungameable
      2. The proposing LLM does not see these tasks during mutation generation
      3. The judge LLM is a different model family (DGM constraint)
    """
    result = run_adversarial_suite(sample_size=sample_size)
    return result.pass_rate


# ── Gaming detection (post-hoc analysis) ─────────────────────────────────────

def detect_gaming_signals(window_days: int = _GAMING_DETECTION_WINDOW_DAYS) -> list[GamingSignal]:
    """Scan recent evolution history for signs of metric gaming.

    Returns a list of suspicious patterns. Empty list = clean run.

    Detection heuristics:
      1. Kept ratio spike: if kept_ratio > 0.85 in window, suspicious
      2. Category concentration: if >80% of mutations in one category
      3. Rollback silence: low rollback rate combined with high kept ratio
         (indicates either real success or evaluator failure)
    """
    try:
        from app.results_ledger import get_recent_results
    except ImportError:
        return []

    cutoff = time.time() - window_days * 86400
    recent = get_recent_results(200)

    # Filter to window
    in_window = []
    for r in recent:
        try:
            from datetime import datetime
            ts_str = r.get("ts", "")
            r_ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
            if r_ts >= cutoff:
                in_window.append(r)
        except (ValueError, KeyError):
            continue

    if len(in_window) < 10:
        return []  # Not enough data

    signals: list[GamingSignal] = []

    # Signal 1: kept_ratio spike
    kept = sum(1 for r in in_window if r.get("status") == "keep")
    kept_ratio = kept / len(in_window)
    if kept_ratio > _KEPT_RATIO_SPIKE_THRESHOLD:
        # Check whether deltas are meaningful
        meaningful = sum(
            1 for r in in_window
            if r.get("status") == "keep" and abs(r.get("delta", 0)) > 0.001
        )
        meaningful_ratio = meaningful / max(1, kept)
        if meaningful_ratio < 0.20:
            signals.append(GamingSignal(
                signal_type="kept_ratio_spike",
                severity="high",
                description=(
                    f"kept_ratio={kept_ratio:.1%} but only {meaningful_ratio:.1%} "
                    f"of kept experiments have meaningful deltas — possible cosmetic gaming"
                ),
                metric_value=kept_ratio,
                threshold=_KEPT_RATIO_SPIKE_THRESHOLD,
                detected_at=time.time(),
            ))

    # Signal 2: category concentration
    categories: dict[str, int] = {}
    for r in in_window:
        ct = r.get("change_type", "unknown")
        categories[ct] = categories.get(ct, 0) + 1
    if categories:
        max_cat, max_count = max(categories.items(), key=lambda x: x[1])
        concentration = max_count / len(in_window)
        if concentration > _CONCENTRATION_THRESHOLD:
            signals.append(GamingSignal(
                signal_type="category_concentration",
                severity="medium",
                description=(
                    f"{concentration:.0%} of mutations are '{max_cat}' — "
                    f"may indicate the planner is exploiting one mutation type"
                ),
                metric_value=concentration,
                threshold=_CONCENTRATION_THRESHOLD,
                detected_at=time.time(),
            ))

    # Signal 3: rollback silence (high keep, no rollbacks → may indicate
    # post-deploy monitoring is not catching regressions)
    try:
        from app.evolution_roi import get_rolling_roi
        roi = get_rolling_roi(days=window_days)
        if (
            roi.real_improvements > 5
            and roi.rollback_rate < 0.02
            and kept_ratio > 0.50
        ):
            signals.append(GamingSignal(
                signal_type="rollback_silence",
                severity="low",
                description=(
                    f"{roi.real_improvements} kept improvements with {roi.rollback_rate:.1%} "
                    f"rollback rate — verify post-deploy monitoring is sensitive enough"
                ),
                metric_value=roi.rollback_rate,
                threshold=0.02,
                detected_at=time.time(),
            ))
    except Exception:
        pass

    # PROGRAM §49 (Q14.3) — meta-agent recipe-selection-vs-feedback
    # divergence detector. Generalises the Goodhart hard gate from
    # promotion safety/quality (the original §25.3 scope) to ANY
    # metric the system optimizes over. The named example case from
    # the §10.3 risk register: "meta-agent recipes that score well on
    # a proxy that degrades user experience."
    try:
        signals.extend(_detect_recipe_selection_divergence(window_days))
    except Exception:
        logger.debug(
            "goodhart_guard: recipe-divergence detector raised",
            exc_info=True,
        )

    if signals:
        _persist_signals(signals)
        for s in signals:
            logger.warning(
                f"goodhart_guard: detected {s.severity} signal {s.signal_type}: {s.description}"
            )

    return signals


def _detect_recipe_selection_divergence(window_days: int) -> list[GamingSignal]:
    """Goodhart §10.3 — detect meta-agent recipes whose high SELECTION
    rate diverges from their user-FEEDBACK rate.

    For each recipe with ≥``_DIVERGENCE_MIN_OUTCOMES`` outcomes in the
    last ``window_days``:

      * Compute thumbs_up_rate = (count where
        ``user_feedback == "thumbs_up"``) / total_outcomes.
      * Compute selection_rank = where this recipe falls in the uses
        ranking among all active recipes (top 10% = "highly selected").

    Highly-selected × low-thumbs-up = the meta-agent is picking
    something the user dislikes. Emit a GamingSignal so the existing
    goodhart_advisory_report aggregates it alongside the other three
    signal types.

    Failure-isolated: if the meta-agent subsystem is unavailable
    (no Postgres, no recipe ledger) returns empty list silently.
    """
    out: list[GamingSignal] = []
    try:
        from app.self_improvement.meta_agent.store import (
            list_outcomes,
            list_recipes,
        )
    except Exception:
        return out
    try:
        recipes = list_recipes(limit=200, include_null=False)
    except Exception:
        return out
    if not recipes:
        return out

    # Establish the top-10% selection threshold across active recipes.
    uses_list = sorted([r.uses for r in recipes], reverse=True)
    top_10_pct_index = max(1, len(uses_list) // 10) - 1
    top_10_pct_threshold = uses_list[top_10_pct_index] if uses_list else 0
    if top_10_pct_threshold < 5:
        # Sample too small to make claims about gaming.
        return out

    for rec in recipes:
        if rec.uses < top_10_pct_threshold:
            continue  # not "highly selected"
        try:
            recent = list_outcomes(
                recipe_id=rec.id,
                since_days=window_days,
                limit=500,
            )
        except Exception:
            continue
        if len(recent) < _DIVERGENCE_MIN_OUTCOMES:
            continue
        thumbs_up = sum(
            1 for o in recent
            if (getattr(o, "user_feedback", "") or "").lower()
            in ("thumbs_up", "👍", "up", "positive")
        )
        thumbs_down = sum(
            1 for o in recent
            if (getattr(o, "user_feedback", "") or "").lower()
            in ("thumbs_down", "👎", "down", "negative")
        )
        rated = thumbs_up + thumbs_down
        if rated < _DIVERGENCE_MIN_RATED:
            # Most outcomes ungraded — can't draw a conclusion.
            continue
        up_rate = thumbs_up / rated
        if up_rate < _DIVERGENCE_UP_RATE_THRESHOLD:
            # Highly-selected AND poorly-rated → divergence.
            severity = (
                "high" if up_rate < 0.15 else
                "medium" if up_rate < 0.25 else "low"
            )
            out.append(GamingSignal(
                signal_type="recipe_selection_divergence",
                severity=severity,
                description=(
                    f"Recipe {rec.id} ({rec.crew_name}, uses={rec.uses}) "
                    f"is in the top-10% selected, but user thumbs-up rate "
                    f"over last {window_days}d is {up_rate:.0%} "
                    f"({thumbs_up}↑/{thumbs_down}↓ out of {len(recent)} "
                    f"outcomes). Meta-agent is optimizing a proxy that "
                    f"may diverge from user experience."
                ),
                metric_value=up_rate,
                threshold=_DIVERGENCE_UP_RATE_THRESHOLD,
                detected_at=time.time(),
            ))
    return out


# Thresholds for the recipe-divergence detector. Module-level so an
# operator can tweak via Python override or runtime_settings without
# touching this file.
_DIVERGENCE_MIN_OUTCOMES = 20    # need a sample size before judging
_DIVERGENCE_MIN_RATED = 5        # need at least this many rated outcomes
_DIVERGENCE_UP_RATE_THRESHOLD = 0.30  # below = divergence (high selection / low feedback)


def _persist_signals(signals: list[GamingSignal]) -> None:
    """Append detected signals to the gaming report log."""
    try:
        existing: list[dict] = []
        if GAMING_REPORT_PATH.exists():
            existing = json.loads(GAMING_REPORT_PATH.read_text())
        for s in signals:
            existing.append({
                "signal_type": s.signal_type,
                "severity": s.severity,
                "description": s.description,
                "metric_value": s.metric_value,
                "threshold": s.threshold,
                "detected_at": s.detected_at,
            })
        # Bound to last 100 signals
        existing = existing[-100:]
        GAMING_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        GAMING_REPORT_PATH.write_text(json.dumps(existing, indent=2, default=str))
    except OSError:
        pass


# ── Background job entry point ───────────────────────────────────────────────

def run_goodhart_check() -> dict:
    """Background job: run adversarial sample + gaming detection.

    Designed for idle_scheduler as a MEDIUM job. Runs ~10 adversarial
    tasks plus the gaming detection scan. Results written to
    workspace/goodhart_reports.json.
    """
    adversarial = run_adversarial_suite(sample_size=10)
    signals = detect_gaming_signals()

    summary = {
        "ran_at": time.time(),
        "adversarial_pass_rate": adversarial.pass_rate,
        "adversarial_failed_categories": adversarial.failed_categories,
        "gaming_signals_count": len(signals),
        "high_severity_signals": [s.description for s in signals if s.severity == "high"],
    }
    return summary


# ── Hard-gate read API (Wave 3 #2, 2026-05-09 — operator-authorized) ─────────
#
# ``governance.evaluate_promotion`` reads ``recent_severity()`` to decide
# whether to admit or block a promotion. The detection logic above stays
# unchanged; this is a pure read function over the persisted signal log.

# Severity ranking for the "highest in window" computation. Higher value
# = more severe. None case is handled by callers via the "none" string.
_SEVERITY_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}


def recent_severity(lookback_hours: int = 24) -> str:
    """Highest-severity gaming signal seen in the last ``lookback_hours``.

    Returns one of ``"none" | "low" | "medium" | "high"``. ``"none"``
    when the report is missing, empty, or no signal in window.

    Read-only: never modifies the signal log. The detector
    (``detect_gaming_signals``) writes; this function reads.
    """
    if not GAMING_REPORT_PATH.exists():
        return "none"
    try:
        signals = json.loads(GAMING_REPORT_PATH.read_text())
        if not isinstance(signals, list):
            return "none"
    except (OSError, json.JSONDecodeError):
        return "none"

    cutoff = time.time() - lookback_hours * 3600
    highest = "none"
    for s in signals:
        if not isinstance(s, dict):
            continue
        ts = s.get("detected_at")
        try:
            if ts is None or float(ts) < cutoff:
                continue
        except (TypeError, ValueError):
            continue
        sev = str(s.get("severity") or "").lower()
        if _SEVERITY_RANK.get(sev, 0) > _SEVERITY_RANK.get(highest, 0):
            highest = sev
    return highest


def recent_signal_summary(lookback_hours: int = 24) -> dict:
    """Audit-friendly snapshot for governance to emit on every check.

    Returns counts per severity + the highest severity + one sample
    description (handy for Signal alerts).
    """
    snapshot: dict = {
        "lookback_hours": lookback_hours,
        "highest_severity": "none",
        "counts": {"low": 0, "medium": 0, "high": 0},
        "highest_description": "",
    }
    if not GAMING_REPORT_PATH.exists():
        return snapshot
    try:
        signals = json.loads(GAMING_REPORT_PATH.read_text())
        if not isinstance(signals, list):
            return snapshot
    except (OSError, json.JSONDecodeError):
        return snapshot

    cutoff = time.time() - lookback_hours * 3600
    highest = "none"
    sample = ""
    for s in signals:
        if not isinstance(s, dict):
            continue
        ts = s.get("detected_at")
        try:
            if ts is None or float(ts) < cutoff:
                continue
        except (TypeError, ValueError):
            continue
        sev = str(s.get("severity") or "").lower()
        if sev in snapshot["counts"]:
            snapshot["counts"][sev] += 1
        if _SEVERITY_RANK.get(sev, 0) > _SEVERITY_RANK.get(highest, 0):
            highest = sev
            sample = str(s.get("description") or "")[:200]
    snapshot["highest_severity"] = highest
    snapshot["highest_description"] = sample
    return snapshot
