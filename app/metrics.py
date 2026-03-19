"""
metrics.py — Composite scalar metric system for evolution measurement.

Inspired by Karpathy's autoresearch: every experiment needs a fixed,
comparable metric. For autoresearch it's val_bpb; for us it's a
composite score combining task success rate, error rate, self-heal
effectiveness, and system capability breadth.

The composite_score is a single float (higher = better) that the
evolution loop uses for keep/discard decisions.
"""

import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

SKILLS_DIR = Path("/app/workspace/skills")


# ── Component metrics ────────────────────────────────────────────────────────

def _task_success_rate() -> float:
    """Fraction of recent tasks that completed without error (0.0-1.0)."""
    try:
        from app.conversation_store import count_recent_tasks
        total, successful = count_recent_tasks(hours=24)
        if total == 0:
            return 1.0  # no data = assume healthy
        return successful / total
    except Exception:
        return 1.0


def _error_rate_24h() -> float:
    """Errors per hour over the last 24 hours. Lower is better."""
    try:
        from app.self_heal import get_recent_errors
        errors = get_recent_errors(100)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        recent = [
            e for e in errors
            if e.get("ts", "") > cutoff.isoformat()
        ]
        return len(recent) / 24.0
    except Exception:
        return 0.0


def _self_heal_rate() -> float:
    """Fraction of errors that were successfully diagnosed (0.0-1.0)."""
    try:
        from app.self_heal import get_recent_errors
        errors = get_recent_errors(50)
        if not errors:
            return 1.0
        diagnosed = sum(1 for e in errors if e.get("diagnosed"))
        return diagnosed / len(errors)
    except Exception:
        return 1.0


def _skill_count() -> int:
    """Number of learned skill files."""
    try:
        if not SKILLS_DIR.exists():
            return 0
        return sum(
            1 for f in SKILLS_DIR.glob("*.md")
            if f.name != "learning_queue.md"
        )
    except Exception:
        return 0


def _output_quality_score() -> float:
    """Average confidence from recent self-reports (0.0-1.0).

    Maps self-report confidence levels to numeric scores and averages
    over recent reports. This captures the agents' own assessment of
    output quality — a proxy metric until external evaluation is added.
    """
    conf_map = {"high": 1.0, "medium": 0.6, "low": 0.3}
    try:
        from app.memory.chromadb_manager import retrieve_with_metadata
        items = retrieve_with_metadata("self_reports", "confidence assessment", n=20)
        if not items:
            return 0.7  # no data = assume decent
        scores = []
        for item in items:
            meta = item.get("metadata", {})
            conf = meta.get("confidence", "medium")
            scores.append(conf_map.get(conf, 0.6))
        return sum(scores) / len(scores) if scores else 0.7
    except Exception:
        return 0.7


def _avg_response_time() -> float:
    """Average task response time in seconds over the last 24 hours."""
    try:
        from app.conversation_store import avg_response_time
        return avg_response_time(hours=24)
    except Exception:
        return 0.0


def _evolution_efficiency() -> float:
    """Fraction of recent experiments that were kept (0.0-1.0)."""
    try:
        from app.results_ledger import get_recent_results
        results = get_recent_results(20)
        if not results:
            return 0.5  # no data = neutral
        kept = sum(1 for r in results if r.get("status") == "keep")
        return kept / len(results)
    except Exception:
        return 0.5


# ── Composite score ──────────────────────────────────────────────────────────

def composite_score() -> float:
    """
    Single scalar metric — higher is better.

    This is the 'val_bpb' equivalent: the ONE number the evolution loop
    uses to decide keep vs discard.

    Components (weighted):
      - task_success_rate (0.30): core purpose — do tasks succeed?
      - error_rate_24h    (0.20): system stability — how many errors?
      - self_heal_rate    (0.15): resilience — can the system fix itself?
      - output_quality    (0.15): quality — how confident are agents in output?
      - skill_breadth     (0.10): capability — how much has it learned?
      - response_time     (0.10): efficiency — how fast are responses?
    """
    success = _task_success_rate()
    errors = _error_rate_24h()
    heal = _self_heal_rate()
    quality = _output_quality_score()
    skills = _skill_count()
    resp_time = _avg_response_time()

    # Normalize error rate: 0 errors/hr = 1.0, 5+ errors/hr = 0.0
    error_score = max(0.0, 1.0 - errors / 5.0)

    # Normalize skill count: 0 = 0.0, 20+ = 1.0 (diminishing returns)
    skill_score = min(1.0, skills / 20.0)

    # Normalize response time: 0s = 1.0, 60s+ = 0.0
    if resp_time <= 0:
        time_score = 1.0
    else:
        time_score = max(0.0, 1.0 - resp_time / 60.0)

    score = (
        0.30 * success
        + 0.20 * error_score
        + 0.15 * heal
        + 0.15 * quality
        + 0.10 * skill_score
        + 0.10 * time_score
    )

    return round(score, 6)


def compute_metrics() -> dict:
    """
    Full metrics snapshot — used for evolution context and dashboard reporting.

    Returns a dict with all component metrics + the composite score.
    """
    success = _task_success_rate()
    errors = _error_rate_24h()
    heal = _self_heal_rate()
    quality = _output_quality_score()
    skills = _skill_count()
    resp_time = _avg_response_time()
    evo_eff = _evolution_efficiency()
    score = composite_score()

    return {
        "task_success_rate": round(success, 4),
        "error_rate_24h": round(errors, 4),
        "self_heal_rate": round(heal, 4),
        "output_quality": round(quality, 4),
        "skill_count": skills,
        "avg_response_time_s": round(resp_time, 2),
        "evolution_efficiency": round(evo_eff, 4),
        "composite_score": score,
        "measured_at": datetime.now(timezone.utc).isoformat(),
    }


def format_metrics(metrics: dict) -> str:
    """Human-readable metrics summary for Signal messages."""
    return (
        f"Composite Score: {metrics['composite_score']:.4f}\n"
        f"Task Success Rate: {metrics['task_success_rate']:.1%}\n"
        f"Output Quality: {metrics['output_quality']:.1%}\n"
        f"Error Rate (24h): {metrics['error_rate_24h']:.2f}/hr\n"
        f"Self-Heal Rate: {metrics['self_heal_rate']:.1%}\n"
        f"Skills: {metrics['skill_count']}\n"
        f"Avg Response Time: {metrics['avg_response_time_s']:.1f}s\n"
        f"Evolution Efficiency: {metrics['evolution_efficiency']:.1%}"
    )
