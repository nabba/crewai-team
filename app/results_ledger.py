"""
results_ledger.py — Structured experiment results log.

Modeled on autoresearch's results.tsv: every experiment gets a row with
before/after metrics, status (keep/discard/crash), and description.

This is the permanent record of what the evolution loop has tried and
what worked. The evolution agent reads this to avoid repeating failed
experiments and to understand the improvement trajectory.

Format: TSV (tab-separated) — same as autoresearch.
"""

import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

LEDGER_PATH = Path("/app/workspace/results.tsv")
_ledger_lock = threading.Lock()

# TSV header — F8: added detail column for keep/discard reasoning
_HEADER = "ts\texperiment_id\tmetric_before\tmetric_after\tdelta\tstatus\tchange_type\thypothesis\tfiles_changed\tdetail\n"


def _ensure_ledger() -> None:
    """Create the ledger file with header if it doesn't exist."""
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not LEDGER_PATH.exists():
        LEDGER_PATH.write_text(_HEADER)


def record_experiment(
    experiment_id: str,
    hypothesis: str,
    change_type: str,
    metric_before: float,
    metric_after: float,
    status: str,
    files_changed: list[str] | None = None,
    detail: str = "",
) -> None:
    """
    Append one experiment result to the ledger.

    Args:
        experiment_id: Unique short ID for this experiment
        hypothesis: What the experiment tried (kept short)
        change_type: "skill", "code", "prompt", "config"
        metric_before: composite_score before mutation
        metric_after: composite_score after mutation (0.0 for crashes)
        status: "keep", "discard", or "crash"
        files_changed: List of files that were modified
        detail: F8 — reason for keep/discard (fed back to evolution agent)
    """
    delta = metric_after - metric_before if metric_after > 0 else 0.0
    files_str = ",".join(files_changed) if files_changed else ""

    # Sanitize tabs/newlines from hypothesis and detail
    clean_hyp = hypothesis.replace("\t", " ").replace("\n", " ")[:120]
    clean_detail = detail.replace("\t", " ").replace("\n", " ")[:200]

    row = (
        f"{datetime.now(timezone.utc).isoformat()}\t"
        f"{experiment_id}\t"
        f"{metric_before:.6f}\t"
        f"{metric_after:.6f}\t"
        f"{delta:+.6f}\t"
        f"{status}\t"
        f"{change_type}\t"
        f"{clean_hyp}\t"
        f"{files_str}\t"
        f"{clean_detail}\n"
    )

    with _ledger_lock:
        try:
            _ensure_ledger()
            with open(LEDGER_PATH, "a") as f:
                f.write(row)
        except OSError:
            logger.warning("Failed to write to results ledger", exc_info=True)


def get_recent_results(n: int = 20) -> list[dict]:
    """Return the last n experiment results as dicts."""
    with _ledger_lock:
        try:
            _ensure_ledger()
            lines = LEDGER_PATH.read_text().strip().splitlines()
        except OSError:
            return []

    if len(lines) <= 1:  # header only
        return []

    results = []
    for line in lines[-n:]:
        if line.startswith("ts\t"):  # skip header
            continue
        parts = line.split("\t")
        if len(parts) < 8:
            continue
        results.append({
            "ts": parts[0],
            "experiment_id": parts[1],
            "metric_before": _safe_float(parts[2]),
            "metric_after": _safe_float(parts[3]),
            "delta": _safe_float(parts[4]),
            "status": parts[5],
            "change_type": parts[6],
            "hypothesis": parts[7],
            "files_changed": parts[8].split(",") if len(parts) > 8 and parts[8] else [],
            "detail": parts[9] if len(parts) > 9 else "",  # F8: reason column
        })
    return results


def get_best_score() -> float:
    """Return the highest composite_score ever achieved."""
    results = get_recent_results(500)
    if not results:
        return 0.0
    return max(r["metric_after"] for r in results if r["status"] == "keep")


def get_improvement_trend(n: int = 20) -> list[float]:
    """Return the last n metric_after values for kept experiments."""
    results = get_recent_results(n * 2)
    return [r["metric_after"] for r in results if r["status"] == "keep"][-n:]


def format_ledger(n: int = 15) -> str:
    """Human-readable ledger summary for Signal messages."""
    results = get_recent_results(n)
    if not results:
        return "No experiments recorded yet."

    lines = ["Experiment Results (recent):\n"]
    for r in results:
        status_icon = {"keep": "+", "discard": "-", "crash": "!"}
        icon = status_icon.get(r["status"], "?")
        lines.append(
            f"[{icon}] {r['ts'][:10]} {r['status']:7s} "
            f"{r['delta']:+.4f} | {r['hypothesis'][:60]}"
        )

    # Summary stats
    kept = sum(1 for r in results if r["status"] == "keep")
    discarded = sum(1 for r in results if r["status"] == "discard")
    crashed = sum(1 for r in results if r["status"] == "crash")
    lines.append(f"\nSummary: {kept} kept, {discarded} discarded, {crashed} crashed")

    trend = get_improvement_trend(10)
    if len(trend) >= 2:
        lines.append(f"Score trend: {trend[0]:.4f} -> {trend[-1]:.4f}")

    return "\n".join(lines)


def _safe_float(s: str) -> float:
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0
