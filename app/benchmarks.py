"""
benchmarks.py — Automated benchmarking system.

Tracks improvement metrics over time. Stores metrics in a JSON journal
following the same pattern as evolution.py.

Metrics tracked:
- task_completion_time: seconds per crew execution
- quality_score: derived from self-report confidence distribution
- proactive_intervention_rate: proactive triggers per N tasks
- memory_hit_rate: % of memory retrievals returning results
- policy_utilization_rate: % of loaded policies that were referenced
- self_assessment_accuracy: confidence vs critic review alignment
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

BENCHMARK_PATH = Path("/app/workspace/benchmarks.json")
MAX_ENTRIES = 1000


def _load_journal() -> list[dict]:
    try:
        if BENCHMARK_PATH.exists():
            return json.loads(BENCHMARK_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        pass
    return []


def _save_journal(entries: list[dict]) -> None:
    try:
        BENCHMARK_PATH.parent.mkdir(parents=True, exist_ok=True)
        from app.safe_io import safe_write_json
        safe_write_json(BENCHMARK_PATH, entries[-MAX_ENTRIES:])
    except OSError:
        logger.warning("Failed to write benchmark journal", exc_info=True)


def record_metric(
    metric_name: str,
    value: float,
    metadata: dict = None,
) -> None:
    """Record a single metric measurement."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "metric": metric_name,
        "value": value,
    }
    if metadata:
        entry["metadata"] = metadata

    journal = _load_journal()
    journal.append(entry)
    _save_journal(journal)


def get_benchmark_summary(last_n_runs: int = 10) -> dict:
    """Calculate rolling averages for all tracked metrics.

    Returns a dict with metric names as keys and their average values.
    """
    journal = _load_journal()
    if not journal:
        return {"status": "No benchmark data yet."}

    # Group by metric name
    by_metric: dict[str, list[float]] = {}
    for entry in journal[-last_n_runs * 6:]:  # ~6 metrics per run
        name = entry.get("metric", "")
        value = entry.get("value")
        if name and value is not None:
            by_metric.setdefault(name, []).append(float(value))

    # Calculate averages
    summary = {}
    for name, values in by_metric.items():
        recent = values[-last_n_runs:]
        summary[name] = {
            "avg": round(sum(recent) / len(recent), 2),
            "min": round(min(recent), 2),
            "max": round(max(recent), 2),
            "count": len(recent),
        }

    summary["total_entries"] = len(journal)
    return summary


def get_benchmark_trend(metric_name: str, last_n: int = 20) -> list[float]:
    """Return a time series of values for a specific metric."""
    journal = _load_journal()
    values = [
        float(e["value"])
        for e in journal
        if e.get("metric") == metric_name and e.get("value") is not None
    ]
    return values[-last_n:]


def compare_benchmarks(period1_n: int = 10, period2_n: int = 10) -> dict:
    """Compare two periods to detect improvement or regression.

    Period 1 = older (baseline), Period 2 = recent.
    Returns per-metric comparison with direction.
    """
    journal = _load_journal()
    if not journal:
        return {"status": "No data"}

    # Group by metric
    by_metric: dict[str, list[float]] = {}
    for entry in journal:
        name = entry.get("metric", "")
        value = entry.get("value")
        if name and value is not None:
            by_metric.setdefault(name, []).append(float(value))

    comparison = {}
    for name, values in by_metric.items():
        total = len(values)
        if total < period1_n + period2_n:
            continue

        period1 = values[-(period1_n + period2_n):-period2_n]
        period2 = values[-period2_n:]

        avg1 = sum(period1) / len(period1)
        avg2 = sum(period2) / len(period2)

        if avg1 == 0:
            change_pct = 0
        else:
            change_pct = round(((avg2 - avg1) / abs(avg1)) * 100, 1)

        direction = "improved" if change_pct > 5 else "declined" if change_pct < -5 else "stable"

        comparison[name] = {
            "period1_avg": round(avg1, 2),
            "period2_avg": round(avg2, 2),
            "change_pct": change_pct,
            "direction": direction,
        }

    return comparison


def format_benchmarks_for_display() -> str:
    """Format benchmark summary for human-readable display."""
    summary = get_benchmark_summary()
    if "status" in summary:
        return summary["status"]

    lines = ["Benchmark Summary:\n"]
    for key, val in summary.items():
        if key == "total_entries":
            lines.append(f"Total data points: {val}")
            continue
        if isinstance(val, dict):
            avg = val.get("avg", "?")
            count = val.get("count", "?")
            lines.append(f"  {key}: avg={avg} (n={count})")

    # Add trend comparison if enough data
    comparison = compare_benchmarks()
    if comparison and "status" not in comparison:
        lines.append("\nTrend Analysis:")
        for key, val in comparison.items():
            direction = val.get("direction", "?")
            change = val.get("change_pct", 0)
            lines.append(f"  {key}: {direction} ({change:+.1f}%)")

    return "\n".join(lines)
