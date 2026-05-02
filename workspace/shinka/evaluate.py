"""
ShinkaEvolve evaluator for AndrusAI agent evolution.

This evaluator is called by ShinkaEvolve for each candidate program.
It imports the evolved code, runs the fixed test suite, and writes
metrics + correctness to the results directory for shinka to consume.

Pre-2026-04-29 the validator rejected list outputs because it checked
``isinstance(run_output, tuple)`` strictly. ShinkaEvolve serialises
sub-process results through JSON IPC, which round-trips tuples to
lists — so every initial-program evaluation came back as a list,
failed validation, and shinka recorded 'Incorrect' with score 0.0.
The fix accepts any 2-element sequence (tuple OR list).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from shinka.core import run_shinka_eval

logger = logging.getLogger(__name__)

# A 2-element output is required: (score, details_dict).
# After JSON IPC the tuple becomes a list, so we accept either.
_ScoreDetails = Sequence  # tuple[float, dict] or list at runtime


# ── Validation ───────────────────────────────────────────────────────────────

def validate_evaluation(run_output: Any) -> tuple[bool, Optional[str]]:
    """Validate that the evolved program produces a usable score + details.

    Accepts any 2-element ordered container — JSON IPC turns tuples into
    lists, so the historical strict ``isinstance(_, tuple)`` check rejected
    every shinka-evaluated program.

    Returns:
        (is_valid, error_message)
    """
    # Accept tuple or list, reject everything else
    if not isinstance(run_output, (tuple, list)):
        return False, f"Expected sequence of (score, details), got {type(run_output).__name__}"
    if len(run_output) != 2:
        return False, f"Expected length 2, got length {len(run_output)}"

    score, details = run_output[0], run_output[1]

    if not isinstance(score, (int, float)):
        return False, f"Score must be numeric, got {type(score).__name__}"

    if not (0.0 <= float(score) <= 1.0):
        return False, f"Score {score} outside [0.0, 1.0] range"

    if not isinstance(details, dict):
        return False, f"Details must be dict, got {type(details).__name__}"

    return True, None


def aggregate_metrics(results: list) -> dict[str, Any]:
    """Aggregate evaluation results into ShinkaEvolve's metrics format.

    Each item in ``results`` is the (score, details) sequence returned by
    ``run_evaluation``. ShinkaEvolve uses ``combined_score`` as the primary
    fitness signal; ``public`` / ``private`` are surfaced in the dashboard
    and run summary respectively.
    """
    if not results:
        return {"combined_score": 0.0, "error": "No results", "public": {}, "private": {}}

    first = results[0]
    if not isinstance(first, (tuple, list)) or len(first) != 2:
        return {
            "combined_score": 0.0,
            "error": f"Malformed result: {type(first).__name__}",
            "public": {},
            "private": {},
        }

    score, details = first[0], first[1]
    if not isinstance(details, dict):
        details = {}

    return {
        "combined_score": float(score),
        "public": {
            "tool_accuracy": details.get("tool_accuracy", 0.0),
            "route_accuracy": details.get("route_accuracy", 0.0),
        },
        "private": {
            "tool_correct": details.get("tool_correct", 0),
            "tool_total": details.get("tool_total", 0),
            "route_correct": details.get("route_correct", 0),
            "route_total": details.get("route_total", 0),
        },
    }


# ── Diagnostic capture ───────────────────────────────────────────────────────

def _write_diagnostic(results_dir: str, payload: dict) -> None:
    """Persist a small JSON file capturing what we received from shinka.

    Lets the next debugger see exactly what arrived without having to
    re-run shinka. Best-effort; never raises.
    """
    try:
        path = Path(results_dir) / "andrus_diagnostic.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str))
    except OSError:
        pass


def main(program_path: str, results_dir: str) -> tuple[dict, bool, Optional[str]]:
    """ShinkaEvolve evaluation entry point.

    Returns:
        (metrics_dict, correct_flag, error_message_or_None)
    """
    started_at = time.time()
    print(f"Evaluating program: {program_path}")
    print(f"Results dir: {results_dir}")
    os.makedirs(results_dir, exist_ok=True)

    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_evaluation",
        num_runs=1,
        validate_fn=validate_evaluation,
        aggregate_metrics_fn=aggregate_metrics,
    )

    # Always write a diagnostic — invaluable when shinka's own outputs are missing
    _write_diagnostic(results_dir, {
        "duration_s": round(time.time() - started_at, 2),
        "program_path": program_path,
        "correct": correct,
        "error_msg": error_msg,
        "metrics_keys": list(metrics.keys()) if isinstance(metrics, dict) else None,
        "combined_score": metrics.get("combined_score") if isinstance(metrics, dict) else None,
    })

    if correct:
        score = metrics.get("combined_score", 0)
        print(f"Evaluation passed. Score: {score:.4f}")
    else:
        print(f"Evaluation failed: {error_msg}")

    return metrics, correct, error_msg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AndrusAI ShinkaEvolve evaluator")
    parser.add_argument("--program_path", type=str, default="initial.py")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()
    main(args.program_path, args.results_dir)
