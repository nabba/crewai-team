"""qos — Quality-of-service measurement (PROGRAM §51 Q16 Theme 6).

Sibling to:
  * ``app/healing/monitors/latency_slo.py`` (39th healing monitor)
  * The future React dashboard at /cp/quality

Public API:
  * ``answer_regression`` — frozen Q-A pairs + quarterly LLM judge
"""
from __future__ import annotations

from app.qos.answer_regression import (
    FROZEN_QA_PAIRS,
    Verdict,
    RegressionRun,
    run_regression,
    latest_run,
    list_runs,
)

__all__ = [
    "FROZEN_QA_PAIRS",
    "Verdict",
    "RegressionRun",
    "run_regression",
    "latest_run",
    "list_runs",
]
