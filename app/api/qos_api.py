"""REST API for QoS surfaces.

PROGRAM §51 — Q16 Theme 6. Endpoints under ``/api/cp/quality/*``:

  GET /latency                — latency SLO history + baselines
  GET /answer-regression      — latest run + history
  POST /answer-regression/run — force a regression run (operator)
  GET /rpt1-calibration       — RPT-1 self-calibration advisory (Theme 6.3)

The endpoints are read-only except the explicit force-run POST.
Failure-isolated: any upstream breakage returns a 200 with
``available=false`` rather than 500.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cp/quality", tags=["quality"])


@router.get("/latency")
def get_latency() -> dict:
    """Latency SLO history + trailing 4-week baselines."""
    try:
        from app.healing.monitors.latency_slo import history_snapshot
        return history_snapshot()
    except Exception as exc:
        logger.debug("qos_api.latency: %s", exc, exc_info=True)
        return {"available": False, "history": [], "baselines": {}}


@router.get("/answer-regression")
def get_answer_regression(limit: int = 20) -> dict:
    """Latest run snapshot + last N historical rows."""
    if not 1 <= limit <= 40:
        raise HTTPException(400, "limit must be in [1, 40]")
    try:
        from app.qos.answer_regression import latest_run, list_runs
        return {
            "latest": latest_run(),
            "history": list_runs(limit=limit),
        }
    except Exception as exc:
        logger.debug("qos_api.answer_regression: %s", exc, exc_info=True)
        return {"available": False, "latest": None, "history": []}


@router.post("/answer-regression/run")
def post_answer_regression_run() -> dict:
    """Force-run the regression suite NOW, bypassing the cadence
    gate. Cost-bearing — requires ``answer_regression_llm_enabled``
    if you want the LLM judge."""
    try:
        from app.qos.answer_regression import run_regression
        run = run_regression(force=True)
        if run is None:
            return {"ok": False, "reason": "master switch off"}
        return {"ok": True, "run": run.to_dict()}
    except Exception as exc:
        raise HTTPException(500, f"force-run failed: {type(exc).__name__}: {exc}")


# ── Theme 6.3 — RPT-1 calibration advisory ──────────────────────────────


def _read_jsonl(path: Path, *, limit: int = 200) -> list[dict]:
    """Read newest-last JSONL, return last N rows. Tolerant of broken
    lines."""
    if not path.exists():
        return []
    out: list[dict] = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    except OSError:
        return []
    return out[-limit:]


@router.get("/rpt1-calibration")
def get_rpt1_calibration() -> dict:
    """RPT-1 self-calibration: predictions vs outcomes per kind.

    Reads ``workspace/sentience/rpt1_calibration_state.json`` for the
    summary (Brier score, ECE, 10-bucket curve per kind) and the
    tail of ``workspace/sentience/rpt1_predictions.jsonl`` for recent
    forecasts.

    The displayed numbers are **advisory** — they never feed back into
    the predictive layer. This endpoint is the missing operator view
    the Q5.6 closure left unwired.
    """
    try:
        from app.paths import WORKSPACE_ROOT
        ws = Path(WORKSPACE_ROOT)
    except Exception:
        ws = Path("/app/workspace")
    state_path = ws / "sentience" / "rpt1_calibration_state.json"
    preds_path = ws / "sentience" / "rpt1_predictions.jsonl"
    summary: dict = {}
    if state_path.exists():
        try:
            summary = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            summary = {}
    recent = _read_jsonl(preds_path, limit=100)
    n_resolved = sum(
        1 for r in recent
        if r.get("resolution_at") and r.get("outcome") is not None
    )
    n_pending = len(recent) - n_resolved
    return {
        "available": bool(summary) or bool(recent),
        "summary": summary,
        "recent_count": len(recent),
        "n_resolved_recent": n_resolved,
        "n_pending_recent": n_pending,
        "advisory_only": True,
        "note": (
            "RPT-1 calibration is pinned write-only — these numbers "
            "are surfaced for operator review and NEVER feed back into "
            "the predictive layer."
        ),
    }
