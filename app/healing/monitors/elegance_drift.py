"""elegance_drift — continuous code-quality observation.

The existing ``app.code_quality`` module is a mutation GATE — it scores
quality before/after a proposed change and rejects regressions. But it
only fires when the system already wants to mutate something. There is
no signal when *existing* code quietly rots: a hand-edit drops a type
hint, a refactor leaves a 30-McCabe branch in place, a stale ruff rule
starts triggering.

This monitor closes that gap by running the same ``measure_file_at_path``
across every ``app/**/*.py`` file weekly, persisting rolling history,
and alerting on per-file composite regressions versus the 8-week median.

Operates entirely on the public API of ``app.code_quality`` — no
duplicated scoring logic. Failure-isolated per file; one unparseable
module never blocks the rest of the scan.
"""
from __future__ import annotations

import json
import logging
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


NAME = "elegance_drift"
CADENCE_SECONDS = 24 * 3600
MASTER_SWITCH_KEY = "elegance_drift_monitor_enabled"
_INTERNAL_CADENCE_S = 7 * 24 * 3600

# A composite drop of more than this versus the rolling median is a
# regression worth surfacing. Matches the mutation-gate threshold in
# ``code_quality.QUALITY_REGRESSION_THRESHOLD`` for consistency.
_REGRESSION_DELTA = 0.10
# Number of historic samples to median over. 8 weeks at weekly cadence
# = ~2 months, enough signal to smooth one-off scans.
_MEDIAN_WINDOW = 8
# How many samples per file we keep on disk. Older entries pruned.
_HISTORY_CAP_PER_FILE = 26
# Skip directories whose files mutate on every run (caches, generated).
_SKIP_PARTS: frozenset[str] = frozenset({"__pycache__", ".pytest_cache"})
# Cap how many regressors we list in the Signal body — avoid spam.
_ALERT_TOP_N = 5


def _workspace_root() -> Path:
    env = os.environ.get("WORKSPACE_ROOT")
    if env:
        return Path(env)
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT)
    except Exception:
        return Path("/app/workspace")


def _history_path() -> Path:
    return _workspace_root() / "code_quality" / "elegance_history.json"


def _state_path() -> Path:
    return _workspace_root() / "healing" / "elegance_drift_state.json"


def _app_root() -> Path:
    here = Path(__file__).resolve()
    # crewai-team/app/healing/monitors/elegance_drift.py → crewai-team/app
    return here.parents[2]


# ── enable / cadence gates ─────────────────────────────────────────────


def _enabled() -> bool:
    try:
        from app.runtime_settings import get_elegance_drift_monitor_enabled
        return get_elegance_drift_monitor_enabled()
    except Exception:
        return True


def _read_state() -> dict[str, Any]:
    p = _state_path()
    if not p.exists():
        return {"last_run": 0.0, "outstanding_alerts": {}}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"last_run": 0.0, "outstanding_alerts": {}}


def _write_state(state: dict[str, Any]) -> None:
    p = _state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(p)


def _cadence_due(state: dict[str, Any]) -> bool:
    return (time.time() - float(state.get("last_run") or 0)) >= _INTERNAL_CADENCE_S


# ── history persistence ────────────────────────────────────────────────


def _read_history() -> dict[str, list[dict[str, Any]]]:
    p = _history_path()
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    return raw


def _write_history(history: dict[str, list[dict[str, Any]]]) -> None:
    p = _history_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(history, sort_keys=True), encoding="utf-8")
    tmp.replace(p)


def _append_sample(
    history: dict[str, list[dict[str, Any]]],
    path: str,
    sample: dict[str, Any],
) -> None:
    series = history.setdefault(path, [])
    series.append(sample)
    if len(series) > _HISTORY_CAP_PER_FILE:
        del series[: len(series) - _HISTORY_CAP_PER_FILE]


# ── core scan ──────────────────────────────────────────────────────────


def _iter_python_files(root: Path):
    for p in sorted(root.rglob("*.py")):
        if any(part in _SKIP_PARTS for part in p.parts):
            continue
        yield p


def _classify(prior: list[float], current: float) -> tuple[str, float | None]:
    """Return ('baseline'|'ok'|'regressed', median_or_None)."""
    if len(prior) < _MEDIAN_WINDOW // 2:
        return "baseline", None
    window = prior[-_MEDIAN_WINDOW:]
    median = statistics.median(window)
    if current < median - _REGRESSION_DELTA:
        return "regressed", median
    return "ok", median


def _emit_alert(summary: str, regressors: list[dict[str, Any]]) -> None:
    body_lines = ["Composite score dropped >10% vs 8-week median for:"]
    for r in regressors[:_ALERT_TOP_N]:
        body_lines.append(
            f"• {r['path']}: {r['median']:.2f} → {r['current']:.2f}"
        )
    try:
        from app.notify import notify
        notify(
            title="📉 Code-elegance regression",
            body="\n".join(body_lines),
            url="/cp/code-health",
            topic=f"elegance_drift:{len(regressors)}",
            arbitrate=True,
        )
    except Exception:
        logger.debug("elegance_drift: notify failed", exc_info=True)
    try:
        from app.identity.continuity_ledger import record_event
        record_event(
            kind="architectural_debt_drift",
            actor="elegance_drift_monitor",
            summary=summary,
            detail={
                "regressors_count": len(regressors),
                "top": [
                    {
                        "path": r["path"],
                        "median": r["median"],
                        "current": r["current"],
                    }
                    for r in regressors[:_ALERT_TOP_N]
                ],
            },
        )
    except Exception:
        logger.debug("elegance_drift: ledger emit failed", exc_info=True)


def run() -> dict[str, Any]:
    """One pass: scan, persist, alert on regressions."""
    summary: dict[str, Any] = {
        "checked": False, "scanned": 0, "skipped": 0, "regressors": 0,
        "baseline_filled": 0, "errors": 0,
    }
    if not _enabled():
        summary["disabled"] = True
        return summary
    state = _read_state()
    if not _cadence_due(state):
        summary["skipped_cadence"] = True
        return summary

    try:
        from app.code_quality import measure_file_at_path
    except Exception:
        logger.debug("elegance_drift: code_quality unavailable", exc_info=True)
        summary["errors"] += 1
        return summary

    history = _read_history()
    regressors: list[dict[str, Any]] = []
    now_iso = datetime.now(timezone.utc).isoformat()
    app_root = _app_root()
    repo_root = app_root.parent

    for path in _iter_python_files(app_root):
        try:
            score = measure_file_at_path(path)
        except Exception:
            summary["errors"] += 1
            continue
        if score is None:
            summary["skipped"] += 1
            continue
        try:
            rel = str(path.relative_to(repo_root))
        except ValueError:
            rel = str(path)

        prior_composites = [s.get("composite", 0.0) for s in history.get(rel, [])]
        verdict, median = _classify(prior_composites, score.composite)
        _append_sample(history, rel, {"ts": now_iso, "composite": score.composite})
        summary["scanned"] += 1
        if verdict == "baseline":
            summary["baseline_filled"] += 1
            continue
        if verdict == "regressed":
            assert median is not None  # narrows type
            regressors.append({
                "path": rel,
                "current": round(score.composite, 3),
                "median": round(median, 3),
                "delta": round(score.composite - median, 3),
            })

    _write_history(history)
    regressors.sort(key=lambda r: r["delta"])  # most-negative first
    summary["regressors"] = len(regressors)
    summary["top_regressors"] = regressors[:_ALERT_TOP_N]

    if regressors:
        msg = (
            f"{len(regressors)} files dropped >10% composite vs 8-week median"
        )
        _emit_alert(msg, regressors)

    state["last_run"] = time.time()
    state["last_summary"] = {
        k: v for k, v in summary.items() if k != "top_regressors"
    }
    _write_state(state)
    summary["checked"] = True
    return summary
