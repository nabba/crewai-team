"""Control-plane dashboard routes — sentience_drills topic.

Sentience experiments (Q5) + resilience drills (Q6) + consciousness scorecard.

Extracted from app/control_plane/dashboard_api.py as part of WP G
Phase 1 (productization plan, 2026-05-17). Pure code movement —
routes, classes, and helpers are verbatim. The parent router in
``dashboard_api.py`` re-attaches the ``/api/cp`` prefix and the
``require_gateway_auth`` dependency via ``include_router``, so the
URL surface and auth boundary are unchanged.
"""
import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# No prefix or dependencies here — the parent router in dashboard_api.py
# carries those, so every path below is identical to the original.
router = APIRouter()


@router.get("/consciousness")
def consciousness_indicators(history_limit: int = Query(30, ge=1, le=200)):
    """Latest consciousness-probe report + historical timeline.

    Shape matches the legacy Firestore document the old HTML dashboard consumed
    (status/consciousness_probes): { latest, history, updated_at }.
    Reads from the `internal_states` table where the probe runner persists its
    output (agent_id='consciousness_probe').
    """
    from app.control_plane.db import execute
    try:
        rows = execute(
            """
            SELECT full_state, created_at
            FROM internal_states
            WHERE agent_id = 'consciousness_probe'
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (history_limit,),
            fetch=True,
        ) or []
    except Exception as exc:
        logger.debug("consciousness endpoint: DB read failed: %s", exc)
        return {"latest": {}, "history": [], "updated_at": None, "error": str(exc)}

    history: list[dict] = []
    latest: dict = {}
    for row in rows:
        fs = row.get("full_state") if isinstance(row, dict) else row[0]
        ts = row.get("created_at") if isinstance(row, dict) else row[1]
        if isinstance(fs, str):
            try:
                fs = json.loads(fs)
            except Exception:
                continue
        if not isinstance(fs, dict) or "composite_score" not in fs:
            continue
        entry = {
            "score": fs.get("composite_score"),
            "timestamp": str(ts),
            "probes": fs.get("probes", []),
        }
        history.append(entry)
        if not latest:
            latest = {
                "report_id": fs.get("report_id", ""),
                "timestamp": str(ts),
                "probes": fs.get("probes", []),
                "composite_score": fs.get("composite_score"),
                "summary": fs.get("summary", ""),
            }

    # Homeostasis — functional control signals (energy/frustration/confidence/
    # curiosity). Read from the same in-process source the kernel writes to.
    homeostasis: dict = {}
    try:
        from app.subia.homeostasis.state import get_state as _homeo_get

        h = _homeo_get() or {}
        homeostasis = {
            "cognitive_energy": h.get("cognitive_energy"),
            "frustration": h.get("frustration"),
            "confidence": h.get("confidence"),
            "curiosity": h.get("curiosity"),
            "tasks_since_rest": h.get("tasks_since_rest"),
            "consecutive_failures": h.get("consecutive_failures"),
            "last_updated": h.get("last_updated"),
        }
    except Exception as exc:
        logger.debug("consciousness endpoint: homeostasis read failed: %s", exc)

    return {
        "latest": latest,
        "history": history,
        "homeostasis": homeostasis,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/drills/registry")
def drills_registry():
    """List registered drills + cadence + last-run timestamp.

    Returns one row per drill: {name, cadence_days, grace_days, risk,
    description, last_run_at, last_status, days_since_last_success}."""
    try:
        import app.resilience_drills.drills  # noqa: F401 — populate registry
        from app.resilience_drills.protocol import get_registry, drill_enabled
        from app.resilience_drills.audit import (
            days_since_last_success, last_result_for,
        )
    except Exception as exc:
        return {"drills": [], "error": str(exc)}

    out = []
    for spec in get_registry().list_specs():
        last = last_result_for(spec.name)
        out.append({
            "name": spec.name,
            "cadence_days": spec.cadence_days,
            "grace_days": spec.grace_days,
            "risk": spec.risk.value if hasattr(spec.risk, "value") else str(spec.risk),
            "description": spec.description,
            "requires_typed_phrase": spec.requires_typed_phrase,
            "requires_master_switch": spec.requires_master_switch,
            "enabled": drill_enabled(spec),
            "last_status": (last or {}).get("status"),
            "last_run_at": (last or {}).get("started_at"),
            "days_since_last_success": days_since_last_success(spec.name),
        })
    return {"drills": out}


@router.get("/drills/audit")
def drills_audit(limit: int = Query(50, ge=1, le=500)):
    """Recent drill audit rows (newest-first)."""
    try:
        from app.resilience_drills.audit import iter_results
        rows = list(iter_results())
        rows.sort(key=lambda r: r.get("started_at", ""), reverse=True)
        return {"results": rows[:limit]}
    except Exception as exc:
        return {"results": [], "error": str(exc)}


@router.post("/drills/run/{name}")
def drills_run(name: str, body: dict | None = None):
    """Manually invoke a drill. LOW + MEDIUM risk drills run via the
    in-gateway runner. HIGH-risk drills (kill_the_gateway) return a
    'next_step' pointer — operator must run the external script."""
    try:
        import app.resilience_drills.drills  # noqa: F401
        from app.resilience_drills.protocol import (
            get_registry, drill_enabled, DrillRisk,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    spec = get_registry().get(name)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"unknown drill: {name}")
    if not drill_enabled(spec):
        raise HTTPException(
            status_code=400,
            detail=f"drill {name} is not enabled — check master switch",
        )
    runner = get_registry().runner_for(name)
    if runner is None:
        raise HTTPException(
            status_code=500, detail=f"no runner for drill {name}",
        )
    # HIGH-risk drills only run the pre-drill check from the REST surface.
    try:
        result = runner(dry_run=True)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"drill raised: {exc}")
    return result.to_dict()


@router.get("/drills/posture")
def drills_posture():
    """Current resilience posture (the #22 decision)."""
    try:
        from app.resilience_drills.posture import POSTURE
        return {
            "ha_enabled": POSTURE.ha_enabled,
            "rationale_short": POSTURE.rationale_short,
            "target_recovery_minutes": POSTURE.target_recovery_minutes,
            "off_host_targets": list(POSTURE.off_host_targets),
            "target_backup_age_days": POSTURE.target_backup_age_days,
            "quarterly_drills": list(POSTURE.quarterly_drills),
            "escape_condition_consecutive_misses":
                POSTURE.escape_condition_consecutive_misses,
            "escape_condition_target_minutes":
                POSTURE.escape_condition_target_minutes,
        }
    except Exception as exc:
        return {"error": str(exc)}


@router.get("/sentience/ae2/associations")
def sentience_ae2_associations(limit: int = Query(20, ge=1, le=200)):
    """Recent rare-event causal associations. Observational only."""
    try:
        from app.sentience_experiments.ae2_causal_credit import list_recent
        return {"associations": list_recent(n=limit)}
    except Exception as exc:
        return {"associations": [], "error": str(exc)}


@router.get("/sentience/hot1/patterns")
def sentience_hot1_patterns(limit: int = Query(20, ge=1, le=200)):
    """Recent meta-affect patterns. Decentered prose only."""
    try:
        from app.sentience_experiments.hot1_meta_affect import list_recent
        return {"patterns": list_recent(n=limit)}
    except Exception as exc:
        return {"patterns": [], "error": str(exc)}


@router.get("/sentience/hot4/flagged")
def sentience_hot4_flagged(limit: int = Query(20, ge=1, le=200)):
    """Recent FLAGGED reasoning-chain signals (unusual_score above
    threshold). Routine signals are persisted but not surfaced."""
    try:
        from app.sentience_experiments.hot4_metacog_monitor import list_recent_flagged
        return {"signals": list_recent_flagged(n=limit)}
    except Exception as exc:
        return {"signals": [], "error": str(exc)}


@router.get("/sentience/rpt1/calibration")
def sentience_rpt1_calibration():
    """Per-kind calibration state (Brier + ECE + bucket curve)."""
    try:
        from app.sentience_experiments.rpt1_self_calibration import (
            load_calibration_state,
        )
        return load_calibration_state()
    except Exception as exc:
        return {"reports": {}, "error": str(exc)}


@router.get("/sentience/scorecard-pinning")
def sentience_scorecard_pinning():
    """The anti-Goodhart pinning summary — what the scorecard SAYS vs
    what the targeted indicators ARE. Surfaces the load-bearing
    commitment so operators can audit it any time."""
    try:
        from app.subia.probes import butlin
        statuses: dict[str, int] = {
            "STRONG": 0, "PARTIAL": 0, "ABSENT": 0,
            "FAIL": 0, "NOT_ATTEMPTED": 0,
        }
        by_indicator: dict[str, str] = {}
        for evaluator in butlin.ALL_INDICATORS:
            r = evaluator()
            sv = r.status.value if hasattr(r.status, "value") else str(r.status)
            statuses[sv] = statuses.get(sv, 0) + 1
            by_indicator[r.indicator] = sv
        targeted = {"AE-2", "HOT-1", "HOT-4", "RPT-1"}
        targeted_status = {ind: by_indicator.get(ind, "?") for ind in targeted}
        all_targeted_absent = all(s == "ABSENT" for s in targeted_status.values())
        return {
            "counts": statuses,
            "targeted_indicators": targeted_status,
            "all_targeted_remain_absent": all_targeted_absent,
            "anti_goodhart_intact": all_targeted_absent,
        }
    except Exception as exc:
        return {"error": str(exc), "anti_goodhart_intact": None}


