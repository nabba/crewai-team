"""Control-plane dashboard routes — sentience_drills topic.

Sentience experiments (Q5) + resilience drills (Q6) + consciousness scorecard.

Extracted from app/control_plane/dashboard_api.py as part of WP G
Phase 1 (2026-05-17); wired into the parent router via
``include_router`` in Phase 2 (2026-05-18). The parent router in
``dashboard_api.py`` carries the ``/api/cp`` prefix and the
``require_gateway_auth`` dependency, both of which propagate to
every route here — so the URL surface and auth boundary are
identical to the pre-Phase-1 monolith.
"""
import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query, Request
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
    """List registered drills + cadence + last-run timestamp + Q18 state.

    Returns one row per drill with both the legacy §44 fields and the
    Q18 v2 fields ({state, consecutive_failures, next_attempt_after,
    has_baseline, …}). Existing React callers consume only the legacy
    fields; the new /cp/drills page consumes the v2 fields."""
    try:
        import app.resilience_drills.drills  # noqa: F401 — populate registry
        from app.resilience_drills.protocol import get_registry, drill_enabled
        from app.resilience_drills.audit import (
            days_since_last_success, last_result_for,
        )
        from app.resilience_drills import state as st
        from app.resilience_drills import baseline as bl
    except Exception as exc:
        return {"drills": [], "error": str(exc)}

    out = []
    for spec in get_registry().list_specs():
        last = last_result_for(spec.name)
        state = st.load_or_initialize(spec.name, warmup_days=spec.warmup_days)
        baseline = bl.load(spec.name)
        ok_now, runnable_reason = st.is_runnable_now(state)
        out.append({
            "name": spec.name,
            "cadence_days": spec.cadence_days,
            "grace_days": spec.grace_days,
            "warmup_days": spec.warmup_days,
            "risk": spec.risk.value if hasattr(spec.risk, "value") else str(spec.risk),
            "description": spec.description,
            "requires_typed_phrase": spec.requires_typed_phrase,
            "requires_master_switch": spec.requires_master_switch,
            "enabled": drill_enabled(spec),
            "last_status": (last or {}).get("status"),
            "last_run_at": (last or {}).get("started_at"),
            "days_since_last_success": days_since_last_success(spec.name),
            # Q18 — state-machine fields
            "state": state.state.value,
            "consecutive_failures": state.consecutive_failures,
            "consecutive_code_errors": state.consecutive_code_errors,
            "last_failure_class": state.last_failure_class,
            "last_failure_summary": state.last_failure_summary,
            "next_attempt_after": state.next_attempt_after,
            "warming_up_until": state.warming_up_until,
            "quarantined_at": state.quarantined_at,
            "quarantined_reason": state.quarantined_reason,
            "muted_at": state.muted_at,
            "muted_until": state.muted_until,
            "is_runnable_now": ok_now,
            "runnable_reason": runnable_reason,
            "has_baseline": baseline is not None,
            "baseline_ratified_at": baseline.ratified_at if baseline else None,
        })
    return {"drills": out}


@router.get("/drills/{name}")
def drill_detail(name: str):
    """Detail view for one drill: state + baseline + recent results +
    transitions. Used by the React drill-detail drawer."""
    try:
        import app.resilience_drills.drills  # noqa: F401
        from app.resilience_drills.protocol import get_registry, drill_enabled
        from app.resilience_drills.audit import iter_results, days_since_last_success
        from app.resilience_drills import state as st
        from app.resilience_drills import baseline as bl
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    spec = get_registry().get(name)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"unknown drill: {name}")

    state = st.load_or_initialize(spec.name, warmup_days=spec.warmup_days)
    baseline = bl.load(name)
    recent = []
    for row in iter_results():
        if row.get("drill_name") == name:
            recent.append(row)
    recent.sort(key=lambda r: r.get("started_at", ""), reverse=True)
    recent_observations = []
    for row in recent[:20]:
        obs = row.get("observation")
        if obs is not None:
            recent_observations.append({
                "observed_at": row.get("started_at"),
                "measurements": obs,
                "status": row.get("status"),
                "failure_class": row.get("failure_class"),
            })

    ok_now, runnable_reason = st.is_runnable_now(state)
    return {
        "spec": {
            "name": spec.name,
            "cadence_days": spec.cadence_days,
            "grace_days": spec.grace_days,
            "warmup_days": spec.warmup_days,
            "risk": spec.risk.value if hasattr(spec.risk, "value") else str(spec.risk),
            "description": spec.description,
            "requires_typed_phrase": spec.requires_typed_phrase,
            "requires_master_switch": spec.requires_master_switch,
            "enabled": drill_enabled(spec),
        },
        "state": state.to_dict(),
        "baseline": baseline.to_dict() if baseline else None,
        "is_runnable_now": ok_now,
        "runnable_reason": runnable_reason,
        "days_since_last_success": days_since_last_success(name),
        "recent_results": recent[:20],
        "recent_observations": recent_observations,
    }


@router.post("/drills/{name}/ratify-baseline")
async def drills_ratify_baseline(name: str, request: Request):
    """Operator-facing: ratify the latest observation (or a supplied
    one) as the baseline. Body shape::

        {
          "observation_at": "2026-05-18T...",   # optional — defaults to latest
          "tolerances": {"key": {"rule": "min", "value": 1}, ...},
          "notes": "..."
        }
    """
    try:
        import app.resilience_drills.drills  # noqa: F401
        from app.resilience_drills.protocol import get_registry
        from app.resilience_drills.audit import iter_results
        from app.resilience_drills.baseline import Observation, ratify_from_observation
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    spec = get_registry().get(name)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"unknown drill: {name}")
    body = await request.json() if request else {}

    target_at = (body or {}).get("observation_at")
    chosen: dict | None = None
    for row in iter_results():
        if row.get("drill_name") != name:
            continue
        if not row.get("observation"):
            continue
        if target_at and row.get("started_at") != target_at:
            continue
        if chosen is None or (row.get("started_at") or "") > (
                chosen.get("started_at") or ""):
            chosen = row
        if target_at:
            break
    if chosen is None:
        raise HTTPException(
            status_code=400,
            detail=("no observation available for ratification — "
                    "the drill must run at least once with an "
                    "observation payload first"),
        )

    observation = Observation(
        drill_name=name,
        observed_at=chosen.get("started_at", ""),
        measurements=dict(chosen.get("observation") or {}),
    )
    baseline = ratify_from_observation(
        observation,
        operator=str((body or {}).get("operator") or "operator-react"),
        tolerances=(body or {}).get("tolerances") or {},
        notes=str((body or {}).get("notes") or ""),
    )
    return {"ok": True, "baseline": baseline.to_dict()}


@router.post("/drills/{name}/unquarantine")
async def drills_unquarantine(name: str, request: Request):
    """Operator unquarantines a drill. Drill transitions to WATCH and
    becomes immediately runnable. Body: {"reason": "..."}."""
    try:
        from app.resilience_drills import state as st
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    body = await request.json() if request else {}
    record = st.unquarantine(
        name,
        operator=str((body or {}).get("operator") or "operator-react"),
        reason=str((body or {}).get("reason") or ""),
    )
    if record is None:
        raise HTTPException(status_code=404, detail=f"unknown drill: {name}")
    return {"ok": True, "state": record.to_dict()}


@router.post("/drills/{name}/mute")
async def drills_mute(name: str, request: Request):
    """Operator mutes a drill. Body::

        {"reason": "...", "until_iso": "2026-06-01T00:00:00Z"}

    ``until_iso`` is optional; when omitted, mute is indefinite.
    """
    try:
        from app.resilience_drills import state as st
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    body = await request.json() if request else {}
    record = st.mute(
        name,
        operator=str((body or {}).get("operator") or "operator-react"),
        reason=str((body or {}).get("reason") or ""),
        until_iso=(body or {}).get("until_iso"),
    )
    return {"ok": True, "state": record.to_dict() if record else None}


@router.post("/drills/{name}/unmute")
async def drills_unmute(name: str, request: Request):
    try:
        from app.resilience_drills import state as st
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    body = await request.json() if request else {}
    record = st.unmute(
        name,
        operator=str((body or {}).get("operator") or "operator-react"),
    )
    if record is None:
        raise HTTPException(status_code=404, detail=f"unknown drill: {name}")
    return {"ok": True, "state": record.to_dict()}


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
    """Manually invoke a drill via the Q18 orchestrator.

    The orchestrator threads state + baseline + audit; backoff is
    bypassed for operator-triggered invocations (the operator
    explicitly asked) but QUARANTINED / MUTED are still respected
    — those represent operator-meaningful states the operator
    should explicitly lift via the dedicated endpoints.

    LOW + MEDIUM risk drills run via the in-gateway runner. HIGH-risk
    drills (kill_the_gateway) return a 'next_step' pointer — operator
    must run the external script.
    """
    try:
        import app.resilience_drills.drills  # noqa: F401
        from app.resilience_drills.protocol import (
            get_registry, drill_enabled,
        )
        from app.resilience_drills.runner import invoke_drill_by_name
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
    result = invoke_drill_by_name(name, triggered_by="operator")
    if result is None:
        raise HTTPException(
            status_code=500, detail=f"no runner for drill {name}",
        )
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


