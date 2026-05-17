"""Control-plane dashboard routes — governance_ops topic.

Governance + proposals + ops (errors, anomalies, deploys, delegation, meta-agent).

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


@router.get("/governance/pending")
def pending_governance(project_id: str = Query(None)):
    from app.control_plane.governance import get_governance
    return get_governance().get_pending(project_id)


@router.post("/governance/{request_id}/approve")
def approve_governance(request_id: str):
    from app.control_plane.governance import get_governance
    ok = get_governance().approve(request_id)
    if not ok:
        raise HTTPException(404, "Request not found or already resolved")
    return {"status": "approved"}


@router.post("/governance/{request_id}/reject")
def reject_governance(request_id: str):
    from app.control_plane.governance import get_governance
    ok = get_governance().reject(request_id)
    if not ok:
        raise HTTPException(404, "Request not found or already resolved")
    return {"status": "rejected"}


class DelegationUpdate(BaseModel):
    enabled: bool


@router.get("/delegation")
def get_delegation_settings():
    """Return {crew: bool} for every crew that supports delegation mode."""
    try:
        from app.crews.delegation_settings import get_all
        return {"settings": get_all()}
    except Exception as exc:
        raise HTTPException(500, f"delegation settings unavailable: {exc}")


@router.post("/delegation/{crew}")
def set_delegation_setting(crew: str, body: DelegationUpdate):
    """Enable or disable delegation mode for a specific crew."""
    try:
        from app.crews.delegation_settings import set_enabled
        updated = set_enabled(crew, body.enabled)
        if crew not in updated:
            raise HTTPException(404, f"unknown crew: {crew}")
        return {"settings": updated, "crew": crew, "enabled": body.enabled}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"delegation toggle failed: {exc}")


class MetaAgentUpdate(BaseModel):
    enabled: bool


@router.get("/meta-agent")
def get_meta_agent_settings():
    """Return current meta-agent state for every supported crew.

    Includes ``master_env_on`` and per-crew env overrides so the Org
    Chart can render an "env override active" badge — the JSON toggle
    has no effect when an env var is set.
    """
    try:
        from app.self_improvement.meta_agent import (
            meta_agent_settings, is_master_on, explicit_flag_for,
        )
        settings = meta_agent_settings.get_all()
        env_overrides = {
            crew: explicit_flag_for(crew)
            for crew in settings
            if explicit_flag_for(crew) is not None
        }
        return {
            "settings": settings,
            "master_env_on": is_master_on(),
            "env_overrides": env_overrides,
        }
    except Exception as exc:
        raise HTTPException(500, f"meta-agent settings unavailable: {exc}")


@router.post("/meta-agent/{crew}")
def set_meta_agent_setting(crew: str, body: MetaAgentUpdate):
    """Enable or disable the meta-agent path for a specific crew.

    Note: this writes to the JSON layer only. If a per-crew env
    override is set, the JSON change has no effect until the env var
    is unset. The GET response surfaces ``env_overrides`` so the UI
    can flag this state.
    """
    try:
        from app.self_improvement.meta_agent import meta_agent_settings
        updated = meta_agent_settings.set_enabled(crew, body.enabled)
        if crew not in updated:
            raise HTTPException(404, f"unknown crew: {crew}")
        return {"settings": updated, "crew": crew, "enabled": body.enabled}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"meta-agent toggle failed: {exc}")


@router.get("/errors")
def recent_errors(limit: int = Query(20, ge=1, le=200)):
    """Recent errors + pattern counts from the self-heal journal."""
    recent: list[dict] = []
    patterns: dict[str, int] = {}
    err: str | None = None
    try:
        from app.healing.error_diagnosis import get_recent_errors, get_error_patterns
        recent = list(get_recent_errors(limit) or [])
        patterns = dict(get_error_patterns() or {})
    except Exception as exc:
        err = str(exc)
        logger.debug("errors endpoint: %s", exc)
    return {
        "recent": recent,
        "patterns": patterns,
        "total_recent": len(recent),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "error": err,
    }


@router.get("/anomalies")
def recent_anomalies(limit: int = Query(20, ge=1, le=200)):
    """Recent statistical anomaly alerts from the detector."""
    alerts: list[dict] = []
    err: str | None = None
    try:
        from app.anomaly_detector import get_recent_alerts
        alerts = list(get_recent_alerts(limit) or [])
    except Exception as exc:
        err = str(exc)
        logger.debug("anomalies endpoint: %s", exc)
    return {
        "recent_alerts": alerts,
        "total": len(alerts),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "error": err,
    }


@router.get("/error_audit")
def error_audit():
    """Permanent error monitor — aggregated stats + open anomalies.

    The shape is whatever ``app.observability.error_monitor.snapshot()``
    returns; the React dashboard renders it directly. The monitor itself
    runs on a 5-min cron registered in main.py.
    """
    try:
        from app.observability.error_monitor import snapshot
        return snapshot()
    except Exception as exc:
        logger.debug("error_audit endpoint: %s", exc)
        return {
            "summary": {"total_24h": 0, "total_1h": 0, "hourly_avg_24h": 0, "trend": "stable"},
            "top_patterns_24h": [],
            "trend_hourly": [],
            "active_anomalies": [],
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "error": str(exc),
        }


@router.post("/error_audit/anomaly/{anomaly_id}/acknowledge")
def acknowledge_anomaly(anomaly_id: str):
    """Mark an open anomaly as acknowledged (silenced; preserves history).

    Auto-resolution still runs in the background — if the underlying spike
    fades, the entry transitions to ``resolved``. Acknowledgement is the
    "I've seen this and don't need it on the dashboard anymore" signal.
    """
    try:
        from app.observability.error_monitor import acknowledge
        ok = acknowledge(anomaly_id)
        return {"ok": bool(ok), "anomaly_id": anomaly_id}
    except Exception as exc:
        logger.debug("acknowledge_anomaly endpoint: %s", exc)
        return {"ok": False, "anomaly_id": anomaly_id, "error": str(exc)}


@router.get("/deploys")
def recent_deploys(limit: int = Query(20, ge=1, le=200)):
    """Recent entries from the self-deploy pipeline log."""
    from pathlib import Path as _Path
    import json as _json
    entries: list[dict] = []
    err: str | None = None
    try:
        path = _Path("/app/workspace/deploy_log.json")
        if path.exists():
            try:
                raw = _json.loads(path.read_text() or "[]")
                if isinstance(raw, list):
                    entries = raw[-limit:][::-1]  # newest first
            except Exception as exc:
                err = f"deploy log parse: {exc}"
    except Exception as exc:
        err = str(exc)
    auto_deploy = None
    try:
        from app.config import get_settings
        auto_deploy = bool(getattr(get_settings(), "evolution_auto_deploy", False))
    except Exception:
        pass
    return {
        "recent": entries,
        "auto_deploy_enabled": auto_deploy,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "error": err,
    }


class ProposalAction(BaseModel):
    action: str   # "approve" | "reject" | "rollback"


@router.post("/proposals/{proposal_id}/action")
def apply_proposal_action(proposal_id: int, body: ProposalAction):
    """Apply ``action`` to the identified proposal and return the
    result text.  Synchronous — unlike the old Firestore-polled path
    which had a 0–5 minute latency, this applies immediately and the
    client gets the outcome in the HTTP response.

    Actions:
      * ``approve``  — calls ``app.proposals.approve_proposal(pid)``
      * ``reject``   — calls ``app.proposals.reject_proposal(pid)``
      * ``rollback`` — deferred: rollback still flows through Signal
        because it involves interactive undo of a deployed change.
    """
    action = (body.action or "").strip().lower()
    if action not in ("approve", "reject", "rollback"):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown action: {action!r} (expected approve/reject/rollback)",
        )
    try:
        if action == "approve":
            from app.proposals import approve_proposal
            result = approve_proposal(proposal_id)
        elif action == "reject":
            from app.proposals import reject_proposal
            result = reject_proposal(proposal_id)
        else:
            # Rollback surface kept intentionally — calls return a
            # pointer to Signal where rollback interactivity lives.
            result = f"Rollback #{proposal_id} — use Signal for rollback"
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)[:300])
    return {
        "proposal_id": proposal_id,
        "action": action,
        "result": str(result)[:4000],
    }


