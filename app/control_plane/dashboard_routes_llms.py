"""Control-plane dashboard routes — llms topic.

LLM catalogue + roles + discovery + promotions + pins + judges.

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

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# No prefix or dependencies here — the parent router in dashboard_api.py
# carries those, so every path below is identical to the original.
router = APIRouter()


@router.get("/llms/catalog")
def llm_catalog():
    """Current live LLM catalog + role assignments + configured cost mode.

    Reads the runtime ``CATALOG`` dict (mutated by the catalog builder) so
    newly-discovered models appear without a service restart.
    """
    models: list[dict] = []
    err: str | None = None
    mode = "balanced"
    try:
        from app.llm_catalog import CATALOG
        for name, entry in CATALOG.items():
            data = dict(entry)
            data["name"] = name
            models.append(data)
    except Exception as exc:
        err = str(exc)
        logger.debug("llms/catalog endpoint: %s", exc)
    # Read the live runtime mode (dashboard switch / Signal command /
    # env-config startup) so the dashboard reflects what the resolver
    # is actually using. Falls back to "balanced" on any failure.
    try:
        from app.llm_mode import get_mode
        mode = get_mode() or "balanced"
    except Exception:
        pass
    role_assignments: dict[str, str] = {}
    public_roles: list[str] = []
    modes_list: list[str] = []
    try:
        from app.llm_catalog import (
            resolve_role_default,
            PUBLIC_ROLES,
            RUNTIME_MODES,
        )
        public_roles = list(PUBLIC_ROLES)
        modes_list = list(RUNTIME_MODES)
        for role in public_roles:
            try:
                resolved = resolve_role_default(role, mode)
                if resolved:
                    role_assignments[role] = resolved
            except Exception:
                continue
    except Exception:
        pass
    return {
        "models": models,
        "role_assignments": role_assignments,
        # ``mode`` is the canonical unified axis. ``cost_mode`` is kept
        # as an alias in the payload for one release so legacy clients
        # keep working; migrate readers to ``mode``.
        "mode": mode,
        "cost_mode": mode,
        "roles": public_roles,     # single source of truth for the UI pin dialog
        "modes": modes_list,
        "cost_modes": modes_list,  # alias for legacy clients
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "error": err,
    }


@router.get("/llms/roles")
def llm_role_assignments_endpoint():
    """Explicit role → model assignments stored in PostgreSQL overrides table."""
    rows: list[dict] = []
    err: str | None = None
    try:
        from app.llm_role_assignments import list_assignments
        rows = list(list_assignments(active_only=True) or [])
    except Exception as exc:
        err = str(exc)
        logger.debug("llms/roles endpoint: %s", exc)
    return {
        "assignments": rows,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "error": err,
    }


@router.get("/llms/discovery")
def llm_discovery_status(limit: int = Query(50, ge=1, le=500)):
    """Recently-discovered models + their benchmarking/promotion status."""
    from app.control_plane.db import execute
    models: list[dict] = []
    err: str | None = None
    try:
        rows = execute(
            """SELECT model_id, provider, display_name, context_window,
                      cost_input_per_m, cost_output_per_m, multimodal, tool_calling,
                      benchmark_score, benchmark_role, per_role_scores,
                      status, promoted_tier, promoted_roles,
                      created_at, updated_at, promoted_at
               FROM control_plane.discovered_models
               ORDER BY COALESCE(updated_at, created_at) DESC
               LIMIT %s""",
            (limit,),
            fetch=True,
        ) or []
        models = rows
    except Exception as exc:
        err = str(exc)
        logger.debug("llms/discovery status: %s", exc)
    return {
        "discovered": models,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "error": err,
    }


class DiscoveryRun(BaseModel):
    max_benchmarks: int = 3


@router.post("/llms/discovery/run")
def llm_discovery_run(body: DiscoveryRun):
    """Trigger a discovery cycle synchronously. Returns summary counts."""
    try:
        from app.llm_discovery import run_discovery_cycle
        result = run_discovery_cycle(max_benchmarks=max(1, min(body.max_benchmarks, 10)))
        return {"status": "ok", "result": result}
    except Exception as exc:
        logger.warning("llms/discovery/run failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


class PromoteRequest(BaseModel):
    model: str
    reason: str = ""


@router.get("/llms/promotions")
def llm_promotions_endpoint():
    """List currently-promoted models (global boost)."""
    try:
        from app.llm_promotions import list_promotions_with_detail
        return {
            "promotions": list_promotions_with_detail(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        logger.debug("llms/promotions endpoint: %s", exc)
        return {"promotions": [], "error": str(exc)}


@router.post("/llms/promote")
def llm_promote_endpoint(body: PromoteRequest):
    """Promote a catalog model — becomes resolver's first choice where it fits."""
    try:
        from app.llm_promotions import promote
        ok = promote(
            body.model,
            promoted_by="user:dashboard",
            reason=body.reason or "dashboard promotion",
        )
        if not ok:
            raise HTTPException(
                status_code=400,
                detail=f"model {body.model!r} not in live CATALOG",
            )
        return {"status": "ok", "model": body.model}
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("llms/promote failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


class DemoteRequest(BaseModel):
    model: str


@router.post("/llms/demote")
def llm_demote_endpoint(body: DemoteRequest):
    """Remove a promotion. Model returns to the regular scored pool."""
    try:
        from app.llm_promotions import demote
        demote(body.model)
        return {"status": "ok", "model": body.model}
    except Exception as exc:
        logger.warning("llms/demote failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


class PinRequest(BaseModel):
    """Hand-pin request body.

    Clients should send ``mode`` (the unified runtime-mode axis).
    ``cost_mode`` is accepted as a legacy alias; if both are present,
    ``mode`` wins.
    """
    role: str
    mode: str | None = None
    cost_mode: str | None = None  # legacy alias
    model: str
    reason: str = ""

    def resolved_mode(self) -> str:
        return (self.mode or self.cost_mode or "balanced")


@router.get("/llms/pins")
def llm_pins_endpoint():
    """List currently-active hand pins."""
    try:
        from app.llm_role_assignments import list_pins
        return {
            "pins": list_pins(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        logger.debug("llms/pins endpoint: %s", exc)
        return {"pins": [], "error": str(exc)}


@router.post("/llms/pin")
def llm_pin_endpoint(body: PinRequest):
    """Hand-pin a model to (role, mode) — hard resolver override."""
    try:
        from app.llm_role_assignments import pin_role
        mode = body.resolved_mode()
        ok = pin_role(
            body.role, mode, body.model,
            assigned_by="user:dashboard",
            reason=body.reason or "dashboard pin",
        )
        if not ok:
            raise HTTPException(
                status_code=400,
                detail=f"pin rejected — {body.model!r} not in live CATALOG",
            )
        return {"status": "ok", "role": body.role,
                "mode": mode, "cost_mode": mode,  # alias for legacy clients
                "model": body.model}
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("llms/pin failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


class UnpinRequest(BaseModel):
    role: str
    mode: str | None = None
    cost_mode: str | None = None  # legacy alias

    def resolved_mode(self) -> str:
        return (self.mode or self.cost_mode or "balanced")


@router.post("/llms/unpin")
def llm_unpin_endpoint(body: UnpinRequest):
    """Remove hand pins for (role, mode). Resolver takes back over."""
    try:
        from app.llm_role_assignments import unpin_role
        mode = body.resolved_mode()
        n = unpin_role(body.role, mode)
        return {"status": "ok", "retired": n,
                "role": body.role, "mode": mode, "cost_mode": mode}
    except Exception as exc:
        logger.warning("llms/unpin failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


class JudgePinRequest(BaseModel):
    """Pin a specific catalog model as the judge for a provider family.

    Overrides the dynamic top-intelligence rotation. Use when the
    auto-picked judge is too slow / expensive / biased and you want a
    deterministic alternative.
    """
    provider_family: str
    model: str
    reason: str = ""


class JudgeUnpinRequest(BaseModel):
    provider_family: str


@router.get("/llms/judges")
def llm_judges_endpoint():
    """Return the active cross-eval judge rotation, pins, and agreement stats.

    Powers the dashboard's Judges panel. Three sections:

      * ``rotation`` — the 3-judge panel currently used by discovery
        / re-benchmarking, post-pin overrides. Each entry includes the
        provider family, catalog key, and whether it came from a pin.
      * ``pins`` — every active row in ``judge_pins`` for the operator
        override view (with reason / pinned_by / pinned_at).
      * ``agreement`` — last-24h aggregate stats so the user can spot
        high-disagreement panels and OpenRouter-fallback frequency.
    """
    rotation: list[dict] = []
    pins: list[dict] = []
    agreement: dict = {}
    err: str | None = None
    try:
        from app.llm_discovery import _discover_judges
        from app.llm_catalog import CATALOG
        from app.llm_judge_pins import list_pins as _list_judge_pins, list_pins_detailed
        from app.llm_judge_telemetry import agreement_stats

        pinned = _list_judge_pins()
        for catalog_key, family in _discover_judges():
            entry = CATALOG.get(catalog_key) or {}
            strengths = entry.get("strengths", {}) or {}
            rotation.append({
                "catalog_key": catalog_key,
                "provider_family": family,
                "tier": entry.get("tier"),
                "provider": entry.get("provider"),
                "reasoning_score": strengths.get("reasoning"),
                "pinned": pinned.get(family) == catalog_key,
            })
        pins = list_pins_detailed()
        agreement = agreement_stats(window_hours=24)
    except Exception as exc:
        err = str(exc)
        logger.debug("llms/judges endpoint: %s", exc)
    return {
        "rotation": rotation,
        "pins": pins,
        "agreement": agreement,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "error": err,
    }


@router.post("/llms/judges/pin")
def llm_judges_pin_endpoint(body: JudgePinRequest):
    """Pin ``body.model`` as the judge for ``body.provider_family``."""
    try:
        from app.llm_judge_pins import pin_judge
        ok = pin_judge(
            body.provider_family.strip().lower(),
            body.model,
            pinned_by="user:dashboard",
            reason=body.reason or "dashboard pin",
        )
        if not ok:
            raise HTTPException(
                status_code=400,
                detail=f"pin rejected — {body.model!r} not in live CATALOG",
            )
        return {"status": "ok",
                "provider_family": body.provider_family,
                "model": body.model}
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("llms/judges/pin failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/llms/judges/unpin")
def llm_judges_unpin_endpoint(body: JudgeUnpinRequest):
    """Remove the judge pin for ``body.provider_family``."""
    try:
        from app.llm_judge_pins import unpin_judge
        removed = unpin_judge(body.provider_family.strip().lower())
        return {"status": "ok",
                "provider_family": body.provider_family,
                "removed": removed}
    except Exception as exc:
        logger.warning("llms/judges/unpin failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/llms/judge-evaluations")
def llm_judge_evaluations_endpoint(
    limit: int = Query(50, ge=1, le=500),
    candidate_model: str | None = Query(None),
):
    """Return recent multi-judge scoring panels for the agreement table."""
    try:
        from app.llm_judge_telemetry import list_recent
        rows = list_recent(limit=limit, candidate_model=candidate_model)
        # Numeric fields come back as Decimal; coerce for JSON.
        for r in rows:
            for k in ("mean_score", "std_dev"):
                if r.get(k) is not None:
                    r[k] = float(r[k])
            scores = r.get("scores")
            if scores is not None:
                r["scores"] = [float(s) if s is not None else None for s in scores]
        return {
            "evaluations": rows,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        logger.debug("llms/judge-evaluations endpoint: %s", exc)
        return {"evaluations": [], "error": str(exc)}


