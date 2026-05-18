"""Control-plane dashboard routes — budgets_costs topic.

Budgets + costs + audit + credit-alerts — financial/audit surface.

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


# ── Module-level constants ──────────────────────────────────────────────
# Relocated from the original dashboard_api.py during the WP G Phase 1
# split (productization plan, 2026-05-17). These are owned here (budgets/
# costs is the primary consumer) and re-exported via the dashboard_api
# composition root's ``__getattr__`` for any back-compat caller.

_CREW_TO_AGENT: dict[str, str] = {
    "research":      "researcher",
    "coding":        "coder",
    "writing":       "writer",
    "pim":           "pim",
    "financial":     "financial_analyst",
    "media":         "media_analyst",
    "creative":      "creative_crew",
    "desktop":       "desktop",
    "devops":        "devops",
    "repo_analysis": "repo_analyst",
    "tech_radar":    "researcher",
}

# Canonical crew roster — the "kind" tag distinguishes user-addressable
# crews (natural-language dispatch) from internal crews (orchestration,
# quality review, reflection, self-learning).  Used by the cost-by-crew
# views here AND by ``get_crew_tasks`` in dashboard_routes_ops_misc.py
# (which imports it from this module — see the cross-topic import there).
_KNOWN_CREWS: tuple[tuple[str, str], ...] = (
    # User-addressable (11)
    ("research",       "user"),
    ("coding",         "user"),
    ("writing",        "user"),
    ("media",          "user"),
    ("creative",       "user"),
    ("pim",            "user"),
    ("financial",      "user"),
    ("desktop",        "user"),
    ("repo_analysis",  "user"),
    ("devops",         "user"),
    ("tech_radar",     "user"),
    # Internal (4)
    ("commander",        "internal"),
    ("critic",           "internal"),
    ("retrospective",    "internal"),
    ("self_improvement", "internal"),
)

_TOKEN_PERIODS = ("hour", "day", "week", "month", "year")


class BudgetOverride(BaseModel):
    project_id: str
    agent_role: str
    new_limit: float
    approver: str = "user"


class BudgetPauseToggle(BaseModel):
    project_id: str
    agent_role: str
    paused: bool
    approver: str = "user"


class BudgetSet(BaseModel):
    project_id: str
    agent_role: str
    limit_usd: float
    limit_tokens: int = None


@router.get("/budgets")
def get_budgets(project_id: str = Query(None)):
    from app.control_plane.budgets import get_budget_enforcer
    return get_budget_enforcer().get_status(project_id)


@router.get("/budgets/forecast")
def budgets_forecast(
    project_id: str = Query(None),
    history_months: int = Query(12, ge=2, le=36),
    forecast_months: int = Query(6, ge=1, le=12),
):
    """Cross-reference the cost-trend forecast against current budget
    caps. Returns the months projected to breach the aggregate budget.

    Observational only — system never auto-raises caps in response.
    PROGRAM §40.1 — Q3.1 (2026-05-11).
    """
    try:
        from app.control_plane.budgets import forecast_breach_periods
        breaches = forecast_breach_periods(
            history_months=history_months,
            forecast_months=forecast_months,
            project_id=project_id,
        )
    except Exception as exc:
        logger.debug("budgets/forecast failed: %s", exc, exc_info=True)
        breaches = []
    return {
        "breaches": breaches,
        "params": {
            "history_months": history_months,
            "forecast_months": forecast_months,
            "project_id": project_id,
        },
        "as_of": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/budgets")
def set_budget(body: BudgetSet):
    from app.control_plane.budgets import get_budget_enforcer
    get_budget_enforcer().set_budget(body.project_id, body.agent_role, body.limit_usd, body.limit_tokens)
    return {"status": "set"}


@router.post("/budgets/override")
def override_budget(body: BudgetOverride):
    from app.control_plane.budgets import get_budget_enforcer
    get_budget_enforcer().override_budget(body.project_id, body.agent_role, body.new_limit, body.approver)
    return {"status": "overridden"}


@router.post("/budgets/pause")
def toggle_budget_pause(body: BudgetPauseToggle):
    """Flip the is_paused flag for an agent's current-period budget row.

    Used by the Pause/Resume button on each Budget card.  Auto-creates
    the row at the project default limit if missing so a fresh agent
    can be pre-paused before it ever spends.
    """
    from app.control_plane.budgets import get_budget_enforcer
    enforcer = get_budget_enforcer()
    ok = enforcer.set_paused(body.project_id, body.agent_role, body.paused, body.approver)
    if not ok:
        # No row yet — seed one then flip.
        enforcer.set_budget(body.project_id, body.agent_role, 50.0)
        enforcer.set_paused(body.project_id, body.agent_role, body.paused, body.approver)
    return {"status": "paused" if body.paused else "unpaused"}


@router.get("/audit")
def get_audit_log(
    project_id: str = Query(None),
    actor: str = Query(None),
    action: str = Query(None),
    limit: int = Query(50),
):
    from app.control_plane.audit import get_audit
    return get_audit().query(
        project_id=project_id, actor=actor,
        action_prefix=action, limit=limit,
    )


@router.get("/audit/costs")
def audit_costs(project_id: str = Query(None)):
    from app.control_plane.audit import get_audit
    return get_audit().cost_summary(project_id)


def _per_crew_costs(project_id: str | None) -> list[dict]:
    """Per-crew (request-routing unit) from the SQLite request_costs
    tracker. ``crew`` here is what tracker.crew_name was set to at
    delegation time: 'research', 'coding', 'research+coding', …"""
    out: list[dict] = []
    try:
        from app.llm_benchmarks import get_crew_cost_stats
        for row in get_crew_cost_stats("year", project_id=project_id) or []:
            name = row.get("crew") or "unknown"
            requests = int(row.get("requests") or 0)
            avg_tokens = float(row.get("avg_tokens") or 0.0)
            out.append({
                "actor": name,
                "calls": requests,
                "total_cost": float(row.get("total_cost_usd") or 0.0),
                "total_tokens": int(avg_tokens * requests),
            })
    except Exception as exc:
        logger.debug("_per_crew_costs failed: %s", exc)
    return out


def _per_agent_role_costs(project_id: str | None) -> list[dict]:
    """Per-agent-role from control_plane.budgets.agent_role. One row per
    individual actor (coder, researcher, writer, critic, commander,
    self_improver, …) rather than per routing unit."""
    try:
        from app.control_plane.db import execute
        sql = (
            """SELECT COALESCE(NULLIF(agent_role, ''), 'unknown') AS actor,
                      SUM(spent_usd)    AS total_cost,
                      SUM(spent_tokens) AS total_tokens
                 FROM control_plane.budgets
                WHERE spent_usd > 0
            """
        )
        params: tuple = ()
        if project_id:
            sql += " AND project_id::text = %s"
            params = (project_id,)
        sql += " GROUP BY COALESCE(NULLIF(agent_role, ''), 'unknown')"
        return [
            {
                "actor": r["actor"],
                "calls": 0,  # budgets table doesn't carry a per-row call count
                "total_cost": float(r["total_cost"] or 0),
                "total_tokens": int(r["total_tokens"] or 0),
            }
            for r in execute(sql, params, fetch=True) or []
        ]
    except Exception as exc:
        logger.debug("_per_agent_role_costs failed: %s", exc)
        return []


def _per_agent_costs_derived(project_id: str | None) -> list[dict]:
    """Per-agent-role cost, derived from two sources so the view has
    useful data even when the budgets reconcile is still catching up:

    1. ``llm_benchmarks.request_costs`` — mapped through _CREW_TO_AGENT.
       Compound crew labels like 'research+coding' split their cost
       equally across the component agents.  Unknown crew names pass
       through unchanged.
    2. ``control_plane.budgets`` — merged in for any agent_role that
       didn't appear from the crew mapping (e.g. internal agents,
       'unknown' fallback).
    """
    by_actor: dict[str, dict] = {}

    def _acc(name: str, calls: int, cost: float, tokens: int) -> None:
        e = by_actor.setdefault(
            name, {"actor": name, "calls": 0, "total_cost": 0.0, "total_tokens": 0}
        )
        e["calls"] += calls
        e["total_cost"] += cost
        e["total_tokens"] += tokens

    # Source 1 — SQLite crew-level stats, fanned out to agent roles.
    for row in _per_crew_costs(project_id):
        crew = row["actor"]
        components = crew.split("+") if "+" in crew else [crew]
        n = max(len(components), 1)
        for c in components:
            role = _CREW_TO_AGENT.get(c.strip(), c.strip() or "unknown")
            _acc(
                role,
                row["calls"] // n if n else 0,
                row["total_cost"] / n,
                row["total_tokens"] // n if n else 0,
            )

    # Source 2 — budgets rows not already covered by the mapping.
    for row in _per_agent_role_costs(project_id):
        name = row["actor"]
        if name in by_actor:
            continue
        _acc(name, row["calls"], row["total_cost"], row["total_tokens"])

    return list(by_actor.values())


def _is_internal_agent(name: str) -> bool:
    internal = {n for n, kind in _KNOWN_CREWS if kind == "internal"}
    internal |= {"self_improver", "meta_evolver", "observer", "introspector"}
    # Idle-scheduler buckets (the rolled-up "idle_scheduler" plus every
    # per-job name set by app/idle_scheduler.py:_run_single_job's
    # agent_scope wrapper) are background work, not a user-addressable
    # crew. Classify them as internal so the Cost-by-Crew chart stays
    # uncluttered.
    if name == "idle_scheduler" or "-" in name:  # job names use hyphens
        return True
    return name in internal


def _shape_response(items: list[dict]) -> dict:
    items = sorted(items, key=lambda x: x["total_cost"], reverse=True)
    return {
        "by_actor": items,
        "total_cost": round(sum(i["total_cost"] for i in items), 6),
    }


@router.get("/costs/by-agent")
def costs_by_agent(project_id: str = Query(None)):
    """Cost per individual agent role (coder, researcher, writer, critic,
    commander, self_improver, …).  Derived from the SQLite per-crew
    tracker via _CREW_TO_AGENT plus any budgets rows not covered by the
    mapping. Answers "who did the LLM work" — distinct from
    /costs/by-crew which answers "which workload routed it there"."""
    return _shape_response(_per_agent_costs_derived(project_id))


@router.get("/costs/by-crew")
def costs_by_crew(project_id: str = Query(None)):
    """Cost per user-addressable crew (research, coding, writing, …).
    Anything not classified as an internal orchestration agent counts
    as a crew, so compound labels like 'research+coding' still show up
    here."""
    items = [c for c in _per_crew_costs(project_id) if not _is_internal_agent(c["actor"])]
    return _shape_response(items)


@router.get("/costs/by-internal-agent")
def costs_by_internal_agent(project_id: str = Query(None)):
    """Cost per internal orchestration agent (commander, critic,
    retrospective, self_improver). Reads the same budgets table as
    /costs/by-agent, filtered to the internal set."""
    items = [c for c in _per_agent_role_costs(project_id) if _is_internal_agent(c["actor"])]
    return _shape_response(items)


@router.get("/costs/daily")
def costs_daily(project_id: str = Query(None), days: int = Query(30)):
    from app.control_plane.db import execute
    rows = execute(
        """SELECT DATE(timestamp) as day,
                  SUM(cost_usd) as total_cost,
                  SUM(tokens) as total_tokens,
                  COUNT(*) as call_count
           FROM control_plane.audit_log
           WHERE cost_usd IS NOT NULL
             AND (%s IS NULL OR project_id::text = %s)
             AND timestamp >= NOW() - INTERVAL '%s days'
           GROUP BY DATE(timestamp)
           ORDER BY day DESC""",
        (project_id, project_id, days), fetch=True,
    )
    return rows or []


@router.get("/embedding-migration")
def embedding_migration_status():
    """Read-only embedding-migration status. PROGRAM §40 Item 12.

    Returns plan + state-machine snapshot + verifier report + window
    summary so the React side can render the migration card without
    issuing four separate calls. Safe to poll.
    """
    out: dict = {
        "plan": None,
        "state": None,
        "verify": None,
        "shadow_window": None,
        "switches": None,
        "as_of": datetime.now(timezone.utc).isoformat(),
    }
    try:
        from app.memory.embedding_migration import plan as plan_mod
        from app.memory.embedding_migration import state as state_mod
        from app.memory.embedding_migration import shadow_read as sr_mod
        from app.memory.embedding_migration import verify as verify_mod
        from app.runtime_settings import (
            get_embedding_migration_dual_write_enabled,
            get_embedding_migration_shadow_read_enabled,
            get_embedding_migration_cutover_enabled,
        )
        plan = plan_mod.load_plan()
        out["plan"] = plan.to_dict() if plan else None
        out["state"] = state_mod.get_state().to_dict()
        out["shadow_window"] = sr_mod.get_window_summary()
        # Verifier is best-effort; don't crash the endpoint if it raises.
        try:
            out["verify"] = verify_mod.verify().to_dict()
        except Exception as exc:
            out["verify"] = {"ok": False, "error": str(exc)}
        out["switches"] = {
            "dual_write_enabled": bool(get_embedding_migration_dual_write_enabled()),
            "shadow_read_enabled": bool(get_embedding_migration_shadow_read_enabled()),
            "cutover_enabled": bool(get_embedding_migration_cutover_enabled()),
        }
    except Exception as exc:
        logger.debug("embedding-migration endpoint failed", exc_info=True)
        out["error"] = str(exc)
    return out


@router.get("/source-ledger/state")
def source_ledger_state():
    """PROGRAM §56 iter-3 — operator-facing ledger health summary.

    Returns per-KB stats (row count, byte size, chain-verify status,
    compaction history, off-host upload state) plus the 8 master
    switches. Read-only; safe to poll. Powers the ``SourceLedgerCard``
    React component on /cp/settings.

    Chain verify is capped at >50k-row ledgers so the endpoint doesn't
    stall on huge histories.
    """
    out: dict = {
        "kbs": [],
        "switches": {},
        "as_of": datetime.now(timezone.utc).isoformat(),
    }
    try:
        from app.memory.source_ledger import state_summary
        from app.runtime_settings import (
            get_chromadb_source_ledger_enabled,
            get_chromadb_ledger_bootstrap_enabled,
            get_chromadb_ledger_drift_replay_enabled,
            get_chromadb_ledger_s3_upload_enabled,
            get_chromadb_ledger_gdrive_upload_enabled,
            get_chromadb_ledger_compaction_enabled,
            get_drill_source_ledger_replay_enabled,
            get_drill_embedding_rotation_enabled,
        )
        out["kbs"] = state_summary().get("kbs", [])
        out["switches"] = {
            "source_ledger_enabled": bool(get_chromadb_source_ledger_enabled()),
            "bootstrap_enabled": bool(get_chromadb_ledger_bootstrap_enabled()),
            "drift_replay_enabled": bool(get_chromadb_ledger_drift_replay_enabled()),
            "compaction_enabled": bool(get_chromadb_ledger_compaction_enabled()),
            "s3_upload_enabled": bool(get_chromadb_ledger_s3_upload_enabled()),
            "gdrive_upload_enabled": bool(get_chromadb_ledger_gdrive_upload_enabled()),
            "drill_replay_enabled": bool(get_drill_source_ledger_replay_enabled()),
            "drill_rotation_enabled": bool(get_drill_embedding_rotation_enabled()),
        }
    except Exception as exc:
        logger.debug("source-ledger/state endpoint failed", exc_info=True)
        out["error"] = str(exc)
    return out


@router.get("/costs/trends")
def costs_trends(
    project_id: str = Query(None),
    history_months: int = Query(12, ge=2, le=36),
    forecast_months: int = Query(6, ge=0, le=12),
    anomaly_window: int = Query(30, ge=7, le=90),
    anomaly_z: float = Query(3.0, ge=2.0, le=5.0),
):
    """Trend bundle for the React CostTrendsCard.

    PROGRAM §40 — Q3 Item 14. Returns ``{summary, monthly, forecast,
    anomalies, params, as_of}``. Read-only on ``audit_log``; safe per
    page-load (single round-trip per panel; no caching).
    """
    from app.control_plane.cost_trends import get_cost_trends
    return get_cost_trends(
        history_months=history_months,
        forecast_months=forecast_months,
        anomaly_window=anomaly_window,
        anomaly_z=anomaly_z,
        project_id=project_id,
    )


@router.get("/tokens")
def token_usage(project_id: str | None = Query(None)):
    """Aggregated token usage, request-level cost stats, and a simple monthly
    projection. Mirrors the payload the legacy dashboard consumed from the
    Firestore `status/tokens` + `status/request_costs` documents.

    When ``project_id`` is supplied, only rows tagged with that project are
    aggregated. Rows recorded before the per-project tagging migration landed
    have a NULL ``project_id`` and are excluded from filtered responses.
    """
    try:
        from app.llm_benchmarks import (
            get_token_stats,
            get_request_cost_stats,
            get_crew_cost_stats,
        )
    except Exception as exc:
        logger.debug("tokens endpoint: llm_benchmarks import failed: %s", exc)
        return {
            "stats": {p: [] for p in _TOKEN_PERIODS},
            "request_costs": {p: {} for p in ("day", "week", "month")},
            "by_crew": {"day": []},
            "projection": {"day_cost_usd": 0.0, "mtd_cost_usd": 0.0, "projected_monthly_usd": 0.0},
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "error": str(exc),
        }

    stats = {p: get_token_stats(p, project_id=project_id) for p in _TOKEN_PERIODS}
    request_costs = {
        p: get_request_cost_stats(p, project_id=project_id) for p in ("day", "week", "month")
    }

    day_cost = sum(float(r.get("cost_usd") or 0) for r in stats.get("day", []))
    month_cost = sum(float(r.get("cost_usd") or 0) for r in stats.get("month", []))

    return {
        "stats": stats,
        "request_costs": request_costs,
        "by_crew": {"day": get_crew_cost_stats("day", project_id=project_id)},
        "projection": {
            "day_cost_usd": round(day_cost, 6),
            "mtd_cost_usd": round(month_cost, 6),
            "projected_monthly_usd": round(day_cost * 30, 4),
        },
        "project_id": project_id,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/credit-alerts")
def credit_alerts():
    """Active credit/billing alerts grouped by provider.

    Response shape::

        {
          "alerts": {
            "openrouter": {
              "provider": "openrouter",
              "error": "Error code: 402 — requires more credits ...",
              "url": "https://openrouter.ai/settings/credits",
              "ts": "2026-04-28T19:42:18+00:00",
              "resolved": false
            }, ...
          },
          "count": 1
        }
    """
    try:
        from app.firebase.publish import _active_alerts
        return {
            "alerts": dict(_active_alerts),
            "count": len(_active_alerts),
        }
    except Exception as e:
        return {"error": str(e), "alerts": {}, "count": 0}


class CreditAlertDismiss(BaseModel):
    provider: str


@router.post("/credit-alerts/dismiss")
def dismiss_credit_alert(body: CreditAlertDismiss):
    """Manually clear a credit alert for a provider. Use after topping up."""
    try:
        from app.firebase.publish import resolve_credit_alert, _active_alerts
        if body.provider not in _active_alerts:
            return {"status": "not_found", "provider": body.provider}
        resolve_credit_alert(body.provider)
        logger.info(f"credit alert dismissed by user: {body.provider}")
        return {"status": "dismissed", "provider": body.provider}
    except Exception as e:
        raise HTTPException(500, str(e)[:200])


