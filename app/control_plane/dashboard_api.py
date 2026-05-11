"""Control Plane dashboard API routes.

Provides REST endpoints for the React dashboard.
All routes prefixed with /api/cp/.

Auth (Phase B1): when ``GATEWAY_AUTH_REQUIRED=1`` is set, every route
on this router requires ``Authorization: Bearer <gateway-secret>``.
Default behaviour (env var unset) is pass-through, preserving the
laptop developer experience. See :mod:`app.control_plane.auth_dep`.
"""
import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.control_plane.auth_dep import require_gateway_auth

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/api/cp",
    tags=["control-plane"],
    dependencies=[Depends(require_gateway_auth)],
)

# ── Request models ───────────────────────────────────────────────────────────

class ProjectCreate(BaseModel):
    name: str
    mission: str = ""
    description: str = ""

class TicketCreate(BaseModel):
    title: str
    description: str = ""
    project_id: str = ""
    priority: int = 5

class TicketUpdate(BaseModel):
    status: str = ""
    result_summary: str = ""

class CommentCreate(BaseModel):
    author: str = "user"
    content: str

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

# ── Projects ─────────────────────────────────────────────────────────────────

@router.get("/projects")
def list_projects():
    from app.control_plane.projects import get_projects
    return get_projects().list_all()

@router.post("/projects")
def create_project(body: ProjectCreate):
    from app.control_plane.projects import get_projects
    result = get_projects().create(body.name, body.mission, body.description)
    if not result:
        raise HTTPException(400, "Failed to create project")
    return result

@router.get("/projects/{project_id}")
def get_project(project_id: str):
    from app.control_plane.projects import get_projects
    proj = get_projects().get_by_id(project_id)
    if not proj:
        raise HTTPException(404, "Project not found")
    return proj

@router.get("/projects/{project_id}/status")
def project_status(project_id: str):
    from app.control_plane.projects import get_projects
    return get_projects().get_status(project_id)

# ── Tickets ──────────────────────────────────────────────────────────────────

@router.get("/tickets")
def list_tickets(
    project_id: str = Query(None),
    status: str = Query(None),
    limit: int = Query(50),
):
    from app.control_plane.tickets import get_tickets
    if status:
        from app.control_plane.db import execute
        rows = execute(
            """SELECT * FROM control_plane.tickets
               WHERE (%s IS NULL OR project_id::text = %s)
                 AND status = %s
               ORDER BY created_at DESC LIMIT %s""",
            (project_id, project_id, status, limit), fetch=True,
        )
        return rows or []
    return get_tickets().get_recent(project_id, limit)

@router.get("/tickets/board")
def ticket_board(project_id: str = Query(None)):
    from app.control_plane.tickets import get_tickets
    return get_tickets().get_board(project_id)

@router.post("/tickets")
def create_ticket(body: TicketCreate):
    from app.control_plane.tickets import get_tickets
    from app.control_plane.projects import get_projects
    pid = body.project_id or get_projects().get_active_project_id()
    result = get_tickets().create_manual(body.title, pid, body.description, body.priority)
    if not result:
        raise HTTPException(400, "Failed to create ticket")
    return result

@router.get("/tickets/{ticket_id}")
def get_ticket(ticket_id: str):
    from app.control_plane.tickets import get_tickets
    ticket = get_tickets().get(ticket_id)
    if not ticket:
        raise HTTPException(404, "Ticket not found")
    return ticket

@router.put("/tickets/{ticket_id}")
def update_ticket(ticket_id: str, body: TicketUpdate):
    from app.control_plane.tickets import get_tickets
    tm = get_tickets()
    requeued = False
    if body.status == "done":
        tm.complete(ticket_id, body.result_summary or "Closed")
    elif body.status == "failed":
        tm.fail(ticket_id, body.result_summary or "Failed")
    elif body.status:
        from app.control_plane.db import execute
        execute(
            "UPDATE control_plane.tickets SET status = %s, updated_at = NOW() WHERE id = %s",
            (body.status, ticket_id),
        )
        if body.status == "todo":
            # Drag-to-todo from the dashboard means "run this again" — spawn
            # Commander in the background so a crew actually picks it up
            # instead of the ticket sitting orphaned.
            requeued = _requeue_ticket_async(ticket_id)
    return {"status": "updated", "requeued": requeued}


def _requeue_ticket_async(ticket_id: str) -> bool:
    """Fire-and-forget: dispatch the ticket's title through Commander so the
    existing routing pipeline assigns it to a crew and runs it.

    Leaves the original ticket in the 'todo' column as a historical marker
    and comments on it so the audit trail is obvious. Returns True when a
    background worker was spawned, False when the prerequisites couldn't be
    assembled (missing ticket, Commander unavailable, etc).
    """
    import threading
    try:
        from app.control_plane.tickets import get_tickets
        ticket = get_tickets().get(ticket_id)
        if not ticket:
            return False
        title = (ticket.get("title") or "").strip()
        if not title:
            return False

        def _worker():
            try:
                try:
                    get_tickets().add_comment(
                        ticket_id, "dashboard",
                        "Re-queued via dashboard drag-to-todo; routing through Commander.",
                    )
                except Exception:
                    logger.debug("requeue: comment write failed", exc_info=True)
                try:
                    from app.agents.commander import Commander
                    Commander().handle(title, sender="dashboard")
                except Exception:
                    logger.warning("requeue: commander dispatch failed", exc_info=True)
            except Exception:
                logger.debug("requeue: worker crashed", exc_info=True)

        threading.Thread(
            target=_worker,
            name=f"ticket-requeue-{ticket_id[:8]}",
            daemon=True,
        ).start()
        return True
    except Exception:
        logger.debug("requeue: setup failed", exc_info=True)
        return False

@router.post("/tickets/{ticket_id}/comments")
def add_comment(ticket_id: str, body: CommentCreate):
    from app.control_plane.tickets import get_tickets
    get_tickets().add_comment(ticket_id, body.author, body.content)
    return {"status": "added"}

# ── Budgets ──────────────────────────────────────────────────────────────────

@router.get("/budgets")
def get_budgets(project_id: str = Query(None)):
    from app.control_plane.budgets import get_budget_enforcer
    return get_budget_enforcer().get_status(project_id)


@router.get("/companion/tensions")
def companion_tensions(status: str = Query("OPEN"), min_freshness: float = Query(0.0, ge=0.0, le=1.0)):
    """Q4#16 (PROGRAM §41) — open questions the companion tracks on
    the operator's behalf. Read-only; mutations go through the
    POST endpoints below.
    """
    try:
        from app.companion.tensions import list_tensions
        tensions = list_tensions(status=status, min_freshness=min_freshness) or []
    except Exception as exc:
        logger.debug("companion/tensions failed: %s", exc, exc_info=True)
        return {"tensions": [], "error": str(exc)}
    return {
        "tensions": [
            {
                **t.to_dict(),
                "freshness": round(t.freshness(), 4),
            }
            for t in tensions
        ],
        "as_of": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/companion/cross-modal-patterns")
def companion_cross_modal_patterns(n: int = Query(20, ge=1, le=100), min_strength: float = Query(0.7, ge=0.0, le=1.0)):
    """Q4#15 (PROGRAM §41) — recent convergence patterns the system
    detected across modalities (conversations, emails, calendar,
    feedback, affect, tickets). Read-only.
    """
    try:
        from app.companion.cross_modal_patterns import list_recent_patterns
        patterns = list_recent_patterns(n=n, min_strength=min_strength) or []
    except Exception as exc:
        logger.debug("companion/cross-modal-patterns failed: %s", exc, exc_info=True)
        return {"patterns": [], "error": str(exc)}
    return {
        "patterns": patterns,
        "as_of": datetime.now(timezone.utc).isoformat(),
    }


class TensionResolve(BaseModel):
    resolution: str


@router.post("/companion/tensions/{tid}/resolve")
def companion_tensions_resolve(tid: str, body: TensionResolve):
    """Operator marks a tension RESOLVED with a short resolution note."""
    try:
        from app.companion.tensions import resolve_tension
        t = resolve_tension(tid, body.resolution)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    if t is None:
        raise HTTPException(status_code=404, detail=f"tension {tid!r} not found")
    return t.to_dict()


class TensionCreate(BaseModel):
    question: str
    workspace_id: str | None = None


@router.post("/companion/tensions")
def companion_tensions_create(body: TensionCreate):
    """Operator manually files a tension. Use case: operator says
    'I'm wondering about X' via /cp/companion rather than letting
    the regex detector pick it up from conversation."""
    try:
        from app.companion.tensions import create_tension
        t = create_tension(
            question=body.question,
            workspace_id=body.workspace_id,
            detection_source="manual:operator",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    if t is None:
        raise HTTPException(
            status_code=400,
            detail="tension rejected (question length or OPEN cap reached)",
        )
    return t.to_dict()


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

# ── Audit ────────────────────────────────────────────────────────────────────

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

# ── Idle scheduler observability (Phase E3) ──────────────────────────────────

@router.get("/idle/jobs")
def list_idle_jobs():
    """Snapshot of every known idle job's current state.

    Closes the cross-area gap where the dashboard previously showed
    only Firebase heartbeats while the idle scheduler ran ~100 jobs
    invisibly. Each entry includes failure_count, in_cooldown,
    cooldown_until_ts, seconds_since_last_success/failure, and
    currently_running. Read-only; calling this never affects the
    scheduler's behaviour.

    The ``inbound_dlq`` block reports the load-shed dead-letter
    queue's active backend (memory or redis) and current depth, so
    the React pane has a single endpoint for "what's happening in
    the background?" without making operators inspect env vars.
    """
    from app.idle_scheduler import get_job_snapshot, is_enabled, is_idle
    out: dict = {
        "scheduler_enabled": is_enabled(),
        "scheduler_idle": is_idle(),
        "jobs": get_job_snapshot(),
    }
    try:
        from app.dead_letter_inbound import backend_info as _dlq_info
        out["inbound_dlq"] = _dlq_info()
    except Exception:
        # DLQ module is optional from the dashboard's POV — never block
        # the snapshot if the import or backend probe fails.
        pass
    return out

# ── Governance ───────────────────────────────────────────────────────────────

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

# ── Org Chart ────────────────────────────────────────────────────────────────

@router.get("/org-chart")
def get_org_chart_api():
    from app.control_plane.org_chart import get_org_chart
    return get_org_chart()


# ── Delegation-mode toggles (shown on Org Chart page) ────────────────────────
# When ON for a crew, tasks go to Coordinator + specialists instead of a
# single monolithic agent.  See app/crews/delegation_settings.py.

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


# ── Meta-agent toggles (shown on Org Chart page) ─────────────────────────────
# When ON for a crew, run_single_agent_crew routes through the meta-agent
# selector — picks a learned recipe (force_tier × extra_tools × task_hint)
# from cross-run history, applies it as a bounded augmentation, records the
# outcome. Orthogonal to delegation: meta-agent fires only on the
# single-agent dispatch path.  See app/self_improvement/meta_agent/.
#
# Resolution order (see feature_flag.is_meta_agent_enabled): per-crew
# env override → master env → persisted JSON (this endpoint) → OFF.

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


# ── System Health (aggregated from existing systems) ─────────────────────────

@router.get("/health")
def control_plane_health():
    """Aggregated system health for dashboard."""
    from app.control_plane.db import execute_scalar
    from app.control_plane.governance import get_governance
    ticket_count = execute_scalar("SELECT COUNT(*) FROM control_plane.tickets") or 0
    audit_count = execute_scalar("SELECT COUNT(*) FROM control_plane.audit_log") or 0
    pending = get_governance().pending_count()
    return {
        "status": "ok",
        "tickets_total": ticket_count,
        "audit_entries": audit_count,
        "governance_pending": pending,
    }

# ── Costs ────────────────────────────────────────────────────────────────────

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


# Crew-name → agent-role mapping.  Matches the argument pairs passed to
# run_single_agent_crew in each crew module (e.g. coding_crew.py passes
# crew_name="coding", agent_role="coder").  Used to derive per-agent
# costs from the long-lived SQLite request_costs history so the Cost
# by Agent chart has real data from day one instead of waiting for the
# budgets reconcile to accumulate.
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


# Canonical crew roster — defined here (early) so the
# ``_INTERNAL_AGENT_NAMES`` derivation below can reference it without a
# forward-reference NameError.  The "kind" tag distinguishes
# user-addressable crews (natural-language dispatch) from internal
# crews (orchestration, quality review, reflection, self-learning).
# This is the single source of truth used to backfill the ``/tasks``
# response so every crew stays visible in the dashboard even when
# Firestore hasn't seen it yet.
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

# Classify actor names by CREW_REGISTRY kind without making the frontend
# the source of truth.  Some rows come from tracker.crew_name
# (e.g. "research+coding", "tom_research") — treat anything that isn't
# explicitly internal as user-level so ad-hoc compound names surface in
# "Cost by Crew" rather than the internal bucket.  Evaluated lazily so
# this helper stays valid even though _KNOWN_CREWS is defined later in
# this module.
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


# ── Operations: errors, anomalies, self-deploy pipeline ──────────────────────

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

# ── Tech radar (scoped operational memory) ───────────────────────────────────

@router.get("/tech-radar")
def tech_radar(limit: int = Query(20, ge=1, le=100)):
    """Technology-discovery items collected during idle scans.

    Mirrors the parsing the Firestore publisher performs: memory items stored
    in ``scope_tech_radar`` follow ``[category] title: summary. Action: ...``.
    """
    import re as _re
    discoveries: list[dict] = []
    err: str | None = None
    try:
        from app.memory.scoped_memory import retrieve_operational
        items = retrieve_operational("scope_tech_radar", "technology discovery", n=limit) or []
        for item in items:
            text = item if isinstance(item, str) else str(item)
            m = _re.match(r'\[(\w+)\]\s*(.+?):\s*(.+?)(?:\.\s*Action:\s*(.+))?$', text, _re.DOTALL)
            if m:
                discoveries.append({
                    "category": m.group(1),
                    "title": m.group(2).strip(),
                    "summary": m.group(3).strip(),
                    "action": (m.group(4) or "").strip(),
                })
            else:
                discoveries.append({
                    "category": "unknown",
                    "title": text[:80],
                    "summary": text[:200],
                    "action": "",
                })
    except Exception as exc:
        err = str(exc)
        logger.debug("tech-radar endpoint: %s", exc)

    # Search backend health — lets the React app explain *why* the radar may
    # be quiet (e.g. "Brave quota exhausted, falling back to SearXNG").
    search_status: dict = {}
    try:
        from app.tools.web_search import get_search_status
        search_status = get_search_status()
    except Exception as exc:
        logger.debug("tech-radar endpoint: search status failed: %s", exc)

    return {
        "discoveries": discoveries,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "error": err,
        "search_status": search_status,
    }

# ── LLM catalog, role assignments, discovery control ─────────────────────────

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


# ── Promotions (layer 2 of the resolver's authority cake) ────────────

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


# ── Hand pins (layer 3 — hard override) ───────────────────────────────

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

# ── Cross-eval judges (rotation + pins + agreement telemetry) ────────────────

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


# ── Crew tasks (live execution + roster) ─────────────────────────────────────

# (``_KNOWN_CREWS`` is defined higher up in the module so
# ``_INTERNAL_AGENT_NAMES`` can derive from it without a forward-ref
# NameError.)

@router.get("/tasks")
def get_crew_tasks(limit: int = Query(20, ge=1, le=200), project_id: str | None = Query(None)):
    """Return recent crew tasks + crew statuses + full agent roster.

    Reads the Control Plane `crew_tasks` table (Postgres).  The legacy
    Firestore `tasks` / `crews` collections are still mirrored by
    `app.firebase.crew_tracking` for backwards-observability but are no
    longer on the dashboard read path — no more 429 quota banners.

    - `tasks`  — last N tasks from crew_tasks (running or completed)
    - `crews`  — per-crew latest status derived from crew_tasks, merged
                 with the canonical registry so idle crews still show up
    - `agents` — the PostgreSQL org-chart roster so every agent /
                 subagent is represented even when idle
    """
    err: str | None = None
    tasks: list[dict] = []
    crews: list[dict] = []

    try:
        from app.control_plane.crew_tasks import list_recent, crew_statuses
        tasks = list_recent(limit=limit, project_id=project_id)
        crews = crew_statuses()
    except Exception as exc:
        err = f"tasks read: {exc}"
        logger.debug("tasks endpoint: %s", err)

    # Ensure every known crew appears in the list even if no tasks yet.
    # Each crew carries a "kind" tag so the dashboard can group user-
    # addressable crews separately from internal orchestration crews.
    known_kinds = {name: kind for name, kind in _KNOWN_CREWS}
    seen_crews = {c.get("name") for c in crews}
    for c in crews:
        name = c.get("name")
        if name in known_kinds and "kind" not in c:
            c["kind"] = known_kinds[name]
    for name, kind in _KNOWN_CREWS:
        if name not in seen_crews:
            crews.append({"name": name, "state": "idle", "kind": kind})

    agents: list[dict] = []
    try:
        from app.control_plane.org_chart import get_org_chart
        agents = get_org_chart() or []
    except Exception as exc:
        logger.debug("tasks endpoint: org_chart read failed: %s", exc)

    return {
        "tasks": tasks,
        "crews": crews,
        "agents": agents,
        "project_id": project_id,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "error": err,
    }


@router.get("/tasks/{task_id}/timeline")
def get_task_timeline(task_id: str):
    """Return the fine-grained execution flow of a single crew task.

    Response shape:
        {
          "task": { ...crew_tasks row... },
          "spans": [                          # nested tree
            {
              "id": 42,
              "span_type": "agent",
              "name": "Researcher",
              "started_at": "...",
              "completed_at": "...",
              "state": "running|completed|failed",
              "detail": {...},
              "children": [ ...spans... ]
            },
            ...
          ],
          "updated_at": "..."
        }

    Spans are populated by ``app.crews.span_events`` as CrewAI emits
    agent/tool/llm-call lifecycle events. Call this endpoint repeatedly
    (poll at 2s) while ``task.state == 'running'`` to watch the flow
    build; stop when state transitions to ``completed`` / ``failed``.
    """
    from app.control_plane.db import execute_one
    from app.control_plane.crew_task_spans import list_spans

    task: dict | None = None
    try:
        task = execute_one(
            """
            SELECT id, crew, project_id, state, summary, result_preview,
                   error, model, tokens_used, cost_usd,
                   parent_task_id, is_sub_agent,
                   delegated_from, delegated_to, delegation_reason,
                   started_at, completed_at, last_updated
              FROM control_plane.crew_tasks
             WHERE id = %s
            """,
            (task_id,),
        )
    except Exception as exc:
        logger.debug("tasks/timeline: crew_tasks lookup failed: %s", exc)

    if not task:
        raise HTTPException(status_code=404, detail=f"task {task_id!r} not found")

    spans_flat = list_spans(task_id)

    # Build the tree in-Python: group by parent_span_id, attach children
    # in started_at order. Root spans have parent_span_id IS NULL.
    by_id: dict[int, dict] = {}
    for s in spans_flat:
        s["children"] = []
        by_id[s["id"]] = s
    roots: list[dict] = []
    for s in spans_flat:
        parent_id = s.get("parent_span_id")
        if parent_id and parent_id in by_id:
            by_id[parent_id]["children"].append(s)
        else:
            roots.append(s)

    return {
        "task": task,
        "spans": roots,
        "span_count": len(spans_flat),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


# ── Token usage & cost projection ────────────────────────────────────────────

_TOKEN_PERIODS = ("hour", "day", "week", "month", "year")

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

# ── Consciousness Indicators (Garland / Butlin-Chalmers) ─────────────────────

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


# ── Proposal actions (evolution approve/reject) ─────────────────────
#
# Replaces the legacy ``report_proposal_actions`` Firestore polling
# loop.  Any client (React dashboard, CLI, Signal bot, external script)
# that wants to act on a pending evolution proposal POSTs here; the
# gateway applies the action synchronously via the ``app.proposals``
# module (same functions the Firestore poller used to call).

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


# ── Observability snapshots ─────────────────────────────────────────
#
# These endpoints expose the Postgres-backed observability snapshot
# store (see ``app/observability/snapshots.py``) so the React dashboard
# can read them without going through Firestore.  Part of the
# Firebase-dependency reduction: as publishers migrate to write
# snapshots, the corresponding dashboard pages switch to these
# endpoints and the Firestore collection for that concern goes unread.


@router.get("/observability/snapshots/{kind}/latest")
def observability_snapshot_latest(kind: str):
    """Return the most recent snapshot of the given ``kind``, or 404 if
    nothing has been recorded.  Shape: ``{ts, kind, payload}``."""
    from app.observability.snapshots import latest as _latest
    snap = _latest(kind)
    if snap is None:
        raise HTTPException(
            status_code=404,
            detail=f"No snapshot recorded for kind={kind!r}",
        )
    return {
        "ts": snap.ts.isoformat() if hasattr(snap.ts, "isoformat") else str(snap.ts),
        "kind": snap.kind,
        "payload": snap.payload,
    }


@router.get("/observability/snapshots/{kind}/recent")
def observability_snapshot_recent(
    kind: str,
    limit: int = Query(default=50, ge=1, le=500),
):
    """Return the most recent ``limit`` snapshots for ``kind``, newest
    first.  Empty list if nothing has been recorded for that kind."""
    from app.observability.snapshots import recent as _recent
    snaps = _recent(kind, limit=limit)
    return {
        "kind": kind,
        "count": len(snaps),
        "items": [
            {
                "ts": s.ts.isoformat() if hasattr(s.ts, "isoformat") else str(s.ts),
                "payload": s.payload,
            }
            for s in snaps
        ],
    }


@router.get("/observability/snapshots")
def observability_snapshot_kinds():
    """Return every distinct ``kind`` currently recorded + its newest
    timestamp.  Useful for dashboards to discover what's available
    without enumerating hardcoded kinds.
    """
    try:
        from app.control_plane.db import execute
        rows = execute(
            """
            SELECT kind, MAX(ts) AS latest_ts, COUNT(*) AS count
              FROM observability_snapshots
          GROUP BY kind
          ORDER BY MAX(ts) DESC
            """,
            fetch=True,
        )
    except Exception:
        rows = []
    return {
        "kinds": [
            {
                "kind": r.get("kind") if isinstance(r, dict) else r[0],
                "latest_ts": (
                    (r.get("latest_ts") if isinstance(r, dict) else r[1]).isoformat()
                    if (r.get("latest_ts") if isinstance(r, dict) else r[1])
                    else None
                ),
                "count": r.get("count") if isinstance(r, dict) else r[2],
            }
            for r in (rows or [])
        ],
    }


# ── Credit alerts (top-up requests) ──────────────────────────────────────
#
# When OpenRouter / Anthropic / OpenAI / Google return a credit-exhausted
# error, the rate_throttle wrapper writes an entry into firebase.publish's
# _active_alerts dict. The dashboard polls these endpoints to render a
# "Top up <provider>" card on the Budgets page.
#
# Why a manual dismiss endpoint when the wrapper auto-resolves on next
# success? Because the credit-failover transparently routes subsequent
# requests to local Ollama, so the failed provider may never be called
# again — and the auto-resolve path never fires. Manual dismiss clears
# the card after the user has topped up.

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


# ── Research adapters status ─────────────────────────────────────────────
#
# Lightweight status endpoint so the dashboard / Signal can answer
# "is Apollo activated?" without a round-trip through the orchestrator.
# Keys are read live from os.environ on every call so adding a key
# and recreating the gateway shows up immediately.

@router.get("/research/adapters")
def research_adapter_status():
    """Show which paid research adapters are configured.

    Response shape::

      {
        "adapters": {
          "apollo":        {"configured": false, "env_key": "APOLLO_API_KEY"},
          "linkedin_data": {"configured": false, "env_key": "PROXYCURL_API_KEY"}
        },
        "source_priority_now": ["regulator", "company_site", "search"],
        "source_priority_full": ["regulator", "company_site", "apollo",
                                 "linkedin_data", "search"]
      }
    """
    try:
        from app.tools.research_orchestrator import (
            get_paid_adapter_status, default_source_priority,
        )
    except Exception as e:
        return {"error": f"orchestrator import failed: {e}"}
    status = get_paid_adapter_status()
    return {
        "adapters": {
            "apollo": {
                "configured": status.get("apollo", False),
                "env_key": "APOLLO_API_KEY",
                "provides": ["head_of_sales", "head_of_sales_linkedin",
                             "head_of_sales_email"],
                "homepage": "https://apollo.io/settings/api",
            },
            "linkedin_data": {
                "configured": status.get("linkedin_data", False),
                "env_key": "PROXYCURL_API_KEY",
                "provides": ["head_of_sales", "head_of_sales_linkedin"],
                "homepage": "https://nubela.co/proxycurl/api-keys",
            },
        },
        "source_priority_now": default_source_priority(),
        "source_priority_full": [
            "regulator", "company_site", "apollo", "linkedin_data", "search",
        ],
    }


# ── Transfer Insight Layer (Phase 17d) ───────────────────────────────────────

@router.get("/transfer-memory/overview")
def transfer_memory_overview():
    """Top-line metrics for the Transfer Insight Layer."""
    from app.transfer_memory.dashboard import get_overview
    return get_overview()


@router.get("/transfer-memory/by-source-kind")
def transfer_memory_by_source_kind():
    """Per-source-kind compile + outcome counters."""
    from app.transfer_memory.dashboard import get_by_source_kind
    return get_by_source_kind()


@router.get("/transfer-memory/recent")
def transfer_memory_recent(days: int = Query(7, ge=1, le=90)):
    """Compile + retrieval activity over the trailing N days."""
    from app.transfer_memory.dashboard import get_recent_activity
    return get_recent_activity(days=days)


@router.get("/transfer-memory/top-performers")
def transfer_memory_top(n: int = Query(10, ge=1, le=100)):
    """Most-surfaced records with no negative-transfer entries."""
    from app.transfer_memory.dashboard import get_top_performers
    return get_top_performers(n=n)


@router.get("/transfer-memory/worst-performers")
def transfer_memory_worst(n: int = Query(10, ge=1, le=100)):
    """Records with the most negative-transfer entries."""
    from app.transfer_memory.dashboard import get_worst_performers
    return get_worst_performers(n=n)


@router.get("/transfer-memory/sanitizer-stats")
def transfer_memory_sanitizer_stats():
    """Hard-reject + demotion totals across all compiled rows."""
    from app.transfer_memory.dashboard import get_sanitizer_stats
    return get_sanitizer_stats()


@router.get("/transfer-memory/promotion-candidates")
def transfer_memory_promotion_candidates():
    """Eligible records waiting for promotion (operator review)."""
    from app.transfer_memory.dashboard import get_promotion_candidates
    return get_promotion_candidates()


@router.get("/transfer-memory/negative-transfer")
def transfer_memory_negative_transfer():
    """Tag distribution and recent negative-transfer events."""
    from app.transfer_memory.dashboard import get_negative_transfer_stats
    return get_negative_transfer_stats()


@router.get("/transfer-memory/source-target-matrix")
def transfer_memory_source_target_matrix():
    """Source-domain × target-domain co-occurrence from shadow retrievals."""
    from app.transfer_memory.dashboard import get_source_to_target_matrix
    return get_source_to_target_matrix()


@router.post("/transfer-memory/promote/{record_id}")
def transfer_memory_promote(record_id: str):
    """Operator-driven manual promotion of a single shadow record.

    Bypasses the cadence guard but still requires the record to pass
    eligibility checks (age, surface count, no negative attribution).
    """
    from app.transfer_memory.promotion import manual_promote
    ok = manual_promote(record_id)
    if not ok:
        raise HTTPException(409, "Promotion rejected — record ineligible or update failed")
    return {"promoted": True, "skill_record_id": record_id}


# ── Chat surface (mirror of the Signal conversation) ────────────────────────
#
# Two endpoints back the React /cp/chat tab:
#   GET  /api/cp/chat/messages?sender=...&limit=50  — recent history (oldest→newest)
#   POST /api/cp/chat/send   {sender, message}      — dispatch through Commander
# Plus a static catalogue:
#   GET  /api/cp/signal-commands                    — every slash / NL command


class ChatSendBody(BaseModel):
    sender: str = "andrus"
    message: str


_PRIMARY_USER_ALIASES = frozenset({"andrus", "owner", "primary", "me", "user"})


def _resolve_chat_sender(raw: str | None) -> str:
    """Resolve the chat-tab ``sender`` query value to the wire-level id
    Signal writes under so the React Chat tab and the Signal channel
    share one conversation thread.

    Signal messages are keyed on ``settings.signal_owner_number`` (a
    phone number); the React tab passes ``"andrus"`` for ergonomics.
    Without this translation the dashboard would read an empty bucket
    keyed on ``HMAC("andrus")`` while the real history sat under
    ``HMAC("+372...")``.
    """
    s = (raw or "").strip()
    if not s or s.lower() in _PRIMARY_USER_ALIASES:
        try:
            from app.config import get_settings
            owner = (get_settings().signal_owner_number or "").strip()
            if owner:
                return owner
        except Exception:
            pass
        return "andrus"
    return s


@router.get("/chat/messages")
def chat_messages(sender: str = Query("andrus"), limit: int = Query(50, ge=1, le=500)):
    """Recent conversation history with the given sender.

    Returns oldest-first so the React UI can append-render straight to a
    scroll area without reversing. Each entry: ``{role, content, ts}``.
    The ``sender`` query is resolved to the configured primary-user
    phone number when it equals "andrus" / "owner" / "me" so the React
    Chat tab sees the same thread Signal writes to.
    """
    resolved = _resolve_chat_sender(sender)
    try:
        from app.conversation_store import get_recent_messages
        rows = get_recent_messages(resolved, limit=limit)
    except Exception as exc:
        logger.debug("chat_messages: read failed: %s", exc)
        return {"sender": sender, "messages": [], "error": str(exc)}
    rows.reverse()  # store returns DESC; chat wants ASC for append
    return {"sender": sender, "messages": rows}


@router.post("/chat/send")
def chat_send(body: ChatSendBody):
    """Send a chat message — same dispatch path as a Signal message.

    Routes through ``Commander.handle()`` so every existing slash
    command, recovery loop, project router, and lifecycle hook fires
    exactly as if the message had arrived from Signal. Sender defaults
    to ``andrus`` and is resolved to ``settings.signal_owner_number``
    so the conversation thread is shared with Signal.
    """
    text = (body.message or "").strip()
    if not text:
        raise HTTPException(400, "empty message")
    sender = _resolve_chat_sender(body.sender)
    try:
        from app.agents.commander import Commander
    except Exception as exc:
        raise HTTPException(503, f"commander unavailable: {exc}")
    try:
        reply = Commander().handle(text, sender=sender) or ""
    except Exception as exc:
        logger.warning("chat_send: commander dispatch failed: %s", exc, exc_info=True)
        raise HTTPException(500, f"dispatch failed: {exc}")
    return {"sender": sender, "message": text, "reply": str(reply)}


# ── System status — comprehensive monitoring pane ──────────────────────────
#
# Single aggregator endpoint behind /cp/monitor. Probes each subsystem with
# a short timeout, returns a flat list of status rows the React page groups
# by category. Credit-exhaustion errors are interpreted via
# firebase.publish.detect_credit_error and surfaced with their top-up link.


def _probe(name: str, category: str, fn, *, link: str | None = None) -> dict:
    """Run a probe with a soft deadline + uniform error surface."""
    import time as _t
    t0 = _t.monotonic()
    try:
        result = fn()
        latency_ms = int((_t.monotonic() - t0) * 1000)
        if isinstance(result, dict):
            result.setdefault("name", name)
            result.setdefault("category", category)
            result.setdefault("latency_ms", latency_ms)
            if link and "link" not in result:
                result["link"] = link
            return result
        return {
            "name": name, "category": category, "status": "ok",
            "message": str(result) if result else "responding",
            "latency_ms": latency_ms, "link": link,
        }
    except Exception as exc:
        return {
            "name": name, "category": category, "status": "error",
            "message": _interpret_error(exc),
            "link": _credit_link_for(exc) or link,
            "latency_ms": int((_t.monotonic() - t0) * 1000),
        }


_KNOWN_ERROR_HINTS: tuple[tuple[str, str], ...] = (
    ("connection refused", "Service is down or not listening"),
    ("timeout",            "Service did not respond in time"),
    ("name or service",    "DNS / hostname unresolved"),
    ("no route to host",   "Network unreachable"),
    ("unauthorized",       "Auth failed — check API key / credentials"),
    ("authentication",     "Auth failed — check API key / credentials"),
    ("permission denied",  "Auth / permission rejected"),
    ("forbidden",          "Forbidden — check API key / scopes"),
    ("insufficient",       "Out of credits — top up to continue"),
    ("payment required",   "Out of credits — top up to continue"),
    ("rate_limit",         "Provider rate-limited the request"),
    ("quota",              "Quota exhausted"),
    ("402",                "Out of credits — top up to continue"),
    ("401",                "Auth failed — check API key"),
    ("403",                "Forbidden — check API key / scopes"),
    ("404",                "Endpoint not found"),
    ("429",                "Rate-limited"),
    ("500",                "Upstream server error"),
    ("502",                "Upstream gateway error"),
    ("503",                "Upstream service unavailable"),
    ("504",                "Upstream gateway timeout"),
)


def _interpret_error(exc: Exception | str) -> str:
    """Turn raw exceptions into a short human-readable hint."""
    raw = str(exc).strip()
    low = raw.lower()
    for needle, hint in _KNOWN_ERROR_HINTS:
        if needle in low:
            return f"{hint} — {raw[:160]}"
    return raw[:200] or type(exc).__name__ if isinstance(exc, Exception) else raw[:200]


def _credit_link_for(exc: Exception | str) -> str | None:
    """If the exception looks like credit exhaustion, return the
    provider's top-up URL so the dashboard can render a 'Top up' button."""
    try:
        from app.firebase.publish import detect_credit_error, _CREDIT_URLS
        provider = detect_credit_error(exc)
        if provider:
            return _CREDIT_URLS.get(provider)
    except Exception:
        pass
    return None


@router.get("/system-status")
def system_status():
    """Aggregated monitoring view used by the React /cp/monitor page.

    Each check is a row with ``status ∈ {ok, warn, error}``, a
    one-line ``message``, an optional ``link`` (used for credit-
    top-up CTAs), and a measured ``latency_ms``. Probes have a soft
    timeout and never raise — failures land as ``error`` rows.
    """
    checks: list[dict] = []

    # ── Containers ─────────────────────────────────────────────
    def _pg():
        from app.control_plane.db import execute_scalar
        n = execute_scalar("SELECT COUNT(*) FROM control_plane.budgets")
        return {"status": "ok", "message": f"connected · {n} budget rows"}
    checks.append(_probe("PostgreSQL (control plane)", "Containers", _pg))

    def _chroma():
        from app.memory.chromadb_manager import get_client
        client = get_client()
        cols = client.list_collections() if client else []
        return {"status": "ok", "message": f"connected · {len(cols)} collections"}
    checks.append(_probe("ChromaDB", "Containers", _chroma))

    def _neo4j():
        from app.subia.belief import neo4j_mirror
        drv = neo4j_mirror._get_driver()
        if drv is None:
            return {"status": "warn", "message": "driver not configured (NEO4J_URL unset?)"}
        with drv.session() as s:
            s.run("RETURN 1").consume()
        return {"status": "ok", "message": "connected"}
    checks.append(_probe("Neo4j", "Containers", _neo4j))

    def _gateway():
        return {"status": "ok", "message": "responding (you're reading me)"}
    checks.append(_probe("Gateway HTTP", "Containers", _gateway))

    # ── Gateways / messaging ───────────────────────────────────
    def _signal():
        """Probe vanilla signal-cli's --http JSON-RPC endpoint.

        Pre-2026-05-10 fix this hit GET /v1/about, which is the path
        used by the *separate* signal-cli-rest-api Docker wrapper —
        not vanilla signal-cli.  Vanilla signal-cli's --http mode is
        JSON-RPC at POST /api/v1/rpc.  The 404 from /v1/about
        triggered a false "Endpoint not found" ERROR in the System
        Monitor for any deployment using upstream signal-cli (the
        host-native dev path).
        """
        import json as _json
        import urllib.request as _u
        from app.config import get_settings
        s = get_settings()
        url = (s.signal_http_url or "").rstrip("/")
        if not url:
            return {"status": "warn", "message": "signal_http_url not configured"}
        body = _json.dumps({
            "jsonrpc": "2.0", "method": "version", "id": 1,
        }).encode("utf-8")
        req = _u.Request(
            f"{url}/api/v1/rpc",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with _u.urlopen(req, timeout=2) as r:
            payload = _json.loads(r.read().decode("utf-8"))
        version = (payload.get("result") or {}).get("version") or "?"
        if "error" in payload:
            return {
                "status": "error",
                "message": f"rpc error: {payload['error']}",
            }
        return {
            "status": "ok",
            "message": f"daemon responding ({s.signal_owner_number} · v{version})",
        }
    checks.append(_probe("Signal-cli daemon", "Messaging", _signal))

    def _bridge():
        """Probe the host bridge by exercising the same code path the
        rest of the system uses to decide if it's wired.

        Pre-2026-05-10 fix this gated on the legacy ``bridge_enabled``
        setting (env ``BRIDGE_ENABLED=1``).  Since §27.3 the actual
        wiring uses per-agent ``BRIDGE_TOKEN_<AGENT>`` tokens via
        ``app.bridge_client.get_bridge(...)`` — ``BRIDGE_ENABLED`` is
        no longer the source of truth.  The probe was reporting
        "host bridge disabled" while the bridge was fully functional.
        """
        # Try the modern per-agent path first.  ``change_requests`` is
        # the canonical agent token used by the React /cp/changes
        # surface; if it's wired, the bridge is operational.
        try:
            from app.bridge_client import get_bridge
            client = get_bridge("change_requests")
            if client and client.is_available():
                return {
                    "status": "ok",
                    "message": "responding via per-agent token (change_requests)",
                }
        except Exception:
            pass

        # Fall back to the legacy enable check + raw /health probe so
        # laptop dev setups with BRIDGE_ENABLED=1 + BRIDGE_SHARED_SECRET
        # still surface a sensible status.
        import urllib.request as _u
        from app.config import get_settings
        s = get_settings()
        if not s.bridge_enabled:
            return {
                "status": "warn",
                "message": (
                    "host bridge: no per-agent BRIDGE_TOKEN_* tokens "
                    "configured AND BRIDGE_ENABLED=0"
                ),
            }
        url = f"http://{s.bridge_host}:{s.bridge_port}/health"
        with _u.urlopen(url, timeout=2) as r:
            ok = r.status == 200
        return {
            "status": "ok" if ok else "error",
            "message": f"port {s.bridge_port}: http {r.status}",
        }
    checks.append(_probe("Host bridge", "Messaging", _bridge))

    # ── Internal subsystems ────────────────────────────────────
    def _idle():
        from app.idle_scheduler import is_enabled, _currently_running_job
        running = _currently_running_job
        if not is_enabled():
            return {"status": "warn", "message": "background tasks OFF (Settings → Background tasks)"}
        msg = f"running · current: {running}" if running else "running · idle"
        return {"status": "ok", "message": msg}
    checks.append(_probe("Idle scheduler", "Internal", _idle))

    def _self_heal():
        """Self-heal journal status — count entries within a real time
        window, not the whole FIFO buffer.

        Pre-2026-05-10 fix this called ``get_recent_errors(50)`` which
        returns the last 50 entries regardless of timestamp.  So any
        install with historical errors (most have weeks-old residue
        from before fixes landed) was permanently in WARN — even when
        zero errors had been recorded in the last 24 h.  The "50
        recent errors, top: coding:BadRequestError×16" alert was
        technically true but misleading: those 50 entries spanned
        2026-04-02 → 2026-04-28 with NO entries from the past 24 h.

        Now: filter by ``ts`` against a 24 h window.  Only entries
        actually fired recently count toward the WARN status.  When
        the journal has historical residue but no fresh entries, the
        operator sees ``"no errors in last 24h (N historical in
        journal)"`` instead of a phantom WARN.
        """
        from datetime import datetime, timedelta, timezone

        from app.healing.error_diagnosis import get_recent_errors
        # Pull a generous slice so the time-window cut isn't truncated
        # by FIFO ordering on a busy day.
        all_recent = list(get_recent_errors(200) or [])
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)

        def _ts_in_window(entry: dict) -> bool:
            ts_str = entry.get("ts") or ""
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                return False
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return ts >= cutoff

        recent_24h = [e for e in all_recent if _ts_in_window(e)]
        if not recent_24h:
            total = len(all_recent)
            if total > 0:
                msg = f"no errors in last 24h ({total} historical in journal)"
            else:
                msg = "no recent errors"
            return {"status": "ok", "message": msg}

        # Compute top pattern over the WINDOW, not all-time, so the
        # operator sees what's actually firing now.
        windowed_patterns: dict[str, int] = {}
        for e in recent_24h:
            key = f"{e.get('crew', '?')}:{e.get('error_type', '?')}"
            windowed_patterns[key] = windowed_patterns.get(key, 0) + 1
        top = sorted(
            windowed_patterns.items(), key=lambda kv: -kv[1],
        )[:1]
        top_str = f", top: {top[0][0]}×{top[0][1]}" if top else ""
        return {
            "status": "warn",
            "message": f"{len(recent_24h)} errors in last 24h{top_str}",
        }
    checks.append(_probe("Self-heal journal", "Internal", _self_heal))

    def _budget_reconcile():
        from app.control_plane.db import execute_one
        row = execute_one(
            "SELECT MAX(updated_at) AS last_ts FROM control_plane.budgets WHERE spent_usd > 0"
        )
        if not row or not row.get("last_ts"):
            return {"status": "warn", "message": "no recent reconcile activity"}
        return {"status": "ok", "message": f"last write {row['last_ts']}"}
    checks.append(_probe("Budget reconcile", "Internal", _budget_reconcile))

    # ── External services / credit alerts ──────────────────────
    try:
        from app.firebase.publish import _active_alerts, _CREDIT_URLS
        for provider, alert in _active_alerts.items():
            checks.append({
                "name": f"{provider.capitalize()} credit",
                "category": "External services",
                "status": "error",
                "message": (alert.get("error") or "credit exhausted")[:200],
                "link": alert.get("url") or _CREDIT_URLS.get(provider),
                "since": alert.get("ts"),
                "latency_ms": 0,
            })
        if not _active_alerts:
            checks.append({
                "name": "LLM provider credits",
                "category": "External services",
                "status": "ok",
                "message": "no active credit alerts",
                "link": None,
                "latency_ms": 0,
            })
    except Exception as exc:
        checks.append({
            "name": "Credit alerts feed",
            "category": "External services",
            "status": "warn",
            "message": _interpret_error(exc),
            "link": None,
            "latency_ms": 0,
        })

    # ── Roll-up summary (counts + worst status) ────────────────
    by_cat: dict[str, dict[str, int]] = {}
    worst = "ok"
    rank = {"ok": 0, "warn": 1, "error": 2}
    for c in checks:
        cat = c.get("category", "Other")
        st = c.get("status", "ok")
        bucket = by_cat.setdefault(cat, {"ok": 0, "warn": 0, "error": 0})
        bucket[st] = bucket.get(st, 0) + 1
        if rank.get(st, 0) > rank.get(worst, 0):
            worst = st

    return {
        "checks": checks,
        "by_category": by_cat,
        "overall": worst,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/signal-commands")
def signal_commands():
    """Hand-curated catalogue of every Signal slash / NL command.

    Source of truth: ``app/agents/commander/command_registry.py``.
    Used by the React /cp/chat sidebar so the user can scan the full
    surface without scrolling through ``commands.py`` source.
    """
    from app.agents.commander.command_registry import to_payload, categories
    return {
        "categories": categories(),
        "commands": to_payload(),
    }
