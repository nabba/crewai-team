"""Control Plane dashboard API routes.

Provides REST endpoints for the React dashboard and SSE for real-time updates.
All routes prefixed with /api/cp/.
"""
import json
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/cp", tags=["control-plane"])


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
    return {"status": "updated"}

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

@router.get("/costs/by-agent")
def costs_by_agent(project_id: str = Query(None)):
    from app.control_plane.audit import get_audit
    return get_audit().cost_summary(project_id)

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
