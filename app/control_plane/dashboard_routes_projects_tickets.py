"""Control-plane dashboard routes — projects_tickets topic.

Projects + Tickets routes — kanban board, project metadata, ticket lifecycle.

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


