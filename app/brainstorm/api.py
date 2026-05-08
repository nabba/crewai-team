"""HTTP API for the brainstorm subsystem.

All routes are prefixed ``/api/cp/brainstorm`` and inherit the gateway-auth
dependency. They wrap the same facilitator the Signal handler and CLI use,
so any session is interchangeable across surfaces.

Senders
-------
The web surface defaults to a single "owner" sender so React sessions share
the store with Signal sessions of the same user. Override the default via
the ``BRAINSTORM_WEB_SENDER`` env var. Callers may also pass an explicit
``sender`` query parameter on every endpoint.

Long-running endpoints
----------------------
``POST /sessions``, ``/respond``, ``/skip`` and ``/finish`` may take tens of
seconds (multi-agent gathering + Writer-agent report). Block synchronously
for now — promote to background tasks if interactive UX needs it.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.brainstorm import facilitator, store
from app.brainstorm.facilitator import FacilitatorError, StepDelivery
from app.brainstorm.multi_agent import DEFAULT_ROSTER, AgentResponse
from app.brainstorm.session import BrainstormSession
from app.brainstorm.techniques import get as get_technique
from app.brainstorm.techniques import registry as technique_registry
from app.control_plane.auth_dep import require_gateway_auth

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/cp/brainstorm",
    tags=["brainstorm"],
    dependencies=[Depends(require_gateway_auth)],
)


# ── Sender resolution ────────────────────────────────────────────────────


def _default_sender() -> str:
    """Default sender for web requests — shared with Signal owner if configured.

    Resolution order:
      1. ``BRAINSTORM_WEB_SENDER`` env var (explicit override)
      2. settings.signal_owner_number (so Signal + web share sessions)
      3. ``"web:default"`` fallback
    """
    explicit = os.environ.get("BRAINSTORM_WEB_SENDER")
    if explicit:
        return explicit
    try:
        from app.config import get_settings
        owner = getattr(get_settings(), "signal_owner_number", None) or ""
        if owner:
            return owner
    except Exception:
        pass
    return "web:default"


def _resolve_sender(sender: str | None) -> str:
    resolved = (sender or "").strip() or _default_sender()
    # Make sure the affect attachment hook can see who's interacting —
    # brainstorm sessions are user-initiated and should bump last_seen.
    try:
        from app.project_context import set_current_sender_id
        set_current_sender_id(resolved)
    except Exception:
        pass
    return resolved


# ── Request / response models ────────────────────────────────────────────


class TechniqueInfo(BaseModel):
    name: str
    title: str
    description: str
    total_steps: int


class SessionCreate(BaseModel):
    technique: str
    topic: str
    with_agents: int = Field(default=0, ge=0, le=len(DEFAULT_ROSTER))


class MessageBody(BaseModel):
    message: str = ""


class AgentResponsePayload(BaseModel):
    role: str
    text: str
    duration_s: float = 0.0
    error: Optional[str] = None

    @classmethod
    def from_dataclass(cls, r: AgentResponse) -> "AgentResponsePayload":
        return cls(role=r.role, text=r.text, duration_s=r.duration_s, error=r.error)


class StepDeliveryPayload(BaseModel):
    prompt: Optional[str] = None
    seed: list[AgentResponsePayload] = []
    react: list[AgentResponsePayload] = []

    @classmethod
    def from_delivery(cls, d: StepDelivery) -> "StepDeliveryPayload":
        return cls(
            prompt=d.prompt,
            seed=[AgentResponsePayload.from_dataclass(r) for r in d.seed],
            react=[AgentResponsePayload.from_dataclass(r) for r in d.react],
        )


class SessionResponse(BaseModel):
    session: dict[str, Any]
    delivery: Optional[StepDeliveryPayload] = None
    advanced: Optional[bool] = None


def _serialize_session(s: BrainstormSession) -> dict[str, Any]:
    """Public-facing session shape for the dashboard."""
    technique = get_technique(s.technique)
    total_steps = technique.total_steps() if technique else None
    return {
        "session_id": s.session_id,
        "sender": s.sender,
        "topic": s.topic,
        "technique": s.technique,
        "technique_title": technique.title if technique else s.technique,
        "technique_description": technique.description if technique else "",
        "step_index": s.technique_state.step_index,
        "total_steps": total_steps,
        "is_complete_state_machine": (
            technique.is_complete(s.technique_state) if technique else False
        ),
        "status": s.status,
        "mode": s.mode,
        "participants": list(s.participants),
        "transcript": list(s.transcript),
        "agent_rounds": list(s.agent_rounds),
        "responses": list(s.technique_state.responses),
        "created_at": s.created_at,
        "updated_at": s.updated_at,
        "final_report_path": s.final_report_path,
        "final_report": s.final_report,
    }


# ── Read endpoints ────────────────────────────────────────────────────────


@router.get("/techniques")
def list_techniques() -> list[TechniqueInfo]:
    out = []
    for name, t in technique_registry().items():
        out.append(
            TechniqueInfo(
                name=name,
                title=t.title,
                description=t.description,
                total_steps=t.total_steps() or 0,
            )
        )
    out.sort(key=lambda i: i.name)
    return out


@router.get("/sessions")
def list_sessions(
    sender: str | None = Query(None),
    include_other_senders: bool = Query(False),
) -> list[dict[str, Any]]:
    """List sessions for ``sender`` (default = web sender), newest first.

    Set ``include_other_senders=true`` to see all sessions in the store
    regardless of sender (useful for an admin view of CLI + Signal sessions).
    """
    if include_other_senders:
        sessions = store.list_sessions(sender=None)
    else:
        sessions = store.list_sessions(sender=_resolve_sender(sender))
    return [_serialize_session(s) for s in sessions]


@router.get("/sessions/active")
def get_active_session(sender: str | None = Query(None)) -> dict[str, Any]:
    """Return the active session for the sender, or ``{"session": null}``."""
    s = store.get_active(_resolve_sender(sender))
    return {"session": _serialize_session(s) if s else None}


@router.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    s = store.load(session_id)
    if s is None:
        raise HTTPException(404, f"Session {session_id} not found")
    return _serialize_session(s)


# ── Write endpoints ───────────────────────────────────────────────────────


@router.post("/sessions")
def start_session(
    body: SessionCreate,
    sender: str | None = Query(None),
) -> SessionResponse:
    n = body.with_agents if body.with_agents > 0 else None
    try:
        session, delivery = facilitator.start(
            _resolve_sender(sender),
            body.technique,
            body.topic,
            with_agents=n,
        )
    except FacilitatorError as exc:
        raise HTTPException(400, str(exc)) from exc
    return SessionResponse(
        session=_serialize_session(session),
        delivery=StepDeliveryPayload.from_delivery(delivery),
    )


@router.post("/sessions/{session_id}/respond")
def respond(
    session_id: str,
    body: MessageBody = Body(...),
    sender: str | None = Query(None),
) -> SessionResponse:
    """Record a user response. Path session_id is required for safety;
    we 409 if it doesn't match the sender's currently-active session.
    """
    resolved = _resolve_sender(sender)
    active = store.get_active(resolved)
    if active is None:
        raise HTTPException(409, "No active brainstorm session for this sender")
    if active.session_id != session_id:
        raise HTTPException(
            409,
            f"Session {session_id} is not the active session "
            f"(active: {active.session_id}). Resume it first.",
        )
    try:
        session, delivery, advanced = facilitator.respond(resolved, body.message)
    except FacilitatorError as exc:
        raise HTTPException(400, str(exc)) from exc
    return SessionResponse(
        session=_serialize_session(session),
        delivery=StepDeliveryPayload.from_delivery(delivery),
        advanced=advanced,
    )


@router.post("/sessions/{session_id}/skip")
def skip_step(
    session_id: str,
    sender: str | None = Query(None),
) -> SessionResponse:
    resolved = _resolve_sender(sender)
    active = store.get_active(resolved)
    if active is None or active.session_id != session_id:
        raise HTTPException(409, "Session is not active for this sender")
    try:
        session, delivery = facilitator.skip(resolved)
    except FacilitatorError as exc:
        raise HTTPException(400, str(exc)) from exc
    return SessionResponse(
        session=_serialize_session(session),
        delivery=StepDeliveryPayload.from_delivery(delivery),
    )


@router.post("/sessions/{session_id}/pause")
def pause_session(
    session_id: str, sender: str | None = Query(None)
) -> dict[str, Any]:
    resolved = _resolve_sender(sender)
    active = store.get_active(resolved)
    if active is None or active.session_id != session_id:
        raise HTTPException(409, "Session is not active for this sender")
    try:
        s = facilitator.pause(resolved)
    except FacilitatorError as exc:
        raise HTTPException(400, str(exc)) from exc
    return _serialize_session(s)


@router.post("/sessions/{session_id}/resume")
def resume_session(
    session_id: str, sender: str | None = Query(None)
) -> SessionResponse:
    resolved = _resolve_sender(sender)
    result = facilitator.resume(resolved, session_id=session_id)
    if result is None:
        raise HTTPException(
            404, f"Session {session_id} cannot be resumed for this sender"
        )
    session, delivery = result
    return SessionResponse(
        session=_serialize_session(session),
        delivery=StepDeliveryPayload.from_delivery(delivery),
    )


@router.post("/sessions/{session_id}/cancel")
def cancel_session(
    session_id: str, sender: str | None = Query(None)
) -> dict[str, Any]:
    resolved = _resolve_sender(sender)
    active = store.get_active(resolved)
    if active is None or active.session_id != session_id:
        raise HTTPException(409, "Session is not active for this sender")
    s = facilitator.cancel(resolved)
    if s is None:
        raise HTTPException(404, "No active session to cancel")
    return _serialize_session(s)


@router.post("/sessions/{session_id}/finish")
def finish_session(
    session_id: str,
    sender: str | None = Query(None),
    generate_report: bool = Query(True),
) -> dict[str, Any]:
    """Close the session and (by default) run the Writer-agent report.

    Long-running. Returns the full session including ``final_report``.
    """
    resolved = _resolve_sender(sender)
    active = store.get_active(resolved)
    target = store.load(session_id)
    if target is None:
        raise HTTPException(404, f"Session {session_id} not found")
    if active is not None and active.session_id != session_id:
        raise HTTPException(
            409,
            f"Cannot finish {session_id}: a different session ({active.session_id}) is active.",
        )
    try:
        s = facilitator.finish(resolved, generate_report=generate_report)
    except FacilitatorError as exc:
        raise HTTPException(400, str(exc)) from exc
    return _serialize_session(s)


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> dict[str, Any]:
    deleted = store.delete(session_id)
    if not deleted:
        raise HTTPException(404, f"Session {session_id} not found")
    return {"deleted": session_id}
