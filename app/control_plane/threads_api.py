"""Control plane — long-horizon thread endpoints at /api/cp/threads.

Operators (via React or curl) can:

  GET  /api/cp/threads                     list (open by default)
  GET  /api/cp/threads/{id}                detail
  POST /api/cp/threads                     create new thread
  POST /api/cp/threads/{id}/sub-question   add sub-question
  POST /api/cp/threads/{id}/resolve-sq     resolve a sub-question
  POST /api/cp/threads/{id}/blocker        add a blocker
  POST /api/cp/threads/{id}/clear-blockers clear blockers + un-block
  POST /api/cp/threads/{id}/note           append a note
  POST /api/cp/threads/{id}/link-cr        link a child change-request id
  POST /api/cp/threads/{id}/link-inquiry   link an inquiry slug
  POST /api/cp/threads/{id}/transition     state transition: blocked /
                                            in_progress / resolved / abandoned

Auth: require_gateway_auth dependency.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.control_plane.auth_dep import require_gateway_auth
from app.threads import (
    InvalidThreadTransition,
    ThreadStatus,
    abandon_thread,
    add_blocker,
    add_subquestion,
    clear_blockers,
    create_thread,
    get,
    link_crew_task,
    link_inquiry,
    list_all,
    list_open,
    mark_blocked,
    mark_in_progress,
    record_note,
    resolve_subquestion,
    resolve_thread,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/cp/threads",
    tags=["control-plane", "threads"],
    dependencies=[Depends(require_gateway_auth)],
)


class _CreateBody(BaseModel):
    title: str = Field(..., min_length=1)
    description: str = Field(default="")


class _SubQuestionBody(BaseModel):
    text: str = Field(..., min_length=1)


class _ResolveSqBody(BaseModel):
    subquestion_id: str = Field(..., min_length=1)
    resolution: str = Field(default="")


class _BlockerBody(BaseModel):
    text: str = Field(..., min_length=1)


class _NoteBody(BaseModel):
    text: str = Field(..., min_length=1)


class _LinkCrBody(BaseModel):
    crew_task_id: str = Field(..., min_length=1)


class _LinkInquiryBody(BaseModel):
    inquiry_slug: str = Field(..., min_length=1)


class _TransitionBody(BaseModel):
    transition: str = Field(...)
    blocker: str | None = Field(default=None)
    summary: str | None = Field(default=None)
    reason: str | None = Field(default=None)


def _serialize(t) -> dict[str, Any]:
    if t is None:
        return {}
    d = t.to_dict()
    d["is_terminal"] = t.is_terminal
    d["open_subquestion_count"] = len(t.open_subquestions)
    d["resolved_subquestion_count"] = len(t.resolved_subquestions)
    return d


def _require(thread_id: str):
    t = get(thread_id)
    if t is None:
        raise HTTPException(
            status_code=404, detail=f"thread {thread_id!r} not found",
        )
    return t


@router.get("")
def list_threads(
    open_only: bool = Query(default=True),
    limit: int = Query(default=100, ge=1, le=500),
):
    items = list_open(limit=limit) if open_only else list_all(limit=limit)
    return {"count": len(items), "threads": [_serialize(t) for t in items]}


@router.get("/{thread_id}")
def get_thread(thread_id: str):
    return _serialize(_require(thread_id))


@router.post("")
def create(body: _CreateBody):
    t = create_thread(title=body.title, description=body.description)
    return {"ok": True, "thread": _serialize(t)}


@router.post("/{thread_id}/sub-question")
def add_sq(thread_id: str, body: _SubQuestionBody):
    _require(thread_id)
    try:
        t = add_subquestion(thread_id, body.text)
    except InvalidThreadTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"ok": True, "thread": _serialize(t)}


@router.post("/{thread_id}/resolve-sq")
def resolve_sq(thread_id: str, body: _ResolveSqBody):
    _require(thread_id)
    try:
        t = resolve_subquestion(thread_id, body.subquestion_id, body.resolution)
    except InvalidThreadTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"ok": True, "thread": _serialize(t)}


@router.post("/{thread_id}/blocker")
def add_blocker_route(thread_id: str, body: _BlockerBody):
    _require(thread_id)
    try:
        t = add_blocker(thread_id, body.text)
    except InvalidThreadTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"ok": True, "thread": _serialize(t)}


@router.post("/{thread_id}/clear-blockers")
def clear_route(thread_id: str):
    _require(thread_id)
    t = clear_blockers(thread_id)
    return {"ok": True, "thread": _serialize(t)}


@router.post("/{thread_id}/note")
def add_note(thread_id: str, body: _NoteBody):
    _require(thread_id)
    try:
        t = record_note(thread_id, body.text)
    except InvalidThreadTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"ok": True, "thread": _serialize(t)}


@router.post("/{thread_id}/link-cr")
def link_cr(thread_id: str, body: _LinkCrBody):
    _require(thread_id)
    t = link_crew_task(thread_id, body.crew_task_id)
    return {"ok": True, "thread": _serialize(t)}


@router.post("/{thread_id}/link-inquiry")
def link_inq(thread_id: str, body: _LinkInquiryBody):
    _require(thread_id)
    t = link_inquiry(thread_id, body.inquiry_slug)
    return {"ok": True, "thread": _serialize(t)}


@router.post("/{thread_id}/transition")
def transition(thread_id: str, body: _TransitionBody):
    _require(thread_id)
    try:
        if body.transition == ThreadStatus.BLOCKED.value:
            t = mark_blocked(thread_id, blocker=body.blocker)
        elif body.transition == ThreadStatus.IN_PROGRESS.value:
            t = mark_in_progress(thread_id)
        elif body.transition == ThreadStatus.RESOLVED.value:
            t = resolve_thread(thread_id, summary=body.summary or "")
        elif body.transition == ThreadStatus.ABANDONED.value:
            if not body.reason:
                raise HTTPException(
                    status_code=400, detail="abandon requires reason",
                )
            t = abandon_thread(thread_id, reason=body.reason)
        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"unsupported transition {body.transition!r}. "
                    f"Valid: blocked / in_progress / resolved / abandoned."
                ),
            )
    except InvalidThreadTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"ok": True, "thread": _serialize(t)}
