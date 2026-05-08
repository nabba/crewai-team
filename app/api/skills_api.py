"""
skills_api.py — control-plane endpoints for the React /cp/skills page.

Routes:

    GET   /api/cp/skills            list all saved skills
    GET   /api/cp/skills/{name}     show one skill
    POST  /api/cp/skills            save a new skill (or overwrite existing)
    DELETE /api/cp/skills/{name}    remove a skill
    POST  /api/cp/skills/{name}/run substitute args + dispatch via Commander
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cp/skills", tags=["skills"])


def _verify(request: Request) -> None:
    """Mirror the auth pattern used by config_api.verify_gateway_secret."""
    from app.api.config_api import verify_gateway_secret
    if not verify_gateway_secret(request):
        raise HTTPException(status_code=401, detail="Unauthorized")


@router.get("")
async def list_skills_endpoint():
    from app.skills import list_skills
    return {"skills": [s.to_dict() for s in list_skills()]}


@router.get("/{name}")
async def get_skill_endpoint(name: str):
    from app.skills import get_skill
    s = get_skill(name)
    if not s:
        raise HTTPException(status_code=404, detail=f"no skill named {name!r}")
    return s.to_dict()


@router.post("")
async def save_skill_endpoint(request: Request):
    """Create or overwrite a skill.

    Body: {name, task_template, description?, force_tier?, extra_tools?, task_hint?}
    """
    _verify(request)
    payload = await request.json()
    try:
        from app.skills import save_skill
        skill = save_skill(
            name=payload.get("name", ""),
            task_template=payload.get("task_template", ""),
            description=payload.get("description", ""),
            force_tier=payload.get("force_tier"),
            extra_tools=payload.get("extra_tools") or [],
            task_hint=payload.get("task_hint", ""),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        import json as _json
        from app.audit import log_security_event
        log_security_event(
            "skill_save",
            _json.dumps({"name": skill.name, "args": skill.args_schema}),
        )
    except Exception:
        pass
    return skill.to_dict()


@router.delete("/{name}")
async def delete_skill_endpoint(name: str, request: Request):
    _verify(request)
    from app.skills import delete_skill
    return {"removed": delete_skill(name)}


@router.post("/{name}/run")
async def run_skill_endpoint(name: str, request: Request):
    """Run a skill via the Commander. Body: {args: {k: v}, sender?: str}."""
    _verify(request)
    payload = await request.json()
    args = payload.get("args") or {}
    sender = payload.get("sender") or "+control-plane"

    try:
        # Late import to avoid pulling commander into the test surface for
        # endpoints that don't need it.
        from app.agents.commander import Commander  # type: ignore
        commander = Commander()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Commander unavailable: {exc}",
        )

    try:
        from app.skills import run_skill
        result = run_skill(name, args, sender, commander)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"no skill named {name!r}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        import json as _json
        from app.audit import log_security_event
        log_security_event(
            "skill_run",
            _json.dumps({"name": name, "args": list(args.keys())}),
        )
    except Exception:
        pass
    return {"result": result}
