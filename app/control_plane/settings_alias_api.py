"""Settings-endpoint alias under /api/cp/settings.

PROGRAM §46.12 — The React settings cards (Person, Resilience,
Architecture, InlineEvolve, Travel) all POST to ``/api/cp/settings``
and GET ``/api/cp/settings``. The canonical handler is
``POST /config/runtime_settings``; this alias forwards the request
shape and re-uses the same setter dispatcher.

The shape is intentionally identical to ``/config/runtime_settings``
so existing React code never had to know about the rename — the
operator-visible URL stayed stable.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from app.control_plane.auth_dep import require_gateway_auth

logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/api/cp",
    tags=["control-plane", "settings-alias"],
    dependencies=[Depends(require_gateway_auth)],
)


@router.get("/settings")
def get_settings() -> dict[str, Any]:
    """Return the live runtime-settings snapshot.

    Mirrors ``GET /config/runtime_settings`` so the React
    ``useRuntimeSettingsQuery`` hook can read the full state via the
    same /api/cp/* prefix it uses for everything else.
    """
    from app.runtime_settings import snapshot
    return snapshot()


@router.post("/settings")
async def set_settings(request: Request) -> dict[str, Any]:
    """Apply a settings patch. Body is a plain JSON object whose
    keys/values match the runtime_settings schema; the canonical
    handler routes each key to its setter.

    Delegates to ``app.api.config_api.set_runtime_settings_endpoint``
    so we have ONE setter dispatcher; this alias is just URL
    plumbing. Auth is enforced via ``require_gateway_auth`` (the
    canonical handler's ``verify_gateway_secret`` is a redundant
    second check that returns the same 401 we already raised).
    """
    from app.api.config_api import set_runtime_settings_endpoint
    return await set_runtime_settings_endpoint(request)
