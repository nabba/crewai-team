"""Control plane — read-only system-state browser at /api/cp/system-state.

Phase 5.1 endpoint. Surfaces git head + gateway uptime + recent crew
runs + tool registry size to the React control plane. Operators read
this for:

  * Pre-deploy sanity checks ("is everything I expect to be there?").
  * Post-incident triage ("what was running when X happened?").
  * Phase 5.3 React UI: change-request review needs to show
    deployment context alongside the proposed diff.

Auth: same `require_gateway_auth` dependency as the rest of /cp/.
Default ON in K8s, OFF on laptop dev.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query

from app.control_plane.auth_dep import require_gateway_auth
from app.system_state import get_system_state

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/cp/system-state",
    tags=["control-plane", "system-state"],
    dependencies=[Depends(require_gateway_auth)],
)


@router.get("")
def read_system_state(
    window_hours: int = Query(
        24,
        description="Hours of history to include in `git.files_changed_last_24h` and `recent_crew_runs`. Default 24.",
        ge=1,
        le=168,
    ),
    use_cache: bool = Query(
        True,
        description="Use the 5-second internal cache. Set False to force a fresh read.",
    ),
):
    """Compose deployment state from git, gateway, registry, and crew-run sources.

    Each top-level section has an `available` boolean. Operators
    (and the React UI) MUST check it before reading other fields —
    sources can be transiently unreachable and we degrade rather
    than fail.

    Caching: 5 seconds. Cheap to call freely.
    """
    return get_system_state(
        window_hours=window_hours,
        use_cache=use_cache,
    )
