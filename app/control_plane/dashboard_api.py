"""Control Plane dashboard API routes — composition root.

Provides REST endpoints for the React dashboard. All routes prefixed
with ``/api/cp/``.

Auth (Phase B1): when ``GATEWAY_AUTH_REQUIRED=1`` is set, every route
on this router requires ``Authorization: Bearer <gateway-secret>``.
Default behaviour (env var unset) is pass-through, preserving the
laptop developer experience. See :mod:`app.control_plane.auth_dep`.

The 110 route handlers live in eight topic-keyed
``dashboard_routes_*.py`` modules. This file is now just the
composition root: it owns the parent ``router`` (which carries the
``/api/cp`` prefix + ``require_gateway_auth`` dependency) and
``include_router``s each split. FastAPI propagates the parent's
prefix and dependencies to every child route, so the URL surface
and auth boundary are identical to the pre-Phase-1 monolith.

History — WP G ran in two phases:

* Phase 1 (2026-05-17 23:48) — extracted 110 handlers into eight
  ``dashboard_routes_*.py`` modules. The extract was complete but
  the wire-up step was deferred and the monolith retained the
  original copies; the splits were unmounted (orphaned).
* Phase 2 (2026-05-18) — completed the refactor: synced the single
  in-place monolith edit (``/source-ledger/state`` docstring) into
  its split, deleted the 110 duplicate handlers from this file,
  wired the splits via ``include_router`` below.

Adding a new topic module: drop a ``dashboard_routes_<topic>.py``
with its own ``router = APIRouter()`` (no prefix, no deps), then
add one ``from ... import router as _<topic>_router`` line and one
``router.include_router(_<topic>_router)`` line below.
"""
import logging

from fastapi import APIRouter, Depends

from app.control_plane.auth_dep import require_gateway_auth
from app.control_plane.dashboard_routes_budgets_costs import (
    router as _budgets_costs_router,
)
from app.control_plane.dashboard_routes_companion import (
    router as _companion_router,
)
from app.control_plane.dashboard_routes_governance_ops import (
    router as _governance_ops_router,
)
from app.control_plane.dashboard_routes_llms import router as _llms_router
from app.control_plane.dashboard_routes_ops_misc import (
    router as _ops_misc_router,
)
from app.control_plane.dashboard_routes_projects_tickets import (
    router as _projects_tickets_router,
)
from app.control_plane.dashboard_routes_sentience_drills import (
    router as _sentience_drills_router,
)
from app.control_plane.dashboard_routes_transfer_memory import (
    router as _transfer_memory_router,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/cp",
    tags=["control-plane"],
    dependencies=[Depends(require_gateway_auth)],
)

router.include_router(_projects_tickets_router)
router.include_router(_budgets_costs_router)
router.include_router(_companion_router)
router.include_router(_governance_ops_router)
router.include_router(_llms_router)
router.include_router(_ops_misc_router)
router.include_router(_sentience_drills_router)
router.include_router(_transfer_memory_router)
