"""Control plane — substrate status endpoint at /api/cp/substrate/status.

Productization plan T2.4. Surfaces the substrate facade
(``app.substrate.gather_substrate_status``) as one read-only JSON
endpoint, consumed by the React ``/cp/status`` landing page.

Pure read. Never raises — the facade's per-probe error isolation is
the safety boundary; this endpoint just transmits the snapshot.

Auth: same ``require_gateway_auth`` dependency as the rest of /cp/.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends

from app.control_plane.auth_dep import require_gateway_auth

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/cp/substrate",
    tags=["control-plane", "substrate"],
    dependencies=[Depends(require_gateway_auth)],
)


@router.get("/status")
def get_substrate_status() -> dict[str, Any]:
    """Return the substrate status snapshot as JSON.

    Sections (each with a safe default):
      - timestamp, inflight_tasks, inbound_queue_depth, dlq_depth
      - memory: ChromaDB / Postgres / Neo4j / Mem0 health + 30d drift
      - subia: live flag, integrity, kernel state
      - self_improvement: pending CRs / proposals / last deploy
      - resources: disk free, host substrate state + alerts
      - continuity: last backup age, ledger event count
      - health: monitor state count
      - settings: active feature flags
      - errors: per-probe failure strings (always present)
    """
    try:
        from app.substrate import gather_substrate_status
        snap = gather_substrate_status()
        return snap.to_dict()
    except Exception as exc:
        # Belt-and-suspenders — gather_substrate_status is supposed to
        # never raise, but if importing the package itself fails we still
        # want a useful error response rather than a 500.
        logger.exception("substrate_api: gather failed")
        return {
            "timestamp": "",
            "ok": False,
            "errors": [f"substrate_api: {exc}"],
        }
