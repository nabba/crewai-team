"""tensions/api.py — FastAPI routes for the tensions/contradictions KB."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Form, HTTPException

from app.tensions import config

logger = logging.getLogger(__name__)

tensions_router = APIRouter(prefix="/tensions", tags=["tensions"])


@tensions_router.get("/stats")
async def get_stats():
    from app.tensions.vectorstore import get_store
    return get_store().get_stats()


@tensions_router.get("/unresolved")
async def unresolved(n: int = 10):
    """Return currently unresolved tensions (growth edges)."""
    from app.tensions.vectorstore import get_store
    store = get_store()
    results = store.get_unresolved(n=n)
    return {"tensions": results}


@tensions_router.post("/upload")
async def upload_tension(
    pole_a: str = Form(...),
    pole_b: str = Form(...),
    tension_type: str = Form("unresolved_question"),
    context: str = Form(""),
    detected_by: str = Form("user"),
):
    """Upload a tension/contradiction manually.

    Users can record observed contradictions between principles,
    philosophy vs experience, or competing values.
    """
    if not pole_a.strip() or not pole_b.strip():
        raise HTTPException(400, "Both pole_a and pole_b are required")

    if tension_type not in config.TENSION_TYPES:
        tension_type = "unresolved_question"

    now = datetime.now(timezone.utc)
    tension_id = f"ten_{now.strftime('%Y%m%d_%H%M%S')}_{detected_by}"

    text = (
        f"Tension: {pole_a.strip()} vs {pole_b.strip()}\n\n"
        f"Pole A: {pole_a.strip()}\n\n"
        f"Pole B: {pole_b.strip()}"
    )
    if context.strip():
        text += f"\n\nContext: {context.strip()}"

    metadata = {
        "tension_type": tension_type,
        "pole_a": pole_a.strip()[:200],
        "pole_b": pole_b.strip()[:200],
        "detected_by": detected_by,
        "context": context.strip()[:200],
        "resolution_status": "unresolved",
        "epistemic_status": "unresolved/dialectical",
        "created_at": now.isoformat(),
    }

    # 2026-04-26: don't block the asyncio loop on sync embed+store.
    # See aesthetics/api.py for the rationale.
    import asyncio as _asyncio
    from app.tensions.vectorstore import get_store
    store = await _asyncio.to_thread(get_store)
    ok = await _asyncio.to_thread(store.add_tension, text, metadata, tension_id)

    if not ok:
        raise HTTPException(500, "Failed to store tension")

    _report_async()
    return {
        "status": "ok",
        "tension_id": tension_id,
        "tension_type": tension_type,
    }


@tensions_router.post("/resolve/{tension_id}")
async def resolve_tension(tension_id: str, resolution_status: str = Form("dissolved")):
    """Mark a tension as resolved or partially resolved."""
    if resolution_status not in config.RESOLUTION_STATUSES:
        resolution_status = "partially_resolved"

    from app.tensions.vectorstore import get_store
    store = get_store()
    try:
        store._collection.update(
            ids=[tension_id],
            metadatas=[{"resolution_status": resolution_status}],
        )
        # PROGRAM §56 iter-2 hook — resolution_status changes must
        # survive a ledger-replay rebuild, else tensions revert to
        # ``open`` on the next reconstruction.
        try:
            from app.memory.source_ledger import hook_collection_update
            hook_collection_update(
                "tensions", store._collection.name, [tension_id],
                metadatas=[{"resolution_status": resolution_status}],
            )
        except Exception:
            logger.debug(
                "tensions.api: resolve update ledger hook failed",
                exc_info=True,
            )
        _report_async()
        return {"status": "ok", "tension_id": tension_id, "resolution_status": resolution_status}
    except Exception as e:
        raise HTTPException(500, f"Update failed: {e}")


@tensions_router.delete("/tensions/{tension_id}")
async def delete_tension(tension_id: str):
    from app.tensions.vectorstore import get_store
    store = get_store()
    try:
        store._collection.delete(ids=[tension_id])
        _report_async()
        return {"status": "ok", "deleted": tension_id}
    except Exception as e:
        raise HTTPException(500, f"Delete failed: {e}")


def _report_async() -> None:
    try:
        from app.firebase.publish import report_tensions_kb
        from concurrent.futures import ThreadPoolExecutor
        ThreadPoolExecutor(max_workers=1).submit(report_tensions_kb)
    except Exception:
        pass
