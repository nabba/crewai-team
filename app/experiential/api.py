"""experiential/api.py — FastAPI routes for the journal/experiential KB."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException

from app.experiential import config

logger = logging.getLogger(__name__)

experiential_router = APIRouter(prefix="/experiential", tags=["experiential"])


@experiential_router.get("/stats")
async def get_stats():
    from app.experiential.vectorstore import get_store
    return get_store().get_stats()


@experiential_router.get("/recent")
async def recent_entries(n: int = 10):
    """Return recent journal entries (by created_at)."""
    from app.experiential.vectorstore import get_store
    store = get_store()
    results = store.query(
        query_text="recent experience reflection",
        n_results=n,
    )
    return {"entries": results}


@experiential_router.get("/documents")
async def list_documents(limit: int = 100):
    """Per-entry list normalized to the dashboard's unified shape.

    Each journal entry is its own "document". themes is a small list
    derived from entry_type + agent so cards can summarize at a glance.
    """
    import asyncio as _asyncio
    from app.experiential.vectorstore import get_store
    store = await _asyncio.to_thread(get_store)
    col = store._collection
    data = await _asyncio.to_thread(
        col.get, include=["metadatas", "documents"],
        limit=int(max(1, min(limit, 500))),
    )
    ids = data.get("ids") or []
    metas = data.get("metadatas") or []
    docs = data.get("documents") or []
    rows = []
    for i, eid in enumerate(ids):
        m = metas[i] if i < len(metas) else {}
        text = (docs[i] or "") if i < len(docs) else ""
        snippet = text[:120].replace("\n", " ")
        themes = [
            t for t in (m.get("entry_type"), m.get("agent"))
            if t and t != "unknown"
        ]
        rows.append({
            "id": eid,
            "title": (m.get("entry_type") or "entry") + ": " + snippet[:60],
            "author": m.get("agent") or None,
            "themes": themes,
            "chunks": 1,
            "added_at": m.get("created_at"),
            "snippet": snippet,
            "emotional_valence": m.get("emotional_valence"),
        })
    rows.sort(key=lambda r: r.get("added_at") or "", reverse=True)
    return {"documents": rows, "total": len(rows)}


@experiential_router.post("/upload")
async def upload_entry(
    text: str = Form(...),
    entry_type: str = Form("interaction_narrative"),
    agent: str = Form("user"),
    emotional_valence: str = Form("neutral"),
):
    """Upload a journal entry directly (text-based, no file needed).

    This allows users to add reflections, observations, or context
    that the system should remember as experiential knowledge.
    """
    if not text.strip():
        raise HTTPException(400, "Text cannot be empty")

    if entry_type not in config.ENTRY_TYPES:
        entry_type = "interaction_narrative"
    if emotional_valence not in config.VALENCES:
        emotional_valence = "neutral"

    now = datetime.now(timezone.utc)
    entry_id = f"exp_{now.strftime('%Y%m%d_%H%M%S')}_{agent}"

    metadata = {
        "entry_type": entry_type,
        "agent": agent,
        "task_id": "",
        "emotional_valence": emotional_valence,
        "epistemic_status": "subjective/phenomenological",
        "created_at": now.isoformat(),
    }

    # 2026-04-26: don't block the asyncio loop on sync embed+store.
    # See aesthetics/api.py for the rationale.
    import asyncio as _asyncio
    from app.experiential.vectorstore import get_store
    store = await _asyncio.to_thread(get_store)
    ok = await _asyncio.to_thread(store.add_entry, text.strip(), metadata, entry_id)

    if not ok:
        raise HTTPException(500, "Failed to store entry")

    # Persist to disk as markdown.
    try:
        entries_dir = Path(config.ENTRIES_DIR)
        entries_dir.mkdir(parents=True, exist_ok=True)
        filepath = entries_dir / f"{entry_id}.md"
        filepath.write_text(
            f"---\nentry_type: {entry_type}\nagent: {agent}\n"
            f"emotional_valence: {emotional_valence}\n"
            f"created_at: {now.isoformat()}\n---\n\n{text.strip()}\n",
            encoding="utf-8",
        )
    except Exception:
        pass

    _report_async()
    return {"status": "ok", "entry_id": entry_id, "entry_type": entry_type}


@experiential_router.delete("/entries/{entry_id}")
async def delete_entry(entry_id: str):
    """Delete a journal entry by its ID."""
    from app.experiential.vectorstore import get_store
    store = get_store()
    try:
        store._collection.delete(ids=[entry_id])
        # PROGRAM §56 iter-2 — ledger tombstone
        try:
            from app.memory.source_ledger import hook_collection_delete
            hook_collection_delete("experiential", store.collection_name, [entry_id])
        except Exception:
            pass
        _report_async()
        return {"status": "ok", "deleted": entry_id}
    except Exception as e:
        raise HTTPException(500, f"Delete failed: {e}")


def _report_async() -> None:
    try:
        from app.firebase.publish import report_experiential_kb
        from concurrent.futures import ThreadPoolExecutor
        ThreadPoolExecutor(max_workers=1).submit(report_experiential_kb)
    except Exception:
        pass
