"""
Fiction Inspiration — FastAPI Routes
======================================
Dashboard endpoints for uploading fiction texts (up to 20MB).
Uses direct HTTP upload instead of Firestore queue for large files.

Mounted in main.py as: app.include_router(fiction_router)
"""

import asyncio
import logging
import re
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

logger = logging.getLogger(__name__)

fiction_router = APIRouter(prefix="/fiction", tags=["fiction"])

FICTION_LIBRARY_DIR = Path("/app/workspace/fiction_library")
FICTION_TEXTS_DIR = FICTION_LIBRARY_DIR / "texts"
MAX_UPLOAD_SIZE = 20 * 1024 * 1024  # 20 MB


def _ensure_dirs():
    FICTION_LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    FICTION_TEXTS_DIR.mkdir(parents=True, exist_ok=True)


@fiction_router.post("/upload")
async def upload_fiction_text(
    file: UploadFile = File(...),
    author: str = Form(""),
    title: str = Form(""),
    theme: str = Form(""),
    # Must be Form() — a bare bool is treated as a query param so the
    # multipart "overwrite=true" body field would be silently dropped.
    overwrite: bool = Form(False),
):
    """Upload a .md/.txt fiction file (up to 20MB) into the fiction library.

    Refuses duplicates by default — pass ``overwrite=true`` to replace.
    """
    _ensure_dirs()

    filename = file.filename or "upload.md"
    ext = Path(filename).suffix.lower()
    if ext not in (".md", ".txt"):
        raise HTTPException(400, f"Only .md and .txt files are supported, got {ext}")

    safe_name = re.sub(r"[^\w\-.]", "_", filename)

    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(413, f"File too large ({len(content)} bytes). Max {MAX_UPLOAD_SIZE // (1024*1024)}MB.")
    if not content.strip():
        raise HTTPException(400, "File is empty.")

    # ── Duplicate check (2026-04-26) ─────────────────────────────────
    from app.api.kb_dedup import find_duplicate
    try:
        from app.fiction_inspiration import _get_collection
        col = await asyncio.to_thread(_get_collection)
    except Exception:
        col = None
    dup = await asyncio.to_thread(
        find_duplicate,
        new_content=content,
        new_filename=safe_name,
        existing_files_dir=FICTION_TEXTS_DIR,
        collection=col,
        filename_meta_key="source_file",
    )
    if dup and not overwrite:
        raise HTTPException(status_code=409, detail=dup.as_detail())
    if dup and overwrite:
        try:
            (FICTION_TEXTS_DIR / dup.existing_filename).unlink(missing_ok=True)
            if col is not None:
                # PROGRAM §56 iter-2 — resolve ids first so we can
                # tombstone them; then delete by ids (avoids two
                # different code paths for the same logical op).
                def _fetch_ids() -> list[str]:
                    try:
                        existing = col.get(where={"source_file": dup.existing_filename})
                        return list(existing.get("ids") or [])
                    except Exception:
                        return []
                ids_to_drop = await asyncio.to_thread(_fetch_ids)
                if ids_to_drop:
                    await asyncio.to_thread(col.delete, ids=ids_to_drop)
                    try:
                        from app.memory.source_ledger import hook_collection_delete
                        col_name = getattr(col, "name", "fiction")
                        # Fiction lives in the ``memory`` KB (collection
                        # is opened via chromadb_manager.get_client()
                        # which writes to /app/workspace/memory). The
                        # earlier "knowledge" tag was a kb_name
                        # mis-tag — corrected 2026-05-18 ultrathink pass.
                        hook_collection_delete("memory", col_name, ids_to_drop)
                    except Exception:
                        pass
                else:
                    # Fallback: original where-delete path (no ids found
                    # to ledger but at least don't leak stale rows).
                    await asyncio.to_thread(
                        col.delete, where={"source_file": dup.existing_filename},
                    )
        except Exception:
            logger.debug("fiction: pre-overwrite cleanup failed", exc_info=True)

    text = content.decode("utf-8", errors="replace")

    # Prepend frontmatter if metadata provided and not already present
    if not text.startswith("---") and (author or title or theme):
        fm_lines = ["---"]
        if author:
            fm_lines.append(f"author: {author}")
        if title:
            fm_lines.append(f"title: {title}")
        if theme:
            fm_lines.append(f"themes: [{theme}]")
        fm_lines.append("source_type: fiction")
        fm_lines.append("epistemic_status: imaginary")
        fm_lines.append("---")
        text = "\n".join(fm_lines) + "\n\n" + text

    # Save to fiction library
    dest = FICTION_TEXTS_DIR / safe_name
    dest.write_text(text, encoding="utf-8")

    # Ingest into ChromaDB
    try:
        from app.fiction_inspiration import ingest_book
        result = await asyncio.to_thread(ingest_book, dest, False)
        chunks = result.get("chunks", 0)
    except Exception as e:
        logger.error(f"Fiction ingestion failed for {safe_name}: {e}")
        raise HTTPException(500, f"Ingestion failed: {str(e)[:200]}")

    # Report status to dashboard
    try:
        from app.fiction_inspiration import _report_fiction_status
        await asyncio.to_thread(_report_fiction_status)
    except Exception:
        pass

    return {
        "status": "ok",
        "filename": safe_name,
        "chunks_created": chunks,
        "characters": len(text),
    }


@fiction_router.get("/documents")
async def fiction_documents():
    """Return per-book document list with author, themes, chunks, and
    added_at — normalized to the dashboard's unified shape.

    Aggregates ChromaDB chunks by ``source_file``; each book contributes
    one row carrying its title, author, themes (parsed from the JSON
    string we store), ingestion timestamp, and chunk count.
    """
    import asyncio as _asyncio
    import json as _json
    from app.fiction_inspiration import _get_collection
    try:
        col = await _asyncio.to_thread(_get_collection)
        # Pull metadatas in one pass; for very large libraries (>50k chunks)
        # the orchestrator's chunked-paginate path could be plugged in later.
        data = await _asyncio.to_thread(
            col.get, include=["metadatas"], limit=20_000,
        )
        metas = data.get("metadatas") or []
        by_source: dict[str, dict] = {}
        for m in metas:
            sf = m.get("source_file") or m.get("source") or "unknown"
            if sf not in by_source:
                themes_raw = m.get("themes") or "[]"
                if isinstance(themes_raw, str):
                    try:
                        themes = _json.loads(themes_raw)
                    except Exception:
                        themes = []
                else:
                    themes = list(themes_raw)
                by_source[sf] = {
                    "id": sf,
                    "title": m.get("book_title") or sf,
                    "author": m.get("author") or "Unknown",
                    "themes": [str(t) for t in themes][:8],
                    "genre": m.get("genre"),
                    "chunks": 0,
                    "added_at": m.get("ingested_at"),
                    "source": sf,
                }
            by_source[sf]["chunks"] += 1
        rows = sorted(by_source.values(), key=lambda r: r["title"].lower())
        return {"documents": rows, "total": len(rows)}
    except Exception as e:
        raise HTTPException(500, str(e)[:200])


@fiction_router.get("/status")
@fiction_router.get("/stats")  # alias — frontend symmetry with episteme/aesthetics/tensions
async def fiction_status():
    """Return fiction library statistics.

    Both ``/status`` and ``/stats`` resolve here so the React dashboard
    can use either name. Returns ``total_chunks`` and ``total_documents``
    so the UI's chunk-count and doc-count badges both populate.
    """
    try:
        from app.fiction_inspiration import _get_collection
        col = _get_collection()
        count = col.count()
        # Approximate document count from collection metadata if available;
        # fall back to chunk count as a worst-case overestimate that's
        # still useful as a "stuff is here" signal.
        try:
            meta = col.get(include=["metadatas"], limit=10000)
            docs = {m.get("source") for m in (meta.get("metadatas") or []) if m}
            doc_count = len(docs) or 0
        except Exception:
            doc_count = 0
        return {
            "status": "ok",
            "total_chunks": count,
            "total_documents": doc_count,
            "collection_name": "fiction_inspiration",
        }
    except Exception as e:
        raise HTTPException(500, str(e)[:200])


@fiction_router.post("/reingest")
async def reingest_fiction():
    """Re-ingest all fiction books."""
    try:
        from app.fiction_inspiration import ingest_library
        result = await asyncio.to_thread(ingest_library)
        return {"status": "ok", **result}
    except Exception as e:
        raise HTTPException(500, str(e)[:200])
