"""
kb.py — Knowledge Base API endpoints.

Extracted from main.py. Handles file upload, ingestion, status, removal, and reset.
"""

import asyncio
import logging
import os
import re
import threading

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

logger = logging.getLogger(__name__)

router = APIRouter(tags=["knowledge-base"])

# Lazy singleton for KnowledgeStore (heavy init — loads embedding model)
_kb_store = None
_kb_store_lock = threading.Lock()

ALLOWED_EXTENSIONS = {
    ".pdf", ".docx", ".pptx", ".xlsx", ".csv",
    ".txt", ".md", ".html", ".htm", ".json",
}
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB


def _get_kb_store():
    global _kb_store
    if _kb_store is None:
        with _kb_store_lock:
            if _kb_store is None:
                from app.knowledge_base.vectorstore import KnowledgeStore
                _kb_store = KnowledgeStore()
    return _kb_store


def _stamp_content_hash(collection, source_name: str, content_hash: str) -> None:
    """Add ``content_hash`` to every chunk metadata for a freshly-ingested
    document. Lets future uploads match by content even after rename.

    Run AFTER the store's add_document() succeeds, so we know which
    chunk IDs to update. Non-fatal — caller wraps in try/except.
    """
    if collection is None or not source_name or not content_hash:
        return
    try:
        existing = collection.get(
            where={"source": source_name},
            include=["metadatas"],
        )
        ids = existing.get("ids") or []
        metas = existing.get("metadatas") or []
        if not ids:
            return
        # Update each chunk metadata with content_hash; ChromaDB's update()
        # is the right primitive (vs delete+re-add which would lose
        # embeddings).
        new_metas = []
        for m in metas:
            new = dict(m or {})
            new["content_hash"] = content_hash
            new_metas.append(new)
        collection.update(ids=ids, metadatas=new_metas)
        # PROGRAM §56 iter-2 hook — replay rebuilds the KB from the
        # source ledger, so metadata updates must be mirrored or
        # content_hash will be invisible to the rebuild.
        try:
            from app.memory.source_ledger import hook_collection_update
            hook_collection_update(
                "knowledge", collection.name, list(ids),
                metadatas=new_metas,
            )
        except Exception:
            logger.debug(
                "_stamp_content_hash: ledger update hook failed",
                exc_info=True,
            )
    except Exception:
        logger.debug(
            "_stamp_content_hash: update failed (non-fatal — dedup will "
            "still work on filename match)",
            exc_info=True,
        )


@router.post("/upload")
async def kb_upload(
    file: UploadFile = File(...),
    category: str = Form("general"),
    # Form() not bare bool — without this FastAPI treats it as a query
    # param and ignores the form body, so the dashboard's "Replace"
    # button (which sends overwrite as a multipart field) was being
    # silently dropped and the upload would 409 again.
    overwrite: bool = Form(False),
):
    """Ingest an uploaded file into the knowledge base.

    Refuses duplicates by default — pass ``overwrite=true`` (form field
    or query param) to replace an existing copy.
    """
    import tempfile

    filename = file.filename or "upload"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    category = re.sub(r"[^a-zA-Z0-9_\-]", "", category or "general") or "general"

    tmp_path = None
    try:
        contents = await file.read()
        if len(contents) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail="File too large (max 50 MB)")
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file")

        # ── Duplicate check (2026-04-26) ─────────────────────────────
        # Look for an existing document with the same filename or hash;
        # 409 unless caller explicitly opted into overwrite.
        from app.api.kb_dedup import (
            find_duplicate, compute_content_hash,
        )
        from pathlib import Path as _Path
        # Sanitize the original filename so it's safe to use as a path component
        # (some users upload "résumé final.pdf" or files with slashes).
        sanitized_name = re.sub(r"[^\w\-.]", "_", filename) or "upload"
        store = await asyncio.to_thread(_get_kb_store)
        col = getattr(store, "_collection", None) or getattr(store, "collection", None)
        dup = await asyncio.to_thread(
            find_duplicate,
            new_content=contents,
            new_filename=sanitized_name,
            existing_files_dir=None,           # KB store doesn't keep files on disk
            collection=col,
            filename_meta_key="source",
        )
        if dup and not overwrite:
            raise HTTPException(status_code=409, detail=dup.as_detail())
        if dup and overwrite:
            try:
                await asyncio.to_thread(store.remove_document, dup.existing_filename)
            except Exception:
                logger.debug("KB upload: pre-overwrite remove failed", exc_info=True)

        # Write to a temp directory but PRESERVE the user's filename so
        # the chunk metadata's ``source`` field carries something a
        # human can recognize (and so subsequent dedup checks can match
        # by filename — without this trick every upload looked like a
        # new document with a random tmpfile name).
        tmp_dir = tempfile.mkdtemp(prefix="kb_upload_")
        tmp_path = str(_Path(tmp_dir) / sanitized_name)
        with open(tmp_path, "wb") as tmp_file:
            tmp_file.write(contents)

        result = await asyncio.to_thread(store.add_document, tmp_path, category=category)

        # Stamp content_hash onto every chunk we just added so future
        # uploads can match by hash even if the filename changes.
        try:
            content_hash = compute_content_hash(contents)
            await asyncio.to_thread(
                _stamp_content_hash, col, sanitized_name, content_hash,
            )
        except Exception:
            logger.debug("KB upload: hash stamp failed (non-fatal)", exc_info=True)

        if not result.success:
            raise HTTPException(status_code=422, detail=result.error or "Ingestion failed")

        return {
            "status": "ok",
            "source": result.source,
            "format": result.format,
            "chunks_created": result.chunks_created,
            "total_characters": result.total_characters,
            "document_id": result.document_id,
            "category": category,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"KB upload error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            # Also remove the parent tmpdir we created (mkdtemp leaves
            # the directory behind even after we delete the file).
            try:
                parent = os.path.dirname(tmp_path)
                if parent and parent.startswith("/tmp") and "kb_upload_" in parent:
                    os.rmdir(parent)
            except OSError:
                pass


@router.get("/status")
@router.get("/stats")  # alias for symmetry with episteme/aesthetics/tensions
async def kb_status():
    """Return knowledge base statistics."""
    try:
        store = await asyncio.to_thread(_get_kb_store)
        stats = await asyncio.to_thread(store.stats)
        return {"status": "ok", **stats}
    except Exception as exc:
        logger.error(f"KB status error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/documents")
async def kb_documents():
    """Return the document list for the main Knowledge Base, normalized
    to the unified shape the dashboard's DocumentList component expects:

      [
        {
          "id":       "<source filename>",
          "title":    "<source filename or extracted title>",
          "themes":   ["<category>", "<tag1>", "<tag2>", ...],
          "chunks":   <int>,
          "added_at": "<ISO datetime>",
          "size_bytes": <int>?
        }, ...
      ]

    The underlying ``stats()`` already gathers per-document data
    (source, format, category, tags, total_chunks, ingested_at). We
    just remap the field names so the React side has one consistent
    shape across all KBs.
    """
    try:
        store = await asyncio.to_thread(_get_kb_store)
        stats = await asyncio.to_thread(store.stats)
        docs = stats.get("documents") or []
        out = []
        for d in docs:
            tags = d.get("tags") or []
            if isinstance(tags, str):
                tags = [tags]
            cat = d.get("category", "")
            themes = [t for t in [cat, *tags] if t and t not in ("general",)]
            out.append({
                "id": d.get("source", ""),
                "title": d.get("source", ""),
                "author": None,
                "themes": themes,
                "chunks": d.get("total_chunks", 0),
                "added_at": d.get("ingested_at"),
                "source": d.get("source"),
                "format": d.get("format"),
            })
        return {"documents": out, "total": len(out)}
    except Exception as exc:
        logger.error(f"KB documents error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/remove")
async def kb_remove(request: Request):
    """Remove a document by source_path."""
    try:
        body = await request.json()
        source_path = body.get("source_path", "")
        if not source_path:
            raise HTTPException(status_code=400, detail="source_path required")
        store = await asyncio.to_thread(_get_kb_store)
        count = await asyncio.to_thread(store.remove_document, source_path)
        try:
            from app.firebase_reporter import report_knowledge_base
            report_knowledge_base()
        except Exception:
            pass
        return {"status": "ok", "removed": count, "source_path": source_path}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"KB remove error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/reset")
async def kb_reset_endpoint():
    """Reset the entire knowledge base."""
    try:
        store = await asyncio.to_thread(_get_kb_store)
        await asyncio.to_thread(store.reset)
        try:
            from app.firebase_reporter import report_knowledge_base
            report_knowledge_base()
        except Exception:
            pass
        return {"status": "ok", "message": "Knowledge base has been reset"}
    except Exception as exc:
        logger.error(f"KB reset error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# ═══════════════════════════════════════════════════════════════════════════════
# BUSINESS-SPECIFIC KNOWLEDGE BASES
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/businesses")
async def list_business_kbs():
    """List all business knowledge bases with stats."""
    try:
        from app.knowledge_base.business_store import get_registry
        registry = get_registry()
        return {"businesses": registry.list_businesses()}
    except Exception as exc:
        logger.error(f"Business KB list error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/business/{business_id}/status")
async def business_kb_status(business_id: str):
    """Get status of a specific business knowledge base."""
    try:
        from app.knowledge_base.business_store import get_registry
        store = get_registry().get_or_create(business_id)
        stats = await asyncio.to_thread(store.stats)
        stats["business_id"] = business_id
        return stats
    except Exception as exc:
        logger.error(f"Business KB status error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/business/{business_id}/upload")
async def business_kb_upload(
    business_id: str,
    file: UploadFile = File(...),
    category: str = Form("general"),
):
    """Upload a document to a business-specific knowledge base.

    The business KB is automatically created if it doesn't exist.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    ext = os.path.splitext(file.filename.lower())[1]
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {MAX_UPLOAD_SIZE // (1024 * 1024)} MB limit",
        )

    # Sanitize filename and category.
    safe_name = re.sub(r"[^\w\-.]", "_", file.filename)
    category = re.sub(r"[^a-zA-Z0-9_\-]", "", category or "general") or "general"

    import tempfile
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=ext, prefix=f"biz_{business_id}_",
    ) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        from app.knowledge_base.business_store import get_registry
        store = get_registry().get_or_create(business_id)
        result = await asyncio.to_thread(
            store.add_document, tmp_path, category,
        )

        # Report to Firebase.
        try:
            from app.firebase.publish import report_business_kb
            report_business_kb(business_id)
        except Exception:
            pass

        return {
            "status": "ok",
            "business_id": business_id,
            "source": safe_name,
            "format": ext.lstrip("."),
            "chunks_created": result.chunks_created,
            "total_characters": result.total_characters,
            "category": category,
        }
    except Exception as exc:
        logger.error(f"Business KB upload error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@router.post("/business/{business_id}/remove")
async def business_kb_remove(business_id: str, request: Request):
    """Remove a document from a business knowledge base."""
    try:
        body = await request.json()
        source_path = body.get("source_path", "")
        if not source_path:
            raise HTTPException(status_code=400, detail="source_path required")

        from app.knowledge_base.business_store import get_registry
        store = get_registry().get_or_create(business_id)
        removed = await asyncio.to_thread(store.remove_document, source_path)

        try:
            from app.firebase.publish import report_business_kb
            report_business_kb(business_id)
        except Exception:
            pass

        return {"status": "ok", "removed": removed, "source_path": source_path}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Business KB remove error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/business/{business_id}/reset")
async def business_kb_reset(business_id: str):
    """Reset a business knowledge base (delete all documents)."""
    try:
        from app.knowledge_base.business_store import get_registry
        store = get_registry().get_or_create(business_id)
        await asyncio.to_thread(store.reset)

        try:
            from app.firebase.publish import report_business_kb
            report_business_kb(business_id)
        except Exception:
            pass

        return {"status": "ok", "message": f"Business KB '{business_id}' has been reset"}
    except Exception as exc:
        logger.error(f"Business KB reset error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
