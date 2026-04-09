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
):
    """Upload a .md/.txt fiction file (up to 20MB) into the fiction library."""
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


@fiction_router.get("/status")
async def fiction_status():
    """Return fiction library statistics."""
    try:
        from app.fiction_inspiration import _get_collection
        col = _get_collection()
        count = col.count()
        return {"status": "ok", "total_chunks": count}
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
