"""
files_api — list, download, and dispatch generated artifacts.

Routes:

    GET  /api/cp/files                   list (grouped) under workspace/
    GET  /api/cp/files/download?path=…   stream a single file
    POST /api/cp/files/send              deliver a file via Signal /
                                          Email / Discord

Three artifact roots are surfaced:

    output           workspace/output/        (PDF/DOCX/XLSX/PPTX/HTML decks
                                               — anything the docgen tool produced)
    skills           workspace/skills/        (markdown skill files)
    notes            workspace/notes/         (Obsidian-style notes, if present)

Path traversal is blocked: every requested path is resolved + checked
against ``WORKSPACE_ROOT`` before any read. Files outside the workspace
are rejected with 400.
"""
from __future__ import annotations

import logging
import mimetypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse

from app.paths import WORKSPACE_ROOT

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cp/files", tags=["files"])

# Roots surfaced to the dashboard. Add more by extending this dict.
_ROOTS: dict[str, Path] = {
    "output": WORKSPACE_ROOT / "output",
    "skills": WORKSPACE_ROOT / "skills",
    "notes": WORKSPACE_ROOT / "notes",
}

# File-extension allow-list per root — keep the surface tight so a
# random scratch file doesn't accidentally end up listed in the UI.
_LIST_EXTENSIONS: dict[str, set[str]] = {
    "output": {".pdf", ".docx", ".xlsx", ".pptx", ".html", ".csv", ".png", ".jpg"},
    "skills": {".md"},
    "notes":  {".md", ".pdf"},
}

_MAX_LIST_PER_ROOT = 200


def _verify(request: Request) -> None:
    from app.api.config_api import verify_gateway_secret
    if not verify_gateway_secret(request):
        raise HTTPException(status_code=401, detail="Unauthorized")


def _safe_path_for_download(raw: str) -> Path:
    """Resolve a request path, refuse anything escaping ``WORKSPACE_ROOT``."""
    if not raw or not raw.strip():
        raise HTTPException(status_code=400, detail="path required")
    p = (WORKSPACE_ROOT / raw.lstrip("/")).resolve()
    try:
        p.relative_to(WORKSPACE_ROOT.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="path escapes workspace")
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return p


def _walk(root: Path, allowed_ext: set[str]) -> Iterable[Path]:
    """Yield files under ``root`` whose suffix is in ``allowed_ext``."""
    if not root.exists() or not root.is_dir():
        return []
    out: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if allowed_ext and p.suffix.lower() not in allowed_ext:
            continue
        out.append(p)
    out.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return out[:_MAX_LIST_PER_ROOT]


def _entry(p: Path) -> dict:
    rel = p.relative_to(WORKSPACE_ROOT)
    stat = p.stat()
    return {
        "name": p.name,
        "path": str(rel),
        "size": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
                                                                          .replace("+00:00", "Z"),
        "extension": p.suffix.lower().lstrip("."),
        "mime": mimetypes.guess_type(p.name)[0] or "application/octet-stream",
    }


@router.get("")
async def list_files_endpoint():
    """Return all listable artifacts grouped by root."""
    out: dict[str, list[dict]] = {}
    for name, root in _ROOTS.items():
        ext = _LIST_EXTENSIONS.get(name, set())
        out[name] = [_entry(p) for p in _walk(root, ext)]
    return {"roots": out}


@router.get("/download")
async def download_endpoint(path: str = Query(..., description="Workspace-relative path")):
    """Stream a file. No auth — same model as workspace/output viewing."""
    p = _safe_path_for_download(path)
    return FileResponse(
        path=str(p),
        filename=p.name,
        media_type=mimetypes.guess_type(p.name)[0] or "application/octet-stream",
    )


@router.post("/send")
async def send_endpoint(request: Request):
    """Deliver one file via signal | email | discord.

    Body:
      {
        "channel":   "signal" | "email" | "discord",
        "path":      "<workspace-relative path>",
        "body":      "<short message>",
        "to":        "<email address>"        // email-only
      }
    """
    _verify(request)
    payload = await request.json()
    channel = (payload.get("channel") or "").lower().strip()
    raw_path = payload.get("path") or ""
    body = (payload.get("body") or "").strip()

    file_path = _safe_path_for_download(raw_path)

    try:
        if channel == "signal":
            from app.delivery import send_via_signal
            ok, detail = send_via_signal([file_path], body=body)
        elif channel == "email":
            to = (payload.get("to") or "").strip()
            if not to:
                raise HTTPException(status_code=400, detail="email needs `to`")
            subject = (payload.get("subject") or f"AndrusAI: {file_path.name}").strip()
            from app.delivery import send_via_email
            ok, detail = send_via_email(
                to=to, subject=subject, body=body or f"Sending {file_path.name}.",
                attachment_paths=[file_path],
            )
        elif channel == "discord":
            from app.config import get_settings
            from app.delivery import send_via_discord
            owner = (get_settings().discord_owner_id or "").strip()
            if not owner:
                raise HTTPException(status_code=400, detail="DISCORD_OWNER_ID not set")
            ok, detail = send_via_discord(
                owner, body=body, attachment_paths=[file_path],
            )
        else:
            raise HTTPException(status_code=400, detail=f"unknown channel {channel!r}")
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("files_api.send failed")
        raise HTTPException(status_code=500, detail=f"send failed: {exc}")

    if not ok:
        raise HTTPException(status_code=502, detail=detail)

    try:
        import json as _json
        from app.audit import log_security_event
        log_security_event(
            "files_send",
            _json.dumps({"channel": channel, "path": str(file_path.relative_to(WORKSPACE_ROOT))}),
        )
    except Exception:
        pass
    return {"status": "ok", "detail": detail}
