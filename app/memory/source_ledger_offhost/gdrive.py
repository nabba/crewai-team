"""
gdrive.py — Google Drive off-host uploader for the per-KB source ledger.

PROGRAM §56 (2026-05-17). Uses the existing Google Workspace OAuth
pipe (``app/google_workspace/``) so the operator doesn't need a
separate auth flow — if Google tools are working, Drive upload
works too.

Configuration via env / runtime settings::

    LEDGER_GDRIVE_FOLDER_ID    required — target Drive folder ID
                                (or get/create one named "AndrusAI-Ledgers"
                                under operator's My Drive on first run)

Activated by ``chromadb_ledger_gdrive_upload_enabled`` (default OFF).

Object naming follows the same canonical pattern as the S3 uploader::

    AndrusAI-Ledgers/<host>/<kb>/source_ledger/YYYY-MM-DD.jsonl.gz

Drive does not have S3-style flat prefixes — we materialise the path
as nested folders, creating them on demand. Folders are cached so
the per-day upload only does one API call after the first run.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

from app.memory.source_ledger_offhost.common import (
    OffhostUploadResult,
    canonical_object_key,
    incremental_payload,
    record_upload,
)

logger = logging.getLogger(__name__)


_DEFAULT_ROOT_NAME = "AndrusAI-Ledgers"

# Per-process cache of (parent_id, name) → folder_id so we don't
# repeatedly call drive.files.list during a single daemon pass.
_folder_cache: dict[tuple[str, str], str] = {}


def _enabled() -> bool:
    try:
        from app.runtime_settings import get_chromadb_ledger_gdrive_upload_enabled
        return bool(get_chromadb_ledger_gdrive_upload_enabled())
    except Exception:
        return False


def _get_drive_service():
    """Reuse the cached googleapiclient.discovery.Resource from
    app.google_workspace (the same one the existing gdrive/gmail
    tools use). Returns None on any failure so caller can skip
    silently."""
    try:
        from app.google_workspace import get_service  # type: ignore
        return get_service("drive")
    except Exception:
        logger.debug(
            "source_ledger_offhost.gdrive: google_workspace not configured",
            exc_info=True,
        )
        return None


def _ensure_folder(service, parent_id: str, name: str) -> Optional[str]:
    """Look up a folder by name under parent_id; create if missing.

    Returns the folder ID, or None on any failure. Cached per
    process to keep the steady-state cost at one upload per day per
    KB.
    """
    cache_key = (parent_id, name)
    if cache_key in _folder_cache:
        return _folder_cache[cache_key]
    try:
        query = (
            f"name='{name}' and "
            f"mimeType='application/vnd.google-apps.folder' and "
            f"'{parent_id}' in parents and trashed=false"
        )
        res = service.files().list(q=query, fields="files(id,name)").execute()
        files = res.get("files", [])
        if files:
            folder_id = files[0]["id"]
        else:
            metadata = {
                "name": name,
                "mimeType": "application/vnd.google-apps.folder",
                "parents": [parent_id],
            }
            res = service.files().create(body=metadata, fields="id").execute()
            folder_id = res["id"]
        _folder_cache[cache_key] = folder_id
        return folder_id
    except Exception:
        logger.debug(
            "source_ledger_offhost.gdrive: folder ensure failed for %s under %s",
            name, parent_id, exc_info=True,
        )
        return None


def _root_folder_id(service) -> Optional[str]:
    """Resolve the top-level destination folder.

    Resolution order:
      1. ``LEDGER_GDRIVE_FOLDER_ID`` env var (operator override)
      2. Folder named ``AndrusAI-Ledgers`` under My Drive root,
         creating it on first run.
    """
    override = os.getenv("LEDGER_GDRIVE_FOLDER_ID", "").strip()
    if override:
        return override
    # "root" is Drive's alias for My Drive top level.
    return _ensure_folder(service, "root", _DEFAULT_ROOT_NAME)


def upload_kb_ledger(kb_name: str) -> OffhostUploadResult:
    """Upload today's incremental window for one KB to Google Drive."""
    result = OffhostUploadResult(ok=False, kb_name=kb_name, destination="gdrive")
    started = time.monotonic()

    if not _enabled():
        result.skipped = True
        result.error = "disabled"
        return result

    service = _get_drive_service()
    if service is None:
        result.skipped = True
        result.error = "google_workspace_not_configured"
        return result

    try:
        from googleapiclient.http import MediaInMemoryUpload  # type: ignore
    except Exception:
        result.skipped = True
        result.error = "googleapiclient_http_missing"
        return result

    try:
        payload, n_rows, start_off, end_off = incremental_payload(kb_name, "gdrive")
    except Exception as exc:
        result.error = f"payload_build_failed: {exc}"
        return result
    if not payload:
        result.skipped = True
        result.ok = True
        result.error = "no_new_rows"
        return result

    try:
        root = _root_folder_id(service)
        if root is None:
            result.error = "root_folder_unresolvable"
            return result
        # Materialise the path nesting.
        # AndrusAI-Ledgers / <host> / <kb> / source_ledger / <date>.jsonl.gz
        from app.memory.source_ledger_offhost.common import _host_tag
        host_folder = _ensure_folder(service, root, _host_tag())
        if host_folder is None:
            result.error = "host_folder_unresolvable"
            return result
        kb_folder = _ensure_folder(service, host_folder, kb_name)
        if kb_folder is None:
            result.error = "kb_folder_unresolvable"
            return result
        leaf_folder = _ensure_folder(service, kb_folder, "source_ledger")
        if leaf_folder is None:
            result.error = "leaf_folder_unresolvable"
            return result
        # Day-keyed filename.
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        filename = f"{today}.jsonl.gz"

        # Idempotent overwrite: search for existing file with the same
        # name in this folder; if present, update; else create.
        query = (
            f"name='{filename}' and '{leaf_folder}' in parents and trashed=false"
        )
        existing = service.files().list(q=query, fields="files(id,name)").execute()
        existing_files = existing.get("files", [])
        media = MediaInMemoryUpload(
            payload, mimetype="application/gzip", resumable=False,
        )
        if existing_files:
            file_id = existing_files[0]["id"]
            service.files().update(
                fileId=file_id, media_body=media,
            ).execute()
            object_key = f"gdrive://{file_id}/{filename}"
        else:
            metadata = {"name": filename, "parents": [leaf_folder]}
            res = service.files().create(
                body=metadata, media_body=media, fields="id",
            ).execute()
            file_id = res["id"]
            object_key = f"gdrive://{file_id}/{filename}"

        record_upload(kb_name, "gdrive", end_off, object_key)
        result.ok = True
        result.bytes_uploaded = len(payload)
        result.rows_uploaded = n_rows
        result.object_key = object_key
    except Exception as exc:
        result.error = f"unexpected: {type(exc).__name__}: {exc}"
    result.duration_s = time.monotonic() - started
    return result
