"""
google_workspace.service — cached `googleapiclient.discovery.Resource` builder.

`get_service(api, version)` returns the Google API client for the requested
service ("gmail", "calendar", "docs", "sheets", "slides", "drive"), built
once per process and reused across tool calls so the discovery-doc HTTP
fetch only happens once.

Returns None when:
  - the operator has not run the bootstrap (no credentials)
  - google-api-python-client is not installed (import error)

Tool factories then degrade to an empty list rather than crashing.
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

from app.google_workspace.auth import get_credentials

logger = logging.getLogger(__name__)

_DEFAULT_VERSIONS = {
    "gmail": "v1",
    "calendar": "v3",
    "docs": "v1",
    "sheets": "v4",
    "slides": "v1",
    "drive": "v3",
}

_lock = threading.Lock()
_cache: dict[tuple[str, str], object] = {}


def get_service(api: str, version: Optional[str] = None):
    """Return a cached Google API client, or None if unavailable."""
    api = api.lower()
    version = version or _DEFAULT_VERSIONS.get(api)
    if not version:
        logger.warning(f"google_workspace.get_service: unknown api {api!r}")
        return None

    cache_key = (api, version)
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    creds = get_credentials()
    if creds is None:
        return None

    try:
        from googleapiclient.discovery import build
    except ImportError:
        logger.debug("google_workspace: googleapiclient not installed")
        return None

    with _lock:
        cached = _cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            service = build(api, version, credentials=creds, cache_discovery=False)
        except Exception as exc:
            logger.warning(f"google_workspace: build {api}/{version} failed: {exc}")
            return None
        _cache[cache_key] = service
        return service


def clear_service_cache() -> None:
    """Drop all cached service clients (used by tests + after token re-bootstrap)."""
    with _lock:
        _cache.clear()
