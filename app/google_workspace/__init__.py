"""
app.google_workspace — OAuth-backed access to Gmail, Calendar, Docs, Sheets, Slides.

The package owns three concerns:

  - OAuth installed-app flow + refresh-token storage (auth.py).
  - Lazy-built `googleapiclient.discovery.Resource` instances per API,
    cached so each tool call doesn't repeat the HTTP discovery doc fetch
    (service.py).
  - One-time bootstrap CLI that walks the operator through consent
    (bootstrap.py — runnable as ``python -m app.google_workspace.bootstrap``).

The five tool-family modules under ``app.tools`` (gmail_tools.py,
gcal_tools.py, gdocs_tools.py, gsheets_tools.py, gslides_tools.py) all
build their CrewAI tools on top of this package.

When the operator hasn't run the bootstrap, ``get_credentials()`` returns
None and every tool factory degrades to an empty list — the agent will
simply not see Google tools, but startup never crashes.
"""
from __future__ import annotations

from app.google_workspace.auth import (
    get_credentials,
    is_configured,
    SCOPES,
    TOKEN_PATH,
)
from app.google_workspace.service import get_service, clear_service_cache

__all__ = [
    "get_credentials", "is_configured", "SCOPES", "TOKEN_PATH",
    "get_service", "clear_service_cache",
]
