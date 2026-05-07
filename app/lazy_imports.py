"""
lazy_imports.py — Cached singleton accessors.

Replaces the runtime `__import__("app.config", fromlist=["get_settings"])
.get_settings()` pattern scattered through idle_scheduler.py,
orchestrator.py, healing/health_remediator.py, and consciousness modules.

Those patterns are a DIY workaround for circular imports. The cost is
a real import on every call — 3× per idle loop in one case — and
the circular dependency is never actually resolved, just deferred.

This module provides:
  - An explicit top-level import surface (no __import__ string literals)
  - @lru_cache so the underlying module is resolved once per process
  - A documented list of the dependencies that were previously hidden

Usage:
    from app.lazy_imports import settings
    s = settings()

If you find yourself adding a new entry here, prefer fixing the circular
import first. Add to this module only when a genuine cycle forces it.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def settings() -> Any:
    """Return app.config.get_settings() result, cached for the process lifetime."""
    from app.config import get_settings
    return get_settings()
