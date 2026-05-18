"""system_inventory — auto-generated catalogue of the codebase's subsystems.

Closes the meta-gap behind the May 2026 ultrathink: the system's self-
documentation (CLAUDE.md) drifted from its actual capabilities. Modules
like ``code_quality``, ``architectural_review``, ``differential_test``,
``cascade_evaluator``, ``canary_deploy``, ``evolution_suite`` exist and
work, but agents reasoning over CLAUDE.md don't know they're there.

The inventory is the *live truth* — re-built weekly from the filesystem
plus a few capability signals (``@register_tool`` decorators, healing-
monitor registrations, change-request requestor strings). CLAUDE.md
becomes a stable narrative; the inventory is the source-of-truth
catalogue agents query at runtime.

Public API
----------
``build_snapshot()``
    Walk ``app/``, return a fresh ``InventorySnapshot``.

``get_snapshot()``
    Read the most-recent persisted snapshot (build one if absent).

``query_inventory(*, kind=None, capability=None, keyword=None)``
    Filter modules by ``kind`` (``package``/``module``), a capability tag
    drawn from registered tools, or a free-text substring against the
    module name or docstring summary.

``inventory_summary()``
    One-line counts + the top-level shape, suitable for prompts.

Master switch: ``runtime_settings.system_inventory_enabled`` (default ON).
"""
from __future__ import annotations

from app.system_inventory.scanner import build_snapshot, InventorySnapshot, ModuleEntry
from app.system_inventory.store import (
    get_snapshot,
    inventory_summary,
    persist_snapshot,
    query_inventory,
)

__all__ = [
    "InventorySnapshot",
    "ModuleEntry",
    "build_snapshot",
    "get_snapshot",
    "inventory_summary",
    "persist_snapshot",
    "query_inventory",
]
