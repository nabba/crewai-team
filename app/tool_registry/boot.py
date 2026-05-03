"""Boot scan — import every tool module so decorators fire.

The decorator side-effects ToolRegistry only when the module that
contains it is imported. Phase 1a needs the gateway to call
``boot_registry()`` exactly once at startup (after settings and
logging are wired) so every annotated tool is registered before any
agent is created.

Design choices
--------------
* **Best-effort imports.** A single broken tool module must not
  crash the gateway. Each import is in a try/except; failures are
  logged with the module name and the registry continues.
* **Idempotent.** Calling ``boot_registry()`` twice is fine — the
  registry singleton's ``register`` is idempotent on identical specs.
* **Snapshot + drift after import.** Once all imports complete, we
  snapshot the registry to Postgres and run the drift detector. Both
  are non-fatal; missing Postgres just skips them.

Module-discovery strategy
-------------------------
Tools live across multiple packages. We don't ``walk_packages`` blindly
because some sub-packages have heavy side-effects on import (Forge
generators, Mem0 clients). Instead we maintain an explicit list of
*tool-module roots* — packages where ``@register_tool``-decorated
factories live. Adding a new tool sub-package = adding it to
``TOOL_MODULE_ROOTS`` here.
"""
from __future__ import annotations

import importlib
import logging
import pkgutil

from app.tool_registry.drift import detect_drift, log_drift
from app.tool_registry.persistence import snapshot
from app.tool_registry.registry import ToolRegistry

logger = logging.getLogger(__name__)


# Add a package here when a new tool-bearing module is created.
# Boot will recursively import every module under each entry so
# every @register_tool decorator fires exactly once.
TOOL_MODULE_ROOTS: tuple[str, ...] = (
    "app.tools",
    # Future Phase 1a/1b additions when the corresponding tools get
    # decorated. Listed up-front so adding the decorator is the only
    # remaining step:
    # "app.knowledge_base.tools",
    # "app.aesthetics.tools",
    # "app.experiential.tools",
    # "app.tensions.tools",
    # "app.philosophy",
)


def _import_subpackage(root_name: str) -> int:
    """Recursively import every module in ``root_name``. Returns the
    count of modules successfully imported."""
    try:
        root = importlib.import_module(root_name)
    except Exception as exc:
        logger.warning("tool_registry boot: cannot import root %s: %s", root_name, exc)
        return 0

    if not hasattr(root, "__path__"):
        return 1  # not a package, just a module

    count = 0
    for mod_info in pkgutil.walk_packages(root.__path__, prefix=f"{root_name}."):
        try:
            importlib.import_module(mod_info.name)
            count += 1
        except Exception as exc:
            # Decorator validation errors (unknown capability) get
            # surfaced loudly — they're a configuration bug.
            if isinstance(exc, ValueError) and "register_tool" in str(exc):
                logger.error("tool_registry boot: %s — %s", mod_info.name, exc)
            else:
                logger.debug(
                    "tool_registry boot: skipped %s (%s)", mod_info.name, exc,
                )
    return count


def boot_registry(
    *,
    snapshot_to_postgres: bool = True,
    index_to_chromadb: bool = True,
    sync_forge: bool = True,
) -> ToolRegistry:
    """One-shot boot: import every tool module, snapshot, detect drift,
    re-index for semantic search, sync Forge-generated tools.

    Returns the populated singleton. Safe to call multiple times.

    Args:
        snapshot_to_postgres: Mirror the registry to a Postgres table
            (Phase 1a). Non-fatal on DB failure.
        index_to_chromadb: Build / refresh the ChromaDB collection for
            ``tool_search`` (Phase 1b). Idempotent — re-embeds only
            tools whose description_hash changed. Non-fatal on
            ChromaDB / embed-service failure.
        sync_forge: Pull Forge-generated tools (SHADOW/CANARY/ACTIVE)
            into the in-memory registry (Phase 3). No-op when Forge
            is disabled or unreachable.
    """
    registry = ToolRegistry.instance()
    pre_count = len(registry.all())
    total_imported = 0
    for root in TOOL_MODULE_ROOTS:
        total_imported += _import_subpackage(root)

    # Replay any decorator-produced specs that weren't picked up
    # because their modules were already in sys.modules. This is the
    # path that test setups + multi-import scenarios rely on.
    replayed = registry.replay_decorations()

    # Phase 3: bridge Forge-generated tools into the registry. Each
    # eligible Forge tool (SHADOW / CANARY / ACTIVE) becomes a
    # ToolSpec with its corresponding Tier. No-op when Forge is
    # disabled — the in-memory registry stays at the static set.
    forge_synced = 0
    if sync_forge:
        try:
            from app.tool_registry.forge_bridge import sync_forge_tools
            forge_synced = sync_forge_tools()
        except Exception as exc:
            logger.debug(
                "tool_registry boot: Forge sync failed (%s) — continuing", exc,
            )

    specs = registry.all()
    new_specs = len(specs) - pre_count
    logger.info(
        "tool_registry boot: imported %d modules across %d roots; "
        "replayed %d cached decorator specs; bridged %d Forge tools; "
        "total %d (was %d).",
        total_imported, len(TOOL_MODULE_ROOTS),
        replayed, forge_synced, len(specs), pre_count,
    )

    if snapshot_to_postgres:
        # Drift detection BEFORE writing the new snapshot — we compare
        # current registry against the *prior* snapshot.
        drift = detect_drift(specs)
        log_drift(drift)
        snapshot(specs)

    if index_to_chromadb:
        # Phase 1b — semantic-search index for tool_search.
        try:
            from app.tool_registry.indexer import index_tools
            reindexed, skipped = index_tools(specs)
            if reindexed:
                logger.info(
                    "tool_registry boot: ChromaDB index updated "
                    "(re-embedded %d, skipped %d unchanged).",
                    reindexed, skipped,
                )
        except Exception as exc:
            logger.debug(
                "tool_registry boot: ChromaDB indexing skipped (%s)", exc,
            )

    return registry
