"""forge_bridge.py — sync Forge-generated tools INTO ToolRegistry.

Phase 3 closes the loop Forge was designed for. Today, Forge can
generate sandboxed tools and graduate them through SHADOW → CANARY →
ACTIVE, but agents have no way to *find* a Forge-generated tool —
``tool_search`` only knows about ``@register_tool``-annotated entries.

This module bridges Forge's Postgres-backed tool catalog into the
in-memory ``ToolRegistry``. Phase 3 is **read-only on Forge's side**:
Forge's state machine, schema, audit pipeline, and TIER_IMMUTABLE
files are untouched. The bridge READS from ``forge.registry.list_tools``
and writes to the in-memory registry.

How it works
------------

  1. ``sync_forge_tools()`` queries Forge for all tools in
     {SHADOW, CANARY, ACTIVE} status.
  2. Each tool is mapped to a ``ToolSpec`` with the corresponding
     ``Tier`` (mapping below).
  3. The spec's factory wraps ``forge.runtime.dispatcher.invoke_tool``
     in a CrewAI BaseTool — invoking the wrapper goes through Forge's
     killswitch + budget + audit gates exactly as before.
  4. Specs are added to the registry's side-table so they survive
     ``reset_for_tests()`` and replay correctly on subsequent boots.
  5. ``boot_registry`` calls ``sync_forge_tools()`` after the static
     ``@register_tool`` pass, so Forge tools appear in the catalog
     for ``tool_search`` and the ``/api/cp/tools`` endpoint.

Status → Tier mapping
---------------------
    Forge ToolStatus    → ToolRegistry.Tier
    DRAFT               → (NOT bridged; not yet validated)
    QUARANTINED         → (NOT bridged; failed audit)
    SHADOW              → Tier.SHADOW
    CANARY              → Tier.CANARY
    ACTIVE              → Tier.PRODUCTION
    DEPRECATED          → (NOT bridged; phasing out)
    KILLED              → (NOT bridged; explicitly killed)

The "NOT bridged" statuses simply don't appear in the in-memory
registry. They still exist in Forge's DB; they're just invisible to
agent discovery — which is the same as their current behavior.

Failure modes
-------------
* Forge disabled (``TOOL_FORGE_ENABLED`` unset / Postgres unreachable):
  ``sync_forge_tools()`` returns 0 silently. The registry has only
  static @register_tool entries. Phase 1a/1b/2 unchanged.
* A specific Forge tool can't be wrapped (e.g. its manifest has an
  unsupported source_type): logged, skipped, sync continues.
* The bridge is **never** in the import path of Forge's hot code —
  this module imports Forge lazily, so a Forge bug can't crash the
  registry boot.
"""
from __future__ import annotations

import logging
from typing import Any

from app.tool_registry.types import Lifecycle, Tier, ToolSpec

logger = logging.getLogger(__name__)


# Forge ToolStatus strings (mirror app/forge/manifest.py:ToolStatus enum).
# Kept as strings here to avoid importing forge.manifest at module load
# (forge import has heavy side-effects — DB connections, tier graduation
# subscriptions, etc.).
_BRIDGED_STATUSES: dict[str, Tier] = {
    "SHADOW": Tier.SHADOW,
    "CANARY": Tier.CANARY,
    "ACTIVE": Tier.PRODUCTION,
}


def _is_forge_enabled() -> bool:
    """Cheap check — is the Forge subsystem reachable? Avoids importing
    Forge until we know we'll actually use it."""
    try:
        import os
        if os.environ.get("TOOL_FORGE_ENABLED", "").strip() not in ("1", "true", "True"):
            return False
        # Try a lightweight import — if the module doesn't import, give up.
        import app.forge.registry  # noqa: F401
        return True
    except Exception as exc:
        logger.debug("forge_bridge: Forge unreachable (%s)", exc)
        return False


# ── BaseTool wrapper ────────────────────────────────────────────────


def _make_forge_basetool(tool_id: str, tool_name: str, tool_description: str):
    """Build a CrewAI BaseTool that proxies into ``forge.runtime.dispatcher.invoke_tool``.

    The wrapper preserves Forge's safety properties: every invocation
    goes through Forge's killswitch, budget check, capability audit,
    and SHADOW-tier result-discard semantics. The wrapper just hands
    the CrewAI tool-call interface a callable that ends up at Forge's
    front door.
    """
    try:
        from crewai.tools import BaseTool
        from pydantic import BaseModel, Field
        from typing import Type
    except ImportError:
        return None

    # Capture closure values into local names that don't shadow the
    # class body's own annotation targets (`name: str`, `description: str`).
    _captured_name = tool_name
    _captured_description = tool_description

    class _ForgeToolInput(BaseModel):
        params: dict[str, Any] = Field(
            default_factory=dict,
            description=(
                "JSON-shaped parameters for the Forge tool. Schema "
                "depends on the specific tool — see the tool's "
                "Forge manifest for the parameter contract."
            ),
        )

    class _ForgeProxyTool(BaseTool):
        name: str = _captured_name
        description: str = _captured_description
        args_schema: Type[BaseModel] = _ForgeToolInput

        def _run(self, params: dict[str, Any] | None = None) -> str:
            try:
                from app.forge.runtime.dispatcher import invoke_tool
            except Exception as exc:  # noqa: BLE001
                return f"forge_bridge ERROR: dispatcher unavailable ({exc})"

            try:
                result = invoke_tool(tool_id=tool_id, params=params or {})
            except Exception as exc:  # noqa: BLE001
                return (
                    f"forge_bridge ERROR: invoke_tool({tool_id}) raised "
                    f"{type(exc).__name__}: {exc}"
                )
            # Dispatcher returns a dict; format it as a single string
            # for the LLM. SHADOW-tier outputs land in shadow_result —
            # surfaced separately so the agent knows what mode they got.
            if not isinstance(result, dict):
                return str(result)
            if not result.get("ok", False):
                return (
                    f"forge_bridge: invocation refused — "
                    f"{result.get('error', 'unknown reason')}"
                )
            mode = result.get("mode", "production")
            if mode == "SHADOW" or "shadow_result" in result:
                # Shadow mode: result is logged but withheld from caller.
                return (
                    f"forge_bridge: SHADOW-tier execution OK. Result was "
                    f"computed and logged for operator review, but not "
                    f"returned to the agent (per SHADOW-tier semantics). "
                    f"elapsed_ms={result.get('elapsed_ms')}, "
                    f"capability_used={result.get('capability_used')}"
                )
            return str(result.get("result", result))

    return _ForgeProxyTool


def _make_factory(tool_id: str, tool_name: str, tool_description: str):
    """Closure that builds a fresh BaseTool instance on registry
    ``build_instance`` calls. SINGLETON lifecycle in the registry —
    so each Forge tool gets one BaseTool instance per process."""
    def factory():
        cls = _make_forge_basetool(tool_id, tool_name, tool_description)
        if cls is None:
            raise RuntimeError(f"forge_bridge: cannot build BaseTool for {tool_id}")
        return cls()
    factory.__name__ = f"_forge_factory_{tool_name}"
    return factory


# ── Public API ──────────────────────────────────────────────────────


def sync_forge_tools() -> int:
    """Query Forge's DB and bring eligible tools into the in-memory
    ``ToolRegistry``. Returns the count of tools currently in the
    bridge's authority (regardless of whether they changed this call).

    Behavior:
      * NEW Forge tool → registered in the registry at the right tier.
      * EXISTING Forge tool with same status → no-op (idempotent).
      * Tier-change (SHADOW → CANARY etc.) → ``replace_spec`` updates
        the registry's spec to the new tier; cached instance dropped.
      * Tool DISAPPEARED from Forge (KILLED/DEPRECATED) → unregistered
        from the in-memory registry.

    The "currently in the bridge's authority" return tracks how many
    Forge tools exist in our registry after the sync — useful for
    boot-time logging.
    """
    if not _is_forge_enabled():
        return 0

    try:
        from app.forge.registry import list_tools
    except Exception as exc:
        logger.debug("forge_bridge: cannot import forge.registry (%s)", exc)
        return 0

    from app.tool_registry.registry import ToolRegistry
    registry = ToolRegistry.instance()

    # Track which Forge-bridged names exist in the current Forge DB so
    # we can detect removals (tools that disappeared since last sync).
    current_bridged: set[str] = set()
    new_or_updated = 0

    for forge_status, our_tier in _BRIDGED_STATUSES.items():
        try:
            rows = list_tools(status=forge_status, limit=500)
        except Exception as exc:
            logger.warning(
                "forge_bridge: list_tools(status=%s) failed: %s",
                forge_status, exc,
            )
            continue

        for row in rows:
            try:
                spec = _row_to_spec(row, tier=our_tier)
            except Exception as exc:
                logger.warning(
                    "forge_bridge: skipping forge tool %s (%s)",
                    row.get("tool_id"), exc,
                )
                continue
            current_bridged.add(spec.name)
            existing = registry.get(spec.name)
            if existing is None:
                registry.register(spec)
                new_or_updated += 1
            elif existing.tier != spec.tier or \
                 existing.description_hash != spec.description_hash:
                # Status / description changed — replace.
                registry.replace_spec(spec)
                new_or_updated += 1
            # Else: identical, no-op.

    # Detect removals: tools that were bridged before but no longer
    # appear in {SHADOW, CANARY, ACTIVE}.
    previously_bridged = {
        s.name for s in registry.all()
        if s.source_module.startswith("app.forge.tools.")
    }
    removed = previously_bridged - current_bridged
    for name in removed:
        registry.unregister(name)

    total_bridged = len(current_bridged)
    if new_or_updated or removed:
        logger.info(
            "forge_bridge: %d new/updated Forge tools, %d removed; "
            "total bridged: %d",
            new_or_updated, len(removed), total_bridged,
        )
    return total_bridged


# ── Periodic reconciliation ──────────────────────────────────────────


_RECONCILE_INTERVAL_SEC = 300  # 5 minutes


async def reconciliation_loop(interval_sec: int = _RECONCILE_INTERVAL_SEC) -> None:
    """Run ``sync_forge_tools`` on a periodic timer.

    Picks up Forge tier transitions (SHADOW → CANARY, demotions,
    KILLED tools disappearing) without an explicit pub/sub channel
    from Forge. 5-minute cadence is the soak baseline — at 300s we
    pay one Postgres round-trip per period when Forge is enabled and
    nothing when it's not.

    Spawned at gateway startup via ``asyncio.create_task``. Survives
    transient Postgres outages because each iteration is independent
    and the bridge's failure modes are non-fatal.

    Cancelable: standard asyncio task cancellation. The loop catches
    CancelledError and exits cleanly.
    """
    import asyncio
    logger.info(
        "forge_bridge: reconciliation loop starting (interval=%ds)",
        interval_sec,
    )
    while True:
        try:
            await asyncio.sleep(interval_sec)
        except asyncio.CancelledError:
            logger.info("forge_bridge: reconciliation loop cancelled")
            return
        try:
            await asyncio.to_thread(sync_forge_tools)
        except Exception as exc:  # noqa: BLE001
            logger.debug("forge_bridge: reconciliation iteration failed: %s", exc)


def _row_to_spec(row: dict[str, Any], *, tier: Tier) -> ToolSpec:
    """Convert one Forge tool row into a ToolSpec.

    Capability tags: Forge has its own ``Capability`` enum (manifest.py)
    — they're declarative descriptors of the tool's syscall surface
    (``http.lan``, ``fs.workspace.read``, etc), not the same vocabulary
    as our registry. For Phase 3, we register Forge tools under the
    bounded vocabulary's ``governance.registers-tool`` (closest fit) —
    they're discoverable as a class but ranked by intent/embedding.
    Phase 4+ may extend the vocabulary with Forge-shaped tags.
    """
    tool_id = row["tool_id"]
    name = row.get("name") or tool_id
    description = (row.get("description") or "").strip()
    if not description:
        description = (
            f"Forge-generated tool ({tier.value} tier). "
            f"tool_id={tool_id}. No description on the manifest."
        )

    return ToolSpec(
        name=name,
        capabilities=("registers-tool",),
        tier=tier,
        lifecycle=Lifecycle.SINGLETON,
        description=description,
        args_schema=None,           # Built lazily by the BaseTool factory
        factory=_make_factory(tool_id, name, description),
        guard=lambda: True,          # Forge enforces its own gates at invoke time
        workspace_scope=("*",),
        source_module=f"app.forge.tools.{name}",
    )
