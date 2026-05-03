"""Tool registry — Phase 1a foundation.

Public API::

    from app.tool_registry import register_tool, ToolRegistry, Tier, Lifecycle

The decorator + registry singleton is the source of truth for "what
tools exist". Phase 1a is purely additive: annotating a tool with
``@register_tool`` does not change how that tool is called by existing
agent factories. The registry just *also* knows about it.

Phase 1b adds ``tool_search``; Phase 2 wires it into LoadableAgent.
See ``docs/TOOL_REGISTRY.md`` for the full architecture and
``docs/TOOL_REGISTRY_PHASE_0.md`` for the spike report that motivated
this design.
"""
from app.tool_registry.decorator import register_tool
from app.tool_registry.registry import ToolRegistry
from app.tool_registry.types import Lifecycle, Tier, ToolSpec

__all__ = [
    "register_tool",
    "ToolRegistry",
    "Tier",
    "Lifecycle",
    "ToolSpec",
]
