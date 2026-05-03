"""@register_tool — the only public way to enter the registry.

Module-level decorators fire once per Python process: importlib caches
modules in ``sys.modules``, so re-importing doesn't re-run the body
and doesn't re-fire decorators. That's a problem for tests that want
a clean ToolRegistry singleton — ``reset_for_tests()`` would empty
the registry permanently because the decorators won't run again.

We solve this by keeping a *side-table* (``_DECORATED_SPECS``) of every
spec ever produced by ``@register_tool``. The side-table is populated
at decoration time and never cleared; ``ToolRegistry.replay_decorations()``
copies it back into the current singleton on demand. So:

  * Production: decorator runs at boot → spec lands in singleton + side-table.
  * Test isolation: ``reset_for_tests()`` empties singleton → tests can
    rebuild via ``replay_decorations()``.

Usage::

    from app.tool_registry import register_tool, Tier, Lifecycle

    @register_tool(
        name="pdf_compose",
        capabilities=["renders-pdf", "renders-chart"],
        tier=Tier.PRODUCTION,
        lifecycle=Lifecycle.SINGLETON,
        description="Render a PDF report from data...",
        args_schema=PdfComposeInput,
        guard=lambda: True,
    )
    def pdf_compose_factory(agent_id: str = "coder"):
        return PdfComposeTool()

The decorator is **passive on the function**: it does not wrap or
alter ``pdf_compose_factory``. It only side-effects the global
``ToolRegistry`` singleton. Existing call sites that use the factory
directly continue to work unchanged. This is how the migration stays
zero-risk in Phase 1a — annotating a tool does NOT break the legacy
import path.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Iterable

from app.tool_registry.capabilities import (
    DEPRECATED_CAPABILITIES,
    all_capability_tags,
)
from app.tool_registry.registry import ToolRegistry
from app.tool_registry.types import Lifecycle, Tier, ToolSpec

logger = logging.getLogger(__name__)


# Side-table: every spec ever produced by @register_tool, keyed by name.
# Survives ToolRegistry.reset_for_tests(); used by replay_decorations()
# in test setups where we want a fresh singleton without re-importing
# every tool module.
_DECORATED_SPECS: dict[str, ToolSpec] = {}


def _get_decorated_specs() -> dict[str, ToolSpec]:
    """Read access to the side-table — used by ToolRegistry.replay_decorations."""
    return dict(_DECORATED_SPECS)


def _validate_capabilities(name: str, capabilities: Iterable[str]) -> tuple[str, ...]:
    """Reject unknown tags; warn (don't reject) deprecated ones.

    The registry's promise is that every active tag is from the
    bounded vocabulary. If a tool tries to declare a typo or a tag
    we never approved, we fail loudly here at import time — better
    a startup error than a silent discovery miss.
    """
    valid_tags = all_capability_tags()
    cleaned: list[str] = []
    for tag in capabilities:
        if tag in valid_tags:
            cleaned.append(tag)
            continue
        if tag in DEPRECATED_CAPABILITIES:
            logger.warning(
                "register_tool: tool %r declares deprecated capability %r — %s",
                name, tag, DEPRECATED_CAPABILITIES[tag],
            )
            cleaned.append(tag)
            continue
        raise ValueError(
            f"register_tool: tool {name!r} declared unknown capability "
            f"{tag!r}. Add it to app/tool_registry/capabilities.py first "
            f"(governance-grade — requires PR review)."
        )
    if not cleaned:
        raise ValueError(
            f"register_tool: tool {name!r} must declare at least one capability."
        )
    return tuple(cleaned)


def register_tool(
    *,
    name: str,
    capabilities: Iterable[str],
    description: str,
    args_schema: type | None = None,
    factory: Callable[..., Any] | None = None,
    tier: Tier = Tier.PRODUCTION,
    lifecycle: Lifecycle = Lifecycle.SINGLETON,
    guard: Callable[[], bool] | None = None,
    workspace_scope: Iterable[str] = ("*",),
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that registers a tool factory in the global registry.

    Args:
        name: Tool name. Must be unique across the registry.
        capabilities: One or more tags from
            ``app/tool_registry/capabilities.py``.
        description: LLM-facing description. Becomes the tool's
            BaseTool.description after loading. Same prose-discipline
            as today (worked examples + anti-patterns). The hash is
            recomputed on every boot for drift detection.
        args_schema: Pydantic model defining the tool's call args.
            Optional — some tools have no args.
        factory: Callable that returns a BaseTool instance. If omitted,
            the decorated callable IS the factory.
        tier: Default PRODUCTION; Forge-generated tools come in as
            SHADOW.
        lifecycle: Default SINGLETON.
        guard: Callable returning True iff the tool's runtime
            environment is wired up (env vars set, etc). If guard()
            returns False, the tool is visible in the registry but
            cannot be loaded by an agent.
        workspace_scope: Workspace IDs the tool is allowed in, or
            ``("*",)`` for all workspaces.

    Returns:
        The decorator. Applied to the factory function.
    """
    capabilities_tuple = _validate_capabilities(name, capabilities)
    workspace_tuple = tuple(workspace_scope)
    if not workspace_tuple:
        workspace_tuple = ("*",)

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        chosen_factory = factory if factory is not None else fn
        spec = ToolSpec(
            name=name,
            capabilities=capabilities_tuple,
            tier=tier,
            lifecycle=lifecycle,
            description=description,
            args_schema=args_schema,
            factory=chosen_factory,
            guard=guard or (lambda: True),
            workspace_scope=workspace_tuple,
            source_module=getattr(fn, "__module__", "<unknown>"),
        )
        # Store in side-table FIRST so reset_for_tests + replay can repopulate.
        _DECORATED_SPECS[name] = spec
        ToolRegistry.instance().register(spec)
        return fn  # passive — leave the factory untouched

    return decorator
