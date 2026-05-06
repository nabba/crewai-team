"""factory.py — registry-backed LoadableAgent construction.

Phase 2 promotes the Phase 0 spike from "static available_tools dict"
to "registry-driven discoverable catalog". An agent declares the
capabilities it always needs (core) + the capabilities it might want
(discoverable), and the factory wires it up against ``ToolRegistry``.

The factory is **hybrid**: it accepts both raw tool instances (for
agent-specific tools that aren't yet annotated, like per-agent memory
tools) AND capability tags (for the catalog-driven path). This keeps
Phase 2 incremental — pilot agents can migrate in stages without
requiring a sweeping annotation pass first.

Public API
----------

    build_loadable_agent(
        role="Introspector",
        goal="...",
        backstory="...",
        llm=llm,
        agent_id="introspector",
        core_tools=[memory_tools, scoped_tools, ...],   # eager, agent-specific
        core_capabilities=["reads-file"],               # eager, registry-resolved
        discoverable_capabilities=["renders-pdf",       # lazy, registry-resolved
                                   "searches-web"],
        agent_tier=Tier.PRODUCTION,
        workspace=None,
    ) → LoadableAgent

Resolution order:
  1. ``core_tools`` — passed directly to LoadableAgent's binder.
  2. ``core_capabilities`` — registry resolves matching tool specs;
     each spec's factory is invoked once, instances added to core.
  3. ``discoverable_capabilities`` + ``discoverable_names`` —
     registry collects matching specs; their factories are wrapped
     in lazy callables and added to ``available_tools``. Agent only
     pays for instantiation when it loads.

The agent gets ``load_tool`` + ``list_available_tools`` automatically
from LoadableAgent's superclass.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Iterable

from app.tool_registry import Tier, ToolRegistry
from app.tool_runtime.loadable_agent import LoadableAgent

logger = logging.getLogger(__name__)


def _resolve_capability_specs(
    *,
    capabilities: Iterable[str],
    workspace: str | None,
    agent_tier: Tier,
) -> list:
    """Pull all registry specs matching any of ``capabilities``,
    after applying the standard tier + workspace gates."""
    registry = ToolRegistry.instance()
    seen_names: set[str] = set()
    out: list = []
    for tag in capabilities:
        for spec in registry.by_capability(tag):
            if spec.name in seen_names:
                continue
            # Tier gate: tool's tier must be ≥ agent's authorization
            if not _tier_passes(spec.tier, agent_tier):
                continue
            # Workspace gate
            if workspace is not None and "*" not in spec.workspace_scope and \
               workspace not in spec.workspace_scope:
                continue
            seen_names.add(spec.name)
            out.append(spec)
    return out


def _tier_passes(spec_tier: Tier, agent_tier: Tier) -> bool:
    """Match the discovery layer's tier gate (see discovery.py)."""
    rank = {Tier.SHADOW: 0, Tier.CANARY: 1, Tier.PRODUCTION: 2, Tier.IMMUTABLE: 3}
    return rank[spec_tier] >= rank[agent_tier]


def _make_lazy_factory(spec, agent_id: str) -> Callable[[], Any]:
    """Wrap a registry spec's factory in a no-arg callable that the
    LoadableAgent binder can call on demand."""
    def factory() -> Any:
        return ToolRegistry.instance().build_instance(spec.name, agent_id=agent_id)
    factory.__name__ = f"_lazy_{spec.name}"
    return factory


def build_loadable_agent(
    *,
    role: str,
    goal: str,
    backstory: str,
    llm: Any,
    agent_id: str,
    core_tools: list[Any] | None = None,
    core_capabilities: Iterable[str] | None = None,
    discoverable_capabilities: Iterable[str] | None = None,
    discoverable_names: Iterable[str] | None = None,
    agent_tier: Tier = Tier.PRODUCTION,
    workspace: str | None = None,
    **agent_kwargs: Any,
) -> LoadableAgent:
    """Build a registry-backed LoadableAgent.

    See module docstring for the resolution order and rationale.
    Returns a fully-constructed LoadableAgent ready for execution.
    """
    registry = ToolRegistry.instance()
    eager_tools: list[Any] = list(core_tools or [])

    # Eager-load tools matching core_capabilities.
    if core_capabilities:
        for spec in _resolve_capability_specs(
            capabilities=core_capabilities,
            workspace=workspace,
            agent_tier=agent_tier,
        ):
            try:
                instance = registry.build_instance(spec.name, agent_id=agent_id)
                eager_tools.append(instance)
            except Exception as exc:
                logger.warning(
                    "build_loadable_agent: skipping core tool %r (%s)",
                    spec.name, exc,
                )

    # Discoverable lazy tools: by capability tag + optional explicit name list.
    available: dict[str, Callable[[], Any]] = {}
    eager_names = {t.name for t in eager_tools if hasattr(t, "name")}

    if discoverable_capabilities:
        for spec in _resolve_capability_specs(
            capabilities=discoverable_capabilities,
            workspace=workspace,
            agent_tier=agent_tier,
        ):
            if spec.name in eager_names:
                continue  # already loaded eagerly
            available[spec.name] = _make_lazy_factory(spec, agent_id)

    if discoverable_names:
        for name in discoverable_names:
            if name in available or name in eager_names:
                continue
            spec = registry.get(name)
            if spec is None:
                logger.warning(
                    "build_loadable_agent: discoverable name %r not in registry — "
                    "skipping. Has it been annotated yet?",
                    name,
                )
                continue
            if not _tier_passes(spec.tier, agent_tier):
                continue
            if workspace is not None and "*" not in spec.workspace_scope and \
               workspace not in spec.workspace_scope:
                continue
            available[name] = _make_lazy_factory(spec, agent_id)

    logger.info(
        "build_loadable_agent[%s]: %d core tools, %d discoverable.",
        agent_id, len(eager_tools), len(available),
    )

    agent = LoadableAgent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=llm,
        core_tools=eager_tools,
        available_tools=available,
        **agent_kwargs,
    )

    # Wire cache-aware telemetry so each LLM call writes a row to
    # workspace/observability/loadable_agent_usage.jsonl. Phase 5 acceptance
    # criterion 3 (token economics) is validated against this log.
    try:
        from app.tool_runtime.telemetry import install_cache_telemetry
        install_cache_telemetry(agent, agent_id)
    except Exception as exc:
        logger.warning(
            "build_loadable_agent[%s]: telemetry install failed (%s) — "
            "agent runs unchanged, but token economics will not be observable.",
            agent_id, exc,
        )

    return agent
