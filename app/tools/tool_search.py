"""tool_search — agent-callable discovery primitive.

Phase 1b. Lets an agent ask "what tools exist for capability X /
intent Y?" before committing to a workflow. Read-only — does not
load anything (Phase 2 adds auto-load).

Pattern of use::

    # Agent doesn't know which tool produces a PDF report:
    tool_search(intent="render Estonia forest report PDF")
      → ranked: pdf_compose (0.92), generate_pdf (0.71), …

    # Agent knows the capability tag:
    tool_search(capabilities=["renders-pdf", "renders-chart"])
      → ranked by exact-tag match score + tier rank

    # Combination — both signals contribute:
    tool_search(
        intent="forest report PDF",
        capabilities=["renders-pdf"],
    )

The tool surfaces itself to agents as a CrewAI BaseTool. It's added
to the coder + writer agents alongside their existing toolsets in
Phase 1b — purely additive, agents can use it or ignore it.

Why it ranks instead of just listing
------------------------------------
A flat list of all 11+ tools would push the agent back to surface-
keyword matching ("hmm, this tool name has 'pdf' in it"). Ranked
output with a one-line `reason` per match is what teaches the agent
to trust the registry's signal: "matches capability renders-pdf;
semantic match d=0.31; tier=production". The reason field is the
LLM's epistemic ground truth for the ranking.

Why it doesn't auto-load (yet)
------------------------------
Auto-load requires LoadableAgent — the executor needs to re-render
its tool schemas mid-iteration when a new tool is added. That's
Phase 2's promotion. For now, the agent can call ``tool_search`` to
discover, then either use a tool that's *already* in its inventory
(common case in Phase 1b: coder ships with all 11 annotated tools)
or know that one isn't reachable in this run.
"""
from __future__ import annotations

import logging
from typing import Type

logger = logging.getLogger(__name__)


# ── Tool registry annotation (Phase 1b) ─────────────────────────────
# tool_search is itself a tool — and itself goes through the registry.
# Self-referential in a useful way: agents can `tool_search` for
# "find a tool" and surface this very tool. We tag it under
# `governance` since it's a system-introspection capability.

try:
    from app.tool_registry import Lifecycle, Tier, register_tool

    @register_tool(
        name="tool_search",
        capabilities=["registers-tool"],   # closest existing tag
        description=(
            "Search the tool registry by capability and/or intent. "
            "Returns ranked candidates with a short reason for each "
            "ranking. Use this BEFORE assuming a tool exists or "
            "doesn't — the registry is the source of truth."
        ),
        tier=Tier.PRODUCTION,
        lifecycle=Lifecycle.SINGLETON,
        workspace_scope=("*",),
    )
    def _tool_search_registry_factory():
        return ToolSearchTool()
except ImportError:
    pass


# ── BaseTool implementation ─────────────────────────────────────────


def _build_tool_search_class():
    """Build the ToolSearchTool class lazily so import-time failures
    in crewai/pydantic don't poison the module."""
    from crewai.tools import BaseTool
    from pydantic import BaseModel, Field

    class _ToolSearchInput(BaseModel):
        intent: str = Field(
            default="",
            description=(
                "Natural-language description of what you want to do. "
                "E.g. 'render a forest report PDF', 'send a Signal "
                "message with attachments', 'fetch satellite forest-"
                "loss data'. Short queries (<3 words) are treated as "
                "noise — pair them with `capabilities=[...]` instead."
            ),
        )
        capabilities: list[str] = Field(
            default_factory=list,
            description=(
                "Optional list of capability tags from the bounded "
                "vocabulary (see /api/cp/tools/capabilities for the "
                "full list). Examples: 'renders-pdf', 'sends-signal', "
                "'reads-satellite'. When provided, tools declaring "
                "any of these tags get a ranking boost; combined "
                "with `intent` for hybrid ranking."
            ),
        )
        limit: int = Field(
            default=5,
            description=(
                "Max number of ranked candidates to return. Default "
                "5; max useful is ~10."
            ),
        )

    class ToolSearchTool(BaseTool):
        name: str = "tool_search"
        description: str = (
            "Search the tool registry for tools matching a capability "
            "or intent. Returns a ranked list of candidates — name, "
            "tier, capabilities, a one-line reason for the ranking, "
            "and the tool's own description.\n\n"
            "USE THIS when:\n"
            "  - You're not sure if a tool exists for what you need.\n"
            "  - You need a capability and want to see all tools "
            "that provide it.\n"
            "  - You're deciding between two similar tools.\n\n"
            "Pattern:\n"
            "  tool_search(\n"
            "      intent='render a forest deforestation PDF',\n"
            "      capabilities=['renders-pdf', 'renders-chart'],\n"
            "  )\n\n"
            "The result tells you what's possible. In Phase 2 you'll "
            "be able to load the suggested tools dynamically; for "
            "now this is a pure-discovery tool — use it to plan, "
            "then use the tool from your existing inventory."
        )
        args_schema: Type[BaseModel] = _ToolSearchInput

        def _run(self, intent: str = "", capabilities: list[str] | None = None,
                 limit: int = 5) -> str:
            from app.tool_registry import Tier
            from app.tool_registry.discovery import search_tools

            try:
                matches = search_tools(
                    intent=intent or "",
                    capabilities=capabilities or [],
                    workspace=None,    # Phase 1b: no workspace propagation yet
                    agent_tier=Tier.PRODUCTION,  # safe default
                    limit=max(1, min(limit, 10)),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("tool_search: discovery failed: %s", exc)
                return f"tool_search ERROR: {type(exc).__name__}: {exc}"

            if not matches:
                # Subjectless intent + no caps → empty (the 4-layer defense)
                return (
                    "No matching tools.\n\n"
                    "If your intent was very short ('go', 'run it'), "
                    "try a more specific phrase or pass `capabilities` "
                    "(see /api/cp/tools/capabilities for the vocabulary)."
                )

            lines = [f"Found {len(matches)} matching tool(s) (highest-ranked first):"]
            for m in matches:
                lines.append("")
                lines.append(f"  {m.name}  [{m.spec.tier.value}]  score={m.score:.2f}")
                lines.append(f"    capabilities: {list(m.spec.capabilities)}")
                lines.append(f"    why: {m.reason}")
                # Truncate long descriptions to keep the response readable.
                desc = m.spec.description
                if len(desc) > 220:
                    desc = desc[:217] + "..."
                lines.append(f"    {desc}")
                if not m.spec.is_loadable:
                    lines.append(
                        "    NOTE: not loadable in this deployment "
                        "(missing env config or unreachable dependency)."
                    )
            return "\n".join(lines)

    return ToolSearchTool


# Lazily build the class so test imports don't blow up if crewai/pydantic
# aren't fully wired.
try:
    ToolSearchTool = _build_tool_search_class()
except Exception as exc:
    logger.debug("tool_search: BaseTool class build deferred: %s", exc)
    ToolSearchTool = None  # type: ignore[assignment]


def create_tool_search_tools(agent_id: str = "coder") -> list:
    """Factory used by the legacy agent factories (coder.py, writer.py)
    to add tool_search to their inventory in Phase 1b. Always returns
    a 1-element list (or [] if crewai/pydantic aren't importable)."""
    global ToolSearchTool
    if ToolSearchTool is None:
        try:
            ToolSearchTool = _build_tool_search_class()
        except Exception as exc:
            logger.warning("tool_search: cannot build class: %s", exc)
            return []
    return [ToolSearchTool()]
