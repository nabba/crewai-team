"""LoadableAgent — Agent subclass that wires a ToolBinder into the executor.

Phase 0 spike. Two responsibilities:

  1. Construct with `core_tools` + `available_tools` (a {name: factory}
     mapping); the executor only sees core_tools at boot, so the system
     prompt's tool-description block is small.
  2. Inject a `load_tool` BaseTool into the agent's core toolset so
     the model can pull a tool from the catalog mid-task; the binder
     handles the actual instantiation + dirty-flag bookkeeping.

The custom executor (LoadableAgentExecutor) re-renders openai_tools
between iterations whenever the binder reports dirty.

This is deliberately minimal — Phase 1 will replace `available_tools`
with a real registry + `tool_search` semantic discovery. The shape of
the public API on Agent is meant to survive that promotion unchanged.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Type

from crewai import Agent
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from app.tool_runtime.binder import ToolBinder, ToolNotAvailable
from app.tool_runtime.loadable_executor import LoadableAgentExecutor

logger = logging.getLogger(__name__)


def _make_load_tool_for_binder(binder: ToolBinder) -> BaseTool:
    """Build a `load_tool` BaseTool that closes over the agent's binder."""

    class _LoadToolInput(BaseModel):
        name: str = Field(
            description=(
                "Name of the tool to load from the registry. Use exact "
                "name. Use `list_available_tools` (no args) to see what "
                "names exist."
            ),
        )

    class LoadTool(BaseTool):
        name: str = "load_tool"
        description: str = (
            "Load a tool by name from this agent's registry. After "
            "loading, the tool's schema is announced to you on the "
            "next step and you call it normally. Idempotent — calling "
            "this with an already-loaded tool name is a no-op.\n\n"
            "Use this when you need a capability you don't currently "
            "have. Do NOT speculatively load tools you might not use; "
            "each load adds tokens to subsequent steps."
        )
        args_schema: Type[BaseModel] = _LoadToolInput

        def _run(self, name: str) -> str:
            try:
                instance = binder.load(name)
            except ToolNotAvailable:
                catalog = ", ".join(binder.catalog_names()) or "(empty)"
                return (
                    f"ERROR: no tool named '{name}' in this registry. "
                    f"Available: {catalog}"
                )
            return (
                f"OK — '{name}' loaded. Its schema will be announced "
                f"on your next step. Call it normally then."
            )

    return LoadTool()


def _make_list_available_tool(binder: ToolBinder) -> BaseTool:
    class _ListInput(BaseModel):
        pass

    class ListAvailableTools(BaseTool):
        name: str = "list_available_tools"
        description: str = (
            "List the names of tools you can `load_tool(name)` from "
            "this agent's registry. Returns names only; descriptions "
            "are visible after loading. Use this BEFORE inventing a "
            "tool name — the catalog is the source of truth."
        )
        args_schema: Type[BaseModel] = _ListInput

        def _run(self) -> str:
            names = binder.catalog_names()
            loaded = set(binder.loaded_names)
            if not names:
                return "(catalog empty)"
            lines = ["Available tools (load with load_tool(name=...)):"]
            for n in names:
                marker = " [loaded]" if n in loaded else ""
                lines.append(f"  - {n}{marker}")
            return "\n".join(lines)

    return ListAvailableTools()


class LoadableAgent(Agent):
    """Agent with a ToolBinder driving its tool list.

    Construction:
        agent = LoadableAgent(
            role=..., goal=..., backstory=..., llm=...,
            core_tools=[file_manager, knowledge_search],   # always in system prompt
            available_tools={                              # discoverable, lazy
                "pdf_compose": lambda: PdfComposeTool(),
                "signal_send_attachment": lambda: ...,
            },
        )

    The agent automatically gains `load_tool` and `list_available_tools`
    in addition to the core_tools provided.
    """

    _binder: ToolBinder = PrivateAttr()

    def __init__(
        self,
        *,
        core_tools: list[Any] | None = None,
        available_tools: dict[str, Callable[[], Any]] | None = None,
        **agent_kwargs: Any,
    ) -> None:
        binder = ToolBinder(
            core_tools=list(core_tools or []),
            available=dict(available_tools or {}),
        )
        # Inject the binder-aware control tools into core. They CANNOT
        # be in `available` because the agent needs them to discover.
        binder._core.append(_make_load_tool_for_binder(binder))
        binder._core.append(_make_list_available_tool(binder))

        agent_kwargs["tools"] = binder.tools  # initial value; never mutated by us
        super().__init__(**agent_kwargs)
        self._binder = binder

    @property
    def binder(self) -> ToolBinder:
        return self._binder

    def create_agent_executor(
        self, tools: list[BaseTool] | None = None, task: Any = None
    ) -> None:
        """Override CrewAI's executor creation to use LoadableAgentExecutor."""
        # Force the custom executor class for this agent's lifetime.
        self.executor_class = LoadableAgentExecutor
        super().create_agent_executor(tools=tools, task=task)
        # Wire the binder onto the executor so its loop can read it.
        if self.agent_executor is not None:
            self.agent_executor.binder = self._binder
