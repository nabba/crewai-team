"""tool_runtime — Phase 0 spike for dynamic tool loading in CrewAI.

This package is the *spike scaffolding* that proves the LoadableAgent
pattern works against CrewAI 1.14.x. It is NOT yet wired into any
production agent. Phase 1 will replace it (or promote it) with the
real registry-backed implementation.

Goals validated by this spike:
  1. Mutating an agent's toolset mid-task is feasible by overriding
     CrewAgentExecutor._invoke_loop_native_tools to recompute the
     openai_tools / available_functions captures on every iteration.
  2. New tools can be announced to the model via a synthetic user
     message ("you now have these tools") that lands in the turn
     region of the prompt, leaving the system prefix cacheable.
  3. The pattern composes cleanly with the existing optional_tool_group
     surface — agents that opt in get LoadableAgent; others stay on
     stock crewai.Agent.

See docs/TOOL_REGISTRY_PHASE_0.md for the full memo and findings.
"""
from app.tool_runtime.binder import ToolBinder
from app.tool_runtime.loadable_agent import LoadableAgent

__all__ = ["LoadableAgent", "ToolBinder"]
