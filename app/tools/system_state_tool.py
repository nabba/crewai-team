"""system_state_tool.py — agent-callable read of deployment state.

Phase 5.1 component. Lets any agent ask "what does the system look
like right now?" before acting on conversation history that might
be stale.

Typical use: an agent receives a task that mentions "fix the bug"
or "X is broken" and wants to verify the bug actually exists before
proposing a fix. Calling ``get_system_state()`` returns:

  * git head (host source repo) — has the file been modified recently?
  * gateway uptime — has the gateway restarted since the alleged failure?
  * recent crew runs — has the crew actually been failing, or is the
    "broken" claim from an old conversation that's been resolved?

Without this, the agent has no choice but to take the user's word
(or the conversation history's word) for whether something needs
fixing. With it, the agent can ground its judgement in real state.

Capability tag: ``reads-deployment-state`` — added to the bounded
vocabulary for this purpose.

The tool is read-only and has no side effects. It's safe to call
from any agent at any time; output is cached for 5 seconds in the
underlying ``get_system_state`` so calls are cheap.
"""
from __future__ import annotations

import json
import logging
from typing import Type

logger = logging.getLogger(__name__)


def _build_tool_class():
    """Build the BaseTool class lazily so import-time failures in
    crewai/pydantic don't poison the module."""
    from crewai.tools import BaseTool
    from pydantic import BaseModel, Field

    class _SystemStateInput(BaseModel):
        window_hours: int = Field(
            default=24,
            description=(
                "Hours of history to include in `git.files_changed_last_24h` "
                "and `recent_crew_runs`. Default 24."
            ),
        )

    class GetSystemStateTool(BaseTool):
        name: str = "get_system_state"
        description: str = (
            "Read deployment state: git head + uptime + recent crew "
            "outcomes + tool-registry size. USE THIS before assuming "
            "a bug exists or a crew is broken — conversation history "
            "may reflect a problem that's already been fixed.\n\n"
            "Returns a structured dict with sections (git, gateway, "
            "tier_immutable, tools, recent_crew_runs). Each section "
            "has an `available` bool — check it before reading "
            "fields, since transient infrastructure failures degrade "
            "individual sources gracefully.\n\n"
            "Pattern (use before proposing a code change):\n"
            "  state = get_system_state()\n"
            "  pim_runs = state['recent_crew_runs']['by_crew'].get('pim', [])\n"
            "  if any(r['ok'] for r in pim_runs[:3]):\n"
            "      # PIM has succeeded recently — the 'broken' claim\n"
            "      # is likely stale.\n"
            "      ..."
        )
        args_schema: Type[BaseModel] = _SystemStateInput

        def _run(self, window_hours: int = 24) -> str:
            try:
                from app.system_state import get_system_state
                state = get_system_state(window_hours=window_hours)
            except Exception as exc:  # noqa: BLE001
                return f"get_system_state ERROR: {type(exc).__name__}: {exc}"
            try:
                return json.dumps(state, indent=2, default=str)
            except Exception:
                return str(state)

    return GetSystemStateTool


# Build the class at module import — propagates ImportError loud,
# but caught at the registry-decoration level below.
try:
    GetSystemStateTool = _build_tool_class()
except Exception as exc:  # noqa: BLE001
    logger.debug("system_state_tool: BaseTool class build deferred: %s", exc)
    GetSystemStateTool = None  # type: ignore[assignment]


def create_system_state_tools(agent_id: str = "default") -> list:
    """Factory used by agent factories for explicit Tool injection.
    Returns a 1-element list (or [] if crewai/pydantic unavailable)."""
    global GetSystemStateTool
    if GetSystemStateTool is None:
        try:
            GetSystemStateTool = _build_tool_class()
        except Exception:
            return []
    return [GetSystemStateTool()]


# ── Tool registry annotation (Phase 5.1) ────────────────────────────
try:
    from app.tool_registry import Lifecycle, Tier, register_tool

    @register_tool(
        name="get_system_state",
        capabilities=["reads-deployment-state"],
        description=(
            "Read deployment state (git head, gateway uptime, recent "
            "crew runs, tool registry size, TIER_IMMUTABLE count). "
            "Use BEFORE assuming a bug exists or a crew is broken — "
            "conversation history may reflect resolved issues. "
            "Returns sectioned dict; each section's `available` bool "
            "indicates whether the source was reachable."
        ),
        tier=Tier.PRODUCTION,
        lifecycle=Lifecycle.SINGLETON,
    )
    def _system_state_registry_factory():
        tools = create_system_state_tools()
        if not tools:
            raise RuntimeError("system_state_tool: factory returned empty list")
        return tools[0]
except ImportError:
    pass
