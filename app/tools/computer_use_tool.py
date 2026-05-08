"""
computer_use_tool.py — agent-callable last-resort UI control.

Exposes a single CrewAI tool, ``computer_use``, that runs the
vision-driven loop with hard budget + step caps. Disabled by default —
the factory only returns a tool when:

  - ``runtime_settings.get_vision_cu_enabled()`` is True (Phase 0 toggle), AND
  - the Anthropic SDK can be imported, AND
  - Playwright is installed (for the browser backend), AND
  - the monthly cap hasn't been reached.

The Commander soul instructs all agents to prefer Playwright + AppleScript
first; this tool is the fallback when neither can drive the target UI.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def create_computer_use_tools(agent_id: str = "coder") -> list:
    """Build CrewAI BaseTool instances for vision-driven UI control.

    Returns an empty list when the feature is off, the Anthropic SDK is
    missing, or Playwright is unavailable. Never raises.
    """
    try:
        from app.runtime_settings import get_vision_cu_enabled
        if not get_vision_cu_enabled():
            return []
    except Exception:
        return []

    # Bail if the Anthropic SDK isn't available — this tool can't function.
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return []

    # And bail if Playwright is missing (the default backend needs it).
    try:
        import playwright  # noqa: F401
    except ImportError:
        return []

    try:
        from crewai.tools import tool
    except ImportError:
        return []

    @tool("computer_use")
    def computer_use_tool(task: str, start_url: str = "about:blank") -> str:
        """Drive a browser UI by vision when no scriptable path exists.

        Use only as a LAST RESORT, after Playwright (browser_tools) and
        AppleScript (desktop_tools) have failed or are clearly unsuitable.
        Bounded at 30 steps and $0.50 per task; refuses when the monthly
        $/cap is reached.

        Args:
            task: Plain-English description of what to accomplish.
            start_url: Optional URL to open before the loop starts.

        Returns: short text describing the outcome — either the model's
            final answer or a refusal explaining why the task couldn't run.
        """
        from app.computer_use import run_task
        result = run_task(task=task, start_url=start_url)
        if result.refused_reason:
            return f"computer_use refused: {result.refused_reason}"
        if not result.success:
            return f"computer_use finished without a clear result (steps={result.steps})"
        return result.text

    return [computer_use_tool]
