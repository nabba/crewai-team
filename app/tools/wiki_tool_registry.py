"""
Wiki tool registry — factory function to create wiki tools for CrewAI agents.

Usage:
    from app.tools.wiki_tool_registry import create_wiki_tools
    tools = create_wiki_tools()          # all 4 tools
    tools = create_wiki_tools("read")    # just WikiReadTool
"""

from app.tools.wiki_tools import (
    WikiReadTool,
    WikiWriteTool,
    WikiSearchTool,
    WikiLintTool,
)


_TOOL_MAP = {
    "read": WikiReadTool,
    "write": WikiWriteTool,
    "search": WikiSearchTool,
    "lint": WikiLintTool,
}


def create_wiki_tools(*names: str) -> list:
    """Create wiki tools for a CrewAI agent.

    Args:
        *names: Tool names to include ('read', 'write', 'search', 'lint').
                If empty, returns all 4 tools.

    Returns:
        List of instantiated wiki tool objects.
    """
    if not names:
        return [cls() for cls in _TOOL_MAP.values()]

    tools = []
    for name in names:
        name = name.strip().lower()
        if name not in _TOOL_MAP:
            raise ValueError(
                f"Unknown wiki tool '{name}'. "
                f"Valid names: {', '.join(sorted(_TOOL_MAP))}"
            )
        tools.append(_TOOL_MAP[name]())
    return tools
