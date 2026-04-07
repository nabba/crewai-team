"""
composio_tool.py — Composio integration for 850+ SaaS connections.

Provides agents access to external services (Gmail, Calendar, Slack,
GitHub, Jira, Notion, Sheets, etc.) via Composio's unified SDK.

Setup:
    1. Set COMPOSIO_API_KEY in .env
    2. pip install composio-crewai (in requirements.txt)
    3. Connect apps via: composio add github, composio add gmail, etc.
       OR connect via Signal: "composio connect github"

If Composio is not installed or API key not set, tools gracefully degrade.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_composio_available = None
_toolset_instance = None


def _check_composio() -> bool:
    """Check if Composio SDK is installed and API key is set."""
    global _composio_available
    if _composio_available is not None:
        return _composio_available
    try:
        from composio_crewai import ComposioToolSet
        api_key = os.getenv("COMPOSIO_API_KEY", "")
        if not api_key:
            logger.info("composio_tool: COMPOSIO_API_KEY not set — SaaS integrations unavailable")
            _composio_available = False
            return False
        _composio_available = True
    except ImportError:
        logger.info("composio_tool: composio-crewai not installed — SaaS integrations unavailable")
        _composio_available = False
    return _composio_available


def _get_toolset():
    """Get or create a singleton ComposioToolSet."""
    global _toolset_instance
    if _toolset_instance is not None:
        return _toolset_instance
    try:
        from composio_crewai import ComposioToolSet
        api_key = os.getenv("COMPOSIO_API_KEY", "")
        _toolset_instance = ComposioToolSet(api_key=api_key)
        logger.info("composio_tool: ComposioToolSet initialized")
        return _toolset_instance
    except Exception as e:
        logger.warning(f"composio_tool: initialization failed — {e}")
        return None


def is_available() -> bool:
    """Check if Composio is available."""
    return _check_composio()


def get_composio_tools(actions: list[str] | None = None, apps: list[str] | None = None) -> list:
    """Get Composio tools for CrewAI agent assignment.

    Args:
        actions: Specific Composio actions (e.g., ["GITHUB_LIST_REPOS", "GMAIL_SEND_EMAIL"])
        apps: App names to get all tools for (e.g., ["github", "gmail"])

    Returns: List of CrewAI-compatible tool objects, or empty if unavailable.
    """
    if not _check_composio():
        return []

    toolset = _get_toolset()
    if not toolset:
        return []

    try:
        if actions:
            return toolset.get_tools(actions=actions)
        if apps:
            return toolset.get_tools(apps=apps)
        # Default: return tools for connected apps only
        return toolset.get_tools()
    except Exception as e:
        logger.warning(f"composio_tool: get_tools failed — {e}")
        return []


def list_connected_apps() -> dict:
    """List connected Composio app integrations."""
    if not _check_composio():
        return {"available": False, "reason": "Composio not configured"}

    toolset = _get_toolset()
    if not toolset:
        return {"available": False, "reason": "Toolset init failed"}

    try:
        from composio import ComposioToolSet as CoreToolSet
        core = CoreToolSet(api_key=os.getenv("COMPOSIO_API_KEY", ""))
        connected = core.get_connected_accounts()
        apps = []
        for acc in connected:
            apps.append({
                "app": getattr(acc, "appUniqueId", str(acc)),
                "status": getattr(acc, "status", "unknown"),
            })
        return {"available": True, "connected": apps, "count": len(apps)}
    except Exception as e:
        return {"available": True, "connected": [], "count": 0, "note": str(e)[:200]}


def execute_action(action: str, params: dict = None) -> dict:
    """Execute a specific Composio action directly.

    Args:
        action: Action name (e.g., "GITHUB_LIST_REPOS", "GMAIL_SEND_EMAIL")
        params: Action parameters

    Returns: {success, result} or {success: False, error}
    """
    if not _check_composio():
        return {"success": False, "error": "Composio not configured"}

    toolset = _get_toolset()
    if not toolset:
        return {"success": False, "error": "Toolset init failed"}

    try:
        result = toolset.execute_action(action=action, params=params or {})
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)[:300]}


def format_status() -> str:
    """Human-readable Composio status for Signal."""
    if not _check_composio():
        return "Composio not available. Install: pip install composio-crewai"

    info = list_connected_apps()
    if not info.get("connected"):
        return (
            "🔌 Composio: active but no apps connected.\n"
            "Connect apps with:\n"
            "  composio connect github\n"
            "  composio connect gmail\n"
            "  composio connect slack\n"
            "  composio connect notion\n"
            "Or visit: https://app.composio.dev/connections"
        )

    lines = ["🔌 Composio: active"]
    for app in info["connected"]:
        status = "✅" if app.get("status") == "active" else "⚠️"
        lines.append(f"  {status} {app.get('app', '?')}")
    return "\n".join(lines)
