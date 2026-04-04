"""Org chart — agent hierarchy and reporting lines."""
import logging
from app.control_plane.db import execute

logger = logging.getLogger(__name__)


def get_org_chart() -> list[dict]:
    """Return full agent hierarchy."""
    return execute(
        """SELECT agent_role, display_name, reports_to, job_description,
                  soul_file, default_model, sort_order
           FROM control_plane.org_chart
           ORDER BY sort_order""",
        fetch=True,
    ) or []


def get_agent(role: str) -> dict | None:
    """Get a single agent's org chart entry."""
    from app.control_plane.db import execute_one
    return execute_one(
        "SELECT * FROM control_plane.org_chart WHERE agent_role = %s",
        (role,),
    )


def get_reports(manager_role: str) -> list[dict]:
    """Get agents that report to a manager."""
    return execute(
        """SELECT agent_role, display_name, job_description
           FROM control_plane.org_chart
           WHERE reports_to = %s
           ORDER BY sort_order""",
        (manager_role,), fetch=True,
    ) or []


def format_org_chart() -> str:
    """Human-readable org chart for Signal."""
    agents = get_org_chart()
    if not agents:
        return "No agents in org chart."
    lines = ["🏢 Org Chart:"]
    for a in agents:
        indent = "  " if a.get("reports_to") else ""
        arrow = f"→ reports to {a['reports_to']}" if a.get("reports_to") else "(CEO)"
        lines.append(f"{indent}{a['display_name']} ({a['agent_role']}) {arrow}")
        if a.get("job_description"):
            lines.append(f"{indent}  {a['job_description'][:80]}")
    return "\n".join(lines)
