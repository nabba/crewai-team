"""Multi-project isolation — wraps existing project_isolation.py with PostgreSQL persistence.

Each project gets: its own ticket queue, budget allocations, audit trail (filtered),
ChromaDB collection namespace, and Mem0 user_id namespace.
"""
import logging
import threading
from typing import Optional

from app.control_plane.db import execute, execute_one, execute_scalar

logger = logging.getLogger(__name__)


class ProjectManager:
    """Multi-project CRUD and scoping."""

    _active_project_id: Optional[str] = None
    _lock = threading.Lock()

    def create(self, name: str, mission: str = "", description: str = "",
               config: dict = None) -> dict:
        """Create a new project."""
        import json
        row = execute_one(
            """INSERT INTO control_plane.projects (name, description, mission, config_json)
               VALUES (%s, %s, %s, %s)
               RETURNING id, name, mission, is_active, created_at""",
            (name, description, mission, json.dumps(config or {})),
        )
        if row:
            from app.control_plane.audit import get_audit
            get_audit().log(
                actor="user", action="project.created",
                project_id=str(row["id"]),
                resource_type="project",
                detail={"name": name, "mission": mission[:200]},
            )
        return row or {}

    def list_all(self) -> list[dict]:
        """List all projects."""
        return execute(
            """SELECT id, name, description, mission, is_active, created_at
               FROM control_plane.projects ORDER BY created_at""",
            fetch=True,
        ) or []

    def get_by_name(self, name: str) -> dict | None:
        """Get a project by name."""
        return execute_one(
            "SELECT * FROM control_plane.projects WHERE name = %s",
            (name,),
        )

    def get_by_id(self, project_id: str) -> dict | None:
        """Get a project by ID."""
        return execute_one(
            "SELECT * FROM control_plane.projects WHERE id = %s",
            (project_id,),
        )

    def get_default_project_id(self) -> str | None:
        """Get the 'default' project ID."""
        return execute_scalar(
            "SELECT id FROM control_plane.projects WHERE name = 'default'"
        )

    def get_active_project_id(self) -> str:
        """Get the currently active project ID. Falls back to default."""
        with self._lock:
            if self._active_project_id:
                return self._active_project_id
        return str(self.get_default_project_id() or "")

    def switch(self, project_name: str) -> dict | None:
        """Switch active project context."""
        project = self.get_by_name(project_name)
        if not project:
            return None
        with self._lock:
            self._active_project_id = str(project["id"])
        # Also activate in the existing project_isolation system
        try:
            from app.project_isolation import get_manager
            pm = get_manager()
            pm.activate(project_name)
        except Exception:
            pass
        logger.info(f"control_plane: switched to project '{project_name}'")
        return project

    def get_status(self, project_id: str) -> dict:
        """Get project summary: tickets, budget, recent activity."""
        from app.control_plane.tickets import get_tickets
        from app.control_plane.budgets import get_budget_enforcer

        board = get_tickets().get_board(project_id)
        budgets = get_budget_enforcer().get_status(project_id)
        project = self.get_by_id(project_id)

        return {
            "project": project,
            "tickets": board.get("counts", {}),
            "budgets": budgets,
        }

    def format_list(self) -> str:
        """Human-readable project list for Signal."""
        projects = self.list_all()
        if not projects:
            return "No projects configured."
        active_id = self.get_active_project_id()
        lines = ["📋 Projects:"]
        for p in projects:
            marker = " ◀ active" if str(p["id"]) == active_id else ""
            lines.append(f"  {p['name']}: {p.get('mission', '—')[:80]}{marker}")
        return "\n".join(lines)


# ── Singleton ────────────────────────────────────────────────────────────────

_manager: Optional[ProjectManager] = None
_lock = threading.Lock()


def get_projects() -> ProjectManager:
    global _manager
    if _manager is None:
        with _lock:
            if _manager is None:
                _manager = ProjectManager()
    return _manager
