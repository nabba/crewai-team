"""Multi-project isolation — wraps existing project_isolation.py with PostgreSQL persistence.

Each project gets: its own ticket queue, budget allocations, audit trail (filtered),
ChromaDB collection namespace, and Mem0 user_id namespace.

Active-project persistence (added 2026-05-09)
---------------------------------------------
Pre-fix, the active project was pure class-level state — wiped on
gateway restart. Users would explicitly ``switch workspace to eesti
mets``, the gateway would later restart for any reason, and the
NEXT task would silently land under ``default`` (or whichever project
the auto-detector grabbed first). The user saw the right "Switched to
…" reply but their tickets ended up in the wrong project anyway.

The fix persists ``(active_project_id, active_project_source)`` to
``workspace/control_plane/active_project.json`` via ``JsonStore``:
  * On every successful ``switch()`` we write through.
  * On first read after process start, we lazy-load.
  * The default-fallback path is unchanged — still the ``default``
    project if no persisted state exists.

Auto-detector picks (source="auto") are also persisted so the
sticky-user-pick guard survives restart. Without that, after a
restart any keyword-trigger could suddenly steal the active project
from the user's explicit choice.
"""
import json
import logging
import threading
from pathlib import Path

from app.control_plane.db import execute, execute_one, execute_scalar
from app.json_store import JsonStore

logger = logging.getLogger(__name__)


_ACTIVE_PROJECT_STORE = JsonStore(
    "workspace/control_plane/active_project.json",
    default={},
)


def _read_persisted_active() -> tuple[str | None, str | None]:
    """Read ``(project_id, source)`` from the on-disk store. Returns
    ``(None, None)`` if no file or unparseable."""
    try:
        data = _ACTIVE_PROJECT_STORE.load() or {}
        pid = data.get("project_id")
        source = data.get("source")
        if isinstance(pid, str) and pid:
            return pid, (source if isinstance(source, str) else None)
    except Exception as exc:
        logger.warning("active_project: load failed: %s", exc)
    return None, None


def _write_persisted_active(project_id: str | None, source: str | None) -> None:
    """Write ``(project_id, source)`` through to disk. ``None`` clears
    the file (the manager goes back to the default-fallback path)."""
    try:
        if project_id is None:
            _ACTIVE_PROJECT_STORE.save({})
            return
        from datetime import datetime, timezone
        _ACTIVE_PROJECT_STORE.save({
            "project_id": project_id,
            "source": source,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as exc:
        logger.warning("active_project: persist failed: %s", exc)


class ProjectManager:
    """Multi-project CRUD and scoping."""

    _active_project_id: str | None = None
    # Source of the current `_active_project_id` value:
    #   "user"   → explicit user action (Signal command, dashboard click)
    #   "auto"   → keyword-detection in main.py (low-trust)
    #   None     → no explicit choice yet (falls back to "default")
    # The keyword-detection layer must NEVER overwrite a "user" pick.
    _active_project_source: str | None = None
    _lock = threading.Lock()
    # Set True after we've consulted the on-disk store at least once.
    # Lazy-load on first get/switch so we don't pay the disk read at
    # import time.
    _persisted_loaded: bool = False

    def _ensure_persisted_loaded(self) -> None:
        """Lazy-restore the on-disk active-project state on first use.

        Called from inside ``_lock`` by both ``get_active_project_id``
        and ``switch``. Idempotent — flips ``_persisted_loaded`` so
        subsequent calls are no-ops.
        """
        if ProjectManager._persisted_loaded:
            return
        pid, source = _read_persisted_active()
        if pid:
            # Verify the project still exists. If a project was
            # deleted while the gateway was down, drop the stale
            # persisted state and fall through to default.
            try:
                row = self.get_by_id(pid)
            except Exception:
                row = None
            if row:
                ProjectManager._active_project_id = pid
                ProjectManager._active_project_source = source
                logger.info(
                    "control_plane: restored active project from disk: "
                    "%s (source=%s)",
                    row.get("name"), source,
                )
            else:
                logger.warning(
                    "control_plane: persisted active project %s no longer "
                    "exists; clearing", pid,
                )
                _write_persisted_active(None, None)
        ProjectManager._persisted_loaded = True

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
            # Auto-create per-business knowledge base collection.
            try:
                from app.knowledge_base.business_store import get_registry
                get_registry().create_store(name)
            except Exception:
                pass
        return row or {}

    def list_all(self) -> list[dict]:
        """List all projects."""
        return execute(
            """SELECT id, name, description, mission, is_active, created_at
               FROM control_plane.projects ORDER BY created_at""",
            fetch=True,
        ) or []

    def get_by_name(self, name: str) -> dict | None:
        """Get a project by name (case-insensitive)."""
        return execute_one(
            "SELECT * FROM control_plane.projects WHERE LOWER(name) = LOWER(%s)",
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
        """Get the currently active project ID. Falls back to default.

        Lazy-restores any persisted active project on first call after
        process start. After that, the in-memory copy is the source of
        truth (writes go through ``switch`` which updates both).
        """
        with self._lock:
            self._ensure_persisted_loaded()
            if self._active_project_id:
                return self._active_project_id
        return str(self.get_default_project_id() or "")

    def switch(self, project_name: str, *, source: str = "user") -> dict | None:
        """Switch active project context.

        ``source`` records who initiated the switch:
          * ``"user"`` (default) — Signal command, dashboard click, API call.
            Once a user pick is recorded, ``switch(..., source="auto")``
            is ignored — keyword-detection cannot overwrite explicit intent.
          * ``"auto"`` — keyword-detection in ``main.py`` (low trust).

        Name matching is case-insensitive (see get_by_name). Passes the
        canonical DB name to project_isolation.activate() to avoid
        case-mismatch ValueError on the in-memory dict.
        """
        project = self.get_by_name(project_name)
        if not project:
            return None
        canonical_name = project.get("name") or project_name
        new_pid = str(project["id"])
        with self._lock:
            self._ensure_persisted_loaded()
            # Sticky-user-pick guard. If the user has already chosen a
            # project this session, only another user action can change
            # it. The auto-detector keeps suggesting based on keywords,
            # but those suggestions stop overriding the explicit pick.
            #
            # Bug fix 2026-05-02: previously the auto-detector ran on
            # every Signal message and silently re-routed the user to
            # PLG / Archibal / KaiCart based on keywords like "estonia"
            # or "event". Users who had explicitly switched (e.g.
            # ``switch workspace to eesti mets``) saw their tickets
            # land under PLG instead, with no log line explaining why.
            #
            # Bug fix 2026-05-09: persist the active project to disk
            # so it survives gateway restart. Pre-fix, an explicit
            # ``switch workspace to eesti mets`` was lost across
            # restarts and tickets again ended up under default/PLG
            # (the same failure shape, different cause).
            if (
                source == "auto"
                and self._active_project_source == "user"
                and self._active_project_id
            ):
                logger.debug(
                    "control_plane: ignoring auto-detect switch to '%s' — "
                    "explicit user pick is sticky (current=%s)",
                    canonical_name, self._active_project_id,
                )
                return project  # return the row but don't change state
            self._active_project_id = new_pid
            self._active_project_source = source
        # Persist outside the lock — the on-disk write is independently
        # consistent (atomic via JsonStore's tempfile+rename).
        _write_persisted_active(new_pid, source)
        # Also activate in the existing project_isolation system using the
        # canonical name (not the user-supplied casing).
        try:
            from app.project_isolation import get_manager
            pm = get_manager()
            pm.activate(canonical_name)
        except Exception:
            logger.debug(
                f"control_plane: project_isolation activate failed for "
                f"'{canonical_name}'", exc_info=True,
            )
        logger.info(
            f"control_plane: switched to project '{canonical_name}' (source={source})",
        )
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

_manager: ProjectManager | None = None
_lock = threading.Lock()

def get_projects() -> ProjectManager:
    global _manager
    if _manager is None:
        with _lock:
            if _manager is None:
                _manager = ProjectManager()
    return _manager
