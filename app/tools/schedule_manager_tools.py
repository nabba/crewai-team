"""
schedule_manager_tools.py — User-configurable scheduled automations.

Stores schedules in workspace/schedules.json.
Integrates with APScheduler for cron-based execution.
Zero external dependencies (uses existing APScheduler).

Usage:
    from app.tools.schedule_manager_tools import create_schedule_tools
    tools = create_schedule_tools("commander")

    # At startup (main.py):
    from app.tools.schedule_manager_tools import register_user_schedules
    register_user_schedules(scheduler)
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_SCHEDULES_PATH = Path("/app/workspace/schedules.json")


def _load_schedules() -> list[dict]:
    """Load user schedules from JSON file."""
    if not _SCHEDULES_PATH.exists():
        return []
    try:
        return json.loads(_SCHEDULES_PATH.read_text())
    except Exception:
        return []


def _save_schedules(schedules: list[dict]) -> None:
    """Save user schedules to JSON file."""
    _SCHEDULES_PATH.parent.mkdir(parents=True, exist_ok=True)
    _SCHEDULES_PATH.write_text(json.dumps(schedules, indent=2))


def _validate_cron(cron_expr: str) -> bool:
    """Validate a cron expression (5 fields)."""
    parts = cron_expr.strip().split()
    if len(parts) != 5:
        return False
    try:
        from apscheduler.triggers.cron import CronTrigger
        CronTrigger.from_crontab(cron_expr)
        return True
    except Exception:
        return False


def register_user_schedules(scheduler) -> int:
    """Load saved schedules and register them as APScheduler jobs.

    Called once at startup from main.py lifespan.
    Returns number of schedules registered.
    """
    schedules = _load_schedules()
    registered = 0
    for sched in schedules:
        if not sched.get("enabled", True):
            continue
        cron = sched.get("cron", "")
        name = sched.get("name", "")
        task = sched.get("task", "")
        if not cron or not name or not task:
            continue
        try:
            from apscheduler.triggers.cron import CronTrigger
            trigger = CronTrigger.from_crontab(cron)
            job_id = f"user_schedule_{name}"

            # Check if job already exists (idempotent)
            if scheduler.get_job(job_id):
                continue

            scheduler.add_job(
                _execute_scheduled_task,
                trigger,
                id=job_id,
                kwargs={"name": name, "task": task},
                replace_existing=True,
            )
            registered += 1
            logger.info(f"schedule_manager: registered '{name}' ({cron})")
        except Exception as e:
            logger.warning(f"schedule_manager: failed to register '{name}': {e}")
    return registered


def _execute_scheduled_task(name: str, task: str) -> None:
    """Execute a scheduled task by routing it through the Commander.

    Wrapped with ``notify_on_complete`` (Phase 7) so each user-defined
    schedule pings Signal + Web Push when it finishes — success or
    failure. The decorator runs around `_run_user_schedule` so the outer
    error logging stays in place.
    """
    from app.notify import notify_on_complete

    @notify_on_complete(
        label=f"Schedule: {name}",
        metadata={"job_id": f"schedule:{name}"},
        # Q4.1 (PROGRAM §41.4) — user-defined scheduled tasks route
        # through the arbiter: many of them are routine ("daily
        # weather", "morning poetry") and should arbitrate. Failures
        # are operationally meaningful (user notices it stopped
        # running) so they bypass arbitration via critical_on_failure.
        arbitrate=True,
        topic=f"schedule:{name}",
        critical_on_failure=True,
    )
    def _run() -> None:
        # Update last_run
        schedules = _load_schedules()
        for sched in schedules:
            if sched["name"] == name:
                sched["last_run"] = datetime.now(timezone.utc).isoformat()
                break
        _save_schedules(schedules)

        # Route through Commander
        from app.agents.commander.orchestrator import Commander
        commander = Commander()
        commander.handle(task, sender="scheduler")

    logger.info(f"schedule_manager: executing '{name}': {task[:100]}")
    try:
        _run()
    except Exception as e:
        # The decorator already pinged on the failure path; log + swallow
        # so APScheduler doesn't disable the job.
        logger.error(f"schedule_manager: task '{name}' failed: {e}")


def handle_webhook_event(event_type: str, payload: dict) -> str:
    """Handle an incoming webhook event. Match against schedule triggers."""
    schedules = _load_schedules()
    matched = [
        s for s in schedules
        if s.get("trigger_event") == event_type and s.get("enabled", True)
    ]
    if not matched:
        return f"No schedules match event: {event_type}"

    results = []
    for sched in matched:
        try:
            _execute_scheduled_task(sched["name"], sched["task"])
            results.append(f"Triggered: {sched['name']}")
        except Exception as e:
            results.append(f"Failed: {sched['name']} ({e})")
    return "; ".join(results)


def create_schedule_tools(agent_id: str) -> list:
    """Create schedule management tools. Always available."""
    try:
        from crewai.tools import BaseTool
        from pydantic import BaseModel, Field
        from typing import Type
    except ImportError:
        return []

    # ── Tool definitions ──────────────────────────────────────────

    class _CreateScheduleInput(BaseModel):
        name: str = Field(
            description="Unique name for this schedule (e.g. 'morning-email-check')"
        )
        cron: str = Field(
            description="Cron expression (5 fields): minute hour day month weekday. "
            "Examples: '0 9 * * *' (daily 9am), '0 9 * * 1-5' (weekdays 9am), "
            "'*/30 * * * *' (every 30 min)"
        )
        task: str = Field(
            description="Task description that will be sent to the Commander "
            "(e.g. 'Check email and summarize unread messages')"
        )

    class CreateScheduleTool(BaseTool):
        name: str = "create_schedule"
        description: str = (
            "Create a new scheduled automation. Specify a cron expression "
            "and a task description. The task runs automatically at the scheduled times."
        )
        args_schema: Type[BaseModel] = _CreateScheduleInput

        def _run(self, name: str, cron: str, task: str) -> str:
            if not _validate_cron(cron):
                return f"Invalid cron expression: {cron}. Use 5 fields: minute hour day month weekday."

            schedules = _load_schedules()

            # Check for duplicate name
            for s in schedules:
                if s["name"] == name:
                    return f"Schedule '{name}' already exists. Delete it first to recreate."

            new_schedule = {
                "name": name,
                "cron": cron,
                "task": task,
                "enabled": True,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "last_run": "",
            }
            schedules.append(new_schedule)
            _save_schedules(schedules)

            # Try to register with running scheduler
            try:
                from app.main import _scheduler_instance
                if _scheduler_instance:
                    from apscheduler.triggers.cron import CronTrigger
                    trigger = CronTrigger.from_crontab(cron)
                    _scheduler_instance.add_job(
                        _execute_scheduled_task,
                        trigger,
                        id=f"user_schedule_{name}",
                        kwargs={"name": name, "task": task},
                        replace_existing=True,
                    )
            except Exception:
                pass  # Will be picked up on next restart

            return f"Schedule '{name}' created: {cron} -> {task[:100]}"

    class _ListSchedulesInput(BaseModel):
        pass

    class ListSchedulesTool(BaseTool):
        name: str = "list_schedules"
        description: str = "List all user-configured scheduled automations."
        args_schema: Type[BaseModel] = _ListSchedulesInput

        def _run(self) -> str:
            schedules = _load_schedules()
            if not schedules:
                return "No scheduled automations configured."

            lines = [f"{len(schedules)} schedule(s):"]
            for s in schedules:
                status = "ON" if s.get("enabled", True) else "OFF"
                last_run = s.get("last_run", "never")
                lines.append(
                    f"  [{status}] {s['name']}: {s['cron']}\n"
                    f"       Task: {s['task'][:80]}\n"
                    f"       Last run: {last_run}"
                )
            return "\n".join(lines)

    class _DeleteScheduleInput(BaseModel):
        name: str = Field(description="Name of the schedule to delete")

    class DeleteScheduleTool(BaseTool):
        name: str = "delete_schedule"
        description: str = "Delete a scheduled automation by name."
        args_schema: Type[BaseModel] = _DeleteScheduleInput

        def _run(self, name: str) -> str:
            schedules = _load_schedules()
            original_len = len(schedules)
            schedules = [s for s in schedules if s["name"] != name]

            if len(schedules) == original_len:
                return f"Schedule '{name}' not found."

            _save_schedules(schedules)

            # Remove from running scheduler
            try:
                from app.main import _scheduler_instance
                if _scheduler_instance:
                    _scheduler_instance.remove_job(f"user_schedule_{name}")
            except Exception:
                pass

            return f"Schedule '{name}' deleted."

    class _TriggerScheduleInput(BaseModel):
        name: str = Field(description="Name of the schedule to trigger immediately")

    class TriggerScheduleTool(BaseTool):
        name: str = "trigger_schedule"
        description: str = "Run a scheduled task immediately (once, regardless of cron)."
        args_schema: Type[BaseModel] = _TriggerScheduleInput

        def _run(self, name: str) -> str:
            schedules = _load_schedules()
            target = next((s for s in schedules if s["name"] == name), None)
            if not target:
                return f"Schedule '{name}' not found."

            try:
                _execute_scheduled_task(name, target["task"])
                return f"Schedule '{name}' triggered successfully."
            except Exception as e:
                return f"Error triggering schedule: {str(e)[:200]}"

    return [
        CreateScheduleTool(),
        ListSchedulesTool(),
        DeleteScheduleTool(),
        TriggerScheduleTool(),
    ]
