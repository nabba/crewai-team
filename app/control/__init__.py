"""
Control path — background job scheduling and system management.

This package provides a clean boundary around the background control plane.
APScheduler runs cron-based jobs (self-improvement, evolution, auditing).
The idle scheduler runs cooperative round-robin jobs when no user tasks
are active.

Components:
  - idle_scheduler: cooperative multitasking when system is idle
  - APScheduler jobs: cron-based periodic tasks (registered in main.py lifespan)
  - Firebase listeners: real-time Firestore config changes
"""

from app.idle_scheduler import (
    start as start_idle_scheduler,
    stop as stop_idle_scheduler,
    notify_task_start,
    notify_task_end,
    is_enabled as is_background_enabled,
)

__all__ = [
    "start_idle_scheduler",
    "stop_idle_scheduler",
    "notify_task_start",
    "notify_task_end",
    "is_background_enabled",
]
