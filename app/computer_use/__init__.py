"""
app.computer_use — vision-driven UI automation, opt-in fallback only.

Phase 6 ships the agent loop, monthly USD cap, per-task hard cap, and
per-step audit trail for Anthropic's native computer-use tool against
Claude Haiku 4.5. The default backend is a Playwright-controlled headless
Chromium so this works inside the existing Docker container without a new
desktop VM. A swap-in backend with a full X11 desktop is the next iteration.

Key invariants:

  - Disabled by default. The runtime toggle ``vision_cu_enabled`` (Phase 0)
    must be on. Tool factories return ``[]`` otherwise.
  - Hard caps enforced INSIDE the runner, not just by the model:
      · ≤ 30 steps per task (``MAX_STEPS_PER_TASK``)
      · ≤ $0.50 per task (``MAX_USD_PER_TASK``)
      · ≤ ``vision_cu_monthly_cap_usd`` per calendar month
        (defaults to $10 from Phase 0; adjustable in /cp/settings)
  - Every step writes a ``computer_use_step`` audit entry so an operator
    reading the hash-chained ledger can replay any task.
  - Last-resort routing — Commander prefers Playwright/AppleScript first.

Public surface:

    create_computer_use_tools(agent_id)   CrewAI BaseTool factory; returns []
                                          unless feature is enabled
    run_task(task, ...)                   programmatic entry for the loop
    BudgetExceeded                        raised when monthly or per-task cap
                                          would be breached
    snapshot()                            current spend + remaining budget
"""
from __future__ import annotations

from app.computer_use.budget import (
    snapshot, BudgetExceeded, MAX_STEPS_PER_TASK, MAX_USD_PER_TASK,
)
from app.computer_use.runner import run_task

__all__ = [
    "snapshot", "BudgetExceeded",
    "MAX_STEPS_PER_TASK", "MAX_USD_PER_TASK",
    "run_task",
]
