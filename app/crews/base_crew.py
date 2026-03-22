"""
base_crew.py — Shared crew execution logic.

Extracts the identical boilerplate from coding_crew, writing_crew,
and media_crew into a reusable `run_single_agent_crew()` function.

Each simple crew had ~40 lines of identical code:
  - Timer start, ETA estimation, Firebase reporting
  - Belief state updates (working → completed/failed)
  - Force tier from difficulty, agent creation
  - Task creation, Crew kickoff
  - Benchmark recording, error handling, self-healing

Now each crew is ~15 lines: define template + call this function.
"""

import logging
import time as _time

from crewai import Task, Crew, Process

from app.firebase_reporter import crew_started, crew_completed, crew_failed
from app.memory.belief_state import update_belief
from app.benchmarks import record_metric
from app.llm_selector import difficulty_to_tier
from app.sanitize import wrap_user_input
from app.self_heal import diagnose_and_fix

logger = logging.getLogger(__name__)


def run_single_agent_crew(
    crew_name: str,
    agent_role: str,
    create_agent_fn,
    task_template: str,
    task_description: str,
    expected_output: str,
    parent_task_id: str = None,
    difficulty: int = 5,
) -> str:
    """Run a single-agent crew with all standard boilerplate.

    Args:
        crew_name: Firebase crew name (e.g. "coding", "writing")
        agent_role: Belief state role (e.g. "coder", "writer")
        create_agent_fn: Factory function (force_tier) → Agent
        task_template: Template string with {user_input} placeholder
        task_description: The user's task (gets wrapped in template)
        expected_output: Expected output description for CrewAI
        parent_task_id: Optional parent task for sub-agent tracking
        difficulty: Task difficulty (1-10)

    Returns:
        The crew's output as a string.
    """
    start = _time.monotonic()

    from app.conversation_store import estimate_eta
    task_id = crew_started(
        crew_name,
        f"{crew_name.title()}: {task_description[:100]}",
        eta_seconds=estimate_eta(crew_name),
        parent_task_id=parent_task_id,
    )
    update_belief(agent_role, "working", current_task=task_description[:100])

    from app.llm_mode import get_mode
    force_tier = difficulty_to_tier(difficulty, get_mode())
    agent = create_agent_fn(force_tier=force_tier)

    task = Task(
        description=task_template.format(user_input=wrap_user_input(task_description)),
        expected_output=expected_output,
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )

    try:
        result = str(crew.kickoff())
        duration = _time.monotonic() - start

        update_belief(agent_role, "completed", current_task=task_description[:100])
        record_metric("task_completion_time", duration, {"crew": crew_name})
        crew_completed(crew_name, task_id, result[:200])
        return result
    except Exception as exc:
        update_belief(agent_role, "failed", current_task=task_description[:100])
        crew_failed(crew_name, task_id, str(exc)[:200])
        diagnose_and_fix(crew_name, task_description, exc)
        raise
