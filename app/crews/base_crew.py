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
    from app.llm_mode import get_mode
    force_tier = difficulty_to_tier(difficulty, get_mode())
    agent = create_agent_fn(force_tier=force_tier)

    # Extract model name from the agent's LLM for the task record
    _model_name = ""
    try:
        _model_name = getattr(agent, 'llm', None) and getattr(agent.llm, 'model', '') or ''
    except Exception:
        pass

    task_id = crew_started(
        crew_name,
        f"{crew_name.title()}: {task_description[:100]}",
        eta_seconds=estimate_eta(crew_name),
        parent_task_id=parent_task_id,
        model=_model_name,
    )
    update_belief(agent_role, "working", current_task=task_description[:100])

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

        # L4: Autobiographical journal entry (append-only, ~100 bytes)
        try:
            import json as _json
            from datetime import datetime as _dt, timezone as _tz
            _journal = Path("/app/workspace/journal.jsonl")
            with open(_journal, "a") as _jf:
                _jf.write(_json.dumps({
                    "ts": _dt.now(_tz.utc).isoformat(),
                    "crew": crew_name,
                    "task": task_description[:200],
                    "result": "success",
                    "duration_s": round(duration, 1),
                }) + "\n")
        except Exception:
            pass

        # Capture token usage from active request tracker
        _tokens = 0; _model = ""; _cost = 0.0
        try:
            from app.rate_throttle import get_active_tracker
            t = get_active_tracker()
            if t:
                _tokens = t.total_tokens
                _model = ", ".join(sorted(t.models_used)) if t.models_used else ""
                _cost = t.total_cost_usd
        except Exception:
            pass
        crew_completed(crew_name, task_id, result[:2000],
                       tokens_used=_tokens, model=_model, cost_usd=_cost)
        return result
    except Exception as exc:
        update_belief(agent_role, "failed", current_task=task_description[:100])
        crew_failed(crew_name, task_id, str(exc)[:200])

        # L4: Journal failure entry
        try:
            import json as _json
            from datetime import datetime as _dt, timezone as _tz
            _journal = Path("/app/workspace/journal.jsonl")
            with open(_journal, "a") as _jf:
                _jf.write(_json.dumps({
                    "ts": _dt.now(_tz.utc).isoformat(),
                    "crew": crew_name,
                    "task": task_description[:200],
                    "result": "failed",
                    "error": str(exc)[:100],
                    "duration_s": round(_time.monotonic() - start, 1),
                }) + "\n")
        except Exception:
            pass

        diagnose_and_fix(crew_name, task_description, exc, task_id=task_id)
        raise
