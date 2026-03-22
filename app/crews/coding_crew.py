import logging
import time as _time
from crewai import Task, Crew, Process
from app.agents.coder import create_coder
from app.sanitize import wrap_user_input
from app.firebase_reporter import crew_started, crew_completed, crew_failed
from app.self_heal import diagnose_and_fix
from app.memory.belief_state import update_belief
from app.benchmarks import record_metric
from app.llm_selector import difficulty_to_tier

logger = logging.getLogger(__name__)

# Static task template — extracted for Anthropic prompt prefix caching.
CODING_TASK_TEMPLATE = """\
Complete the following coding task:

{user_input}

Write clean, well-documented code. Test it by executing it in the Docker sandbox.
If the code fails, debug and fix it. Save the final working code to a file using
the file_manager tool.

Return the working code along with its output.
"""


class CodingCrew:
    def run(self, task_description: str, parent_task_id: str = None, difficulty: int = 5) -> str:
        """Run a coding crew on the given task."""
        _start = _time.monotonic()
        from app.conversation_store import estimate_eta
        task_id = crew_started("coding", f"Code: {task_description[:100]}",
                               eta_seconds=estimate_eta("coding"), parent_task_id=parent_task_id)
        update_belief("coder", "working", current_task=task_description[:100])
        from app.llm_mode import get_mode
        force_tier = difficulty_to_tier(difficulty, get_mode())
        coder = create_coder(force_tier=force_tier)

        # S6: Policy loading moved to commander._run_crew() parallel context fetch
        task = Task(
            description=CODING_TASK_TEMPLATE.format(user_input=wrap_user_input(task_description)),
            expected_output="Working code with execution output, saved to a file.",
            agent=coder,
        )

        crew = Crew(
            agents=[coder],
            tasks=[task],
            process=Process.sequential,
            verbose=True,
        )

        try:
            result = crew.kickoff()
            result_str = str(result)

            update_belief("coder", "completed", current_task=task_description[:100])
            record_metric("task_completion_time", _time.monotonic() - _start, {"crew": "coding"})
            crew_completed("coding", task_id, result_str[:200])
            return result_str
        except Exception as exc:
            update_belief("coder", "failed", current_task=task_description[:100])
            crew_failed("coding", task_id, str(exc)[:200])
            diagnose_and_fix("coding", task_description, exc)
            raise

