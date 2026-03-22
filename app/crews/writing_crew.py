import logging
import time as _time
from crewai import Task, Crew, Process
from app.agents.writer import create_writer
from app.sanitize import wrap_user_input
from app.firebase_reporter import crew_started, crew_completed, crew_failed
from app.self_heal import diagnose_and_fix
from app.memory.belief_state import update_belief
from app.benchmarks import record_metric
from app.llm_selector import difficulty_to_tier

logger = logging.getLogger(__name__)

# Static task template — extracted for Anthropic prompt prefix caching.
WRITING_TASK_TEMPLATE = """\
Complete the following writing task:

{user_input}

First, check team memory for any relevant research or context. Then write clear,
well-structured content. Adapt the length and format based on the destination:
- Signal messages: concise, under 1500 characters
- Files: can be longer, use Markdown formatting

If the output is a document or report, save it using the file_manager tool.
"""


class WritingCrew:
    def run(self, task_description: str, parent_task_id: str = None, difficulty: int = 5) -> str:
        """Run a writing crew on the given task."""
        _start = _time.monotonic()
        from app.conversation_store import estimate_eta
        task_id = crew_started("writing", f"Write: {task_description[:100]}",
                               eta_seconds=estimate_eta("writing"), parent_task_id=parent_task_id)
        update_belief("writer", "working", current_task=task_description[:100])
        from app.llm_mode import get_mode
        force_tier = difficulty_to_tier(difficulty, get_mode())
        writer = create_writer(force_tier=force_tier)

        # S6: Policy loading moved to commander._run_crew() parallel context fetch
        task = Task(
            description=WRITING_TASK_TEMPLATE.format(user_input=wrap_user_input(task_description)),
            expected_output="Well-written content appropriate for the destination format.",
            agent=writer,
        )

        crew = Crew(
            agents=[writer],
            tasks=[task],
            process=Process.sequential,
            verbose=True,
        )

        try:
            result = crew.kickoff()
            result_str = str(result)

            update_belief("writer", "completed", current_task=task_description[:100])
            record_metric("task_completion_time", _time.monotonic() - _start, {"crew": "writing"})
            crew_completed("writing", task_id, result_str[:200])
            return result_str
        except Exception as exc:
            update_belief("writer", "failed", current_task=task_description[:100])
            crew_failed("writing", task_id, str(exc)[:200])
            diagnose_and_fix("writing", task_description, exc)
            raise

