import logging
import time as _time
from crewai import Task, Crew, Process
from app.agents.writer import create_writer
from app.agents.critic import create_critic
from app.sanitize import wrap_user_input
from app.firebase_reporter import crew_started, crew_completed, crew_failed
from app.self_heal import diagnose_and_fix
from app.memory.belief_state import update_belief
from app.policies.policy_loader import load_relevant_policies
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

After completing the task, use the self_report tool to assess your confidence in the
content quality and completeness. Then use store_reflection to record what went well
and what could improve about your writing approach.
"""


class WritingCrew:
    def run(self, task_description: str, parent_task_id: str = None, difficulty: int = 5) -> str:
        """Run a writing crew on the given task."""
        _start = _time.monotonic()
        task_id = crew_started("writing", f"Write: {task_description[:100]}",
                               eta_seconds=90, parent_task_id=parent_task_id)
        update_belief("writer", "working", current_task=task_description[:100])
        from app.llm_mode import get_mode
        force_tier = difficulty_to_tier(difficulty, get_mode())
        writer = create_writer(force_tier=force_tier)

        policies = load_relevant_policies(task_description, "writer")
        policies_block = f"\n{policies}\n" if policies else ""

        task = Task(
            description=(
                policies_block
                + WRITING_TASK_TEMPLATE.format(user_input=wrap_user_input(task_description))
            ),
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

            # Critic review — editorial quality check
            result_str = self._critic_review(result_str, task_description)

            update_belief("writer", "completed", current_task=task_description[:100])
            record_metric("task_completion_time", _time.monotonic() - _start, {"crew": "writing"})
            crew_completed("writing", task_id, result_str[:200])
            return result_str
        except Exception as exc:
            update_belief("writer", "failed", current_task=task_description[:100])
            crew_failed("writing", task_id, str(exc)[:200])
            diagnose_and_fix("writing", task_description, exc)
            raise

    def _critic_review(self, result: str, task_description: str) -> str:
        """Run a Critic agent to review the writing output for quality."""
        try:
            critic = create_critic()
            review_task = Task(
                description=(
                    f"Review this writing output for accuracy, completeness, and quality.\n\n"
                    f"Task: {task_description[:200]}\n\n"
                    f"Writing output to review:\n{result[:4000]}\n\n"
                    f"Check for:\n"
                    f"1. Is the content accurate and complete?\n"
                    f"2. Does it match the requested format and tone?\n"
                    f"3. Are there unsupported claims or missing sources?\n"
                    f"4. Is the structure clear and well-organized?\n\n"
                    f"Provide a brief review with any issues found. "
                    f"Use self_report to assess your review confidence."
                ),
                expected_output="Brief editorial review with issues and suggestions.",
                agent=critic,
            )
            crew = Crew(
                agents=[critic], tasks=[review_task],
                process=Process.sequential, verbose=True,
            )
            review = str(crew.kickoff()).strip()
            if review:
                result += f"\n\n---\n\n**[Critic Review]**\n{review}"
        except Exception:
            logger.warning("Critic review failed, continuing without it", exc_info=True)
        return result
