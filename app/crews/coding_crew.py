import logging
import time as _time
from crewai import Task, Crew, Process
from app.agents.coder import create_coder
from app.agents.critic import create_critic
from app.sanitize import wrap_user_input
from app.firebase_reporter import crew_started, crew_completed, crew_failed
from app.self_heal import diagnose_and_fix
from app.memory.belief_state import update_belief
from app.policies.policy_loader import load_relevant_policies
from app.benchmarks import record_metric

logger = logging.getLogger(__name__)


class CodingCrew:
    def run(self, task_description: str, parent_task_id: str = None) -> str:
        """Run a coding crew on the given task."""
        _start = _time.monotonic()
        task_id = crew_started("coding", f"Code: {task_description[:100]}",
                               eta_seconds=180, parent_task_id=parent_task_id)
        update_belief("coder", "working", current_task=task_description[:100])
        coder = create_coder()

        policies = load_relevant_policies(task_description, "coder")
        policies_block = f"\n{policies}\n" if policies else ""

        task = Task(
            description=f"""{policies_block}Complete the following coding task:

{wrap_user_input(task_description)}

Write clean, well-documented code. Test it by executing it in the Docker sandbox.
If the code fails, debug and fix it. Save the final working code to a file using
the file_manager tool.

Return the working code along with its output.

After completing the task, use the self_report tool to assess your confidence in the
code quality, completeness, and any risks. Then use store_reflection to record lessons
learned about your coding approach.
""",
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

            # Critic review — adversarial code review
            result_str = self._critic_review(result_str, task_description)

            update_belief("coder", "completed", current_task=task_description[:100])
            record_metric("task_completion_time", _time.monotonic() - _start, {"crew": "coding"})
            crew_completed("coding", task_id, result_str[:200])
            return result_str
        except Exception as exc:
            update_belief("coder", "failed", current_task=task_description[:100])
            crew_failed("coding", task_id, str(exc)[:200])
            diagnose_and_fix("coding", task_description, exc)
            raise

    def _critic_review(self, result: str, task_description: str) -> str:
        """Run a Critic agent to review the code output for quality."""
        try:
            critic = create_critic()
            review_task = Task(
                description=(
                    f"Review this code output for correctness, security, and completeness.\n\n"
                    f"Task: {task_description[:200]}\n\n"
                    f"Code output to review:\n{result[:4000]}\n\n"
                    f"Check for:\n"
                    f"1. Does the code solve what was asked?\n"
                    f"2. Security issues (injection, path traversal, unsafe operations)?\n"
                    f"3. Edge cases not handled?\n"
                    f"4. Are tests adequate?\n\n"
                    f"Provide a brief review with any issues found. "
                    f"Use self_report to assess your review confidence."
                ),
                expected_output="Brief code quality review with issues and suggestions.",
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
