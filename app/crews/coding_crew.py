"""coding_crew.py — Code generation and sandbox execution crew."""

from app.agents.coder import create_coder
from app.crews.base_crew import run_single_agent_crew

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
        return run_single_agent_crew(
            crew_name="coding",
            agent_role="coder",
            create_agent_fn=create_coder,
            task_template=CODING_TASK_TEMPLATE,
            task_description=task_description,
            expected_output="Working code with execution output, saved to a file.",
            parent_task_id=parent_task_id,
            difficulty=difficulty,
        )
