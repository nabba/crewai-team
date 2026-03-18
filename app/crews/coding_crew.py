from crewai import Task, Crew, Process
from app.agents.coder import create_coder
from app.sanitize import wrap_user_input
from app.firebase_reporter import crew_started, crew_completed, crew_failed
from app.self_heal import diagnose_and_fix


class CodingCrew:
    def run(self, task_description: str, parent_task_id: str = None) -> str:
        """Run a coding crew on the given task."""
        task_id = crew_started("coding", f"Code: {task_description[:100]}",
                               eta_seconds=180, parent_task_id=parent_task_id)
        coder = create_coder()

        task = Task(
            description=f"""Complete the following coding task:

{wrap_user_input(task_description)}

Write clean, well-documented code. Test it by executing it in the Docker sandbox.
If the code fails, debug and fix it. Save the final working code to a file using
the file_manager tool.

Return the working code along with its output.
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
            crew_completed("coding", task_id, result_str[:200])
            return result_str
        except Exception as exc:
            crew_failed("coding", task_id, str(exc)[:200])
            diagnose_and_fix("coding", task_description, exc)
            raise
