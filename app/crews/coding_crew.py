from crewai import Task, Crew, Process
from app.agents.coder import create_coder
from app.sanitize import wrap_user_input


class CodingCrew:
    def run(self, task_description: str) -> str:
        """Run a coding crew on the given task."""
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

        result = crew.kickoff()
        return str(result)
