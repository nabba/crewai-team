from crewai import Task, Crew, Process
from app.agents.writer import create_writer
from app.sanitize import wrap_user_input


class WritingCrew:
    def run(self, task_description: str) -> str:
        """Run a writing crew on the given task."""
        writer = create_writer()

        task = Task(
            description=f"""Complete the following writing task:

{wrap_user_input(task_description)}

First, check team memory for any relevant research or context. Then write clear,
well-structured content. Adapt the length and format based on the destination:
- Signal messages: concise, under 1500 characters
- Files: can be longer, use Markdown formatting

If the output is a document or report, save it using the file_manager tool.
""",
            expected_output="Well-written content appropriate for the destination format.",
            agent=writer,
        )

        crew = Crew(
            agents=[writer],
            tasks=[task],
            process=Process.sequential,
            verbose=True,
        )

        result = crew.kickoff()
        return str(result)
