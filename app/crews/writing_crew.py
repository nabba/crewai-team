from crewai import Task, Crew, Process
from app.agents.writer import create_writer
from app.sanitize import wrap_user_input
from app.firebase_reporter import crew_started, crew_completed, crew_failed


class WritingCrew:
    def run(self, task_description: str, parent_task_id: str = None) -> str:
        """Run a writing crew on the given task."""
        task_id = crew_started("writing", f"Write: {task_description[:100]}",
                               eta_seconds=90, parent_task_id=parent_task_id)
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

        try:
            result = crew.kickoff()
            result_str = str(result)
            crew_completed("writing", task_id, result_str[:200])
            return result_str
        except Exception as exc:
            crew_failed("writing", task_id, str(exc)[:200])
            raise
