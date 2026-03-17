from crewai import Task, Crew, Process
from app.agents.researcher import create_researcher
from app.sanitize import wrap_user_input


class ResearchCrew:
    def run(self, topic: str) -> str:
        """Run a research crew on the given topic."""
        researcher = create_researcher()

        task = Task(
            description=f"""Research the following topic thoroughly:

{wrap_user_input(topic)}

Search the web for at least 3 high-quality sources. Read articles and extract key
information. If any YouTube videos are relevant, extract their transcripts.
Store all findings in team memory.

Compile a structured research report with:
1. Key findings
2. Important details and data points
3. Sources (with URLs)
""",
            expected_output="A structured research report with key findings, details, and cited sources.",
            agent=researcher,
        )

        crew = Crew(
            agents=[researcher],
            tasks=[task],
            process=Process.sequential,
            verbose=True,
        )

        result = crew.kickoff()
        return str(result)
