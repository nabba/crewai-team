import json
import logging
import re
from crewai import Agent, Task, Crew, Process, LLM
from app.agents.researcher import create_researcher
from app.config import get_settings, get_anthropic_api_key
from app.sanitize import wrap_user_input
from app.self_heal import diagnose_and_fix
from app.firebase_reporter import (
    crew_started, crew_completed, crew_failed, update_sub_agent_progress,
)
from app.crews.parallel_runner import run_parallel

logger = logging.getLogger(__name__)
settings = get_settings()


class ResearchCrew:
    def run(self, topic: str, parent_task_id: str = None) -> str:
        """Run research, spawning sub-agents in parallel for complex topics."""
        task_id = crew_started(
            "research", f"Research: {topic[:100]}",
            eta_seconds=120, parent_task_id=parent_task_id,
        )

        try:
            subtopics = self._plan_research(topic)

            if len(subtopics) <= 1:
                return self._run_single(topic, task_id)

            logger.info(f"Research crew spawning {len(subtopics)} sub-agents")
            return self._run_parallel(topic, subtopics, task_id)
        except Exception as exc:
            crew_failed("research", task_id, str(exc)[:200])
            diagnose_and_fix("research", topic, exc)
            raise

    def _plan_research(self, topic: str) -> list[str]:
        """Quick LLM call to split topic into 1-4 parallel subtopics."""
        try:
            llm = LLM(
                model=f"anthropic/{settings.specialist_model}",
                api_key=get_anthropic_api_key(),
                max_tokens=1024,
            )
            agent = Agent(
                role="Research Planner",
                goal="Break a research topic into independent subtopics.",
                backstory="You plan research by identifying 1-4 independent angles.",
                llm=llm, verbose=False,
            )
            task = Task(
                description=(
                    f"Break this research topic into 1-4 independent subtopics that can be "
                    f"researched in parallel. Topic: {topic[:500]}\n\n"
                    f"Reply with ONLY a JSON array of strings:\n"
                    f'["subtopic 1", "subtopic 2"]\n\n'
                    f"If simple, return a single-item array."
                ),
                expected_output='A JSON array of 1-4 subtopic strings',
                agent=agent,
            )
            crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
            raw = re.sub(r'^```(?:json)?\s*', '', str(crew.kickoff()).strip())
            raw = re.sub(r'\s*```$', '', raw)
            subtopics = json.loads(raw)
            if isinstance(subtopics, list) and all(isinstance(s, str) for s in subtopics):
                return subtopics[:settings.max_sub_agents]
        except Exception:
            logger.warning("Research planning failed, using single agent", exc_info=True)
        return [topic]

    def _run_single(self, topic: str, task_id: str) -> str:
        """Single-agent research for simple topics."""
        researcher = create_researcher()
        task = Task(
            description=f"""Research the following topic thoroughly:

{wrap_user_input(topic)}

Search the web for at least 3 high-quality sources. Read articles and extract key
information. Store all findings in team memory.

Compile a structured research report with:
1. Key findings
2. Important details and data points
3. Sources (with URLs)
""",
            expected_output="A structured research report with key findings and sources.",
            agent=researcher,
        )
        crew = Crew(agents=[researcher], tasks=[task], process=Process.sequential, verbose=True)
        result_str = str(crew.kickoff())
        crew_completed("research", task_id, result_str[:200])
        return result_str

    def _run_parallel(self, topic: str, subtopics: list[str], parent_task_id: str) -> str:
        """Spawn one sub-agent per subtopic, run in parallel, then synthesize."""

        def make_sub_fn(subtopic: str):
            def fn():
                sub_id = crew_started(
                    "research", f"Sub: {subtopic[:80]}",
                    eta_seconds=90, parent_task_id=parent_task_id,
                )
                try:
                    researcher = create_researcher()
                    task = Task(
                        description=(
                            f"Research this specific subtopic thoroughly:\n\n"
                            f"{wrap_user_input(subtopic)}\n\n"
                            f"Search the web, read at least 2 sources. "
                            f"Store key findings in shared team memory. "
                            f"Return a concise summary with sources."
                        ),
                        expected_output="Research findings with sources.",
                        agent=researcher,
                    )
                    crew = Crew(
                        agents=[researcher], tasks=[task],
                        process=Process.sequential, verbose=True,
                    )
                    result = str(crew.kickoff())
                    crew_completed("research", sub_id, result[:200])
                    return result
                except Exception as exc:
                    crew_failed("research", sub_id, str(exc)[:200])
                    raise
            return fn

        parallel_tasks = [
            (f"research-{i}", make_sub_fn(st))
            for i, st in enumerate(subtopics)
        ]

        results = run_parallel(parallel_tasks)

        completed_count = sum(1 for r in results if r.success)
        update_sub_agent_progress("research", parent_task_id, completed_count, len(subtopics))

        # Phase 3: Synthesize
        return self._synthesize(topic, results, parent_task_id)

    def _synthesize(self, topic: str, results: list, parent_task_id: str) -> str:
        """Combine parallel research results into a unified report."""
        successful = [r.result for r in results if r.success and r.result]
        failed = [r.label for r in results if not r.success]

        if not successful:
            return "Research failed: no sub-agents returned results."

        combined_input = "\n\n---\n\n".join(successful)
        researcher = create_researcher()
        task = Task(
            description=(
                f"Synthesize these parallel research findings into one unified report on: {topic}\n\n"
                f"Individual findings:\n{combined_input[:6000]}\n\n"
                f"Create a cohesive report with: key findings, details, and all sources."
                + (f"\n\nNote: {len(failed)} sub-tasks failed." if failed else "")
            ),
            expected_output="A unified research report combining all findings.",
            agent=researcher,
        )
        crew = Crew(agents=[researcher], tasks=[task], process=Process.sequential, verbose=True)
        result = str(crew.kickoff())
        crew_completed("research", parent_task_id, result[:200])
        return result
