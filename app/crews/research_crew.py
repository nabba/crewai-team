import json
import logging
import re
from crewai import Agent, Task, Crew, Process
from app.agents.researcher import create_researcher
from app.config import get_settings
from app.llm_factory import create_specialist_llm
from app.sanitize import wrap_user_input
from app.self_heal import diagnose_and_fix
from app.firebase_reporter import (
    crew_started, crew_completed, crew_failed, update_sub_agent_progress,
)
from app.crews.parallel_runner import run_parallel
from app.memory.belief_state import update_belief
from app.agents.critic import create_critic
from app.policies.policy_loader import load_relevant_policies
from app.benchmarks import record_metric

logger = logging.getLogger(__name__)
settings = get_settings()


class ResearchCrew:
    def run(self, topic: str, parent_task_id: str = None) -> str:
        """Run research, spawning sub-agents in parallel for complex topics."""
        task_id = crew_started(
            "research", f"Research: {topic[:100]}",
            eta_seconds=120, parent_task_id=parent_task_id,
        )

        import time as _time
        _start = _time.monotonic()

        update_belief("researcher", "working", current_task=topic[:100])
        try:
            subtopics = self._plan_research(topic)

            if len(subtopics) <= 1:
                result = self._run_single(topic, task_id)
            else:
                logger.info(f"Research crew spawning {len(subtopics)} sub-agents")
                result = self._run_parallel(topic, subtopics, task_id)
                # Heterogeneous debate — only for complex multi-subtopic research
                result = self._debate_round(result, topic)

            # Critic review step
            result = self._critic_review(result, topic)

            update_belief("researcher", "completed", current_task=topic[:100])
            record_metric("task_completion_time", _time.monotonic() - _start, {"crew": "research"})
            return result
        except Exception as exc:
            update_belief("researcher", "failed", current_task=topic[:100])
            crew_failed("research", task_id, str(exc)[:200])
            diagnose_and_fix("research", topic, exc)
            raise

    def _plan_research(self, topic: str) -> list[str]:
        """Quick LLM call to split topic into 1-4 parallel subtopics."""
        try:
            llm = create_specialist_llm(max_tokens=1024, role="research")
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
        policies = load_relevant_policies(topic, "researcher")
        policies_block = f"\n{policies}\n" if policies else ""
        task = Task(
            description=f"""{policies_block}Research the following topic thoroughly:

{wrap_user_input(topic)}

Search the web for at least 3 high-quality sources. Read articles and extract key
information. Store all findings in team memory.

Compile a structured research report with:
1. Key findings
2. Important details and data points
3. Sources (with URLs)

After completing your research, use the self_report tool to assess your confidence,
completeness, and any blockers. Then use store_reflection to record what you learned
about your own performance on this task.
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
                max_retries = 2
                last_exc = None
                for attempt in range(1, max_retries + 1):
                    try:
                        researcher = create_researcher()
                        task = Task(
                            description=(
                                f"Research this specific subtopic thoroughly:\n\n"
                                f"{wrap_user_input(subtopic)}\n\n"
                                f"IMPORTANT: Focus ONLY on your assigned subtopic. "
                                f"Do not attempt to cover the broader topic.\n\n"
                                f"Search the web, read at least 2 sources. "
                                f"Store key findings in shared team memory. "
                                f"Return a concise summary with sources. "
                                f"Then use self_report to assess your confidence and completeness."
                            ),
                            expected_output="Research findings with sources.",
                            agent=researcher,
                        )
                        crew = Crew(
                            agents=[researcher], tasks=[task],
                            process=Process.sequential, verbose=True,
                        )
                        result = str(crew.kickoff())
                        if not result or result.strip().lower() in ("none", ""):
                            raise ValueError("Empty LLM response")
                        crew_completed("research", sub_id, result[:200])
                        return result
                    except Exception as exc:
                        last_exc = exc
                        err_str = str(exc).lower()
                        is_transient = any(k in err_str for k in (
                            "invalid response", "none or empty", "empty",
                            "connection", "timeout", "busy",
                        ))
                        if is_transient and attempt < max_retries:
                            import time
                            wait = 3 * attempt
                            logger.warning(
                                f"Sub-agent '{subtopic[:40]}' attempt {attempt} failed "
                                f"(transient), retrying in {wait}s: {exc}"
                            )
                            time.sleep(wait)
                            continue
                        break
                crew_failed("research", sub_id, str(last_exc)[:200])
                raise last_exc
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

    def _debate_round(self, result: str, topic: str) -> str:
        """Heterogeneous debate: two challenger perspectives review the synthesis.

        Uses different reasoning strategies (not identical agents) to catch
        different classes of errors — per Penn State/Shanghai AI Lab research
        on heterogeneous MAD outperforming homogeneous approaches.
        """
        try:
            # Use research role (qwen3:30b-a3b) — fast MoE model.
            # architecture role picks deepseek-r1:32b which is too slow for debate.
            llm = create_specialist_llm(max_tokens=2048, role="research")

            # Two challengers with genuinely different reasoning strategies
            skeptic = Agent(
                role="Research Skeptic",
                goal="Find the weakest claims, missing counter-evidence, and logical gaps.",
                backstory=(
                    "You are a rigorous academic reviewer. Your job is to find flaws "
                    "in reasoning, unsupported claims, and missing counter-evidence. "
                    "Be specific — cite which claims are weak and why."
                ),
                llm=llm, tools=[], verbose=False,
            )
            practitioner = Agent(
                role="Practitioner Reviewer",
                goal="Evaluate practical applicability — what's missing for someone who needs to act on this?",
                backstory=(
                    "You are a domain practitioner who needs to make real decisions "
                    "based on this research. Evaluate: Is it actionable? What context "
                    "is missing? What would you need to know before acting on this?"
                ),
                llm=llm, tools=[], verbose=False,
            )

            truncated = result[:4000]

            def run_skeptic():
                task = Task(
                    description=(
                        f"Review this research for weaknesses:\n\n"
                        f"Topic: {topic[:200]}\n\n{truncated}\n\n"
                        f"List the top 3 weakest claims or logical gaps. Be specific."
                    ),
                    expected_output="Top 3 weaknesses with specific citations.",
                    agent=skeptic,
                )
                crew = Crew(agents=[skeptic], tasks=[task], process=Process.sequential, verbose=False)
                return str(crew.kickoff())

            def run_practitioner():
                task = Task(
                    description=(
                        f"Review this research for practical applicability:\n\n"
                        f"Topic: {topic[:200]}\n\n{truncated}\n\n"
                        f"What's missing for someone who needs to act on this? "
                        f"List the top 3 gaps from a practitioner perspective."
                    ),
                    expected_output="Top 3 practical gaps or missing context.",
                    agent=practitioner,
                )
                crew = Crew(agents=[practitioner], tasks=[task], process=Process.sequential, verbose=False)
                return str(crew.kickoff())

            # Run challengers in parallel
            challenges = run_parallel([
                ("skeptic", run_skeptic),
                ("practitioner", run_practitioner),
            ])

            valid_challenges = [c.result for c in challenges if c.success and c.result]
            if not valid_challenges:
                return result

            # Resolution: researcher incorporates valid challenges
            researcher = create_researcher()
            challenge_text = "\n\n---\n\n".join(valid_challenges)
            resolve_task = Task(
                description=(
                    f"Your research on '{topic[:200]}' received these challenges:\n\n"
                    f"{challenge_text[:3000]}\n\n"
                    f"Original research:\n{truncated}\n\n"
                    f"Incorporate valid criticisms into an improved version. "
                    f"Address legitimate gaps. Dismiss unfounded challenges with reasoning. "
                    f"Return the improved research report."
                ),
                expected_output="Improved research report addressing valid challenges.",
                agent=researcher,
            )
            crew = Crew(agents=[researcher], tasks=[resolve_task], process=Process.sequential, verbose=True)
            resolved = str(crew.kickoff())
            if resolved and len(resolved) > 50:
                logger.info("research_crew: debate_round improved research via heterogeneous MAD")
                return resolved

        except Exception:
            logger.warning("Debate round failed, continuing with original result", exc_info=True)
        return result

    def _critic_review(self, result: str, topic: str) -> str:
        """Run a Critic agent to review the research output for quality."""
        try:
            critic = create_critic()
            review_task = Task(
                description=(
                    f"Review this research output for accuracy, gaps, and unjustified claims.\n\n"
                    f"Topic: {topic[:200]}\n\n"
                    f"Research output to review:\n{result[:4000]}\n\n"
                    f"Check for:\n"
                    f"1. Are claims supported by cited sources?\n"
                    f"2. Are there obvious gaps or missing perspectives?\n"
                    f"3. Is the confidence level justified by the evidence?\n"
                    f"4. Are there any contradictions?\n\n"
                    f"Provide a brief review with any issues found and suggestions. "
                    f"Use self_report to assess your review confidence."
                ),
                expected_output="Brief quality review with issues and suggestions.",
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
                f"Create a cohesive report with: key findings, details, and all sources. "
                f"After synthesizing, use self_report to assess overall research quality "
                f"and store_reflection to note what worked and what could improve."
                + (f"\n\nNote: {len(failed)} sub-tasks failed." if failed else "")
            ),
            expected_output="A unified research report combining all findings.",
            agent=researcher,
        )
        crew = Crew(agents=[researcher], tasks=[task], process=Process.sequential, verbose=True)
        result = str(crew.kickoff())
        crew_completed("research", parent_task_id, result[:200])
        return result
