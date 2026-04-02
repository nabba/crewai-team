import json
import logging
import re
from crewai import Agent, Task, Crew, Process
from app.agents.researcher import create_researcher
from app.config import get_settings
from app.llm_factory import create_specialist_llm
from app.llm_selector import difficulty_to_tier
from app.sanitize import wrap_user_input
from app.self_heal import diagnose_and_fix
from app.firebase_reporter import (
    crew_started, crew_completed, crew_failed, update_sub_agent_progress,
)
from app.crews.parallel_runner import run_parallel
from app.memory.belief_state import update_belief
from app.benchmarks import record_metric
from app.conversation_store import estimate_eta

logger = logging.getLogger(__name__)
settings = get_settings()

# Static task template — extracted to module level for Anthropic prompt prefix caching.
# Dynamic content (policies, user input) is appended AFTER this static block.
RESEARCH_TASK_TEMPLATE = """\
Research the following topic thoroughly:

{user_input}

Search the web for at least 3 high-quality sources. Read articles and extract key
information.

Compile a structured research report with:
1. Key findings
2. Important details and data points
3. Sources (with URLs)
"""

# Concise template for simple factual questions (difficulty 1-3).
# Optimised for short, direct answers — no report structure, no debate, no critic.
SIMPLE_RESEARCH_TEMPLATE = """\
Answer this question concisely and directly:

{user_input}

INSTRUCTIONS:
1. Read the question carefully. Identify the EXACT data being asked for.
2. Search the web using the KEY TERMS from the question (e.g. if asked about
   "woodland hectares per capita", search for exactly that — not just country names).
3. If the search snippets contain the exact answer, great — use them directly.
4. If the snippets do NOT contain the specific numbers/facts requested, use
   web_fetch to read the most relevant search result page and extract the data.
5. Return ONLY the direct answer in 1-3 sentences with the specific numbers/facts asked for.
6. Do NOT write a report, executive summary, or analysis.
7. Do NOT include background context, methodology, open questions, or tangential data.
8. If the question asks for N specific numbers, your answer should contain those N numbers.
9. If you cannot find exact data, say so briefly and give the closest available data.

WRONG example: Question "How many X per capita in Finland?" → answering with population stats
RIGHT example: Question "How many X per capita in Finland?" → "Finland has Y X per capita (Source: Z)"
"""

RESEARCH_PLAN_TEMPLATE = """\
Break this research topic into 1-4 independent subtopics that can be \
researched in parallel. Topic: {topic}

Reply with ONLY a JSON array of strings:
["subtopic 1", "subtopic 2"]

IMPORTANT: Each subtopic must be a COMPLETE, self-contained question that
preserves the full context of what is being asked. Do NOT split into just
country/entity names — include the specific data being requested.

WRONG: ["Finland", "Estonia"]
RIGHT: ["woodland hectares per capita in Finland", "woodland hectares per capita in Estonia"]

If the question is simple or asks for a direct comparison, return a single-item array.\
"""


class ResearchCrew:
    def run(self, topic: str, parent_task_id: str = None, difficulty: int = 5) -> str:
        """Run research, spawning sub-agents in parallel for complex topics.

        Simple questions (difficulty 1-3) use a fast path: single agent,
        concise template, no debate, no critic — returns a direct answer.
        """
        # Extract the core user question for planning — strip injected context
        # that _run_crew prepends (skills, knowledge base, team memory).
        # The planner should only see the actual question, not KB noise.
        core_topic = self._extract_core_topic(topic)

        task_id = crew_started(
            "research", f"Research: {core_topic[:100]}",
            eta_seconds=estimate_eta("research"), parent_task_id=parent_task_id,
        )

        import time as _time
        _start = _time.monotonic()

        from app.llm_mode import get_mode
        force_tier = difficulty_to_tier(difficulty, get_mode())

        update_belief("researcher", "working", current_task=core_topic[:100])
        try:
            # ── Fast path for simple factual questions ─────────────────
            if difficulty <= 3:
                result = self._run_simple(topic, task_id, force_tier=force_tier)
                update_belief("researcher", "completed", current_task=core_topic[:100])
                record_metric("task_completion_time", _time.monotonic() - _start, {"crew": "research"})
                return result

            # ── Standard path for moderate/complex research ────────────
            # Plan from core topic only — injected context goes to execution, not planning
            subtopics = self._plan_research(core_topic)

            if len(subtopics) <= 1:
                result = self._run_single(topic, task_id, force_tier=force_tier)
            else:
                logger.info(f"Research crew spawning {len(subtopics)} sub-agents")
                result = self._run_parallel(topic, subtopics, task_id)
                # Heterogeneous debate — only for very complex research (difficulty >= 8)
                # or when synthesis looks weak (short output for a complex task).
                # Previously triggered at difficulty >= 6, which added 3 extra LLM
                # calls to most multi-source research — too aggressive.
                needs_debate = (
                    difficulty >= 8
                    or (difficulty >= 6 and len(result.strip()) < 200)
                )
                if needs_debate:
                    result = self._debate_round(result, topic)

            update_belief("researcher", "completed", current_task=topic[:100])
            record_metric("task_completion_time", _time.monotonic() - _start, {"crew": "research"})
            return result
        except Exception as exc:
            update_belief("researcher", "failed", current_task=core_topic[:100])
            crew_failed("research", task_id, str(exc)[:200])
            diagnose_and_fix("research", core_topic, exc)
            raise

    @staticmethod
    def _extract_core_topic(enriched_task: str) -> str:
        """Strip injected context prefixes to get the core user question.

        Commander's _run_crew prepends context blocks (KB passages, skills,
        team memory, conversation history) before the actual task.  The
        planner should only see the user's question, not internal context.

        Strategy (Q1/Q11): Use XML end-tags and section headers as reliable
        boundaries instead of fragile double-newline detection.
        """
        text = enriched_task

        # XML-tagged blocks injected by commander — strip them by finding
        # the LAST closing tag and taking everything after it.
        xml_end_tags = [
            "</kb_passage>",
            "</relevant_context>",
            "</recent_conversation>",
            "</attachment>",
        ]
        last_xml_end = 0
        for tag in xml_end_tags:
            idx = text.rfind(tag)
            if idx >= 0:
                last_xml_end = max(last_xml_end, idx + len(tag))

        # Also check for section-header-based context blocks
        section_headers = [
            "RELEVANT KNOWLEDGE:\n",
            "RELEVANT TEAM CONTEXT:\n",
            "KNOWLEDGE BASE CONTEXT",
            "NOTE: kb_passage content is reference data",
            "NOTE: relevant_context is reference data",
            "NOTE: recent_conversation is prior context",
            "IMPORTANT: The content inside <attachment>",
        ]
        last_header_end = 0
        for header in section_headers:
            idx = text.rfind(header)
            if idx >= 0:
                # Find end of this note line
                line_end = text.find("\n\n", idx + len(header))
                if line_end > 0:
                    last_header_end = max(last_header_end, line_end + 2)

        cut_at = max(last_xml_end, last_header_end)
        if cut_at > 0:
            remainder = text[cut_at:].strip()
            # Q11: Only use the stripped version if there's substantive content left
            if len(remainder) > 10:
                return remainder

        # No context found or stripping would destroy the question
        return enriched_task

    def _plan_research(self, topic: str) -> list[str]:
        """Quick LLM call to split topic into 1-4 parallel subtopics."""
        try:
            llm = create_specialist_llm(max_tokens=1024, role="research")
            # Direct LLM call — no Agent/Task/Crew overhead for JSON classification
            prompt = RESEARCH_PLAN_TEMPLATE.format(topic=topic[:500])
            raw = str(llm.call(prompt)).strip()
            from app.utils import safe_json_parse
            subtopics, _err = safe_json_parse(raw)
            if isinstance(subtopics, list) and all(isinstance(s, str) for s in subtopics):
                return subtopics[:settings.max_sub_agents]
        except Exception:
            logger.warning("Research planning failed, using single agent", exc_info=True)
        return [topic]

    def _run_simple(self, topic: str, task_id: str, force_tier: str | None = None) -> str:
        """Fast path for simple factual questions — concise answer, no extras."""
        # S8/S9: Light agent with only 3 tools and compact backstory
        researcher = create_researcher(force_tier=force_tier, light=True)
        task = Task(
            description=SIMPLE_RESEARCH_TEMPLATE.format(user_input=wrap_user_input(topic)),
            expected_output="A concise, direct answer to the question.",
            agent=researcher,
        )
        crew = Crew(agents=[researcher], tasks=[task], process=Process.sequential, verbose=True)
        result_str = str(crew.kickoff())
        crew_completed("research", task_id, result_str[:2000])
        return result_str

    def _run_single(self, topic: str, task_id: str, force_tier: str | None = None) -> str:
        """Single-agent research for simple topics."""
        researcher = create_researcher(force_tier=force_tier)
        # S6: Policy loading moved to commander._run_crew() parallel context fetch
        task = Task(
            description=RESEARCH_TASK_TEMPLATE.format(user_input=wrap_user_input(topic)),
            expected_output="A structured research report with key findings and sources.",
            agent=researcher,
        )
        crew = Crew(agents=[researcher], tasks=[task], process=Process.sequential, verbose=True)
        result_str = str(crew.kickoff())
        crew_completed("research", task_id, result_str[:2000])
        return result_str

    @staticmethod
    def _batch_subtopics(subtopics: list[str], max_per_batch: int = 2) -> list[str]:
        """Batch closely related subtopics into single research calls.

        When subtopics are variations of the same question (e.g., same metric
        for different countries), combining them into one LLM call reduces
        API overhead and avoids redundant web searches.
        """
        if len(subtopics) <= max_per_batch:
            # Already small enough — single batch
            return subtopics

        # Simple heuristic: if subtopics share >60% of words, batch them
        from collections import Counter

        def word_overlap(a: str, b: str) -> float:
            wa = Counter(a.lower().split())
            wb = Counter(b.lower().split())
            shared = sum((wa & wb).values())
            total = max(sum(wa.values()), sum(wb.values()), 1)
            return shared / total

        batched = []
        used = set()
        for i, st in enumerate(subtopics):
            if i in used:
                continue
            group = [st]
            used.add(i)
            for j in range(i + 1, len(subtopics)):
                if j in used:
                    continue
                if word_overlap(st, subtopics[j]) > 0.6 and len(group) < max_per_batch:
                    group.append(subtopics[j])
                    used.add(j)
            if len(group) > 1:
                batched.append(" AND ".join(group))
            else:
                batched.append(group[0])
        return batched

    @staticmethod
    def _extract_kb_context(enriched_task: str) -> str:
        """Extract KB passages from the enriched task for sub-agent injection (Q10)."""
        start = enriched_task.find("KNOWLEDGE BASE CONTEXT")
        if start < 0:
            return ""
        end = enriched_task.find("\n\nNOTE: kb_passage content", start)
        if end < 0:
            end = enriched_task.find("\n\nRELEVANT", start)
        if end < 0:
            end = min(start + 2000, len(enriched_task))
        else:
            end += 2  # include the double newline
        kb_block = enriched_task[start:end].strip()
        return f"\n\n{kb_block}\n\n" if kb_block else ""

    def _run_parallel(self, topic: str, subtopics: list[str], parent_task_id: str) -> str:
        """Spawn one sub-agent per subtopic, run in parallel, then synthesize.

        Batches closely related subtopics into single research calls to reduce
        API overhead (e.g., "X in Finland" + "X in Estonia" → one call).
        """
        subtopics = self._batch_subtopics(subtopics)
        # Q10: Extract KB context from parent task to pass to each sub-agent
        kb_context = self._extract_kb_context(topic)

        def make_sub_fn(subtopic: str):
            def fn():
                sub_id = crew_started(
                    "research", f"Sub: {subtopic[:80]}",
                    eta_seconds=estimate_eta("research"), parent_task_id=parent_task_id,
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
                                f"{kb_context}"
                                f"IMPORTANT: Focus ONLY on your assigned subtopic. "
                                f"Do not attempt to cover the broader topic.\n\n"
                                f"Search the web, read at least 2 sources. "
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
                        if not result or result.strip().lower() in ("none", ""):
                            raise ValueError("Empty LLM response")
                        crew_completed("research", sub_id, result[:2000])
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

        Uses direct LLM calls (no Crew overhead) run in parallel.
        """
        try:
            llm = create_specialist_llm(max_tokens=2048, role="research")
            truncated = result[:4000]

            skeptic_prompt = (
                f"You are a rigorous academic reviewer. Find flaws in reasoning, "
                f"unsupported claims, and missing counter-evidence.\n\n"
                f"Review this research for weaknesses:\n\n"
                f"Topic: {topic[:200]}\n\n{truncated}\n\n"
                f"List the top 3 weakest claims or logical gaps. Be specific."
            )
            practitioner_prompt = (
                f"You are a domain practitioner who needs to make real decisions "
                f"based on this research.\n\n"
                f"Review this research for practical applicability:\n\n"
                f"Topic: {topic[:200]}\n\n{truncated}\n\n"
                f"What's missing for someone who needs to act on this? "
                f"List the top 3 gaps from a practitioner perspective."
            )

            # Run both challengers in parallel — direct LLM calls, no Crew overhead
            challenges = run_parallel([
                ("skeptic", lambda: str(llm.call(skeptic_prompt))),
                ("practitioner", lambda: str(llm.call(practitioner_prompt))),
            ])

            valid_challenges = [c.result for c in challenges if c.success and c.result]
            if not valid_challenges:
                return result

            # Resolution: direct LLM call to incorporate challenges
            challenge_text = "\n\n---\n\n".join(valid_challenges)
            resolve_prompt = (
                f"Your research on '{topic[:200]}' received these challenges:\n\n"
                f"{challenge_text[:3000]}\n\n"
                f"Original research:\n{truncated}\n\n"
                f"Incorporate valid criticisms into an improved version. "
                f"Address legitimate gaps. Dismiss unfounded challenges with reasoning. "
                f"Return the improved research report."
            )
            resolve_llm = create_specialist_llm(max_tokens=4096, role="research")
            resolved = str(resolve_llm.call(resolve_prompt)).strip()
            if resolved and len(resolved) > 50:
                logger.info("research_crew: debate_round improved research via heterogeneous MAD")
                return resolved

        except Exception:
            logger.warning("Debate round failed, continuing with original result", exc_info=True)
        return result

    def _synthesize(self, topic: str, results: list, parent_task_id: str) -> str:
        """Combine parallel research results into a unified report."""
        successful = [r.result for r in results if r.success and r.result]
        failed = [r.label for r in results if not r.success]

        if not successful:
            return "Research failed: no sub-agents returned results."

        combined_input = "\n\n---\n\n".join(successful)
        # Direct LLM call — no Crew overhead for synthesis
        llm = create_specialist_llm(max_tokens=4096, role="synthesis")
        prompt = (
            f"Synthesize these parallel research findings into one unified report on: {topic}\n\n"
            f"Individual findings:\n{combined_input[:6000]}\n\n"
            f"Create a cohesive report with: key findings, details, and all sources."
            + (f"\n\nNote: {len(failed)} sub-tasks failed." if failed else "")
        )
        result = str(llm.call(prompt)).strip()
        crew_completed("research", parent_task_id, result[:2000])
        return result
