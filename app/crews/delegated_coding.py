"""
delegated_coding.py — Design + Coordinator + Execution + Debug crew.

Engaged when delegation_settings.is_enabled("coding") is True. Each
sub-agent stays ≤ 18 tools so Anthropic strict-mode works.

The pipeline runs in two phases:

  Phase 1 (Design)  : Design Specialist produces a technical spec.
                      No code is written — just a contract for Phase 2.
  Phase 2 (Code)    : Coordinator implements against the spec, delegating
                      to Execution Specialist for runs and Debug Specialist
                      on failures.

Splitting design from implementation reduces BadRequestError and TimeoutError
on complex tasks where a single agent otherwise tries to think and code
simultaneously and exceeds the model's effective working memory.

This is the same idea as the rejected exp_202604290007_1172, refactored
into the existing crew rather than a parallel CrewAI-bypassing module.
"""
from __future__ import annotations

import logging

from crewai import Crew, Task, Process

from app.agents.specialists import (
    create_coding_coordinator,
    create_design_specialist,
    create_execution_specialist,
    create_debug_specialist,
)
from app.sanitize import wrap_user_input
from app.crews.lifecycle import crew_lifecycle
from app.llm_selector import difficulty_to_tier

logger = logging.getLogger(__name__)

# Difficulty threshold above which the explicit Design phase runs.
# Trivial tasks (difficulty ≤ this) go straight to implementation to
# avoid spending an LLM round-trip on a one-line spec.
_DESIGN_PHASE_DIFFICULTY_FLOOR = 5

_DESIGN_TASK_TEMPLATE = """\
Produce a TECHNICAL SPECIFICATION for the following coding task.
Do NOT write code yet — only the spec.

{user_input}

{tool_inventory_section}\
Required sections (keep each short):
1. Summary
2. Assumptions / non-goals
3. File-by-file proposed changes
4. Key APIs / interfaces
5. Error handling and validation
6. Testing plan
7. Risks and mitigations

If the task is trivial, say so and produce a one-line spec.

CRITICAL — when the inventory above lists a specialist tool that fits the
task (e.g. `gee_run_script` for satellite/forest/Hansen analysis,
`firecrawl_extract` for structured web scraping), the spec MUST name it
and follow the pattern in its description.  Do NOT propose writing a
custom Python pipeline that re-implements what a registered tool does —
the executor will get capped/timed-out and the user will get nothing.
"""

_DELEGATED_CODING_TASK_TEMPLATE = """\
Implement the following coding task.

{user_input}

{design_spec_section}\
{tool_inventory_section}\
Process:
1. Write the code yourself (you have file_manager + memory tools).
2. Delegate to the Execution Specialist to RUN the code and capture real output.
3. If the run fails, delegate to the Debug Specialist for diagnosis.
4. Apply the fix and re-delegate execution.
5. When it runs clean, return the final working code + its real output.

OUTPUT RULES:
 - Return ONLY the final deliverable — working code plus actual execution output.
 - Do NOT narrate your delegation steps.
 - Save the final code to a file via file_manager if appropriate.

CRITICAL — when the inventory above lists a specialist tool that fits
the task (e.g. `gee_run_script` for satellite/forest/Hansen analysis,
`firecrawl_extract` for web scraping), CALL IT directly via the
Execution Specialist.  Do NOT write a custom Python pipeline that
re-implements what a registered tool does — the executor's per-tool
budget is 180s, and a naive Python loop calling Earth Engine per-year
will burn that budget on every iteration.  Read each tool's
description carefully — many tools document the FAST pattern explicitly
(e.g. `gee_run_script` shows `# GOOD` single-call frequencyHistogram
vs `# BAD` per-year loop).  Follow the FAST pattern exactly.
"""


class DelegatedCodingCrew:
    def run(
        self,
        task_description: str,
        parent_task_id: str | None = None,
        difficulty: int = 5,
    ) -> str:
        from app.llm_mode import get_mode
        force_tier = difficulty_to_tier(difficulty, get_mode())

        with crew_lifecycle(
            crew_name="coding",
            agent_role="coder",
            task_title=f"Coding (delegated): {task_description[:100]}",
            task_description=task_description,
            parent_task_id=parent_task_id,
            mode="delegated",
        ) as ctx:
            coordinator = create_coding_coordinator(force_tier=force_tier)
            executor = create_execution_specialist(force_tier=force_tier)
            debugger = create_debug_specialist(force_tier=force_tier)

            # Phase 1 — Design (skipped for trivial tasks)
            design_spec = self._maybe_run_design_phase(
                task_description=task_description,
                difficulty=difficulty,
                force_tier=force_tier,
            )

            # Phase 2 — Coordinator implements, delegates run/debug as needed
            spec_section = (
                f"Design spec to follow (the implementer's contract):\n{design_spec}\n\n"
                if design_spec
                else ""
            )
            # 2026-05-02 audit Week 2.5 — also inject the tool inventory
            # into the implementation prompt.  Week 2 only injected it
            # into the design phase; verification dispatch v5 showed the
            # design spec correctly named gee_run_script, but the
            # implementation/executor still wrote slow per-year loops
            # because its prompt didn't carry the same nudge.  Same
            # helper, called over (coordinator + executor + debugger)
            # so the implementation prompt sees every tool the team
            # collectively has.
            impl_inventory = _render_tool_inventory_section(
                coordinator, executor, debugger
            )
            task = Task(
                description=_DELEGATED_CODING_TASK_TEMPLATE.format(
                    user_input=wrap_user_input(task_description),
                    design_spec_section=spec_section,
                    tool_inventory_section=impl_inventory,
                ),
                expected_output=(
                    "Working code with real execution output, saved to a file if appropriate."
                ),
                agent=coordinator,
            )

            crew = Crew(
                agents=[coordinator, executor, debugger],
                tasks=[task],
                process=Process.hierarchical,
                manager_llm=coordinator.llm,
                verbose=False,
            )

            result_str = str(crew.kickoff())
            ctx.set_outcome(result_str)
            return result_str

    def _maybe_run_design_phase(
        self,
        *,
        task_description: str,
        difficulty: int,
        force_tier: str | None,
    ) -> str:
        """Run the Design Specialist for non-trivial tasks. Return the spec text.

        Returns an empty string when the design phase is skipped (trivial task)
        or when the design call fails — in either case the implementer falls
        back to its previous behaviour.
        """
        if difficulty < _DESIGN_PHASE_DIFFICULTY_FLOOR:
            return ""

        try:
            designer = create_design_specialist(force_tier=force_tier)
            # 2026-05-02 audit Week 2 — inject the executor + designer's
            # actual tool inventory into the design prompt so the spec
            # explicitly references registered specialist tools (e.g.
            # gee_run_script with the documented frequencyHistogram
            # pattern) instead of proposing a re-implementation.
            executor = create_execution_specialist(force_tier=force_tier)
            inventory_section = _render_tool_inventory_section(designer, executor)
            design_task = Task(
                description=_DESIGN_TASK_TEMPLATE.format(
                    user_input=wrap_user_input(task_description),
                    tool_inventory_section=inventory_section,
                ),
                expected_output=(
                    "A concise technical specification covering the seven required sections."
                ),
                agent=designer,
            )
            design_crew = Crew(
                agents=[designer],
                tasks=[design_task],
                process=Process.sequential,
                verbose=False,
            )
            spec = str(design_crew.kickoff()).strip()
            if not spec:
                return ""
            logger.info(f"delegated_coding: design phase produced {len(spec)} chars of spec")
            return spec
        except Exception as exc:
            # Graceful degradation — never let the design phase break the run.
            logger.warning(f"delegated_coding: design phase failed, proceeding without spec: {exc}")
            return ""


def _render_tool_inventory_section(*agents) -> str:
    """Render a Markdown bullet list of unique tools attached to the
    given agents, with category-grouped headings and the first 100 chars
    of each tool's description.

    Returns an empty string when no tools are attached (graceful no-op).
    Used by ``_maybe_run_design_phase`` to inject the executor + designer
    tool inventory into the design prompt — closes the loop where the
    design LLM was producing specs that ignored registered specialist
    tools.  See PHASE_4_recommendation_memo.md (Shift 1) for context.
    """
    seen: dict[str, object] = {}
    for agent in agents:
        for t in (getattr(agent, "tools", None) or []):
            name = getattr(t, "name", "")
            if name and name not in seen:
                seen[name] = t
    if not seen:
        return ""

    # Group by an inferred "category bucket" purely from name conventions.
    # We don't read the registry's category field directly here because
    # not every tool is registered yet — Week 2 only annotates ~10 of the
    # ~45 tool sources.  This bucketer is a close-enough fallback.
    def _bucket(n: str) -> str:
        n = n.lower()
        if n.startswith("gee_") or "earth_engine" in n:
            return "Geospatial / satellite"
        if n.startswith("firecrawl_") or n in ("web_search", "web_fetch"):
            return "Web research / scraping"
        if n.startswith("browser_"):
            return "Browser automation"
        if n in ("execute_code", "run_python") or "sandbox" in n or "bridge" in n:
            return "Code execution"
        if n in ("file_manager", "read_attachment", "ocr_extract_text"):
            return "Files & attachments"
        if n.startswith("memory_") or n.startswith("scoped_memory_") or n.startswith("mem0_"):
            return "Memory"
        if "knowledge" in n or "search_journal" in n or "research_knowledge" in n:
            return "Knowledge bases"
        if n.startswith("generate_") or "wiki_" in n:
            return "Output generation"
        return "Other"

    grouped: dict[str, list] = {}
    for name, t in seen.items():
        grouped.setdefault(_bucket(name), []).append((name, t))

    lines = ["## Tools available to designer + executor (use these — do not re-implement):", ""]
    # Stable display order — most-likely-needed buckets first.
    for bucket in (
        "Code execution", "Geospatial / satellite", "Web research / scraping",
        "Browser automation", "Files & attachments", "Knowledge bases",
        "Memory", "Output generation", "Other",
    ):
        items = grouped.get(bucket)
        if not items:
            continue
        lines.append(f"### {bucket}")
        for name, t in sorted(items):
            desc = (getattr(t, "description", "") or "").strip()
            # First sentence or 100 chars, whichever is shorter — keeps
            # the prompt budget focused on actionable info.
            desc = " ".join(desc.split())
            first_period = desc.find(".")
            if 30 < first_period < 200:
                desc = desc[: first_period + 1]
            else:
                desc = desc[:200]
            lines.append(f"- `{name}` — {desc}")
        lines.append("")
    return "\n".join(lines) + "\n"
