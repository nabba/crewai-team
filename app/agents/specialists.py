"""
specialists.py — narrow-focus sub-agents for delegation-mode crews.

Each specialist carries a tightly-scoped toolkit (≤ 18 tools) so the
coordinator can delegate subtasks without exceeding any provider's tool
limit.  The coordinator itself gets a small core set plus CrewAI's
delegation meta-tools (delegate_work, ask_question).

This module is ONLY imported by delegated crew paths.  Legacy single-agent
paths continue to use app/agents/researcher.py, coder.py, etc. unchanged.
"""
from __future__ import annotations

import logging

from crewai import Agent

from app.llm_factory import create_specialist_llm
from app.souls.loader import compose_backstory

logger = logging.getLogger(__name__)


# ── Shared helpers ───────────────────────────────────────────────────

def _safe(factory, *args, **kwargs):
    """Try a tool-factory and swallow errors.  Specialists are opt-in —
    missing tools shouldn't kill the crew."""
    try:
        result = factory(*args, **kwargs)
        if isinstance(result, list):
            return result
        return [result] if result else []
    except Exception:
        return []


# ── Domain-tool stop-gap (Week 1 audit fix — replaces ad-hoc holes) ─
#
# Phase 3 of the 2026-05-02 deep audit found that none of the four
# delegated_coding specialists (designer/coordinator/executor/debugger)
# had `gee_run_script` attached, despite the routing prompt explicitly
# naming GEE for satellite-imagery tasks.  Root cause: the standalone
# `coder` agent factory wires GEE in directly; the specialist factories
# were a separate code path that never got the import.
#
# This helper is the stop-gap.  Week 2 of the audit ships a declarative
# tool-registry mechanism (`assemble_tools(agent_id, categories=...)`)
# that supersedes both this helper and the rest of the open-coded tool
# wiring across the four specialist factories.  When that lands, this
# helper and its callers can be deleted in one PR.
#
# Until then: every coding-crew specialist gets the same domain tools.
def _extend_specialist_domain_tools(tools: list) -> None:
    """Append cross-specialist domain tools (geospatial, etc.) to *tools*.

    Each domain tool source is wrapped in `_safe` — missing dependencies
    or unconfigured services degrade silently rather than killing the
    crew.  Order matches priority: most-likely-needed last so cap-by-
    priority dropping kicks in on less-needed tools first.
    """
    # Geospatial — `gee_run_script` for Google Earth Engine.  Returns []
    # when GOOGLE_APPLICATION_CREDENTIALS is unset (the standalone case
    # at agent factory startup), so this is safe to call unconditionally.
    tools.extend(_safe(
        lambda: __import__(
            "app.tools.gee_tool", fromlist=["create_gee_tools"]
        ).create_gee_tools("coder")
    ))


# ── Web specialist ───────────────────────────────────────────────────
# Handles: web search, page fetch, youtube transcripts, browser
# navigation, firecrawl scraping, MCP-served web tools.

_WEB_BACKSTORY = (
    "You are the Web Research Specialist.  Your job is simple: given a search "
    "query, return the most relevant up-to-date web content as structured text.  "
    "Use web_search for keyword lookups, web_fetch to read the contents of a "
    "specific URL, get_youtube_transcript for videos, browser_fetch for "
    "JavaScript-heavy pages that web_fetch can't handle.  Always cite the URL "
    "you pulled from.  Never speculate beyond what the sources say.  Keep "
    "responses compact — you're one step in a larger research pipeline."
)


def create_web_specialist(force_tier: str | None = None) -> Agent:
    llm = create_specialist_llm(max_tokens=4096, role="research", force_tier=force_tier)
    tools: list = []
    from app.tools.web_search import web_search
    from app.tools.web_fetch import web_fetch
    from app.tools.youtube_transcript import get_youtube_transcript
    tools.extend([web_search, web_fetch, get_youtube_transcript])

    # Browser / firecrawl — optional
    for mod, fn in [
        ("app.tools.browser_tools", "create_browser_tools"),
        ("app.tools.firecrawl_tools", "create_firecrawl_tools"),
    ]:
        tools.extend(_safe(lambda m=mod, f=fn: __import__(m, fromlist=[f]).__dict__[f]()))

    # Research orchestrator — fallback path: if the coordinator delegated a
    # sub-matrix to this leaf, still use the structured pipeline rather
    # than individual web_search calls.  The coordinator should have
    # caught this already (MATRIX MODE), but this keeps the leaf honest.
    with optional_tool_group('specialists', 'research_orchestrator'):
        from app.tools.research_orchestrator import research_orchestrator
        tools.append(research_orchestrator)

    return Agent(
        role="Web Research Specialist",
        goal="Fetch authoritative web sources and return verbatim extracts with URLs.",
        backstory=_WEB_BACKSTORY,
        llm=llm,
        tools=tools,
        allow_delegation=False,  # Leaf agent — can't delegate further
        max_iter=6,
        verbose=False,
    )


# ── Document specialist ──────────────────────────────────────────────
# Handles: PDFs, OCR, attachments, local files, wiki reads, KB searches,
# research-KB, journal search.

_DOC_BACKSTORY = (
    "You are the Document Research Specialist.  Your focus is structured "
    "information: PDF documents, images needing OCR, user-attached files, "
    "the enterprise knowledge base, the research knowledge base (episteme), "
    "the experiential journal, and the internal wiki.  When asked for data "
    "you read the relevant sources and return faithful extracts with source "
    "citations (filename, page or section).  You do NOT invent data — if a "
    "source doesn't cover the asked question, say so plainly."
)


def create_document_specialist(force_tier: str | None = None) -> Agent:
    llm = create_specialist_llm(max_tokens=4096, role="research", force_tier=force_tier)
    tools: list = []
    from app.tools.attachment_reader import read_attachment
    from app.tools.file_manager import file_manager
    from app.knowledge_base.tools import KnowledgeSearchTool
    tools += [read_attachment, file_manager, KnowledgeSearchTool()]

    # OCR, PDF, wiki, research KB, journal — all optional
    for mod, fn, args in [
        ("app.tools.ocr_tool", "create_ocr_tool", ()),
        ("app.tools.wiki_tool_registry", "create_wiki_tools", ("read",)),
        ("app.episteme.tools", "get_episteme_tools", ("researcher",)),
        ("app.experiential.tools", "get_experiential_tools", ("researcher",)),
    ]:
        tools.extend(_safe(
            lambda m=mod, f=fn, a=args: __import__(m, fromlist=[f]).__dict__[f](*a),
        ))

    return Agent(
        role="Document Research Specialist",
        goal="Extract verified facts from documents, KBs, and the journal with citations.",
        backstory=_DOC_BACKSTORY,
        llm=llm,
        tools=tools,
        allow_delegation=False,
        max_iter=6,
        verbose=False,
    )


# ── Synthesis specialist ─────────────────────────────────────────────
# Handles: reasoning/dialectics, philosophy, aesthetic judgement,
# tensions, experiential journaling, self-report.

_SYNTH_BACKSTORY = (
    "You are the Synthesis Specialist.  When the coordinator has gathered "
    "raw facts from web/document specialists, your role is to integrate them "
    "into a coherent, well-reasoned response.  You use the philosophy KB for "
    "ethical/framework grounding, conceptual_blend for creative recombinations, "
    "find_counter_argument to stress-test claims, and the tensions/experiential "
    "stores to surface relevant past reflections.  Your output is the final "
    "synthesised answer: sourced, balanced, phone-readable."
)


def create_synthesis_specialist(force_tier: str | None = None) -> Agent:
    # Synthesis is the one that writes the FINAL long-form answer after
    # research/document specialists have gathered raw facts.  8192 tokens
    # so "extensive document" requests don't get chopped mid-section,
    # which was causing vetting to reject for "response is cut off" and
    # reflexion to re-run the whole 6-min hierarchical flow.
    llm = create_specialist_llm(max_tokens=8192, role="writing", force_tier=force_tier)
    tools: list = []

    # Philosophy KB + dialectics
    for mod, fn, args in [
        ("app.philosophy.rag_tool", "PhilosophyRAGTool", ()),
        ("app.philosophy.dialectics_tool", "FindCounterArgumentTool", ()),
    ]:
        with optional_tool_group('specialists', 'unknown'):
            cls = __import__(mod, fromlist=[fn]).__dict__[fn]
            tools.append(cls())

    # Conceptual blend
    with optional_tool_group('specialists', 'blend_tool'):
        from app.philosophy.blend_tool import create_conceptual_blend_tool
        t = create_conceptual_blend_tool()
        if t:
            tools.append(t)

    # Tensions / experiential / aesthetic — all with their full read+write surface
    for mod, fn, args in [
        ("app.tensions.tools", "get_tension_tools", ("writer",)),
        ("app.experiential.tools", "get_experiential_tools", ("writer",)),
        ("app.aesthetics.tools", "get_aesthetic_tools", ("writer",)),
    ]:
        tools.extend(_safe(
            lambda m=mod, f=fn, a=args: __import__(m, fromlist=[f]).__dict__[f](*a),
        ))

    return Agent(
        role="Synthesis Specialist",
        goal="Integrate gathered facts into a balanced, sourced final response.",
        backstory=_SYNTH_BACKSTORY,
        llm=llm,
        tools=tools,
        allow_delegation=False,
        max_iter=6,
        verbose=False,
    )


# ── Research coordinator ─────────────────────────────────────────────
# Small tool set — mostly orchestration.  Delegation meta-tools
# (delegate_work, ask_question) are injected by CrewAI when
# allow_delegation=True.

_COORD_BACKSTORY = compose_backstory("researcher") + (
    "\n\n"
    "═══ MATRIX MODE (MANDATORY when task is a table) ═══\n"
    "If the user asks for a TABLE / MATRIX / STRUCTURED LIST of 3+ entities\n"
    "(companies, products, people) each with 2+ attributes (name, URL, email,\n"
    "description, etc.) — you MUST call ``research_orchestrator`` FIRST,\n"
    "before any delegation.  The orchestrator handles the whole matrix in\n"
    "one structured pass with partial streaming, per-domain circuit breakers,\n"
    "known-hard field short-circuits, and a budget.  Do NOT use delegation\n"
    "for matrix tasks — the per-subject-per-field reasoning burns budget\n"
    "on work the orchestrator does in parallel.\n"
    "\n"
    "Matrix task recognition:\n"
    "  • \"give me a table of X for Y companies\" → orchestrator\n"
    "  • \"find email + URL + LinkedIn for N entities\" → orchestrator\n"
    "  • \"compare features of N products across M criteria\" → orchestrator\n"
    "\n"
    "DELEGATION MODE — for single-topic deep research (not matrix tasks):\n"
    "You coordinate a team of three specialists:\n"
    "  • Web Research Specialist — for live web content and URLs\n"
    "  • Document Research Specialist — for PDFs, KB passages, journal entries\n"
    "  • Synthesis Specialist — for final dialectical integration\n"
    "Use the delegate_work tool to dispatch sub-queries.  Give each "
    "specialist a focused, concrete instruction.  When you have enough raw "
    "material, delegate final integration to the Synthesis Specialist.  "
    "Return ONLY the final synthesised answer to the user — not a running "
    "commentary of your delegations."
)


def create_research_coordinator(force_tier: str | None = None) -> Agent:
    # 8192 tokens because the coordinator is the final voice to the user
    # when it chooses to answer directly (simple classifications) — 4096
    # truncated mid-section on deep-research prompts.
    llm = create_specialist_llm(max_tokens=8192, role="research", force_tier=force_tier)
    tools: list = []

    # ── Research orchestrator — FIRST in the tool list ────────────────
    # LLMs bias toward earlier-listed tools; we want matrix tasks to
    # pick this instantly instead of reaching for delegation or MCP
    # search.  The 2026-04-24 task #75 (coordinator on a weaker-tier
    # model after credit failover) ignored the MATRIX MODE backstory
    # rule and spent 15 min on MCP-server installation before the
    # orchestrator ever got considered.  Tool-order priority is the
    # simplest forcing function.
    try:
        from app.tools.research_orchestrator import research_orchestrator
        tools.append(research_orchestrator)
    except Exception:
        pass  # fail-soft: delegation path still works without the orchestrator

    # Coordinator needs memory access to remember context across delegations
    from app.tools.memory_tool import create_memory_tools
    from app.tools.scoped_memory_tool import create_scoped_memory_tools
    from app.tools.mem0_tools import create_mem0_tools
    tools.extend(create_memory_tools(collection="research"))
    tools.extend(create_scoped_memory_tools("researcher"))
    tools.extend(create_mem0_tools("researcher"))

    # A minimal web_search so simple follow-ups don't need a full delegation
    from app.tools.web_search import web_search
    tools.append(web_search)

    # read_attachment + file_manager so the coordinator can directly
    # read uploaded PDFs/CSVs and previous .md reports without needing
    # to delegate to a leaf specialist for basic file I/O.
    with optional_tool_group('specialists', 'file_manager'):
        from app.tools.file_manager import file_manager
        from app.tools.attachment_reader import read_attachment
        tools.extend([file_manager, read_attachment])

    return Agent(
        role="Research Coordinator",
        goal=(
            "Answer the user's research question by delegating sub-queries to "
            "Web, Document, and Synthesis specialists, then presenting the "
            "integrated result."
        ),
        backstory=_COORD_BACKSTORY,
        llm=llm,
        tools=tools,
        allow_delegation=True,  # CrewAI adds delegate_work + ask_question
        max_iter=10,
        verbose=False,
    )


# ── Coding: Execution specialist ─────────────────────────────────────
# Handles: actually running code, file I/O, sandbox, host bridge.

_EXEC_BACKSTORY = (
    "You are the Execution Specialist.  Given working (or draft) code, "
    "your job is to EXECUTE it: run it in the sandbox, capture stdout/"
    "stderr/return codes, write output files, and report the concrete "
    "results.  You don't design algorithms — you run them.  Prefer "
    "Docker sandbox execution over host shell unless the coordinator "
    "explicitly asks for host access.  Always return both the code you "
    "ran and its actual output."
)


# ── Coding: Design specialist ─────────────────────────────────────────
# Handles: producing a technical specification BEFORE any code is written.
# Splitting design from implementation reduces BadRequestError/TimeoutError
# on complex tasks where the model otherwise tries to think and code at
# the same time. The spec becomes the contract the implementer must satisfy.

_DESIGN_BACKSTORY = (
    "You are the Design Specialist.  When given a coding task, you produce "
    "a concise technical specification — NOT code — with these sections: "
    "summary, assumptions, file-by-file proposed changes, key APIs, error "
    "handling, testing plan, and risks.  Your spec is the contract the "
    "implementer follows.  You read the existing codebase for context, "
    "consult the experiential journal for past similar work, and check "
    "the tensions store for known contradictions.  Keep specs short — "
    "one screen of text is the target.  If a task is trivial, say so and "
    "produce a one-line spec."
)


def create_design_specialist(force_tier: str | None = None) -> Agent:
    llm = create_specialist_llm(max_tokens=4096, role="coding", force_tier=force_tier)
    tools: list = []

    # Read-only tooling — design is a thinking phase, not a writing phase.
    from app.tools.memory_tool import create_memory_tools
    from app.tools.scoped_memory_tool import create_scoped_memory_tools
    from app.tools.mem0_tools import create_mem0_tools
    from app.tools.file_manager import file_manager
    from app.tools.attachment_reader import read_attachment
    tools.extend(create_memory_tools(collection="coding"))
    tools.extend(create_scoped_memory_tools("coder"))
    tools.extend(create_mem0_tools("coder"))
    tools.extend([file_manager, read_attachment])
    _extend_specialist_domain_tools(tools)

    return Agent(
        role="Design Specialist",
        goal=(
            "Produce a technical specification for a coding task before "
            "any code is written, so the implementer has a clear contract "
            "to satisfy."
        ),
        backstory=_DESIGN_BACKSTORY,
        llm=llm,
        tools=tools,
        allow_delegation=False,
        max_iter=4,
        verbose=False,
    )


def create_execution_specialist(force_tier: str | None = None) -> Agent:
    llm = create_specialist_llm(max_tokens=4096, role="coding", force_tier=force_tier)
    tools: list = []
    from app.tools.file_manager import file_manager
    from app.tools.attachment_reader import read_attachment
    tools.extend([file_manager, read_attachment])

    # Code execution tools
    for mod, fn, args in [
        ("app.tools.code_execution", "create_code_tools", ()),
        ("app.tools.sandbox_tools", "create_sandbox_tools", ()),
        ("app.tools.bridge_tools", "create_bridge_tools", ("coder",)),
    ]:
        tools.extend(_safe(
            lambda m=mod, f=fn, a=args: __import__(m, fromlist=[f]).__dict__[f](*a),
        ))
    _extend_specialist_domain_tools(tools)

    return Agent(
        role="Execution Specialist",
        goal="Run code and capture real output (stdout, stderr, artifacts).",
        backstory=_EXEC_BACKSTORY,
        llm=llm,
        tools=tools,
        allow_delegation=False,
        max_iter=6,
        verbose=False,
    )


# ── Coding: Debug specialist ─────────────────────────────────────────
# Handles: reading past error patterns, tension records, failure
# journals, and team decisions to inform debugging.

_DEBUG_BACKSTORY = (
    "You are the Debug Specialist.  When the Execution Specialist reports "
    "a failure, you diagnose WHY.  You read the experiential journal for "
    "similar past failures, consult the tensions store for documented "
    "contradictions, check team memory for decisions made on adjacent "
    "code, and propose targeted fixes.  You don't run code yourself — "
    "you return a DIAGNOSIS + specific line-level suggestions."
)


def create_debug_specialist(force_tier: str | None = None) -> Agent:
    llm = create_specialist_llm(max_tokens=4096, role="coding", force_tier=force_tier)
    tools: list = []
    from app.knowledge_base.tools import KnowledgeSearchTool
    tools.append(KnowledgeSearchTool())

    # File + attachment access — the coordinator and execution specialist
    # both have these; the debug specialist historically did not, so when
    # the coding crew was (mis-)routed a file-heavy task and delegated to
    # Debug for "what's going wrong", Debug had no way to actually look
    # at the input file.  Same pattern as the 2026-04-24 desktop-agent fix.
    from app.tools.file_manager import file_manager
    from app.tools.attachment_reader import read_attachment
    tools.extend([file_manager, read_attachment])

    # Experiential, tensions, team decisions
    for mod, fn, args in [
        ("app.experiential.tools", "get_experiential_tools", ("coder",)),
        ("app.tensions.tools", "get_tension_tools", ("coder",)),
        ("app.episteme.tools", "get_episteme_tools", ("coder",)),
    ]:
        tools.extend(_safe(
            lambda m=mod, f=fn, a=args: __import__(m, fromlist=[f]).__dict__[f](*a),
        ))

    # Web search — for known-error patterns ("stack trace ...")
    from app.tools.web_search import web_search
    tools.append(web_search)
    _extend_specialist_domain_tools(tools)

    return Agent(
        role="Debug Specialist",
        goal="Diagnose code failures by cross-referencing journal, tensions, and past decisions.",
        backstory=_DEBUG_BACKSTORY,
        llm=llm,
        tools=tools,
        allow_delegation=False,
        max_iter=6,
        verbose=False,
    )


# ── Coding coordinator ───────────────────────────────────────────────

_CODE_COORD_BACKSTORY = compose_backstory("coder") + (
    "\n\n"
    "DELEGATION MODE — You coordinate a coding team of two specialists:\n"
    "  • Execution Specialist — runs code in the sandbox and returns actual output\n"
    "  • Debug Specialist — diagnoses failures using journal, tensions, past decisions\n"
    "Your job: write the code yourself, then delegate EXECUTION to verify it works.  "
    "If execution fails, delegate to Debug for a diagnosis, apply the suggested fix, "
    "and re-delegate execution.  Return the final working code with its real output.  "
    "Do NOT narrate your delegations — just deliver the working solution."
)


def create_coding_coordinator(force_tier: str | None = None) -> Agent:
    llm = create_specialist_llm(max_tokens=4096, role="coding", force_tier=force_tier)
    tools: list = []

    from app.tools.memory_tool import create_memory_tools
    from app.tools.scoped_memory_tool import create_scoped_memory_tools
    from app.tools.mem0_tools import create_mem0_tools
    from app.tools.file_manager import file_manager
    from app.tools.attachment_reader import read_attachment
    tools.extend(create_memory_tools(collection="coding"))
    tools.extend(create_scoped_memory_tools("coder"))
    tools.extend(create_mem0_tools("coder"))
    tools.extend([file_manager, read_attachment])
    _extend_specialist_domain_tools(tools)

    return Agent(
        role="Coding Coordinator",
        goal=(
            "Design the code, delegate its execution to the Execution Specialist, "
            "consult the Debug Specialist on failures, and return working code "
            "with actual output."
        ),
        backstory=_CODE_COORD_BACKSTORY,
        llm=llm,
        tools=tools,
        allow_delegation=True,
        max_iter=12,
        verbose=False,
    )


# ── Writing: Research specialist (lighter than research-crew's Web+Doc split) ─

_WRITE_RESEARCH_BACKSTORY = (
    "You are the Writing Research Specialist.  Given a writing task, you "
    "gather the necessary factual material: quick web searches for current "
    "facts, knowledge-base lookups for company/product data, and philosophy "
    "KB for thematic depth.  You return a structured research brief — NOT "
    "the final prose.  The Synthesis Specialist turns your brief into the "
    "finished piece."
)


def create_writing_research_specialist(force_tier: str | None = None) -> Agent:
    llm = create_specialist_llm(max_tokens=4096, role="writing", force_tier=force_tier)
    tools: list = []
    from app.tools.web_search import web_search
    from app.tools.web_fetch import web_fetch
    from app.tools.attachment_reader import read_attachment
    from app.knowledge_base.tools import KnowledgeSearchTool
    tools.extend([web_search, web_fetch, read_attachment, KnowledgeSearchTool()])

    # Philosophy + episteme
    for mod, fn, args in [
        ("app.philosophy.rag_tool", "PhilosophyRAGTool", ()),
        ("app.episteme.tools", "get_episteme_tools", ("writer",)),
        ("app.experiential.tools", "get_experiential_tools", ("writer",)),
    ]:
        with optional_tool_group('specialists', 'unknown'):
            cls_or_fn = __import__(mod, fromlist=[fn]).__dict__[fn]
            if isinstance(cls_or_fn, type):
                tools.append(cls_or_fn())
            else:
                tools.extend(_safe(lambda c=cls_or_fn, a=args: c(*a)))

    return Agent(
        role="Writing Research Specialist",
        goal="Gather factual material and return a structured brief.",
        backstory=_WRITE_RESEARCH_BACKSTORY,
        llm=llm,
        tools=tools,
        allow_delegation=False,
        max_iter=6,
        verbose=False,
    )


# ── Writing coordinator ──────────────────────────────────────────────
# For writing we reuse the Synthesis Specialist defined above (it's
# already tuned for integrating material into final prose).

_WRITE_COORD_BACKSTORY = compose_backstory("writer") + (
    "\n\n"
    "DELEGATION MODE — You coordinate a writing team of two specialists:\n"
    "  • Writing Research Specialist — gathers facts and builds a brief\n"
    "  • Synthesis Specialist — integrates material into finished prose\n"
    "Simple writing tasks (short emails, concise notes) you handle yourself.  "
    "For longer/substantive pieces: delegate the Research Specialist to gather "
    "sources and build a brief, then delegate the Synthesis Specialist to "
    "produce the final draft.  Revise as needed.  Return the finished piece "
    "only — no delegation commentary."
)


def create_writing_coordinator(force_tier: str | None = None) -> Agent:
    # 8192 so long-form writing (reports, essays) isn't cut off mid-section.
    llm = create_specialist_llm(max_tokens=8192, role="writing", force_tier=force_tier)
    tools: list = []

    from app.tools.memory_tool import create_memory_tools
    from app.tools.scoped_memory_tool import create_scoped_memory_tools
    from app.tools.mem0_tools import create_mem0_tools
    from app.tools.file_manager import file_manager
    tools.extend(create_memory_tools(collection="writing"))
    tools.extend(create_scoped_memory_tools("writer"))
    tools.extend(create_mem0_tools("writer"))
    tools.append(file_manager)

    return Agent(
        role="Writing Coordinator",
        goal=(
            "Produce finished writing by delegating research and synthesis "
            "to specialists, then polishing the result."
        ),
        backstory=_WRITE_COORD_BACKSTORY,
        llm=llm,
        tools=tools,
        allow_delegation=True,
        max_iter=10,
        verbose=False,
    )
