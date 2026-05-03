"""researcher.py — Researcher agent (web + KB + Mem0 + episteme).

Phase 4a migration target for the dynamic-tool-loading architecture.
Migration is **opt-in via env flag**: ``LOADABLE_RESEARCHER=1`` (or
the master ``LOADABLE_AGENT_EXPERIMENTAL=1``). Default behavior
stays on the stock crewai.Agent path. Failsafe: experimental factory
exception → fallback to legacy.

Migration scope:
  * The full path (``light=False``) — 30+ tools, where LoadableAgent
    provides the most token-cost win — is the migration target.
  * The light path (``light=True``) — 5-10 tools — stays on the
    legacy factory unchanged. The cost overhead of the registry is
    not worth it for the light path's already-small tool surface.

Per-agent flag resolution (see app/tool_runtime/feature_flags.py):
  ``LOADABLE_RESEARCHER`` overrides ``LOADABLE_AGENT_EXPERIMENTAL``
  when set. So an operator running with ``LOADABLE_AGENT_EXPERIMENTAL=1``
  can keep researcher on stock by setting ``LOADABLE_RESEARCHER=0``.
"""
import logging

from crewai import Agent
from app.agents._common import optional_tool_group
from app.config import get_settings

logger = logging.getLogger(__name__)
from app.llm_factory import create_specialist_llm
from app.tools.web_search import web_search
from app.tools.web_fetch import web_fetch
from app.tools.youtube_transcript import get_youtube_transcript
from app.tools.memory_tool import create_memory_tools
from app.tools.scoped_memory_tool import create_scoped_memory_tools
from app.tools.file_manager import file_manager
from app.tools.attachment_reader import read_attachment
from app.tools.self_report_tool import create_self_report_tool
from app.tools.reflection_tool import ReflectionTool
from app.souls.loader import compose_backstory
from app.tools.mem0_tools import create_mem0_tools
from app.knowledge_base.tools import KnowledgeSearchTool

settings = get_settings()

RESEARCHER_BACKSTORY = compose_backstory("researcher")

# S9: Compact backstory for simple tasks — just the essential identity and rules.
# Saves ~2000 tokens per LLM call (×4-6 calls = 8000-12000 tokens per task).
_COMPACT_RESEARCHER_BACKSTORY = (
    "You are a Researcher — an intelligence analyst who finds, verifies, and "
    "synthesizes information from web sources. You distrust information by default "
    "and verify across multiple sources. Never fabricate URLs or data. "
    "Distinguish between verified fact, inference, and speculation.\n"
    "CRITICAL: You HAVE web browsing tools (web_search, web_fetch). You CAN "
    "and MUST use them to answer questions about current events, live websites, "
    "or any information you don't already know. NEVER say 'I cannot browse the "
    "internet' — you can. Use your tools.\n"
    "MATRIX TASKS: If the user asks for a table / matrix of data on 3+ entities "
    "with 2+ attributes each, escalate — say 'this needs the full researcher' "
    "and stop. Do NOT try to fill the matrix with hand-chained web_search calls "
    "on the light path; you will time out. The full researcher has a "
    "`research_orchestrator` tool designed for this exact shape.\n"
    "Any <reference_context> block in the task is silent background (current date, "
    "season, user location). Use it only if the user's question depends on 'now' or "
    "'here'. Never mention, quote, describe, or reason aloud about it — the user "
    "did not write it and does not see it. Answer the user's actual question only."
)


def create_researcher(force_tier: str | None = None, light: bool = False, task_id: str = "") -> Agent:
    """Create a researcher agent.

    Phase 4a: dispatches between legacy (stock CrewAI) and LoadableAgent
    based on ``LOADABLE_RESEARCHER`` (or master ``LOADABLE_AGENT_EXPERIMENTAL``)
    env. Default OFF; light path always uses legacy regardless of the
    flag (its small tool surface doesn't benefit from registry overhead).

    Args:
        force_tier: Override model tier selection.
        light: If True, use minimal tools and compact backstory (S8/S9).
               Used for difficulty ≤ 3 simple factual questions.
        task_id: If set, adds blackboard tools scoped to this task.
    """
    # Light path: always legacy. The compact backstory + 5-10 tools
    # already avoid the bloat that LoadableAgent solves for; running
    # registry plumbing here would add overhead without payoff.
    if light:
        return _legacy_create_researcher(force_tier=force_tier, light=True, task_id=task_id)

    # Full path: dispatch on flag.
    from app.tool_runtime.feature_flags import is_loadable_for
    if is_loadable_for("researcher"):
        try:
            agent = _build_loadable_researcher(force_tier=force_tier, task_id=task_id)
            logger.info("researcher: built LoadableAgent (Phase 4a experimental path)")
            return agent
        except Exception as exc:
            logger.warning(
                "researcher: LoadableAgent path failed (%s) — falling back to legacy. "
                "Set LOADABLE_RESEARCHER=0 to silence this until the issue is fixed.",
                exc, exc_info=True,
            )
    return _legacy_create_researcher(force_tier=force_tier, light=False, task_id=task_id)


def _legacy_create_researcher(force_tier: str | None = None, light: bool = False, task_id: str = "") -> Agent:
    """Stock-CrewAI researcher — kept as the default path during
    Phase 4a soak. Once the LoadableAgent variant is validated across
    a parity panel, this becomes the fallback for outages."""
    # max_tokens = 8192 so the researcher can produce an extensive report
    # (the user often asks for "extensive document with numbers and
    # commentaries").  At 4096 the output was getting cut off mid-section
    # — vetting would reject "response is cut off mid-sentence in Section
    # 5.1" and reflexion retry re-ran the whole 6-min crew producing the
    # same cut-off result.  Claude Sonnet/Opus models accept 8K natively.
    llm = create_specialist_llm(max_tokens=8192, role="research", force_tier=force_tier)

    if light:
        # S8→S10: "Medium-light" — enough tools for most factual questions
        # without the full 40+ tool overhead.  Adds file/attachment access,
        # Mem0 conversational context, and Firecrawl so the agent can actually
        # answer current-events and web-content questions routed as difficulty ≤ 3.
        mem0_tools = create_mem0_tools("researcher")
        tools = [web_search, web_fetch, file_manager, read_attachment, KnowledgeSearchTool()] + mem0_tools
        # Firecrawl search+scrape
        with optional_tool_group("researcher", "firecrawl"):
            from app.tools.firecrawl_tools import create_firecrawl_tools
            fc = create_firecrawl_tools()
            if fc:
                tools.extend(fc[:2])  # scrape + search (keep tool count low)
        backstory = _COMPACT_RESEARCHER_BACKSTORY
    else:
        # R5: Removed world_model_tool (write-only, never read) and self-awareness
        # tools (self_report, reflection) — telemetry now runs in post-crew hook.
        # This saves ~600 tokens per LLM call (3 fewer tool descriptions).
        memory_tools = create_memory_tools(collection="researcher")
        scoped_tools = create_scoped_memory_tools("researcher")
        mem0_tools = create_mem0_tools("researcher")
        # New KB tools (Phase 2/3): episteme for research grounding + journal for past experience.
        from app.episteme.tools import get_episteme_tools
        from app.experiential.tools import get_experiential_tools
        tools = [web_search, web_fetch, get_youtube_transcript, file_manager, read_attachment, KnowledgeSearchTool()] + memory_tools + scoped_tools + mem0_tools + get_episteme_tools() + get_experiential_tools("researcher")
        # Firecrawl tools (scrape, extract, search, map) — gracefully empty if unavailable
        with optional_tool_group("researcher", "firecrawl"):
            from app.tools.firecrawl_tools import create_firecrawl_tools
            fc_tools = create_firecrawl_tools()
            if fc_tools:
                tools.extend(fc_tools[:4])  # scrape, search, extract, map (not crawl)
        # Composio SaaS tools (Gmail, GitHub, Slack, etc.) — gracefully empty if unavailable
        with optional_tool_group("researcher", "composio"):
            from app.tools.composio_tool import get_composio_tools
            composio_tools = get_composio_tools()
            if composio_tools:
                tools.extend(composio_tools[:10])
        # Host Bridge tools (read/write host files, execute commands, LAN access)
        with optional_tool_group("researcher", "bridge"):
            from app.tools.bridge_tools import create_bridge_tools
            bridge_tools = create_bridge_tools("researcher")
            if bridge_tools:
                tools.extend(bridge_tools)
        # Wiki tools (read, write, search — researcher is primary wiki contributor)
        with optional_tool_group("researcher", "wiki"):
            from app.tools.wiki_tool_registry import create_wiki_tools
            wiki_tools = create_wiki_tools("read", "write", "search", "slides")
            if wiki_tools:
                tools.extend(wiki_tools)
        # Blackboard tools (deposit/read findings — P1 research synthesis)
        if task_id:
            with optional_tool_group("researcher", "blackboard"):
                from app.tools.blackboard_tool import create_blackboard_tools
                bb_tools = create_blackboard_tools(task_id, "researcher")
                tools.extend(bb_tools)
        # Tension tools — researcher records contradictions found during research
        with optional_tool_group("researcher", "tensions"):
            from app.tensions.tools import get_tension_tools
            tools.extend(get_tension_tools("researcher"))
        # OCR tool — extract text from images (screenshots, documents, receipts)
        with optional_tool_group("researcher", "ocr"):
            from app.tools.ocr_tool import create_ocr_tool
            ocr = create_ocr_tool()
            if ocr:
                tools.append(ocr)
        # Research orchestrator — structured (subjects × fields) matrix
        # research with partial streaming + per-domain circuit breakers.
        # Prevents the "LLM retry-loop burns 45 min for no deliverable"
        # failure mode when the user asks for a table of N companies × M
        # attributes.  See app/tools/research_orchestrator.py.
        with optional_tool_group("researcher", "research_orchestrator"):
            from app.tools.research_orchestrator import research_orchestrator
            tools.append(research_orchestrator)
        backstory = RESEARCHER_BACKSTORY

    return Agent(
        role="Researcher",
        goal="Find accurate, comprehensive information on any topic using web search, article reading, and YouTube transcripts.",
        backstory=backstory,
        llm=llm,
        tools=tools,
        max_execution_time=300,
        verbose=True,
    )


def _build_loadable_researcher(
    *, force_tier: str | None = None, task_id: str = "",
) -> Agent:
    """Phase 4a path — LoadableAgent backed by the tool registry.

    Same backstory + LLM + per-agent tools as the legacy full path.
    The 30+ existing tools stay as eager core (most are per-agent
    state — filtered by ``collection="researcher"`` — and not yet
    registry-annotated). The agent gains discoverable capabilities
    via ``load_tool`` / ``tool_search`` for catalog tools the
    researcher might want to pull on demand.

    The eager set mirrors the legacy ``light=False`` build exactly,
    so behavior parity is high — what changes is the *discoverability*
    of additional tools and the cache cost of running this agent
    (Phase 1c predicted ~33% of stock).
    """
    from app.tool_registry import Tier
    from app.tool_runtime.factory import build_loadable_agent

    llm = create_specialist_llm(max_tokens=8192, role="research", force_tier=force_tier)

    # Mirror the full-path eager toolset.
    memory_tools = create_memory_tools(collection="researcher")
    scoped_tools = create_scoped_memory_tools("researcher")
    mem0_tools = create_mem0_tools("researcher")
    from app.episteme.tools import get_episteme_tools
    from app.experiential.tools import get_experiential_tools

    eager: list = [
        web_search, web_fetch, get_youtube_transcript,
        file_manager, read_attachment, KnowledgeSearchTool(),
    ] + memory_tools + scoped_tools + mem0_tools \
      + get_episteme_tools() + get_experiential_tools("researcher")

    # Optional groups — preserve legacy graceful-degradation semantics.
    with optional_tool_group("researcher", "firecrawl"):
        from app.tools.firecrawl_tools import create_firecrawl_tools
        fc_tools = create_firecrawl_tools()
        if fc_tools:
            eager.extend(fc_tools[:4])
    with optional_tool_group("researcher", "composio"):
        from app.tools.composio_tool import get_composio_tools
        composio_tools = get_composio_tools()
        if composio_tools:
            eager.extend(composio_tools[:10])
    with optional_tool_group("researcher", "bridge"):
        from app.tools.bridge_tools import create_bridge_tools
        bridge_tools = create_bridge_tools("researcher")
        if bridge_tools:
            eager.extend(bridge_tools)
    with optional_tool_group("researcher", "wiki"):
        from app.tools.wiki_tool_registry import create_wiki_tools
        wiki_tools = create_wiki_tools("read", "write", "search", "slides")
        if wiki_tools:
            eager.extend(wiki_tools)
    if task_id:
        with optional_tool_group("researcher", "blackboard"):
            from app.tools.blackboard_tool import create_blackboard_tools
            eager.extend(create_blackboard_tools(task_id, "researcher"))
    with optional_tool_group("researcher", "tensions"):
        from app.tensions.tools import get_tension_tools
        eager.extend(get_tension_tools("researcher"))
    with optional_tool_group("researcher", "ocr"):
        from app.tools.ocr_tool import create_ocr_tool
        ocr = create_ocr_tool()
        if ocr:
            eager.append(ocr)
    with optional_tool_group("researcher", "research_orchestrator"):
        from app.tools.research_orchestrator import research_orchestrator
        eager.append(research_orchestrator)

    return build_loadable_agent(
        role="Researcher",
        goal="Find accurate, comprehensive information on any topic using web search, article reading, and YouTube transcripts.",
        backstory=RESEARCHER_BACKSTORY,
        llm=llm,
        agent_id="researcher",
        # Eager: every tool the legacy path would have surfaced.
        # Phase 5 sweeps the per-agent tools into PER_AGENT registry
        # entries and shrinks this list to ~5 universal tools.
        core_tools=eager,
        # Discoverable: catalog tools the researcher might want.
        # Forge-bridged SHADOW/CANARY tools tagged ``registers-tool``
        # also surface here, so the researcher can pick up any
        # operator-promoted tool without an agent rewrite.
        discoverable_capabilities=[
            "renders-pdf",            # synthesize report PDFs
            "sends-signal",            # deliver findings to the user
            "renders-chart",           # visualize comparisons
            "fetches-geodata",         # map / region queries
            "executes-code",           # numeric processing of findings
        ],
        agent_tier=Tier.PRODUCTION,
        max_execution_time=300,
        verbose=True,
    )
