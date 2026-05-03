"""coder.py — Coder agent (code execution + sandboxes + Forge).

Phase 4c migration target — the highest-stakes Phase 4 migration so
far. The coder runs code in the sandbox, calls Forge to generate
new tools, and produces user-deliverable artifacts (PDFs over Signal).
A migration regression here is more visible than researcher/writer.

Migration is **opt-in via env flag**: ``LOADABLE_CODER=1`` (or
master ``LOADABLE_AGENT_EXPERIMENTAL=1``). Default behavior stays
on the stock crewai.Agent path. Failsafe: experimental factory
exception → fallback to legacy.

Per-agent flag resolution (see app/tool_runtime/feature_flags.py):
  ``LOADABLE_CODER`` overrides ``LOADABLE_AGENT_EXPERIMENTAL`` when
  set. So an operator running with ``LOADABLE_AGENT_EXPERIMENTAL=1``
  can keep coder on stock by setting ``LOADABLE_CODER=0``.
"""
import logging

from crewai import Agent
from app.agents._common import optional_tool_group
from app.config import get_settings
from app.llm_factory import create_specialist_llm
from app.tools.code_executor import execute_code
from app.tools.file_manager import file_manager
from app.tools.web_search import web_search
from app.tools.memory_tool import create_memory_tools
from app.tools.scoped_memory_tool import create_scoped_memory_tools
from app.tools.attachment_reader import read_attachment
from app.souls.loader import compose_backstory
from app.tools.mem0_tools import create_mem0_tools
from app.knowledge_base.tools import KnowledgeSearchTool
from app.fiction_inspiration import get_fiction_tools, FICTION_AWARENESS_PROMPT

logger = logging.getLogger(__name__)
settings = get_settings()

CODER_BACKSTORY = compose_backstory("coder") + FICTION_AWARENESS_PROMPT


def create_coder(force_tier: str | None = None) -> Agent:
    """Create a coder agent.

    Phase 4c: dispatches between legacy (stock CrewAI) and LoadableAgent
    based on ``LOADABLE_CODER`` (or master ``LOADABLE_AGENT_EXPERIMENTAL``)
    env. Default OFF; failsafe falls back to legacy on factory error.
    """
    from app.tool_runtime.feature_flags import is_loadable_for
    if is_loadable_for("coder"):
        try:
            agent = _build_loadable_coder(force_tier=force_tier)
            logger.info("coder: built LoadableAgent (Phase 4c experimental path)")
            return agent
        except Exception as exc:
            logger.warning(
                "coder: LoadableAgent path failed (%s) — falling back to legacy. "
                "Set LOADABLE_CODER=0 to silence this until the issue is fixed.",
                exc, exc_info=True,
            )
    return _legacy_create_coder(force_tier=force_tier)


def _legacy_create_coder(force_tier: str | None = None) -> Agent:
    """Stock-CrewAI coder — kept as the default path during Phase 4c
    soak. Once the LoadableAgent variant passes its parity panel,
    this becomes the fallback for outages.

    Highest-stakes migration in Phase 4 — the coder runs code, calls
    Forge, produces user-deliverable PDFs. Keeping legacy as fallback
    means a Phase 4c bug never breaks the user-facing flow.
    """
    llm = create_specialist_llm(max_tokens=4096, role="coding", force_tier=force_tier)
    # R5: Self-awareness tools removed — telemetry runs in post-crew hook
    memory_tools = create_memory_tools(collection="coder")
    scoped_tools = create_scoped_memory_tools("coder")
    mem0_tools = create_mem0_tools("coder")

    # New KB tools (Phase 2/3): aesthetics for elegant code patterns + journal for past experience.
    from app.aesthetics.tools import get_aesthetic_tools
    from app.experiential.tools import get_experiential_tools
    tools = [execute_code, file_manager, web_search, read_attachment, KnowledgeSearchTool()] + memory_tools + scoped_tools + mem0_tools + get_fiction_tools() + get_aesthetic_tools("coder") + get_experiential_tools("coder")
    # Tension tools — coder records conflicts between approaches (e.g. speed vs readability)
    with optional_tool_group("coder", "tensions"):
        from app.tensions.tools import get_tension_tools
        tools.extend(get_tension_tools("coder"))
    # Host Bridge tools (read/write host files, execute commands on Mac)
    with optional_tool_group("coder", "bridge"):
        from app.tools.bridge_tools import create_bridge_tools
        bridge_tools = create_bridge_tools("coder")
        if bridge_tools:
            tools.extend(bridge_tools)
    # Wiki tools (read, write — coder updates technical architecture pages)
    with optional_tool_group("coder", "wiki"):
        from app.tools.wiki_tool_registry import create_wiki_tools
        tools.extend(create_wiki_tools("read", "write"))
    # Google Earth Engine — satellite-imagery analysis for forest /
    # land-use questions. Tool only registers when
    # GOOGLE_APPLICATION_CREDENTIALS points at a readable service-
    # account JSON; otherwise create_gee_tools returns []. See
    # app/tools/gee_tool.py + .env.example for the setup walkthrough.
    with optional_tool_group("coder", "gee"):
        from app.tools.gee_tool import create_gee_tools
        tools.extend(create_gee_tools("coder"))
    # PDF composition + Signal delivery — closes the loop "agent
    # produces a PDF" → "user opens the PDF on their phone."
    # pdf_compose runs matplotlib + reportlab in-process and writes
    # to /app/workspace/output/; signal_send_attachment maps those
    # files to host paths and hands them to signal-cli. Pre-2026-05-03
    # the agent had matplotlib + reportlab installed but no tool
    # surface, so it wrote Python source as text and never actually
    # delivered the PDF — that's the gap these two tools close.
    with optional_tool_group("coder", "pdf"):
        from app.tools.pdf_compose import create_pdf_tools
        tools.extend(create_pdf_tools("coder"))
    with optional_tool_group("coder", "signal_attachment"):
        from app.tools.signal_attachment import create_signal_attachment_tools
        tools.extend(create_signal_attachment_tools("coder"))
    # tool_search — discovery primitive (Phase 1b). Lets the agent
    # query the registry by capability tag + intent before assuming
    # a tool exists. Read-only; Phase 2 promotes it to auto-load.
    with optional_tool_group("coder", "tool_search"):
        from app.tools.tool_search import create_tool_search_tools
        tools.extend(create_tool_search_tools("coder"))
    # Forge generator — only exposed when both TOOL_FORGE_ENABLED and
    # TOOL_FORGE_AGENT_GENERATION_ENABLED are set. Lets Coder register a new
    # sandboxed tool through the audit pipeline. Tool lands in SHADOW at best;
    # promotion past SHADOW requires manual human approval.
    with optional_tool_group("coder", "forge_generator"):
        from app.forge.generator_tool import get_forge_generator_tool
        forge_tool = get_forge_generator_tool()
        if forge_tool is not None:
            tools.append(forge_tool)

    return Agent(
        role="Coder",
        goal="Write, test, and debug code across any language. Execute code safely in a Docker sandbox.",
        backstory=CODER_BACKSTORY,
        llm=llm,
        tools=tools,
        max_execution_time=300,
        verbose=True,
    )


def _build_loadable_coder(*, force_tier: str | None = None) -> Agent:
    """Phase 4c path — LoadableAgent backed by the tool registry.

    Same backstory + LLM + per-agent tools as the legacy factory.
    Eager toolset mirrors the legacy build exactly so behavior parity
    is high — what changes is the *discoverability* of additional
    tools (catalog tools the coder might want pulled in mid-task) and
    the cache cost (Phase 1c predicted ~33% of stock).

    Coder-specific discoverable capabilities:
      * ``fetches-geodata`` — geodata fetcher (geographic processing).
      * ``converts-currency`` / ``fetches-finance`` — financial data
        munging that the coder doesn't ship with eagerly.
      * ``registers-tool`` — Forge-bridged SHADOW/CANARY tools surface
        here automatically (Phase 3 bridge), so the coder can pick up
        any operator-promoted Forge tool without an agent rewrite.
    """
    from app.tool_registry import Tier
    from app.tool_runtime.factory import build_loadable_agent

    llm = create_specialist_llm(max_tokens=4096, role="coding", force_tier=force_tier)

    # Mirror the legacy eager toolset.
    memory_tools = create_memory_tools(collection="coder")
    scoped_tools = create_scoped_memory_tools("coder")
    mem0_tools = create_mem0_tools("coder")

    from app.aesthetics.tools import get_aesthetic_tools
    from app.experiential.tools import get_experiential_tools

    eager: list = [
        execute_code, file_manager, web_search,
        read_attachment, KnowledgeSearchTool(),
    ] + memory_tools + scoped_tools + mem0_tools \
      + get_fiction_tools() + get_aesthetic_tools("coder") \
      + get_experiential_tools("coder")

    # Optional groups — preserve legacy graceful-degradation semantics.
    with optional_tool_group("coder", "tensions"):
        from app.tensions.tools import get_tension_tools
        eager.extend(get_tension_tools("coder"))
    with optional_tool_group("coder", "bridge"):
        from app.tools.bridge_tools import create_bridge_tools
        bridge_tools = create_bridge_tools("coder")
        if bridge_tools:
            eager.extend(bridge_tools)
    with optional_tool_group("coder", "wiki"):
        from app.tools.wiki_tool_registry import create_wiki_tools
        eager.extend(create_wiki_tools("read", "write"))
    with optional_tool_group("coder", "gee"):
        from app.tools.gee_tool import create_gee_tools
        eager.extend(create_gee_tools("coder"))
    with optional_tool_group("coder", "pdf"):
        from app.tools.pdf_compose import create_pdf_tools
        eager.extend(create_pdf_tools("coder"))
    with optional_tool_group("coder", "signal_attachment"):
        from app.tools.signal_attachment import create_signal_attachment_tools
        eager.extend(create_signal_attachment_tools("coder"))
    with optional_tool_group("coder", "tool_search"):
        from app.tools.tool_search import create_tool_search_tools
        eager.extend(create_tool_search_tools("coder"))
    with optional_tool_group("coder", "forge_generator"):
        from app.forge.generator_tool import get_forge_generator_tool
        forge_tool = get_forge_generator_tool()
        if forge_tool is not None:
            eager.append(forge_tool)

    return build_loadable_agent(
        role="Coder",
        goal="Write, test, and debug code across any language. Execute code safely in a Docker sandbox.",
        backstory=CODER_BACKSTORY,
        llm=llm,
        agent_id="coder",
        # Eager: every tool the legacy path would have surfaced.
        # Phase 5 sweeps the per-agent tools into PER_AGENT registry
        # entries and shrinks this list to ~5 universal tools.
        core_tools=eager,
        # Discoverable: catalog tools NOT already in eager that the
        # coder might want on demand.
        discoverable_capabilities=[
            "fetches-geodata",        # geodata processing
            "converts-currency",      # currency conversions
            "fetches-finance",        # ECB / forex data
        ],
        agent_tier=Tier.PRODUCTION,
        max_execution_time=300,
        verbose=True,
    )
