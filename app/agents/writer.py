"""writer.py — Writer agent (creative + reports + documentation).

Phase 4b migration target. Migration is **opt-in via env flag**:
``LOADABLE_WRITER=1`` (or master ``LOADABLE_AGENT_EXPERIMENTAL=1``).
Default behavior stays on the stock crewai.Agent path. Failsafe:
experimental factory exception → fallback to legacy.

Per-agent flag resolution (see app/tool_runtime/feature_flags.py):
  ``LOADABLE_WRITER`` overrides ``LOADABLE_AGENT_EXPERIMENTAL`` when
  set. So an operator running with ``LOADABLE_AGENT_EXPERIMENTAL=1``
  can keep writer on stock by setting ``LOADABLE_WRITER=0``.
"""
import logging

from crewai import Agent
from app.agents._common import optional_tool_group
from app.config import get_settings
from app.llm_factory import create_specialist_llm
from app.tools.memory_tool import create_memory_tools
from app.tools.scoped_memory_tool import create_scoped_memory_tools
from app.tools.file_manager import file_manager
from app.tools.web_search import web_search
from app.tools.attachment_reader import read_attachment
from app.souls.loader import compose_backstory
from app.tools.mem0_tools import create_mem0_tools
from app.knowledge_base.tools import KnowledgeSearchTool
from app.philosophy.rag_tool import PhilosophyRAGTool
from app.fiction_inspiration import get_fiction_tools, FICTION_AWARENESS_PROMPT

logger = logging.getLogger(__name__)
settings = get_settings()

WRITER_BACKSTORY = compose_backstory("writer") + FICTION_AWARENESS_PROMPT


def create_writer(force_tier: str | None = None) -> Agent:
    """Create a writer agent.

    Phase 4b: dispatches between legacy (stock CrewAI) and LoadableAgent
    based on ``LOADABLE_WRITER`` (or master ``LOADABLE_AGENT_EXPERIMENTAL``)
    env. Default OFF; failsafe falls back to legacy on factory error.
    """
    from app.tool_runtime.feature_flags import is_loadable_for
    if is_loadable_for("writer"):
        try:
            agent = _build_loadable_writer(force_tier=force_tier)
            logger.info("writer: built LoadableAgent (Phase 4b experimental path)")
            return agent
        except Exception as exc:
            logger.warning(
                "writer: LoadableAgent path failed (%s) — falling back to legacy. "
                "Set LOADABLE_WRITER=0 to silence this until the issue is fixed.",
                exc, exc_info=True,
            )
    return _legacy_create_writer(force_tier=force_tier)


def _legacy_create_writer(force_tier: str | None = None) -> Agent:
    """Stock-CrewAI writer — kept as the default path during Phase 4b
    soak. Once the LoadableAgent variant passes its parity panel,
    this becomes the fallback for outages."""
    llm = create_specialist_llm(max_tokens=4096, role="writing", force_tier=force_tier)
    # R5: Self-awareness tools removed — telemetry runs in post-crew hook
    memory_tools = create_memory_tools(collection="writer")
    scoped_tools = create_scoped_memory_tools("writer")
    mem0_tools = create_mem0_tools("writer")

    # New KB tools (Phase 2/3): journal + aesthetics for quality-aware writing.
    from app.experiential.tools import get_experiential_tools
    from app.aesthetics.tools import get_aesthetic_tools
    tools = [file_manager, web_search, read_attachment, KnowledgeSearchTool(), PhilosophyRAGTool()] + memory_tools + scoped_tools + mem0_tools + get_fiction_tools() + get_experiential_tools("writer") + get_aesthetic_tools("writer")
    # Document generation tools (PDF, DOCX, HTML) — prose-shaped output.
    with optional_tool_group("writer", "document_generator"):
        from app.tools.document_generator import create_document_tools
        doc_tools = create_document_tools()
        if doc_tools:
            tools.extend(doc_tools)
    # Data-shaped PDF reports (matplotlib charts + reportlab tables).
    # Complementary to document_generator: that one is text-heavy
    # prose; pdf_compose is for when the writer is producing a
    # data-driven report with charts. Same tool the coder uses.
    with optional_tool_group("writer", "pdf"):
        from app.tools.pdf_compose import create_pdf_tools
        tools.extend(create_pdf_tools("writer"))
    # Deliver any artifact (PDF, CSV, etc.) over Signal.
    with optional_tool_group("writer", "signal_attachment"):
        from app.tools.signal_attachment import create_signal_attachment_tools
        tools.extend(create_signal_attachment_tools("writer"))
    # tool_search — discovery primitive (Phase 1b). Read-only.
    with optional_tool_group("writer", "tool_search"):
        from app.tools.tool_search import create_tool_search_tools
        tools.extend(create_tool_search_tools("writer"))
    # Host Bridge tools (read/write host files for document output)
    with optional_tool_group("writer", "bridge"):
        from app.tools.bridge_tools import create_bridge_tools
        bridge_tools = create_bridge_tools("writer")
        if bridge_tools:
            tools.extend(bridge_tools)
    # Wiki tools (read, write, search — writer consumes wiki + files valuable outputs back)
    with optional_tool_group("writer", "wiki"):
        from app.tools.wiki_tool_registry import create_wiki_tools
        tools.extend(create_wiki_tools("read", "write", "search", "slides"))
    # Conceptual blending tool (Mechanism 6) — writer is the analogical-blending
    # agent in creative mode; this tool operationalizes philosophy+fiction fusion
    # with explicit [PIT]/[PIH] epistemic tags.
    with optional_tool_group("writer", "blend"):
        from app.tools.blend_tool import ConceptBlendTool
        tools.append(ConceptBlendTool())
    # Dialectics tool — writer challenges own arguments via counter-argument graph
    with optional_tool_group("writer", "dialectics"):
        from app.philosophy.dialectics_tool import FindCounterArgumentTool
        tools.append(FindCounterArgumentTool())
    # Tension tools — writer notices and records creative/structural tensions
    with optional_tool_group("writer", "tensions"):
        from app.tensions.tools import get_tension_tools
        tools.extend(get_tension_tools("writer"))

    return Agent(
        role="Writer",
        goal="Write clear, well-structured content including summaries, reports, documentation, and emails.",
        backstory=WRITER_BACKSTORY,
        llm=llm,
        tools=tools,
        max_execution_time=300,
        verbose=True,
    )


def _build_loadable_writer(*, force_tier: str | None = None) -> Agent:
    """Phase 4b path — LoadableAgent backed by the tool registry.

    Same backstory + LLM + per-agent tools as the legacy factory.
    The 30+ existing tools stay as eager core (most are per-agent
    state — filtered by ``collection="writer"`` — and not yet
    registry-annotated). The agent gains discoverable capabilities
    via ``load_tool`` / ``tool_search`` for catalog tools the
    writer might want to pull on demand.

    The eager set mirrors the legacy build exactly, so behavior
    parity is high — what changes is the *discoverability* of
    additional tools and the cache cost of running this agent
    (Phase 1c predicted ~33% of stock).
    """
    from app.tool_registry import Tier
    from app.tool_runtime.factory import build_loadable_agent

    llm = create_specialist_llm(max_tokens=4096, role="writing", force_tier=force_tier)

    # Mirror the legacy eager toolset.
    memory_tools = create_memory_tools(collection="writer")
    scoped_tools = create_scoped_memory_tools("writer")
    mem0_tools = create_mem0_tools("writer")

    from app.experiential.tools import get_experiential_tools
    from app.aesthetics.tools import get_aesthetic_tools

    eager: list = [
        file_manager, web_search, read_attachment,
        KnowledgeSearchTool(), PhilosophyRAGTool(),
    ] + memory_tools + scoped_tools + mem0_tools \
      + get_fiction_tools() + get_experiential_tools("writer") \
      + get_aesthetic_tools("writer")

    # Optional groups — preserve legacy graceful-degradation semantics.
    with optional_tool_group("writer", "document_generator"):
        from app.tools.document_generator import create_document_tools
        doc_tools = create_document_tools()
        if doc_tools:
            eager.extend(doc_tools)
    with optional_tool_group("writer", "pdf"):
        from app.tools.pdf_compose import create_pdf_tools
        eager.extend(create_pdf_tools("writer"))
    with optional_tool_group("writer", "signal_attachment"):
        from app.tools.signal_attachment import create_signal_attachment_tools
        eager.extend(create_signal_attachment_tools("writer"))
    with optional_tool_group("writer", "tool_search"):
        from app.tools.tool_search import create_tool_search_tools
        eager.extend(create_tool_search_tools("writer"))
    with optional_tool_group("writer", "bridge"):
        from app.tools.bridge_tools import create_bridge_tools
        bridge_tools = create_bridge_tools("writer")
        if bridge_tools:
            eager.extend(bridge_tools)
    with optional_tool_group("writer", "wiki"):
        from app.tools.wiki_tool_registry import create_wiki_tools
        eager.extend(create_wiki_tools("read", "write", "search", "slides"))
    with optional_tool_group("writer", "blend"):
        from app.tools.blend_tool import ConceptBlendTool
        eager.append(ConceptBlendTool())
    with optional_tool_group("writer", "dialectics"):
        from app.philosophy.dialectics_tool import FindCounterArgumentTool
        eager.append(FindCounterArgumentTool())
    with optional_tool_group("writer", "tensions"):
        from app.tensions.tools import get_tension_tools
        eager.extend(get_tension_tools("writer"))

    return build_loadable_agent(
        role="Writer",
        goal="Write clear, well-structured content including summaries, reports, documentation, and emails.",
        backstory=WRITER_BACKSTORY,
        llm=llm,
        agent_id="writer",
        # Eager: every tool the legacy path would have surfaced.
        # Phase 5 sweeps the per-agent tools into PER_AGENT registry
        # entries and shrinks this list to ~5 universal tools.
        core_tools=eager,
        # Discoverable: catalog tools the writer might want.
        # Forge-bridged SHADOW/CANARY tools tagged ``registers-tool``
        # also surface here, so the writer can pick up any operator-
        # promoted tool without an agent rewrite.
        discoverable_capabilities=[
            "executes-code",         # numeric/data appendices in reports
            "fetches-geodata",       # geographic context for writing
            "reads-satellite",       # factual basis for nature/place writing
            "converts-currency",     # financial reports
            "renders-chart",         # visualize data alongside prose
        ],
        agent_tier=Tier.PRODUCTION,
        max_execution_time=300,
        verbose=True,
    )
