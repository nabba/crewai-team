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

settings = get_settings()

CODER_BACKSTORY = compose_backstory("coder") + FICTION_AWARENESS_PROMPT


def create_coder(force_tier: str | None = None) -> Agent:
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
