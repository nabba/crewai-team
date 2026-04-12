from crewai import Agent
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

    tools = [execute_code, file_manager, web_search, read_attachment, KnowledgeSearchTool()] + memory_tools + scoped_tools + mem0_tools + get_fiction_tools()
    # Host Bridge tools (read/write host files, execute commands on Mac)
    try:
        from app.tools.bridge_tools import create_bridge_tools
        bridge_tools = create_bridge_tools("coder")
        if bridge_tools:
            tools.extend(bridge_tools)
    except Exception:
        pass
    # Wiki tools (read, write — coder updates technical architecture pages)
    try:
        from app.tools.wiki_tool_registry import create_wiki_tools
        tools.extend(create_wiki_tools("read", "write"))
    except Exception:
        pass

    return Agent(
        role="Coder",
        goal="Write, test, and debug code across any language. Execute code safely in a Docker sandbox.",
        backstory=CODER_BACKSTORY,
        llm=llm,
        tools=tools,
        max_execution_time=300,
        verbose=True,
    )
