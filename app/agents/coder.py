from crewai import Agent
from app.config import get_settings
from app.llm_factory import create_specialist_llm
from app.tools.code_executor import execute_code
from app.tools.file_manager import file_manager
from app.tools.web_search import web_search
from app.tools.memory_tool import create_memory_tools
from app.tools.scoped_memory_tool import create_scoped_memory_tools
from app.tools.attachment_reader import read_attachment
from app.tools.self_report_tool import create_self_report_tool
from app.tools.reflection_tool import ReflectionTool
from app.souls.loader import compose_backstory
from app.tools.mem0_tools import create_mem0_tools
from app.knowledge_base.tools import KnowledgeSearchTool

settings = get_settings()

CODER_BACKSTORY = compose_backstory("coder")


def create_coder(force_tier: str | None = None) -> Agent:
    llm = create_specialist_llm(max_tokens=4096, role="coding", force_tier=force_tier)
    memory_tools = create_memory_tools(collection="coder")
    scoped_tools = create_scoped_memory_tools("coder")
    mem0_tools = create_mem0_tools("coder")
    awareness_tools = [
        create_self_report_tool("coder"),
        ReflectionTool(agent_role="coder"),
    ]

    return Agent(
        role="Coder",
        goal="Write, test, and debug code across any language. Execute code safely in a Docker sandbox.",
        backstory=CODER_BACKSTORY,
        llm=llm,
        tools=[execute_code, file_manager, web_search, read_attachment, KnowledgeSearchTool()] + memory_tools + scoped_tools + mem0_tools + awareness_tools,
        verbose=True,
    )
