from crewai import Agent
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

settings = get_settings()

WRITER_BACKSTORY = compose_backstory("writer")


def create_writer(force_tier: str | None = None) -> Agent:
    llm = create_specialist_llm(max_tokens=4096, role="writing", force_tier=force_tier)
    # R5: Self-awareness tools removed — telemetry runs in post-crew hook
    memory_tools = create_memory_tools(collection="writer")
    scoped_tools = create_scoped_memory_tools("writer")
    mem0_tools = create_mem0_tools("writer")

    return Agent(
        role="Writer",
        goal="Write clear, well-structured content including summaries, reports, documentation, and emails.",
        backstory=WRITER_BACKSTORY,
        llm=llm,
        tools=[file_manager, web_search, read_attachment, KnowledgeSearchTool()] + memory_tools + scoped_tools + mem0_tools,
        verbose=True,
    )
