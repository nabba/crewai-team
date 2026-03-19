from crewai import Agent
from app.config import get_settings
from app.llm_factory import create_specialist_llm
from app.tools.memory_tool import create_memory_tools
from app.tools.scoped_memory_tool import create_scoped_memory_tools
from app.tools.file_manager import file_manager
from app.tools.web_search import web_search
from app.tools.attachment_reader import read_attachment
from app.tools.self_report_tool import create_self_report_tool
from app.tools.reflection_tool import ReflectionTool
from app.souls.loader import compose_backstory

settings = get_settings()

WRITER_BACKSTORY = compose_backstory("writer")


def create_writer(force_tier: str | None = None) -> Agent:
    llm = create_specialist_llm(max_tokens=4096, role="writing", force_tier=force_tier)
    memory_tools = create_memory_tools(collection="writer")
    scoped_tools = create_scoped_memory_tools("writer")
    awareness_tools = [
        create_self_report_tool("writer"),
        ReflectionTool(agent_role="writer"),
    ]

    return Agent(
        role="Writer",
        goal="Write clear, well-structured content including summaries, reports, documentation, and emails.",
        backstory=WRITER_BACKSTORY,
        llm=llm,
        tools=[file_manager, web_search, read_attachment] + memory_tools + scoped_tools + awareness_tools,
        verbose=True,
    )
