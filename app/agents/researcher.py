from crewai import Agent
from app.config import get_settings
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

settings = get_settings()

RESEARCHER_BACKSTORY = compose_backstory("researcher")


def create_researcher(force_tier: str | None = None) -> Agent:
    llm = create_specialist_llm(max_tokens=4096, role="research", force_tier=force_tier)
    memory_tools = create_memory_tools(collection="researcher")
    scoped_tools = create_scoped_memory_tools("researcher")
    awareness_tools = [
        create_self_report_tool("researcher"),
        ReflectionTool(agent_role="researcher"),
    ]

    return Agent(
        role="Researcher",
        goal="Find accurate, comprehensive information on any topic using web search, article reading, and YouTube transcripts.",
        backstory=RESEARCHER_BACKSTORY,
        llm=llm,
        tools=[web_search, web_fetch, get_youtube_transcript, file_manager, read_attachment] + memory_tools + scoped_tools + awareness_tools,
        verbose=True,
    )
