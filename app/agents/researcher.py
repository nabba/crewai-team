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
    "Distinguish between verified fact, inference, and speculation."
)


def create_researcher(force_tier: str | None = None, light: bool = False) -> Agent:
    """Create a researcher agent.

    Args:
        force_tier: Override model tier selection.
        light: If True, use minimal tools and compact backstory (S8/S9).
               Used for difficulty ≤ 3 simple factual questions.
    """
    llm = create_specialist_llm(max_tokens=4096, role="research", force_tier=force_tier)

    if light:
        # S8: Only the tools needed for a quick factual lookup
        tools = [web_search, web_fetch, KnowledgeSearchTool()]
        backstory = _COMPACT_RESEARCHER_BACKSTORY
    else:
        # R5: Removed world_model_tool (write-only, never read) and self-awareness
        # tools (self_report, reflection) — telemetry now runs in post-crew hook.
        # This saves ~600 tokens per LLM call (3 fewer tool descriptions).
        memory_tools = create_memory_tools(collection="researcher")
        scoped_tools = create_scoped_memory_tools("researcher")
        mem0_tools = create_mem0_tools("researcher")
        tools = [web_search, web_fetch, get_youtube_transcript, file_manager, read_attachment, KnowledgeSearchTool()] + memory_tools + scoped_tools + mem0_tools
        backstory = RESEARCHER_BACKSTORY

    return Agent(
        role="Researcher",
        goal="Find accurate, comprehensive information on any topic using web search, article reading, and YouTube transcripts.",
        backstory=backstory,
        llm=llm,
        tools=tools,
        verbose=True,
    )
