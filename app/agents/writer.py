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
from app.philosophy.rag_tool import PhilosophyRAGTool

settings = get_settings()

WRITER_BACKSTORY = compose_backstory("writer")


def create_writer(force_tier: str | None = None) -> Agent:
    llm = create_specialist_llm(max_tokens=4096, role="writing", force_tier=force_tier)
    # R5: Self-awareness tools removed — telemetry runs in post-crew hook
    memory_tools = create_memory_tools(collection="writer")
    scoped_tools = create_scoped_memory_tools("writer")
    mem0_tools = create_mem0_tools("writer")

    tools = [file_manager, web_search, read_attachment, KnowledgeSearchTool(), PhilosophyRAGTool()] + memory_tools + scoped_tools + mem0_tools
    # Document generation tools (PDF, DOCX, HTML)
    try:
        from app.tools.document_generator import create_document_tools
        doc_tools = create_document_tools()
        if doc_tools:
            tools.extend(doc_tools)
    except Exception:
        pass
    # Host Bridge tools (read/write host files for document output)
    try:
        from app.tools.bridge_tools import create_bridge_tools
        bridge_tools = create_bridge_tools("writer")
        if bridge_tools:
            tools.extend(bridge_tools)
    except Exception:
        pass

    return Agent(
        role="Writer",
        goal="Write clear, well-structured content including summaries, reports, documentation, and emails.",
        backstory=WRITER_BACKSTORY,
        llm=llm,
        tools=tools,
        verbose=True,
    )
