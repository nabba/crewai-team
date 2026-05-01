"""repo_analyst.py — Repository analysis agent."""

from crewai import Agent
from app.llm_factory import create_specialist_llm
from app.tools.web_search import web_search
from app.tools.file_manager import file_manager
from app.tools.memory_tool import create_memory_tools
from app.tools.scoped_memory_tool import create_scoped_memory_tools
from app.tools.mem0_tools import create_mem0_tools
from app.knowledge_base.tools import KnowledgeSearchTool
from app.souls.loader import compose_backstory


REPO_ANALYST_BACKSTORY = compose_backstory("repo_analyst")


def create_repo_analyst(force_tier: str | None = None) -> Agent:
    llm = create_specialist_llm(max_tokens=4096, role="coding", force_tier=force_tier)
    memory_tools = create_memory_tools(collection="repo_analyst")
    scoped_tools = create_scoped_memory_tools("repo_analyst")
    mem0_tools = create_mem0_tools("repo_analyst")

    tools = [web_search, file_manager, KnowledgeSearchTool()] + memory_tools + scoped_tools + mem0_tools

    # Repo analysis tools
    with optional_tool_group('repoanalyst', 'repo_analysis_tools'):
        from app.tools.repo_analysis_tools import create_repo_analysis_tools
        repo_tools = create_repo_analysis_tools("repo_analyst")
        if repo_tools:
            tools.extend(repo_tools)

    # Document generation for reports
    with optional_tool_group('repoanalyst', 'document_generator'):
        from app.tools.document_generator import create_document_tools
        doc_tools = create_document_tools()
        if doc_tools:
            tools.extend(doc_tools)

    # Bridge tools
    with optional_tool_group('repoanalyst', 'bridge_tools'):
        from app.tools.bridge_tools import create_bridge_tools
        bridge_tools = create_bridge_tools("repo_analyst")
        if bridge_tools:
            tools.extend(bridge_tools)

    return Agent(
        role="Repository Analyst",
        goal="Analyze codebases: structure, tech stack, metrics, architecture, and produce reports.",
        backstory=REPO_ANALYST_BACKSTORY,
        llm=llm,
        tools=tools,
        max_execution_time=600,
        verbose=True,
    )
