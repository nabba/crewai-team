"""financial_analyst.py — Financial analysis agent (market data, modeling, reports)."""

from crewai import Agent
from app.agents._common import optional_tool_group
from app.llm_factory import create_specialist_llm
from app.tools.web_search import web_search
from app.tools.web_fetch import web_fetch
from app.tools.file_manager import file_manager
from app.tools.memory_tool import create_memory_tools
from app.tools.scoped_memory_tool import create_scoped_memory_tools
from app.tools.mem0_tools import create_mem0_tools
from app.knowledge_base.tools import KnowledgeSearchTool
from app.souls.loader import compose_backstory


FINANCIAL_BACKSTORY = compose_backstory("financial_analyst")


def create_financial_analyst(force_tier: str | None = None) -> Agent:
    llm = create_specialist_llm(max_tokens=4096, role="research", force_tier=force_tier)
    memory_tools = create_memory_tools(collection="financial")
    scoped_tools = create_scoped_memory_tools("financial")
    mem0_tools = create_mem0_tools("financial")

    tools = [web_search, web_fetch, file_manager, KnowledgeSearchTool()] + memory_tools + scoped_tools + mem0_tools

    # Financial data tools
    with optional_tool_group("financial", "financial_tools"):
        from app.tools.financial_tools import create_financial_tools
        fin_tools = create_financial_tools("financial")
        if fin_tools:
            tools.extend(fin_tools)

    # Document generation for reports
    with optional_tool_group("financial", "document_generator"):
        from app.tools.document_generator import create_document_tools
        doc_tools = create_document_tools()
        if doc_tools:
            tools.extend(doc_tools)

    # Bridge tools
    with optional_tool_group("financial", "bridge"):
        from app.tools.bridge_tools import create_bridge_tools
        bridge_tools = create_bridge_tools("financial")
        if bridge_tools:
            tools.extend(bridge_tools)

    # Wiki tools
    with optional_tool_group("financial", "wiki"):
        from app.tools.wiki_tool_registry import create_wiki_tools
        tools.extend(create_wiki_tools("read", "write"))

    return Agent(
        role="Financial Analyst",
        goal="Provide rigorous, data-driven financial analysis with clear methodology and sourcing.",
        backstory=FINANCIAL_BACKSTORY,
        llm=llm,
        tools=tools,
        max_execution_time=600,  # Financial analysis may need more time
        verbose=True,
    )
