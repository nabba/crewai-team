"""
Enterprise Knowledge Base — RAG-powered document store for CrewAI agents.

Adds the ability to ingest documents (PDF, DOCX, PPTX, XLSX, CSV, TXT, MD,
HTML, JSON, URLs) into a persistent ChromaDB vector store.  Agents can then
search the knowledge base at runtime via CrewAI tools.

Usage:
    from app.knowledge_base.tools import get_knowledge_tools
    from app.knowledge_base.vectorstore import KnowledgeStore
"""

from app.knowledge_base.vectorstore import KnowledgeStore
from app.knowledge_base.tools import (
    KnowledgeSearchTool,
    KnowledgeIngestTool,
    KnowledgeStatusTool,
    get_knowledge_tools,
)

__all__ = [
    "KnowledgeStore",
    "KnowledgeSearchTool",
    "KnowledgeIngestTool",
    "KnowledgeStatusTool",
    "get_knowledge_tools",
]
