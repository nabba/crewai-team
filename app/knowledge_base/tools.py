"""
CrewAI Tools for Enterprise Knowledge Base.

Three tools:
  1. KnowledgeSearchTool   -- Query the knowledge base (all agents get this)
  2. KnowledgeIngestTool   -- Add documents (admin use)
  3. KnowledgeStatusTool   -- Check what's in the knowledge base
"""

import json
import logging
from typing import Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from app.knowledge_base.vectorstore import KnowledgeStore

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Singleton store instance (lazy, shared across all tools)
# ─────────────────────────────────────────────────────────────────────────────
_store_instance: Optional[KnowledgeStore] = None


def get_store() -> KnowledgeStore:
    global _store_instance
    if _store_instance is None:
        _store_instance = KnowledgeStore()
    return _store_instance


def set_store(store: KnowledgeStore) -> None:
    global _store_instance
    _store_instance = store


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1: Knowledge Search
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeSearchInput(BaseModel):
    query: str = Field(
        description=(
            "The question or topic to search for in the knowledge base. "
            "Be specific and use domain terminology for best results."
        )
    )
    category: Optional[str] = Field(
        default=None,
        description=(
            "Optional: filter results by category "
            "(e.g., 'policy', 'product', 'finance', 'technical'). "
            "Leave empty to search all."
        ),
    )
    top_k: int = Field(
        default=6,
        description="Number of results to return (1-15). Default 6.",
    )


class KnowledgeSearchTool(BaseTool):
    name: str = "search_knowledge_base"
    description: str = (
        "Search the knowledge base for specific information. "
        "Returns relevant text passages from ingested documents "
        "(policies, product docs, reports, data files, etc.) with source "
        "attribution. Use this whenever you need factual information that "
        "may be in the knowledge base. Be specific in your query."
    )
    args_schema: Type[BaseModel] = KnowledgeSearchInput

    def _run(self, query: str, category: Optional[str] = None, top_k: int = 6) -> str:
        try:
            store = get_store()
        except Exception as e:
            return f"Knowledge base unavailable: {e}"

        results = store.query(
            question=query,
            top_k=min(top_k, 15),
            category=category,
        )

        if not results:
            return (
                f"No relevant information found in the knowledge base "
                f"for: '{query}'. The knowledge base may not contain "
                "information on this topic."
            )

        parts = [f"Found {len(results)} relevant passages:\n"]
        for i, r in enumerate(results, 1):
            parts.append(
                f"--- Result {i} (relevance: {r['score']:.0%}) ---\n"
                f"Source: {r['source']} | Category: {r['category']}\n"
                f"Content:\n{r['text']}\n"
            )
        parts.append(
            "\n[Cite the source document when using this information.]"
        )
        return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2: Knowledge Ingestion
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeIngestInput(BaseModel):
    source: str = Field(
        description=(
            "File path or URL to ingest. Supported: "
            "PDF, DOCX, PPTX, XLSX, CSV, TXT, MD, HTML, JSON, URLs."
        )
    )
    category: str = Field(
        default="general",
        description="Category for filtering (e.g., 'policy', 'product', 'finance').",
    )
    tags: Optional[str] = Field(
        default=None,
        description="Comma-separated tags (e.g., 'Q1-2026,board').",
    )


class KnowledgeIngestTool(BaseTool):
    name: str = "ingest_to_knowledge_base"
    description: str = (
        "Add a document (file or URL) to the knowledge base. "
        "The document will be processed, chunked, and stored for retrieval. "
        "Supported: PDF, DOCX, PPTX, XLSX, CSV, TXT, MD, HTML, JSON, URLs."
    )
    args_schema: Type[BaseModel] = KnowledgeIngestInput

    def _run(
        self,
        source: str,
        category: str = "general",
        tags: Optional[str] = None,
    ) -> str:
        try:
            store = get_store()
        except Exception as e:
            return f"Knowledge base unavailable: {e}"

        tag_list = [t.strip() for t in tags.split(",")] if tags else []

        result = store.add_document(
            source=source,
            category=category,
            tags=tag_list,
        )

        if result.success:
            return (
                f"Ingested '{result.source}':\n"
                f"  Format: {result.format}\n"
                f"  Chunks: {result.chunks_created}\n"
                f"  Characters: {result.total_characters:,}\n"
                f"  Category: {category}"
            )
        else:
            return f"Failed to ingest '{source}': {result.error}"


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3: Knowledge Base Status
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeStatusInput(BaseModel):
    detail_level: str = Field(
        default="summary",
        description="'summary' for overview, 'full' for complete document list.",
    )


class KnowledgeStatusTool(BaseTool):
    name: str = "knowledge_base_status"
    description: str = (
        "Check what documents and knowledge are in the knowledge base. "
        "Returns document count, categories, and optionally a document list."
    )
    args_schema: Type[BaseModel] = KnowledgeStatusInput

    def _run(self, detail_level: str = "summary") -> str:
        try:
            store = get_store()
        except Exception as e:
            return f"Knowledge base unavailable: {e}"

        stats = store.stats()

        parts = [
            "Knowledge Base Status:",
            f"  Documents: {stats['total_documents']}",
            f"  Chunks: {stats['total_chunks']}",
            f"  Characters: {stats['total_characters']:,}",
            f"  Est. tokens: ~{stats['estimated_tokens']:,}",
            "",
            "Categories:",
        ]

        for cat, count in sorted(stats["categories"].items()):
            parts.append(f"  {cat}: {count} chunks")

        if detail_level == "full" and stats["documents"]:
            parts.append("\nDocuments:")
            for doc in stats["documents"]:
                parts.append(
                    f"  - {doc['source']} ({doc['format']}) | "
                    f"cat: {doc['category']} | "
                    f"chunks: {doc['total_chunks']} | "
                    f"added: {doc['ingested_at'][:10]}"
                )

        return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: Get all tools as a list
# ─────────────────────────────────────────────────────────────────────────────

def get_knowledge_tools(
    store: Optional[KnowledgeStore] = None,
    include_ingest: bool = False,
) -> list[BaseTool]:
    """
    Return knowledge base tools ready for CrewAI agents.

    By default returns search + status only.  Set include_ingest=True
    to also include the ingestion tool.
    """
    if store:
        set_store(store)

    tools = [KnowledgeSearchTool(), KnowledgeStatusTool()]
    if include_ingest:
        tools.append(KnowledgeIngestTool())
    return tools
