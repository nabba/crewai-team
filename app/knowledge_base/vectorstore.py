"""
Vector Store Layer (ChromaDB)

Persistent vector storage for enterprise knowledge.
All data stored on disk and survives process restarts.
"""

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import chromadb

from app.knowledge_base import config
from app.knowledge_base.ingestion import (
    DocumentChunk,
    IngestionResult,
    chunk_text,
    ingest_document,
)

logger = logging.getLogger(__name__)


class KnowledgeStore:
    """
    Persistent enterprise knowledge base backed by ChromaDB.

    Usage:
        store = KnowledgeStore()
        store.add_document("/path/to/policy.pdf", category="policy")
        results = store.query("What is our refund policy?")
    """

    def __init__(
        self,
        persist_dir: str = config.CHROMA_PERSIST_DIR,
        collection_name: str = config.CHROMA_COLLECTION_NAME,
        embedding_model: str = config.EMBEDDING_MODEL,
    ):
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        # B6: Reuse the shared ChromaDB client from chromadb_manager instead of
        # creating a duplicate PersistentClient (which loaded a second
        # SentenceTransformer model). Now uses the Ollama Metal GPU embedding
        # backend shared with the memory system.
        from app.memory.chromadb_manager import get_client
        self._client = get_client()

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            f"KnowledgeStore initialized: {self._collection.count()} chunks "
            f"in collection '{collection_name}'"
        )

    # ─────────────────────────────────────────────
    # Document Management
    # ─────────────────────────────────────────────

    def add_document(
        self,
        source: str,
        category: str = "general",
        tags: Optional[list[str]] = None,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
    ) -> IngestionResult:
        """Ingest a document and add its chunks to the knowledge base."""
        chunks, result = ingest_document(
            source=source,
            category=category,
            tags=tags,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        if not result.success or not chunks:
            return result

        # Replace if source already exists
        try:
            existing = self._collection.get(
                where={"source_path": source},
                limit=1,
            )
            if existing and existing["ids"]:
                logger.info(f"Source '{source}' already exists. Replacing.")
                self.remove_document(source)
        except Exception:
            pass  # where clause may fail if no docs exist yet

        from app.memory.chromadb_manager import embed
        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=[embed(c.text) for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )

        logger.info(f"Added {len(chunks)} chunks from '{result.source}'")
        return result

    def add_text(
        self,
        text: str,
        source_name: str = "manual_entry",
        category: str = "general",
        tags: Optional[list[str]] = None,
    ) -> IngestionResult:
        """Add raw text directly to the knowledge base (no file needed)."""
        tags = tags or []
        now = datetime.now(timezone.utc).isoformat()

        text_chunks = chunk_text(text)
        if not text_chunks:
            return IngestionResult(
                source=source_name,
                format="text",
                chunks_created=0,
                total_characters=len(text),
                success=False,
                error="Text too short to produce meaningful chunks.",
            )

        chunks = []
        for i, chunk_content in enumerate(text_chunks):
            metadata = {
                "source": source_name,
                "source_path": f"manual://{source_name}",
                "format": "text",
                "category": category,
                "tags": json.dumps(tags),
                "chunk_index": i,
                "total_chunks": len(text_chunks),
                "ingested_at": now,
                "char_count": len(chunk_content),
            }
            chunks.append(DocumentChunk(text=chunk_content, metadata=metadata))

        from app.memory.chromadb_manager import embed
        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=[embed(c.text) for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )

        return IngestionResult(
            source=source_name,
            format="text",
            chunks_created=len(chunks),
            total_characters=len(text),
            success=True,
        )

    def remove_document(self, source_path: str) -> int:
        """Remove all chunks from a specific source."""
        try:
            existing = self._collection.get(where={"source_path": source_path})
            if existing and existing["ids"]:
                self._collection.delete(ids=existing["ids"])
                count = len(existing["ids"])
                logger.info(f"Removed {count} chunks from '{source_path}'")
                return count
        except Exception as e:
            logger.error(f"Failed to remove document '{source_path}': {e}")
        return 0

    # ─────────────────────────────────────────────
    # Retrieval
    # ─────────────────────────────────────────────

    def query(
        self,
        question: str,
        top_k: int = config.DEFAULT_TOP_K,
        category: Optional[str] = None,
        tags: Optional[list[str]] = None,
        min_score: float = config.MIN_RELEVANCE_SCORE,
    ) -> list[dict]:
        """Query the knowledge base and return relevant chunks with metadata."""
        if self._collection.count() == 0:
            return []

        where_filter = None
        if category:
            where_filter = {"category": category}

        from app.memory.chromadb_manager import embed
        results = self._collection.query(
            query_embeddings=[embed(question)],
            n_results=min(top_k, self._collection.count()),
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        formatted = []
        for doc, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            score = 1.0 - distance  # cosine distance -> similarity
            if score < min_score:
                continue

            if tags:
                chunk_tags = json.loads(meta.get("tags", "[]"))
                if not any(t in chunk_tags for t in tags):
                    continue

            formatted.append({
                "text": doc,
                "source": meta.get("source", "unknown"),
                "score": round(score, 4),
                "category": meta.get("category", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "total_chunks": meta.get("total_chunks", 0),
                "ingested_at": meta.get("ingested_at", ""),
                "metadata": meta,
            })

        return formatted

    # ─────────────────────────────────────────────
    # Inventory & Stats
    # ─────────────────────────────────────────────

    def list_documents(self) -> list[dict]:
        """List all unique documents in the knowledge base."""
        all_data = self._collection.get(include=["metadatas"])
        if not all_data["metadatas"]:
            return []

        docs = {}
        for meta in all_data["metadatas"]:
            source_path = meta.get("source_path", "unknown")
            if source_path not in docs:
                docs[source_path] = {
                    "source": meta.get("source", "unknown"),
                    "source_path": source_path,
                    "format": meta.get("format", "unknown"),
                    "category": meta.get("category", "general"),
                    "tags": json.loads(meta.get("tags", "[]")),
                    "total_chunks": meta.get("total_chunks", 0),
                    "ingested_at": meta.get("ingested_at", ""),
                }

        return sorted(docs.values(), key=lambda d: d["ingested_at"], reverse=True)

    def stats(self) -> dict:
        """Return knowledge base statistics."""
        docs = self.list_documents()
        all_data = self._collection.get(include=["metadatas"])

        categories = {}
        total_chars = 0
        for meta in all_data["metadatas"]:
            cat = meta.get("category", "general")
            categories[cat] = categories.get(cat, 0) + 1
            total_chars += meta.get("char_count", 0)

        return {
            "total_chunks": self._collection.count(),
            "total_documents": len(docs),
            "total_characters": total_chars,
            "estimated_tokens": total_chars // 4,
            "categories": categories,
            "documents": docs,
        }

    def reset(self) -> None:
        """Delete all data from the knowledge base."""
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection.name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning("Knowledge base has been reset.")
