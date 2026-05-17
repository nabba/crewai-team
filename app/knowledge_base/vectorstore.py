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

import chromadb

# Use direct module import to avoid circular dependency:
# __init__.py imports vectorstore → vectorstore imports __init__ (for config) → circular
import app.knowledge_base.config as config
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

        # Use a dedicated PersistentClient for the KB so it reads from
        # config.CHROMA_PERSIST_DIR (/app/workspace/knowledge), NOT the
        # shared memory client which points to /app/workspace/memory.
        # Both use the same embed() function (Ollama GPU / CPU fallback).
        self._client = chromadb.PersistentClient(path=persist_dir)

        col = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Handle embedding dimension mismatch (e.g. stale collection from a
        # prior model). If mismatch: delete and recreate — documents must be
        # re-ingested. All embeddings are now pinned to 768-dim (nomic-embed-text).
        from app.memory.chromadb_manager import get_embed_dim
        try:
            if col.count() > 0:
                sample = col.peek(1)  # returns embeddings by default in chromadb 1.x
                embs = sample.get("embeddings") if sample else None
                if embs is not None and len(embs) > 0 and embs[0] is not None and len(embs[0]) > 0:
                    existing_dim = len(embs[0])
                    current_dim = get_embed_dim()
                    if existing_dim != current_dim:
                        logger.warning(
                            f"KnowledgeStore: collection '{collection_name}' has "
                            f"{existing_dim}-dim embeddings but current model produces "
                            f"{current_dim}-dim — recreating (documents must be re-ingested)"
                        )
                        self._client.delete_collection(collection_name)
                        col = self._client.get_or_create_collection(
                            name=collection_name,
                            metadata={"hnsw:space": "cosine"},
                        )
        except Exception as e:
            if "dimension" in str(e).lower():
                logger.warning(f"KnowledgeStore: dimension error on init — recreating: {e}")
                try:
                    self._client.delete_collection(collection_name)
                    col = self._client.get_or_create_collection(
                        name=collection_name,
                        metadata={"hnsw:space": "cosine"},
                    )
                except Exception:
                    pass

        self._collection = col
        self._collection_name = collection_name
        logger.info(
            f"KnowledgeStore initialized: {self._collection.count()} chunks "
            f"in collection '{collection_name}' at '{persist_dir}'"
        )

    def _ensure_collection(self):
        """Re-fetch collection if the cached reference points to a stale UUID."""
        try:
            self._collection.count()
        except Exception:
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )

    # ─────────────────────────────────────────────
    # Document Management
    # ─────────────────────────────────────────────

    def add_document(
        self,
        source: str,
        category: str = "general",
        tags: list[str] | None = None,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
    ) -> IngestionResult:
        """Ingest a document and add its chunks to the knowledge base."""
        self._ensure_collection()
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
        chunk_ids = [c.chunk_id for c in chunks]
        chunk_docs = [c.text for c in chunks]
        chunk_metas = [c.metadata for c in chunks]
        self._collection.add(
            ids=chunk_ids,
            documents=chunk_docs,
            embeddings=[embed(t) for t in chunk_docs],
            metadatas=chunk_metas,
        )
        # PROGRAM §56 — source ledger dual-write.
        try:
            from app.memory.source_ledger import hook_collection_add
            hook_collection_add(
                "knowledge", self._collection.name,
                chunk_ids, chunk_docs, chunk_metas,
            )
        except Exception:
            logger.debug("KnowledgeBase: source_ledger hook failed", exc_info=True)

        logger.info(f"Added {len(chunks)} chunks from '{result.source}'")
        return result

    def add_text(
        self,
        text: str,
        source_name: str = "manual_entry",
        category: str = "general",
        tags: list[str] | None = None,
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
        chunk_ids = [c.chunk_id for c in chunks]
        chunk_docs = [c.text for c in chunks]
        chunk_metas = [c.metadata for c in chunks]
        self._collection.add(
            ids=chunk_ids,
            documents=chunk_docs,
            embeddings=[embed(t) for t in chunk_docs],
            metadatas=chunk_metas,
        )
        # PROGRAM §56 — source ledger dual-write.
        try:
            from app.memory.source_ledger import hook_collection_add
            hook_collection_add(
                "knowledge", self._collection.name,
                chunk_ids, chunk_docs, chunk_metas,
            )
        except Exception:
            logger.debug("KnowledgeBase.add_text: source_ledger hook failed", exc_info=True)

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
                # PROGRAM §56 iter-2 — ledger tombstone
                try:
                    from app.memory.source_ledger import hook_collection_delete
                    hook_collection_delete(
                        "knowledge", self._collection.name, list(existing["ids"]),
                    )
                except Exception:
                    pass
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
        category: str | None = None,
        tags: list[str] | None = None,
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

    def query_reranked(
        self,
        question: str,
        top_k: int = config.DEFAULT_TOP_K,
        category: str | None = None,
        tags: list[str] | None = None,
        min_score: float = config.MIN_RELEVANCE_SCORE,
    ) -> list[dict]:
        """Two-stage retrieval: vector top-20 → temporal freshness → cross-encoder re-rank.

        Falls back to plain ``query()`` if the retrieval orchestrator is
        unavailable (graceful degradation).
        """
        try:
            from app.retrieval.reranker import rerank
            from app.retrieval.temporal import apply_temporal_decay
        except Exception:
            return self.query(question, top_k, category, tags, min_score)

        # Stage 1: broad vector retrieval.
        from app.retrieval.config import RERANK_TOP_K_INPUT
        candidates = self.query(
            question=question,
            top_k=RERANK_TOP_K_INPUT,
            category=category,
            tags=tags,
            min_score=min_score,
        )
        if not candidates:
            return []

        # Stage 1.5: temporal freshness weighting (enterprise docs have ingested_at).
        candidates = apply_temporal_decay(candidates, timestamp_field="ingested_at")

        # Stage 2: cross-encoder re-ranking.
        return rerank(question, candidates, top_k=top_k)

    # ─────────────────────────────────────────────
    # Inventory & Stats
    # ─────────────────────────────────────────────

    def list_documents(self) -> list[dict]:
        """List all unique documents in the knowledge base."""
        self._ensure_collection()
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
        self._ensure_collection()
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
        self._ensure_collection()
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning("Knowledge base has been reset.")
