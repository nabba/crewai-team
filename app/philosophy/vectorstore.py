"""
Philosophy Vector Store — Dedicated ChromaDB collection for philosophical texts.

Separate from the main knowledge base and operational memory.  Uses the same
Ollama Metal GPU embedding pipeline as the rest of the system for consistency.

Usage:
    store = PhilosophyStore()
    store.add_documents(chunks, metadatas)
    results = store.query("What does Aristotle say about virtue?")
"""

import logging
from pathlib import Path

import chromadb

from app.philosophy import config

logger = logging.getLogger(__name__)

class PhilosophyStore:
    """
    Persistent vector store for humanist philosophical texts.

    Key differences from enterprise KB:
    - Separate collection (`philosophy_humanist`)
    - Larger chunks optimized for argumentative coherence
    - Metadata schema: author, tradition, era, title, section
    - Read-heavy, write-rare (ingest via dashboard, query via agents)
    """

    def __init__(
        self,
        persist_dir: str = config.CHROMA_PERSIST_DIR,
        collection_name: str = config.COLLECTION_NAME,
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        self._client = chromadb.PersistentClient(path=str(self.persist_dir))

        col = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Handle embedding dimension mismatch from stale collections.
        # All embeddings pinned to 768-dim (nomic-embed-text).
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
                            f"PhilosophyStore: dimension mismatch ({existing_dim} vs "
                            f"{current_dim}) — recreating collection (re-ingest needed)"
                        )
                        self._client.delete_collection(self.collection_name)
                        col = self._client.get_or_create_collection(
                            name=self.collection_name,
                            metadata={"hnsw:space": "cosine"},
                        )
        except Exception as e:
            if "dimension" in str(e).lower():
                logger.warning(f"PhilosophyStore: dimension error — recreating: {e}")
                try:
                    self._client.delete_collection(self.collection_name)
                    col = self._client.get_or_create_collection(
                        name=self.collection_name,
                        metadata={"hnsw:space": "cosine"},
                    )
                except Exception:
                    pass

        self._collection = col
        logger.info(
            f"PhilosophyStore initialized: {self._collection.count()} chunks "
            f"in '{self.collection_name}' at '{persist_dir}'"
        )

    def add_documents(
        self,
        chunks: list[str],
        metadatas: list[dict],
        ids: list[str] | None = None,
    ) -> int:
        """Add document chunks with metadata to the philosophy collection.

        Returns number of chunks added.
        """
        if not chunks:
            return 0

        if len(chunks) != len(metadatas):
            raise ValueError(
                f"chunks ({len(chunks)}) and metadatas ({len(metadatas)}) must match"
            )

        if ids is None:
            existing = self._collection.count()
            ids = [f"phil_{existing + i:06d}" for i in range(len(chunks))]

        from app.memory.chromadb_manager import embed

        batch_size = 50
        total_added = 0

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            batch_embeddings = [embed(c) for c in batch_chunks]

            self._collection.add(
                documents=batch_chunks,
                metadatas=batch_meta,
                embeddings=batch_embeddings,
                ids=batch_ids,
            )
            total_added += len(batch_chunks)
            # PROGRAM §56 — source ledger dual-write.
            try:
                from app.memory.source_ledger import hook_collection_add
                hook_collection_add(
                    "philosophy", self.collection_name,
                    batch_ids, batch_chunks, batch_meta,
                )
            except Exception:
                logger.debug("PhilosophyStore: source_ledger hook failed", exc_info=True)

        logger.info(
            f"Ingested {total_added} chunks. Collection total: {self._collection.count()}"
        )
        return total_added

    def query(
        self,
        query_text: str,
        n_results: int = config.DEFAULT_TOP_K,
        where_filter: dict | None = None,
        min_score: float = config.MIN_RELEVANCE_SCORE,
    ) -> list[dict]:
        """Query the philosophy collection.

        Returns list of result dicts with keys: text, metadata, score, id
        """
        count = self._collection.count()
        if count == 0:
            return []

        from app.memory.chromadb_manager import embed

        query_params = {
            "query_embeddings": [embed(query_text)],
            "n_results": min(n_results, count),
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            query_params["where"] = where_filter

        try:
            results = self._collection.query(**query_params)
        except Exception as e:
            logger.error(f"Philosophy query failed: {e}")
            return []

        formatted = []
        if results and results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                distance = results["distances"][0][i] if results["distances"] else 1.0
                score = 1.0 - distance
                if score < min_score:
                    continue

                formatted.append({
                    "text": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "score": round(score, 4),
                    "id": results["ids"][0][i] if results["ids"] else None,
                })

        return formatted

    def query_reranked(
        self,
        query_text: str,
        n_results: int = config.DEFAULT_TOP_K,
        where_filter: dict | None = None,
        min_score: float = config.MIN_RELEVANCE_SCORE,
    ) -> list[dict]:
        """Two-stage retrieval: vector top-20 → cross-encoder re-rank.

        Falls back to plain ``query()`` if the retrieval orchestrator is
        unavailable (graceful degradation).
        """
        try:
            from app.retrieval.reranker import rerank
        except Exception:
            return self.query(query_text, n_results, where_filter, min_score)

        from app.retrieval.config import RERANK_TOP_K_INPUT
        candidates = self.query(
            query_text=query_text,
            n_results=RERANK_TOP_K_INPUT,
            where_filter=where_filter,
            min_score=min_score,
        )
        if not candidates:
            return []

        return rerank(query_text, candidates, top_k=n_results)

    def remove_by_source(self, source_file: str) -> int:
        """Remove all chunks from a specific source file."""
        try:
            existing = self._collection.get(
                where={"source_file": source_file},
            )
            if existing and existing["ids"]:
                self._collection.delete(ids=existing["ids"])
                count = len(existing["ids"])
                # PROGRAM §56 iter-2 — ledger tombstone
                try:
                    from app.memory.source_ledger import hook_collection_delete
                    hook_collection_delete("philosophy", self.collection_name, list(existing["ids"]))
                except Exception:
                    logger.debug("PhilosophyStore: ledger delete hook failed", exc_info=True)
                logger.info(f"Removed {count} chunks from '{source_file}'")
                return count
        except Exception as e:
            logger.error(f"Failed to remove '{source_file}': {e}")
        return 0

    def reset_collection(self) -> None:
        """Drop and recreate the collection."""
        logger.warning(f"Resetting philosophy collection '{self.collection_name}'!")
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def get_stats(self) -> dict:
        """Return collection statistics."""
        count = self._collection.count()

        traditions: set[str] = set()
        authors: set[str] = set()
        titles: set[str] = set()
        source_files: set[str] = set()

        if count > 0:
            # Get all metadata (philosophy KB is small enough)
            all_data = self._collection.get(include=["metadatas"])
            if all_data and all_data["metadatas"]:
                for meta in all_data["metadatas"]:
                    traditions.add(meta.get("tradition", "Unknown"))
                    authors.add(meta.get("author", "Unknown"))
                    titles.add(meta.get("title", "Unknown"))
                    source_files.add(meta.get("source_file", "Unknown"))

        return {
            "collection_name": self.collection_name,
            "total_chunks": count,
            "total_texts": len(source_files - {"Unknown"}),
            "traditions": sorted(traditions - {"Unknown"}),
            "authors": sorted(authors - {"Unknown"}),
            "titles": sorted(titles - {"Unknown"}),
            "persist_dir": str(self.persist_dir),
        }

    def list_texts(self) -> list[dict]:
        """Return per-document metadata extracted from stored chunks."""
        count = self._collection.count()
        if count == 0:
            return []

        all_data = self._collection.get(include=["metadatas"])
        if not all_data or not all_data["metadatas"]:
            return []

        # Group chunks by source_file, take first chunk's metadata as representative
        by_source: dict[str, dict] = {}
        chunk_counts: dict[str, int] = {}
        for meta in all_data["metadatas"]:
            sf = meta.get("source_file", "Unknown")
            chunk_counts[sf] = chunk_counts.get(sf, 0) + 1
            if sf not in by_source:
                by_source[sf] = {
                    "filename": sf,
                    "title": meta.get("title", sf),
                    "author": meta.get("author", "Unknown"),
                    "tradition": meta.get("tradition", "Unknown"),
                    "era": meta.get("era", "Unknown"),
                }

        for sf, info in by_source.items():
            info["chunks"] = chunk_counts.get(sf, 0)

        return sorted(by_source.values(), key=lambda d: d["title"])

# ── Singleton accessor ────────────────────────────────────────────────────────
_store: PhilosophyStore | None = None

def get_store() -> PhilosophyStore:
    """Lazy-singleton accessor for the philosophy store."""
    global _store
    if _store is None:
        _store = PhilosophyStore()
    return _store
