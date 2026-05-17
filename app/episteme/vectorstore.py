"""
episteme/vectorstore.py — ChromaDB vector store for research/metacognitive knowledge.

Follows the PhilosophyStore pattern exactly.
"""

from __future__ import annotations

import logging
from pathlib import Path

import chromadb

from app.episteme import config

logger = logging.getLogger(__name__)


class EpistemeStore:
    """Persistent vector store for research papers and design knowledge.

    Metadata schema per chunk:
        source_file, author, paper_type, domain, epistemic_status,
        date, title, section
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

        # Handle embedding dimension mismatch.
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
                            "EpistemeStore: dimension mismatch (%d vs %d) — recreating",
                            existing_dim, current_dim,
                        )
                        self._client.delete_collection(self.collection_name)
                        col = self._client.get_or_create_collection(
                            name=self.collection_name,
                            metadata={"hnsw:space": "cosine"},
                        )
        except Exception as e:
            if "dimension" in str(e).lower():
                logger.warning("EpistemeStore: dimension error — recreating: %s", e)
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
            "EpistemeStore initialized: %d chunks in '%s' at '%s'",
            self._collection.count(), self.collection_name, persist_dir,
        )

    def add_documents(
        self,
        chunks: list[str],
        metadatas: list[dict],
        ids: list[str] | None = None,
    ) -> int:
        if not chunks:
            return 0
        if len(chunks) != len(metadatas):
            raise ValueError("chunks and metadatas must have same length")

        if ids is None:
            existing = self._collection.count()
            ids = [f"epi_{existing + i:06d}" for i in range(len(chunks))]

        from app.memory.chromadb_manager import embed

        batch_size = 50
        total_added = 0
        for i in range(0, len(chunks), batch_size):
            bc = chunks[i:i + batch_size]
            bm = metadatas[i:i + batch_size]
            bi = ids[i:i + batch_size]
            be = [embed(c) for c in bc]
            self._collection.add(
                documents=bc, metadatas=bm, embeddings=be, ids=bi,
            )
            total_added += len(bc)
            # PROGRAM §56 — source ledger dual-write. Failure-isolated.
            try:
                from app.memory.source_ledger import hook_collection_add
                hook_collection_add("episteme", self.collection_name, bi, bc, bm)
            except Exception:
                logger.debug("EpistemeStore: source_ledger hook failed", exc_info=True)

        logger.info("Ingested %d chunks. Total: %d", total_added, self._collection.count())
        return total_added

    def query(
        self,
        query_text: str,
        n_results: int = config.DEFAULT_TOP_K,
        where_filter: dict | None = None,
        min_score: float = config.MIN_RELEVANCE_SCORE,
    ) -> list[dict]:
        count = self._collection.count()
        if count == 0:
            return []

        from app.memory.chromadb_manager import embed

        query_params: dict = {
            "query_embeddings": [embed(query_text)],
            "n_results": min(n_results, count),
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            query_params["where"] = where_filter

        try:
            results = self._collection.query(**query_params)
        except Exception as e:
            logger.error("Episteme query failed: %s", e)
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
        try:
            from app.retrieval.reranker import rerank
        except Exception:
            return self.query(query_text, n_results, where_filter, min_score)

        from app.retrieval.config import RERANK_TOP_K_INPUT
        candidates = self.query(query_text, RERANK_TOP_K_INPUT, where_filter, min_score)
        if not candidates:
            return []
        return rerank(query_text, candidates, top_k=n_results)

    def remove_by_source(self, source_file: str) -> int:
        try:
            existing = self._collection.get(where={"source_file": source_file})
            if existing and existing["ids"]:
                self._collection.delete(ids=existing["ids"])
                count = len(existing["ids"])
                # PROGRAM §56 iter-2 — ledger tombstone
                try:
                    from app.memory.source_ledger import hook_collection_delete
                    hook_collection_delete("episteme", self.collection_name, list(existing["ids"]))
                except Exception:
                    logger.debug("EpistemeStore: ledger delete hook failed", exc_info=True)
                logger.info("Removed %d chunks from '%s'", count, source_file)
                return count
        except Exception as e:
            logger.error("Failed to remove '%s': %s", source_file, e)
        return 0

    def get_stats(self) -> dict:
        count = self._collection.count()
        paper_types: set[str] = set()
        authors: set[str] = set()
        titles: set[str] = set()
        source_files: set[str] = set()

        if count > 0:
            all_data = self._collection.get(include=["metadatas"])
            if all_data and all_data["metadatas"]:
                for meta in all_data["metadatas"]:
                    paper_types.add(meta.get("paper_type", "Unknown"))
                    authors.add(meta.get("author", "Unknown"))
                    titles.add(meta.get("title", "Unknown"))
                    source_files.add(meta.get("source_file", "Unknown"))

        return {
            "collection_name": self.collection_name,
            "total_chunks": count,
            "total_texts": len(source_files - {"Unknown"}),
            "paper_types": sorted(paper_types - {"Unknown"}),
            "authors": sorted(authors - {"Unknown"}),
            "titles": sorted(titles - {"Unknown"}),
            "persist_dir": str(self.persist_dir),
        }

    def list_texts(self) -> list[dict]:
        count = self._collection.count()
        if count == 0:
            return []

        all_data = self._collection.get(include=["metadatas"])
        if not all_data or not all_data["metadatas"]:
            return []

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
                    "paper_type": meta.get("paper_type", "Unknown"),
                    "domain": meta.get("domain", "Unknown"),
                }

        for sf, info in by_source.items():
            info["chunks"] = chunk_counts.get(sf, 0)

        return sorted(by_source.values(), key=lambda d: d["title"])

    def reset_collection(self) -> None:
        logger.warning("Resetting episteme collection '%s'!", self.collection_name)
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )


_store: EpistemeStore | None = None

def get_store() -> EpistemeStore:
    global _store
    if _store is None:
        _store = EpistemeStore()
    return _store
