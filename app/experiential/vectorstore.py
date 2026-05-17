"""
experiential/vectorstore.py — ChromaDB vector store for journal entries.

Follows the PhilosophyStore pattern.
"""

from __future__ import annotations

import logging
from pathlib import Path

import chromadb

from app.experiential import config

logger = logging.getLogger(__name__)


class ExperientialStore:
    """Persistent vector store for experiential journal entries.

    Metadata schema per chunk:
        source_file, entry_type, agent, task_id, emotional_valence,
        epistemic_status, created_at
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
                            "ExperientialStore: dimension mismatch — recreating"
                        )
                        self._client.delete_collection(self.collection_name)
                        col = self._client.get_or_create_collection(
                            name=self.collection_name,
                            metadata={"hnsw:space": "cosine"},
                        )
        except Exception as e:
            if "dimension" in str(e).lower():
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
            "ExperientialStore initialized: %d entries in '%s'",
            self._collection.count(), self.collection_name,
        )

    def add_entry(
        self,
        text: str,
        metadata: dict,
        entry_id: str | None = None,
    ) -> bool:
        """Add a single journal entry."""
        if not text.strip():
            return False

        if entry_id is None:
            existing = self._collection.count()
            entry_id = f"exp_{existing:06d}"

        # Ensure immutable epistemic status.
        metadata.setdefault("epistemic_status", "subjective/phenomenological")

        from app.memory.chromadb_manager import embed
        try:
            self._collection.add(
                documents=[text],
                metadatas=[metadata],
                embeddings=[embed(text)],
                ids=[entry_id],
            )
            # PROGRAM §56 — source ledger dual-write.
            try:
                from app.memory.source_ledger import hook_collection_add
                hook_collection_add(
                    "experiential", self.collection_name,
                    [entry_id], [text], [metadata],
                )
            except Exception:
                logger.debug("ExperientialStore: source_ledger hook failed", exc_info=True)
            return True
        except Exception as e:
            logger.error("Failed to add journal entry: %s", e)
            return False

    def add_documents(
        self,
        chunks: list[str],
        metadatas: list[dict],
        ids: list[str] | None = None,
    ) -> int:
        if not chunks:
            return 0

        if ids is None:
            existing = self._collection.count()
            ids = [f"exp_{existing + i:06d}" for i in range(len(chunks))]

        for m in metadatas:
            m.setdefault("epistemic_status", "subjective/phenomenological")

        from app.memory.chromadb_manager import embed

        batch_size = 50
        total = 0
        for i in range(0, len(chunks), batch_size):
            bc = chunks[i:i + batch_size]
            bm = metadatas[i:i + batch_size]
            bi = ids[i:i + batch_size]
            self._collection.add(
                documents=bc, metadatas=bm,
                embeddings=[embed(c) for c in bc], ids=bi,
            )
            total += len(bc)
            # PROGRAM §56 — source ledger dual-write.
            try:
                from app.memory.source_ledger import hook_collection_add
                hook_collection_add("experiential", self.collection_name, bi, bc, bm)
            except Exception:
                logger.debug("ExperientialStore: source_ledger hook failed", exc_info=True)
        return total

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

        params: dict = {
            "query_embeddings": [embed(query_text)],
            "n_results": min(n_results, count),
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            params["where"] = where_filter

        try:
            results = self._collection.query(**params)
        except Exception as e:
            logger.error("Experiential query failed: %s", e)
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
            from app.retrieval.temporal import apply_temporal_decay
        except Exception:
            return self.query(query_text, n_results, where_filter, min_score)

        from app.retrieval.config import RERANK_TOP_K_INPUT
        candidates = self.query(query_text, RERANK_TOP_K_INPUT, where_filter, min_score)
        if not candidates:
            return []

        # Journal entries benefit from recency — weight newer entries higher.
        candidates = apply_temporal_decay(candidates, timestamp_field="created_at")
        return rerank(query_text, candidates, top_k=n_results)

    def get_stats(self) -> dict:
        count = self._collection.count()
        entry_types: set[str] = set()
        agents: set[str] = set()

        if count > 0:
            all_data = self._collection.get(include=["metadatas"])
            if all_data and all_data["metadatas"]:
                for meta in all_data["metadatas"]:
                    entry_types.add(meta.get("entry_type", "Unknown"))
                    agents.add(meta.get("agent", "Unknown"))

        return {
            "collection_name": self.collection_name,
            "total_entries": count,
            "entry_types": sorted(entry_types - {"Unknown"}),
            "agents": sorted(agents - {"Unknown"}),
        }

    def reset_collection(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )


_store: ExperientialStore | None = None

def get_store() -> ExperientialStore:
    global _store
    if _store is None:
        _store = ExperientialStore()
    return _store
