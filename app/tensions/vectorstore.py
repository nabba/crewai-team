"""
tensions/vectorstore.py — ChromaDB vector store for unresolved tensions.
"""

from __future__ import annotations

import logging
from pathlib import Path

import chromadb

from app.tensions import config

logger = logging.getLogger(__name__)


class TensionStore:
    """Persistent vector store for unresolved contradictions and tensions.

    Metadata schema:
        tension_type, pole_a, pole_b, detected_by, context,
        resolution_status, epistemic_status, created_at
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
                    if len(embs[0]) != get_embed_dim():
                        logger.warning("TensionStore: dimension mismatch — recreating")
                        self._client.delete_collection(self.collection_name)
                        col = self._client.get_or_create_collection(
                            name=self.collection_name,
                            metadata={"hnsw:space": "cosine"},
                        )
        except Exception as e:
            if "dimension" in str(e).lower():
                logger.warning("TensionStore: dimension error — recreating: %s", e)
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
            "TensionStore initialized: %d tensions in '%s'",
            self._collection.count(), self.collection_name,
        )

    def add_tension(
        self,
        text: str,
        metadata: dict,
        tension_id: str | None = None,
    ) -> bool:
        if not text.strip():
            return False

        if tension_id is None:
            existing = self._collection.count()
            tension_id = f"ten_{existing:06d}"

        metadata.setdefault("epistemic_status", "unresolved/dialectical")
        metadata.setdefault("resolution_status", "unresolved")

        from app.memory.chromadb_manager import embed
        try:
            self._collection.add(
                documents=[text],
                metadatas=[metadata],
                embeddings=[embed(text)],
                ids=[tension_id],
            )
            # PROGRAM §56 — source ledger dual-write.
            try:
                from app.memory.source_ledger import hook_collection_add
                hook_collection_add(
                    "tensions", self.collection_name,
                    [tension_id], [text], [metadata],
                )
            except Exception:
                logger.debug("TensionsStore: source_ledger hook failed", exc_info=True)
            return True
        except Exception as e:
            logger.error("Failed to add tension: %s", e)
            return False

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
            logger.error("Tension query failed: %s", e)
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

    def get_unresolved(self, n: int = 10) -> list[dict]:
        """Return unresolved tensions (growth edges)."""
        return self.query(
            query_text="unresolved tension conflict paradox",
            n_results=n,
            where_filter={"resolution_status": "unresolved"},
        )

    def get_stats(self) -> dict:
        count = self._collection.count()
        tension_types: set[str] = set()
        detectors: set[str] = set()
        statuses: dict[str, int] = {}

        if count > 0:
            all_data = self._collection.get(include=["metadatas"])
            if all_data and all_data["metadatas"]:
                for meta in all_data["metadatas"]:
                    tension_types.add(meta.get("tension_type", "Unknown"))
                    detectors.add(meta.get("detected_by", "Unknown"))
                    status = meta.get("resolution_status", "unresolved")
                    statuses[status] = statuses.get(status, 0) + 1

        return {
            "collection_name": self.collection_name,
            "total_tensions": count,
            "tension_types": sorted(tension_types - {"Unknown"}),
            "detected_by": sorted(detectors - {"Unknown"}),
            "resolution_statuses": statuses,
        }

    def reset_collection(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )


_store: TensionStore | None = None

def get_store() -> TensionStore:
    global _store
    if _store is None:
        _store = TensionStore()
    return _store
