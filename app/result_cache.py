"""
result_cache.py — Semantic result caching for crew outputs.

Caches crew results in ChromaDB keyed by the task description embedding.
Before dispatching a crew, check if a semantically similar task was answered
recently (cosine similarity >= threshold). Returns cached result if found,
avoiding redundant LLM calls for recurring or near-identical questions.

Cache entries expire after a configurable TTL (default 1 hour).
"""

import logging
import threading
import time
import uuid
from typing import Optional

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "result_cache"
_store_counter = 0  # E3: only prune every 50 stores
_SIMILARITY_THRESHOLD = 0.92  # cosine similarity cutoff
_TTL_SECONDS = 3600  # 1 hour default
_MAX_CACHED = 500  # prune beyond this

_lock = threading.Lock()


def _get_collection():
    from app.memory.chromadb_manager import get_client
    client = get_client()
    return client.get_or_create_collection(
        _COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _embed(text: str) -> list[float]:
    from app.memory.chromadb_manager import embed
    return embed(text)


def lookup(crew_name: str, task: str) -> Optional[str]:
    """Check cache for a semantically similar previous result.

    Returns the cached result string if found, or None.
    """
    try:
        col = _get_collection()
        if col.count() == 0:
            return None

        results = col.query(
            query_embeddings=[_embed(task)],
            n_results=3,
            where={"crew": crew_name},
            include=["documents", "metadatas", "distances"],
        )

        if not results["documents"] or not results["documents"][0]:
            return None

        now = time.time()
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # Cosine distance: 0 = identical, 2 = opposite
            # Similarity = 1 - distance
            similarity = 1.0 - dist
            cached_at = meta.get("cached_at", 0)
            ttl = meta.get("ttl", _TTL_SECONDS)

            if similarity >= _SIMILARITY_THRESHOLD and (now - cached_at) < ttl:
                logger.info(
                    f"result_cache HIT: crew={crew_name}, "
                    f"similarity={similarity:.3f}, age={now - cached_at:.0f}s"
                )
                return doc

        return None
    except Exception:
        logger.debug("result_cache lookup failed", exc_info=True)
        return None


def store(crew_name: str, task: str, result: str, ttl: int = _TTL_SECONDS):
    """Cache a crew result for future semantic lookups."""
    try:
        col = _get_collection()
        col.add(
            documents=[result],
            embeddings=[_embed(task)],
            metadatas=[{
                "crew": crew_name,
                "task_preview": task[:200],
                "cached_at": time.time(),
                "ttl": ttl,
            }],
            ids=[str(uuid.uuid4())],
        )
        logger.debug(f"result_cache STORE: crew={crew_name}, task={task[:80]}")

        # E3: Prune old entries only every 50 stores (not every time).
        # col.count() is O(n) and _prune() loads all metadata — expensive.
        global _store_counter
        _store_counter += 1
        if _store_counter % 50 == 0 and col.count() > _MAX_CACHED:
            _prune(col)
    except Exception:
        logger.debug("result_cache store failed", exc_info=True)


def _prune(col):
    """Remove expired entries from the cache."""
    try:
        all_data = col.get(include=["metadatas"])
        now = time.time()
        to_delete = []
        for id_, meta in zip(all_data["ids"], all_data["metadatas"]):
            cached_at = meta.get("cached_at", 0)
            ttl = meta.get("ttl", _TTL_SECONDS)
            if (now - cached_at) > ttl:
                to_delete.append(id_)
        if to_delete:
            col.delete(ids=to_delete)
            logger.debug(f"result_cache pruned {len(to_delete)} expired entries")
    except Exception:
        logger.debug("result_cache prune failed", exc_info=True)


def invalidate(crew_name: str = None):
    """Clear cache entries, optionally filtered by crew."""
    try:
        col = _get_collection()
        if crew_name:
            all_data = col.get(where={"crew": crew_name}, include=["metadatas"])
            if all_data["ids"]:
                col.delete(ids=all_data["ids"])
        else:
            from app.memory.chromadb_manager import get_client
            get_client().delete_collection(_COLLECTION_NAME)
    except Exception:
        logger.debug("result_cache invalidate failed", exc_info=True)
