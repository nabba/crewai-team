"""
result_cache.py — Semantic result caching for crew outputs.

Caches crew results in ChromaDB keyed by the task description embedding.
Before dispatching a crew, check if a semantically similar task was answered
recently (cosine similarity >= threshold). Returns cached result if found,
avoiding redundant LLM calls for recurring or near-identical questions.

Cache entries expire after a configurable TTL (default 1 hour).
"""

import logging
import os
import threading
import time
import uuid

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "result_cache"
_store_counter = 0  # E3: only prune every 50 stores
_SIMILARITY_THRESHOLD = 0.92  # cosine similarity cutoff
# Stage 3.5 — bump TTL 1h → 6h and cap 500 → 2000. Safe because entries are
# still guarded by the similarity threshold; old results on changed topics
# won't re-match because the embedding drifts. Env-overridable for tuning.
_TTL_SECONDS = int(os.environ.get("RESULT_CACHE_TTL_S", "21600"))      # 6 hours
_MAX_CACHED = int(os.environ.get("RESULT_CACHE_MAX", "2000"))          # was 500
_BYPASS = os.environ.get("CACHE_BYPASS", "0") == "1"                    # testing knob

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

def lookup(crew_name: str, task: str) -> str | None:
    """Check cache for a semantically similar previous result.

    Returns the cached result string if found, or None.
    """
    try:
        # Stage 3.5 — CACHE_BYPASS=1 disables entirely (benchmark/debug).
        if _BYPASS:
            return None
        # Q9: Skip cache for short/vague queries — their embeddings are
        # unstable and match unrelated cached results.  "Compare two frontier
        # models" (30 chars) matched a cached DSPy distillation result.
        if len(task.strip()) < 60:
            return None

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
            _ledger_delete("result_cache", to_delete)
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
                _ledger_delete("result_cache", all_data["ids"])
        else:
            from app.memory.chromadb_manager import get_client
            get_client().delete_collection(_COLLECTION_NAME)
    except Exception:
        logger.debug("result_cache invalidate failed", exc_info=True)


def _ledger_delete(collection: str, ids: list[str]) -> None:
    """PROGRAM §56 iter-2 — mirror chromadb deletes into the source
    ledger as tombstones so replay-rebuild doesn't resurrect deleted
    rows. Failure-isolated."""
    try:
        from app.memory.source_ledger import hook_collection_delete
        hook_collection_delete("memory", collection, list(ids))
    except Exception:
        logger.debug("result_cache: ledger delete hook failed", exc_info=True)


def invalidate_by_task(task: str, *, crew_name: str | None = None) -> int:
    """Invalidate any cache entry whose stored task closely matches ``task``.

    Called from abort / timeout / stall-kill paths in ``handle_task`` so
    that a task which got partially-cached mid-flight but never produced
    a real deliverable cannot resurface as a cache HIT on the user's
    next identical request.

    Matches by semantic similarity (same ChromaDB distance used for
    ``lookup``) with a STRICT threshold (0.98+) so we never evict
    legitimate entries for unrelated tasks.  When ``crew_name`` is
    provided, only entries for that crew are considered.

    Returns the number of entries deleted.  Never raises.
    """
    if not task or len(task.strip()) < 10:
        return 0
    try:
        col = _get_collection()
        if col.count() == 0:
            return 0

        where = {"crew": crew_name} if crew_name else None
        results = col.query(
            query_embeddings=[_embed(task)],
            n_results=5,
            where=where,
            include=["metadatas", "distances"],
        )
        ids_out = results.get("ids") or [[]]
        dists = results.get("distances") or [[]]
        if not ids_out or not ids_out[0]:
            return 0

        # Stricter-than-lookup threshold — we only want to evict
        # exact-or-near-exact matches so a noisy but related request
        # doesn't nuke legitimate cached answers.
        _INVALIDATE_SIMILARITY = 0.98
        to_delete: list[str] = []
        for id_, dist in zip(ids_out[0], dists[0]):
            similarity = 1.0 - dist
            if similarity >= _INVALIDATE_SIMILARITY:
                to_delete.append(id_)

        if to_delete:
            col.delete(ids=to_delete)
            _ledger_delete("result_cache", to_delete)
            logger.info(
                "result_cache: INVALIDATE_BY_TASK crew=%s n=%d "
                "(task=%r)", crew_name or "*", len(to_delete), task[:80],
            )
        return len(to_delete)
    except Exception:
        logger.debug("result_cache invalidate_by_task failed", exc_info=True)
        return 0
