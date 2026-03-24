import chromadb
import functools
import threading
import uuid
from sentence_transformers import SentenceTransformer
from pathlib import Path

PERSIST_DIR = Path("/app/workspace/memory")
TEAM_COLLECTION = "team_shared"

_model = SentenceTransformer("all-MiniLM-L6-v2")  # Runs locally, no API


@functools.lru_cache(maxsize=512)
def _embed_cached(text: str) -> tuple:
    """LRU-cached embedding computation.

    Avoids re-encoding the same text multiple times per request
    (e.g., result_cache lookup + context fetch + cache store all
    embedding the same task string).  Returns tuple for hashability.
    """
    return tuple(_model.encode(text).tolist())


def embed(text: str) -> list[float]:
    """Get embedding for text, using LRU cache."""
    return list(_embed_cached(text))

# Thread-safe singleton — prevents lock contention when multiple threads
# each try to create their own PersistentClient pointing to the same dir.
_client = None
_client_lock = threading.Lock()


def get_client():
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is None:
            _client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    return _client


# E4: Cache collection objects — avoid get_or_create_collection() per operation.
# Also cache count() to avoid O(n) scan on every retrieve call.
_collections: dict[str, object] = {}
_count_cache: dict[str, int] = {}  # collection → last known count


def _get_col(name: str):
    """Get a ChromaDB collection, caching the object for reuse."""
    if name not in _collections:
        _collections[name] = get_client().get_or_create_collection(name)
    return _collections[name]


def _get_count(col, name: str) -> int:
    """Get collection count, using cached value when available."""
    if name not in _count_cache:
        _count_cache[name] = col.count()
    return _count_cache[name]


def store(collection_name: str, text: str, metadata: dict = None):
    # H1: Validate content before storage to prevent memory poisoning attacks.
    try:
        from app.sanitize import validate_content
        if not validate_content(text):
            import logging
            logging.getLogger(__name__).warning(
                f"Memory store BLOCKED — injection pattern detected in "
                f"collection={collection_name}: {text[:80]!r}"
            )
            return
    except ImportError:
        pass
    col = _get_col(collection_name)
    embedding = embed(text)
    col.add(
        documents=[text],
        embeddings=[embedding],
        metadatas=[metadata or {}],
        ids=[str(uuid.uuid4())],
    )
    # Invalidate count cache for this collection
    _count_cache.pop(collection_name, None)


def retrieve(collection_name: str, query: str, n: int = 5) -> list[str]:
    n = min(max(1, n), 50)
    col = _get_col(collection_name)
    cnt = _get_count(col, collection_name)
    if cnt == 0:
        return []
    embedding = embed(query)
    results = col.query(
        query_embeddings=[embedding], n_results=min(n, cnt)
    )
    return results["documents"][0]


def store_team(text: str, metadata: dict = None):
    """Store in the shared team-wide collection (cross-crew sharing)."""
    store(TEAM_COLLECTION, text, metadata)


def retrieve_team(query: str, n: int = 5) -> list[str]:
    """Retrieve from the shared team-wide collection."""
    return retrieve(TEAM_COLLECTION, query, n)


def retrieve_with_metadata(
    collection_name: str, query: str, n: int = 5
) -> list[dict]:
    """Retrieve documents with their metadata and distances."""
    n = min(max(1, n), 50)
    col = _get_col(collection_name)
    cnt = _get_count(col, collection_name)
    if cnt == 0:
        return []
    embedding = embed(query)  # E4: use cached embed(), not raw _model.encode()
    results = col.query(
        query_embeddings=[embedding],
        n_results=min(n, cnt),
        include=["documents", "metadatas", "distances"],
    )
    items = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    for doc, meta, dist in zip(docs, metas, dists):
        items.append({"document": doc, "metadata": meta or {}, "distance": dist})
    return items


def retrieve_filtered(
    collection_name: str, query: str, where: dict, n: int = 5
) -> list[str]:
    """Retrieve documents filtered by a ChromaDB 'where' clause."""
    n = min(max(1, n), 50)
    col = _get_col(collection_name)
    cnt = _get_count(col, collection_name)
    if cnt == 0:
        return []
    embedding = embed(query)  # E4: use cached embed()
    try:
        results = col.query(
            query_embeddings=[embedding],
            n_results=min(n, cnt),
            where=where,
        )
        return results["documents"][0] if results["documents"] else []
    except Exception:
        return retrieve(collection_name, query, n)
