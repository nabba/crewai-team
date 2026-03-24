"""
chromadb_manager.py — ChromaDB vector memory with Metal-accelerated embeddings.

Embedding strategy (in priority order):
  1. Ollama nomic-embed-text via Metal GPU (~15ms/call, 768-dim)
  2. CPU SentenceTransformer all-MiniLM-L6-v2 fallback (~500ms/call, 384-dim)

The Ollama path calls the native macOS Ollama instance via HTTP, which uses
Metal GPU for inference.  If Ollama is unreachable, falls back to the CPU
SentenceTransformer model loaded in-process.

IMPORTANT: Switching embedding models changes the vector dimension.
ChromaDB collections created with one dimension are incompatible with another.
On dimension mismatch, the collection is automatically recreated (old data lost).
This is acceptable because ChromaDB stores operational/ephemeral data — the
persistent knowledge is in Mem0 (Postgres+Neo4j) and skill files on disk.
"""

import chromadb
import functools
import logging
import os
import threading
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

PERSIST_DIR = Path("/app/workspace/memory")
TEAM_COLLECTION = "team_shared"

# ── Embedding backend selection ──────────────────────────────────────────────

# Ollama URL (from inside Docker: host.docker.internal; native: localhost)
_OLLAMA_URL = os.environ.get(
    "OLLAMA_EMBED_URL",
    os.environ.get("LOCAL_LLM_BASE_URL", "http://host.docker.internal:11434"),
)
_OLLAMA_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
_EMBED_DIM = 0  # auto-detected on first call
_embed_backend = "unknown"  # "ollama" or "cpu"
_backend_lock = threading.Lock()

# Lazy-loaded CPU fallback
_cpu_model = None


def _get_cpu_model():
    global _cpu_model
    if _cpu_model is None:
        from sentence_transformers import SentenceTransformer
        _cpu_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Loaded CPU fallback embedding model: all-MiniLM-L6-v2 (384-dim)")
    return _cpu_model


def _ollama_embed(text: str) -> list[float] | None:
    """Get embedding from Ollama via Metal GPU. Returns None on failure."""
    try:
        import requests
        resp = requests.post(
            f"{_OLLAMA_URL}/api/embeddings",
            json={"model": _OLLAMA_MODEL, "prompt": text},
            timeout=10,
        )
        if resp.status_code == 200:
            emb = resp.json().get("embedding")
            if emb:
                return emb
        return None
    except Exception:
        return None


def _cpu_embed(text: str) -> list[float]:
    """Get embedding from CPU SentenceTransformer (fallback)."""
    return _get_cpu_model().encode(text).tolist()


def _detect_backend() -> tuple[str, int]:
    """Detect which embedding backend to use. Returns (backend, dim)."""
    global _embed_backend, _EMBED_DIM
    # Try Ollama first
    emb = _ollama_embed("test")
    if emb:
        _embed_backend = "ollama"
        _EMBED_DIM = len(emb)
        logger.info(
            f"Embedding backend: Ollama Metal GPU ({_OLLAMA_MODEL}, "
            f"{_EMBED_DIM}-dim, ~15ms/call)"
        )
        return _embed_backend, _EMBED_DIM
    # Fallback to CPU
    emb = _cpu_embed("test")
    _embed_backend = "cpu"
    _EMBED_DIM = len(emb)
    logger.warning(
        f"Embedding backend: CPU SentenceTransformer (all-MiniLM-L6-v2, "
        f"{_EMBED_DIM}-dim, ~500ms/call) — Ollama not available at {_OLLAMA_URL}"
    )
    return _embed_backend, _EMBED_DIM


def _raw_embed(text: str) -> list[float]:
    """Get embedding using the detected backend."""
    global _embed_backend, _EMBED_DIM
    if _embed_backend == "unknown":
        with _backend_lock:
            if _embed_backend == "unknown":
                _detect_backend()
    if _embed_backend == "ollama":
        emb = _ollama_embed(text)
        if emb:
            return emb
        # Ollama went down — fall back to CPU for this call
        logger.warning("Ollama embedding failed, falling back to CPU for this call")
        return _cpu_embed(text)
    return _cpu_embed(text)


@functools.lru_cache(maxsize=512)
def _embed_cached(text: str) -> tuple:
    """LRU-cached embedding computation.

    Avoids re-encoding the same text multiple times per request.
    Returns tuple for hashability.
    """
    return tuple(_raw_embed(text))


def embed(text: str) -> list[float]:
    """Get embedding for text, using LRU cache + Metal GPU."""
    return list(_embed_cached(text))


def get_embed_dim() -> int:
    """Return the dimension of the current embedding model."""
    global _EMBED_DIM
    if _EMBED_DIM == 0:
        embed("dimension probe")
    return _EMBED_DIM


# ── ChromaDB client ──────────────────────────────────────────────────────────

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
_count_cache: dict[str, int] = {}


def _get_col(name: str):
    """Get a ChromaDB collection, caching the object for reuse.

    If the collection's embedding dimension doesn't match the current model,
    recreate it (operational data is ephemeral — skill files and Mem0 persist).
    """
    if name not in _collections:
        client = get_client()
        col = client.get_or_create_collection(name)
        # Check dimension compatibility if collection has data
        try:
            if col.count() > 0:
                # Peek at one embedding to check dimension
                sample = col.peek(1)
                if sample and sample.get("embeddings") and sample["embeddings"][0]:
                    existing_dim = len(sample["embeddings"][0])
                    current_dim = get_embed_dim()
                    if existing_dim != current_dim:
                        logger.warning(
                            f"Collection '{name}' has {existing_dim}-dim embeddings "
                            f"but model produces {current_dim}-dim — recreating collection"
                        )
                        client.delete_collection(name)
                        col = client.get_or_create_collection(name)
        except Exception:
            pass  # dimension check is best-effort
        _collections[name] = col
    return _collections[name]


def _get_count(col, name: str) -> int:
    """Get collection count, using cached value when available."""
    if name not in _count_cache:
        _count_cache[name] = col.count()
    return _count_cache[name]


# ── Store / Retrieve operations ──────────────────────────────────────────────

def store(collection_name: str, text: str, metadata: dict = None):
    # H1: Validate content before storage to prevent memory poisoning attacks.
    try:
        from app.sanitize import validate_content
        if not validate_content(text):
            logger.warning(
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
    embedding = embed(query)
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
    embedding = embed(query)
    try:
        results = col.query(
            query_embeddings=[embedding],
            n_results=min(n, cnt),
            where=where,
        )
        return results["documents"][0] if results["documents"] else []
    except Exception:
        return retrieve(collection_name, query, n)
