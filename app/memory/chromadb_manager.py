"""
chromadb_manager.py — ChromaDB vector memory with Metal-accelerated embeddings.

Embedding strategy (in priority order):
  1. Ollama nomic-embed-text via Metal GPU (~15ms/call, 768-dim)
  2. Refused — CPU fallback disabled to prevent 384→768 dimension corruption.

ALL embeddings system-wide are pinned to 768-dim (nomic-embed-text).
If Ollama is unreachable, embed() raises EmbeddingUnavailableError.
This protects ChromaDB collections from silent data corruption caused by
mixing 384-dim and 768-dim vectors.

IMPORTANT: Never change _EMBED_DIM without migrating ALL ChromaDB collections
AND all pgvector columns (agent_experiences, workspace_items, beliefs).
"""

import chromadb
import functools
import logging
import os
import threading
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Stage 2: shared httpx client for Ollama embeddings ──────────────────────
# Module-level Client with connection pooling + keepalive. Saves 3-8ms per
# embed by reusing the TCP/TLS connection across the hundreds of embed calls
# per user request. Lazy-init so imports don't fail if httpx isn't installed.
_ollama_http_client = None
_ollama_http_lock = threading.Lock()


def _get_ollama_http():
    global _ollama_http_client
    if _ollama_http_client is not None:
        return _ollama_http_client
    with _ollama_http_lock:
        if _ollama_http_client is not None:
            return _ollama_http_client
        try:
            import httpx
            _ollama_http_client = httpx.Client(
                timeout=10.0,
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=40),
            )
        except ImportError:
            _ollama_http_client = False  # sentinel: httpx unavailable
        return _ollama_http_client

PERSIST_DIR = Path("/app/workspace/memory")
TEAM_COLLECTION = "team_shared"

# ── Embedding backend selection ──────────────────────────────────────────────

# Ollama URL (from inside Docker: host.docker.internal; native: localhost)
_OLLAMA_URL = os.environ.get(
    "OLLAMA_EMBED_URL",
    os.environ.get("LOCAL_LLM_BASE_URL", "http://host.docker.internal:11434"),
)
_OLLAMA_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
_EMBED_DIM = 768  # IMMUTABLE — pinned to Ollama nomic-embed-text dimension.
                   # All ChromaDB collections + pgvector columns depend on this.
_embed_backend = "unknown"  # "ollama" or "unavailable" (cpu fallback removed)
_backend_lock = threading.Lock()


class EmbeddingUnavailableError(RuntimeError):
    """Raised when Ollama embedding backend is unavailable."""
    pass


def _ollama_embed(text: str) -> list[float] | None:
    """Get embedding from Ollama via Metal GPU. Returns None on failure.

    Uses a shared pooled httpx.Client for TCP keepalive (~3-8ms/call saved).
    Falls back to the legacy `requests.post` path if httpx is unavailable.
    """
    client = _get_ollama_http()
    try:
        if client and client is not False:
            resp = client.post(
                f"{_OLLAMA_URL}/api/embeddings",
                json={"model": _OLLAMA_MODEL, "prompt": text},
            )
            if resp.status_code == 200:
                emb = resp.json().get("embedding")
                if emb:
                    return emb
            return None
        # Fallback: httpx not installed
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


def _detect_backend() -> tuple[str, int]:
    """Detect the embedding backend. Only Ollama (768-dim) is supported."""
    global _embed_backend
    emb = _ollama_embed("test")
    if emb:
        _embed_backend = "ollama"
        actual_dim = len(emb)
        if actual_dim != _EMBED_DIM:
            logger.error(
                f"CRITICAL: Ollama {_OLLAMA_MODEL} returned {actual_dim}-dim "
                f"but system is pinned to {_EMBED_DIM}-dim. "
                f"Check OLLAMA_EMBED_MODEL setting."
            )
        logger.info(
            f"Embedding backend: Ollama Metal GPU ({_OLLAMA_MODEL}, "
            f"{_EMBED_DIM}-dim, ~15ms/call)"
        )
        return _embed_backend, _EMBED_DIM
    _embed_backend = "unavailable"
    logger.warning(
        f"Embedding backend: UNAVAILABLE — Ollama not reachable at {_OLLAMA_URL}. "
        f"Store/retrieve operations will skip until Ollama is available."
    )
    return _embed_backend, _EMBED_DIM


def _raw_embed(text: str) -> list[float]:
    """Get 768-dim embedding from Ollama.

    Raises EmbeddingUnavailableError if Ollama is down. No CPU fallback —
    mixing 384-dim and 768-dim embeddings silently corrupts vector stores.
    """
    global _embed_backend
    if _embed_backend == "unknown":
        with _backend_lock:
            if _embed_backend == "unknown":
                _detect_backend()
    if _embed_backend == "unavailable":
        # Retry Ollama — it may have come back
        emb = _ollama_embed(text)
        if emb:
            with _backend_lock:
                _embed_backend = "ollama"
            logger.info("Embedding backend recovered: Ollama available again")
            return emb
        raise EmbeddingUnavailableError(
            "Ollama embedding unavailable — all embeddings are pinned to "
            f"768-dim ({_OLLAMA_MODEL}). No CPU fallback."
        )
    # _embed_backend == "ollama"
    emb = _ollama_embed(text)
    if emb:
        return emb
    # Ollama went down mid-session — refuse to produce wrong-dimension vectors
    raise EmbeddingUnavailableError(
        f"Ollama embedding failed mid-session — refusing to produce "
        f"non-{_EMBED_DIM}-dim vectors"
    )


@functools.lru_cache(maxsize=4096)
def _embed_cached(text: str) -> tuple:
    """LRU-cached embedding computation (L1, in-proc).

    Avoids re-encoding the same text multiple times per request. Size bumped
    from 512 → 4096 in Stage 3 since sentience runs many embeds per request.
    Returns tuple for hashability.
    """
    # L2: check disk cache first — survives container restart.
    try:
        from app.memory import disk_cache as _dc
        cached = _dc.embed_get(text)
        if cached is not None and len(cached) == _EMBED_DIM:
            return tuple(cached)
    except Exception:
        pass

    vec = _raw_embed(text)
    # Write-through to L2 (fire-and-forget).
    try:
        from app.memory import disk_cache as _dc
        _dc.embed_put(text, list(vec))
    except Exception:
        pass
    return tuple(vec)


def embed(text: str) -> list[float]:
    """Get embedding for text, using LRU cache + Metal GPU."""
    return list(_embed_cached(text))


def get_embed_dim() -> int:
    """Return the pinned embedding dimension (768 for Ollama nomic-embed-text)."""
    return _EMBED_DIM


# ── ChromaDB client ──────────────────────────────────────────────────────────

_client = None
_client_lock = threading.Lock()

# Q3.1 (2026-05-11) — KB-rooted client registry. The default ``get_client()``
# points at PERSIST_DIR (the ``memory`` KB). Knowledge bases other than
# ``memory`` (philosophy, episteme, knowledge, experiential, tensions,
# aesthetics, …) live under their own workspace subdirectories. Callers
# that need to operate on those KBs — embedding-migration dual-write,
# cutover, the chromadb_rebuild CLI — use ``get_kb_client(kb_name)`` so
# they reach the right persist dir instead of silently writing into
# ``workspace/memory``.
_kb_clients: dict[str, object] = {}


def get_client():
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is None:
            _client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    return _client


def get_kb_client(kb_name: str):
    """Return the ChromaDB client rooted at ``workspace/<kb_name>``.

    For ``kb_name == "memory"`` (the legacy default), this is the same
    singleton ``get_client()`` returns. For other KBs (philosophy /
    episteme / knowledge / experiential / tensions / aesthetics), this
    opens a separate PersistentClient and caches it.

    All clients live until process exit; cache is process-local.
    """
    name = (kb_name or "").strip()
    if not name or name == "memory":
        return get_client()
    cached = _kb_clients.get(name)
    if cached is not None:
        return cached
    with _client_lock:
        cached = _kb_clients.get(name)
        if cached is not None:
            return cached
        try:
            from app.paths import WORKSPACE_ROOT
            persist_dir = Path(WORKSPACE_ROOT) / name
        except Exception:
            persist_dir = Path("/app/workspace") / name
        client = chromadb.PersistentClient(path=str(persist_dir))
        _kb_clients[name] = client
        return client


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
                sample = col.peek(1)  # returns embeddings by default in chromadb 1.x
                embs = sample.get("embeddings") if sample else None
                if embs is not None and len(embs) > 0 and embs[0] is not None and len(embs[0]) > 0:
                    existing_dim = len(sample["embeddings"][0])
                    current_dim = get_embed_dim()
                    if existing_dim != current_dim:
                        logger.warning(
                            f"ChromaDB: dimension mismatch in '{name}' "
                            f"(stored={existing_dim}, model={current_dim}). Recreating."
                        )
                        try:
                            from app.self_awareness.journal import get_journal, JournalEntry, JournalEntryType
                            get_journal().write(JournalEntry(
                                entry_type=JournalEntryType.ERROR,
                                summary=f"ChromaDB '{name}' recreated: dims {existing_dim}→{current_dim}",
                                outcome="degraded",
                            ))
                        except Exception:
                            pass
                        client.delete_collection(name)
                        col = client.get_or_create_collection(name)
        except Exception as e:
            # If peek fails with dimension error, recreate the collection
            if "dimension" in str(e).lower():
                logger.warning(f"Collection '{name}' dimension error — recreating: {e}")
                try:
                    client.delete_collection(name)
                    col = client.get_or_create_collection(name)
                except Exception:
                    pass
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
    # Generate ONE id so source + shadow share it (dual-write hook).
    doc_id = str(uuid.uuid4())
    try:
        col.add(
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata or {}],
            ids=[doc_id],
        )
    except Exception as e:
        # Dimension mismatch: collection has 384-dim but model produces 768-dim
        # Recreate the collection and retry (operational data is ephemeral)
        if "dimension" in str(e).lower():
            logger.warning(f"Dimension mismatch in '{collection_name}' — recreating and retrying")
            _collections.pop(collection_name, None)
            _count_cache.pop(collection_name, None)
            client = get_client()
            client.delete_collection(collection_name)
            col = client.get_or_create_collection(collection_name)
            _collections[collection_name] = col
            col.add(
                documents=[text],
                embeddings=[embedding],
                metadatas=[metadata or {}],
                ids=[doc_id],
            )
        else:
            raise
    _count_cache.pop(collection_name, None)
    # PROGRAM §40 Item 12 — best-effort shadow write. Hook is a no-op
    # unless the migration master switch is on AND the state machine
    # is in a phase that wants shadow writes. Failures swallowed —
    # never block the source write path.
    try:
        from app.memory.embedding_migration.dual_write import maybe_dual_write
        maybe_dual_write(collection_name, doc_id, text, metadata)
    except Exception:
        logger.debug("chromadb_manager: dual_write hook failed", exc_info=True)


def retrieve(collection_name: str, query: str, n: int = 5) -> list[str]:
    n = min(max(1, n), 50)
    col = _get_col(collection_name)
    cnt = _get_count(col, collection_name)
    if cnt == 0:
        return []
    embedding = embed(query)
    try:
        results = col.query(
            query_embeddings=[embedding], n_results=min(n, cnt)
        )
    except Exception as e:
        if "dimension" in str(e).lower():
            logger.warning(
                f"Dimension mismatch in '{collection_name}' during retrieve — "
                f"recreating collection (old data lost): {e}"
            )
            _collections.pop(collection_name, None)
            _count_cache.pop(collection_name, None)
            client = get_client()
            client.delete_collection(collection_name)
            client.get_or_create_collection(collection_name)
            return []
        raise
    # PROGRAM §40 Item 12 — best-effort shadow-read divergence sample.
    # Hook is a no-op unless the migration master switch is on AND the
    # state machine is in SHADOW_READ / READY. Sampling is internal.
    try:
        from app.memory.embedding_migration.shadow_read import maybe_shadow_read
        observed_ids = (results.get("ids") or [[]])[0]
        maybe_shadow_read(
            collection_name, query, list(observed_ids), n_results=n,
        )
    except Exception:
        logger.debug("chromadb_manager: shadow_read hook failed", exc_info=True)
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
    try:
        results = col.query(
            query_embeddings=[embedding],
            n_results=min(n, cnt),
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        if "dimension" in str(e).lower():
            logger.warning(
                f"Dimension mismatch in '{collection_name}' during retrieve — "
                f"recreating collection (old data lost): {e}"
            )
            _collections.pop(collection_name, None)
            _count_cache.pop(collection_name, None)
            client = get_client()
            client.delete_collection(collection_name)
            client.get_or_create_collection(collection_name)
            return []
        raise
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
