"""
Mem0 persistent memory manager — cross-session fact extraction and graph memory.

Mem0 adds two capabilities ChromaDB doesn't provide:
  1. Automatic fact extraction from conversations (LLM-based)
  2. Entity relationship graph via Neo4j

This coexists with ChromaDB — Mem0 handles persistent cross-session knowledge,
ChromaDB handles real-time operational state (beliefs, policies, self-reports).

All functions degrade gracefully: if Mem0 is unavailable (postgres/neo4j down,
mem0 disabled), they return empty results and log warnings.
"""
import logging
import re
import threading

logger = logging.getLogger(__name__)

_client = None
_client_lock = threading.Lock()
_init_failed = False

# Input limits to prevent resource exhaustion
_MAX_FACT_LENGTH = 10_000       # max bytes for a single fact
_MAX_MESSAGE_LENGTH = 50_000    # max bytes for conversation extraction
_MAX_QUERY_LENGTH = 2_000       # max bytes for search queries

def _sanitize_exc(exc: Exception) -> str:
    """Redact connection strings and credentials from exception messages."""
    msg = str(exc)
    msg = re.sub(r'postgresql://[^@\s]+@[^\s/]+', 'postgresql://***@***', msg)
    msg = re.sub(r'bolt://[^\s]+', 'bolt://***', msg)
    msg = re.sub(r'password[=:]\S+', 'password=***', msg, flags=re.IGNORECASE)
    return msg


def _bump_error_counter() -> None:
    """Increment the Prometheus connection-error counter, swallowing any
    failure to import the metrics module (keeps the warning path total-trust).

    Used as the last line of every ``except Exception`` block in this file
    so the BotArmyPostgresDown alert picks up degraded Mem0 even when
    running against managed RDS / Cloud SQL where ``up{job=...-postgres}``
    isn't a meaningful probe.
    """
    try:
        from app.observability.metrics import MEM0_POSTGRES_CONNECTION_ERRORS_TOTAL
        MEM0_POSTGRES_CONNECTION_ERRORS_TOTAL.inc()
    except Exception:
        pass

def _validate_text(text: str, max_bytes: int) -> bool:
    """Validate text input: non-empty, within size limit, valid UTF-8, no null bytes."""
    if not text or not isinstance(text, str):
        return False
    if '\x00' in text:
        return False
    if len(text.encode('utf-8', errors='replace')) > max_bytes:
        return False
    return True

def _get_config() -> dict:
    """Build Mem0 config from application settings."""
    from app.config import get_settings
    s = get_settings()

    pg_url = s.mem0_postgres_url  # property — builds from components
    if not pg_url:
        raise ValueError("mem0: MEM0_POSTGRES_PASSWORD not set — cannot connect to postgres")

    config = {
        "vector_store": {
            "provider": "pgvector",
            "config": {
                "connection_string": pg_url,
                "collection_name": "crewai_memories",
            },
        },
        "llm": {
            "provider": "litellm",
            "config": {
                "model": s.mem0_llm_model,
                "temperature": 0.1,
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": s.mem0_embedder_model,
            },
        },
        "version": "v1.1",
    }

    # Graph store is optional — only add if Neo4j URL and password are configured
    neo4j_url = s.mem0_neo4j_url
    neo4j_pw = s.mem0_neo4j_password.get_secret_value()
    if neo4j_url and neo4j_pw:
        config["graph_store"] = {
            "provider": "neo4j",
            "config": {
                "url": neo4j_url,
                "username": s.mem0_neo4j_user,
                "password": neo4j_pw,
            },
        }

    return config

def get_client():
    """Thread-safe singleton Mem0 Memory client."""
    global _client, _init_failed
    if _init_failed:
        return None
    if _client is not None:
        return _client
    with _client_lock:
        if _client is not None:
            return _client
        if _init_failed:
            return None
        try:
            from app.config import get_settings
            if not get_settings().mem0_enabled:
                logger.info("mem0: disabled via settings")
                _init_failed = True
                return None

            from mem0 import Memory
            config = _get_config()
            _client = Memory.from_config(config)
            logger.info("mem0: client initialised (pgvector + neo4j)")
            return _client
        except Exception as exc:
            logger.warning(f"mem0: init failed, running without persistent memory: {_sanitize_exc(exc)}")
            _init_failed = True
            _bump_error_counter()
            return None

def _get_user_id() -> str:
    """Default user_id for the single-user system."""
    from app.config import get_settings
    return get_settings().mem0_user_id

# ── Public API ────────────────────────────────────────────────────────────────

def store_memory(
    text: str,
    agent_id: str | None = None,
    metadata: dict | None = None,
) -> dict | None:
    """Store a fact/finding in Mem0.

    Pattern C hybrid scoping:
    - With agent_id: private to that agent
    - Without agent_id: shared pool (all agents can see)
    """
    client = get_client()
    if not client:
        return None
    if not _validate_text(text, _MAX_FACT_LENGTH):
        logger.debug("mem0: store rejected — invalid or oversized text")
        return None
    try:
        messages = [{"role": "user", "content": text}]
        kwargs = {"user_id": _get_user_id()}
        if agent_id:
            kwargs["agent_id"] = agent_id
        if metadata:
            kwargs["metadata"] = metadata
        result = client.add(messages, **kwargs)
        logger.debug(f"mem0: stored memory (agent={agent_id}): {text[:80]}")
        return result
    except Exception as exc:
        logger.warning(f"mem0: store failed: {_sanitize_exc(exc)}")
        _bump_error_counter()
        return None

# ── Stage 5.4: async writes — move Mem0 fact extraction off critical path ──
# Mem0's `.add()` calls an LLM to extract facts; this can take 1-3 s per call.
# Most callers don't consume the return value — they fire-and-forget — yet
# previously blocked the user response. We now enqueue those writes onto a
# small bounded thread pool and return immediately.

import atexit as _atexit
from concurrent.futures import ThreadPoolExecutor as _TPE

_async_write_pool: _TPE | None = None
_async_pool_lock = threading.Lock()


def _get_async_pool() -> _TPE:
    global _async_write_pool
    if _async_write_pool is not None:
        return _async_write_pool
    with _async_pool_lock:
        if _async_write_pool is None:
            _async_write_pool = _TPE(max_workers=2, thread_name_prefix="mem0-writer")
            _atexit.register(lambda: _async_write_pool and _async_write_pool.shutdown(wait=False))
        return _async_write_pool


def store_memory_async(text: str, agent_id: str | None = None, metadata: dict | None = None) -> None:
    """Fire-and-forget wrapper around store_memory().

    Use this on hot-path writes where the caller doesn't need the return value.
    Never blocks, never raises — failures are logged inside store_memory itself.
    """
    try:
        _get_async_pool().submit(store_memory, text, agent_id, metadata)
    except Exception as exc:
        logger.debug(f"mem0: async enqueue failed, falling through to sync: {exc}")
        store_memory(text, agent_id, metadata)


def store_conversation_async(
    messages: list[dict],
    agent_id: str | None = None,
    run_id: str | None = None,
) -> None:
    """Fire-and-forget wrapper around store_conversation().

    Biggest perceived-latency win in Stage 5.4 — Mem0's LLM-based fact
    extraction on a conversation can take 1-3 s. This enqueues the write so
    the user response returns immediately.
    """
    try:
        _get_async_pool().submit(store_conversation, messages, agent_id, run_id)
    except Exception as exc:
        logger.debug(f"mem0: async enqueue failed, falling through to sync: {exc}")
        store_conversation(messages, agent_id, run_id)


def store_conversation(
    messages: list[dict],
    agent_id: str | None = None,
    run_id: str | None = None,
) -> dict | None:
    """Store a conversation for automatic fact extraction.

    messages format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    This is the key Mem0 differentiator — LLM-based extraction of facts.
    """
    client = get_client()
    if not client:
        return None
    # Validate all messages
    total_bytes = 0
    for msg in messages:
        content = msg.get("content", "")
        if not isinstance(content, str):
            return None
        total_bytes += len(content.encode('utf-8', errors='replace'))
    if total_bytes > _MAX_MESSAGE_LENGTH:
        logger.debug("mem0: conversation too large, skipping extraction")
        return None
    try:
        kwargs = {"user_id": _get_user_id()}
        if agent_id:
            kwargs["agent_id"] = agent_id
        if run_id:
            kwargs["run_id"] = run_id
        result = client.add(messages, **kwargs)
        logger.debug(f"mem0: stored conversation ({len(messages)} msgs, agent={agent_id})")
        return result
    except Exception as exc:
        logger.warning(f"mem0: conversation store failed: {_sanitize_exc(exc)}")
        _bump_error_counter()
        return None

def search_memory(
    query: str,
    agent_id: str | None = None,
    n: int = 5,
) -> list[dict]:
    """Search memories. Returns list of {memory, score, ...} dicts.

    - With agent_id: searches that agent's private pool
    - Without agent_id: searches shared pool
    """
    client = get_client()
    if not client:
        return []
    if not _validate_text(query, _MAX_QUERY_LENGTH):
        return []
    n = min(max(1, n), 20)  # cap results
    try:
        # Mem0 ≥ 1.x rejects top-level `user_id` / `agent_id` on search();
        # both must travel inside `filters={}`.
        filters: dict = {"user_id": _get_user_id()}
        if agent_id:
            filters["agent_id"] = agent_id
        results = client.search(query=query, filters=filters, limit=n)
        # Mem0 returns {"results": [...]} or a list directly depending on version
        if isinstance(results, dict):
            return results.get("results", [])
        return results if isinstance(results, list) else []
    except Exception as exc:
        logger.warning(f"mem0: search failed: {_sanitize_exc(exc)}")
        _bump_error_counter()
        return []

def search_shared(query: str, n: int = 5) -> list[dict]:
    """Search the shared memory pool (no agent_id filter)."""
    return search_memory(query, agent_id=None, n=n)

def search_agent(query: str, agent_id: str, n: int = 5) -> list[dict]:
    """Search an agent's private memory pool."""
    return search_memory(query, agent_id=agent_id, n=n)

def get_all_memories(agent_id: str | None = None) -> list[dict]:
    """Get all stored memories for a user/agent."""
    client = get_client()
    if not client:
        return []
    try:
        kwargs = {"user_id": _get_user_id()}
        if agent_id:
            kwargs["agent_id"] = agent_id
        results = client.get_all(**kwargs)
        if isinstance(results, dict):
            return results.get("results", [])
        return results if isinstance(results, list) else []
    except Exception as exc:
        logger.warning(f"mem0: get_all failed: {_sanitize_exc(exc)}")
        _bump_error_counter()
        return []
