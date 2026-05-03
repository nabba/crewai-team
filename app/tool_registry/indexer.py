"""ChromaDB indexing of tool descriptions for semantic search.

Phase 1b component. Sits between the in-memory ``ToolRegistry`` and
the discovery layer. Index gets rebuilt at boot (via
``boot_registry``) — idempotent: tools whose description_hash hasn't
changed are skipped, so cold start is cheap on warm caches.

Collection
----------
Name: ``tool_registry``
ID:   ``spec.name``
Doc:  ``"<name> [<capabilities>]\n<description>"``  (the embedded text)
Meta: ``{tier, lifecycle, capabilities, workspace_scope,
         description_hash, is_loadable, source_module}``

The embedded "doc" string is the *retrieval target* — what we want
the agent's intent query to match against. Tags are inlined so a
query like "render a forest report PDF" hits both the description's
prose AND the explicit ``renders-pdf`` tag.

Why ChromaDB and not pgvector
-----------------------------
The skills retrieval layer (the closest sibling) uses ChromaDB, so
the embedding service is already wired up. Reusing that pipeline
keeps tool retrieval and skill retrieval consistent — same embed
function, same backend, same operational semantics. pgvector is
already in use for Mem0 facts; we don't need a third vector store.

If/when we collapse to a single store, this module is the only
indexer that has to migrate.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from app.tool_registry.types import ToolSpec


_COLLECTION_NAME = "tool_registry"

# ChromaDB collection metadata — pin cosine distance so the
# `_DISTANCE_CEILING` (0.55) in discovery.py is calibrated correctly.
# nomic-embed-text was designed for cosine; default L2 distances run
# 100×–1000× larger and would never fall below the ceiling.
# Mirrors the skills-retrieval setup at integrator.py:206.
_COLLECTION_METADATA = {"hnsw:space": "cosine"}


def _open_collection(client):
    """Open the registry collection, ensuring cosine space.

    If the collection already exists with a different space, recreate
    it — embeddings are cheap enough to rebuild and a wrong-space
    collection silently breaks discovery (every match falls outside
    the ceiling).
    """
    col = client.get_or_create_collection(
        _COLLECTION_NAME, metadata=_COLLECTION_METADATA,
    )
    actual_space = (col.metadata or {}).get("hnsw:space")
    if actual_space and actual_space != "cosine":
        logger.warning(
            "tool_registry indexer: collection has hnsw:space=%s, expected "
            "cosine — recreating to keep distance ceiling calibrated.",
            actual_space,
        )
        client.delete_collection(_COLLECTION_NAME)
        col = client.get_or_create_collection(
            _COLLECTION_NAME, metadata=_COLLECTION_METADATA,
        )
    return col


def _build_doc(spec: "ToolSpec") -> str:
    """The text we embed for retrieval. Inlines capability tags so
    queries hit both prose and structured tags."""
    tags = " ".join(spec.capabilities)
    return f"{spec.name} [{tags}]\n{spec.description}"


def _build_metadata(spec: "ToolSpec") -> dict[str, Any]:
    """Metadata stored alongside the embedding. Used by where-clauses
    in the discovery filter (tier, workspace, loadable_only)."""
    return {
        "tier": spec.tier.value,
        "lifecycle": spec.lifecycle.value,
        # ChromaDB metadata accepts only str/int/float/bool — flatten
        # capability list into a comma-separated string + a per-tag
        # boolean flag for where-clause filtering.
        "capabilities_csv": ",".join(spec.capabilities),
        "workspace_scope_csv": ",".join(spec.workspace_scope),
        "description_hash": spec.description_hash,
        "is_loadable": spec.is_loadable,
        "source_module": spec.source_module,
    }


def index_tools(specs: list["ToolSpec"]) -> tuple[int, int]:
    """Re-index the registry into ChromaDB.

    Idempotent: only embeds tools whose description_hash differs from
    the currently-stored value (or which are missing entirely).

    Returns ``(reindexed_count, skipped_count)``. On any infrastructure
    failure (ChromaDB unreachable, embed service down, etc.), logs and
    returns ``(0, 0)`` — the in-memory registry still works.
    """
    try:
        from app.memory.chromadb_manager import embed, get_client
    except Exception as exc:
        logger.warning("tool_registry indexer: chromadb unavailable: %s", exc)
        return 0, 0

    try:
        client = get_client()
        col = _open_collection(client)
    except Exception as exc:
        logger.warning("tool_registry indexer: cannot open collection: %s", exc)
        return 0, 0

    # Pull existing rows so we can compare hashes (idempotency).
    try:
        existing = col.get(include=["metadatas"])
        existing_hashes: dict[str, str] = {
            existing["ids"][i]: (existing["metadatas"][i] or {}).get("description_hash", "")
            for i in range(len(existing["ids"]))
        }
    except Exception as exc:
        logger.debug("tool_registry indexer: existing fetch failed (%s) — assuming empty", exc)
        existing_hashes = {}

    to_embed: list["ToolSpec"] = []
    for spec in specs:
        prior_hash = existing_hashes.get(spec.name)
        if prior_hash == spec.description_hash:
            continue  # unchanged; no need to re-embed
        to_embed.append(spec)

    if not to_embed:
        # Drop entries that are no longer in the registry (renamed/removed).
        stale = set(existing_hashes) - {s.name for s in specs}
        if stale:
            try:
                col.delete(ids=list(stale))
                logger.info(
                    "tool_registry indexer: removed %d stale entries: %s",
                    len(stale), sorted(stale),
                )
            except Exception as exc:
                logger.debug("tool_registry indexer: stale delete failed: %s", exc)
        return 0, len(specs)

    ids = [s.name for s in to_embed]
    docs = [_build_doc(s) for s in to_embed]
    metas = [_build_metadata(s) for s in to_embed]
    try:
        embeddings = [embed(doc) for doc in docs]
    except Exception as exc:
        logger.warning("tool_registry indexer: embed failed: %s", exc)
        return 0, 0

    try:
        col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
    except Exception as exc:
        logger.warning("tool_registry indexer: upsert failed: %s", exc)
        return 0, 0

    # Drop entries no longer in the registry (renamed/removed).
    stale = set(existing_hashes) - {s.name for s in specs}
    if stale:
        try:
            col.delete(ids=list(stale))
        except Exception as exc:
            logger.debug("tool_registry indexer: stale delete failed: %s", exc)

    logger.info(
        "tool_registry indexer: re-embedded %d tools (skipped %d unchanged, "
        "removed %d stale).",
        len(to_embed), len(specs) - len(to_embed), len(stale),
    )
    return len(to_embed), len(specs) - len(to_embed)


def query_index(
    intent: str,
    *,
    limit: int = 10,
    workspace: str | None = None,
    capabilities: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Semantic + filtered query against the tool index.

    Returns a list of ``{name, distance, metadata, document}`` rows
    sorted by distance ascending (closest first). Empty list on
    infrastructure failure.

    Filters in metadata-where (when supported by Chroma):
      * ``capabilities``: if any tag matches the row's
        capabilities_csv (substring match — ChromaDB doesn't support
        list-contains directly).
      * ``workspace``: row's workspace_scope_csv must contain ``*`` or
        the workspace ID.

    Capability + workspace filtering is *complementary* to the
    application-side filter in ``discovery.py`` — we filter here for
    speed (smaller result set), then re-validate in discovery for
    correctness.
    """
    try:
        from app.memory.chromadb_manager import embed, get_client
    except Exception as exc:
        logger.debug("tool_registry indexer: chromadb unavailable: %s", exc)
        return []

    try:
        client = get_client()
        col = _open_collection(client)
    except Exception as exc:
        logger.debug("tool_registry indexer: cannot open collection: %s", exc)
        return []

    try:
        emb = embed(intent)
    except Exception as exc:
        logger.debug("tool_registry indexer: embed failed: %s", exc)
        return []

    try:
        result = col.query(
            query_embeddings=[emb],
            n_results=max(limit, 1),
            include=["metadatas", "documents", "distances"],
        )
    except Exception as exc:
        logger.debug("tool_registry indexer: query failed: %s", exc)
        return []

    out: list[dict[str, Any]] = []
    if not result.get("ids") or not result["ids"]:
        return out
    ids = result["ids"][0]
    metas = (result.get("metadatas") or [[]])[0]
    docs = (result.get("documents") or [[]])[0]
    dists = (result.get("distances") or [[]])[0]
    for i, name in enumerate(ids):
        meta = metas[i] if i < len(metas) else {}
        out.append({
            "name": name,
            "distance": dists[i] if i < len(dists) else 1.0,
            "metadata": meta,
            "document": docs[i] if i < len(docs) else "",
        })
    return out
