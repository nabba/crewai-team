"""Embedding-model migration framework.

PROGRAM §40 (2026-05-10) — Q3 Item 12.

Migrating an embedding model is the highest-stakes substrate change
the system can do. Today we have:

    _EMBED_DIM = 768  # IMMUTABLE — pinned to Ollama nomic-embed-text

Every ChromaDB collection plus every pgvector column inherits that
dimension. A future move to (say) ``mxbai-embed-large`` (1024-dim)
would silently corrupt every retrieval if done casually.

This framework gives operators a deliberate, reversible path:

  1. **Plan** — declare ``source → target`` model + dim, list the KBs
     and pgvector tables to migrate, persist as
     ``workspace/embedding_migration/plan.json``.
  2. **Dual-write** — every new write embeds with both models in
     parallel; the target embedding goes into a SHADOW collection
     suffix. Reads still go to source.
  3. **Backfill** — drain the existing source collections through
     the target embedder into the shadow collections.
  4. **Shadow-read verification** — for a sampled fraction of live
     queries, run the same query against shadow and record the NDCG@10
     divergence. Convergence threshold is the cutover gate.
  5. **Cutover** — operator-initiated, gated by a Tier-3 amendment
     because changing ``_EMBED_DIM`` rewrites a TIER_IMMUTABLE
     constant. Atomic swap: shadow becomes live, source archives.
  6. **Stand-down** — a configurable retention window keeps the old
     source collections so a regression can be rolled back. After the
     window expires the source is deleted.

Module boundary:

  * ``plan.py``       — typed migration plan + serialization
  * ``state.py``      — runtime state machine over runtime_settings
  * ``dual_write.py`` — embedded helpers called from chromadb_manager
  * ``shadow_read.py``— sampled NDCG@10 divergence telemetry
  * ``verify.py``     — pre-cutover invariants (counts, dim, sanity)
  * ``cutover.py``    — Tier-3-gated atomic swap
  * ``dry_run.py``    — full pipeline against a SANDBOX collection only

The framework runs entirely in OBSERVATIONAL mode by default. The
operator turns each phase on explicitly via the React `/cp/settings`
embedding-migration card.
"""
from __future__ import annotations

__all__ = [
    "plan", "state", "dual_write", "shadow_read",
    "verify", "cutover", "dry_run",
]
