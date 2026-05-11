"""Dual-write helpers — embed once, write twice.

PROGRAM §40 (2026-05-10) — Q3 Item 12.

Called from ``chromadb_manager.store(...)`` when migration phase is in
``DUAL_WRITE / BACKFILLING / SHADOW_READ / READY``. The hook:

  1. Embeds the new document with the TARGET model (separate HTTP
     call to the target endpoint; never reuses the source embedding).
  2. Writes to the SHADOW collection at ``<collection>__shadow_<plan_id>``.
  3. Increments the ``shadow_writes`` counter.

Failure model:
  * Target embedding failure → log + skip the shadow write. **Never
    block the source write.** A migration that interrupts production
    is worse than a migration that drops some shadow rows.
  * Shadow collection write failure → log + skip. Same rationale.
  * Target endpoint unreachable → exponential backoff with a soft
    1s ceiling so we don't pile up.

The module is pure helper code — no decorators, no heuristics. The
runtime hook in ``chromadb_manager`` decides when to call us based on
``state.dual_write_enabled()``.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from app.memory.embedding_migration import plan as plan_mod
from app.memory.embedding_migration import state as state_mod

logger = logging.getLogger(__name__)


_SHADOW_SUFFIX = "__shadow"


def shadow_collection_name(source_name: str, plan_id: str) -> str:
    """``team_shared`` + ``ollama-nomic-to-mxbai-2026-Q3`` →
    ``team_shared__shadow_ollama-nomic-to-mxbai-2026-Q3``.

    Suffixed by plan id so a re-attempted migration can run alongside
    a stale shadow from a previous abandoned plan."""
    sanitized = plan_id.replace("/", "_").replace(" ", "_")
    return f"{source_name}{_SHADOW_SUFFIX}_{sanitized}"


# ── Target embedder client (lazy + pluggable) ────────────────────────────


_target_client_lock = threading.Lock()
_target_http = None


def _get_target_http_client():
    """Reuse one httpx client across calls (keepalive)."""
    global _target_http
    if _target_http is not None:
        return _target_http
    with _target_client_lock:
        if _target_http is not None:
            return _target_http
        try:
            import httpx
            _target_http = httpx.Client(
                timeout=10.0,
                limits=httpx.Limits(
                    max_keepalive_connections=10, max_connections=20,
                ),
            )
        except Exception:
            _target_http = False
        return _target_http


@dataclass
class _Backoff:
    """Tiny soft backoff so a flaky target endpoint can't pile up."""
    failures: int = 0
    last_failure_at: float = 0.0
    cooldown_s: float = 0.0


_backoff = _Backoff()
_backoff_lock = threading.Lock()


def _on_failure() -> None:
    with _backoff_lock:
        _backoff.failures += 1
        _backoff.last_failure_at = time.monotonic()
        # Quadratic up to a 60s ceiling: 1, 4, 9, 16, 25, 36, 49, 60, 60, …
        _backoff.cooldown_s = min(60.0, _backoff.failures ** 2)


def _on_success() -> None:
    with _backoff_lock:
        _backoff.failures = 0
        _backoff.cooldown_s = 0.0


def _within_backoff() -> bool:
    with _backoff_lock:
        if _backoff.cooldown_s <= 0:
            return False
        return (time.monotonic() - _backoff.last_failure_at) < _backoff.cooldown_s


# ── Target embedding ─────────────────────────────────────────────────────


def _embed_target(text: str, target_model) -> list[float] | None:
    """Run the target model. Returns ``None`` on failure (logged).

    Provider-specific dispatch — Ollama is the most likely target
    today; OpenAI / Voyage are forward-looking and stub out cleanly.
    """
    if _within_backoff():
        return None
    client = _get_target_http_client()
    if client is False or client is None:
        return None
    try:
        provider = (target_model.provider or "").lower()
        if provider == "ollama":
            url = (target_model.base_url or "http://localhost:11434").rstrip("/")
            r = client.post(
                f"{url}/api/embeddings",
                json={"model": target_model.name, "prompt": text},
                timeout=10.0,
            )
            r.raise_for_status()
            data = r.json()
            emb = data.get("embedding")
            if not emb or len(emb) != target_model.dim:
                logger.debug(
                    "embedding_migration.dual_write: target dim mismatch "
                    "expected=%d got=%d",
                    target_model.dim, len(emb) if emb else 0,
                )
                _on_failure()
                return None
            _on_success()
            return list(emb)
        else:
            # Other providers fall through to "not implemented" — operator
            # wires the call when authoring the plan. Keeps this module
            # provider-light.
            logger.debug(
                "embedding_migration.dual_write: provider %s not implemented",
                provider,
            )
            return None
    except Exception:
        logger.debug(
            "embedding_migration.dual_write: target embed failed",
            exc_info=True,
        )
        _on_failure()
        return None


# ── Plan-target lookup ───────────────────────────────────────────────────


def _matching_target(plan, source_collection: str):
    """Return the plan target (or None) whose collection matches the
    live ``source_collection`` being written to. The matching is
    collection-name-only because the chromadb_manager hook doesn't know
    which KB the call originated from — but since the plan validator
    refuses non-memory KBs today, the routing collapses to: any
    memory-KB write whose collection matches a planned target.

    When the allowlist is widened in a Q3.x follow-up, the caller (the
    hook in chromadb_manager.store) will need to pass the KB context
    in too. For now: matched-by-collection-name is correct."""
    if plan is None:
        return None
    for t in plan.targets:
        if t.kind != "chromadb":
            continue
        if (t.collection or "") == source_collection:
            return t
    return None


# ── Public hook (called from chromadb_manager.store) ─────────────────────


def maybe_dual_write(
    source_collection: str, doc_id: str, text: str, metadata: dict | None,
) -> None:
    """Best-effort shadow write. Never raises.

    Routing: looks up the plan target whose collection matches the
    live source_collection; opens a KB-rooted chromadb client (so the
    shadow lands in the right persist directory); writes there.
    """
    try:
        if not state_mod.dual_write_enabled():
            return
        plan = plan_mod.load_plan()
        if plan is None or not plan.plan_id:
            return
        # Skip dual-write for the shadow collection itself (never recurse).
        if _SHADOW_SUFFIX in source_collection:
            return
        target = _matching_target(plan, source_collection)
        if target is None:
            # The write is to a collection not under migration. Skip
            # silently — the plan only covers a subset of collections.
            return
        target_emb = _embed_target(text, plan.target)
        if target_emb is None:
            return
        from app.memory import chromadb_manager
        # Q3.1 — KB-rooted client. For the legacy ``memory`` KB this is
        # the same singleton ``get_client()`` returns, so behavior is
        # unchanged. For future KBs (once the plan-validator allowlist
        # widens) the shadow lands in the right persist dir.
        client = chromadb_manager.get_kb_client(target.kb or "memory")
        shadow_name = shadow_collection_name(source_collection, plan.plan_id)
        shadow_col = client.get_or_create_collection(shadow_name)
        shadow_col.add(
            ids=[doc_id],
            embeddings=[target_emb],
            documents=[text],
            metadatas=[metadata or {}],
        )
        state_mod.increment_shadow_write(1)
    except Exception:
        # Per docstring: never block the source write. Swallow
        # everything; a healing alert can be wired later if shadow
        # write failures get noisy.
        logger.debug(
            "embedding_migration.dual_write: maybe_dual_write failed",
            exc_info=True,
        )


# ── Backfill driver ──────────────────────────────────────────────────────


def backfill_one_collection(
    source_collection: str, batch_size: int = 100, max_rows: int | None = None,
) -> dict:
    """Stream rows from the source collection, embed with the target
    model, write to the shadow collection. Returns a summary dict.

    Idempotent: skips rows already present in the shadow collection
    (id-based dedup). Routes through the plan target's KB so the
    shadow lands in the correct persist directory."""
    plan = plan_mod.load_plan()
    if plan is None:
        return {"ok": False, "error": "no plan loaded"}
    cur_state = state_mod.get_state()
    if cur_state.phase not in (
        state_mod.PHASE_DUAL_WRITE, state_mod.PHASE_BACKFILLING,
    ):
        return {"ok": False, "error": f"phase {cur_state.phase} disallows backfill"}

    target = _matching_target(plan, source_collection)
    if target is None:
        return {
            "ok": False,
            "error": (
                f"collection {source_collection!r} is not a target in plan "
                f"{plan.plan_id!r} — backfill would create an orphan shadow"
            ),
        }

    from app.memory import chromadb_manager
    client = chromadb_manager.get_kb_client(target.kb or "memory")
    try:
        src = client.get_collection(source_collection)
    except Exception as exc:
        return {"ok": False, "error": f"source not found: {exc}"}
    shadow_name = shadow_collection_name(source_collection, plan.plan_id)
    shadow = client.get_or_create_collection(shadow_name)

    # Collect existing shadow IDs for dedup. For very large shadows
    # this is the bottleneck; pagination + a set keeps memory bounded.
    existing: set[str] = set()
    offset = 0
    while True:
        chunk = shadow.get(limit=batch_size, offset=offset, include=[])
        ids = chunk.get("ids") or []
        if not ids:
            break
        existing.update(ids)
        if len(ids) < batch_size:
            break
        offset += len(ids)

    rows_processed = 0
    rows_written = 0
    rows_skipped_dedup = 0
    embed_failures = 0

    offset = 0
    while True:
        chunk = src.get(
            limit=batch_size, offset=offset,
            include=["documents", "metadatas"],
        )
        ids = chunk.get("ids") or []
        if not ids:
            break
        docs = chunk.get("documents") or [None] * len(ids)
        metas = chunk.get("metadatas") or [{}] * len(ids)
        new_ids: list[str] = []
        new_docs: list[str] = []
        new_metas: list[dict] = []
        new_embs: list[list[float]] = []
        for i, doc_id in enumerate(ids):
            rows_processed += 1
            if doc_id in existing:
                rows_skipped_dedup += 1
                continue
            if not docs[i]:
                continue
            emb = _embed_target(docs[i], plan.target)
            if emb is None:
                embed_failures += 1
                continue
            new_ids.append(doc_id)
            new_docs.append(docs[i])
            new_metas.append(metas[i] if isinstance(metas[i], dict) else {})
            new_embs.append(emb)
        if new_ids:
            try:
                shadow.add(
                    ids=new_ids, embeddings=new_embs,
                    documents=new_docs, metadatas=new_metas,
                )
                rows_written += len(new_ids)
                state_mod.increment_backfill(len(new_ids))
            except Exception:
                logger.exception("backfill: shadow add failed")
                # Don't increment counter on failure
        offset += len(ids)
        if max_rows and rows_processed >= max_rows:
            break
        if len(ids) < batch_size:
            break

    return {
        "ok": True,
        "source_collection": source_collection,
        "shadow_collection": shadow_name,
        "rows_processed": rows_processed,
        "rows_written": rows_written,
        "rows_skipped_dedup": rows_skipped_dedup,
        "embed_failures": embed_failures,
    }
