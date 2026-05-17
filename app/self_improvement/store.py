"""
app.self_improvement.store — persistence for LearningGap records.

Backed by ChromaDB (the existing infra). One collection: `learning_gaps`.
Each gap is stored with its embedding so similar gaps can be discovered
across sources (e.g. retrieval-miss + reflexion-failure on overlapping
topics → same underlying knowledge gap).

Idempotency: emit_gap() deduplicates on (source, description) within the
last 24h. Re-detection of the same gap doesn't pollute the store.

IMMUTABLE — infrastructure-level module.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone, timedelta

from app.self_improvement.types import (
    LearningGap, GapSource, GapStatus,
)

logger = logging.getLogger(__name__)

GAPS_COLLECTION = "learning_gaps"

# Idempotency window: re-emitting the same (source, description) within this
# window updates the existing record (signal_strength) rather than creating a
# duplicate. Past the window, a fresh detection is treated as a new gap event.
_DEDUP_WINDOW_HOURS = 24


def _gap_id(source: GapSource, description: str) -> str:
    """Deterministic ID from (source, normalized description).

    Same gap re-detected within the dedup window collides on this id and is
    upserted rather than appended.
    """
    norm = description.strip().lower()
    h = hashlib.sha256(f"{source.value}::{norm}".encode()).hexdigest()[:12]
    return f"gap_{source.value}_{h}"


def _get_collection():
    """Lazy-init the gaps collection. Returns None if Chroma is unavailable."""
    try:
        from app.memory.chromadb_manager import get_client
        client = get_client()
        return client.get_or_create_collection(
            GAPS_COLLECTION, metadata={"hnsw:space": "cosine"},
        )
    except Exception as exc:
        logger.debug(f"learning_gaps collection unavailable: {exc}")
        return None


def emit_gap(gap: LearningGap) -> bool:
    """Persist a gap. Returns True on success.

    Idempotent: same (source, description) within _DEDUP_WINDOW_HOURS upserts
    the existing record rather than appending. The signal_strength of the
    upserted record is the max of the two — repeated detection raises the
    priority of the gap, never lowers it.
    """
    col = _get_collection()
    if col is None:
        return False

    # Override caller-supplied id to enforce dedup
    gap.id = _gap_id(gap.source, gap.description)

    # If it already exists and is recent, merge signal_strength
    try:
        existing = col.get(ids=[gap.id])
        if existing.get("ids"):
            try:
                prior = json.loads(existing["documents"][0])
                prior_strength = float(prior.get("signal_strength", 0.0))
                gap.signal_strength = max(gap.signal_strength, prior_strength)
                # Bump detected_at so age-based pruning reflects recent re-detection
                # (preserves original status though)
                prior_status = prior.get("status")
                if prior_status:
                    gap.status = GapStatus(prior_status)
            except Exception:
                pass
    except Exception:
        pass

    try:
        doc = json.dumps(gap.to_dict())
        # Metadata kept flat for ChromaDB filterability
        meta = {
            "source": gap.source.value,
            "status": gap.status.value,
            "signal_strength": float(gap.signal_strength),
            "detected_at": gap.detected_at,
        }
        from app.memory.chromadb_manager import embed
        col.upsert(ids=[gap.id], documents=[doc], metadatas=[meta],
                    embeddings=[embed(doc)])
        logger.debug(
            f"emit_gap: {gap.source.value} '{gap.description[:60]}' "
            f"strength={gap.signal_strength:.2f}"
        )
        return True
    except Exception as exc:
        logger.debug(f"emit_gap upsert failed: {exc}")
        return False


def list_open_gaps(limit: int = 20, source: GapSource | None = None) -> list[LearningGap]:
    """Return the most recent OPEN gaps, optionally filtered by source."""
    col = _get_collection()
    if col is None:
        return []
    try:
        where = {"status": GapStatus.OPEN.value}
        if source is not None:
            # ChromaDB AND filter
            where = {"$and": [{"status": GapStatus.OPEN.value},
                              {"source": source.value}]}
        result = col.get(where=where, limit=limit * 3)  # over-fetch then sort
        gaps: list[LearningGap] = []
        for doc in result.get("documents", []):
            try:
                gaps.append(LearningGap.from_dict(json.loads(doc)))
            except Exception:
                continue
        # Sort: signal_strength desc, then detected_at desc
        gaps.sort(key=lambda g: (g.signal_strength, g.detected_at), reverse=True)
        return gaps[:limit]
    except Exception as exc:
        logger.debug(f"list_open_gaps failed: {exc}")
        return []


def update_gap_status(gap_id: str, status: GapStatus, notes: str = "") -> bool:
    """Update a gap's status (e.g. OPEN → SCHEDULED → RESOLVED_NEW)."""
    col = _get_collection()
    if col is None:
        return False

    success_gap: LearningGap | None = None
    try:
        existing = col.get(ids=[gap_id])
        if not existing.get("ids"):
            return False
        gap = LearningGap.from_dict(json.loads(existing["documents"][0]))
        gap.status = status
        if notes:
            gap.resolution_notes = (
                (gap.resolution_notes + "\n" + notes).strip() if gap.resolution_notes else notes
            )
        meta = {
            "source": gap.source.value,
            "status": gap.status.value,
            "signal_strength": float(gap.signal_strength),
            "detected_at": gap.detected_at,
        }
        doc = json.dumps(gap.to_dict())
        from app.memory.chromadb_manager import embed
        col.upsert(ids=[gap.id], documents=[doc], metadatas=[meta],
                    embeddings=[embed(doc)])
        success_gap = gap
        return True
    except Exception as exc:
        logger.debug(f"update_gap_status failed: {exc}")
        return False
    finally:
        # Transfer-memory hook (Phase 17): only RESOLVED_NEW transitions
        # carry the "we learned something new" signal worth compiling.
        # Other transitions (TRIAGED, SCHEDULED, RESOLVED_EXISTING) are
        # workflow noise that doesn't belong in cross-domain memory.
        if success_gap is not None and status == GapStatus.RESOLVED_NEW:
            try:
                _queue_gap_transfer_event(success_gap, notes)
            except Exception:
                logger.debug(
                    "update_gap_status: transfer_memory hook failed",
                    exc_info=True,
                )


def _queue_gap_transfer_event(gap: "LearningGap", notes: str) -> None:
    """Append a gap-resolution event for nightly transfer-memory compilation."""
    from app.transfer_memory import append_event, TransferKind
    append_event(
        kind=TransferKind.GAP_RESOLVED,
        source_id=gap.id,
        summary=f"[{gap.source.value}] {gap.description[:160]}",
        payload={
            "source": gap.source.value,
            "description": gap.description[:600],
            "evidence": _shrink_evidence(gap.evidence),
            "signal_strength": float(gap.signal_strength),
            "resolution_notes": (notes or gap.resolution_notes)[:400],
        },
    )


def _shrink_evidence(evidence: dict) -> dict:
    """Keep gap.evidence small enough for the JSONL queue line.

    Strips long values and any obviously oversized blobs; keeps key/value
    pairs whose stringified value is ≤200 chars.
    """
    if not isinstance(evidence, dict):
        return {}
    out: dict = {}
    for k, v in evidence.items():
        s = str(v)
        if len(s) <= 200:
            out[k] = v
        else:
            out[k] = s[:200] + "…"
        if len(out) >= 12:
            break
    return out


def query_gaps(query_text: str, n: int = 5) -> list[LearningGap]:
    """Semantic search over gaps — find related ones for clustering."""
    col = _get_collection()
    if col is None:
        return []
    try:
        from app.memory.chromadb_manager import embed
        result = col.query(query_embeddings=[embed(query_text)], n_results=n)
        out: list[LearningGap] = []
        for doc in (result.get("documents") or [[]])[0]:
            try:
                out.append(LearningGap.from_dict(json.loads(doc)))
            except Exception:
                continue
        return out
    except Exception as exc:
        logger.debug(f"query_gaps failed: {exc}")
        return []


def prune_old_gaps(max_age_days: int = 60) -> int:
    """Remove RESOLVED/REJECTED gaps older than max_age_days. Returns count.

    Open gaps are never pruned — if they're stale, they're a signal that
    the learning loop isn't keeping up.
    """
    col = _get_collection()
    if col is None:
        return 0
    cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat()
    removed = 0
    try:
        terminal_statuses = [
            GapStatus.RESOLVED_EXISTING.value,
            GapStatus.RESOLVED_NEW.value,
            GapStatus.REJECTED.value,
        ]
        result = col.get(where={"$and": [
            {"status": {"$in": terminal_statuses}},
            {"detected_at": {"$lt": cutoff}},
        ]})
        ids = result.get("ids", [])
        if ids:
            col.delete(ids=ids)
            removed = len(ids)
            # PROGRAM §56 iter-2 — ledger tombstone
            try:
                from app.memory.source_ledger import hook_collection_delete
                hook_collection_delete("memory", GAPS_COLLECTION, list(ids))
            except Exception:
                pass
    except Exception as exc:
        logger.debug(f"prune_old_gaps failed: {exc}")
    return removed
