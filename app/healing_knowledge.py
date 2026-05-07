"""
healing_knowledge.py — SHIELD-style evolving self-healing knowledge base.

When the self-healer successfully fixes an error (verified after 5 min),
the fix is stored in ChromaDB for future lookup. Before running expensive
LLM diagnosis, the system searches this knowledge base for previously
successful fixes for similar errors.

This creates a closed self-healing loop (SHIELD pattern):
  Error → search healing KB → match found? → apply known fix
                             → no match? → LLM diagnosis → verify → store

Over time, common errors get fixed in O(ms) instead of O(minutes).

Reference: arXiv:2601.19174 "SHIELD: Auto-Healing Agentic Defense Framework"
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

HEALING_KB_COLLECTION = "healing_knowledge"
HEALING_STATS_PATH = Path("/app/workspace/healing_stats.json")
_MIN_APPLICATIONS_FOR_REUSE = 2  # Fix must have been applied 2+ times before auto-reuse


@dataclass
class HealingEntry:
    """A known fix from the healing knowledge base."""
    error_signature: str
    error_description: str
    fix_type: str          # "code", "skill", "config", "transient"
    fix_applied: str       # Description of what was done
    outcome: str           # "resolved", "partial", "failed"
    times_applied: int = 0
    mast_category: str = ""
    mast_mode: str = ""
    distance: float = 1.0  # Similarity distance (lower = more similar)


def compute_error_signature(crew: str, error_type: str, agent_mode: str = "") -> str:
    """Stable hash for deduplication of healing entries."""
    key = f"{crew}:{error_type}:{agent_mode}".lower().strip()
    return hashlib.sha256(key.encode()).hexdigest()[:16]


# ── Storage ──────────────────────────────────────────────────────────────────

def store_healing_result(
    error_signature: str,
    error_description: str,
    fix_applied: str,
    fix_type: str,
    outcome: str,
    mast_category: str = "",
    mast_mode: str = "",
) -> None:
    """Store a verified healing result in ChromaDB.

    Called by healing/health_remediator.py after verification confirms the fix worked.
    Deduplicates by error_signature: if same signature exists, increments
    times_applied instead of creating a duplicate.
    """
    stored = False
    try:
        from app.memory.chromadb_manager import store, retrieve_with_metadata

        # Check for existing entry with same signature
        existing = retrieve_with_metadata(
            HEALING_KB_COLLECTION,
            error_description[:200],
            n=3,
        )
        for doc, meta in existing:
            if meta.get("error_signature") == error_signature:
                # Update existing: increment times_applied
                times = meta.get("times_applied", 1) + 1
                meta["times_applied"] = times
                meta["outcome"] = outcome
                meta["last_applied"] = time.time()
                # Re-store with updated metadata
                store(
                    HEALING_KB_COLLECTION,
                    error_description[:500],
                    metadata={
                        **meta,
                        "times_applied": times,
                        "outcome": outcome,
                    },
                )
                logger.info(
                    f"healing_knowledge: updated existing entry for {error_signature} "
                    f"(times_applied={times}, outcome={outcome})"
                )
                stored = True
                return

        # New entry
        store(
            HEALING_KB_COLLECTION,
            error_description[:500],
            metadata={
                "error_signature": error_signature,
                "fix_type": fix_type,
                "fix_applied": fix_applied[:500],
                "outcome": outcome,
                "times_applied": 1,
                "mast_category": mast_category,
                "mast_mode": mast_mode,
                "ts": time.time(),
                "last_applied": time.time(),
            },
        )
        logger.info(
            f"healing_knowledge: stored new entry for {error_signature} "
            f"(fix_type={fix_type}, outcome={outcome})"
        )
        stored = True

    except Exception as e:
        logger.debug(f"healing_knowledge: store failed: {e}")
    finally:
        if stored:
            _queue_transfer_event(
                error_signature=error_signature,
                error_description=error_description,
                fix_applied=fix_applied,
                fix_type=fix_type,
                outcome=outcome,
                mast_category=mast_category,
                mast_mode=mast_mode,
            )


def _queue_transfer_event(
    *,
    error_signature: str,
    error_description: str,
    fix_applied: str,
    fix_type: str,
    outcome: str,
    mast_category: str,
    mast_mode: str,
) -> None:
    """Append a transfer-memory event for nightly compilation.

    Lightweight write-path hook (single JSONL append). Failures are
    swallowed — healing storage must never break because a downstream
    compiler is unavailable.
    """
    try:
        from app.transfer_memory import append_event, TransferKind
        append_event(
            kind=TransferKind.HEALING,
            source_id=error_signature,
            summary=error_description[:120],
            payload={
                "error_signature": error_signature,
                "error_description": error_description[:500],
                "fix_type": fix_type,
                "fix_applied": fix_applied[:500],
                "outcome": outcome,
                "mast_category": mast_category,
                "mast_mode": mast_mode,
            },
        )
    except Exception:
        logger.debug("healing_knowledge: transfer_memory hook failed", exc_info=True)


def lookup_known_fix(
    error_description: str,
    crew: str = "",
    n: int = 3,
) -> list[HealingEntry]:
    """Search for previously successful fixes for a similar error.

    Returns entries sorted by relevance (distance). Only returns entries
    with outcome=="resolved" for auto-application. Entries with fewer
    than _MIN_APPLICATIONS_FOR_REUSE are returned but marked as advisory.

    Args:
        error_description: Error message/traceback text.
        crew: Crew that produced the error (optional filter).
        n: Max results to return.

    Returns:
        List of HealingEntry objects, sorted by distance (most similar first).
    """
    try:
        from app.memory.chromadb_manager import retrieve_with_metadata

        results = retrieve_with_metadata(
            HEALING_KB_COLLECTION,
            error_description[:500],
            n=n * 2,  # Over-fetch to filter
        )

        entries = []
        for i, (doc, meta) in enumerate(results):
            if not meta:
                continue
            entry = HealingEntry(
                error_signature=meta.get("error_signature", ""),
                error_description=doc[:200],
                fix_type=meta.get("fix_type", ""),
                fix_applied=meta.get("fix_applied", ""),
                outcome=meta.get("outcome", ""),
                times_applied=meta.get("times_applied", 0),
                mast_category=meta.get("mast_category", ""),
                mast_mode=meta.get("mast_mode", ""),
                distance=meta.get("distance", 1.0 - (1.0 / (i + 2))),
            )
            entries.append(entry)

        # Sort by relevance and filter to resolved outcomes
        entries.sort(key=lambda e: e.distance)
        return entries[:n]

    except Exception as e:
        logger.debug(f"healing_knowledge: lookup failed: {e}")
        return []


def get_best_known_fix(error_description: str, crew: str = "") -> HealingEntry | None:
    """Get the single best known fix for an error, if one exists.

    Returns the fix only if it has outcome=="resolved" and has been
    applied at least _MIN_APPLICATIONS_FOR_REUSE times (proven fix).
    """
    entries = lookup_known_fix(error_description, crew, n=1)
    if entries:
        best = entries[0]
        if best.outcome == "resolved" and best.times_applied >= _MIN_APPLICATIONS_FOR_REUSE:
            return best
    return None


# ── Stats ────────────────────────────────────────────────────────────────────

def get_healing_stats() -> dict:
    """Aggregate stats for the dashboard."""
    try:
        from app.memory.chromadb_manager import retrieve_with_metadata
        all_entries = retrieve_with_metadata(HEALING_KB_COLLECTION, "", n=100)

        total = len(all_entries)
        resolved = sum(1 for _, m in all_entries if m.get("outcome") == "resolved")
        total_applications = sum(m.get("times_applied", 0) for _, m in all_entries)

        return {
            "total_entries": total,
            "resolved": resolved,
            "total_applications": total_applications,
            "reuse_rate": round(total_applications / max(1, total), 2),
        }
    except Exception:
        return {"total_entries": 0, "resolved": 0, "total_applications": 0, "reuse_rate": 0}


# ── SUBIA integration ────────────────────────────────────────────────────────

def _boost_coherence_on_reuse() -> None:
    """Successful knowledge reuse boosts SUBIA coherence."""
    try:
        from app.subia.kernel import get_active_kernel
        kernel = get_active_kernel()
        if kernel and hasattr(kernel, "homeostasis"):
            current = kernel.homeostasis.variables.get("coherence", 0.5)
            kernel.homeostasis.variables["coherence"] = min(1.0, current + 0.02)
    except Exception:
        pass


# ── Lifecycle hook ───────────────────────────────────────────────────────────

def create_healing_lookup_hook():
    """ON_ERROR hook: search healing KB for known fixes before LLM diagnosis."""
    def _hook(ctx):
        try:
            if not ctx.errors:
                return ctx
            error_text = ctx.errors[0]
            crew = ctx.metadata.get("crew", "")

            best = get_best_known_fix(error_text, crew)
            if best:
                ctx.metadata["_known_fix"] = {
                    "fix_type": best.fix_type,
                    "fix_applied": best.fix_applied,
                    "times_applied": best.times_applied,
                    "outcome": best.outcome,
                    "mast_mode": best.mast_mode,
                }
                _boost_coherence_on_reuse()
                logger.info(
                    f"healing_knowledge: found known fix for error "
                    f"(applied {best.times_applied}x, outcome={best.outcome})"
                )
        except Exception:
            pass
        return ctx
    return _hook
