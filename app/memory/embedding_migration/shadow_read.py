"""Shadow-read divergence telemetry — sampled NDCG@10.

PROGRAM §40 (2026-05-10) — Q3 Item 12.

Once the shadow collection is sufficiently backfilled, every Nth live
query also runs against the shadow with the target model's embedding.
We compare the top-10 lists with NDCG@10:

    NDCG@10 = DCG_observed / DCG_ideal

where ``ideal`` is the DCG of the source's top-10. NDCG@10 == 1.0 means
the shadow returned the exact same ranking; lower means divergence.

Operators read the rolling NDCG@10 average from the React migration
card. Cutover unblocks when:

    average NDCG@10 ≥ plan.cutover_threshold_ndcg
    AND
    sample count ≥ plan.cutover_min_shadow_queries

Goodhart safeguard: this metric is for CUTOVER GATING ONLY. The system
never optimises against it (e.g. by retraining the source model to
match shadow). The metric serves as a safety check on the operator's
proposed change, not a target the system pursues.

Sampling: ``EMBED_MIGRATION_SHADOW_SAMPLE_RATE`` env (0.0-1.0;
default 0.05). Set to 1.0 in dry-run for full coverage.
"""
from __future__ import annotations

import logging
import math
import os
import random
import threading
from collections import deque
from dataclasses import dataclass, field

from app.memory.embedding_migration import plan as plan_mod
from app.memory.embedding_migration import state as state_mod
from app.memory.embedding_migration.dual_write import (
    _embed_target, shadow_collection_name,
)

logger = logging.getLogger(__name__)


_DEFAULT_SAMPLE_RATE = 0.05
_ROLLING_WINDOW = 200   # rolling NDCG@10 average window


def _sample_rate() -> float:
    raw = os.getenv(
        "EMBED_MIGRATION_SHADOW_SAMPLE_RATE", str(_DEFAULT_SAMPLE_RATE),
    ).strip()
    try:
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        return _DEFAULT_SAMPLE_RATE


@dataclass
class _Window:
    """Rolling window of recent NDCG@10 measurements."""
    samples: deque = field(default_factory=lambda: deque(maxlen=_ROLLING_WINDOW))
    lock: threading.Lock = field(default_factory=threading.Lock)


_window = _Window()


def _record(ndcg: float) -> tuple[float, int]:
    """Add to rolling window, return (mean, n)."""
    with _window.lock:
        _window.samples.append(float(ndcg))
        n = len(_window.samples)
        mean = sum(_window.samples) / n if n > 0 else 0.0
    return mean, n


def get_window_summary() -> dict:
    """Read-only snapshot of the current window."""
    with _window.lock:
        n = len(_window.samples)
        mean = sum(_window.samples) / n if n > 0 else 0.0
        if n > 1:
            variance = sum((x - mean) ** 2 for x in _window.samples) / (n - 1)
            std = math.sqrt(variance)
        else:
            std = 0.0
        return {
            "mean_ndcg_at_10": round(mean, 6),
            "stdev_ndcg_at_10": round(std, 6),
            "samples": n,
            "min": min(_window.samples) if _window.samples else 0.0,
            "max": max(_window.samples) if _window.samples else 0.0,
        }


def _ndcg_at_10(ideal_ids: list[str], observed_ids: list[str]) -> float:
    """Standard binary-relevance NDCG@10. Each id in ``ideal_ids``
    counts as relevance=1; everything else is 0.

    ``observed_ids`` is the shadow ranking. ``ideal_ids`` is the
    source ranking. NDCG@10 == 1.0 means observed is permutation
    invariant w.r.t. ideal (top-10 is a set match)."""
    if not ideal_ids:
        return 1.0 if not observed_ids else 0.0
    relevance = set(ideal_ids[:10])
    dcg = 0.0
    for rank, doc_id in enumerate(observed_ids[:10], start=1):
        if doc_id in relevance:
            # Binary relevance at position rank: gain = 1
            dcg += 1.0 / math.log2(rank + 1)
    ideal_dcg = sum(1.0 / math.log2(r + 1) for r in range(1, len(relevance) + 1))
    if ideal_dcg <= 0:
        return 1.0 if dcg <= 0 else 0.0
    return dcg / ideal_dcg


# ── Public hook (called from chromadb_manager.retrieve) ──────────────────


def maybe_shadow_read(
    source_collection: str, query_text: str, observed_ids: list[str],
    n_results: int,
) -> None:
    """Best-effort shadow query for divergence telemetry.

    ``observed_ids`` is the source's top-N from the live query (we
    don't repeat the source query — we trust what the manager already
    computed)."""
    try:
        if not state_mod.shadow_read_enabled():
            return
        if random.random() >= _sample_rate():
            return
        plan = plan_mod.load_plan()
        if plan is None or not plan.plan_id:
            return
        if not query_text:
            return

        target_emb = _embed_target(query_text, plan.target)
        if target_emb is None:
            return

        from app.memory import chromadb_manager
        client = chromadb_manager.get_client()
        shadow_name = shadow_collection_name(source_collection, plan.plan_id)
        try:
            shadow = client.get_collection(shadow_name)
        except Exception:
            return

        try:
            res = shadow.query(
                query_embeddings=[target_emb],
                n_results=max(1, min(n_results, 10)),
            )
        except Exception:
            return

        # ChromaDB returns nested lists ([batch][rank]); we only have one batch.
        shadow_ids = (res.get("ids") or [[]])[0]
        ndcg = _ndcg_at_10(observed_ids, list(shadow_ids))
        mean, n = _record(ndcg)
        state_mod.record_shadow_query(ndcg_at_10=mean, window_size=n)
    except Exception:
        logger.debug(
            "embedding_migration.shadow_read: hook failed", exc_info=True,
        )
