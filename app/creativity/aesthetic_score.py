"""Aesthetic scoring shim — wraps the aesthetics ChromaDB pattern store
for brainstorm + reverie consumption.

PROGRAM §46.21 (Q11.5). The aesthetics store (``app.aesthetics``)
holds curated patterns labelled by ``quality_score``. This module
exposes a single ``score(text) -> Optional[float]`` that combines
nearest-pattern similarity with that pattern's curated quality
to produce a 0..1 score the brainstorm report can render.

The score is intentionally LOOSE:

  * No threshold here. The brainstorm report shows the score but
    does NOT auto-drop anything (operator decision Q11.5). Over-
    filtering kills creativity.
  * Returns None when the store is empty / unavailable so the
    caller can omit the column rather than render zeros that
    look like negative judgements.

Formula::

    score = mean(top_k_similarity) × mean(top_k_quality_score)

Both inputs are 0..1, so the score is bounded. ``mean`` over top-3
patterns rather than top-1 dampens single-pattern noise; the
weighting by quality_score means "looks like our worst curated
pattern" doesn't rate as high as "looks like our best."
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


_TOP_K = 3


def score(text: str, *, top_k: int = _TOP_K) -> Optional[float]:
    """Aesthetic score for ``text``. Returns 0..1 or None.

    None is the "no signal" sentinel — caller should render 'n/a'
    or omit the column. Distinguishes from a score of 0.0 which
    means "the patterns we have rate this poorly."

    Failure-isolated: any exception → None.
    """
    text = (text or "").strip()
    if not text:
        return None
    try:
        from app.aesthetics.vectorstore import get_store
    except Exception:
        return None
    try:
        store = get_store()
    except Exception:
        return None
    try:
        count = store._collection.count()
    except Exception:
        return None
    if count == 0:
        return None
    try:
        results = store.query(query_text=text, n_results=max(1, top_k))
    except Exception:
        logger.debug(
            "aesthetic_score: store.query raised", exc_info=True,
        )
        return None
    if not results:
        return None
    # AestheticStore.query returns ``score = 1.0 - distance`` (cosine
    # distance-to-similarity already done at the store layer). Handle
    # legacy ``similarity`` / ``distance`` shapes too for robustness.
    sims: list[float] = []
    qualities: list[float] = []
    for r in results[:top_k]:
        if "score" in r:
            sim = float(r.get("score") or 0.0)
        elif "similarity" in r:
            sim = float(r.get("similarity") or 0.0)
        elif "distance" in r:
            d = float(r.get("distance") or 0.0)
            sim = max(0.0, 1.0 - d / 2.0)
        else:
            continue
        meta = r.get("metadata") or {}
        q = meta.get("quality_score")
        try:
            q_val = float(q) if q is not None else 0.5
        except (TypeError, ValueError):
            q_val = 0.5
        # quality_score may be stored as 0..10 (curator convention)
        # or 0..1 (normalized). Cheap normalisation.
        if q_val > 1.0:
            q_val = q_val / 10.0
        q_val = max(0.0, min(1.0, q_val))
        sims.append(max(0.0, min(1.0, sim)))
        qualities.append(q_val)
    if not sims:
        return None
    mean_sim = sum(sims) / len(sims)
    mean_q = sum(qualities) / len(qualities)
    return round(mean_sim * mean_q, 3)


def score_many(texts: list[str], *, top_k: int = _TOP_K) -> list[Optional[float]]:
    """Vectorised over a list — convenience for the brainstorm
    report walking many responses. Same None semantics."""
    return [score(t, top_k=top_k) for t in texts]
