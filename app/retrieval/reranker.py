"""
retrieval/reranker.py — Re-ranking for two-stage retrieval.

Stage 1 (vector similarity) is fast but imprecise — it matches embeddings.
Stage 2 (cross-encoder / API rerank) is slower but far more accurate — it
reads the actual query-document pair and outputs a relevance score.

Two backends are available, dispatched via ``cfg.RERANKER_BACKEND``:

* ``local`` (default) — sentence-transformers cross-encoder, ~60M params,
  CPU, ~10ms per pair, free + offline.
* ``openrouter`` — OpenRouter ``/rerank`` endpoint (Cohere / Fireworks
  rerankers), higher quality, ~200-400ms latency + ~$1/1k requests. Falls
  through to local on any HTTP / network failure.

Graceful degradation: if a backend is unavailable, the reranker returns
input unchanged (capped to ``top_k``) with a logged warning. No crash,
no silent data loss.

IMMUTABLE — infrastructure-level module.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from app.retrieval import config as cfg

logger = logging.getLogger(__name__)

# ── Lazy singleton ──────────────────────────────────────────────────────────

_lock = threading.Lock()
_model: Any | None = None
_model_failed: bool = False


def _get_model():
    """Lazy-load the cross-encoder model (thread-safe singleton)."""
    global _model, _model_failed
    if _model is not None:
        return _model
    if _model_failed:
        return None

    with _lock:
        # Double-check after acquiring lock.
        if _model is not None:
            return _model
        if _model_failed:
            return None
        try:
            from sentence_transformers import CrossEncoder

            _model = CrossEncoder(cfg.RERANKER_MODEL)
            logger.info(
                "retrieval.reranker: loaded cross-encoder '%s'", cfg.RERANKER_MODEL
            )
            return _model
        except Exception as exc:
            _model_failed = True
            logger.warning(
                "retrieval.reranker: failed to load cross-encoder '%s' — "
                "re-ranking disabled (will pass through): %s",
                cfg.RERANKER_MODEL,
                exc,
            )
            return None


# ── OpenRouter rerank backend ───────────────────────────────────────────────


def _openrouter_rerank(
    query: str,
    documents: list[dict],
    top_k: int,
    text_key: str,
    model: str,
) -> list[dict] | None:
    """Call OpenRouter's ``/rerank`` endpoint.

    Returns the reranked documents (with ``rerank_score`` populated from
    the API's ``relevance_score`` field) on success, or ``None`` on any
    failure — caller falls back to the local cross-encoder.

    Telemetry: latency, document count, and HTTP status are emitted at
    INFO so an operator can compare backends without enabling debug logs.
    """
    try:
        import httpx
        from app.config import get_settings
        api_key = get_settings().openrouter_api_key.get_secret_value()
    except Exception as exc:
        logger.debug("retrieval.reranker: openrouter setup failed: %s", exc)
        return None
    if not api_key:
        return None

    valid_docs: list[dict] = []
    texts: list[str] = []
    for doc in documents:
        text = doc.get(text_key, "")
        if text:
            valid_docs.append(doc)
            texts.append(text)
    if not texts:
        return []

    started = time.monotonic()
    try:
        resp = httpx.post(
            "https://openrouter.ai/api/v1/rerank",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "query": query,
                "documents": texts,
                "top_n": top_k,
            },
            timeout=15.0,
        )
    except Exception as exc:
        logger.warning("retrieval.reranker: openrouter request failed: %s", exc)
        return None

    elapsed_ms = (time.monotonic() - started) * 1000.0
    if resp.status_code != 200:
        logger.warning(
            "retrieval.reranker: openrouter HTTP %d (%.0fms): %s",
            resp.status_code, elapsed_ms, resp.text[:200],
        )
        return None
    try:
        data = resp.json()
    except Exception as exc:
        logger.warning("retrieval.reranker: openrouter response parse failed: %s", exc)
        return None

    results = data.get("results") or []
    reranked: list[dict] = []
    for r in results:
        idx = r.get("index")
        score = r.get("relevance_score")
        if idx is None or not isinstance(idx, int) or not (0 <= idx < len(valid_docs)):
            continue
        doc = valid_docs[idx]
        if score is not None:
            try:
                doc["rerank_score"] = float(score)
            except (TypeError, ValueError):
                pass
        reranked.append(doc)

    logger.info(
        "retrieval.reranker: openrouter model=%s n_in=%d n_out=%d top_k=%d latency=%.0fms",
        model, len(texts), len(reranked), top_k, elapsed_ms,
    )
    return reranked[:top_k]


# ── Public API ──────────────────────────────────────────────────────────────


def rerank(
    query: str,
    documents: list[dict],
    top_k: int = cfg.RERANK_TOP_K_OUTPUT,
    text_key: str = "text",
) -> list[dict]:
    """Re-rank *documents* by cross-encoder relevance to *query*.

    Each document dict must contain a *text_key* field with the passage
    text.  Returns a new list (sorted descending by rerank_score) with
    a ``rerank_score`` field added to each dict.

    If the cross-encoder is unavailable, returns *documents* unchanged
    (graceful degradation).

    Parameters
    ----------
    query : str
        The user/agent query.
    documents : list[dict]
        First-stage candidates, each containing at least ``text_key``.
    top_k : int
        How many to return after re-ranking.
    text_key : str
        Key in each dict that holds the passage text.
    """
    if not documents:
        return []

    # Backend dispatch. ``openrouter`` returns None on any failure, in
    # which case we transparently fall through to the local cross-encoder
    # below — the caller never sees the failure beyond a logged warning.
    if cfg.RERANKER_BACKEND == "openrouter":
        result = _openrouter_rerank(
            query, documents, top_k, text_key, cfg.RERANKER_MODEL_OPENROUTER,
        )
        if result is not None:
            return result
        logger.info("retrieval.reranker: openrouter unavailable, falling back to local")

    model = _get_model()
    if model is None:
        # Graceful degradation — return as-is, capped to top_k.
        for doc in documents:
            doc.setdefault("rerank_score", doc.get("score", 0.0))
        return documents[:top_k]

    # Build (query, passage) pairs for the cross-encoder.
    pairs = []
    valid_docs = []
    for doc in documents:
        text = doc.get(text_key, "")
        if text:
            pairs.append((query, text))
            valid_docs.append(doc)

    if not pairs:
        return []

    try:
        scores = model.predict(pairs)
    except Exception as exc:
        logger.warning("retrieval.reranker: predict failed: %s", exc)
        for doc in documents:
            doc.setdefault("rerank_score", doc.get("score", 0.0))
        return documents[:top_k]

    # Attach scores and sort.
    for doc, score in zip(valid_docs, scores):
        doc["rerank_score"] = float(score)

    valid_docs.sort(key=lambda d: d["rerank_score"], reverse=True)
    return valid_docs[:top_k]
