"""
retrieval/config.py — Configuration for the shared retrieval orchestrator.

All values are env-var-overridable following the project convention.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


# ── Cross-encoder re-ranking ────────────────────────────────────────────────
# Lightweight model (~60M params, CPU, ~10ms per query-document pair).
# Loaded lazily via sentence-transformers CrossEncoder.
RERANKER_MODEL: str = os.environ.get(
    "RETRIEVAL_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# Backend dispatch — "local" runs the cross-encoder above on CPU (fast,
# free, offline); "openrouter" routes through OpenRouter's /rerank endpoint
# (Cohere / Fireworks rerankers — higher quality, ~200-400ms latency,
# ~$1/1k requests). On any HTTP / network failure the OpenRouter path
# falls through to local automatically. Default ``local`` keeps the
# system free + offline-capable; flip to ``openrouter`` only after
# observing the rerank-call telemetry justifies the cost surface.
RERANKER_BACKEND: str = os.environ.get("RETRIEVAL_RERANKER_BACKEND", "local").lower()

# OpenRouter reranker model id. Cohere multilingual v3 is a good default
# for mixed-language corpora; switch via env without a code change.
RERANKER_MODEL_OPENROUTER: str = os.environ.get(
    "RETRIEVAL_RERANKER_MODEL_OPENROUTER", "cohere/rerank-multilingual-v3.0"
)

# First stage retrieves this many candidates via vector similarity …
RERANK_TOP_K_INPUT: int = int(os.environ.get("RETRIEVAL_RERANK_INPUT", "20"))
# … then the cross-encoder re-ranks and returns this many.
RERANK_TOP_K_OUTPUT: int = int(os.environ.get("RETRIEVAL_RERANK_OUTPUT", "5"))

# ── Temporal freshness weighting ────────────────────────────────────────────
TEMPORAL_HALF_LIFE_HOURS: float = float(
    os.environ.get("RETRIEVAL_TEMPORAL_HALF_LIFE", "168.0")  # 7 days
)
TEMPORAL_WEIGHT: float = float(
    os.environ.get("RETRIEVAL_TEMPORAL_WEIGHT", "0.15")
)

# ── Query decomposition ────────────────────────────────────────────────────
DECOMPOSITION_ENABLED: bool = os.environ.get(
    "RETRIEVAL_DECOMPOSITION", "1"
) == "1"
DECOMPOSITION_MAX_SUBQUERIES: int = int(
    os.environ.get("RETRIEVAL_MAX_SUBQUERIES", "4")
)
# Minimum query length (chars) before decomposition kicks in.
DECOMPOSITION_MIN_LENGTH: int = int(
    os.environ.get("RETRIEVAL_DECOMP_MIN_LEN", "100")
)

# ── Parallel retrieval ──────────────────────────────────────────────────────
MAX_PARALLEL_COLLECTIONS: int = int(
    os.environ.get("RETRIEVAL_MAX_PARALLEL", "6")
)
RETRIEVE_TIMEOUT_S: float = float(
    os.environ.get("RETRIEVAL_TIMEOUT", "5.0")
)


@dataclass
class RetrievalConfig:
    """Runtime-overridable config passed to RetrievalOrchestrator."""

    rerank_enabled: bool = True
    rerank_top_k_input: int = RERANK_TOP_K_INPUT
    rerank_top_k_output: int = RERANK_TOP_K_OUTPUT

    decomposition_enabled: bool = DECOMPOSITION_ENABLED
    decomposition_min_length: int = DECOMPOSITION_MIN_LENGTH
    max_subqueries: int = DECOMPOSITION_MAX_SUBQUERIES

    temporal_enabled: bool = False
    temporal_field: str = "ingested_at"
    temporal_half_life_hours: float = TEMPORAL_HALF_LIFE_HOURS
    temporal_weight: float = TEMPORAL_WEIGHT

    max_parallel: int = MAX_PARALLEL_COLLECTIONS
    timeout_s: float = RETRIEVE_TIMEOUT_S
