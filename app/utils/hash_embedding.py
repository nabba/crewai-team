"""Deterministic hashing-trick embedding (Phase E #7, 2026-05-09).

Originally lived in two places:

  * ``app/healing/llm_output_drift.py`` (drift detector fallback)
  * ``app/companion/lessons_learned.py``  (rejection-cluster centroids)

Both used identical 256-d SHA-1 hashing, identical cosine helpers,
and tuned thresholds against the resulting cosine distribution.
Keeping two copies invited drift; this module is the single source.

Usage::

    from app.utils.hash_embedding import embed, cosine

    v = embed("any text")
    s = cosine(v, w)

The vector is L2-normalized so cosine == dot product.

Properties this design relies on:

  * Stable across processes — SHA-1 is deterministic, no PRNG.
  * Same text → same vector.
  * Identical text → cosine 1.0.
  * Completely different text → cosine ≈ 0.0 (orthogonal-ish).
  * Similar-but-not-identical text → cosine in [0.3, 0.8] range,
    much lower than a real LLM embedding would give. Callers
    should set thresholds with that in mind.

Not a substitute for a real embedding model — use this when:
  * The real embedding endpoint is unavailable.
  * The task is change-detection (same-vs-different), not semantic
    similarity.
  * Determinism matters more than fidelity.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Sequence

# Default dim. Bigger dims spread tokens over more buckets → fewer
# collisions → cleaner cosine for "similar but not identical" text.
DEFAULT_DIM = 256

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-]{2,}")


def _tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")]


def embed(text: str, dim: int = DEFAULT_DIM) -> list[float]:
    """Return an L2-normalized hashing-trick embedding of ``text``."""
    vec = [0.0] * dim
    for tok in _tokenize(text):
        h = hashlib.sha1(tok.encode("utf-8")).digest()
        for i, byte in enumerate(h):
            slot = (i * 13 + byte) % dim
            sign = 1.0 if byte % 2 == 0 else -1.0
            vec[slot] += sign
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity. Robust to empty / mismatched-length vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)
