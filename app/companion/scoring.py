"""Scoring helpers — novelty, quality, transferability.

Novelty is embedding-based via ChromaDB (1 - max_similarity vs prior
workspace ideas). Quality and transferability are LLM-as-judge with a
fixed rubric, run on the cheap tier of the model cascade.

All three return [0.0, 1.0]. Failures degrade gracefully so a broken
ChromaDB / LLM gateway never crashes the cycle:
  - novelty unavailable      → 1.0 (assume novel; gate elsewhere)
  - quality unavailable      → 0.5 (neutral)
  - transferability unavailable → 0.5 (neutral)

Cost: at most 2 cheap LLM calls per cycle (quality + transferability) —
trivial against the $1/day workspace budget. Phase 12 wires per-call cost
attribution into the budget ledger; for now the cost is implicit.
"""

from __future__ import annotations

import logging
import re

from app.companion import idea_store as _idea_store

logger = logging.getLogger(__name__)

MIN_TEXT_FOR_SCORING = 30


def compute_novelty(text: str, workspace_id: str, *,
                    top_k: int = 5) -> float:
    """1 - max similarity vs existing workspace ideas. 1.0 when no history.

    Uses ChromaDB cosine distance (∈ [0, 2]). Maps to similarity ∈ [0, 1]
    via ``similarity = 1 - distance/2``. Falls back to 1.0 if ChromaDB
    is unavailable — workspace's first idea is also 1.0 by definition.
    """
    if not text or not text.strip():
        return 0.0
    try:
        results = _idea_store.search_similar(workspace_id, text, top_k=top_k)
    except Exception as exc:
        logger.debug("companion.scoring: novelty search failed: %s", exc)
        return 1.0
    if not results:
        return 1.0
    distances = [r["distance"] for r in results
                 if r.get("distance") is not None]
    if not distances:
        return 1.0
    max_sim = 1.0 - min(distances) / 2.0
    return _clamp01(1.0 - max_sim)


def compute_quality(text: str) -> float:
    """LLM-as-judge quality rubric. Returns [0, 1]. Neutral 0.5 on failure."""
    if not text or not text.strip() or len(text) < MIN_TEXT_FOR_SCORING:
        return 0.0
    try:
        return _llm_score(_QUALITY_RUBRIC.format(text=text[:2000]))
    except Exception as exc:
        logger.debug("companion.scoring: quality LLM failed: %s", exc)
        return 0.5


def compute_transferability(text: str) -> float:
    """LLM classifier: 1.0 = abstract/structural, 0.0 = workspace-specific.

    The result feeds Phase 13's cross-workspace transfer gate (only ideas
    at or above ``transferability_threshold`` are considered for sanitised
    propagation to other workspaces).
    """
    if not text or not text.strip() or len(text) < MIN_TEXT_FOR_SCORING:
        return 0.0
    try:
        return _llm_score(_TRANSFERABILITY_RUBRIC.format(text=text[:2000]))
    except Exception as exc:
        logger.debug("companion.scoring: transferability LLM failed: %s", exc)
        return 0.5


# ── Rubrics ────────────────────────────────────────────────────────────────

_QUALITY_RUBRIC = """\
Score this idea on the rubric below. Output ONLY a single number 0–10.

Rubric:
  0–2  vague, generic, or trivially obvious
  3–4  reasonable but not insightful
  5–6  solid; clear value but no surprise
  7–8  insightful; non-obvious connection
  9–10 deeply illuminating, unusually clear

IDEA:
{text}

Score (0–10):"""


_TRANSFERABILITY_RUBRIC = """\
Score how abstract / structural this idea is. Output ONLY a single \
number 0–10.

  0–2  fully workspace-specific (named entities, domain lingo dominate)
  3–4  mostly specific
  5–6  mixed
  7–8  largely abstract; principles + structure dominate
  9–10 pure structural pattern; transfers across domains

IDEA:
{text}

Score (0–10):"""


# ── Internal ────────────────────────────────────────────────────────────────

def _llm_score(prompt: str) -> float:
    """Call the cheap-tier LLM, parse the first number, return [0, 1]."""
    raw = _invoke_judge(prompt)
    m = re.search(r"\b(\d+(?:\.\d+)?)", str(raw))
    if not m:
        return 0.5
    val = float(m.group(1))
    return _clamp01(val / 10.0)


def _invoke_judge(prompt: str) -> str:
    """Indirection over the LLM factory call, for testability."""
    from app.llm_factory import create_specialist_llm
    llm = create_specialist_llm(max_tokens=20, role="critic")
    return str(llm.call(prompt))


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))
