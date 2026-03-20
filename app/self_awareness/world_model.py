"""
world_model.py — Causal world model memory (L2 Self-Awareness).

Stores cause→effect beliefs and prediction-vs-reality comparisons.
Agents learn from environmental outcomes over time by building an
evolving model of how the system and world behave.

Uses existing scoped memory infrastructure — no new storage backend.

Memory scopes:
  scope_world_model   — causal beliefs (cause → effect patterns)
  scope_predictions   — pre-execution predictions and post-execution comparisons
"""

import logging
from app.memory.scoped_memory import store_scoped, retrieve_operational

logger = logging.getLogger(__name__)

SCOPE_CAUSAL = "scope_world_model"
SCOPE_PREDICTIONS = "scope_predictions"


# ── Causal belief storage ────────────────────────────────────────────────────

def store_causal_belief(
    cause: str,
    effect: str,
    confidence: str = "medium",
    source: str = "observed",
) -> None:
    """Store a cause→effect observation in the world model.

    Args:
        cause: What triggered the effect (e.g. "Ollama timeout after 10s")
        effect: What happened as a result (e.g. "Cascade to OpenRouter added 3s latency")
        confidence: How confident we are — "high", "medium", or "low"
        source: Where this knowledge came from — "observed", "inferred", "reported"
    """
    text = f"CAUSE: {cause} → EFFECT: {effect} [confidence={confidence}, source={source}]"
    store_scoped(
        SCOPE_CAUSAL, text,
        {"type": "causal", "confidence": confidence, "source": source},
        importance="high",
    )
    logger.debug(f"World model: stored causal belief ({confidence}): {cause[:60]} → {effect[:60]}")


def recall_relevant_beliefs(query: str, n: int = 3) -> list[str]:
    """Recall causal beliefs relevant to a query.

    Returns up to n beliefs, ranked by semantic similarity + recency.
    """
    try:
        return retrieve_operational(SCOPE_CAUSAL, query, n)
    except Exception:
        logger.debug("World model: failed to recall beliefs", exc_info=True)
        return []


# ── Prediction tracking ──────────────────────────────────────────────────────

def store_prediction(
    task_id: str,
    prediction: str,
    context: str = "",
) -> None:
    """Store a pre-execution prediction for later comparison.

    Args:
        task_id: Unique identifier for the task (e.g. crew_name + timestamp)
        prediction: What we expect to happen
        context: Additional context about the prediction
    """
    text = f"PREDICTION [{task_id}]: {prediction}"
    if context:
        text += f". Context: {context}"
    store_scoped(
        SCOPE_PREDICTIONS, text,
        {"type": "prediction", "task_id": task_id},
    )


def store_prediction_result(
    task_id: str,
    prediction: str,
    actual: str,
    lesson: str,
) -> None:
    """Compare prediction to actual outcome, store the lesson.

    Called after task execution to update the world model with
    what we learned from the prediction-reality gap.

    Args:
        task_id: Same task_id used in store_prediction
        prediction: What we expected
        actual: What actually happened
        lesson: What we learned from the gap
    """
    text = (
        f"PREDICTION REVIEW [{task_id}]: "
        f"Expected: {prediction}. Actual: {actual}. Lesson: {lesson}"
    )
    # High importance if prediction was wrong — we learned something
    importance = "high" if prediction.lower() != actual.lower() else "normal"
    store_scoped(
        SCOPE_PREDICTIONS, text,
        {"type": "prediction_result", "task_id": task_id},
        importance=importance,
    )
    logger.debug(f"World model: prediction result [{task_id}] — importance={importance}")


def recall_relevant_predictions(query: str, n: int = 3) -> list[str]:
    """Recall past prediction outcomes relevant to the current task.

    Useful for avoiding repeated mistakes or confirming known patterns.
    """
    try:
        return retrieve_operational(SCOPE_PREDICTIONS, query, n)
    except Exception:
        logger.debug("World model: failed to recall predictions", exc_info=True)
        return []
