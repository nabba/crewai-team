"""Pre-call cost estimation for budget enforcement.

Estimates the cost of an LLM call BEFORE it happens, so budget enforcement
can reject calls that would exceed limits.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Rough token estimates per prompt length (chars → tokens ≈ chars/4)
_CHARS_PER_TOKEN = 4


def estimate_tokens(prompt: str, max_output_tokens: int = 1024) -> tuple[int, int]:
    """Estimate input and output tokens.

    Returns (estimated_input_tokens, estimated_output_tokens).
    """
    input_tokens = max(len(prompt) // _CHARS_PER_TOKEN, 10)
    return input_tokens, max_output_tokens


def estimate_cost(
    model: str,
    prompt: str = "",
    input_tokens: int = 0,
    output_tokens: int = 1024,
) -> float:
    """Estimate cost in USD for an LLM call.

    Uses the model's cost_input_per_m and cost_output_per_m from the catalog.
    Returns estimated cost in USD.
    """
    if not input_tokens and prompt:
        input_tokens = max(len(prompt) // _CHARS_PER_TOKEN, 10)

    try:
        from app.llm_catalog import get_model
        info = get_model(model) if model else None
        if not info:
            # Unknown model — use a conservative estimate ($2/M input, $6/M output)
            return (input_tokens * 2.0 + output_tokens * 6.0) / 1_000_000

        cost_in = info.get("cost_input_per_m", 0) or 0
        cost_out = info.get("cost_output_per_m", 0) or 0

        # Local models are free
        if info.get("tier") == "local":
            return 0.0

        return (input_tokens * cost_in + output_tokens * cost_out) / 1_000_000
    except Exception:
        # Conservative fallback
        return (input_tokens * 2.0 + output_tokens * 6.0) / 1_000_000
