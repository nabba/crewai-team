"""
utils.py — Shared utility functions.

safe_json_parse: Robust JSON parsing for LLM output with markdown fence
stripping, size limits, and structured error reporting.
"""

import json
import re
from typing import Any


_MAX_JSON_SIZE = 100_000  # bytes


def safe_json_parse(
    text: str,
    max_size: int = _MAX_JSON_SIZE,
) -> tuple[Any | None, str]:
    """Parse JSON from LLM output, stripping markdown fences and validating size.

    Returns:
        (parsed_value, "") on success
        (None, error_message) on failure
    """
    if not text or not isinstance(text, str):
        return None, "empty or non-string input"

    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    cleaned = re.sub(r'^```(?:json)?\s*', '', text.strip())
    cleaned = re.sub(r'\s*```$', '', cleaned)

    # Size check
    if len(cleaned.encode('utf-8', errors='replace')) > max_size:
        return None, f"JSON too large ({len(cleaned)} chars, max {max_size} bytes)"

    try:
        result = json.loads(cleaned)
        return result, ""
    except json.JSONDecodeError as exc:
        # Provide a useful error snippet
        pos = exc.pos or 0
        snippet = cleaned[max(0, pos - 20):pos + 20]
        return None, f"JSON parse error at pos {pos}: {exc.msg} near '{snippet}'"
