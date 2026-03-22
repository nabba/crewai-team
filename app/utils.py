"""
utils.py — Shared utility functions used across the codebase.

Centralizes common patterns to eliminate duplication:
  - safe_json_parse: Robust JSON parsing for LLM output
  - now_iso: UTC timestamp string (replaces 20+ inline datetime calls)
  - load_json_file / save_json_file: Journal-style file I/O with error handling
  - truncate: Safe string truncation (replaces 80+ inline [:N] slices)
"""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MAX_JSON_SIZE = 100_000  # bytes


# ── Timestamp ────────────────────────────────────────────────────────────────

def now_iso() -> str:
    """Return current UTC time as ISO 8601 string. Replaces 20+ inline calls."""
    return datetime.now(timezone.utc).isoformat()


# ── JSON File I/O ────────────────────────────────────────────────────────────

def load_json_file(path: Path, default=None):
    """Load a JSON file, returning default on any error.

    Replaces the identical load pattern in self_heal.py, benchmarks.py,
    auditor.py, variant_archive.py, proposals.py, etc.
    """
    if default is None:
        default = []
    try:
        if path.exists():
            return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        pass
    return default


def save_json_file(path: Path, data, max_entries: int = 0) -> bool:
    """Save data as JSON, optionally capping list entries.

    Returns True on success, False on failure.
    Replaces the identical save pattern in self_heal.py, benchmarks.py, etc.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if max_entries > 0 and isinstance(data, list):
            data = data[-max_entries:]
        path.write_text(json.dumps(data, indent=2))
        return True
    except OSError:
        logger.debug(f"Failed to write {path}", exc_info=True)
        return False


# ── String helpers ───────────────────────────────────────────────────────────

def truncate(text: str, max_len: int = 200) -> str:
    """Truncate a string to max_len chars. Replaces 80+ inline [:N] slices."""
    if not text:
        return ""
    return text[:max_len]


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
