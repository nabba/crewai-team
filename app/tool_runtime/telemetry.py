"""telemetry.py — capture cache hit/miss tokens per LLM call.

Phase 2 needs to validate the Phase 1c analytical model with real
cache behavior from Anthropic's API. Anthropic returns
``cache_creation_input_tokens`` and ``cache_read_input_tokens`` in
its response usage; litellm passes those through. CrewAI's existing
``TokenCalcHandler`` captures total input/output tokens, but not the
cache fields — so we add our own callback.

Usage
-----
The callback is wired automatically when LoadableAgent's executor
is instrumented (Phase 2 calls ``install_cache_telemetry`` on the
agent in its factory). Each LLM call writes a row to
``workspace/observability/loadable_agent_usage.jsonl`` with::

    {
      "ts": "2026-05-03T...Z",
      "agent_id": "introspector",
      "iteration": 3,
      "input_tokens": 234,
      "output_tokens": 89,
      "cache_creation_input_tokens": 1500,
      "cache_read_input_tokens": 8200,
      "model": "claude-3-5-sonnet-...",
    }

After 50+ calls, ``analyze_telemetry()`` summarizes hit/miss ratios
and compares against the Phase 1c model's predictions. If real
behavior diverges materially (>±15%), the analytical model is
recalibrated and the gate is re-run.

Failure semantics
-----------------
The callback never raises. If the response payload doesn't have
cache fields (older litellm version, non-Anthropic model), it logs
zeros and continues. Logging file errors are swallowed.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_TELEMETRY_PATH = Path("/app/workspace/observability/loadable_agent_usage.jsonl")
_WRITE_LOCK = threading.Lock()


def _ensure_dir() -> None:
    try:
        _TELEMETRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def record_call_usage(
    *,
    agent_id: str,
    iteration: int,
    response: Any,
    model: str | None = None,
) -> None:
    """Write one telemetry row from a litellm response.

    ``response`` is whatever ``get_llm_response`` returns from
    inside the executor; the wrapped litellm completion has a
    ``usage`` attribute we extract from.
    """
    try:
        usage = getattr(response, "usage", None) or {}
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        elif hasattr(usage, "__dict__"):
            usage = dict(usage.__dict__)
        elif not isinstance(usage, dict):
            usage = {}
    except Exception:
        usage = {}

    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "agent_id": agent_id,
        "iteration": iteration,
        "input_tokens": int(usage.get("input_tokens", 0) or 0),
        "output_tokens": int(usage.get("output_tokens", 0) or 0),
        # Anthropic-specific fields. May be None for non-Anthropic models.
        "cache_creation_input_tokens": int(
            usage.get("cache_creation_input_tokens", 0) or 0
        ),
        "cache_read_input_tokens": int(
            usage.get("cache_read_input_tokens", 0) or 0
        ),
        "model": model or "",
    }

    _ensure_dir()
    try:
        with _WRITE_LOCK:
            with _TELEMETRY_PATH.open("a") as f:
                f.write(json.dumps(row) + "\n")
    except Exception as exc:
        logger.debug("loadable_agent telemetry: write failed: %s", exc)


# ── Analysis ────────────────────────────────────────────────────────


def load_telemetry(*, agent_id: str | None = None, limit: int | None = None) -> list[dict]:
    """Read recorded telemetry rows. Filter by agent_id; return at
    most ``limit`` newest rows. Empty list if the file doesn't exist."""
    if not _TELEMETRY_PATH.exists():
        return []
    rows: list[dict] = []
    try:
        with _TELEMETRY_PATH.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if agent_id and row.get("agent_id") != agent_id:
                    continue
                rows.append(row)
    except Exception as exc:
        logger.debug("loadable_agent telemetry: read failed: %s", exc)
        return []
    if limit:
        rows = rows[-limit:]
    return rows


def analyze_telemetry(*, agent_id: str | None = None) -> dict[str, Any]:
    """Summarize cache hit/miss patterns from recorded calls.

    Returns:
        {
            "calls": int,
            "total_input": int,
            "total_output": int,
            "total_cache_creation": int,
            "total_cache_read": int,
            "cache_read_pct": float,   # of total input tokens
            "effective_input_tokens": float,  # under Anthropic's cache pricing
            "vs_uncached": dict,        # what it would have cost without cache
        }

    Empty / zero values when no telemetry exists.
    """
    rows = load_telemetry(agent_id=agent_id)
    if not rows:
        return {
            "calls": 0,
            "total_input": 0,
            "total_output": 0,
            "total_cache_creation": 0,
            "total_cache_read": 0,
            "cache_read_pct": 0.0,
            "effective_input_tokens": 0.0,
            "vs_uncached": {},
        }

    total_input = sum(r.get("input_tokens", 0) for r in rows)
    total_output = sum(r.get("output_tokens", 0) for r in rows)
    total_cache_creation = sum(
        r.get("cache_creation_input_tokens", 0) for r in rows
    )
    total_cache_read = sum(r.get("cache_read_input_tokens", 0) for r in rows)

    # Effective tokens under Anthropic's cache pricing
    # (1.0× fresh, 1.25× cache write, 0.10× cache read).
    fresh_input = total_input  # input_tokens is fresh-only in Anthropic's accounting
    effective = (
        1.00 * fresh_input
        + 1.25 * total_cache_creation
        + 0.10 * total_cache_read
    )
    uncached_equivalent = fresh_input + total_cache_creation + total_cache_read

    return {
        "calls": len(rows),
        "total_input": total_input,
        "total_output": total_output,
        "total_cache_creation": total_cache_creation,
        "total_cache_read": total_cache_read,
        "cache_read_pct": (
            total_cache_read / max(uncached_equivalent, 1)
        ),
        "effective_input_tokens": round(effective, 1),
        "vs_uncached": {
            "uncached_total_input": uncached_equivalent,
            "savings_ratio": (
                round(1 - effective / max(uncached_equivalent, 1), 3)
                if uncached_equivalent else 0.0
            ),
        },
    }
