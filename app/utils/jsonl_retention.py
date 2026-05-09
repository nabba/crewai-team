"""Bounded JSONL retention helper (Phase F #7, 2026-05-09).

Multiple modules under ``app/`` shipped append-only JSONL ledgers
without retention policy. Over years they grow unboundedly. Examples:

  * ``workspace/training/adapter_quality_history.jsonl``
  * ``workspace/healing/llm_drift_history.jsonl``
  * ``workspace/governance_proposals.jsonl``
  * ``workspace/proposed_experiments.jsonl``
  * ``workspace/training/retirement_proposals.jsonl``

This module exposes ``cap_jsonl(path, max_lines)`` — keep the last
``max_lines`` lines of a JSONL file by atomic rewrite. Designed to be
called inline by writers ("write a row, then enforce the cap") so we
don't need a separate retention monitor.

Atomic-rewrite semantics: read → tail-slice → write to ``.tmp`` →
``replace``. If the original is below the cap, no rewrite happens.

Best-effort and silent on failure — losing a few rows is preferable
to crashing a writer because the cap can't be enforced.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def cap_jsonl(path: Path | str, max_lines: int) -> int:
    """Truncate ``path`` to the last ``max_lines`` lines. Returns the
    number of lines dropped (0 when no rewrite needed).

    Idempotent — calling repeatedly is a no-op once the file is
    within bounds.
    """
    p = Path(path)
    if not p.exists() or max_lines <= 0:
        return 0
    try:
        with p.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        logger.debug("jsonl_retention: read failed for %s", p, exc_info=True)
        return 0
    if len(lines) <= max_lines:
        return 0
    keep = lines[-max_lines:]
    dropped = len(lines) - len(keep)
    try:
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text("".join(keep), encoding="utf-8")
        tmp.replace(p)
    except OSError:
        logger.debug("jsonl_retention: rewrite failed for %s", p, exc_info=True)
        return 0
    return dropped


def append_with_cap(path: Path | str, json_line: str, max_lines: int) -> None:
    """Append ``json_line`` (no trailing newline) to ``path`` and
    enforce the cap. Best-effort; silent on failure.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        with p.open("a", encoding="utf-8") as f:
            if not json_line.endswith("\n"):
                f.write(json_line + "\n")
            else:
                f.write(json_line)
    except OSError:
        logger.debug("jsonl_retention: append failed for %s", p, exc_info=True)
        return
    cap_jsonl(p, max_lines)
