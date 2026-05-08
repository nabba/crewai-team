"""
computer_use.audit — per-step structured logging.

Two endpoints:

  - ``log_step`` writes a JSON line per agent step (action + result) into
    ``workspace/computer_use_steps.jsonl`` for diagnosing reruns and
    spotting drift (e.g. the model suddenly clicking outside the viewport).

  - ``log_lifecycle`` writes a security_event into the hash-chained audit
    ledger (the same ledger settings_change uses), with actor
    ``computer_use``. This is the durable record that an operator can grep.

Both helpers are best-effort — they never raise in the runner's hot path.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from app.paths import WORKSPACE_ROOT

logger = logging.getLogger(__name__)

_STEPS_PATH = WORKSPACE_ROOT / "computer_use_steps.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def log_step(step: int, action: str, *, payload: dict[str, Any] | None = None,
             result: str | None = None, screenshot_kb: int | None = None,
             cost_usd: float | None = None) -> None:
    """Append one step record. Failure is silent (audit is best-effort)."""
    try:
        record = {
            "ts": _now_iso(),
            "step": int(step),
            "action": action,
            "payload": payload or {},
            "result": (result or "")[:500],
            "screenshot_kb": screenshot_kb,
            "cost_usd": cost_usd,
        }
        _STEPS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _STEPS_PATH.open("a") as fp:
            fp.write(json.dumps(record) + "\n")
    except Exception:
        logger.debug("computer_use.audit.log_step failed", exc_info=True)


def log_lifecycle(event: str, detail: dict[str, Any]) -> None:
    """Write ``computer_use_<event>`` into the hash-chained security log.

    ``event`` is a short verb such as ``start``, ``finish``, ``refuse``,
    ``budget_exceeded``. ``detail`` is JSON-serialisable.
    """
    try:
        from app.audit import log_security_event
        log_security_event(
            f"computer_use_{event}",
            json.dumps(detail, default=str),
        )
    except Exception:
        logger.debug("computer_use.audit.log_lifecycle failed", exc_info=True)


def recent_steps(n: int = 50) -> list[dict[str, Any]]:
    """Return the last N step records (oldest first)."""
    if not _STEPS_PATH.exists():
        return []
    try:
        with _STEPS_PATH.open("r") as fp:
            lines = fp.readlines()
        rows: list[dict[str, Any]] = []
        for line in lines[-n:]:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
        return rows
    except Exception:
        logger.debug("computer_use.audit.recent_steps failed", exc_info=True)
        return []
