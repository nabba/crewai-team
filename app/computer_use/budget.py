"""
computer_use.budget — monthly + per-task USD caps for vision computer use.

A single JSON file under ``workspace/computer_use_spend.json`` holds the
running spend for the current calendar month, plus a tail of recent task
records (cost + step count + truncated description) for diagnostics.

Three caps are enforced:

    MAX_STEPS_PER_TASK   30      hard
    MAX_USD_PER_TASK     $0.50   hard
    monthly cap          variable, read from runtime_settings (Phase 0)

The monthly cap defaults to $10 from the Phase 0 wiring; the operator can
raise or lower it from the React /cp/settings page at any time.

Pricing for Haiku 4.5 (claude-haiku-4-5-20251001) as of May 2026:
    input  $1   per 1M tokens     (cache write 1.25x, cache read 0.10x)
    output $5   per 1M tokens

`estimate_cost_usd` covers a single API turn given Anthropic's usage block.
The runner accumulates these and calls `record_task_cost` once on exit.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any

from app.paths import WORKSPACE_ROOT

logger = logging.getLogger(__name__)

# ── Hard caps ──────────────────────────────────────────────────────────────
MAX_STEPS_PER_TASK = 30
MAX_USD_PER_TASK = 0.50

# Haiku 4.5 prices (USD per 1M tokens)
HAIKU_INPUT_PER_1M = 1.0
HAIKU_OUTPUT_PER_1M = 5.0
HAIKU_CACHE_WRITE_PER_1M = 1.25
HAIKU_CACHE_READ_PER_1M = 0.10

_STORE_PATH = WORKSPACE_ROOT / "computer_use_spend.json"
_TAIL_LIMIT = 100  # how many recent task records to retain in the file
_lock = threading.Lock()


class BudgetExceeded(Exception):
    """Raised when a vision-CU task would cross either the per-task cap
    or the monthly cap. The runner converts this into a refusal string."""

    def __init__(self, scope: str, spent: float, cap: float, *, suggestion: str = ""):
        self.scope = scope
        self.spent = spent
        self.cap = cap
        self.suggestion = suggestion
        super().__init__(
            f"vision-cu {scope} cap reached: ${spent:.2f} of ${cap:.2f}"
            + (f". {suggestion}" if suggestion else "")
        )


# ── Persistence ───────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _current_month() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def _load() -> dict[str, Any]:
    if not _STORE_PATH.exists():
        return _empty_state()
    try:
        data = json.loads(_STORE_PATH.read_text())
    except Exception as exc:
        logger.warning(f"computer_use.budget: failed to load: {exc}")
        return _empty_state()
    if not isinstance(data, dict):
        return _empty_state()
    # Reset on month rollover.
    if data.get("month") != _current_month():
        return _empty_state()
    return data


def _empty_state() -> dict[str, Any]:
    return {"month": _current_month(), "spent_usd": 0.0, "tasks": []}


def _save(state: dict[str, Any]) -> None:
    _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _STORE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(_STORE_PATH)


# ── Public API ─────────────────────────────────────────────────────────────

def estimate_cost_usd(usage: dict[str, int]) -> float:
    """Compute the cost in USD for a single Anthropic ``usage`` block.

    The block is what the SDK returns: input_tokens, output_tokens, plus
    cache_creation_input_tokens and cache_read_input_tokens when prompt
    caching is enabled. Conservative — uses normal-input pricing for any
    field we don't explicitly recognise.
    """
    if not usage:
        return 0.0
    input_t = int(usage.get("input_tokens", 0) or 0)
    output_t = int(usage.get("output_tokens", 0) or 0)
    cache_write_t = int(usage.get("cache_creation_input_tokens", 0) or 0)
    cache_read_t = int(usage.get("cache_read_input_tokens", 0) or 0)
    return (
        (input_t       / 1_000_000) * HAIKU_INPUT_PER_1M
        + (output_t      / 1_000_000) * HAIKU_OUTPUT_PER_1M
        + (cache_write_t / 1_000_000) * HAIKU_CACHE_WRITE_PER_1M
        + (cache_read_t  / 1_000_000) * HAIKU_CACHE_READ_PER_1M
    )


def get_monthly_cap_usd() -> float:
    """Read the live monthly cap from runtime_settings (Phase 0)."""
    try:
        from app.runtime_settings import get_vision_cu_monthly_cap_usd
        return float(get_vision_cu_monthly_cap_usd())
    except Exception:
        return 10.0  # Sensible fallback matches the .env default.


def check_can_start() -> None:
    """Raise BudgetExceeded if a new task can't even start.

    Called BEFORE the first model call. Doesn't reserve budget; the runner
    re-checks after each step via ``check_step_within_budget``.
    """
    cap = get_monthly_cap_usd()
    with _lock:
        state = _load()
    spent = float(state.get("spent_usd", 0.0))
    if spent >= cap:
        raise BudgetExceeded(
            "monthly", spent, cap,
            suggestion="Raise the cap from /cp/settings or wait for next month.",
        )


def check_step_within_budget(task_spent_usd: float) -> None:
    """Raise BudgetExceeded if continuing the current step would exceed
    either the per-task cap or push the monthly total over the cap."""
    if task_spent_usd >= MAX_USD_PER_TASK:
        raise BudgetExceeded(
            "per-task", task_spent_usd, MAX_USD_PER_TASK,
            suggestion="Break the task into smaller steps.",
        )
    cap = get_monthly_cap_usd()
    with _lock:
        state = _load()
    monthly = float(state.get("spent_usd", 0.0)) + task_spent_usd
    if monthly >= cap:
        raise BudgetExceeded(
            "monthly", monthly, cap,
            suggestion="Raise the cap from /cp/settings or wait for next month.",
        )


def record_task_cost(
    task_summary: str,
    *,
    cost_usd: float,
    steps: int,
    success: bool,
    refused_reason: str = "",
) -> None:
    """Persist a finished task's spend. Trims the tail to ``_TAIL_LIMIT``."""
    with _lock:
        state = _load()
        state["spent_usd"] = float(state.get("spent_usd", 0.0)) + float(cost_usd)
        tasks = list(state.get("tasks") or [])
        tasks.append({
            "ts": _now_iso(),
            "summary": (task_summary or "")[:240],
            "cost_usd": round(float(cost_usd), 4),
            "steps": int(steps),
            "success": bool(success),
            "refused_reason": refused_reason,
        })
        state["tasks"] = tasks[-_TAIL_LIMIT:]
        _save(state)
    logger.info(
        f"computer_use.budget: recorded task ${cost_usd:.4f} steps={steps} "
        f"success={success}"
    )


def snapshot() -> dict[str, Any]:
    """Return the current month's spend + cap for the React page."""
    cap = get_monthly_cap_usd()
    with _lock:
        state = _load()
    spent = float(state.get("spent_usd", 0.0))
    return {
        "month": state.get("month", _current_month()),
        "monthly_cap_usd": cap,
        "spent_usd": round(spent, 4),
        "remaining_usd": round(max(0.0, cap - spent), 4),
        "task_count": len(state.get("tasks") or []),
        "recent_tasks": list(state.get("tasks") or [])[-10:][::-1],
    }
