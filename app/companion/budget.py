"""Per-workspace daily cost ledger.

Daily reset is implicit: when ``state.cost_day_key`` falls behind
``utc_day_key()``, the ledger zeroes before the new charge applies.
"""

from __future__ import annotations

import logging

from app.companion import state as _state
from app.companion.config import CompanionConfig

logger = logging.getLogger(__name__)


def remaining_usd(project_id: str, config: CompanionConfig) -> float:
    """USD remaining in today's budget (>= 0)."""
    s = _state.load(project_id)
    today = _state.utc_day_key()
    spent = s.daily_cost_usd if s.cost_day_key == today else 0.0
    return max(0.0, config.daily_budget_usd - spent)


def is_exhausted(project_id: str, config: CompanionConfig) -> bool:
    return remaining_usd(project_id, config) <= 0.0


def charge(project_id: str, usd: float) -> float:
    """Add a cost to today's ledger. Returns the new daily total.

    Resets the ledger if a UTC-day boundary has been crossed since the
    last charge. Negative charges are ignored (treated as no-op) so a
    buggy caller can't refund an over-budget workspace.
    """
    if usd is None or usd < 0:
        return _state.load(project_id).daily_cost_usd
    s = _state.load(project_id)
    today = _state.utc_day_key()
    if s.cost_day_key != today:
        s.daily_cost_usd = 0.0
        s.cost_day_key = today
    s.daily_cost_usd = round(s.daily_cost_usd + float(usd), 6)
    _state.save(s)
    return s.daily_cost_usd
