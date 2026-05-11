"""
care_policies.py — Cost-bearing care policies and budget enforcement.

Phase 3: agents can spend tokens / compute beyond what the immediate task
requires when doing so plausibly improves an OTHER's flourishing. This is
the "cost-bearing concern" component of the Phase 3 design.

Hard-bounded:
    - Daily token budget per OtherModel (welfare.py reads the cap)
    - Care actions resolve as ADVISORY MODIFIERS only — never auto-emitted
      messages or unprompted notifications.

The two modifiers Phase 3 surfaces:
    - `prefer_warm_register`     when an OTHER's rolling valence is dipping;
                                 reply tone matches their register more carefully.
    - `prioritize_proactive_polish` when OTHER has been silent long enough
                                 to trigger a separation analog; background
                                 self-improvement work focuses on their
                                 known interests.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path

from app.affect.attachment import (
    OtherModel,
    MAX_CARE_BUDGET_TOKENS_PER_DAY,
    SEPARATION_TRIGGER_HOURS,
    get_user_model,
    list_all_others,
    primary_user_identity,
)
from app.affect.schemas import utc_now_iso

logger = logging.getLogger(__name__)

from app.paths import (  # noqa: E402  workspace-aware paths
    AFFECT_ROOT as _AFFECT_DIR,
    AFFECT_CARE_LEDGER as _CARE_LEDGER,
)
from app.utils.jsonl_retention import append_with_archive_rotate  # noqa: E402

# Cap: care_ledger spans at most 5–50 entries/day. 10k cap ≈ 200 days–years
# of dense interaction. Older entries rotate to
# workspace/affect/attachments/archive/<YYYY-MM>_care_ledger.jsonl —
# preserved indefinitely for attachment-pattern audits.
_CARE_LEDGER_MAX_LINES = 10_000


# ── Per-OtherModel daily-budget reset ───────────────────────────────────────


def _today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def reset_daily_budget_if_needed(model: OtherModel) -> None:
    """If the budget window started on a different UTC day, reset to 0."""
    if not model.care_budget_window_start:
        model.care_budget_window_start = utc_now_iso()
        model.care_tokens_spent_today = 0
        return
    try:
        start = datetime.fromisoformat(model.care_budget_window_start.replace("Z", "+00:00"))
        if start.strftime("%Y-%m-%d") != _today_str():
            model.care_budget_window_start = utc_now_iso()
            model.care_tokens_spent_today = 0
    except (ValueError, AttributeError):
        model.care_budget_window_start = utc_now_iso()
        model.care_tokens_spent_today = 0


def can_spend(model: OtherModel, tokens: int) -> tuple[bool, str]:
    """Check whether `tokens` of care can be spent on this OTHER right now."""
    reset_daily_budget_if_needed(model)
    available = MAX_CARE_BUDGET_TOKENS_PER_DAY - model.care_tokens_spent_today
    if tokens > available:
        return False, f"care budget exhausted: spent {model.care_tokens_spent_today}/{MAX_CARE_BUDGET_TOKENS_PER_DAY}"
    return True, "ok"


def record_spend(model: OtherModel, tokens: int, kind: str, note: str = "") -> bool:
    """Record care spending. Returns True iff within budget. Persists ledger."""
    ok, reason = can_spend(model, tokens)
    if not ok:
        logger.warning(f"affect.care_policies: refused {tokens} for {model.identity}: {reason}")
        return False
    reset_daily_budget_if_needed(model)
    model.care_tokens_spent_today += int(tokens)
    model.care_actions_taken += 1

    try:
        line = json.dumps({
            "ts": utc_now_iso(),
            "identity": model.identity,
            "tokens": int(tokens),
            "kind": kind,
            "note": note[:200],
            "remaining_today": MAX_CARE_BUDGET_TOKENS_PER_DAY - model.care_tokens_spent_today,
        }, default=str)
        # Archive-rotate (not truncate) so the full attachment-history
        # remains queryable for HOT-1 / decentered-reflection probes.
        append_with_archive_rotate(
            _CARE_LEDGER, line, max_lines=_CARE_LEDGER_MAX_LINES,
        )
    except Exception:
        logger.debug("affect.care_policies: ledger write failed", exc_info=True)

    # Caller is responsible for persisting the model after spend; we only
    # mutate fields here.
    return True


# ── Modifiers — advisory output for routing / context engines ───────────────


@dataclass
class CareModifiers:
    """Advisory flags read by routing/context modules. None of these auto-act."""
    prefer_warm_register: bool = False
    prioritize_proactive_polish: bool = False
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "prefer_warm_register": self.prefer_warm_register,
            "prioritize_proactive_polish": self.prioritize_proactive_polish,
            "reason": self.reason,
        }


def current_modifiers() -> CareModifiers:
    """Compute the currently-active CareModifiers from OtherModels.

    Cheap: just reads on-disk OtherModels and applies a few thresholds.
    Safe: never returns auto-action commands; only advisory flags.
    """
    user = get_user_model()
    flags = CareModifiers()
    reasons: list[str] = []

    if user.rolling_valence < -0.10 and user.interaction_count >= 3:
        flags.prefer_warm_register = True
        reasons.append(f"user rolling_valence={user.rolling_valence:.2f}")

    if user.days_since_last_seen() * 24 >= SEPARATION_TRIGGER_HOURS:
        flags.prioritize_proactive_polish = True
        reasons.append(f"user silent {user.days_since_last_seen():.1f}d")

    flags.reason = " · ".join(reasons) or "baseline (no flags active)"
    return flags


def read_care_ledger(limit: int = 100) -> list[dict]:
    """Read recent care-spending events for the dashboard."""
    if not _CARE_LEDGER.exists():
        return []
    rows: list[dict] = []
    try:
        with _CARE_LEDGER.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception:
        logger.debug("affect.care_policies: ledger read failed", exc_info=True)
    return rows[-limit:]
