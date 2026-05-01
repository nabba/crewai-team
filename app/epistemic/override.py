"""User-override feedback loop.

When the user disagrees with a gate verdict — typically a peer-review
veto or a blocked output — and forces delivery anyway, that override
is itself a learning signal. Two interpretations are possible:

* **The library is too strict** (false positive): the diagnosis was
  fine despite shaky ledger health, and the bias library should be
  tuned down.
* **The user is overruling for context the system can't see**: the
  user knows something the agent doesn't, and the override is correct
  for this case but not generalizable.

Phase 7 records both kinds with the user's stated reasoning. The
Self-Improver consumes the events as ``LearningGap`` records with
``GapSource.USER_CORRECTION`` (the strongest organic source weight,
0.9). Subsequent tuning is a human-reviewed PR — the agent never
modifies the bias library at runtime.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class OverrideAction(StrEnum):
    """What the user did with a gate verdict they disagreed with."""

    FORCE_PROCEED = "force_proceed"       # ship original despite veto/revise
    USE_REVISION = "use_revision"         # accept the revised text
    ABANDON = "abandon"                   # cancel the action entirely


@dataclass(frozen=True)
class OverrideEvent:
    override_id: str
    task_id: str
    peer_review_id: int | None              # if overriding a peer-review verdict
    blocked_action: str                     # "block" | "revise"
    user_action: OverrideAction
    user_reasoning: str                     # the user's stated reason
    overridden_at: datetime

    def as_jsonable(self) -> dict[str, Any]:
        return {
            "override_id": self.override_id,
            "task_id": self.task_id,
            "peer_review_id": self.peer_review_id,
            "blocked_action": self.blocked_action,
            "user_action": self.user_action.value,
            "user_reasoning": self.user_reasoning,
            "overridden_at": self.overridden_at.isoformat(),
        }


def record_override(
    *,
    task_id: str,
    blocked_action: str,
    user_action: OverrideAction,
    user_reasoning: str,
    peer_review_id: int | None = None,
    flush_to_self_improver: bool = True,
) -> OverrideEvent:
    """Persist an override event and (optionally) feed Self-Improver.

    The override is *always* persisted (per the Phase 0 fire-and-forget
    contract — DB failures swallowed at DEBUG). Self-Improver flushing
    is opt-in per call and best-effort.

    Returns the constructed :class:`OverrideEvent` regardless of
    persistence success. The caller (the orchestrator or a Signal
    command) uses the event id for follow-up flows.
    """
    event = OverrideEvent(
        override_id=f"ovr_{uuid4().hex[:12]}",
        task_id=task_id,
        peer_review_id=peer_review_id,
        blocked_action=blocked_action,
        user_action=user_action,
        user_reasoning=user_reasoning.strip(),
        overridden_at=datetime.now(timezone.utc),
    )

    try:
        from app.epistemic.span_writer import persist_override
        persist_override(event)
    except Exception as exc:
        logger.debug("epistemic record_override: persist failed: %s", exc)

    if flush_to_self_improver:
        _flush_to_self_improver(event)

    return event


def _flush_to_self_improver(event: OverrideEvent) -> bool:
    """Best-effort: feed the override into the Self-Improver loop.

    Uses ``GapSource.USER_CORRECTION`` (signal_strength=0.9) — the
    strongest organic source. The user just told us our gate was
    wrong (or right but worth overriding for unseen context); that's
    a stronger signal than retrieval misses or trajectory attribution.

    Returns True on successful emit, False otherwise. Override
    persistence is independent of this — even if Self-Improver isn't
    available, the override row is still in the DB for human review.
    """
    try:
        from app.self_improvement.types import GapSource, LearningGap
        from app.self_improvement.store import emit_gap
    except ImportError:
        logger.debug(
            "epistemic override: Self-Improver not available; "
            "override %s recorded but not flushed",
            event.override_id,
        )
        return False

    try:
        gap = LearningGap(
            id="",
            source=GapSource.USER_CORRECTION,
            description=(
                f"Epistemic gate override: user {event.user_action.value} "
                f"on a {event.blocked_action} verdict"
            )[:200],
            evidence={
                "override_id": event.override_id,
                "task_id": event.task_id,
                "peer_review_id": event.peer_review_id,
                "blocked_action": event.blocked_action,
                "user_action": event.user_action.value,
                "user_reasoning": event.user_reasoning[:1000],
            },
            signal_strength=0.9,
        )
        return bool(emit_gap(gap))
    except Exception as exc:
        logger.warning(
            "epistemic override: emit_gap failed: %s", exc,
        )
        return False
