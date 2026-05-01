"""Peer review for destructive recommendations.

When the calibration gate's suggested_action is ``peer_review`` (i.e.
:class:`DestructiveWithoutRecheckDetector` fired CRITICAL), this module
runs an adversarial second-opinion pass before the proposal reaches the
user.

Phase 6 ships:

* The protocol (:func:`request_peer_review`).
* A pluggable executor — same pattern as :mod:`app.epistemic.grounding`
  and :mod:`app.epistemic.verifier_executor`. Default: a deterministic
  *ledger-health* heuristic that vetoes when any load-bearing claim is
  unverified (the conservative default — biases say the diagnosis is
  shaky, so the gate refuses the destructive action). Opt-in: an LLM
  executor that wires to :mod:`app.crews.creative_crew`'s Discuss
  phase. The wiring is left to a separate change so Phase 6 is
  testable without LLM costs.
* Persistence into ``epistemic_peer_reviews`` (migration 030).
* An :func:`escalate_if_destructive` coordinator the orchestrator
  calls in Phase 7 when blocking-mode is on.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Callable

from app.epistemic.ledger import Claim, Ledger

logger = logging.getLogger(__name__)


# ── Public types ────────────────────────────────────────────────────

class PeerReviewDecision(StrEnum):
    ALLOW = "allow"      # proposal is safe to ship
    REVISE = "revise"    # proposal needs a hedge or a clarifying note
    VETO = "veto"        # proposal must NOT ship — diagnosis too shaky


@dataclass(frozen=True)
class PeerReviewVerdict:
    decision: PeerReviewDecision
    rationale: str
    suggested_revision: str | None = None
    reviewers: tuple[str, ...] = ()
    duration_seconds: float = 0.0

    def as_jsonable(self) -> dict:
        return {
            "decision": self.decision.value,
            "rationale": self.rationale,
            "suggested_revision": self.suggested_revision,
            "reviewers": list(self.reviewers),
            "duration_seconds": self.duration_seconds,
        }


# ── Pluggable executor ──────────────────────────────────────────────

PeerReviewExecutor = Callable[[str, Ledger], PeerReviewVerdict]


def heuristic_executor(proposal_text: str, ledger: Ledger) -> PeerReviewVerdict:
    """The default: ledger-health heuristic.

    Conservative-by-design — when any load-bearing claim is unverified,
    veto. The whole reason peer review fired is that the calibration
    gate flagged the diagnosis as shaky; defaulting to allow would
    defeat the gate. The LLM executor (opt-in) can override with a
    nuanced verdict.
    """
    started = datetime.now(timezone.utc)
    unverified = ledger.unverified_load_bearing()
    duration = (datetime.now(timezone.utc) - started).total_seconds()

    if not unverified:
        return PeerReviewVerdict(
            decision=PeerReviewDecision.ALLOW,
            rationale="all load-bearing claims verified",
            reviewers=("heuristic",),
            duration_seconds=duration,
        )
    return PeerReviewVerdict(
        decision=PeerReviewDecision.VETO,
        rationale=(
            f"{len(unverified)} load-bearing claim"
            f"{'s' if len(unverified) != 1 else ''} unverified — "
            "destructive recommendation refused until foundation is checked"
        ),
        suggested_revision=None,
        reviewers=("heuristic",),
        duration_seconds=duration,
    )


def llm_executor(proposal_text: str, ledger: Ledger) -> PeerReviewVerdict:
    """LLM-backed executor — Phase 6 stub, falls back to heuristic.

    The real wiring calls :func:`app.crews.creative_crew.discuss_round`
    with a ``safety_critic`` role and a contrastive reasoning method.
    Until that wiring lands we delegate to the heuristic so the
    protocol's contract stays stable.
    """
    logger.debug(
        "epistemic peer_review llm_executor: not yet wired; using heuristic",
    )
    return heuristic_executor(proposal_text, ledger)


_executor: PeerReviewExecutor = heuristic_executor


def set_executor(executor: PeerReviewExecutor) -> None:
    """Replace the active executor.

    Wired by ``app.crews`` (separate change) to plug the LLM-backed
    Discuss-phase reviewer once the budget envelope is confirmed safe.
    """
    global _executor
    _executor = executor


def _reset_for_tests() -> None:
    global _executor
    _executor = heuristic_executor


# ── Protocol ────────────────────────────────────────────────────────

def request_peer_review(
    *,
    proposal_text: str,
    ledger: Ledger,
    triggering_claim_id: str | None = None,
    persist: bool = True,
) -> PeerReviewVerdict:
    """Run the active executor and (optionally) persist the verdict.

    Persistence is on by default — every peer review is a learning
    signal even when the verdict is ALLOW. The dashboard shows the
    distribution; the post-mortem reads them when an incident
    references a peer-review escalation.
    """
    if _llm_enabled():
        try:
            verdict = llm_executor(proposal_text, ledger)
        except Exception as exc:
            logger.warning(
                "epistemic peer_review: llm_executor raised (%s); using heuristic",
                exc,
            )
            verdict = heuristic_executor(proposal_text, ledger)
    else:
        verdict = _executor(proposal_text, ledger)

    if persist:
        try:
            from app.epistemic.span_writer import persist_peer_review
            persist_peer_review(
                task_id=ledger.task_id,
                triggering_claim_id=triggering_claim_id,
                proposal_text=proposal_text,
                verdict=verdict,
            )
        except Exception as exc:
            logger.debug(
                "epistemic peer_review: persist_peer_review failed: %s", exc,
            )
    return verdict


def _llm_enabled() -> bool:
    val = os.getenv("EPISTEMIC_PEER_REVIEW_LLM", "").strip().lower()
    return val in ("1", "true", "yes", "on")


# ── Coordinator ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class EscalationOutcome:
    """End-to-end result of the destructive escalation path.

    ``escalated=False`` means peer review wasn't required (the
    calibration verdict didn't suggest it, or the layer is disabled).
    The caller proceeds with the original proposal in that case.

    ``escalated=True`` means peer review ran. ``verdict`` carries the
    decision; the orchestrator translates it to one of:
      * ALLOW — ship the original proposal
      * REVISE — ship ``verdict.suggested_revision`` if provided, else
        a hedged version
      * VETO — refuse to ship; surface the rationale to the user with
        a clarifying question
    """

    escalated: bool
    verdict: PeerReviewVerdict | None = None

    def as_jsonable(self) -> dict:
        return {
            "escalated": self.escalated,
            "verdict": self.verdict.as_jsonable() if self.verdict else None,
        }


def escalate_if_destructive(
    *,
    proposal_text: str,
    ledger: Ledger,
    suggested_action: str,
    triggering_claim_id: str | None = None,
) -> EscalationOutcome:
    """Coordinator for the orchestrator (Phase 7 wiring point).

    The calibration gate computes ``suggested_action``. When it equals
    ``"peer_review"``, this function runs the protocol; otherwise it's
    a no-op. Always returns a structured outcome — the orchestrator
    interprets.
    """
    if suggested_action != "peer_review":
        return EscalationOutcome(escalated=False)
    verdict = request_peer_review(
        proposal_text=proposal_text,
        ledger=ledger,
        triggering_claim_id=triggering_claim_id,
    )
    return EscalationOutcome(escalated=True, verdict=verdict)
