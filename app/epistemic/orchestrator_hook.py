"""Single-call orchestrator integration point.

The commander orchestrator calls :func:`gate_output` once per delivery,
post-vetting and pre-user-facing send. Everything else flows from that
one call:

  1. Loads the per-task ledger.
  2. Runs :func:`app.epistemic.calibration.calibration_check`.
  3. If the verdict suggests ``peer_review``, escalates via
     :func:`app.epistemic.calibration.escalate`.
  4. Returns a :class:`GateResult` describing the orchestrator's
     options: proceed-as-is, ship a hedged revision, or block with a
     user-facing reason.

The hook respects two layered env vars:

* ``EPISTEMIC_ENABLED`` — master kill switch. Off → no-op.
* ``EPISTEMIC_BLOCKING_MODE`` — gate-enforces verdicts. Off → all
  non-trivial verdicts are passed through as ``proceed=True`` with a
  diagnostic note; the dashboard still shows the matches/escalations
  for monitoring. On → veto outcomes block delivery.

Phase 7 ships ``EPISTEMIC_BLOCKING_MODE=false`` as default. Operators
flip it after the soak window confirms the false-positive rate is
acceptable. The fine-grained
``EPISTEMIC_CALIBRATION_BLOCKS_OUTPUT`` flag (Phase 1) is preserved
for backward compatibility and is read by
:func:`app.epistemic.calibration.calibration_check` directly.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Literal

from app.epistemic import is_enabled
from app.epistemic.calibration import (
    CalibrationVerdict,
    calibration_check,
    escalate,
)
from app.epistemic.peer_review import (
    EscalationOutcome,
    PeerReviewDecision,
)
from app.epistemic.span_writer import load_ledger_for_task

logger = logging.getLogger(__name__)


GateAction = Literal["ship", "revise", "block"]


@dataclass(frozen=True)
class GateResult:
    """Verdict the orchestrator acts on.

    * ``ship`` — deliver ``final_text`` unchanged (or with the hedged
      version when ``revised=True``).
    * ``revise`` — deliver ``final_text`` (a peer-review-suggested
      revision). Treated like ``ship`` for delivery; distinguished
      so the orchestrator can mark the response as edited.
    * ``block`` — refuse to deliver. ``user_visible_reason`` is the
      single sentence the orchestrator surfaces to the user.
    """

    action: GateAction
    final_text: str                         # what the orchestrator should send
    user_visible_reason: str = ""           # populated for ``block`` and ``revise``
    diagnostic_note: str = ""               # for logs / dashboard, never shown to user
    revised: bool = False
    verdict: CalibrationVerdict | None = None
    escalation: EscalationOutcome | None = None
    blocking_mode: bool = False

    def as_jsonable(self) -> dict:
        return {
            "action": self.action,
            "final_text": self.final_text,
            "user_visible_reason": self.user_visible_reason,
            "diagnostic_note": self.diagnostic_note,
            "revised": self.revised,
            "verdict": self.verdict.as_jsonable() if self.verdict else None,
            "escalation": (
                self.escalation.as_jsonable() if self.escalation else None
            ),
            "blocking_mode": self.blocking_mode,
        }


def is_blocking_mode_enabled() -> bool:
    """Whether veto / revise verdicts actually gate delivery.

    Off by default (Phase 7 ships in observe-mode). Operators flip
    ``EPISTEMIC_BLOCKING_MODE=true`` after the soak window confirms
    low false-positive rates.
    """
    val = os.getenv("EPISTEMIC_BLOCKING_MODE", "").strip().lower()
    return val in ("1", "true", "yes", "on")


def gate_output(
    *,
    proposal_text: str,
    task_id: str,
    triggering_claim_id: str | None = None,
) -> GateResult:
    """Run the calibration + peer-review gate for a single delivery.

    Always returns a structured :class:`GateResult`. Never raises —
    any internal failure is reflected as a ``ship`` action with a
    diagnostic note (the user-facing path must not break on telemetry
    or detection failures).
    """
    # Master kill switch — bypass entirely.
    if not is_enabled():
        return GateResult(
            action="ship",
            final_text=proposal_text,
            diagnostic_note="epistemic layer disabled (EPISTEMIC_ENABLED unset)",
            blocking_mode=False,
        )

    blocking = is_blocking_mode_enabled()

    try:
        ledger = load_ledger_for_task(task_id)
    except Exception as exc:
        logger.debug("epistemic gate_output: ledger load failed: %s", exc)
        return GateResult(
            action="ship",
            final_text=proposal_text,
            diagnostic_note=f"ledger unavailable ({exc!r}) — proceeding",
            blocking_mode=blocking,
        )

    try:
        verdict = calibration_check(ledger=ledger)
    except Exception as exc:
        logger.debug("epistemic gate_output: calibration_check failed: %s", exc)
        return GateResult(
            action="ship",
            final_text=proposal_text,
            diagnostic_note=f"calibration_check failed ({exc!r})",
            blocking_mode=blocking,
        )

    # No biases fired → ship.
    if verdict.suggested_action == "ship":
        return GateResult(
            action="ship",
            final_text=proposal_text,
            verdict=verdict,
            blocking_mode=blocking,
        )

    # Calibration suggests verify/hedge but no peer-review escalation needed.
    if verdict.suggested_action != "peer_review":
        return _gate_for_non_critical(
            proposal_text=proposal_text,
            verdict=verdict,
            blocking=blocking,
        )

    # Peer-review escalation path.
    try:
        escalation = escalate(
            proposal_text=proposal_text,
            ledger=ledger,
            verdict=verdict,
            triggering_claim_id=triggering_claim_id,
        )
    except Exception as exc:
        logger.debug("epistemic gate_output: escalate failed: %s", exc)
        return GateResult(
            action="ship",
            final_text=proposal_text,
            diagnostic_note=f"escalate failed ({exc!r})",
            verdict=verdict,
            blocking_mode=blocking,
        )

    return _gate_for_escalation(
        proposal_text=proposal_text,
        verdict=verdict,
        escalation=escalation,
        blocking=blocking,
    )


# ── Internal arms ───────────────────────────────────────────────────

def _gate_for_non_critical(
    *,
    proposal_text: str,
    verdict: CalibrationVerdict,
    blocking: bool,
) -> GateResult:
    """Non-critical verdicts: hedge or verify suggestions.

    In observe-mode we always ship with a diagnostic note (the
    dashboard shows the matches; nothing changes for the user). In
    blocking-mode we hedge by appending a brief disclaimer — the
    actual hedging logic could be more sophisticated, but a one-line
    note preserves the user's flow without silently shipping a
    flagged claim as fact.
    """
    if not blocking:
        return GateResult(
            action="ship",
            final_text=proposal_text,
            diagnostic_note=(
                f"calibration suggested {verdict.suggested_action!r} "
                f"({len(verdict.biases_detected)} match(es)); observe-mode"
            ),
            verdict=verdict,
            blocking_mode=blocking,
        )

    # Blocking-mode: hedge by appending a single-sentence note.
    hedge = (
        "\n\nNote: I have low confidence in part of this — please verify "
        "the load-bearing claims before relying on the recommendation."
    )
    return GateResult(
        action="revise",
        final_text=proposal_text + hedge,
        user_visible_reason="hedged — calibration flagged low-confidence claim(s)",
        diagnostic_note=verdict.note_for_post_mortem,
        revised=True,
        verdict=verdict,
        blocking_mode=blocking,
    )


def _gate_for_escalation(
    *,
    proposal_text: str,
    verdict: CalibrationVerdict,
    escalation: EscalationOutcome,
    blocking: bool,
) -> GateResult:
    """Peer-review escalation outcome → GateResult.

    In observe-mode the verdict is recorded but the orchestrator
    proceeds with the original proposal. In blocking-mode, veto
    refuses delivery and revise replaces the text.
    """
    if escalation.verdict is None:
        return GateResult(
            action="ship",
            final_text=proposal_text,
            diagnostic_note="peer review returned no verdict",
            verdict=verdict,
            escalation=escalation,
            blocking_mode=blocking,
        )

    decision = escalation.verdict.decision
    rationale = escalation.verdict.rationale

    if not blocking:
        return GateResult(
            action="ship",
            final_text=proposal_text,
            diagnostic_note=(
                f"peer-review {decision.value} ({rationale}); observe-mode"
            ),
            verdict=verdict,
            escalation=escalation,
            blocking_mode=blocking,
        )

    if decision is PeerReviewDecision.ALLOW:
        return GateResult(
            action="ship",
            final_text=proposal_text,
            verdict=verdict,
            escalation=escalation,
            blocking_mode=blocking,
        )

    if decision is PeerReviewDecision.REVISE:
        revision = (
            escalation.verdict.suggested_revision
            or _hedge_for_revise(proposal_text, rationale)
        )
        return GateResult(
            action="revise",
            final_text=revision,
            user_visible_reason=f"revised — {rationale}",
            verdict=verdict,
            escalation=escalation,
            revised=True,
            blocking_mode=blocking,
        )

    # VETO
    return GateResult(
        action="block",
        final_text=_blocked_user_message(rationale),
        user_visible_reason=rationale,
        diagnostic_note=(
            f"peer review veto on destructive recommendation: {rationale}"
        ),
        verdict=verdict,
        escalation=escalation,
        blocking_mode=blocking,
    )


def _hedge_for_revise(proposal_text: str, rationale: str) -> str:
    return (
        f"{proposal_text}\n\n"
        f"Caveat: peer review flagged this — {rationale}. "
        "Please verify before proceeding."
    )


def _blocked_user_message(rationale: str) -> str:
    return (
        "I'm pausing on this recommendation: " + rationale + " "
        "Could you confirm whether to proceed anyway, or give me more "
        "context to verify the underlying claims?"
    )
