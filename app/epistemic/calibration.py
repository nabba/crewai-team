"""Pre-output calibration check.

The function the orchestrator calls just before delivering a response to
the user. Reads the per-task bias-match log, decides whether to ship as-is
or take a corrective action.

Phase 1 (this commit) ships in *warn-mode*: ``proceed=True`` is always
returned, even on bias matches. The biases_detected list is populated so
the dashboard sees the activity, but no output is blocked. Phase 7 flips
the blocking flag to ``True`` after a soak window confirms low
false-positive rates.

Suggested actions follow a simple precedence:

    critical → peer_review (Phase 6 wires this)
    high blocking + verifier available → verify
    high non-blocking → hedge
    medium → hedge
    none → ship
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Literal

from app.epistemic.biases import BIAS_LIBRARY, BiasMatch, Severity, severity_rank
from app.epistemic.ledger import Ledger, VerificationStatus

logger = logging.getLogger(__name__)


SuggestedAction = Literal["ship", "hedge", "verify", "peer_review"]


@dataclass(frozen=True)
class CalibrationVerdict:
    proceed: bool
    suggested_action: SuggestedAction
    biases_detected: tuple[BiasMatch, ...] = ()
    forced_verifier_claim_ids: tuple[str, ...] = ()
    note_for_post_mortem: str = ""

    def as_jsonable(self) -> dict:
        return {
            "proceed": self.proceed,
            "suggested_action": self.suggested_action,
            "biases_detected": [m.as_jsonable() for m in self.biases_detected],
            "forced_verifier_claim_ids": list(self.forced_verifier_claim_ids),
            "note_for_post_mortem": self.note_for_post_mortem,
        }


def calibration_check(
    *,
    ledger: Ledger,
    matches: list[BiasMatch] | None = None,
) -> CalibrationVerdict:
    """Decide whether the pending output should ship.

    ``matches`` is optional — if not supplied, the bias matches recorded
    during the task (via the realtime meta-hook) are loaded from
    :mod:`app.epistemic.span_writer`. Tests typically pass it explicitly
    to avoid the DB round-trip.
    """
    if matches is None:
        matches = _load_matches_for_task(ledger.task_id)
    if not matches:
        return CalibrationVerdict(proceed=True, suggested_action="ship")

    blocking_mode = _blocks_output()

    # Pick the worst severity. ``max`` handles single-element lists fine.
    worst = max(matches, key=lambda m: severity_rank(m.severity))
    suggested = _action_for(worst, ledger)

    proceed = True
    if blocking_mode and BIAS_LIBRARY.get(worst.bias_id).blocking:
        proceed = False
    if blocking_mode and worst.severity is Severity.CRITICAL:
        proceed = False

    return CalibrationVerdict(
        proceed=proceed,
        suggested_action=suggested,
        biases_detected=tuple(matches),
        forced_verifier_claim_ids=_pending_verifier_claim_ids(ledger),
        note_for_post_mortem=_summary_note(matches),
    )


# ── Helpers ─────────────────────────────────────────────────────────

def _action_for(match: BiasMatch, ledger: Ledger) -> SuggestedAction:
    """Map a bias match to a suggested corrective action."""
    if match.severity is Severity.CRITICAL:
        return "peer_review"
    if match.bias_id == "inference_as_fact":
        # The verifier is what makes this resolvable cheaply.
        if any(c.verifying_action is not None for c in ledger.unverified_load_bearing()):
            return "verify"
    return "hedge"


# ── Phase 6: peer-review escalation helper ──────────────────────────

def escalate(
    *,
    proposal_text: str,
    ledger: Ledger,
    verdict: CalibrationVerdict,
    triggering_claim_id: str | None = None,
):
    """Run peer review when the calibration verdict suggests it.

    Returns an :class:`~app.epistemic.peer_review.EscalationOutcome`:

      * ``escalated=False`` if the calibration verdict didn't suggest
        peer review — the caller proceeds with the original proposal.
      * ``escalated=True`` with a ``verdict`` (allow/revise/veto) when
        peer review ran. The orchestrator interprets and either ships,
        revises, or refuses.

    This is the Phase 6 wiring point. The orchestrator (Phase 7) calls
    :func:`calibration_check` first, then this helper if the verdict
    is non-trivial. Persisting the peer-review row happens inside
    :func:`request_peer_review`.
    """
    from app.epistemic.peer_review import escalate_if_destructive
    return escalate_if_destructive(
        proposal_text=proposal_text,
        ledger=ledger,
        suggested_action=verdict.suggested_action,
        triggering_claim_id=triggering_claim_id,
    )


def _pending_verifier_claim_ids(ledger: Ledger) -> tuple[str, ...]:
    """Claims the orchestrator should run verifiers for before reissuing."""
    return tuple(
        c.claim_id
        for c in ledger.unverified_load_bearing()
        if c.verifying_action is not None
        and c.status is VerificationStatus.INFERRED
    )


def _summary_note(matches: list[BiasMatch]) -> str:
    counts: dict[str, int] = {}
    for m in matches:
        counts[m.bias_id] = counts.get(m.bias_id, 0) + 1
    return ", ".join(f"{bid}×{n}" for bid, n in sorted(counts.items()))


def _load_matches_for_task(task_id: str) -> list[BiasMatch]:
    """Reconstruct BiasMatch instances from the persistence layer."""
    from app.epistemic.span_writer import list_bias_matches_for_task
    rows = list_bias_matches_for_task(task_id)
    out: list[BiasMatch] = []
    for r in rows:
        try:
            severity = Severity(r["severity"])
        except ValueError:
            logger.debug(
                "epistemic calibration: skipping match with unknown severity %r",
                r.get("severity"),
            )
            continue
        out.append(BiasMatch(
            bias_id=r["bias_id"],
            matched_claim_ids=tuple(r.get("matched_claim_ids", []) or []),
            severity=severity,
            detail=r.get("detail", {}) or {},
        ))
    return out


def _blocks_output() -> bool:
    """Whether calibration violations should block delivery (vs warn).

    Phase 1 default: False (warn-mode). Operators can opt in early via
    ``EPISTEMIC_CALIBRATION_BLOCKS_OUTPUT=true``; Phase 7 flips the
    default after a soak window.
    """
    val = os.getenv("EPISTEMIC_CALIBRATION_BLOCKS_OUTPUT", "").strip().lower()
    return val in ("1", "true", "yes", "on")
