"""User-contradiction handler — the adversarial trigger.

When a user message contradicts a load-bearing claim, this module
runs a deterministic protocol:

  1. **Detect** the contradiction (``detect_contradiction``).
  2. **Re-verify** the foundation by running the load-bearing claim's
     :class:`VerifyingAction` — and ONLY that. No investigation
     expansion, no peripheral exploration.
  3. **Cascade** the outcome:
     * REVERIFIED — foundation re-confirmed; the user may have a new
       question, falls through to normal handling with a noted hedge.
     * FALSIFIED — foundation wrong; original claim plus all
       dependents are superseded; reasoning restarts.
     * UNVERIFIABLE — no verifier ran (default executor / no shape
       in registry / executor returned ``settles=False``); surfaced
       to the user with a hedge.

The narrowness of step 2 is the *whole point* of the protocol — it
structurally prevents the "defending the periphery" failure mode from
the April 2026 reference incident, where an agent investigated mount
tables for several minutes instead of running ``readlink``.

Detection in Phase 3 uses a regex/heuristic classifier (no LLM
dependency). Phase 5+ adds an LLM-based classifier for non-obvious
contradictions, opt-in via ``EPISTEMIC_PUSHBACK_LLM_DETECTOR=true``.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from app.epistemic.ledger import (
    Claim,
    Evidence,
    Ledger,
    Register,
    VerificationStatus,
)
from app.epistemic.verifier_executor import VerifierResult, execute

logger = logging.getLogger(__name__)


# Lower bound on detector confidence below which we treat the message
# as "not a contradiction". Defined here (not as a Settings field)
# because it's a safety boundary — agent-modifiable thresholds on
# self-correction would let the system widen its own gate.
_MIN_CONFIDENCE = 0.60


# ── Public types ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class ContradictionSignal:
    """The classifier's verdict on a single user message."""

    contradicted_claim_id: str
    user_evidence: str        # the user's contradicting statement, verbatim
    confidence: float         # 0.0–1.0
    detected_at: datetime
    detector: str             # "regex" | "llm" — which classifier fired

    def as_jsonable(self) -> dict[str, Any]:
        return {
            "contradicted_claim_id": self.contradicted_claim_id,
            "user_evidence": self.user_evidence,
            "confidence": self.confidence,
            "detected_at": self.detected_at.isoformat(),
            "detector": self.detector,
        }


class FoundationOutcome(StrEnum):
    REVERIFIED = "reverified"     # foundation re-confirmed
    FALSIFIED = "falsified"       # foundation wrong; cascade-invalidate
    UNVERIFIABLE = "unverifiable" # no exact-answer verifier ran


@dataclass(frozen=True)
class FoundationCheckResult:
    """The result of a foundation re-check.

    ``invalidated_claim_ids`` is non-empty only when ``outcome ==
    FALSIFIED`` — it lists every dependent claim that was
    cascade-superseded.
    """

    outcome: FoundationOutcome
    contradicted_claim_id: str
    new_evidence_excerpt: str
    invalidated_claim_ids: tuple[str, ...] = ()
    duration_seconds: float = 0.0

    def as_jsonable(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "contradicted_claim_id": self.contradicted_claim_id,
            "new_evidence_excerpt": self.new_evidence_excerpt,
            "invalidated_claim_ids": list(self.invalidated_claim_ids),
            "duration_seconds": self.duration_seconds,
        }


# ── Detection ────────────────────────────────────────────────────────

# Regex shapes that signal explicit contradiction. Conservative on
# purpose — false positives here trigger an unnecessary verifier run,
# which is cheap; false negatives just leave the protocol unfired,
# which the LLM classifier will eventually catch.
_CONTRADICTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bno,?\s+(?:that's|that is|it's|it is|you're|you are)\s+(?:wrong|incorrect|not right)\b", re.IGNORECASE),
    re.compile(r"\b(?:that's|that is|you're|you are)\s+wrong\b", re.IGNORECASE),
    re.compile(r"\bactually,?\s+\S+\s+(?:is|isn't|was|wasn't|does|doesn't|did|didn't)\b", re.IGNORECASE),
    re.compile(r"\bi\s+(?:just\s+)?checked\b", re.IGNORECASE),
    re.compile(r"\bincorrect\b", re.IGNORECASE),
    re.compile(r"\bnot\s+(?:true|correct|right)\b", re.IGNORECASE),
)


def _looks_like_contradiction(user_input: str) -> bool:
    return any(p.search(user_input) for p in _CONTRADICTION_PATTERNS)


def _candidate_claims(ledger: Ledger, lookback: int) -> list[Claim]:
    """Recent load-bearing claims in emission order, newest last."""
    load_bearing = [c for c in ledger.all() if c.load_bearing]
    if lookback > 0:
        load_bearing = load_bearing[-lookback:]
    return load_bearing


def _best_match(user_input: str, candidates: list[Claim]) -> tuple[Claim, float] | None:
    """Score each candidate by token overlap with the user input.

    Returns the highest-scoring candidate with its confidence, or
    ``None`` if the best score falls below :data:`_MIN_CONFIDENCE`.
    Confidence = (overlapping_tokens / claim_token_count). This is
    crude but sufficient for the obvious-pushback case the regex
    classifier targets.
    """
    if not candidates:
        return None
    user_tokens = _tokens(user_input)
    if not user_tokens:
        return None

    best_claim: Claim | None = None
    best_score = 0.0
    for claim in candidates:
        claim_tokens = _tokens(claim.statement)
        if not claim_tokens:
            continue
        overlap = len(user_tokens & claim_tokens)
        score = overlap / max(1, len(claim_tokens))
        if score > best_score:
            best_score = score
            best_claim = claim
    if best_claim is None:
        return None
    if best_score < _MIN_CONFIDENCE:
        return None
    return best_claim, best_score


def _tokens(text: str) -> set[str]:
    """Lowercase alphanumeric tokens of length ≥ 3 — strips fillers
    while keeping subjects (paths, identifiers, file names)."""
    raw = re.findall(r"[A-Za-z0-9_./-]{3,}", text.lower())
    # Drop super-common words that add overlap noise without signal.
    stopwords = {
        "the", "and", "for", "but", "you", "are", "this", "that",
        "with", "from", "have", "has", "had", "was", "were", "not",
        "yes", "actually", "really", "still", "wrong", "right",
        "correct", "incorrect", "checked",
    }
    return {t for t in raw if t not in stopwords}


def regex_detect_contradiction(
    user_input: str,
    ledger: Ledger,
    *,
    lookback: int = 5,
) -> ContradictionSignal | None:
    """Heuristic contradiction detector. No LLM dependency.

    Two-stage gate:
      1. The user input must contain an explicit contradiction phrase
         (see :data:`_CONTRADICTION_PATTERNS`).
      2. The input must overlap with at least one recent load-bearing
         claim above :data:`_MIN_CONFIDENCE`.

    Both stages must pass — pure phrase detection without claim
    matching would mis-target the foundation re-check at an irrelevant
    claim. Returns ``None`` if either gate fails.
    """
    if not _looks_like_contradiction(user_input):
        return None
    candidates = _candidate_claims(ledger, lookback)
    match = _best_match(user_input, candidates)
    if match is None:
        return None
    claim, confidence = match
    return ContradictionSignal(
        contradicted_claim_id=claim.claim_id,
        user_evidence=user_input.strip(),
        confidence=round(confidence, 3),
        detected_at=datetime.now(timezone.utc),
        detector="regex",
    )


def llm_detect_contradiction(
    user_input: str,
    ledger: Ledger,
    *,
    lookback: int = 5,
) -> ContradictionSignal | None:
    """LLM-based contradiction detector. Phase 3 fallback to regex.

    Wired to a budget-tier specialist with structured output in a
    later phase. The function exists so the dispatcher contract stays
    stable.
    """
    logger.debug(
        "epistemic llm_detect_contradiction: not yet wired; using regex fallback",
    )
    return regex_detect_contradiction(user_input, ledger, lookback=lookback)


def detect_contradiction(
    user_input: str,
    ledger: Ledger,
    *,
    lookback: int = 5,
) -> ContradictionSignal | None:
    """Dispatch to the configured detector."""
    if _llm_enabled():
        try:
            return llm_detect_contradiction(user_input, ledger, lookback=lookback)
        except Exception as exc:
            logger.warning(
                "epistemic detect_contradiction: llm raised (%s); using regex",
                exc,
            )
            return regex_detect_contradiction(user_input, ledger, lookback=lookback)
    return regex_detect_contradiction(user_input, ledger, lookback=lookback)


def _llm_enabled() -> bool:
    val = os.getenv("EPISTEMIC_PUSHBACK_LLM_DETECTOR", "").strip().lower()
    return val in ("1", "true", "yes", "on")


# ── Foundation re-check protocol ────────────────────────────────────

def handle_foundation_check(
    signal: ContradictionSignal,
    ledger: Ledger,
) -> FoundationCheckResult:
    """The deterministic protocol. ONLY runs the verifier.

    Anything else requires explicit user follow-up. This function does
    not branch into "let me also check..." paths — the structural
    narrowness is the whole point.

    Side effects on FALSIFIED outcome:
      * The original claim is superseded (status flipped to
        CONTRADICTED, ``superseded_by`` set to a new replacement claim).
      * Every dependent claim (those whose evidence references the
        original claim) is also superseded.
      * The replacement claim is emitted normally — it goes through the
        realtime detectors and is persisted.
    """
    started = datetime.now(timezone.utc)
    target = ledger.by_id(signal.contradicted_claim_id)
    if target is None:
        return FoundationCheckResult(
            outcome=FoundationOutcome.UNVERIFIABLE,
            contradicted_claim_id=signal.contradicted_claim_id,
            new_evidence_excerpt="claim not in ledger",
            duration_seconds=_elapsed(started),
        )
    if target.verifying_action is None:
        return FoundationCheckResult(
            outcome=FoundationOutcome.UNVERIFIABLE,
            contradicted_claim_id=target.claim_id,
            new_evidence_excerpt="no exact-answer verifier registered for this claim",
            duration_seconds=_elapsed(started),
        )

    result = execute(target.verifying_action)
    if not result.settles:
        return FoundationCheckResult(
            outcome=FoundationOutcome.UNVERIFIABLE,
            contradicted_claim_id=target.claim_id,
            new_evidence_excerpt=(result.stdout or result.stderr or
                                  "executor returned settles=False"),
            duration_seconds=_elapsed(started),
        )

    if result.confirms:
        # Foundation holds. The user's pushback may have been mistaken
        # or aimed at a different claim — the orchestrator can surface
        # the verifier output to clarify.
        return FoundationCheckResult(
            outcome=FoundationOutcome.REVERIFIED,
            contradicted_claim_id=target.claim_id,
            new_evidence_excerpt=_truncate(result.stdout, 300),
            duration_seconds=_elapsed(started),
        )

    # Foundation falsified. Cascade-invalidate dependents and emit a
    # replacement claim flipped to VERIFIED with the new evidence.
    invalidated = _cascade_invalidate(target, result, ledger)
    return FoundationCheckResult(
        outcome=FoundationOutcome.FALSIFIED,
        contradicted_claim_id=target.claim_id,
        new_evidence_excerpt=_truncate(result.stdout, 300),
        invalidated_claim_ids=invalidated,
        duration_seconds=_elapsed(started),
    )


def _cascade_invalidate(
    target: Claim,
    result: VerifierResult,
    ledger: Ledger,
) -> tuple[str, ...]:
    """Build the falsified replacement, supersede target + dependents."""
    # Find dependents BEFORE we mutate the ledger. A dependent is a
    # claim whose evidence references the target's claim_id (kind=
    # "prior_claim" with source_ref == target.claim_id).
    dependents = _dependents_of(target.claim_id, ledger)

    replacement_statement = (
        f"{target.statement} — falsified by user pushback re-check"
    )
    new_evidence = (
        Evidence(
            kind="user_assertion",
            source_ref=f"pushback:{target.claim_id}",
            excerpt=_truncate(result.stdout, 500),
            confidence=1.0,
        ),
    )
    # The replacement carries the new (correct) status. Its statement is
    # textually a marker — the orchestrator uses ``invalidated_claim_ids``
    # to surface "X was wrong; here's the verifier output".
    replacement = Claim.new(
        task_id=target.task_id,
        agent_role=target.agent_role,
        statement=replacement_statement,
        status=VerificationStatus.VERIFIED,
        register=Register.INTERNAL,
        evidence=new_evidence,
        verifying_action=target.verifying_action,
        load_bearing=target.load_bearing,
        tags=target.tags,
        span_id=target.span_id,
    )

    # Supersede the target — emits the replacement and persists.
    ledger.supersede(claim_id=target.claim_id, replacement=replacement)

    # Cascade: every dependent gets superseded too. Their replacements
    # are bare CONTRADICTED markers (we don't try to re-derive them
    # automatically — that's the agent's job after the protocol returns).
    invalidated: list[str] = []
    for dep in dependents:
        dep_replacement = Claim.new(
            task_id=dep.task_id,
            agent_role=dep.agent_role,
            statement=f"{dep.statement} — invalidated by upstream contradiction",
            status=VerificationStatus.CONTRADICTED,
            register=Register.INTERNAL,
            evidence=(),
            load_bearing=dep.load_bearing,
            tags=dep.tags,
            span_id=dep.span_id,
        )
        ledger.supersede(claim_id=dep.claim_id, replacement=dep_replacement)
        invalidated.append(dep.claim_id)
    return tuple(invalidated)


def _dependents_of(claim_id: str, ledger: Ledger) -> list[Claim]:
    """Claims whose evidence references ``claim_id`` as a prior claim."""
    out: list[Claim] = []
    for c in ledger.all():
        if c.claim_id == claim_id:
            continue
        if c.status is VerificationStatus.CONTRADICTED:
            continue
        for ev in c.evidence:
            if ev.kind == "prior_claim" and ev.source_ref == claim_id:
                out.append(c)
                break
    return out


def _elapsed(started: datetime) -> float:
    return (datetime.now(timezone.utc) - started).total_seconds()


def _truncate(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


# ── Coordinator ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class PushbackOutcome:
    """End-to-end result of processing a user message.

    ``signal=None`` means no contradiction was detected — the message
    flows through normal handling. ``check is None and signal is not
    None`` would be a bug; the protocol always runs a foundation check
    once a signal fires.
    """

    signal: ContradictionSignal | None
    check: FoundationCheckResult | None

    @property
    def fired(self) -> bool:
        return self.signal is not None

    def as_jsonable(self) -> dict[str, Any]:
        return {
            "signal": self.signal.as_jsonable() if self.signal else None,
            "check": self.check.as_jsonable() if self.check else None,
        }


def process_user_message(
    user_input: str,
    ledger: Ledger,
    *,
    lookback: int = 5,
    persist: bool = True,
) -> PushbackOutcome:
    """Top-level entry point: detect, check, persist.

    The orchestrator (Phase 5 wiring) calls this once per inbound user
    message, BEFORE the agent generates a response. The returned
    ``PushbackOutcome`` tells the orchestrator whether to:
      * proceed normally (``not outcome.fired``),
      * surface "your pushback was right; here's the new evidence"
        (FALSIFIED outcome),
      * surface "I re-checked, the foundation holds — is your question
        about something else?" (REVERIFIED outcome),
      * surface a hedge ("I can't fully verify this; here's what I
        know") (UNVERIFIABLE outcome).
    """
    signal = detect_contradiction(user_input, ledger, lookback=lookback)
    if signal is None:
        return PushbackOutcome(signal=None, check=None)

    check = handle_foundation_check(signal, ledger)

    if persist:
        try:
            from app.epistemic.span_writer import persist_pushback_event
            persist_pushback_event(
                task_id=ledger.task_id,
                signal=signal,
                check=check,
            )
        except Exception as exc:
            logger.debug("epistemic pushback persist failed: %s", exc)

    return PushbackOutcome(signal=signal, check=check)
