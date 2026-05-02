"""Realtime bias detectors.

Each detector runs on every Claim emission via a single meta-hook
registered with :func:`app.epistemic.registry.register`. The meta-hook
isolates per-detector failures (one buggy detector cannot poison the
others) and persists matches via :mod:`app.epistemic.span_writer`.

Realtime detectors:
  * :class:`InferenceAsFactDetector` (the canonical bias)
  * :class:`RegisterConfidenceMismatchDetector` (felt-vs-stated)
  * :class:`DestructiveWithoutRecheckDetector` (irreversible-action gate)
  * :class:`RecommendationWithoutMeasurementDetector` (measure-first rule)
  * :class:`CausalLayerOverreachDetector` (Pearl L2/L3 without controlled
    evidence — Causal Hierarchy Theorem)

Performance contract: each detector's ``detect()`` call MUST run in
< 5 ms p95. The meta-hook target is < 50 ms p95 across all detectors
(see :data:`app.epistemic.CALIBRATION_HOOK_BUDGET_MS`). A detector
that violates this should be moved to post-hoc.
"""
from __future__ import annotations

import logging
import re
from typing import Iterable

from app.epistemic.biases import BIAS_LIBRARY, BiasMatch
from app.epistemic.detectors import Detector, register_realtime
from app.epistemic.grounding import factual_grounding
from app.epistemic.ledger import (
    CAUSAL_EVIDENCE_KINDS_L2,
    Claim,
    Ledger,
    PchLayer,
    Register,
    VerificationStatus,
)
from app.epistemic.registry import register as register_claim_hook

logger = logging.getLogger(__name__)


# Threshold below which felt grounding triggers the
# register_confidence_mismatch bias. See data/biases.yaml for the
# rationale (matches the affect layer's "low certainty" band).
_GROUNDING_LOW_THRESHOLD = 0.40


class InferenceAsFactDetector(Detector):
    """The canonical failure mode.

    Fires when:
      * ``claim.status == INFERRED`` (the agent did not directly verify),
      * ``claim.verifying_action is not None`` (a cheap exact-answer
        verifier is available — i.e. there's a clear "next move" the
        agent could take instead of asserting),
      * ``claim.register == DECLARATIVE`` (the agent is about to state
        the inference *as fact*, not as a hedge).

    This is the bias from the April 2026 reference incident — see
    ``crewai-team/docs/EPISTEMIC_INTEGRITY.md`` §1.
    """

    bias_id = "inference_as_fact"

    def detect(
        self,
        ledger: Ledger,
        *,
        claim: Claim | None = None,
    ) -> Iterable[BiasMatch]:
        if claim is None:
            return
        if (
            claim.status is VerificationStatus.INFERRED
            and claim.verifying_action is not None
            and claim.register is Register.DECLARATIVE
        ):
            yield BiasMatch(
                bias_id=self.bias_id,
                matched_claim_ids=(claim.claim_id,),
                severity=BIAS_LIBRARY.get(self.bias_id).severity,
                detail={
                    "verifier_tool": claim.verifying_action.tool,
                    "verifier_seconds": claim.verifying_action.estimated_seconds,
                    "register": claim.register.value,
                },
            )


class RegisterConfidenceMismatchDetector(Detector):
    """Felt-vs-stated calibration mismatch.

    Fires when:
      * ``claim.register == DECLARATIVE`` (high-confidence framing),
      * ``claim.load_bearing`` (it matters downstream),
      * the affective layer reports ``factual_grounding < 0.40``.

    If grounding is unavailable (no provider wired, e.g. during tests
    or before Phase 5's affect integration), the detector silently
    skips. ``None`` is *not* treated as "low grounding" — that would
    fire the bias on every declarative load-bearing claim.
    """

    bias_id = "register_confidence_mismatch"

    def detect(
        self,
        ledger: Ledger,
        *,
        claim: Claim | None = None,
    ) -> Iterable[BiasMatch]:
        if claim is None:
            return
        if claim.register is not Register.DECLARATIVE:
            return
        if not claim.load_bearing:
            return
        grounding = factual_grounding()
        if grounding is None:
            return  # signal not available — skip, don't fire spuriously
        if grounding < _GROUNDING_LOW_THRESHOLD:
            yield BiasMatch(
                bias_id=self.bias_id,
                matched_claim_ids=(claim.claim_id,),
                severity=BIAS_LIBRARY.get(self.bias_id).severity,
                detail={
                    "factual_grounding": round(grounding, 3),
                    "threshold": _GROUNDING_LOW_THRESHOLD,
                    "register": claim.register.value,
                },
            )


# Patterns that mark a claim's *statement* as a destructive recommendation.
# Word-boundary anchors keep "rm" from matching inside "permission".
_DESTRUCTIVE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\brm\s+-rf\b", re.IGNORECASE),
    re.compile(r"\brmdir\b", re.IGNORECASE),
    re.compile(r"\bgit\s+reset\s+--hard\b", re.IGNORECASE),
    re.compile(r"\bgit\s+clean\s+-[a-z]*f", re.IGNORECASE),
    re.compile(r"\bforce[-\s]push\b", re.IGNORECASE),
    re.compile(r"\bDROP\s+(?:TABLE|DATABASE|SCHEMA|INDEX)\b", re.IGNORECASE),
    re.compile(r"\bTRUNCATE\b", re.IGNORECASE),
    re.compile(r"\bdelete\s+(?:all|the\s+entire|every|the\s+whole)\b", re.IGNORECASE),
    re.compile(r"\bwipe\s+(?:the|all)\b", re.IGNORECASE),
    re.compile(r"\bdrop\s+(?:the|all)\s+rows?\b", re.IGNORECASE),
)


def _matches_destructive(statement: str) -> bool:
    return any(p.search(statement) for p in _DESTRUCTIVE_PATTERNS)


class DestructiveWithoutRecheckDetector(Detector):
    """Irreversible-action gate.

    Fires when the agent is about to recommend a destructive action
    (rm, DROP, force-push, etc.) AND the load-bearing diagnosis still
    contains unverified claims. The asymmetric recovery cost (a wrong
    rm or DROP is irreversible) makes this CRITICAL severity — Phase 7
    flips it to blocking-mode and routes the claim through
    :mod:`app.epistemic.peer_review`.

    Detection is statement-pattern based (see ``_DESTRUCTIVE_PATTERNS``)
    plus an explicit tag override (``"destructive_recommendation"`` in
    :attr:`Claim.tags`). The tag override lets agents that know they're
    proposing a destructive action surface that fact even if the
    statement doesn't match a pattern (e.g. a custom delete tool).
    """

    bias_id = "destructive_without_recheck"

    def detect(
        self,
        ledger: Ledger,
        *,
        claim: Claim | None = None,
    ) -> Iterable[BiasMatch]:
        if claim is None:
            return
        is_destructive = (
            "destructive_recommendation" in claim.tags
            or _matches_destructive(claim.statement)
        )
        if not is_destructive:
            return
        unverified = ledger.unverified_load_bearing()
        if not unverified:
            return
        # Match-claim ids: every unverified-load-bearing claim, plus the
        # destructive claim itself at the tail. The post-mortem walker
        # uses the order to render "X unverified claims, then this".
        matched_ids = tuple(c.claim_id for c in unverified) + (claim.claim_id,)
        yield BiasMatch(
            bias_id=self.bias_id,
            matched_claim_ids=matched_ids,
            severity=BIAS_LIBRARY.get(self.bias_id).severity,
            detail={
                "destructive_claim_id": claim.claim_id,
                "unverified_count": len(unverified),
            },
        )


# Patterns that mark a claim's *statement* as an optimization
# recommendation. Conservative — these are the explicit shapes; agents
# can also opt-in via the ``"optimization_recommendation"`` tag.
_RECOMMENDATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:recommend|suggest|propose|should)\b[^.!?]*?"
        r"\b(?:optimi[zs]e|improve|reduce|cut|speed\s*up|switch\s+to|use\s+\S+\s+instead)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:would|will)\b[^.!?]*?"
        r"\b(?:improve|optimi[zs]e|reduce|cut|speed\s*up)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:swap|replace)\b[^.!?]*?\bwith\b",
        re.IGNORECASE,
    ),
)


def _matches_recommendation(statement: str) -> bool:
    return any(p.search(statement) for p in _RECOMMENDATION_PATTERNS)


# Tools that produce a measurement. Evidence whose tool head is in
# this set counts as "the agent measured before recommending".
_MEASUREMENT_TOOLS: frozenset[str] = frozenset({
    "perf_eval", "benchmark", "bench", "wrk", "ab",
    "psql", "chroma_count", "kb_count",
    "stat", "wc", "du",
    "git", "git_diff_stat",
    "control_plane.span_metrics",
    "time", "timer", "profile", "measure", "timeit",
})


def _tool_head_from_excerpt(excerpt: str) -> str:
    """Pull the tool head from an evidence excerpt of the form
    ``"$ tool args\\noutput"`` (the format produced by
    :func:`app.epistemic.ledger._format_invocation`)."""
    if not excerpt.startswith("$ "):
        return ""
    first_line = excerpt.split("\n", 1)[0]
    parts = first_line[2:].split(maxsplit=1)
    return parts[0] if parts else ""


class RecommendationWithoutMeasurementDetector(Detector):
    """Measure-first rule.

    Fires when the agent makes an optimization recommendation
    (statement matches :data:`_RECOMMENDATION_PATTERNS` or claim has
    the ``"optimization_recommendation"`` tag) but no evidence in the
    claim came from a tool in :data:`_MEASUREMENT_TOOLS`.

    Seed: the user's April 2026 token-economy episode (memory
    feedback_verify_before_recommending) — three "wins" recommended
    without measurement, all falsified empirically.
    """

    bias_id = "recommendation_without_measurement"

    def detect(
        self,
        ledger: Ledger,
        *,
        claim: Claim | None = None,
    ) -> Iterable[BiasMatch]:
        if claim is None:
            return
        is_recommendation = (
            "optimization_recommendation" in claim.tags
            or _matches_recommendation(claim.statement)
        )
        if not is_recommendation:
            return
        for ev in claim.evidence:
            if ev.kind != "tool_call":
                continue
            head = _tool_head_from_excerpt(ev.excerpt)
            if head in _MEASUREMENT_TOOLS:
                return  # measurement present — no bias
        yield BiasMatch(
            bias_id=self.bias_id,
            matched_claim_ids=(claim.claim_id,),
            severity=BIAS_LIBRARY.get(self.bias_id).severity,
            detail={
                "evidence_tool_count": sum(
                    1 for e in claim.evidence if e.kind == "tool_call"
                ),
                "reason": "no measurement evidence in claim",
            },
        )


# Patterns that mark a statement as a causal/interventional claim
# ("doing X changes Y" — Pearl L2). Conservative on purpose: false
# positives here mean the gate fires on observational claims, which
# is more annoying than dangerous. Word boundaries keep "improved"
# from matching inside "unimproved".
_L2_KEYWORDS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:improved|caused|reduced|increased|decreased|sped\s*up|slowed\s*down)\b",
        re.IGNORECASE,
    ),
    # "made the build faster" / "made X better" — allow multiple words
    # between "made" and the adjective so multi-word objects match.
    re.compile(
        r"\bmade\s+(?:\w+\s+){1,5}?(?:better|worse|faster|slower)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bproduced\s+(?:meaningful|measurable|real|the)\s+(?:improvements?|gains?|wins?|deltas?)\b",
        re.IGNORECASE,
    ),
)

# Patterns that mark a statement as a counterfactual (Pearl L3).
_L3_KEYWORDS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bwould\s+have\s+\w+", re.IGNORECASE),
    re.compile(r"\bif\s+\w+\s+had\b", re.IGNORECASE),
    re.compile(r"\bcounterfactually\b", re.IGNORECASE),
    re.compile(r"\bhad\s+we\s+\w+", re.IGNORECASE),
)


def _infer_pch_layer(statement: str) -> PchLayer:
    """Infer the Pearl Causal Hierarchy layer from a claim's text.

    Order matters: L3 wins over L2 (counterfactual is strictly more
    expressive than interventional). L1 is the default.
    """
    if any(p.search(statement) for p in _L3_KEYWORDS):
        return "L3"
    if any(p.search(statement) for p in _L2_KEYWORDS):
        return "L2"
    return "L1"


def _has_l2_grade_evidence(claim: Claim) -> bool:
    """Does this claim cite L2-grade (controlled-intervention) evidence?

    True if any tag in :data:`Claim.causal_evidence_kinds` is in
    :data:`CAUSAL_EVIDENCE_KINDS_L2`. The detector treats this as
    the explicit "I ran the experiment" affirmation; statement-level
    heuristics never grant L2 evidence on their own.
    """
    return any(
        kind in CAUSAL_EVIDENCE_KINDS_L2
        for kind in claim.causal_evidence_kinds
    )


class CausalLayerOverreachDetector(Detector):
    """Pearl Causal Hierarchy — claim asserts L2/L3 with no controlled evidence.

    The Causal Hierarchy Theorem says you cannot infer layer i from
    layer i-1 alone: L2 ("doing X changes Y") cannot be derived from
    L1 ("X correlates with Y") without a controlled intervention.
    This detector fires when an agent's claim is causal in shape
    (interventional or counterfactual) but the claim cites no
    L2-grade evidence — a controlled experiment, ablation, or
    do-intervention.

    Inferred layer = explicit ``claim.pch_layer`` if set, else the
    layer inferred from the statement text. Fires when inferred layer
    is L2 or L3 and :func:`_has_l2_grade_evidence` is False.

    Seed scenario: the Self-Improver narrative emitting "yesterday's
    experiments produced K meaningful improvements" without attaching
    the experiment_runner span as evidence. The narrative path emits
    explicit ``causal_evidence_kinds=("controlled_experiment",)`` to
    avoid the false positive.
    """

    bias_id = "causal_layer_overreach"

    def detect(
        self,
        ledger: Ledger,
        *,
        claim: Claim | None = None,
    ) -> Iterable[BiasMatch]:
        if claim is None:
            return
        inferred = claim.pch_layer or _infer_pch_layer(claim.statement)
        if inferred == "L1":
            return
        if _has_l2_grade_evidence(claim):
            return
        yield BiasMatch(
            bias_id=self.bias_id,
            matched_claim_ids=(claim.claim_id,),
            severity=BIAS_LIBRARY.get(self.bias_id).severity,
            detail={
                "inferred_layer": inferred,
                "agent_role": claim.agent_role,
                "explicit_layer": claim.pch_layer is not None,
                "reason": "no controlled-intervention evidence on the claim",
            },
        )


# ── Detector instantiation & registration ──────────────────────────
# Done at import time. Importing ``app.epistemic.detectors.realtime``
# attaches the meta-hook to the claim ledger; that's the explicit
# bootstrap step. ``app.epistemic`` does this import in its __init__
# so a plain ``from app.epistemic import Ledger`` is enough.

INFERENCE_AS_FACT = register_realtime(InferenceAsFactDetector())
REGISTER_CONFIDENCE_MISMATCH = register_realtime(RegisterConfidenceMismatchDetector())
DESTRUCTIVE_WITHOUT_RECHECK = register_realtime(DestructiveWithoutRecheckDetector())
RECOMMENDATION_WITHOUT_MEASUREMENT = register_realtime(RecommendationWithoutMeasurementDetector())
CAUSAL_LAYER_OVERREACH = register_realtime(CausalLayerOverreachDetector())


@register_claim_hook
def _realtime_meta_hook(claim: Claim, ledger: Ledger) -> None:
    """Run every realtime detector against ``claim``, persist matches.

    Per-detector failures are isolated: a detector that raises is logged
    at WARNING and skipped — its bias just doesn't fire for this claim.
    The other detectors continue.
    """
    from app.epistemic.detectors import realtime_detectors

    matches: list[BiasMatch] = []
    for detector in realtime_detectors():
        try:
            matches.extend(detector.detect(ledger, claim=claim))
        except Exception as exc:
            logger.warning(
                "epistemic realtime detector %s raised on claim %s: %s",
                detector.__class__.__name__, claim.claim_id, exc,
            )
            continue

    if matches:
        try:
            from app.epistemic.span_writer import persist_bias_matches
            persist_bias_matches(claim_id=claim.claim_id,
                                 task_id=claim.task_id,
                                 matches=matches)
        except Exception as exc:
            logger.debug(
                "epistemic realtime: persist_bias_matches failed: %s", exc,
            )

        # Notify match observers (e.g. affect_bridge for cognitive_failure
        # salience emission). Each observer is isolated — one bad observer
        # doesn't poison the rest.
        from app.epistemic.detectors import match_observers
        for observer in match_observers():
            try:
                observer(matches, claim, ledger)
            except Exception as exc:
                logger.warning(
                    "epistemic realtime: match observer %r raised: %s",
                    getattr(observer, "__qualname__", observer), exc,
                )
