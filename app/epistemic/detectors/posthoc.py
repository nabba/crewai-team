"""Post-hoc bias detectors.

Run inside :func:`app.epistemic.postmortem.synthesize_report` — never on
the user-facing path. Tolerate complexity the realtime detectors cannot
(full ledger graph traversal, cross-event correlations, pushback-trace
analysis) but in exchange must finish within
:data:`POSTHOC_DETECTOR_BUDGET_S` for any single task.

Phase 4 ships four post-hoc detectors:

* :class:`DefendingPeripheryDetector` — after a UNVERIFIABLE pushback,
  did the agent expand the investigation?
* :class:`CoherenceBiasDetector` — chain of inferred claims terminating
  at a declarative load-bearing recommendation.
* :class:`ToolLazinessDetector` — multi-step inference where a cheap
  verifier was available.
* :class:`AnomalyDismissalDetector` — claim retained despite
  contradicting evidence.

The realtime meta-hook does NOT invoke these. They are called only via
:func:`posthoc_detectors` from the post-mortem pipeline.
"""
from __future__ import annotations

import logging
from typing import Iterable, Mapping

from app.epistemic.biases import BIAS_LIBRARY, BiasMatch
from app.epistemic.detectors import Detector, register_posthoc
from app.epistemic.ledger import (
    Claim,
    Ledger,
    Register,
    VerificationStatus,
)

logger = logging.getLogger(__name__)


# Threshold below which a piece of evidence is treated as
# "contradicting" the claim it supports — used by AnomalyDismissalDetector.
_CONTRADICTING_EVIDENCE_CONFIDENCE = 0.30

# After a UNVERIFIABLE pushback, how many subsequent claims constitute
# "investigation expansion"? Conservative default; reference panel
# scenarios pin the exact threshold.
_DEFENDING_PERIPHERY_MIN_CLAIMS = 3

# Coherence bias: minimum chain length of unverified claims feeding into
# a declarative load-bearing terminal.
_COHERENCE_MIN_CHAIN = 3

# Tool laziness: a verifier counts as "cheap" below this cost in seconds.
_TOOL_LAZINESS_MAX_SECONDS = 5.0
# A claim with at least this many evidence rows is treated as having
# done multi-step reasoning (cheap proxy for "the agent worked harder
# than it needed to").
_TOOL_LAZINESS_MIN_EVIDENCE = 3


# ── DefendingPeripheryDetector ──────────────────────────────────────

class DefendingPeripheryDetector(Detector):
    """After a UNVERIFIABLE pushback, did the agent keep going?

    The April 2026 reference incident shape: user pushed back on
    "/etc/foo is not a symlink"; the protocol returned UNVERIFIABLE
    (executor not yet wired); the agent then investigated mount tables
    and devcontainer.json instead of stopping to ask the user. Here we
    catch the same shape post-hoc: any UNVERIFIABLE outcome followed by
    >= 3 subsequent claims fires this bias.

    Pushback events are passed in via ``pushback_events`` rather than
    re-queried from the DB so the post-mortem pipeline owns the read
    boundary.
    """

    bias_id = "defending_periphery"

    def __init__(self, *, pushback_events: list[Mapping] | None = None) -> None:
        # ``pushback_events`` is set per-task by the post-mortem pipeline.
        self._pushback_events: list[Mapping] = list(pushback_events or [])

    def with_events(
        self, pushback_events: list[Mapping],
    ) -> "DefendingPeripheryDetector":
        """Return a fresh detector bound to a specific event list.

        The post-mortem pipeline calls this per-task so detector
        instances stay shareable across tasks.
        """
        return DefendingPeripheryDetector(pushback_events=pushback_events)

    def detect(
        self,
        ledger: Ledger,
        *,
        claim: Claim | None = None,
    ) -> Iterable[BiasMatch]:
        if claim is not None:
            return  # post-hoc only; realtime path is a no-op
        all_claims = ledger.all()
        for event in self._pushback_events:
            if event.get("outcome") != "unverifiable":
                continue
            target_id = event.get("contradicted_claim_id", "")
            detected_at = event.get("detected_at")
            if not target_id or not detected_at:
                continue
            after = self._claims_after(all_claims, detected_at)
            if len(after) < _DEFENDING_PERIPHERY_MIN_CLAIMS:
                continue
            matched = (target_id,) + tuple(c.claim_id for c in after[:_DEFENDING_PERIPHERY_MIN_CLAIMS])
            yield BiasMatch(
                bias_id=self.bias_id,
                matched_claim_ids=matched,
                severity=BIAS_LIBRARY.get(self.bias_id).severity,
                detail={
                    "post_pushback_claim_count": len(after),
                    "pushback_outcome": "unverifiable",
                    "contradicted_claim_id": target_id,
                },
            )

    @staticmethod
    def _claims_after(all_claims: list[Claim], iso_timestamp: str) -> list[Claim]:
        """Filter claims whose ``created_at`` is strictly after ``iso_timestamp``.

        Comparing ISO strings lexicographically works because they're
        all UTC and zero-padded — same format as `datetime.isoformat()`.
        """
        return [c for c in all_claims if c.created_at.isoformat() > iso_timestamp]


# ── CoherenceBiasDetector ───────────────────────────────────────────

class CoherenceBiasDetector(Detector):
    """Narrative-too-clean: a chain of inferred claims feeding into a
    declarative load-bearing recommendation.

    Builds a dependency graph from ``Evidence(kind="prior_claim")``
    edges, then finds any path of length >= 3 ending at a DECLARATIVE
    + load_bearing claim where every node is INFERRED.
    """

    bias_id = "coherence_bias"

    def detect(
        self,
        ledger: Ledger,
        *,
        claim: Claim | None = None,
    ) -> Iterable[BiasMatch]:
        if claim is not None:
            return
        claims_by_id: dict[str, Claim] = {c.claim_id: c for c in ledger.all()}

        # Build child→parents adjacency: child claim references parent
        # claims as prior_claim evidence. We then find all-INFERRED
        # paths terminating at a DECLARATIVE+load_bearing claim.
        parents_of: dict[str, list[str]] = {}
        for c in claims_by_id.values():
            parents = [
                e.source_ref
                for e in c.evidence
                if e.kind == "prior_claim" and e.source_ref in claims_by_id
            ]
            if parents:
                parents_of[c.claim_id] = parents

        emitted_chains: set[tuple[str, ...]] = set()
        for terminal in claims_by_id.values():
            if not (
                terminal.register is Register.DECLARATIVE
                and terminal.load_bearing
            ):
                continue
            for chain in self._chains_to(terminal, claims_by_id, parents_of):
                if len(chain) < _COHERENCE_MIN_CHAIN:
                    continue
                if not all(
                    claims_by_id[cid].status is VerificationStatus.INFERRED
                    for cid in chain
                ):
                    continue
                key = tuple(chain)
                if key in emitted_chains:
                    continue
                emitted_chains.add(key)
                yield BiasMatch(
                    bias_id=self.bias_id,
                    matched_claim_ids=tuple(chain),
                    severity=BIAS_LIBRARY.get(self.bias_id).severity,
                    detail={"chain_length": len(chain)},
                )

    @staticmethod
    def _chains_to(
        terminal: Claim,
        claims_by_id: dict[str, Claim],
        parents_of: dict[str, list[str]],
    ) -> Iterable[list[str]]:
        """Yield every directed chain ending at ``terminal``.

        Iterative DFS with cycle protection. Each yielded chain is a
        list of claim_ids ordered from earliest ancestor to terminal.
        """
        # Stack of (current_id, path_from_root)
        stack: list[tuple[str, list[str]]] = [(terminal.claim_id, [terminal.claim_id])]
        while stack:
            current_id, path = stack.pop()
            parents = parents_of.get(current_id, [])
            if not parents:
                # Path is complete — reverse so root is first.
                yield list(reversed(path))
                continue
            for p in parents:
                if p in path:
                    continue  # cycle guard
                stack.append((p, path + [p]))


# ── ToolLazinessDetector ────────────────────────────────────────────

class ToolLazinessDetector(Detector):
    """Multi-step inference where a cheap verifier was available.

    Flag claims that:
      * are INFERRED + load_bearing,
      * have a verifying_action with estimated_seconds < 5.0,
      * have >= 3 evidence rows (proxy for multi-step reasoning).

    The bias name is borrowed from the reference incident: the agent
    burned several minutes on indirect inference when ``readlink`` was
    a 0.5-second exact-answer command.
    """

    bias_id = "tool_laziness"

    def detect(
        self,
        ledger: Ledger,
        *,
        claim: Claim | None = None,
    ) -> Iterable[BiasMatch]:
        if claim is not None:
            return
        for c in ledger.all():
            if c.status is not VerificationStatus.INFERRED:
                continue
            if not c.load_bearing:
                continue
            if c.verifying_action is None:
                continue
            if c.verifying_action.estimated_seconds >= _TOOL_LAZINESS_MAX_SECONDS:
                continue
            if len(c.evidence) < _TOOL_LAZINESS_MIN_EVIDENCE:
                continue
            yield BiasMatch(
                bias_id=self.bias_id,
                matched_claim_ids=(c.claim_id,),
                severity=BIAS_LIBRARY.get(self.bias_id).severity,
                detail={
                    "verifier_seconds": c.verifying_action.estimated_seconds,
                    "evidence_count": len(c.evidence),
                    "verifier_tool": c.verifying_action.tool,
                },
            )


# ── AnomalyDismissalDetector ────────────────────────────────────────

class AnomalyDismissalDetector(Detector):
    """Claim retained despite contradicting evidence.

    A claim has at least one evidence row with confidence below
    :data:`_CONTRADICTING_EVIDENCE_CONFIDENCE` AND the claim's status
    is not CONTRADICTED. The agent observed disconfirming evidence and
    chose to keep the claim — that's the "explained it away" pattern.

    Conservative threshold: most evidence is in the 0.5–0.9 band, so
    anything < 0.3 is a real outlier worth surfacing.
    """

    bias_id = "anomaly_dismissal"

    def detect(
        self,
        ledger: Ledger,
        *,
        claim: Claim | None = None,
    ) -> Iterable[BiasMatch]:
        if claim is not None:
            return
        for c in ledger.all():
            if c.status is VerificationStatus.CONTRADICTED:
                continue
            contradicting = [
                e for e in c.evidence
                if e.confidence < _CONTRADICTING_EVIDENCE_CONFIDENCE
            ]
            if not contradicting:
                continue
            yield BiasMatch(
                bias_id=self.bias_id,
                matched_claim_ids=(c.claim_id,),
                severity=BIAS_LIBRARY.get(self.bias_id).severity,
                detail={
                    "contradicting_evidence_count": len(contradicting),
                    "min_confidence": min(e.confidence for e in contradicting),
                },
            )


# ── Registration ────────────────────────────────────────────────────
# Done at import time. The post-hoc detector list is consulted by
# :func:`app.epistemic.postmortem.synthesize_report` — never by the
# realtime meta-hook. Re-registration is a no-op (set semantics).

DEFENDING_PERIPHERY = register_posthoc(DefendingPeripheryDetector())
COHERENCE_BIAS = register_posthoc(CoherenceBiasDetector())
TOOL_LAZINESS = register_posthoc(ToolLazinessDetector())
ANOMALY_DISMISSAL = register_posthoc(AnomalyDismissalDetector())
