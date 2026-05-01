"""Post-mortem pipeline.

Aviation-style incident analysis: structured reconstruction of a task's
epistemic failures and concrete behavioral changes, fed back into the
existing Self-Improver 6-stage loop.

Flow:

  task_id → load ledger + bias matches + pushback events
          → run all post-hoc detectors
          → IncidentReport (timeline, root cause, enabling factors,
                            missed signals, behavioral changes)
          → persist (epistemic_incidents)
          → emit_to_self_improver (LearningGap with bias_id evidence)

The Self-Improver integration is the seamless self-evolution coupling
the user asked for: incidents become first-class learning gaps that
the existing pipeline (Gap Detector → Novelty Gate → Learner →
Integrator → Evaluator → Consolidator) consumes without modification.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from app.epistemic.biases import BIAS_LIBRARY, BiasMatch, Severity, severity_rank
from app.epistemic.detectors import posthoc_detectors
from app.epistemic.ledger import Claim, Ledger
from app.epistemic.span_writer import (
    list_bias_matches_for_task,
    list_pushback_events_for_task,
    load_ledger_for_task,
)

logger = logging.getLogger(__name__)


# ── Types ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TimelineEntry:
    """One event in the reconstructed task timeline.

    Ordered by ``at``. Kind is one of:
      * "claim_emit" — a claim was emitted
      * "claim_supersede" — a claim was contradicted
      * "bias_match" — a realtime detector fired
      * "pushback" — user contradiction event
    """

    at: datetime
    kind: str
    summary: str
    claim_id: str | None = None
    bias_id: str | None = None
    severity: Severity | None = None

    def as_jsonable(self) -> dict[str, Any]:
        return {
            "at": self.at.isoformat(),
            "kind": self.kind,
            "summary": self.summary,
            "claim_id": self.claim_id,
            "bias_id": self.bias_id,
            "severity": self.severity.value if self.severity else None,
        }


@dataclass(frozen=True)
class BehavioralChange:
    """A concrete change derived from one or more bias matches.

    Three kinds in Phase 4:

    * ``"verifier_registry_addition"`` — propose adding a new verifier
      shape to the YAML registry (CODEOWNERS PR).
    * ``"feedback_memory_entry"`` — write a feedback memory entry
      (auto-applied by the existing feedback pipeline).
    * ``"ledger_pattern_warning"`` — surface a pattern to the
      operator via the bias library description (no code change).
    """

    kind: str
    target: str
    body: str
    proposed_by: str = "epistemic_postmortem"

    def as_jsonable(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "target": self.target,
            "body": self.body,
            "proposed_by": self.proposed_by,
        }


@dataclass(frozen=True)
class IncidentReport:
    incident_id: str
    task_id: str
    timeline: tuple[TimelineEntry, ...]
    root_cause: BiasMatch
    enabling_factors: tuple[BiasMatch, ...] = ()
    missed_signals: tuple[str, ...] = ()
    behavioral_changes: tuple[BehavioralChange, ...] = ()
    cost: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def severity(self) -> Severity:
        return self.root_cause.severity

    def as_jsonable(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "task_id": self.task_id,
            "timeline": [t.as_jsonable() for t in self.timeline],
            "root_cause": self.root_cause.as_jsonable(),
            "enabling_factors": [f.as_jsonable() for f in self.enabling_factors],
            "missed_signals": list(self.missed_signals),
            "behavioral_changes": [bc.as_jsonable() for bc in self.behavioral_changes],
            "cost": dict(self.cost),
            "severity": self.severity.value,
            "created_at": self.created_at.isoformat(),
        }


# ── Synthesis ───────────────────────────────────────────────────────

def synthesize_report(*, task_id: str) -> IncidentReport | None:
    """Run all post-hoc detectors, gather realtime matches, and build a
    structured report. Returns ``None`` if no biases fired — there's
    nothing to learn from.

    The function is pure with respect to the database except for the
    reads it does. Persistence is the caller's job (see
    :func:`persist_and_emit`).
    """
    ledger = load_ledger_for_task(task_id)
    realtime_rows = list_bias_matches_for_task(task_id)
    pushback_events = list_pushback_events_for_task(task_id)

    realtime_matches = [_match_from_row(r) for r in realtime_rows]
    posthoc_matches = list(_run_posthoc(ledger, pushback_events))
    all_matches = realtime_matches + posthoc_matches

    if not all_matches:
        return None

    timeline = _build_timeline(ledger, realtime_rows, pushback_events)
    root_cause, enabling = _classify(all_matches, timeline)
    behavioral_changes = _derive_changes(all_matches, ledger)
    missed_signals = _missed_signals(all_matches, ledger)

    return IncidentReport(
        incident_id=f"inc_{task_id[:8]}_{datetime.now(timezone.utc):%Y%m%d%H%M%S}_{uuid4().hex[:6]}",
        task_id=task_id,
        timeline=tuple(timeline),
        root_cause=root_cause,
        enabling_factors=tuple(enabling),
        missed_signals=tuple(missed_signals),
        behavioral_changes=tuple(behavioral_changes),
        cost=_cost_summary(realtime_rows, pushback_events),
    )


def _run_posthoc(ledger: Ledger, pushback_events: list[dict]):
    """Execute every registered post-hoc detector with the pushback
    context where applicable."""
    from app.epistemic.detectors.posthoc import DefendingPeripheryDetector

    for detector in posthoc_detectors():
        try:
            # DefendingPeripheryDetector needs the per-task pushback events.
            if isinstance(detector, DefendingPeripheryDetector):
                bound = detector.with_events(pushback_events)
                yield from bound.detect(ledger)
            else:
                yield from detector.detect(ledger)
        except Exception as exc:
            logger.warning(
                "epistemic posthoc detector %s raised: %s",
                detector.__class__.__name__, exc,
            )


def _match_from_row(row: dict) -> BiasMatch:
    """Realtime DB rows → in-memory BiasMatch."""
    try:
        severity = Severity(row["severity"])
    except ValueError:
        severity = Severity.MEDIUM
    return BiasMatch(
        bias_id=row["bias_id"],
        matched_claim_ids=tuple(row.get("matched_claim_ids", []) or []),
        severity=severity,
        detail=row.get("detail", {}) or {},
    )


def _classify(
    all_matches: list[BiasMatch],
    timeline: list[TimelineEntry],
) -> tuple[BiasMatch, list[BiasMatch]]:
    """Pick a root cause and the rest become enabling factors.

    Strategy: highest severity wins; ties broken by earliest timestamp
    in the timeline. This is conservative — a CRITICAL bias that fired
    after several HIGH biases is still the root cause because its
    severity dominates.
    """
    by_severity = sorted(
        all_matches,
        key=lambda m: (-severity_rank(m.severity), _earliest_at(m, timeline)),
    )
    root = by_severity[0]
    enabling = by_severity[1:]
    return root, enabling


def _earliest_at(match: BiasMatch, timeline: list[TimelineEntry]) -> str:
    """Return the ISO timestamp of the earliest timeline entry that
    references this match by bias_id. Used as a tiebreaker."""
    for t in timeline:
        if t.bias_id == match.bias_id:
            return t.at.isoformat()
    return ""


def _build_timeline(
    ledger: Ledger,
    realtime_rows: list[dict],
    pushback_events: list[dict],
) -> list[TimelineEntry]:
    entries: list[TimelineEntry] = []

    for c in ledger.all():
        entries.append(TimelineEntry(
            at=c.created_at,
            kind="claim_emit",
            summary=f"[{c.agent_role}] {c.statement[:200]}",
            claim_id=c.claim_id,
        ))
        if c.superseded_by:
            entries.append(TimelineEntry(
                at=c.created_at,
                kind="claim_supersede",
                summary=f"superseded by {c.superseded_by}",
                claim_id=c.claim_id,
            ))

    for row in realtime_rows:
        try:
            sev = Severity(row.get("severity", "medium"))
        except ValueError:
            sev = Severity.MEDIUM
        at = _parse_iso(row.get("detected_at"))
        entries.append(TimelineEntry(
            at=at,
            kind="bias_match",
            summary=f"{row['bias_id']} fired ({sev.value})",
            claim_id=row.get("claim_id"),
            bias_id=row["bias_id"],
            severity=sev,
        ))

    for event in pushback_events:
        at = _parse_iso(event.get("detected_at"))
        outcome = event.get("outcome", "?")
        entries.append(TimelineEntry(
            at=at,
            kind="pushback",
            summary=f"user contradicted → {outcome}",
            claim_id=event.get("contradicted_claim_id"),
        ))

    entries.sort(key=lambda e: e.at)
    return entries


def _derive_changes(
    matches: list[BiasMatch],
    ledger: Ledger,
) -> list[BehavioralChange]:
    """One BehavioralChange per *kind* of match (deduped by bias_id)."""
    changes: list[BehavioralChange] = []
    seen: set[str] = set()
    for m in matches:
        if m.bias_id in seen:
            continue
        seen.add(m.bias_id)
        change = _change_for(m, ledger)
        if change is not None:
            changes.append(change)
    return changes


def _change_for(match: BiasMatch, ledger: Ledger) -> BehavioralChange | None:
    """Map a bias_id to a concrete proposed change.

    The mapping is deliberately narrow — Phase 4 ships one per bias.
    Future phases (or the Self-Improver's Learner stage) can expand
    these into multi-action proposals.
    """
    bid = match.bias_id

    if bid == "inference_as_fact":
        return BehavioralChange(
            kind="feedback_memory_entry",
            target="agent_register_discipline",
            body=(
                "Match phrasing to evidence. When ledger.status=inferred "
                "and a verifier is available, either run the verifier or "
                "downshift register to hedged."
            ),
        )
    if bid == "register_confidence_mismatch":
        return BehavioralChange(
            kind="feedback_memory_entry",
            target="agent_register_discipline",
            body=(
                "When affective interoception reports factual_grounding < 0.40, "
                "downshift load-bearing claims to hedged register before delivery."
            ),
        )
    if bid == "destructive_without_recheck":
        return BehavioralChange(
            kind="feedback_memory_entry",
            target="agent_destructive_recommendations",
            body=(
                "Before recommending any destructive action, every load-bearing "
                "claim in the diagnosis must be VERIFIED. If any are INFERRED, "
                "run the verifiers first or escalate to peer review."
            ),
        )
    if bid == "recommendation_without_measurement":
        return BehavioralChange(
            kind="feedback_memory_entry",
            target="agent_optimization_discipline",
            body=(
                "Every optimization recommendation must include evidence "
                "from a measurement tool (benchmark, psql, profile, …). "
                "No exceptions for 'obvious' wins — the April 2026 token-economy "
                "incident was three obvious wins, all empirically false."
            ),
        )
    if bid == "defending_periphery":
        return BehavioralChange(
            kind="feedback_memory_entry",
            target="agent_pushback_response",
            body=(
                "When user pushback fires UNVERIFIABLE foundation re-check, "
                "stop and ask the user. Do not investigate adjacent details — "
                "that is the canonical 'defending the periphery' anti-pattern."
            ),
        )
    if bid == "coherence_bias":
        return BehavioralChange(
            kind="feedback_memory_entry",
            target="agent_chain_caution",
            body=(
                "When ≥3 inferred load-bearing claims chain into a "
                "declarative recommendation, downshift the recommendation's "
                "register or break the chain by verifying at least one link."
            ),
        )
    if bid == "tool_laziness":
        verifier = match.detail.get("verifier_tool")
        return BehavioralChange(
            kind="feedback_memory_entry",
            target="agent_verifier_preference",
            body=(
                f"When a cheap verifier ({verifier}) is available "
                "(estimated_seconds < 5), prefer running it over multi-step "
                "inference. 3 seconds of tool call < 3 minutes of indirect reasoning."
            ),
        )
    if bid == "anomaly_dismissal":
        return BehavioralChange(
            kind="feedback_memory_entry",
            target="agent_anomaly_handling",
            body=(
                "When a piece of evidence with confidence < 0.30 attaches to "
                "a non-CONTRADICTED claim, treat it as falsification candidate "
                "first; only retain the claim if the anomaly is independently explained."
            ),
        )
    return None


def _missed_signals(
    matches: list[BiasMatch],
    ledger: Ledger,
) -> list[str]:
    """Plain-English notes about what was in the ledger but ignored.

    Phase 4 ships a small set of high-signal patterns; Phase 5+ can
    extend with cross-task pattern matching from the Transfer Insight
    Layer.
    """
    out: list[str] = []
    unverified = ledger.unverified_load_bearing()
    if unverified:
        out.append(
            f"{len(unverified)} load-bearing claim(s) remained unverified "
            "at incident time"
        )
    by_bias = {m.bias_id for m in matches}
    if "inference_as_fact" in by_bias and "tool_laziness" in by_bias:
        out.append(
            "inference_as_fact + tool_laziness co-fired — agent burned "
            "reasoning cycles when a cheap verifier was sitting in the registry"
        )
    return out


def _cost_summary(realtime_rows: list[dict], pushback_events: list[dict]) -> dict[str, Any]:
    return {
        "realtime_match_count": len(realtime_rows),
        "pushback_event_count": len(pushback_events),
    }


def _parse_iso(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return datetime.now(timezone.utc)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    return datetime.now(timezone.utc)


# ── Self-Improver integration ───────────────────────────────────────

# Severity → signal_strength for the LearningGap. Tuned so HIGH
# biases (the most common shape) sit comfortably above the
# RETRIEVAL_MISS baseline (0.6) without dominating USER_CORRECTION
# (0.9), which is the strongest organic source.
_SEVERITY_TO_STRENGTH: dict[Severity, float] = {
    Severity.LOW: 0.30,
    Severity.MEDIUM: 0.50,
    Severity.HIGH: 0.70,
    Severity.CRITICAL: 0.90,
}


def emit_to_self_improver(report: IncidentReport) -> bool:
    """Best-effort: feed the incident into the Self-Improver loop.

    Uses ``GapSource.LOW_CONFIDENCE`` (most semantically accurate for
    a bias detection — every bias represents a low-confidence call
    that was made declaratively). The bias_id and incident_id ride in
    the evidence dict so the Learner / Integrator can route by type.

    Returns ``True`` on successful emit, ``False`` if the Self-Improver
    isn't available or the call failed. Either way, the IncidentReport
    itself is already persisted by the caller — Self-Improver
    integration is purely additive.
    """
    try:
        from app.self_improvement.types import GapSource, LearningGap
        from app.self_improvement.store import emit_gap
    except ImportError:
        logger.debug(
            "epistemic emit_to_self_improver: gap_detector not available; "
            "skipping (incident %s persisted but not flushed)",
            report.incident_id,
        )
        return False

    bias = BIAS_LIBRARY.get(report.root_cause.bias_id)
    strength = _SEVERITY_TO_STRENGTH.get(report.root_cause.severity, 0.5)

    description = (
        f"Epistemic incident: {bias.name} "
        f"({len(report.enabling_factors)} enabling factors)"
    )

    evidence = {
        "incident_id": report.incident_id,
        "task_id": report.task_id,
        "bias_id": report.root_cause.bias_id,
        "bias_name": bias.name,
        "severity": report.severity.value,
        "matched_claim_ids": list(report.root_cause.matched_claim_ids),
        "enabling_bias_ids": [m.bias_id for m in report.enabling_factors],
        "behavioral_changes": [bc.as_jsonable() for bc in report.behavioral_changes],
        "missed_signals": list(report.missed_signals),
    }

    try:
        gap = LearningGap(
            id="",  # store assigns deterministic id
            source=GapSource.LOW_CONFIDENCE,
            description=description[:200],
            evidence=evidence,
            signal_strength=strength,
        )
        return bool(emit_gap(gap))
    except Exception as exc:
        logger.warning(
            "epistemic emit_to_self_improver: emit_gap raised: %s", exc,
        )
        return False


def persist_and_emit(report: IncidentReport) -> bool:
    """Persist an IncidentReport then attempt Self-Improver integration.

    Returns whether the Self-Improver emit succeeded. Persistence
    failure is logged at DEBUG and not propagated.
    """
    try:
        from app.epistemic.span_writer import persist_incident
        persist_incident(report)
    except Exception as exc:
        logger.debug(
            "epistemic persist_and_emit: persist_incident failed: %s", exc,
        )
    return emit_to_self_improver(report)
