"""Temporal binding (Proposal §3.2).

The proposal asks for parallel pre-task execution of CIL steps 2-6,
followed by a binding step. We DO NOT restructure the loop into true
async parallelism — that breaks well-tested causal ordering invariants
(Step 3 attention reads Step 2 affect, etc.). Instead, we deliver the
SEMANTIC PAYOFF the proposal cares about — "the binding IS the unity
of experience" — by adding a binding REDUCER that integrates the
just-computed signals from steps 2-6 into a single BoundMoment after
they finish, *and* makes that BoundMoment the simultaneously-present
context the next loop reads from.

Conflict resolution rules are explicit and Tier-3 protected (no agent
override).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BoundMoment:
    """The reconciled, simultaneously-present moment.

    Outputs of FEEL / ATTEND / OWN / PREDICT / MONITOR are integrated
    here. Conflicts are explicitly recorded so downstream consumers
    (and the meta-monitor) can see what was reconciled.
    """
    feel: dict = field(default_factory=dict)
    attend: dict = field(default_factory=dict)
    own: dict = field(default_factory=dict)
    predict: dict = field(default_factory=dict)
    monitor: dict = field(default_factory=dict)

    # Bound view derived from the five layers
    salient_focus: list = field(default_factory=list)         # top-N items
    dominant_affect: str = "neutral"
    confidence_unified: float = 0.5
    conflicts: list = field(default_factory=list)             # human-readable strings


def _confidence_unified(feel: dict, monitor: dict, predict: dict) -> tuple[float, list]:
    """Reconcile multiple confidence signals into one with conflict notes."""
    notes: list = []
    affect_urgency = float(feel.get("urgency", 0.0))
    monitor_conf = float(monitor.get("confidence", 0.5))
    predict_conf = float(predict.get("confidence", 0.5))

    # Conflict: high urgency + low confidence + high prediction confidence
    if affect_urgency > 0.6 and monitor_conf < 0.4 and predict_conf > 0.7:
        notes.append(
            "felt urgency + low monitored confidence + high predicted confidence "
            "→ binding favours predicted resolution"
        )
        unified = round((monitor_conf + predict_conf * 2) / 3, 4)
    elif affect_urgency > 0.6 and predict_conf < 0.3:
        notes.append("urgency without predictive grounding → caution")
        unified = round((monitor_conf + predict_conf) / 2, 4)
    else:
        unified = round((monitor_conf + predict_conf) / 2, 4)
    return unified, notes


def temporal_bind(
    *,
    feel: Optional[dict] = None,
    attend: Optional[dict] = None,
    own: Optional[dict] = None,
    predict: Optional[dict] = None,
    monitor: Optional[dict] = None,
    retention: Optional[list] = None,
) -> BoundMoment:
    """Reduce simultaneous signals into a single bound moment.

    `retention` is the SpeciousPresent.retention list — the binder uses
    it to apply a STABILITY BIAS: items that have persisted across the
    retention window are weighted up against items that just appeared
    (this is what mutes the "shiny new thing" reflex that pure novelty
    salience produces).
    """
    feel = feel or {}
    attend = attend or {}
    own = own or {}
    predict = predict or {}
    monitor = monitor or {}

    bm = BoundMoment(feel=feel, attend=attend, own=own,
                     predict=predict, monitor=monitor)

    # ── Salient focus ──────────────────────────────────────────────
    candidates = list(attend.get("focal_items", []) or [])
    stability_set: set = set()
    if retention:
        # Items present in EVERY retention frame
        per_frame = []
        for m in retention:
            per_frame.append(set((m.scene_delta or {}).get("entered", []) or []))
        if per_frame:
            stable = per_frame[0]
            for s in per_frame[1:]:
                stable &= s
            stability_set = stable
    # Re-rank: stable items get +0.1 effective salience
    def _key(it):
        sal = float(it.get("salience", 0.5)) if isinstance(it, dict) else 0.5
        if isinstance(it, dict) and it.get("id") in stability_set:
            sal += 0.1
        return -sal
    bm.salient_focus = sorted(candidates, key=_key)[:5]

    # ── Affect ─────────────────────────────────────────────────────
    bm.dominant_affect = feel.get("dominant_affect", "neutral")

    # ── Confidence ─────────────────────────────────────────────────
    bm.confidence_unified, conflict_notes = _confidence_unified(feel, monitor, predict)
    bm.conflicts.extend(conflict_notes)

    # ── Ownership conflict surfacing ───────────────────────────────
    own_state = own.get("ownership_assignments", {}) or {}
    if own_state and any(v == "external" for v in own_state.values()) \
            and any(v == "self" for v in own_state.values()):
        bm.conflicts.append("mixed ownership in focal scene (self + external)")

    return bm


def temporal_quick_bind(
    *,
    feel: Optional[dict] = None,
    attend: Optional[dict] = None,
) -> BoundMoment:
    """Cheap variant of `temporal_bind` for the compressed CIL path.

    The compressed loop runs only Steps 1-3 (PERCEIVE / FEEL / ATTEND) and
    early-returns before OWN / PREDICT / MONITOR fire. Without this helper,
    compressed cycles produce no BoundMoment at all, so any observability
    surface that reads `details["bound_moment_*"]` sees nothing on the
    common case (the compressed path is the default for unknown operations,
    see `loop.py:130`).

    What this can compute (from FEEL + ATTEND alone):
      * `dominant_affect`     — direct from feel.dominant_affect
      * `salient_focus`       — top-N from attend.focal_items, no stability
                                bias (no retention window passed; the
                                compressed path doesn't read SpeciousPresent)

    What this CANNOT compute:
      * `confidence_unified`  — needs MONITOR.confidence + PREDICT.confidence,
                                which haven't run; left at the dataclass
                                default of 0.5
      * `conflicts`           — ownership conflict needs OWN output; left []

    Tier-3 protected (no agent override) — same discipline as `temporal_bind`.
    Consciousness-roadmap §3.G4. The honest scope is observability uniformity
    across full and compressed loops, NOT correctness recovery (the original
    G4 framing assumed stale-on-kernel which doesn't apply — see audit
    note in CONSCIOUSNESS_ROADMAP.md).
    """
    feel = feel or {}
    attend = attend or {}

    bm = BoundMoment(feel=feel, attend=attend)

    # Salient focus — same shape as the full reducer but without stability
    # bias (no retention frames available on the compressed path).
    candidates = list(attend.get("focal_items", []) or [])

    def _key(it):
        return -float(it.get("salience", 0.5)) if isinstance(it, dict) else -0.5

    bm.salient_focus = sorted(candidates, key=_key)[:5]
    bm.dominant_affect = feel.get("dominant_affect", "neutral")

    return bm
