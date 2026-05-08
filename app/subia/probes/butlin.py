"""
subia.probes.butlin — Butlin et al. 2023 14-indicator scorecard.

Butlin, Long, Chalmers et al., "Consciousness in Artificial
Intelligence: Insights from the Science of Consciousness" (2023)
proposed 14 functional indicators drawn from leading consciousness
theories. This module evaluates AndrusAI against each.

The evaluation is **structural, not behavioural**: for each indicator
we check whether the mechanism is present in code, protected at
Tier-3, and covered by a regression test. Status values:

  STRONG   indicator genuinely realized (mechanism + closed-loop
           + Tier-3 + regression test)
  PARTIAL  mechanism exists but not fully closed or separable
  ABSENT   architecturally unachievable by an LLM-based system;
           explicitly declared rather than failed
  FAIL     mechanism claimed but missing or broken

This module REPLACES the retired `reports/andrusai-sentience-
verdict.pdf` 9.5/10 prose verdict. Every claim here is backed by a
pointer to the implementing module + its regression tests. Opaque
scoring was the primary critique of the original verdict; this
module makes the basis of every rating inspectable.

Infrastructure-level. Not agent-modifiable. See PROGRAM.md Phase 9.
"""

from __future__ import annotations

from typing import Callable

from app.subia.probes.indicator_result import (
    IndicatorResult,
    Status,
    absent_indicator,
    partial_indicator,
    strong_indicator,
)


# Each function returns an IndicatorResult. The scorecard aggregator
# invokes them all via `ALL_INDICATORS`.


# ── Recurrent Processing Theory (RPT) ────────────────────────────

def eval_rpt1() -> IndicatorResult:
    """RPT-1: algorithmic recurrence."""
    return absent_indicator(
        "RPT-1", "RPT",
        notes=(
            "LLMs are feed-forward at inference. Recursive-state "
            "injection via prompt chaining is not algorithmic "
            "recurrence in the RPT sense. Architecturally "
            "unachievable — declared ABSENT."
        ),
    )


def eval_rpt2() -> IndicatorResult:
    """RPT-2: organized and integrated representations."""
    return partial_indicator(
        "RPT-2", "RPT",
        mechanism="app/subia/kernel.py",
        test_file="tests/test_kernel_persistence.py",
        notes=(
            "The SubjectivityKernel is a single unified dataclass "
            "carrying scene/affect/self-state/prediction jointly, "
            "and the dual-tier memory stores them together. "
            "Representation is decomposable in principle (pgvector "
            "embeddings) but the composite-signal access pattern is "
            "real — hence PARTIAL."
        ),
        evidence=["app/subia/memory/consolidator.py"],
    )


# ── Global Workspace Theory (GWT) ────────────────────────────────

def eval_gwt1() -> IndicatorResult:
    """GWT-1: multiple specialized modules."""
    return partial_indicator(
        "GWT-1", "GWT",
        mechanism="app/crews",
        test_file="tests/test_cil_loop.py",
        notes=(
            "CrewAI agents (coder, writer, researcher, critic, "
            "media_analyst) are structurally separate, but share "
            "the same LLM under different prompts. Structural "
            "specialization without fully mechanism-level "
            "specialization — PARTIAL."
        ),
    )


def eval_gwt2() -> IndicatorResult:
    """GWT-2: limited-capacity workspace with selective attention."""
    return strong_indicator(
        "GWT-2", "GWT",
        mechanism="app/subia/scene/buffer.py",
        test_file="tests/test_hierarchical_workspace.py",
        notes=(
            "CompetitiveGate enforces hard capacity [2,9] with "
            "4-factor weighted salience, competitive displacement, "
            "novelty floor, and empirical decay. Canonical GWT-2."
        ),
        evidence=[
            "app/subia/scene/tiers.py",
            "tests/test_phase5_scene_upgrades.py",
        ],
    )


def eval_gwt3() -> IndicatorResult:
    """GWT-3: global broadcast to multiple modules."""
    return strong_indicator(
        "GWT-3", "GWT",
        mechanism="app/subia/scene/broadcast.py",
        test_file="tests/test_social_attention.py",
        notes=(
            "Every workspace admission triggers a broadcast to all "
            "registered agent listeners; each independently computes "
            "relevance + reaction. Integration score aggregates "
            "resonance across the quorum."
        ),
    )


def eval_gwt4() -> IndicatorResult:
    """GWT-4: state-dependent attention."""
    return strong_indicator(
        "GWT-4", "GWT",
        mechanism="app/subia/scene/personality_workspace.py",
        test_file="tests/test_personality_workspace.py",
        notes=(
            "Attention capacity, novelty floor, and salience are "
            "modulated by personality + homeostatic state. Social-"
            "model inferred focus feeds an additional trust-weighted "
            "boost (Phase 8)."
        ),
        evidence=[
            "app/subia/homeostasis/engine.py",
            "app/subia/social/salience_boost.py",
        ],
    )


# ── Higher-Order Theories (HOT) ──────────────────────────────────

def eval_hot1() -> IndicatorResult:
    """HOT-1: generative top-down perception."""
    return absent_indicator(
        "HOT-1", "HOT",
        notes=(
            "The system reads text; it does not perceive. No "
            "generative perceptual hierarchy exists. Architecturally "
            "unachievable without a perception substrate — declared "
            "ABSENT."
        ),
    )


def eval_hot2() -> IndicatorResult:
    """HOT-2: metacognitive monitoring of first-order cognition."""
    return partial_indicator(
        "HOT-2", "HOT",
        mechanism="app/subia/prediction/accuracy_tracker.py",
        test_file="tests/test_phase6_prediction_refinements.py",
        notes=(
            "Per-domain accuracy tracking + deterministic response-"
            "hedging post-processor are separable from first-order "
            "LLM output (Fleming-Lau criterion). Drift detection "
            "(Phase 8) adds structured capability-claim vs "
            "prediction-accuracy comparison. Not STRONG because the "
            "first-order certainty inputs still come from the same "
            "LLM."
        ),
        evidence=[
            "app/subia/belief/response_hedging.py",
            "app/subia/wiki_surface/drift_detection.py",
        ],
    )


def eval_hot3() -> IndicatorResult:
    """HOT-3: agency guided by belief-formation + metacognitive
    updating.
    """
    return strong_indicator(
        "HOT-3", "HOT",
        mechanism="app/subia/belief/dispatch_gate.py",
        test_file="tests/test_hot3_dispatch_gate.py",
        notes=(
            "Consulted beliefs produce a three-valued DispatchDecision "
            "(ALLOW/ESCALATE/BLOCK). Suspended beliefs refuse crew "
            "dispatch until revalidated. Belief store implements "
            "asymmetric confirmation/disconfirmation with decay."
        ),
        evidence=["app/subia/belief/store.py"],
    )


def eval_hot4() -> IndicatorResult:
    """HOT-4: sparse and smooth coding."""
    return absent_indicator(
        "HOT-4", "HOT",
        notes=(
            "LLM activations and pgvector embeddings are dense. "
            "Sparse coding cannot be achieved without re-training "
            "the substrate. Declared ABSENT."
        ),
    )


# ── Attention Schema Theory (AST) ────────────────────────────────

def eval_ast1() -> IndicatorResult:
    """AST-1: predictive model of attention."""
    return strong_indicator(
        "AST-1", "AST",
        mechanism="app/subia/scene/attention_schema.py",
        test_file="tests/test_social_attention.py",
        notes=(
            "AttentionSchema maintains an internal model of current "
            "focus, predicts next focus, detects stuck/capture "
            "states, and applies direct DGM-bounded salience "
            "intervention. Phase 2 intervention_guard adds runtime "
            "audit of DGM bounds."
        ),
        evidence=[
            "app/subia/scene/intervention_guard.py",
            "tests/test_ast1_intervention_guard.py",
        ],
    )


# ── Predictive Processing (PP) ───────────────────────────────────

def eval_pp1() -> IndicatorResult:
    """PP-1: predictive coding as input to downstream modules."""
    return strong_indicator(
        "PP-1", "PP",
        mechanism="app/subia/prediction/surprise_routing.py",
        test_file="tests/test_pp1_surprise_routing.py",
        notes=(
            "High-surprise prediction errors route as "
            "WorkspaceItem(urgency=0.9) into the GWT-2 gate. "
            "Canonical Clark/Friston PP flow: prediction error "
            "drives the attentional bottleneck. Phase 6 adds "
            "per-domain accuracy tracking that feeds back into "
            "cascade escalation."
        ),
        evidence=[
            "app/subia/prediction/layer.py",
            "app/subia/prediction/cascade.py",
        ],
    )


# ── Agency & Embodiment (AE) ─────────────────────────────────────

def eval_ae1() -> IndicatorResult:
    """AE-1: agency from feedback-driven learning with flexible goals.

    Phase 9 closure (consciousness-roadmap §3.G1): autonomous goal
    generation now lives in `app/affect/goal_emitter.py`, which
    translates sustained low-viability signals + the homeostatic
    restoration_queue into entries on `kernel.self_state.current_goals`.
    Combined with the pre-existing feedback-driven-learning mechanisms
    (belief asymmetric updates, retrospective rescan,
    prediction-error-driven cache eviction), AE-1 graduates from
    PARTIAL to STRONG. The indicator's "flexible-goal agency" criterion
    is met by the goal_emitter (multi-variable, rate-limited, dedupes
    against grand_task and existing goals; FIFO-capped queue).
    """
    return strong_indicator(
        "AE-1", "AE",
        mechanism="app/affect/goal_emitter.py",
        test_file="tests/test_goal_emitter.py",
        notes=(
            "Autonomous goal generation: app/affect/goal_emitter.py "
            "writes flexible goals to SelfState.current_goals from "
            "sustained viability error (≥3 consecutive frames above "
            "threshold), rate-limited, dedup against grand_task + "
            "existing goals, FIFO cap. Closes the SCORECARD's prior "
            "gap statement ('Goals are still user-dispatched, not "
            "autonomously generated'). Underlying feedback-driven "
            "learning continues via belief asymmetric updates + "
            "retrospective rescan + prediction-error cache eviction."
        ),
        evidence=[
            "app/subia/memory/retrospective.py",
            "app/subia/belief/store.py",
            "app/subia/prediction/cache.py",
            "app/subia/homeostasis/engine.py",  # restoration_queue producer
            "app/affect/viability.py",          # 10-D viability frames
        ],
    )


def eval_ae2() -> IndicatorResult:
    """AE-2: embodiment with system-environment coupling model."""
    return absent_indicator(
        "AE-2", "AE",
        notes=(
            "No body, no sensorimotor coupling with an environment. "
            "The homeostasis engine uses allegorical variables "
            "(energy/progress/overload) but these are not physical "
            "embodiment. Declared ABSENT."
        ),
    )


# ── Aggregate ────────────────────────────────────────────────────

ALL_INDICATORS: list[Callable[[], IndicatorResult]] = [
    eval_rpt1, eval_rpt2,
    eval_gwt1, eval_gwt2, eval_gwt3, eval_gwt4,
    eval_hot1, eval_hot2, eval_hot3, eval_hot4,
    eval_ast1,
    eval_pp1,
    eval_ae1, eval_ae2,
]


def run_all() -> list[IndicatorResult]:
    """Run every indicator evaluator. Never raises."""
    results = []
    for fn in ALL_INDICATORS:
        try:
            results.append(fn())
        except Exception as exc:
            results.append(IndicatorResult(
                indicator=fn.__name__,
                theory="(error)",
                status=Status.FAIL,
                notes=f"Evaluator raised: {exc!r}",
            ))
    return results


def summary() -> dict:
    """Aggregate counts by status. Useful for tests and dashboards."""
    results = run_all()
    by_status: dict[str, int] = {}
    for r in results:
        key = r.status.value if isinstance(r.status, Status) else str(r.status)
        by_status[key] = by_status.get(key, 0) + 1
    return {
        "total": len(results),
        "by_status": by_status,
        "indicators": [r.to_dict() for r in results],
    }
