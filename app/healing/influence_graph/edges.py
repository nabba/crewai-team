"""Curated influence-graph edges (PROGRAM §49 Q14.2).

This file is the source of truth for "which subsystem's output feeds
which other subsystem's input." Hand-curated because:

  1. Automatic extraction (AST-based import + call analysis) would
     produce noise — many imports don't flow signal, just types.
  2. The edge set is small enough (~50 edges) to maintain by reading
     diffs alongside any subsystem change.
  3. Honest about what we know — if the graph misses an edge, the
     operator sees the gap, vs. an automatic extractor producing
     false confidence with hidden gaps.

When a new subsystem ships:

  1. Add ``InfluenceEdge`` entries to ``EDGES`` for every signal
     this subsystem PRODUCES (the consumers will name their inputs
     here too).
  2. Add edges for every signal this subsystem CONSUMES (looking
     at imports + reads).
  3. Re-run ``find_cycles()`` (test enforces this) — if a new cycle
     appears, the operator sees it in the next pass + the test
     fixture asserts no UNEXPECTED cycles.

Edge kinds (informational; not used by cycle detection):

  * ``DATA`` — A writes a file/row that B reads.
  * ``EVENT`` — A publishes an event B subscribes to.
  * ``RPC`` — A calls into B directly (function call).
  * ``GOVERN`` — A decides whether B can run (rate limit, switch).
  * ``EMBED`` — A's output is the embedding context for B.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass


class EdgeKind(str, enum.Enum):
    DATA = "data"
    EVENT = "event"
    RPC = "rpc"
    GOVERN = "govern"
    EMBED = "embed"


@dataclass(frozen=True)
class InfluenceEdge:
    """One directed edge: producer → consumer via a named signal."""

    producer: str    # node name (subsystem)
    consumer: str    # node name (subsystem)
    signal: str      # human-readable signal label
    kind: EdgeKind = EdgeKind.DATA


# The curated edge list. Order is informational only; cycle detection
# does not depend on it.
EDGES: tuple[InfluenceEdge, ...] = (
    # ── Meta-agent closed loop (Q14.2 drift probe focuses here) ────
    InfluenceEdge(
        "meta_agent.selector", "agent_factory",
        "recipe_selection", EdgeKind.RPC,
    ),
    InfluenceEdge(
        "agent_factory", "crew_lifecycle",
        "agent_built", EdgeKind.RPC,
    ),
    InfluenceEdge(
        "crew_lifecycle", "meta_agent.recorder",
        "outcome", EdgeKind.DATA,
    ),
    InfluenceEdge(
        "meta_agent.recorder", "meta_agent.store",
        "outcome_row", EdgeKind.DATA,
    ),
    InfluenceEdge(
        "meta_agent.store", "meta_agent.selector",
        "recipe_uses_successes", EdgeKind.DATA,
    ),
    InfluenceEdge(
        "crew_lifecycle", "companion.lessons_learned",
        "task_outcome_event", EdgeKind.EVENT,
    ),
    InfluenceEdge(
        "companion.lessons_learned", "meta_agent.selector",
        "lessons_kb_consultation", EdgeKind.DATA,
    ),
    InfluenceEdge(
        "meta_agent.consolidation", "meta_agent.store",
        "recipe_retirement_proposal", EdgeKind.GOVERN,
    ),

    # ── Recovery / runbook / error loop ────────────────────────────
    InfluenceEdge(
        "crew_lifecycle", "error_monitor",
        "error_event", EdgeKind.EVENT,
    ),
    InfluenceEdge(
        "error_monitor", "healing.runbooks",
        "anomaly_record", EdgeKind.RPC,
    ),
    InfluenceEdge(
        "healing.runbooks", "change_requests.lifecycle",
        "create_request", EdgeKind.RPC,
    ),
    InfluenceEdge(
        "change_requests.lifecycle", "auditor",
        "audit_row", EdgeKind.DATA,
    ),
    InfluenceEdge(
        "healing.structured_diagnosis", "change_requests.lifecycle",
        "create_request", EdgeKind.RPC,
    ),
    InfluenceEdge(
        "healing.structured_diagnosis", "healing.diagnosis_telemetry",
        "diagnosis_event", EdgeKind.DATA,
    ),
    InfluenceEdge(
        "healing.diagnosis_telemetry", "healing.diagnosis_auto_tune",
        "approval_rate_series", EdgeKind.DATA,
    ),
    InfluenceEdge(
        "healing.diagnosis_auto_tune", "runtime_settings",
        "threshold_override", EdgeKind.GOVERN,
    ),
    InfluenceEdge(
        "runtime_settings", "healing.structured_diagnosis",
        "current_threshold", EdgeKind.GOVERN,
    ),

    # ── Affect / narrative / consciousness loop ────────────────────
    InfluenceEdge(
        "affect.viability", "affect.hooks",
        "viability_score", EdgeKind.DATA,
    ),
    InfluenceEdge(
        "affect.hooks", "affect.trace",
        "affect_row", EdgeKind.DATA,
    ),
    InfluenceEdge(
        "affect.trace", "narrative.salience",
        "trace_for_episode", EdgeKind.DATA,
    ),
    InfluenceEdge(
        "narrative.salience", "narrative.episode_synth",
        "episode_candidate", EdgeKind.DATA,
    ),
    InfluenceEdge(
        "narrative.episode_synth", "narrative.chapter_consolidator",
        "episode", EdgeKind.DATA,
    ),
    InfluenceEdge(
        "narrative.chapter_consolidator", "identity.continuity_ledger",
        "soul_edit_signal", EdgeKind.EVENT,
    ),
    InfluenceEdge(
        "affect.goal_emitter", "kernel.self_state",
        "current_goals", EdgeKind.DATA,
    ),

    # ── Identity / drift / annual reflection ──────────────────────
    InfluenceEdge(
        "governance_amendment.protocol", "identity.continuity_ledger",
        "tier3_amendment_event", EdgeKind.EVENT,
    ),
    InfluenceEdge(
        "governance_ratchet.protocol", "identity.continuity_ledger",
        "ratchet_event", EdgeKind.EVENT,
    ),
    InfluenceEdge(
        "identity.continuity_ledger", "identity.annual_reflection",
        "drift_summary", EdgeKind.DATA,
    ),
    InfluenceEdge(
        "identity.continuity_ledger", "identity.drift_digest",
        "drift_summary", EdgeKind.DATA,
    ),
    InfluenceEdge(
        "identity.continuity_ledger", "identity.long_term_goal_review",
        "drift_summary", EdgeKind.DATA,
    ),
    InfluenceEdge(
        "identity.annual_reflection", "wiki.value_reflections",
        "essay", EdgeKind.DATA,
    ),

    # ── Sentience experiments (Q5) ──────────────────────────────────
    InfluenceEdge(
        "sentience_experiments.ae2_causal_credit",
        "identity.continuity_ledger",
        "sentience_observation", EdgeKind.EVENT,
    ),
    InfluenceEdge(
        "sentience_experiments.hot1_meta_affect",
        "identity.continuity_ledger",
        "sentience_observation", EdgeKind.EVENT,
    ),
    InfluenceEdge(
        "sentience_experiments.hot4_metacog_monitor",
        "identity.continuity_ledger",
        "sentience_observation", EdgeKind.EVENT,
    ),
    InfluenceEdge(
        "sentience_experiments.rpt1_self_calibration",
        "identity.continuity_ledger",
        "sentience_observation", EdgeKind.EVENT,
    ),

    # ── Proposal bridge / library_radar / dependency_radar ─────────
    InfluenceEdge(
        "self_improvement.capability_gap_analyzer", "proposal_bridge.store",
        "staged_proposal", EdgeKind.DATA,
    ),
    InfluenceEdge(
        "library_radar.proposer", "proposal_bridge.store",
        "staged_proposal", EdgeKind.DATA,
    ),
    InfluenceEdge(
        "paper_pipeline", "proposal_bridge.store",
        "staged_proposal", EdgeKind.DATA,
    ),
    InfluenceEdge(
        "dependency_radar.proposer", "proposal_bridge.store",
        "staged_proposal", EdgeKind.DATA,
    ),
    InfluenceEdge(
        "proposal_bridge.promoter", "change_requests.lifecycle",
        "create_request", EdgeKind.RPC,
    ),

    # ── Companion + interest model + person correlation ────────────
    InfluenceEdge(
        "conversation_store", "companion.interest_model",
        "conversation_text", EdgeKind.DATA,
    ),
    InfluenceEdge(
        "email_tools", "companion.interest_model",
        "email_subjects", EdgeKind.DATA,
    ),
    InfluenceEdge(
        "calendar_tools", "companion.interest_model",
        "event_titles", EdgeKind.DATA,
    ),
    InfluenceEdge(
        "companion.feedback", "companion.topic_weights",
        "topic_feedback", EdgeKind.EVENT,
    ),
    InfluenceEdge(
        "companion.topic_weights", "companion.interest_model",
        "topic_multiplier", EdgeKind.DATA,
    ),
    InfluenceEdge(
        "companion.interest_model", "companion.idle_contemplation",
        "top_topics", EdgeKind.DATA,
    ),

    # ── Workspace publish → SubIA Global Workspace ─────────────────
    InfluenceEdge(
        "workspace_publish", "subia.global_workspace",
        "publish_event", EdgeKind.EVENT,
    ),

    # ── Threads / approaches / lessons ─────────────────────────────
    InfluenceEdge(
        "threads.approaches", "companion.lessons_learned",
        "thread_closure_event", EdgeKind.EVENT,
    ),
    InfluenceEdge(
        "threads.approaches", "threads.store",
        "approaches_summary", EdgeKind.DATA,
    ),
    InfluenceEdge(
        "threads.store", "recovery.thread_consultation",
        "open_threads", EdgeKind.DATA,
    ),

    # ── Healing monitors (observation layer, not in cycles) ────────
    # Most healing monitors are pure readers: they observe other
    # subsystems but don't feed signal back in. Modelled here so the
    # graph is complete for the operator's reading; they appear as
    # leaf consumers in the topology.
    InfluenceEdge(
        "auditor", "healing.auditor_bridge",
        "audit_journal", EdgeKind.DATA,
    ),
    InfluenceEdge(
        "healing.auditor_bridge", "change_requests.lifecycle",
        "create_request", EdgeKind.RPC,
    ),
)


def nodes() -> frozenset[str]:
    """All nodes that appear as producer or consumer in EDGES."""
    s: set[str] = set()
    for e in EDGES:
        s.add(e.producer)
        s.add(e.consumer)
    return frozenset(s)
