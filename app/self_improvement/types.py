"""
app.self_improvement.types — typed records for the self-improvement loop.

Single source of truth for the structured data flowing through the pipeline:
    LearningGap   — observed evidence of missing or weak knowledge
    NoveltyReport — verdict from the Novelty Gate on a candidate text
    SkillDraft    — Learner's proposed skill, awaiting integration
    SkillRecord   — integrated skill living in a KB

All records carry provenance so any artifact can be traced back to the
triggering event.

IMMUTABLE — infrastructure-level types.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum


# ── Enums ────────────────────────────────────────────────────────────────────

class GapSource(str, Enum):
    """Why a learning gap was detected.

    Each source has a distinct evidence shape and signal strength weight
    (see app.self_improvement.gap_detector for weights).
    """
    RETRIEVAL_MISS          = "retrieval_miss"     # KB query returned nothing or low-score
    REFLEXION_FAILURE       = "reflexion_failure"  # crew exhausted reflexion retries
    LOW_CONFIDENCE          = "low_confidence"     # vetting/uncertainty flagged
    USER_CORRECTION         = "user_correction"    # user reaction or "actually..."
    TENSION                 = "tension"            # unresolved Tensions KB entry
    MAPELITES_VOID          = "mapelites_void"     # empty grid cell w/ strong neighbors
    USAGE_DECAY             = "usage_decay"        # skill never retrieved in N days
    # arXiv:2603.10600 — trajectory-informed memory generation
    TRAJECTORY_ATTRIBUTION  = "trajectory_attribution"   # post-hoc causal analysis of a run
    OBSERVER_MIS_PREDICTION = "observer_mis_prediction"  # Observer systematically wrong on a mode
    RECOVERY_REFUSAL        = "recovery_refusal"          # recovery loop tried + exhausted strategies


class GapStatus(str, Enum):
    """Lifecycle stage of a LearningGap."""
    OPEN              = "open"               # detected, not yet acted on
    TRIAGED           = "triaged"            # reviewed, awaiting scheduling
    SCHEDULED         = "scheduled"          # in learning queue
    RESOLVED_EXISTING = "resolved_existing"  # KB already had it (retrieval issue)
    RESOLVED_NEW      = "resolved_new"       # new skill created
    REJECTED          = "rejected"           # not worth pursuing


class NoveltyDecision(str, Enum):
    """Novelty Gate verdict on a candidate text."""
    COVERED   = "covered"    # already in the KBs — reject
    OVERLAP   = "overlap"    # heavy overlap — propose extension
    ADJACENT  = "adjacent"   # nearby — create with cross-link
    NOVEL     = "novel"      # distinct — create


# ── Records ──────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class LearningGap:
    """Structured evidence that the system needs to learn something."""
    id: str
    source: GapSource
    description: str                              # human-readable phrasing
    evidence: dict = field(default_factory=dict)  # task_id, query, error_id, etc.
    signal_strength: float = 0.5                  # 0..1, source-weighted
    detected_at: str = field(default_factory=_now_iso)
    status: GapStatus = GapStatus.OPEN
    resolution_notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["source"] = self.source.value
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "LearningGap":
        d = dict(d)
        if isinstance(d.get("source"), str):
            d["source"] = GapSource(d["source"])
        if isinstance(d.get("status"), str):
            d["status"] = GapStatus(d["status"])
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


@dataclass
class NoveltyReport:
    """Verdict from the Novelty Gate.

    nearest_distance: 0.0 = identical, 1.0 = orthogonal (using cosine distance).
    Higher = more novel. The decision band is computed against NOVELTY_THRESHOLDS.
    """
    decision: NoveltyDecision
    nearest_distance: float
    nearest_text: str = ""
    nearest_kb: str = ""
    nearest_id: str = ""
    rationale: str = ""

    @property
    def should_create(self) -> bool:
        """True when the candidate should result in a new skill."""
        return self.decision in (NoveltyDecision.NOVEL, NoveltyDecision.ADJACENT)

    @property
    def should_extend_existing(self) -> bool:
        """True when the better action is to extend the nearest existing skill."""
        return self.decision == NoveltyDecision.OVERLAP

    @property
    def is_duplicate(self) -> bool:
        """True when the candidate is already covered."""
        return self.decision == NoveltyDecision.COVERED


@dataclass
class SkillDraft:
    """A Learner-produced skill awaiting Integrator routing.

    Phase 3 of the overhaul consumes this. Defined here so the schema is
    fixed before consumers are written.

    Trajectory-sourced fields (arXiv:2603.10600) are optional; when set,
    they carry through to `SkillRecord.provenance` so retrieval can filter
    on tip_type and the originating trajectory can be audited or bulk-
    archived if it turns out to have produced unreliable tips.

    Transfer-memory fields (Phase 17, arXiv:2606.21099-style MTL) are
    populated by `app.transfer_memory.compiler` for cross-domain "Insight"
    memories compiled from healing/evo/grounding/gap sources. Empty for
    trajectory tips and external-topic drafts.
    """
    id: str
    topic: str
    rationale: str
    content_markdown: str
    proposed_kb: str = "episteme"  # episteme | experiential | aesthetics | tensions
    supersedes: list[str] = field(default_factory=list)
    created_from_gap: str = ""
    novelty_at_creation: float = 0.0
    created_at: str = field(default_factory=_now_iso)
    # ── Trajectory-sourced provenance (optional; empty for external topics) ──
    tip_type: str = ""              # "strategy" | "recovery" | "optimization" | ""
    source_trajectory_id: str = ""  # UUID of the Trajectory this was distilled from
    agent_role: str = ""            # crew_name of the trajectory (for retrieval filter)
    # ── Transfer-memory provenance (optional; empty for non-transfer drafts) ──
    source_kind: str = ""           # "trajectory|healing|evo_success|evo_failure|grounding_correction|gap"
    source_domain: str = ""         # "coding|ops|research|grounding|business|safety|evolution|healing"
    transfer_scope: str = ""        # "shadow|project_local|same_domain_only|global_meta"
    project_origin: str = ""        # project name when source was project-scoped
    abstraction_score: float = 0.0  # 0..1, deterministic from transfer_memory.scorer
    leakage_risk: float = 0.0       # 0..1, deterministic from transfer_memory.sanitizer
    negative_transfer_tags: str = ""  # comma-sep tags, set post-hoc by attribution
    evidence_refs: str = ""         # comma-sep refs, e.g. "traj:xyz#step7,evo:abc"


def construct_skill_draft(
    *,
    topic: str,
    rationale: str,
    content_markdown: str,
    proposed_kb: str | None = None,
    created_from_gap: str = "",
    supersedes: list[str] | None = None,
    id_prefix: str = "",
    # Trajectory-sourced provenance
    tip_type: str = "",
    source_trajectory_id: str = "",
    agent_role: str = "",
    # Transfer-memory provenance
    source_kind: str = "",
    source_domain: str = "",
    transfer_scope: str = "",
    project_origin: str = "",
    abstraction_score: float = 0.0,
    leakage_risk: float = 0.0,
    negative_transfer_tags: str = "",
    evidence_refs: str = "",
    # Optional pre-computed novelty; if None the helper runs novelty_report.
    novelty_at_creation: float | None = None,
) -> "SkillDraft":
    """Single source of truth for SkillDraft construction.

    Both `app.trajectory.tip_builder.build_draft` and
    `app.transfer_memory.compiler` route through this helper so the schema,
    ID format, and lazy novelty pre-check stay in one place.

    `id_prefix` distinguishes provenance at a glance:
      "traj" → trajectory-sourced tip      (`draft_traj_<hex>`)
      "xfer" → transfer-memory insight     (`draft_xfer_<hex>`)
      ""     → external-topic draft        (`draft_<hex>`)

    `proposed_kb` semantics:
      None → omit from SkillDraft kwargs; the dataclass default
             ("episteme") applies. Use this for flows that never pre-
             classify and want the integrator's LLM classifier to decide.
      ""   → explicit empty string. tip_builder passes this when the
             tip_type is unknown so the integrator's classifier runs.
      "experiential"|"episteme"|... → caller's pre-classification.

    The novelty pre-check is best-effort — failures fall back to 1.0
    (treated as fully novel) so a Chroma outage cannot block draft creation.
    """
    import uuid as _uuid

    if novelty_at_creation is None:
        novelty_at_creation = 1.0
        try:
            from app.self_improvement.novelty import novelty_report
            rep = novelty_report(content_markdown)
            novelty_at_creation = float(rep.nearest_distance)
        except Exception:
            pass

    prefix = f"draft_{id_prefix}_" if id_prefix else "draft_"

    kwargs: dict = {
        "id": f"{prefix}{_uuid.uuid4().hex[:12]}",
        "topic": topic,
        "rationale": rationale,
        "content_markdown": content_markdown,
        "supersedes": list(supersedes or []),
        "created_from_gap": created_from_gap,
        "novelty_at_creation": float(novelty_at_creation),
        "tip_type": tip_type,
        "source_trajectory_id": source_trajectory_id,
        "agent_role": agent_role,
        "source_kind": source_kind,
        "source_domain": source_domain,
        "transfer_scope": transfer_scope,
        "project_origin": project_origin,
        "abstraction_score": float(abstraction_score),
        "leakage_risk": float(leakage_risk),
        "negative_transfer_tags": negative_transfer_tags,
        "evidence_refs": evidence_refs,
    }
    # Only set proposed_kb when caller passes an explicit value (including
    # ""). Passing None preserves the SkillDraft default ("episteme") for
    # external-topic flows that never pre-classify.
    if proposed_kb is not None:
        kwargs["proposed_kb"] = proposed_kb
    return SkillDraft(**kwargs)


@dataclass
class SkillRecord:
    """An integrated skill, persisted in one of the KBs.

    Phase 3 + 4 read/write this. Defined here so the lifecycle states are
    fixed and observable from day one.
    """
    id: str
    topic: str
    content_markdown: str
    kb: str
    status: str = "active"  # active | superseded | archived
    superseded_by: str = ""
    usage_count: int = 0
    last_used_at: str = ""
    provenance: dict = field(default_factory=dict)  # gap_id, draft_id, …
    created_at: str = field(default_factory=_now_iso)

    # ── Conditional activation (T3-9) ─────────────────────────────────
    # Skill is only surfaced to agents when the current execution context
    # matches all of these predicates. Empty string / empty list = no filter.
    requires_mode: str = ""         # "local" | "cloud" | "hybrid" | "insane" | "" (any)
    requires_tier: str = ""         # "local" | "budget" | "mid" | "premium" | "" (any)
    fallback_for_mode: str = ""     # Show ONLY when this mode is NOT active
    requires_tools: list[str] = field(default_factory=list)

    def matches_context(self, mode: str = "", cost_mode: str = "") -> bool:
        """Check if this skill should activate in the current context.

        Args:
            mode: Current LLM mode (local/cloud/hybrid/insane)
            cost_mode: Current cost mode (budget/balanced/quality)

        Returns True if all conditions are met (or if no conditions set).
        """
        if self.requires_mode and self.requires_mode != mode:
            return False
        if self.fallback_for_mode and self.fallback_for_mode == mode:
            return False
        if self.requires_tier and cost_mode:
            _tier_access = {
                "budget": {"local", "budget"},
                "balanced": {"local", "budget", "mid"},
                "quality": {"local", "budget", "mid", "premium"},
            }
            allowed = _tier_access.get(cost_mode, {"local", "budget", "mid", "premium"})
            if self.requires_tier not in allowed:
                return False
        return True
