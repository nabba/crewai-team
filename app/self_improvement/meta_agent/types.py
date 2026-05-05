"""
app.self_improvement.meta_agent.types — typed records for the meta-agent layer.

Three records carry the meta-agent's state:

    AgentRecipe       — a bounded augmentation applied on top of an existing
                        agent factory's output. NEVER replaces the factory.
    RecipeOutcome     — observed result from applying a recipe to one task.
                        Append-only; the bandit reads this to score recipes.
    RecipeSelection   — the selector's reasoned choice for a single dispatch
                        (recipe + alternatives + score breakdown). Useful
                        for audit trails and the /cp/ops dashboard.

Everything is JSON-serialisable so the same shapes round-trip through
Postgres rows, ChromaDB documents, and the control_plane HTTP API.

IMMUTABLE — infrastructure-level types.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional


# ── Lifecycle helpers ────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── Records ──────────────────────────────────────────────────────────────────

@dataclass
class AgentRecipe:
    """A bounded configuration applied on top of an agent factory output.

    Recipes are EVALUATED, not generated. The MetaAgent searches the
    historical recipe space (force_tier × extra_tool_set × task_hint) by
    similarity to the incoming task and selects the highest-UCB option.
    The factory itself (create_coder, create_researcher, ...) is
    IMMUTABLE — recipes never replace the factory's output, only
    augment it through channels run_single_agent_crew already exposes.

    Why these specific knobs:
        force_tier         — already a parameter of every agent factory
        extra_tool_names   — already supported by run_single_agent_crew
        task_hint          — non-destructive prefix on the task template
        max_execution_time — advisory ceiling; agent factories ignore None

    Anything outside this set (backstory, goal, llm model rules, agent
    class) is in TIER_GATED or TIER_IMMUTABLE and cannot be touched
    from this layer.
    """

    id: str
    crew_name: str                       # "coding", "research", "writing", ...

    # Augmentation knobs (None / empty = "let the factory default win")
    force_tier: Optional[str] = None
    extra_tool_names: list[str] = field(default_factory=list)
    task_hint: str = ""
    max_execution_time: Optional[int] = None

    # Used for similarity matching — canonical text representation of
    # the kind of task this recipe was tried on. The selector embeds
    # this and matches against the embedding of the incoming task.
    task_signature: str = ""

    # Lifecycle + provenance
    created_at: str = field(default_factory=_now_iso)
    proposed_by: str = "meta_agent"      # "meta_agent" | "operator" | "seed"
    notes: str = ""

    # Convergence control: how many outcomes have been observed for this
    # recipe. The selector uses this directly (no extra round-trip).
    uses: int = 0
    successes: int = 0
    last_used_at: str = ""

    @property
    def is_null(self) -> bool:
        """True iff this recipe makes no augmentation (factory-default).

        The selector treats the null recipe as the always-available
        control arm in the bandit.
        """
        return (
            self.force_tier is None
            and not self.extra_tool_names
            and not self.task_hint
            and self.max_execution_time is None
        )

    @property
    def smoothed_success_rate(self) -> float:
        """Bayesian-smoothed success rate ((successes+1)/(uses+2)).

        Smoothing prevents a single early failure from permanently
        burying a recipe; with 0/0 the rate is 0.5 (true uncertainty).
        """
        return (self.successes + 1) / (self.uses + 2)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AgentRecipe":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


@dataclass
class RecipeOutcome:
    """One observed outcome from applying a recipe to a single task.

    Append-only — never updated in place. The bandit reads aggregates
    of these rows; recipe.uses/successes are denormalised counters
    kept in sync by the recorder.
    """

    id: str
    recipe_id: str
    crew_name: str
    task_id: str

    success: bool
    confidence: str = ""                 # "high" | "medium" | "low" | ""
    duration_s: float = 0.0
    cost_estimate: float = 0.0           # tokens × tier rate, advisory
    error_signature: str = ""            # for failure clustering
    user_feedback: str = ""              # "👍" / "👎" / "" if from React
    recorded_at: str = field(default_factory=_now_iso)

    # Snapshot of the embedded task signature at apply time. Lets us
    # later re-cluster outcomes by similarity even if the recipe's
    # task_signature drifts.
    task_signature: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RecipeOutcome":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


@dataclass
class RecipeSelection:
    """The selector's reasoned choice for one dispatch.

    Returned by selector.select_recipe so the caller (and audit logs)
    can see why a recipe won — not just which one. Persisted lightly
    in the outcome row; useful for the /cp/ops dashboard.
    """

    chosen: AgentRecipe
    candidates_considered: int
    score: float                         # final UCB × similarity score
    similarity: float                    # 1 - cosine_distance
    smoothed_success_rate: float
    explored: bool                       # True iff ε-greedy explore branch
    rationale: str = ""

    @property
    def chose_null_recipe(self) -> bool:
        return self.chosen.is_null

    def to_dict(self) -> dict:
        d = {
            "chosen_recipe_id": self.chosen.id,
            "chosen_is_null": self.chosen.is_null,
            "candidates_considered": self.candidates_considered,
            "score": self.score,
            "similarity": self.similarity,
            "smoothed_success_rate": self.smoothed_success_rate,
            "explored": self.explored,
            "rationale": self.rationale,
        }
        return d
