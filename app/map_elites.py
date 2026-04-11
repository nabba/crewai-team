"""
map_elites.py — MAP-Elites quality-diversity grid for agent strategies.

Instead of converging on a single "best" approach, maintains a grid where
each cell represents a different *type* of solution. A simple-but-cheap
strategy coexists with a complex-but-thorough one.

Feature dimensions (configurable):
    - complexity: simple → elaborate (instruction count, specificity)
    - cost: cheap → expensive (model tier usage patterns)
    - specialization: general → domain-specific (task affinity)

Each cell holds the BEST solution of that type. When the Self-Improver
needs inspiration, it draws from multiple cells (double-selection),
not just the global best.

Key patterns from OpenEvolve:
    1. MAP-Elites quality-diversity preservation
    2. Double-selection: performance baseline + diverse inspiration
    3. Artifact feedback loop: structured generation-to-generation feedback
    4. Template stochasticity: controlled prompt randomization

Backed by PostgreSQL (existing pgvector setup) for persistence.

IMMUTABLE — infrastructure-level module.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── IMMUTABLE configuration ──────────────────────────────────────────────────

# Feature dimensions for the MAP-Elites grid
FEATURE_DIMENSIONS = ["complexity", "cost_efficiency", "specialization"]

# Bins per dimension (10 bins × 3 dims = 1000 cells max)
BINS_PER_DIM = 10

# Number of islands for parallel MAP-Elites
NUM_ISLANDS = 3

# Migration between islands
MIGRATION_INTERVAL = 15   # generations
MIGRATION_COUNT = 3        # top N per island

# Double-selection parameters
TOP_K_PERFORMANCE = 3      # top performers for exploitation
DIVERSE_K_INSPIRATION = 2  # diverse exemplars for exploration


# ── Data types ────────────────────────────────────────────────────────────────


@dataclass
class Artifact:
    """Structured feedback from a single evaluation run.

    Generation-to-generation feedback: each evaluation produces artifacts
    that inform the next mutation attempt.
    """
    generation: int = 0
    success: bool = False
    score: float = 0.0
    execution_time_ms: float = 0.0
    stage_reached: str = ""     # "format" | "smoke" | "full"
    stderr: str = ""
    llm_feedback: str = ""
    failure_stage: str = ""
    suggestion: str = ""
    token_usage: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Artifact":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    def format_for_prompt(self) -> str:
        """Format artifact for injection into mutation prompt."""
        icon = "✅" if self.success else "❌"
        lines = [f"Generation {self.generation}: {icon} | Score: {self.score:.2f}"]
        if self.failure_stage:
            lines.append(f"  ⚠️ Failed at: {self.failure_stage}")
        if self.stderr:
            lines.append(f"  stderr: {self.stderr[:200]}")
        if self.llm_feedback:
            lines.append(f"  💡 Feedback: {self.llm_feedback[:200]}")
        if self.suggestion:
            lines.append(f"  🔧 Suggestion: {self.suggestion[:200]}")
        return "\n".join(lines)


@dataclass
class StrategyEntry:
    """A single strategy in the MAP-Elites grid."""
    strategy_id: str = ""
    role: str = ""               # agent role this strategy is for
    prompt_content: str = ""     # the actual prompt text
    fitness_score: float = 0.0
    feature_vector: dict = field(default_factory=dict)  # {dim: 0.0-1.0}
    generation: int = 0
    parent_id: str = ""
    mutation_type: str = ""
    artifacts: list[Artifact] = field(default_factory=list)  # last N artifacts
    created_at: str = ""
    island_id: int = 0

    @property
    def bin_key(self) -> tuple:
        """Discretized position in the grid."""
        return tuple(
            min(int(self.feature_vector.get(d, 0.5) * BINS_PER_DIM), BINS_PER_DIM - 1)
            for d in FEATURE_DIMENSIONS
        )

    def to_dict(self) -> dict:
        return {
            "strategy_id": self.strategy_id,
            "role": self.role,
            "fitness_score": self.fitness_score,
            "feature_vector": self.feature_vector,
            "generation": self.generation,
            "parent_id": self.parent_id,
            "mutation_type": self.mutation_type,
            "artifacts": [a.to_dict() for a in self.artifacts[-5:]],
            "created_at": self.created_at,
            "island_id": self.island_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StrategyEntry":
        artifacts = [Artifact.from_dict(a) for a in d.pop("artifacts", [])]
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        entry = cls(**valid)
        entry.artifacts = artifacts
        return entry


# ── Feature extraction ────────────────────────────────────────────────────────


def extract_features(prompt: str) -> dict[str, float]:
    """Extract behavioral feature vector from a prompt.

    Returns: {complexity, cost_efficiency, specialization} each in [0.0, 1.0]
    """
    # Complexity: based on instruction count, length, specificity markers
    lines = prompt.split("\n")
    instruction_count = sum(1 for l in lines if l.strip().startswith(("-", "•", "*", "1.", "2.")))
    word_count = len(prompt.split())

    complexity = min(1.0, (instruction_count / 30.0) * 0.5 + (word_count / 2000.0) * 0.5)

    # Cost efficiency: inverse of premium model indicators
    premium_signals = sum(1 for kw in ["detailed", "thorough", "comprehensive", "exhaustive",
                                         "deep analysis", "cross-reference", "multiple sources"]
                          if kw in prompt.lower())
    budget_signals = sum(1 for kw in ["concise", "brief", "quick", "efficient", "minimal",
                                        "direct", "simple"]
                         if kw in prompt.lower())
    cost_efficiency = max(0.0, min(1.0, 0.5 + (budget_signals - premium_signals) * 0.1))

    # Specialization: domain-specific keywords
    general_signals = sum(1 for kw in ["general", "any topic", "versatile", "broad", "flexible"]
                          if kw in prompt.lower())
    specific_signals = sum(1 for kw in ["specifically", "domain", "specialized", "expert in",
                                          "focus on", "only when"]
                           if kw in prompt.lower())
    specialization = max(0.0, min(1.0, 0.5 + (specific_signals - general_signals) * 0.1))

    return {
        "complexity": round(complexity, 3),
        "cost_efficiency": round(cost_efficiency, 3),
        "specialization": round(specialization, 3),
    }


# ── MAP-Elites Grid ──────────────────────────────────────────────────────────


class MAPElitesGrid:
    """Quality-diversity grid backed by in-memory dict + PostgreSQL persistence.

    Each cell: (bin_key) → StrategyEntry
    Only the BEST strategy per cell survives (elitism within niche).
    """

    def __init__(self, island_id: int = 0):
        self._grid: dict[tuple, StrategyEntry] = {}
        self._island_id = island_id
        self._lock = threading.Lock()

    def add(self, entry: StrategyEntry) -> bool:
        """Add strategy to grid. Returns True if it replaced an existing entry."""
        entry.island_id = self._island_id
        if not entry.feature_vector:
            entry.feature_vector = extract_features(entry.prompt_content)
        key = entry.bin_key

        with self._lock:
            existing = self._grid.get(key)
            if existing is None or entry.fitness_score > existing.fitness_score:
                self._grid[key] = entry
                return True
            return False

    def get_best_overall(self) -> Optional[StrategyEntry]:
        """Get the globally best strategy across all cells."""
        with self._lock:
            if not self._grid:
                return None
            return max(self._grid.values(), key=lambda e: e.fitness_score)

    def double_select(self) -> dict:
        """Double-selection: performance baseline + diverse inspiration.

        Returns: {performance: [top K], inspiration: [diverse K]}
        """
        with self._lock:
            entries = list(self._grid.values())

        if not entries:
            return {"performance": [], "inspiration": []}

        # Performance selection: top K by fitness
        performance = sorted(entries, key=lambda e: e.fitness_score, reverse=True)
        top_k = performance[:TOP_K_PERFORMANCE]

        # Inspiration selection: maximally diverse entries
        diverse = self._select_diverse(entries, DIVERSE_K_INSPIRATION)

        return {
            "performance": top_k,
            "inspiration": diverse,
        }

    def _select_diverse(self, entries: list[StrategyEntry], n: int) -> list[StrategyEntry]:
        """Greedy farthest-point sampling in feature space."""
        if len(entries) <= n:
            return list(entries)

        selected = [entries[0]]
        remaining = list(entries[1:])

        while len(selected) < n and remaining:
            best_dist = -1.0
            best_idx = 0
            for i, candidate in enumerate(remaining):
                min_dist = min(
                    self._feature_distance(candidate, s) for s in selected
                )
                if min_dist > best_dist:
                    best_dist = min_dist
                    best_idx = i
            selected.append(remaining.pop(best_idx))

        return selected

    def _feature_distance(self, a: StrategyEntry, b: StrategyEntry) -> float:
        """L2 distance in feature space."""
        dist_sq = 0.0
        for dim in FEATURE_DIMENSIONS:
            va = a.feature_vector.get(dim, 0.5)
            vb = b.feature_vector.get(dim, 0.5)
            dist_sq += (va - vb) ** 2
        return math.sqrt(dist_sq)

    @property
    def size(self) -> int:
        return len(self._grid)

    @property
    def coverage(self) -> float:
        """Fraction of grid cells that are filled."""
        total_cells = BINS_PER_DIM ** len(FEATURE_DIMENSIONS)
        return self.size / total_cells if total_cells > 0 else 0.0

    def get_all(self) -> list[StrategyEntry]:
        with self._lock:
            return list(self._grid.values())

    def to_dict(self) -> list[dict]:
        return [e.to_dict() for e in self._grid.values()]


# ── Artifact Feedback Manager ─────────────────────────────────────────────────


class ArtifactManager:
    """Manages generation-to-generation structured feedback.

    Collects execution artifacts (success/failure, scores, errors, suggestions)
    and formats them for injection into the next generation's mutation prompt.
    """

    def __init__(self, max_history: int = 5):
        self._history: dict[str, list[Artifact]] = {}  # role → artifacts
        self._max = max_history

    def record(self, role: str, artifact: Artifact) -> None:
        """Record an artifact for a role."""
        if role not in self._history:
            self._history[role] = []
        self._history[role].append(artifact)
        # Keep only recent
        self._history[role] = self._history[role][-self._max:]

    def get_feedback_context(self, role: str) -> str:
        """Format recent artifacts as prompt context for the next mutation."""
        artifacts = self._history.get(role, [])
        if not artifacts:
            return ""

        lines = [f"## Previous Execution Feedback (Last {len(artifacts)} Generations)\n"]
        for art in artifacts:
            lines.append(art.format_for_prompt())
            lines.append("")

        return "\n".join(lines)

    def get_latest(self, role: str) -> Optional[Artifact]:
        arts = self._history.get(role, [])
        return arts[-1] if arts else None


# ── Template Stochasticity ────────────────────────────────────────────────────

# IMMUTABLE: variation templates per agent role
PROMPT_VARIATIONS: dict[str, dict[str, list[str]]] = {
    "researcher": {
        "methodology": [
            "Start with a broad search, then narrow to specifics.",
            "Begin with the most authoritative sources, then expand.",
            "Cross-reference at least 3 independent sources.",
            "Prioritize recent publications over older sources.",
        ],
        "approach": [
            "Analyze the following research question systematically:",
            "Investigate this topic with academic rigor:",
            "Explore this question, prioritizing primary sources:",
        ],
    },
    "coder": {
        "approach": [
            "Write clean, well-documented code following best practices.",
            "Optimize for readability and maintainability first.",
            "Start with tests, then implement to pass them.",
            "Focus on correctness first, optimize later.",
        ],
        "style": [
            "Use type hints throughout.",
            "Include docstrings on all public functions.",
            "Keep functions under 30 lines where possible.",
        ],
    },
    "writer": {
        "tone": [
            "Write clearly and directly, avoiding jargon.",
            "Adapt your tone to the audience and context.",
            "Be precise with language — every word should earn its place.",
        ],
        "structure": [
            "Open with the key finding, then support with evidence.",
            "Use progressive disclosure: summary first, then details.",
            "Structure for scannability: headers, bullets, bold key terms.",
        ],
    },
    "commander": {
        "delegation": [
            "Delegate to the specialist most suited for each subtask.",
            "Prefer parallel crew execution when subtasks are independent.",
            "Consider task difficulty when choosing between direct answer and crew dispatch.",
        ],
    },
}


def apply_stochasticity(role: str, prompt: str) -> str:
    """Apply controlled randomization to a prompt by injecting variation snippets.

    Each execution gets a different combination, providing exploration pressure
    without modifying the core prompt structure.
    """
    variations = PROMPT_VARIATIONS.get(role, {})
    if not variations:
        return prompt

    # Select one variation per category
    selected = []
    for category, options in variations.items():
        choice = random.choice(options)
        selected.append(f"[{category.upper()}]: {choice}")

    if not selected:
        return prompt

    stochastic_block = "\n".join(selected)
    return f"{prompt}\n\n## Session-Specific Guidance\n{stochastic_block}"


# ── MAP-Elites Strategy Database (multi-island) ──────────────────────────────


class MAPElitesDB:
    """Multi-island MAP-Elites with migration, double-selection, and artifact feedback.

    Integrates:
        - MAP-Elites quality-diversity grid (per island)
        - Double-selection for performance + inspiration
        - Artifact feedback loop for generation-to-generation learning
        - Template stochasticity for controlled randomization
    """

    def __init__(self, role: str = "coder"):
        self._role = role
        self._islands = [MAPElitesGrid(island_id=i) for i in range(NUM_ISLANDS)]
        self._artifacts = ArtifactManager()
        self._generation = 0
        self._dir = Path(f"/app/workspace/map_elites/{role}")
        self._dir.mkdir(parents=True, exist_ok=True)

    def add_strategy(self, entry: StrategyEntry, island_id: int = 0) -> bool:
        """Add a strategy to an island's grid."""
        if 0 <= island_id < len(self._islands):
            return self._islands[island_id].add(entry)
        return False

    def record_artifact(self, artifact: Artifact) -> None:
        """Record execution feedback."""
        self._artifacts.record(self._role, artifact)

    def double_select(self, island_id: int = 0) -> dict:
        """Get performance baseline + diverse inspiration from an island."""
        if 0 <= island_id < len(self._islands):
            return self._islands[island_id].double_select()
        return {"performance": [], "inspiration": []}

    def get_mutation_context(self, island_id: int = 0) -> str:
        """Build full context for generating a mutation.

        Combines:
            1. Performance baseline (what to improve)
            2. Inspiration sources (diverse alternatives)
            3. Artifact feedback (what happened before)
        """
        selection = self.double_select(island_id)
        feedback = self._artifacts.get_feedback_context(self._role)

        lines = []

        # Performance baseline
        if selection["performance"]:
            best = selection["performance"][0]
            lines.append("## Current Best Strategy (to improve)")
            lines.append(f"Score: {best.fitness_score:.3f}")
            lines.append(f"Features: {best.feature_vector}")
            lines.append(f"```\n{best.prompt_content[:2000]}\n```")
            lines.append("")

        # Inspiration
        if selection["inspiration"]:
            lines.append("## Alternative Approaches (for inspiration only)")
            for i, insp in enumerate(selection["inspiration"]):
                lines.append(f"### Inspiration {i+1} (score={insp.fitness_score:.3f}, "
                             f"features={insp.feature_vector})")
                lines.append(f"```\n{insp.prompt_content[:1000]}\n```")
            lines.append("")

        # Artifact feedback
        if feedback:
            lines.append(feedback)

        return "\n".join(lines)

    def migrate(self) -> None:
        """Ring topology migration between islands."""
        for i in range(NUM_ISLANDS):
            target = (i + 1) % NUM_ISLANDS
            source_entries = self._islands[i].get_all()
            if not source_entries:
                continue

            # Top MIGRATION_COUNT by fitness
            top = sorted(source_entries, key=lambda e: e.fitness_score, reverse=True)
            for entry in top[:MIGRATION_COUNT]:
                self._islands[target].add(StrategyEntry(
                    strategy_id=hashlib.sha256(
                        f"{entry.strategy_id}_migrated_{time.time()}".encode()
                    ).hexdigest()[:12],
                    role=entry.role,
                    prompt_content=entry.prompt_content,
                    fitness_score=entry.fitness_score,
                    feature_vector=entry.feature_vector,
                    generation=entry.generation,
                    parent_id=entry.strategy_id,
                    mutation_type="migration",
                    artifacts=entry.artifacts.copy(),
                    created_at=datetime.now(timezone.utc).isoformat(),
                    island_id=target,
                ))

        logger.info(f"map_elites: migration complete for {self._role}")

    def apply_stochasticity(self, prompt: str) -> str:
        """Apply template stochasticity to a prompt."""
        return apply_stochasticity(self._role, prompt)

    def step_generation(self) -> None:
        self._generation += 1

    @property
    def generation(self) -> int:
        return self._generation

    def get_stats(self) -> dict:
        return {
            "role": self._role,
            "generation": self._generation,
            "islands": [
                {
                    "island_id": i,
                    "grid_size": self._islands[i].size,
                    "coverage": f"{self._islands[i].coverage:.1%}",
                    "best_fitness": (
                        self._islands[i].get_best_overall().fitness_score
                        if self._islands[i].get_best_overall() else 0
                    ),
                }
                for i in range(NUM_ISLANDS)
            ],
        }

    def persist(self) -> None:
        """Save state to disk."""
        state = {
            "role": self._role,
            "generation": self._generation,
            "islands": [isl.to_dict() for isl in self._islands],
        }
        path = self._dir / "state.json"
        from app.safe_io import safe_write_json
        safe_write_json(path, state)

    @classmethod
    def load(cls, role: str) -> "MAPElitesDB":
        """Load state from disk."""
        db = cls(role=role)
        path = db._dir / "state.json"
        if path.exists():
            try:
                state = json.loads(path.read_text())
                db._generation = state.get("generation", 0)
                for i, island_data in enumerate(state.get("islands", [])):
                    if i < NUM_ISLANDS:
                        for entry_data in island_data:
                            entry = StrategyEntry.from_dict(entry_data)
                            db._islands[i].add(entry)
            except Exception:
                logger.debug(f"map_elites: failed to load state for {role}", exc_info=True)
        return db

    def format_report(self) -> str:
        stats = self.get_stats()
        lines = [
            f"🗺️ MAP-Elites: {self._role} (gen {self._generation})",
            f"   Feature dims: {FEATURE_DIMENSIONS}",
            "",
        ]
        for isl in stats["islands"]:
            best = self._islands[isl["island_id"]].get_best_overall()
            best_info = f"best={best.fitness_score:.3f}" if best else "empty"
            lines.append(
                f"   Island {isl['island_id']}: {isl['grid_size']} cells "
                f"({isl['coverage']} coverage) {best_info}"
            )
        return "\n".join(lines)


# ── Module-level singletons ──────────────────────────────────────────────────

_databases: dict[str, MAPElitesDB] = {}
_db_lock = threading.Lock()


def get_db(role: str = "coder") -> MAPElitesDB:
    """Get or create a MAP-Elites database for a role."""
    with _db_lock:
        if role not in _databases:
            _databases[role] = MAPElitesDB.load(role)
        return _databases[role]
