"""
island_evolution.py — Island-based population evolution for agent prompts.

Multiple populations (islands) evolve prompt variants in parallel.
Top performers periodically migrate between islands, preventing
premature convergence while balancing diversity and exploitation.

Architecture:
    3 islands × 5 prompt variants each = 15 concurrent candidates
    Tournament selection within each island
    Migration: top 1 from each island → next island (ring topology)
    Fitness: sandboxed evaluation via eval_sandbox.py

Inspired by CodeEvolve's island-based GA adapted for prompt evolution.
Uses EVOLVE-BLOCK markers to constrain mutations to safe regions.

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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

ISLAND_DIR = Path("/app/workspace/island_evolution")

# ── IMMUTABLE configuration ──────────────────────────────────────────────────

NUM_ISLANDS = 3
POP_PER_ISLAND = 5
MIGRATION_INTERVAL = 5       # epochs between migrations
MIGRATION_COUNT = 1           # top N individuals to migrate
TOURNAMENT_SIZE = 3           # tournament selection size
MAX_EPOCHS_PER_SESSION = 20
STAGNATION_THRESHOLD = 5     # epochs without improvement → increase exploration
ELITISM_COUNT = 1             # top N preserved unchanged each epoch


# ── Data types ────────────────────────────────────────────────────────────────


@dataclass
class Individual:
    """A single prompt variant in a population."""
    prompt_id: str = ""
    prompt_content: str = ""
    fitness: float = 0.0
    generation: int = 0
    parent_id: str = ""
    mutation_type: str = ""    # "meta_prompt" | "inspiration" | "depth_exploit" | "random"
    island_id: int = 0
    created_at: str = ""
    ancestors: list[str] = field(default_factory=list)  # lineage for depth exploitation

    def to_dict(self) -> dict:
        return {
            "prompt_id": self.prompt_id,
            "fitness": self.fitness,
            "generation": self.generation,
            "parent_id": self.parent_id,
            "mutation_type": self.mutation_type,
            "island_id": self.island_id,
            "created_at": self.created_at,
            "ancestors": self.ancestors[-10:],  # Keep last 10
        }


@dataclass
class Island:
    """A population of prompt variants."""
    island_id: int = 0
    population: list[Individual] = field(default_factory=list)
    generation: int = 0
    best_fitness: float = 0.0
    stagnation_count: int = 0

    def add(self, individual: Individual) -> None:
        individual.island_id = self.island_id
        self.population.append(individual)
        # Keep population capped
        if len(self.population) > POP_PER_ISLAND * 2:
            self.population.sort(key=lambda i: i.fitness, reverse=True)
            self.population = self.population[:POP_PER_ISLAND]

    def select_parent(self) -> Individual:
        """Tournament selection."""
        if len(self.population) < TOURNAMENT_SIZE:
            return max(self.population, key=lambda i: i.fitness)
        tournament = random.sample(self.population, TOURNAMENT_SIZE)
        return max(tournament, key=lambda i: i.fitness)

    def select_survivors(self) -> None:
        """Keep top POP_PER_ISLAND individuals (elitism + tournament)."""
        if len(self.population) <= POP_PER_ISLAND:
            return

        self.population.sort(key=lambda i: i.fitness, reverse=True)

        # Elitism: keep top ELITISM_COUNT unconditionally
        survivors = self.population[:ELITISM_COUNT]
        remaining = self.population[ELITISM_COUNT:]

        # Tournament for remaining slots
        while len(survivors) < POP_PER_ISLAND and remaining:
            if len(remaining) <= TOURNAMENT_SIZE:
                survivors.extend(remaining)
                break
            tournament = random.sample(remaining, min(TOURNAMENT_SIZE, len(remaining)))
            winner = max(tournament, key=lambda i: i.fitness)
            survivors.append(winner)
            remaining.remove(winner)

        self.population = survivors[:POP_PER_ISLAND]

    def get_top_k(self, k: int = 3) -> list[Individual]:
        """Get top K individuals by fitness."""
        return sorted(self.population, key=lambda i: i.fitness, reverse=True)[:k]

    def update_stagnation(self) -> None:
        """Track if this island is stagnating."""
        current_best = max((i.fitness for i in self.population), default=0)
        if current_best > self.best_fitness + 0.01:  # Meaningful improvement
            self.best_fitness = current_best
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1


# ── Island Evolution Engine ──────────────────────────────────────────────────


class IslandEvolution:
    """Island-based population evolution for agent prompts.

    Usage:
        engine = IslandEvolution(target_role="coder")
        engine.initialize_population(current_prompt)
        results = engine.run_session(max_epochs=20)
        if results["best"]:
            promote(results["best"]["prompt_content"])
    """

    def __init__(self, target_role: str = "coder"):
        self._role = target_role
        self._islands: list[Island] = [
            Island(island_id=i) for i in range(NUM_ISLANDS)
        ]
        self._epoch = 0
        self._dir = ISLAND_DIR / target_role
        self._dir.mkdir(parents=True, exist_ok=True)

    def initialize_population(self, base_prompt: str) -> None:
        """Seed all islands with variants of the base prompt.

        Each island starts with the original + (POP_PER_ISLAND - 1) mutations.
        Different islands use different mutation strategies for diversity.
        """
        strategies = ["meta_prompt", "inspiration", "depth_exploit"]

        for island in self._islands:
            # Seed with original
            island.add(Individual(
                prompt_id=self._gen_id(),
                prompt_content=base_prompt,
                fitness=0.0,
                generation=0,
                mutation_type="seed",
                created_at=datetime.now(timezone.utc).isoformat(),
            ))

            # Generate initial variants
            strategy = strategies[island.island_id % len(strategies)]
            for _ in range(POP_PER_ISLAND - 1):
                variant = self._mutate(base_prompt, strategy, inspirations=[])
                if variant:
                    island.add(Individual(
                        prompt_id=self._gen_id(),
                        prompt_content=variant,
                        fitness=0.0,
                        generation=0,
                        mutation_type=strategy,
                        parent_id="seed",
                        created_at=datetime.now(timezone.utc).isoformat(),
                    ))

        logger.info(f"island_evolution: initialized {NUM_ISLANDS} islands × "
                    f"{POP_PER_ISLAND} variants for role '{self._role}'")

    def run_session(self, max_epochs: int = MAX_EPOCHS_PER_SESSION) -> dict:
        """Run an evolution session.

        Returns: {epochs_run, best: {prompt_content, fitness, ...}, stats}
        """
        from app.idle_scheduler import should_yield

        results = {
            "epochs_run": 0,
            "best": None,
            "stats": {},
        }

        for epoch in range(max_epochs):
            if should_yield():
                logger.info("island_evolution: yielding to user task")
                break

            self._epoch = epoch
            self._run_epoch()
            results["epochs_run"] = epoch + 1

            # Migration
            if epoch > 0 and epoch % MIGRATION_INTERVAL == 0:
                self._migrate()

            # Check stagnation across all islands
            all_stagnant = all(
                i.stagnation_count >= STAGNATION_THRESHOLD for i in self._islands
            )
            if all_stagnant:
                logger.info("island_evolution: all islands stagnant, ending session")
                break

        # Find global best
        all_individuals = []
        for island in self._islands:
            all_individuals.extend(island.population)

        if all_individuals:
            best = max(all_individuals, key=lambda i: i.fitness)
            results["best"] = {
                "prompt_content": best.prompt_content,
                "fitness": best.fitness,
                "generation": best.generation,
                "mutation_type": best.mutation_type,
                "island_id": best.island_id,
            }

        results["stats"] = self._get_stats()
        self._persist()

        return results

    def _run_epoch(self) -> None:
        """Run one epoch across all islands."""
        for island in self._islands:
            # Determine exploration rate based on stagnation
            exploration_rate = 0.3
            if island.stagnation_count >= STAGNATION_THRESHOLD:
                exploration_rate = 0.7  # Increase exploration when stagnating

            new_individuals = []
            for _ in range(POP_PER_ISLAND):
                parent = island.select_parent()

                # Choose mutation strategy
                if random.random() < exploration_rate:
                    strategy = "meta_prompt"
                else:
                    strategy = random.choice(["inspiration", "depth_exploit"])

                # Get inspirations from top performers
                inspirations = island.get_top_k(3)

                # Mutate
                new_content = self._mutate(
                    parent.prompt_content, strategy,
                    inspirations=[i.prompt_content for i in inspirations],
                    lineage=parent.ancestors + [parent.prompt_id],
                )

                if new_content and new_content != parent.prompt_content:
                    # Evaluate fitness
                    fitness = self._evaluate_fitness(new_content)

                    new_individuals.append(Individual(
                        prompt_id=self._gen_id(),
                        prompt_content=new_content,
                        fitness=fitness,
                        generation=self._epoch + 1,
                        parent_id=parent.prompt_id,
                        mutation_type=strategy,
                        island_id=island.island_id,
                        created_at=datetime.now(timezone.utc).isoformat(),
                        ancestors=(parent.ancestors + [parent.prompt_id])[-10:],
                    ))

            # Add new individuals and select survivors
            for ind in new_individuals:
                island.add(ind)
            island.select_survivors()
            island.generation = self._epoch + 1
            island.update_stagnation()

        best_per_island = [round(max((i.fitness for i in isl.population), default=0), 3)
                          for isl in self._islands]
        logger.debug(f"island_evolution: epoch {self._epoch} complete — "
                    f"best fitness per island: {best_per_island}")

    def _migrate(self) -> None:
        """Ring topology migration: top performer from each island → next island."""
        migrants = []
        for island in self._islands:
            top = island.get_top_k(MIGRATION_COUNT)
            migrants.append(top)

        for i, island in enumerate(self._islands):
            source_idx = (i - 1) % NUM_ISLANDS  # From previous island
            for migrant in migrants[source_idx]:
                # Clone to new island
                clone = Individual(
                    prompt_id=self._gen_id(),
                    prompt_content=migrant.prompt_content,
                    fitness=migrant.fitness,
                    generation=migrant.generation,
                    parent_id=migrant.prompt_id,
                    mutation_type="migration",
                    island_id=island.island_id,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    ancestors=migrant.ancestors.copy(),
                )
                island.add(clone)

        logger.info(f"island_evolution: migration complete (epoch {self._epoch})")

    # ── Mutation strategies ───────────────────────────────────────────

    def _mutate(
        self, prompt: str, strategy: str,
        inspirations: list[str] | None = None,
        lineage: list[str] | None = None,
    ) -> Optional[str]:
        """Apply a mutation strategy to generate a new prompt variant.

        Respects EVOLVE-BLOCK markers — only modifies evolvable regions.
        """
        from app.evolve_blocks import has_evolve_blocks, extract_evolvable_content, validate_modification

        try:
            from app.llm_factory import create_specialist_llm
            llm = create_specialist_llm(max_tokens=2048, role="self_improve")

            if strategy == "meta_prompt":
                new_prompt = self._meta_prompt_mutation(llm, prompt)
            elif strategy == "inspiration":
                new_prompt = self._inspiration_mutation(llm, prompt, inspirations or [])
            elif strategy == "depth_exploit":
                new_prompt = self._depth_exploitation(llm, prompt, lineage or [])
            else:
                new_prompt = self._random_mutation(llm, prompt)

            if not new_prompt or len(new_prompt) < 50:
                return None

            # Validate EVOLVE-BLOCK integrity
            if has_evolve_blocks(prompt):
                result = validate_modification(prompt, new_prompt)
                if not result["valid"]:
                    logger.debug(f"island_evolution: mutation rejected — {result['reason']}")
                    return None

            return new_prompt

        except Exception as e:
            logger.debug(f"island_evolution: mutation failed: {e}")
            return None

    def _meta_prompt_mutation(self, llm, prompt: str) -> str:
        """Meta-prompting: an auxiliary LLM rewrites the prompt to improve performance."""
        meta_instruction = (
            "You are a prompt optimization expert. Rewrite this agent system prompt "
            "to improve task success rate and response quality.\n\n"
            "CURRENT PROMPT:\n"
            f"{prompt[:3000]}\n\n"
            "CONSTRAINTS:\n"
            "- Only modify content between EVOLVE-BLOCK markers (if present)\n"
            "- Preserve all FREEZE-BLOCK content exactly\n"
            "- Keep the same overall structure and role identity\n"
            "- Make the instructions clearer, more specific, and actionable\n\n"
            "Return the complete rewritten prompt."
        )
        return str(llm.call(meta_instruction)).strip()

    def _inspiration_mutation(self, llm, prompt: str, inspirations: list[str]) -> str:
        """Inspiration-based crossover: synthesize from top performers."""
        if not inspirations:
            return self._meta_prompt_mutation(llm, prompt)

        insp_text = "\n\n---\n\n".join(
            f"## Inspiration {i+1}:\n{insp[:1500]}"
            for i, insp in enumerate(inspirations[:3])
        )

        instruction = (
            "You are improving an agent system prompt by drawing inspiration "
            "from multiple high-performing variants.\n\n"
            f"CURRENT PROMPT:\n{prompt[:2000]}\n\n"
            f"HIGH-PERFORMING INSPIRATION VARIANTS:\n{insp_text}\n\n"
            "Synthesize a new prompt that integrates the best patterns, "
            "specificity, and approaches from the inspirations. "
            "Do not blindly copy — understand WHY each works and apply those principles.\n\n"
            "CONSTRAINTS:\n"
            "- Only modify content between EVOLVE-BLOCK markers (if present)\n"
            "- Preserve all FREEZE-BLOCK content exactly\n\n"
            "Return the complete improved prompt."
        )
        return str(llm.call(instruction)).strip()

    def _depth_exploitation(self, llm, prompt: str, lineage: list[str]) -> str:
        """Depth exploitation: incremental refinement preserving working components."""
        lineage_note = ""
        if lineage:
            lineage_note = (
                f"\n\nThis prompt has been refined through {len(lineage)} generations. "
                "Focus on targeted, incremental improvements. Preserve everything "
                "that's working well. Only change what can be measurably improved."
            )

        instruction = (
            "You are making a targeted, minimal improvement to this agent system prompt. "
            "This is an incremental refinement — preserve all working components."
            f"{lineage_note}\n\n"
            f"CURRENT PROMPT:\n{prompt[:3000]}\n\n"
            "Make ONE specific improvement. Choose the single change most likely to "
            "improve task success rate. Explain your reasoning briefly, then return "
            "the complete updated prompt."
        )
        return str(llm.call(instruction)).strip()

    def _random_mutation(self, llm, prompt: str) -> str:
        """Random mutation for diversity."""
        mutations = [
            "Add a new specific instruction that would help with edge cases",
            "Reorder the instructions to put the most important ones first",
            "Add a concrete example of ideal behavior",
            "Clarify any ambiguous instructions with more specific language",
            "Add a constraint about output format or structure",
        ]
        chosen = random.choice(mutations)
        instruction = (
            f"Modify this agent system prompt. Specific change: {chosen}\n\n"
            f"PROMPT:\n{prompt[:3000]}\n\n"
            "Return the complete modified prompt."
        )
        return str(llm.call(instruction)).strip()

    # ── Fitness evaluation ────────────────────────────────────────────

    def _evaluate_fitness(self, prompt_content: str) -> float:
        """Evaluate a prompt variant's fitness using the eval sandbox."""
        try:
            from app.eval_sandbox import EvalSandbox
            from app.config import get_settings
            import app.prompt_registry as registry

            s = get_settings()
            if not s.mem0_postgres_url:
                return random.uniform(0.3, 0.7)  # Fallback: random fitness

            sandbox = EvalSandbox(s.mem0_postgres_url, registry)
            # Create a temporary modification to evaluate
            result = sandbox.evaluate_modification(
                role=self._role,
                proposed_content=prompt_content,
                modification_type="island_evolution",
            )

            if result.get("verdict") == "approve":
                return result.get("proposed_score", 0.5)
            elif result.get("verdict") == "reject":
                return max(0.1, result.get("proposed_score", 0.3))
            return 0.4

        except Exception as e:
            logger.debug(f"island_evolution: fitness eval failed: {e}")
            return random.uniform(0.2, 0.5)

    # ── Helpers ───────────────────────────────────────────────────────

    def _gen_id(self) -> str:
        return hashlib.sha256(
            f"{time.time()}{random.random()}".encode()
        ).hexdigest()[:12]

    def _get_stats(self) -> dict:
        stats = {"islands": []}
        for island in self._islands:
            fitnesses = [i.fitness for i in island.population]
            stats["islands"].append({
                "island_id": island.island_id,
                "population_size": len(island.population),
                "generation": island.generation,
                "best_fitness": max(fitnesses) if fitnesses else 0,
                "avg_fitness": sum(fitnesses) / len(fitnesses) if fitnesses else 0,
                "stagnation_count": island.stagnation_count,
            })
        return stats

    def _persist(self) -> None:
        """Save island state to disk."""
        state = {
            "role": self._role,
            "epoch": self._epoch,
            "islands": [],
        }
        for island in self._islands:
            state["islands"].append({
                "island_id": island.island_id,
                "generation": island.generation,
                "best_fitness": island.best_fitness,
                "stagnation_count": island.stagnation_count,
                "population": [ind.to_dict() for ind in island.population],
            })
        path = self._dir / "state.json"
        path.write_text(json.dumps(state, indent=2))

    def format_report(self) -> str:
        """Human-readable evolution report."""
        stats = self._get_stats()
        lines = [
            f"🧬 Island Evolution: {self._role} (epoch {self._epoch})",
            "",
        ]
        for isl in stats["islands"]:
            lines.append(
                f"  Island {isl['island_id']}: pop={isl['population_size']} "
                f"best={isl['best_fitness']:.3f} avg={isl['avg_fitness']:.3f} "
                f"stag={isl['stagnation_count']}"
            )
        return "\n".join(lines)


# ── Module-level entry point for idle scheduler ──────────────────────────────


def run_island_evolution_cycle(target_role: str = "coder") -> dict:
    """Run one island evolution session. Called by idle scheduler."""
    try:
        from app.prompt_registry import get_active_prompt
        base_prompt = get_active_prompt(target_role)
        if not base_prompt:
            return {"status": "no_prompt", "role": target_role}

        engine = IslandEvolution(target_role=target_role)
        engine.initialize_population(base_prompt)
        results = engine.run_session(max_epochs=MAX_EPOCHS_PER_SESSION)

        # Promote best if significantly better
        if results.get("best") and results["best"]["fitness"] > 0.7:
            try:
                from app.prompt_registry import propose_version, promote_version
                new_version = propose_version(
                    target_role,
                    results["best"]["prompt_content"],
                    f"Island evolution: fitness={results['best']['fitness']:.3f}, "
                    f"strategy={results['best']['mutation_type']}",
                )
                promote_version(target_role, new_version)
                logger.info(f"island_evolution: promoted v{new_version:03d} for {target_role} "
                            f"(fitness={results['best']['fitness']:.3f})")
            except Exception:
                logger.debug("island_evolution: promotion failed", exc_info=True)

        return results

    except Exception as e:
        logger.error(f"island_evolution: session failed: {e}")
        return {"status": "error", "error": str(e)[:200]}
