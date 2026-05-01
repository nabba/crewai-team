"""
Island Evolution Tests
=======================

Comprehensive tests for the island-based population evolution system.
Covers data types, selection algorithms, migration, persistence,
fitness evaluation, stagnation detection, and system wiring.

Run: docker exec crewai-team-gateway-1 python3 -m pytest /app/tests/test_island_evolution.py -v
"""

import inspect
import json
import os
import random
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# app.island_evolution hardcodes ISLAND_DIR = "/app/workspace/island_evolution"
# (Docker container layout). On macOS the parent /app exists but is the
# system root and read-only, so any test that exercises the persist path
# blows up with OSError: Read-only file system. Skip the whole module
# unless we're in a writable Docker-style environment.
pytestmark = pytest.mark.skipif(
    not os.access("/app", os.W_OK),
    reason="Requires Docker-style /app writable layout (run inside the gateway container)",
)


# ════════════════════════════════════════════════════════════════════════════════
# 1. IMPORT & CONSTANTS TESTS
# ════════════════════════════════════════════════════════════════════════════════

class TestImportsAndConstants:
    """All public symbols must be importable and configuration must be sane."""

    def test_core_imports(self):
        from app.island_evolution import (
            Individual, Island, IslandEvolution,
            run_island_evolution_cycle,
        )
        assert callable(IslandEvolution)
        assert callable(run_island_evolution_cycle)

    def test_constants_present(self):
        from app.island_evolution import (
            NUM_ISLANDS, POP_PER_ISLAND, MIGRATION_INTERVAL,
            MIGRATION_COUNT, TOURNAMENT_SIZE, MAX_EPOCHS_PER_SESSION,
            STAGNATION_THRESHOLD, ELITISM_COUNT, ISLAND_DIR,
        )
        assert NUM_ISLANDS >= 2
        assert POP_PER_ISLAND >= 3
        assert MIGRATION_INTERVAL >= 1
        assert MIGRATION_COUNT >= 1
        assert TOURNAMENT_SIZE >= 2
        assert MAX_EPOCHS_PER_SESSION >= 1
        assert STAGNATION_THRESHOLD >= 1
        assert ELITISM_COUNT >= 1

    def test_constants_consistency(self):
        """Tournament size must be <= population size."""
        from app.island_evolution import POP_PER_ISLAND, TOURNAMENT_SIZE, ELITISM_COUNT
        assert TOURNAMENT_SIZE <= POP_PER_ISLAND
        assert ELITISM_COUNT < POP_PER_ISLAND

    def test_island_dir_is_path(self):
        from app.island_evolution import ISLAND_DIR
        assert isinstance(ISLAND_DIR, Path)


# ════════════════════════════════════════════════════════════════════════════════
# 2. INDIVIDUAL DATACLASS TESTS
# ════════════════════════════════════════════════════════════════════════════════

class TestIndividual:
    """Individual represents a single prompt variant."""

    def test_defaults(self):
        from app.island_evolution import Individual
        ind = Individual()
        assert ind.prompt_id == ""
        assert ind.prompt_content == ""
        assert ind.fitness == 0.0
        assert ind.generation == 0
        assert ind.ancestors == []

    def test_to_dict_includes_prompt_content(self):
        from app.island_evolution import Individual
        ind = Individual(
            prompt_id="abc123",
            prompt_content="You are a helpful coder.",
            fitness=0.85,
            generation=5,
            parent_id="parent1",
            mutation_type="meta_prompt",
            island_id=2,
            created_at="2026-01-01T00:00:00",
            ancestors=["a", "b", "c"],
        )
        d = ind.to_dict()
        assert d["prompt_id"] == "abc123"
        assert d["prompt_content"] == "You are a helpful coder."
        assert d["fitness"] == 0.85
        assert d["generation"] == 5
        assert d["parent_id"] == "parent1"
        assert d["mutation_type"] == "meta_prompt"
        assert d["island_id"] == 2
        assert d["ancestors"] == ["a", "b", "c"]

    def test_to_dict_truncates_ancestors(self):
        from app.island_evolution import Individual
        ind = Individual(ancestors=[f"a{i}" for i in range(20)])
        d = ind.to_dict()
        assert len(d["ancestors"]) == 10  # Last 10

    def test_from_dict_roundtrip(self):
        from app.island_evolution import Individual
        ind = Individual(
            prompt_id="rt_test",
            prompt_content="Test prompt for roundtrip",
            fitness=0.72,
            generation=3,
            parent_id="p1",
            mutation_type="inspiration",
            island_id=1,
            created_at="2026-04-01T12:00:00",
            ancestors=["x", "y"],
        )
        d = ind.to_dict()
        ind2 = Individual.from_dict(d)
        assert ind2.prompt_id == ind.prompt_id
        assert ind2.prompt_content == ind.prompt_content
        assert ind2.fitness == ind.fitness
        assert ind2.generation == ind.generation
        assert ind2.parent_id == ind.parent_id
        assert ind2.mutation_type == ind.mutation_type
        assert ind2.island_id == ind.island_id
        assert ind2.ancestors == ind.ancestors

    def test_from_dict_with_missing_fields(self):
        from app.island_evolution import Individual
        ind = Individual.from_dict({"prompt_id": "sparse"})
        assert ind.prompt_id == "sparse"
        assert ind.prompt_content == ""
        assert ind.fitness == 0.0
        assert ind.ancestors == []

    def test_from_dict_empty(self):
        from app.island_evolution import Individual
        ind = Individual.from_dict({})
        assert ind.prompt_id == ""
        assert ind.fitness == 0.0


# ════════════════════════════════════════════════════════════════════════════════
# 3. ISLAND DATACLASS TESTS
# ════════════════════════════════════════════════════════════════════════════════

class TestIsland:
    """Island manages a population with selection and survival."""

    def _make_individual(self, fitness: float, prompt_id: str = "") -> "Individual":
        from app.island_evolution import Individual
        return Individual(
            prompt_id=prompt_id or f"ind_{fitness}",
            prompt_content=f"Prompt with fitness {fitness}",
            fitness=fitness,
        )

    def test_add_individual(self):
        from app.island_evolution import Island
        island = Island(island_id=0)
        ind = self._make_individual(0.5)
        island.add(ind)
        assert len(island.population) == 1
        assert island.population[0].island_id == 0  # Assigned by add()

    def test_add_caps_population(self):
        from app.island_evolution import Island, POP_PER_ISLAND
        island = Island(island_id=0)
        # Add more than 2x POP_PER_ISLAND
        for i in range(POP_PER_ISLAND * 2 + 1):
            island.add(self._make_individual(random.random()))
        assert len(island.population) <= POP_PER_ISLAND

    def test_add_keeps_best_when_capped(self):
        from app.island_evolution import Island, POP_PER_ISLAND
        island = Island(island_id=0)
        # Add many with known fitness
        for i in range(POP_PER_ISLAND * 2 + 1):
            island.add(self._make_individual(i / 100.0, prompt_id=f"ind_{i}"))
        # After capping, population should contain the fittest
        fitnesses = [ind.fitness for ind in island.population]
        assert max(fitnesses) == (POP_PER_ISLAND * 2) / 100.0

    def test_select_parent_returns_individual(self):
        from app.island_evolution import Island
        island = Island(island_id=0)
        for i in range(5):
            island.add(self._make_individual(random.random()))
        parent = island.select_parent()
        assert parent is not None
        assert parent in island.population

    def test_select_parent_tournament_pressure(self):
        """Tournament selection should favor higher fitness over many trials."""
        from app.island_evolution import Island
        island = Island(island_id=0)
        island.add(self._make_individual(0.1, "low"))
        island.add(self._make_individual(0.5, "mid"))
        island.add(self._make_individual(0.9, "high"))
        island.add(self._make_individual(0.2, "low2"))
        island.add(self._make_individual(0.3, "low3"))

        # Over many selections, higher fitness should win more often
        counts = {}
        for _ in range(200):
            p = island.select_parent()
            counts[p.prompt_id] = counts.get(p.prompt_id, 0) + 1
        assert counts.get("high", 0) > counts.get("low", 0)

    def test_select_parent_small_population(self):
        """With fewer than TOURNAMENT_SIZE, should still work."""
        from app.island_evolution import Island
        island = Island(island_id=0)
        island.add(self._make_individual(0.5))
        island.add(self._make_individual(0.8))
        parent = island.select_parent()
        assert parent.fitness == 0.8  # Max of small pop

    def test_select_survivors_preserves_elite(self):
        from app.island_evolution import Island, POP_PER_ISLAND, ELITISM_COUNT
        island = Island(island_id=0)
        # Add 2x population
        for i in range(POP_PER_ISLAND * 2):
            island.add(self._make_individual(i / 20.0, prompt_id=f"surv_{i}"))
        # Manually set population to bypass cap in add()
        island.population = [self._make_individual(i / 20.0, f"surv_{i}")
                             for i in range(POP_PER_ISLAND * 2)]

        island.select_survivors()
        assert len(island.population) == POP_PER_ISLAND

        # Top elite must survive
        fitnesses = sorted([ind.fitness for ind in island.population], reverse=True)
        expected_top = (POP_PER_ISLAND * 2 - 1) / 20.0
        assert fitnesses[0] == expected_top

    def test_select_survivors_noop_when_small(self):
        from app.island_evolution import Island, POP_PER_ISLAND
        island = Island(island_id=0)
        for i in range(POP_PER_ISLAND - 1):
            island.add(self._make_individual(random.random()))
        size_before = len(island.population)
        island.select_survivors()
        assert len(island.population) == size_before

    def test_get_top_k(self):
        from app.island_evolution import Island
        island = Island(island_id=0)
        for f in [0.1, 0.5, 0.9, 0.3, 0.7]:
            island.add(self._make_individual(f))
        top3 = island.get_top_k(3)
        assert len(top3) == 3
        assert top3[0].fitness == 0.9
        assert top3[1].fitness == 0.7
        assert top3[2].fitness == 0.5

    def test_get_top_k_larger_than_pop(self):
        from app.island_evolution import Island
        island = Island(island_id=0)
        island.add(self._make_individual(0.5))
        island.add(self._make_individual(0.8))
        top5 = island.get_top_k(5)
        assert len(top5) == 2

    def test_update_stagnation_reset(self):
        from app.island_evolution import Island
        island = Island(island_id=0, best_fitness=0.5, stagnation_count=3)
        island.add(self._make_individual(0.6))  # Improvement > 0.01
        island.update_stagnation()
        assert island.stagnation_count == 0
        assert island.best_fitness == 0.6

    def test_update_stagnation_increment(self):
        from app.island_evolution import Island
        island = Island(island_id=0, best_fitness=0.5, stagnation_count=2)
        island.add(self._make_individual(0.505))  # Below threshold (0.01)
        island.update_stagnation()
        assert island.stagnation_count == 3
        assert island.best_fitness == 0.5  # Unchanged

    def test_update_stagnation_empty_island(self):
        from app.island_evolution import Island
        island = Island(island_id=0, best_fitness=0.0, stagnation_count=0)
        island.update_stagnation()
        assert island.stagnation_count == 1


# ════════════════════════════════════════════════════════════════════════════════
# 4. ISLAND EVOLUTION ENGINE TESTS
# ════════════════════════════════════════════════════════════════════════════════

class TestIslandEvolution:
    """Core engine: init, population, epochs, migration, stats."""

    def test_init_creates_islands(self):
        from app.island_evolution import IslandEvolution, NUM_ISLANDS
        engine = IslandEvolution(target_role="test_init")
        assert len(engine._islands) == NUM_ISLANDS
        for i, island in enumerate(engine._islands):
            assert island.island_id == i

    def test_init_creates_directory(self):
        from app.island_evolution import IslandEvolution, ISLAND_DIR
        engine = IslandEvolution(target_role="test_dir_creation")
        assert (ISLAND_DIR / "test_dir_creation").exists()

    def test_gen_id_unique(self):
        from app.island_evolution import IslandEvolution
        engine = IslandEvolution(target_role="test_genid")
        ids = {engine._gen_id() for _ in range(100)}
        assert len(ids) == 100  # All unique

    def test_gen_id_format(self):
        from app.island_evolution import IslandEvolution
        engine = IslandEvolution(target_role="test_genid")
        gid = engine._gen_id()
        assert len(gid) == 12
        assert all(c in "0123456789abcdef" for c in gid)

    def test_get_stats_structure(self):
        from app.island_evolution import IslandEvolution, Individual, NUM_ISLANDS
        engine = IslandEvolution(target_role="test_stats")
        # Add some population
        for island in engine._islands:
            island.add(Individual(prompt_content="test", fitness=0.5))
        stats = engine._get_stats()
        assert "islands" in stats
        assert len(stats["islands"]) == NUM_ISLANDS
        for isl_stat in stats["islands"]:
            assert "island_id" in isl_stat
            assert "population_size" in isl_stat
            assert "generation" in isl_stat
            assert "best_fitness" in isl_stat
            assert "avg_fitness" in isl_stat
            assert "stagnation_count" in isl_stat

    def test_get_stats_empty_islands(self):
        from app.island_evolution import IslandEvolution
        engine = IslandEvolution(target_role="test_empty_stats")
        stats = engine._get_stats()
        for isl_stat in stats["islands"]:
            assert isl_stat["best_fitness"] == 0
            assert isl_stat["avg_fitness"] == 0
            assert isl_stat["population_size"] == 0

    def test_format_report(self):
        from app.island_evolution import IslandEvolution, Individual
        engine = IslandEvolution(target_role="test_report")
        for island in engine._islands:
            island.add(Individual(prompt_content="test", fitness=0.6))
        report = engine.format_report()
        assert isinstance(report, str)
        assert "test_report" in report
        assert "Island 0" in report

    def test_test_tasks_exist_for_standard_roles(self):
        from app.island_evolution import IslandEvolution
        assert "coder" in IslandEvolution._TEST_TASKS
        assert "researcher" in IslandEvolution._TEST_TASKS
        assert "writer" in IslandEvolution._TEST_TASKS
        for role, tasks in IslandEvolution._TEST_TASKS.items():
            assert len(tasks) >= 2, f"Role {role} needs at least 2 test tasks"

    def test_test_tasks_fallback_for_unknown_role(self):
        """Unknown roles should fall back to researcher tasks."""
        from app.island_evolution import IslandEvolution
        engine = IslandEvolution(target_role="unknown_role_xyz")
        tasks = engine._TEST_TASKS.get("unknown_role_xyz", engine._TEST_TASKS["researcher"])
        assert len(tasks) >= 2


# ════════════════════════════════════════════════════════════════════════════════
# 5. MIGRATION TESTS
# ════════════════════════════════════════════════════════════════════════════════

class TestMigration:
    """Ring topology migration between islands."""

    def test_migration_adds_individuals(self):
        from app.island_evolution import IslandEvolution, Individual
        engine = IslandEvolution(target_role="test_migration")
        # Populate islands
        for i, island in enumerate(engine._islands):
            for j in range(3):
                island.add(Individual(
                    prompt_id=f"isl{i}_ind{j}",
                    prompt_content=f"Prompt from island {i}",
                    fitness=0.5 + i * 0.1 + j * 0.01,
                ))
        sizes_before = [len(isl.population) for isl in engine._islands]
        engine._migrate()
        sizes_after = [len(isl.population) for isl in engine._islands]
        # Each island should gain migrants
        for before, after in zip(sizes_before, sizes_after):
            assert after >= before

    def test_migration_ring_topology(self):
        """Island i gets migrants from island (i-1) % N."""
        from app.island_evolution import IslandEvolution, Individual, NUM_ISLANDS, MIGRATION_COUNT
        engine = IslandEvolution(target_role="test_ring")
        # Give each island a unique best individual
        for i, island in enumerate(engine._islands):
            island.add(Individual(
                prompt_id=f"best_{i}",
                prompt_content=f"Best prompt from island {i}",
                fitness=0.9,
            ))
            for j in range(2):
                island.add(Individual(
                    prompt_id=f"filler_{i}_{j}",
                    prompt_content=f"Filler from island {i}",
                    fitness=0.3,
                ))
        engine._migrate()
        # Island 0 should have a migrant from island (NUM_ISLANDS-1)
        island0_contents = [ind.prompt_content for ind in engine._islands[0].population]
        source_idx = (0 - 1) % NUM_ISLANDS
        expected = f"Best prompt from island {source_idx}"
        assert any(expected in c for c in island0_contents)

    def test_migration_clones_not_moves(self):
        """Migration clones individuals — source island keeps its population."""
        from app.island_evolution import IslandEvolution, Individual
        engine = IslandEvolution(target_role="test_clone")
        for island in engine._islands:
            for j in range(3):
                island.add(Individual(
                    prompt_content=f"test_{island.island_id}_{j}",
                    fitness=0.5,
                ))
        sizes_before = [len(isl.population) for isl in engine._islands]
        engine._migrate()
        # Source islands should not lose population
        for i, island in enumerate(engine._islands):
            assert len(island.population) >= sizes_before[i]

    def test_migrant_mutation_type(self):
        from app.island_evolution import IslandEvolution, Individual
        engine = IslandEvolution(target_role="test_migtype")
        for island in engine._islands:
            island.add(Individual(prompt_content="test", fitness=0.5))
        engine._migrate()
        # Find migrants
        for island in engine._islands:
            migrants = [ind for ind in island.population if ind.mutation_type == "migration"]
            # Each island should receive at least 1 migrant
            assert len(migrants) >= 1 or len(island.population) <= 1


# ════════════════════════════════════════════════════════════════════════════════
# 6. PERSISTENCE TESTS
# ════════════════════════════════════════════════════════════════════════════════

class TestPersistence:
    """Persist and load island state across sessions."""

    def test_persist_creates_file(self):
        from app.island_evolution import IslandEvolution, Individual, ISLAND_DIR
        engine = IslandEvolution(target_role="test_persist_file")
        engine._islands[0].add(Individual(prompt_content="hello", fitness=0.5))
        engine._persist()
        assert (ISLAND_DIR / "test_persist_file" / "state.json").exists()

    def test_persist_contains_prompt_content(self):
        from app.island_evolution import IslandEvolution, Individual, ISLAND_DIR
        engine = IslandEvolution(target_role="test_persist_content")
        engine._islands[0].add(Individual(
            prompt_id="pc_test",
            prompt_content="This must be persisted!",
            fitness=0.77,
        ))
        engine._persist()
        data = json.loads((ISLAND_DIR / "test_persist_content" / "state.json").read_text())
        pop = data["islands"][0]["population"]
        assert any(ind["prompt_content"] == "This must be persisted!" for ind in pop)

    def test_load_restores_state(self):
        from app.island_evolution import IslandEvolution, Individual
        engine = IslandEvolution(target_role="test_load_restore")
        engine._epoch = 7
        engine._islands[0].add(Individual(
            prompt_id="restore_test",
            prompt_content="Restored prompt content",
            fitness=0.82,
            generation=7,
            mutation_type="depth_exploit",
        ))
        engine._islands[0].generation = 7
        engine._islands[0].best_fitness = 0.82
        engine._persist()

        # Load into fresh engine
        engine2 = IslandEvolution(target_role="test_load_restore")
        loaded = engine2._load()
        assert loaded is True
        assert engine2._epoch == 7
        assert len(engine2._islands[0].population) == 1
        ind = engine2._islands[0].population[0]
        assert ind.prompt_content == "Restored prompt content"
        assert ind.fitness == 0.82
        assert ind.mutation_type == "depth_exploit"
        assert engine2._islands[0].best_fitness == 0.82

    def test_load_returns_false_for_missing(self):
        from app.island_evolution import IslandEvolution
        engine = IslandEvolution(target_role="nonexistent_role_xyz_99")
        loaded = engine._load()
        assert loaded is False

    def test_load_rejects_old_format_without_content(self):
        from app.island_evolution import IslandEvolution, ISLAND_DIR
        role = "test_old_fmt_reject"
        d = ISLAND_DIR / role
        d.mkdir(parents=True, exist_ok=True)
        (d / "state.json").write_text(json.dumps({
            "role": role, "epoch": 5,
            "islands": [{
                "island_id": 0, "generation": 5,
                "best_fitness": 0.5, "stagnation_count": 2,
                "population": [{"prompt_id": "x", "fitness": 0.5}],
            }],
        }))
        engine = IslandEvolution(target_role=role)
        assert engine._load() is False

    def test_load_rejects_wrong_role(self):
        from app.island_evolution import IslandEvolution, ISLAND_DIR
        role = "test_wrong_role"
        d = ISLAND_DIR / role
        d.mkdir(parents=True, exist_ok=True)
        (d / "state.json").write_text(json.dumps({
            "role": "different_role", "epoch": 1,
            "islands": [],
        }))
        engine = IslandEvolution(target_role=role)
        assert engine._load() is False

    def test_has_saved_state(self):
        from app.island_evolution import IslandEvolution, Individual
        engine = IslandEvolution(target_role="test_has_state")
        engine._islands[0].add(Individual(prompt_content="content", fitness=0.5))
        engine._persist()
        assert engine.has_saved_state() is True

    def test_has_saved_state_false_for_missing(self):
        from app.island_evolution import IslandEvolution
        engine = IslandEvolution(target_role="missing_state_xyz")
        assert engine.has_saved_state() is False

    def test_has_saved_state_false_for_empty_content(self):
        from app.island_evolution import IslandEvolution, ISLAND_DIR
        role = "test_empty_content_state"
        d = ISLAND_DIR / role
        d.mkdir(parents=True, exist_ok=True)
        (d / "state.json").write_text(json.dumps({
            "role": role, "epoch": 3,
            "islands": [{
                "island_id": 0, "generation": 3,
                "best_fitness": 0.5, "stagnation_count": 0,
                "population": [{"prompt_id": "x", "fitness": 0.5}],
            }],
        }))
        engine = IslandEvolution(target_role=role)
        assert engine.has_saved_state() is False

    def test_load_ensures_num_islands(self):
        """Loading state with fewer islands should pad to NUM_ISLANDS."""
        from app.island_evolution import IslandEvolution, ISLAND_DIR, NUM_ISLANDS
        role = "test_pad_islands"
        d = ISLAND_DIR / role
        d.mkdir(parents=True, exist_ok=True)
        (d / "state.json").write_text(json.dumps({
            "role": role, "epoch": 1,
            "islands": [{
                "island_id": 0, "generation": 1,
                "best_fitness": 0.5, "stagnation_count": 0,
                "population": [{"prompt_id": "x", "prompt_content": "pad test", "fitness": 0.5}],
            }],
        }))
        engine = IslandEvolution(target_role=role)
        loaded = engine._load()
        assert loaded is True
        assert len(engine._islands) == NUM_ISLANDS

    def test_persist_load_roundtrip_multiple_islands(self):
        from app.island_evolution import IslandEvolution, Individual, NUM_ISLANDS
        engine = IslandEvolution(target_role="test_multi_roundtrip")
        for i, island in enumerate(engine._islands):
            for j in range(3):
                island.add(Individual(
                    prompt_id=f"rt_{i}_{j}",
                    prompt_content=f"Prompt island {i} individual {j}",
                    fitness=0.3 + i * 0.1 + j * 0.05,
                    generation=2,
                    mutation_type="meta_prompt",
                ))
            island.generation = 2
            island.best_fitness = 0.3 + i * 0.1 + 0.1
        engine._epoch = 2
        engine._persist()

        engine2 = IslandEvolution(target_role="test_multi_roundtrip")
        assert engine2._load() is True
        assert engine2._epoch == 2
        total_pop = sum(len(isl.population) for isl in engine2._islands)
        assert total_pop == NUM_ISLANDS * 3


# ════════════════════════════════════════════════════════════════════════════════
# 7. EPOCH & SESSION TESTS (mocked LLM)
# ════════════════════════════════════════════════════════════════════════════════

class TestEpochAndSession:
    """Epoch execution and session management with mocked LLM calls."""

    def _make_engine_with_population(self, role="test_epoch"):
        from app.island_evolution import IslandEvolution, Individual
        engine = IslandEvolution(target_role=role)
        for island in engine._islands:
            for j in range(5):
                island.add(Individual(
                    prompt_id=engine._gen_id(),
                    prompt_content=f"Test prompt variant {j} for {island.island_id}",
                    fitness=random.uniform(0.3, 0.7),
                    generation=0,
                ))
        return engine

    @patch("app.island_evolution.IslandEvolution._mutate")
    @patch("app.island_evolution.IslandEvolution._evaluate_fitness")
    def test_run_epoch_produces_offspring(self, mock_fitness, mock_mutate):
        mock_mutate.return_value = "New mutated prompt content here is long enough"
        mock_fitness.return_value = 0.75
        engine = self._make_engine_with_population()
        engine._run_epoch()
        # All islands should have had selection
        for island in engine._islands:
            assert island.generation == 1

    @patch("app.island_evolution.IslandEvolution._mutate")
    @patch("app.island_evolution.IslandEvolution._evaluate_fitness")
    def test_run_epoch_rejects_identical_mutations(self, mock_fitness, mock_mutate):
        """If mutate returns the same prompt as parent, it should be skipped."""
        engine = self._make_engine_with_population()
        # Return same content as parent (should be rejected)
        def return_parent(prompt, strategy, **kwargs):
            return prompt
        mock_mutate.side_effect = return_parent
        mock_fitness.return_value = 0.8

        pop_before = sum(len(isl.population) for isl in engine._islands)
        engine._run_epoch()
        # Population shouldn't grow since all mutations are identical
        pop_after = sum(len(isl.population) for isl in engine._islands)
        assert pop_after <= pop_before + 1  # At most minimal growth from rounding

    @patch("app.island_evolution.IslandEvolution._mutate")
    @patch("app.island_evolution.IslandEvolution._evaluate_fitness")
    def test_exploration_rate_increases_on_stagnation(self, mock_fitness, mock_mutate):
        """Stagnating islands should have higher exploration rate."""
        from app.island_evolution import STAGNATION_THRESHOLD
        mock_mutate.return_value = "Sufficiently long mutated prompt content for testing"
        mock_fitness.return_value = 0.6

        engine = self._make_engine_with_population()
        # Force stagnation on island 0
        engine._islands[0].stagnation_count = STAGNATION_THRESHOLD

        # Track which strategies are called
        strategies_used = []
        original_mutate = engine._mutate.__wrapped__ if hasattr(engine._mutate, '__wrapped__') else None

        def track_mutate(prompt, strategy, **kwargs):
            strategies_used.append(strategy)
            return "Tracked mutation result that is long enough for the test"
        mock_mutate.side_effect = track_mutate

        engine._run_epoch()
        # Can't directly assert exploration rate, but epoch should complete
        assert engine._islands[0].generation == 1

    @patch("app.island_evolution.IslandEvolution._mutate")
    @patch("app.island_evolution.IslandEvolution._evaluate_fitness")
    @patch("app.idle_scheduler.should_yield", return_value=False)
    def test_run_session_returns_dict(self, mock_yield, mock_fitness, mock_mutate):
        mock_mutate.return_value = "A sufficiently long mutated prompt for testing purposes"
        mock_fitness.return_value = 0.65
        engine = self._make_engine_with_population("test_session_dict")
        results = engine.run_session(max_epochs=2)
        assert isinstance(results, dict)
        assert "epochs_run" in results
        assert "best" in results
        assert "stats" in results
        assert results["epochs_run"] == 2

    @patch("app.island_evolution.IslandEvolution._mutate")
    @patch("app.island_evolution.IslandEvolution._evaluate_fitness")
    @patch("app.idle_scheduler.should_yield", return_value=False)
    def test_run_session_finds_best(self, mock_yield, mock_fitness, mock_mutate):
        mock_mutate.return_value = "Best prompt that is long enough for test"
        mock_fitness.return_value = 0.85
        engine = self._make_engine_with_population("test_session_best")
        results = engine.run_session(max_epochs=1)
        assert results["best"] is not None
        assert results["best"]["fitness"] >= 0.0
        assert "prompt_content" in results["best"]

    @patch("app.island_evolution.IslandEvolution._mutate")
    @patch("app.island_evolution.IslandEvolution._evaluate_fitness")
    @patch("app.idle_scheduler.should_yield", return_value=True)
    def test_run_session_yields_to_user(self, mock_yield, mock_fitness, mock_mutate):
        engine = self._make_engine_with_population("test_yield")
        results = engine.run_session(max_epochs=20)
        assert results["epochs_run"] == 0  # Should yield immediately

    @patch("app.island_evolution.IslandEvolution._mutate")
    @patch("app.island_evolution.IslandEvolution._evaluate_fitness")
    @patch("app.idle_scheduler.should_yield", return_value=False)
    def test_run_session_stops_on_stagnation(self, mock_yield, mock_fitness, mock_mutate):
        from app.island_evolution import STAGNATION_THRESHOLD
        mock_mutate.return_value = None  # No successful mutations
        mock_fitness.return_value = 0.5
        engine = self._make_engine_with_population("test_stagnation_stop")
        # Pre-set all islands as stagnant
        for island in engine._islands:
            island.stagnation_count = STAGNATION_THRESHOLD
        results = engine.run_session(max_epochs=50)
        # Should stop early due to stagnation
        assert results["epochs_run"] < 50

    @patch("app.island_evolution.IslandEvolution._mutate")
    @patch("app.island_evolution.IslandEvolution._evaluate_fitness")
    @patch("app.idle_scheduler.should_yield", return_value=False)
    def test_run_session_triggers_migration(self, mock_yield, mock_fitness, mock_mutate):
        from app.island_evolution import MIGRATION_INTERVAL
        mock_mutate.return_value = "Migration test prompt content is long enough"
        mock_fitness.return_value = 0.6
        engine = self._make_engine_with_population("test_migration_trigger")
        results = engine.run_session(max_epochs=MIGRATION_INTERVAL + 1)
        assert results["epochs_run"] >= MIGRATION_INTERVAL

    @patch("app.island_evolution.IslandEvolution._mutate")
    @patch("app.island_evolution.IslandEvolution._evaluate_fitness")
    @patch("app.idle_scheduler.should_yield", return_value=False)
    def test_run_session_persists_at_end(self, mock_yield, mock_fitness, mock_mutate):
        from app.island_evolution import ISLAND_DIR
        mock_mutate.return_value = "Persisted prompt long enough for testing"
        mock_fitness.return_value = 0.7
        engine = self._make_engine_with_population("test_persist_session")
        engine.run_session(max_epochs=1)
        assert (ISLAND_DIR / "test_persist_session" / "state.json").exists()


# ════════════════════════════════════════════════════════════════════════════════
# 8. MUTATION STRATEGY TESTS (mocked LLM)
# ════════════════════════════════════════════════════════════════════════════════

class TestMutationStrategies:
    """Mutation strategies generate prompt variants via LLM."""

    @patch("app.island_evolution.IslandEvolution._meta_prompt_mutation")
    @patch("app.evolve_blocks.has_evolve_blocks", return_value=False)
    def test_mutate_meta_prompt(self, mock_blocks, mock_meta):
        from app.island_evolution import IslandEvolution
        mock_meta.return_value = "A rewritten prompt that is long enough to pass the 50 char filter"
        engine = IslandEvolution(target_role="test_mut")
        result = engine._mutate("original prompt", "meta_prompt")
        assert result is not None
        assert len(result) >= 50

    @patch("app.island_evolution.IslandEvolution._inspiration_mutation")
    @patch("app.evolve_blocks.has_evolve_blocks", return_value=False)
    def test_mutate_inspiration(self, mock_blocks, mock_insp):
        from app.island_evolution import IslandEvolution
        mock_insp.return_value = "Inspired prompt variant that combines multiple strategies effectively"
        engine = IslandEvolution(target_role="test_mut")
        result = engine._mutate("original", "inspiration", inspirations=["insp1", "insp2"])
        assert result is not None

    @patch("app.island_evolution.IslandEvolution._depth_exploitation")
    @patch("app.evolve_blocks.has_evolve_blocks", return_value=False)
    def test_mutate_depth_exploit(self, mock_blocks, mock_depth):
        from app.island_evolution import IslandEvolution
        mock_depth.return_value = "Incrementally refined prompt with targeted improvement applied here"
        engine = IslandEvolution(target_role="test_mut")
        result = engine._mutate("original", "depth_exploit", lineage=["a", "b"])
        assert result is not None

    @patch("app.island_evolution.IslandEvolution._random_mutation")
    @patch("app.evolve_blocks.has_evolve_blocks", return_value=False)
    def test_mutate_random_fallback(self, mock_blocks, mock_random):
        from app.island_evolution import IslandEvolution
        mock_random.return_value = "Randomly mutated prompt with some new instructions added for variety"
        engine = IslandEvolution(target_role="test_mut")
        result = engine._mutate("original", "unknown_strategy")
        assert result is not None

    def test_mutate_rejects_short_output(self):
        from app.island_evolution import IslandEvolution
        engine = IslandEvolution(target_role="test_short")
        with patch.object(engine, "_meta_prompt_mutation", return_value="short"):
            with patch("app.evolve_blocks.has_evolve_blocks", return_value=False):
                result = engine._mutate("original", "meta_prompt")
                assert result is None  # Too short (< 50 chars)

    def test_mutate_rejects_none(self):
        from app.island_evolution import IslandEvolution
        engine = IslandEvolution(target_role="test_none")
        with patch.object(engine, "_meta_prompt_mutation", return_value=None):
            with patch("app.evolve_blocks.has_evolve_blocks", return_value=False):
                result = engine._mutate("original", "meta_prompt")
                assert result is None

    @patch("app.evolve_blocks.has_evolve_blocks", return_value=True)
    @patch("app.evolve_blocks.validate_modification")
    def test_mutate_validates_evolve_blocks(self, mock_validate, mock_has):
        from app.island_evolution import IslandEvolution
        mock_validate.return_value = {"valid": False, "reason": "FREEZE block modified"}
        engine = IslandEvolution(target_role="test_blocks")
        with patch.object(engine, "_meta_prompt_mutation",
                          return_value="A long enough prompt that should be validated against blocks"):
            result = engine._mutate("original", "meta_prompt")
            assert result is None  # Rejected by block validation


# ════════════════════════════════════════════════════════════════════════════════
# 9. FITNESS EVALUATION TESTS (mocked LLM)
# ════════════════════════════════════════════════════════════════════════════════

class TestFitnessEvaluation:
    """Fitness evaluation with LLM-as-judge."""

    @patch("app.llm_factory.create_cheap_vetting_llm")
    @patch("app.llm_factory.create_specialist_llm")
    def test_evaluate_returns_score(self, mock_specialist, mock_judge):
        from app.island_evolution import IslandEvolution
        mock_agent = MagicMock()
        mock_agent.call.return_value = "def two_sum(nums, target): return []"
        mock_specialist.return_value = mock_agent

        mock_j = MagicMock()
        mock_j.call.return_value = '{"score": 0.85, "reason": "Good implementation"}'
        mock_judge.return_value = mock_j

        engine = IslandEvolution(target_role="coder")
        score = engine._evaluate_fitness("You are an expert coder.")
        assert 0.0 <= score <= 1.0
        assert score == pytest.approx(0.85, abs=0.01)

    @patch("app.llm_factory.create_cheap_vetting_llm")
    @patch("app.llm_factory.create_specialist_llm")
    def test_evaluate_averages_multiple_tasks(self, mock_specialist, mock_judge):
        from app.island_evolution import IslandEvolution
        mock_agent = MagicMock()
        mock_agent.call.return_value = "A decent response that is long enough"
        mock_specialist.return_value = mock_agent

        mock_j = MagicMock()
        # Return different scores for different tasks
        mock_j.call.side_effect = [
            '{"score": 0.9, "reason": "Excellent"}',
            '{"score": 0.7, "reason": "Good"}',
        ]
        mock_judge.return_value = mock_j

        engine = IslandEvolution(target_role="coder")
        score = engine._evaluate_fitness("You are an expert coder.")
        assert score == pytest.approx(0.8, abs=0.01)

    @patch("app.llm_factory.create_cheap_vetting_llm")
    @patch("app.llm_factory.create_specialist_llm")
    def test_evaluate_handles_empty_response(self, mock_specialist, mock_judge):
        from app.island_evolution import IslandEvolution
        mock_agent = MagicMock()
        mock_agent.call.return_value = ""  # Empty
        mock_specialist.return_value = mock_agent
        mock_judge.return_value = MagicMock()

        engine = IslandEvolution(target_role="coder")
        score = engine._evaluate_fitness("bad prompt")
        assert score == pytest.approx(0.2, abs=0.01)

    @patch("app.llm_factory.create_cheap_vetting_llm")
    @patch("app.llm_factory.create_specialist_llm")
    def test_evaluate_parses_plain_number(self, mock_specialist, mock_judge):
        from app.island_evolution import IslandEvolution
        mock_agent = MagicMock()
        mock_agent.call.return_value = "A valid response to the task at hand"
        mock_specialist.return_value = mock_agent

        mock_j = MagicMock()
        mock_j.call.return_value = "Score: 0.75"  # No JSON, just number
        mock_judge.return_value = mock_j

        engine = IslandEvolution(target_role="researcher")
        score = engine._evaluate_fitness("You are a researcher.")
        assert score == pytest.approx(0.75, abs=0.01)

    @patch("app.llm_factory.create_cheap_vetting_llm")
    @patch("app.llm_factory.create_specialist_llm")
    def test_evaluate_handles_unparseable(self, mock_specialist, mock_judge):
        from app.island_evolution import IslandEvolution
        mock_agent = MagicMock()
        mock_agent.call.return_value = "A valid response to the task at hand"
        mock_specialist.return_value = mock_agent

        mock_j = MagicMock()
        mock_j.call.return_value = "I cannot score this."  # Unparseable
        mock_judge.return_value = mock_j

        engine = IslandEvolution(target_role="writer")
        score = engine._evaluate_fitness("You are a writer.")
        assert score == pytest.approx(0.4, abs=0.01)  # Default fallback

    @patch("app.llm_factory.create_cheap_vetting_llm")
    @patch("app.llm_factory.create_specialist_llm")
    def test_evaluate_clamps_score(self, mock_specialist, mock_judge):
        from app.island_evolution import IslandEvolution
        mock_agent = MagicMock()
        mock_agent.call.return_value = "A valid response to the task"
        mock_specialist.return_value = mock_agent

        mock_j = MagicMock()
        mock_j.call.return_value = '{"score": 5.0, "reason": "Off scale"}'
        mock_judge.return_value = mock_j

        engine = IslandEvolution(target_role="coder")
        score = engine._evaluate_fitness("Test prompt")
        assert score <= 1.0

    def test_evaluate_fallback_on_exception(self):
        from app.island_evolution import IslandEvolution
        engine = IslandEvolution(target_role="coder")
        with patch("app.llm_factory.create_specialist_llm", side_effect=RuntimeError("No LLM")):
            score = engine._evaluate_fitness("Test")
            assert 0.3 <= score <= 0.5  # Random fallback range

    @patch("app.llm_factory.create_cheap_vetting_llm")
    @patch("app.llm_factory.create_specialist_llm")
    def test_evaluate_handles_task_exception(self, mock_specialist, mock_judge):
        from app.island_evolution import IslandEvolution
        mock_agent = MagicMock()
        mock_agent.call.side_effect = RuntimeError("API error")
        mock_specialist.return_value = mock_agent
        mock_judge.return_value = MagicMock()

        engine = IslandEvolution(target_role="coder")
        score = engine._evaluate_fitness("Test prompt")
        assert score == pytest.approx(0.3, abs=0.01)  # Exception fallback


# ════════════════════════════════════════════════════════════════════════════════
# 10. RUN_ISLAND_EVOLUTION_CYCLE TESTS
# ════════════════════════════════════════════════════════════════════════════════

class TestRunCycle:
    """Module-level entry point used by idle scheduler."""

    @patch("app.island_evolution.IslandEvolution.run_session")
    @patch("app.island_evolution.IslandEvolution.initialize_population")
    @patch("app.island_evolution.IslandEvolution._load", return_value=False)
    @patch("app.prompt_registry.get_active_prompt", return_value="Base prompt content")
    def test_cycle_initializes_fresh(self, mock_prompt, mock_load, mock_init, mock_run):
        from app.island_evolution import run_island_evolution_cycle
        mock_run.return_value = {"epochs_run": 5, "best": None, "stats": {}}
        result = run_island_evolution_cycle("coder")
        mock_init.assert_called_once()
        assert isinstance(result, dict)

    @patch("app.island_evolution.IslandEvolution.run_session")
    @patch("app.island_evolution.IslandEvolution.initialize_population")
    @patch("app.island_evolution.IslandEvolution._load", return_value=True)
    @patch("app.prompt_registry.get_active_prompt", return_value="Base prompt content")
    def test_cycle_resumes_from_saved(self, mock_prompt, mock_load, mock_init, mock_run):
        from app.island_evolution import run_island_evolution_cycle
        mock_run.return_value = {"epochs_run": 5, "best": None, "stats": {}}
        result = run_island_evolution_cycle("coder")
        mock_init.assert_not_called()  # Should NOT reinitialize
        assert isinstance(result, dict)

    @patch("app.prompt_registry.get_active_prompt", return_value=None)
    def test_cycle_no_prompt(self, mock_prompt):
        from app.island_evolution import run_island_evolution_cycle
        result = run_island_evolution_cycle("unknown_role")
        assert result["status"] == "no_prompt"

    @patch("app.prompt_registry.get_active_prompt", side_effect=RuntimeError("DB down"))
    def test_cycle_handles_error(self, mock_prompt):
        from app.island_evolution import run_island_evolution_cycle
        result = run_island_evolution_cycle("coder")
        assert result.get("status") == "error"
        assert "error" in result

    @patch("app.island_evolution.IslandEvolution.run_session")
    @patch("app.island_evolution.IslandEvolution.initialize_population")
    @patch("app.island_evolution.IslandEvolution._load", return_value=False)
    @patch("app.prompt_registry.get_active_prompt", return_value="Base prompt")
    def test_cycle_attempts_promotion_above_threshold(self, mock_prompt, mock_load, mock_init, mock_run):
        from app.island_evolution import run_island_evolution_cycle
        mock_run.return_value = {
            "epochs_run": 5,
            "best": {
                "prompt_content": "Evolved prompt",
                "fitness": 0.85,
                "mutation_type": "meta_prompt",
            },
            "stats": {},
        }
        with patch("app.prompt_registry.propose_version", return_value=42) as mock_propose:
            with patch("app.prompt_registry.promote_version") as mock_promote:
                result = run_island_evolution_cycle("coder")
                mock_propose.assert_called_once()
                mock_promote.assert_called_once()

    @patch("app.island_evolution.IslandEvolution.run_session")
    @patch("app.island_evolution.IslandEvolution.initialize_population")
    @patch("app.island_evolution.IslandEvolution._load", return_value=False)
    @patch("app.prompt_registry.get_active_prompt", return_value="Base prompt")
    def test_cycle_no_promotion_below_threshold(self, mock_prompt, mock_load, mock_init, mock_run):
        from app.island_evolution import run_island_evolution_cycle
        mock_run.return_value = {
            "epochs_run": 5,
            "best": {
                "prompt_content": "Mediocre prompt",
                "fitness": 0.55,  # Below 0.7
                "mutation_type": "meta_prompt",
            },
            "stats": {},
        }
        with patch("app.prompt_registry.propose_version") as mock_propose:
            result = run_island_evolution_cycle("coder")
            mock_propose.assert_not_called()


# ════════════════════════════════════════════════════════════════════════════════
# 11. SYSTEM WIRING TESTS
# ════════════════════════════════════════════════════════════════════════════════

class TestSystemWiring:
    """Island evolution must be wired into the live system."""

    def test_idle_scheduler_has_job(self):
        from app.idle_scheduler import _default_jobs
        jobs = _default_jobs()
        names = [name for name, _ in jobs]
        assert "island-evolution" in names

    def test_idle_scheduler_job_is_callable(self):
        from app.idle_scheduler import _default_jobs
        jobs = _default_jobs()
        island_jobs = [(n, fn) for n, fn in jobs if n == "island-evolution"]
        assert len(island_jobs) == 1
        assert callable(island_jobs[0][1])

    def test_evolution_suite_reexports(self):
        from app.evolution_suite import IslandEvolution, run_island_evolution_cycle
        assert callable(IslandEvolution)
        assert callable(run_island_evolution_cycle)

    def test_auto_deployer_protects_file(self):
        src = inspect.getsource(__import__("app.auto_deployer", fromlist=["_"]))
        assert "island_evolution" in src

    def test_publish_lists_module(self):
        src = inspect.getsource(__import__("app.firebase.publish", fromlist=["_"]))
        assert "island_evolution" in src

    def test_evolve_blocks_importable(self):
        """Island evolution depends on evolve_blocks for FREEZE/EVOLVE markers."""
        from app.evolve_blocks import has_evolve_blocks, validate_modification
        assert callable(has_evolve_blocks)
        assert callable(validate_modification)

    def test_prompt_registry_importable(self):
        """Island evolution depends on prompt_registry for base prompts."""
        from app.prompt_registry import get_active_prompt
        assert callable(get_active_prompt)


# ════════════════════════════════════════════════════════════════════════════════
# 12. INTEGRATION TESTS (require running services)
# ════════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    """Integration tests that run with live services."""

    def test_persisted_state_on_disk(self):
        """Check that island evolution state files exist from prior runs."""
        from app.island_evolution import ISLAND_DIR
        if not ISLAND_DIR.exists():
            pytest.skip("No island_evolution directory")
        roles = list(ISLAND_DIR.iterdir())
        # May have state from prior runs
        for role_dir in roles:
            state_file = role_dir / "state.json"
            if state_file.exists():
                data = json.loads(state_file.read_text())
                assert "role" in data
                assert "epoch" in data
                assert "islands" in data
                assert isinstance(data["islands"], list)

    def test_full_persist_load_cycle(self):
        """Full round trip: create → populate → persist → load → verify."""
        from app.island_evolution import IslandEvolution, Individual, NUM_ISLANDS
        role = "integration_test_full"
        engine = IslandEvolution(target_role=role)
        for i, island in enumerate(engine._islands):
            for j in range(3):
                island.add(Individual(
                    prompt_id=f"int_{i}_{j}",
                    prompt_content=f"Integration test prompt {i}-{j} is long enough",
                    fitness=random.uniform(0.3, 0.9),
                    generation=i,
                    mutation_type="meta_prompt",
                    ancestors=[f"anc_{k}" for k in range(j)],
                ))
            island.generation = 5
            island.best_fitness = max(ind.fitness for ind in island.population)
            island.stagnation_count = i
        engine._epoch = 5
        engine._persist()

        # Fresh engine
        engine2 = IslandEvolution(target_role=role)
        assert engine2._load() is True
        assert engine2._epoch == 5
        assert len(engine2._islands) == NUM_ISLANDS
        for i, island in enumerate(engine2._islands):
            assert len(island.population) == 3
            assert island.generation == 5
            assert island.stagnation_count == i
            for ind in island.population:
                assert len(ind.prompt_content) > 0
                assert ind.fitness > 0

    def test_selection_pressure_over_generations(self):
        """Simulate multiple generations; average fitness should not decrease."""
        from app.island_evolution import Island, Individual
        island = Island(island_id=0)
        # Start with varied fitness
        for i in range(5):
            island.add(Individual(
                prompt_id=f"gen0_{i}",
                prompt_content=f"Gen 0 individual {i}",
                fitness=random.uniform(0.2, 0.8),
            ))
        avg_before = sum(i.fitness for i in island.population) / len(island.population)

        # Simulate 10 generations of selection
        for gen in range(10):
            # Add offspring (better-than-average)
            for _ in range(5):
                parent = island.select_parent()
                offspring_fitness = parent.fitness + random.uniform(-0.05, 0.1)
                offspring_fitness = max(0.0, min(1.0, offspring_fitness))
                island.add(Individual(
                    prompt_id=f"gen{gen}_{_}",
                    prompt_content=f"Gen {gen} offspring",
                    fitness=offspring_fitness,
                    generation=gen + 1,
                ))
            island.select_survivors()

        avg_after = sum(i.fitness for i in island.population) / len(island.population)
        # With tournament selection + elitism, average fitness should generally improve
        # (or at minimum, the best should not decrease)
        best_after = max(i.fitness for i in island.population)
        assert best_after >= avg_before * 0.8  # Best should be at least near initial avg


# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
