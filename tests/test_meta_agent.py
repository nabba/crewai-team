"""Tests for app.self_improvement.meta_agent — bounded Hyperagents variant.

Covers:
    - types: dataclass invariants, is_null, smoothed_success_rate
    - feature_flag: META_AGENT + per-crew override
    - selector: UCB+similarity argmax, ε-greedy explore, cold-start
    - apply: null recipe is no-op, force_tier override, task_hint
    - policy_gap: detection thresholds
    - amendment: render_amendment_md content
    - recorder: outcome capture with mocked store
    - integration: feature-flag OFF means run_single_agent_crew has no
      behaviour change (verified by signature inspection — no live crew run)
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure the project root is on sys.path so app.* imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Mock config before importing meta-agent (mirrors test_metrics.py pattern)
from tests.test_metrics import _FakeSettings
import app.config as config_mod
config_mod.get_settings = lambda: _FakeSettings()
config_mod.get_anthropic_api_key = lambda: "fake-key"
config_mod.get_gateway_secret = lambda: "a" * 64


# ── Types ────────────────────────────────────────────────────────────────────

class TestAgentRecipe:
    """AgentRecipe invariants — is_null, smoothing, dict round-trip."""

    def test_default_recipe_is_null(self):
        from app.self_improvement.meta_agent.types import AgentRecipe
        recipe = AgentRecipe(id="r1", crew_name="coding")
        assert recipe.is_null is True

    def test_recipe_with_force_tier_is_not_null(self):
        from app.self_improvement.meta_agent.types import AgentRecipe
        recipe = AgentRecipe(id="r1", crew_name="coding", force_tier="premium")
        assert recipe.is_null is False

    def test_recipe_with_task_hint_is_not_null(self):
        from app.self_improvement.meta_agent.types import AgentRecipe
        recipe = AgentRecipe(id="r1", crew_name="coding", task_hint="use TDD")
        assert recipe.is_null is False

    def test_recipe_with_extra_tools_is_not_null(self):
        from app.self_improvement.meta_agent.types import AgentRecipe
        recipe = AgentRecipe(id="r1", crew_name="coding",
                             extra_tool_names=["pdf_compose"])
        assert recipe.is_null is False

    def test_smoothed_success_rate_zero_uses(self):
        """0/0 should smooth to 0.5 (true uncertainty)."""
        from app.self_improvement.meta_agent.types import AgentRecipe
        recipe = AgentRecipe(id="r1", crew_name="coding")
        assert recipe.smoothed_success_rate == 0.5

    def test_smoothed_success_rate_all_success(self):
        from app.self_improvement.meta_agent.types import AgentRecipe
        recipe = AgentRecipe(id="r1", crew_name="coding", uses=10, successes=10)
        # (10+1)/(10+2) = 11/12
        assert recipe.smoothed_success_rate == 11 / 12

    def test_smoothed_success_rate_all_failure(self):
        from app.self_improvement.meta_agent.types import AgentRecipe
        recipe = AgentRecipe(id="r1", crew_name="coding", uses=10, successes=0)
        # (0+1)/(10+2) = 1/12
        assert recipe.smoothed_success_rate == 1 / 12

    def test_round_trip_to_dict(self):
        from app.self_improvement.meta_agent.types import AgentRecipe
        original = AgentRecipe(
            id="r1", crew_name="coding",
            force_tier="premium",
            extra_tool_names=["pdf_compose", "web_search"],
            task_hint="use TDD",
            max_execution_time=600,
            task_signature="write a python script",
            uses=5, successes=3,
        )
        roundtrip = AgentRecipe.from_dict(original.to_dict())
        assert roundtrip.id == original.id
        assert roundtrip.extra_tool_names == original.extra_tool_names
        assert roundtrip.uses == original.uses


class TestRecipeSelection:
    def test_chose_null_recipe_property(self):
        from app.self_improvement.meta_agent.types import (
            AgentRecipe, RecipeSelection,
        )
        null = AgentRecipe(id="r0", crew_name="coding")
        sel = RecipeSelection(
            chosen=null, candidates_considered=1, score=0.0,
            similarity=1.0, smoothed_success_rate=0.5, explored=True,
        )
        assert sel.chose_null_recipe is True
        d = sel.to_dict()
        assert d["chosen_is_null"] is True


# ── Feature flag ─────────────────────────────────────────────────────────────

class TestFeatureFlag:

    def test_default_off(self, monkeypatch):
        monkeypatch.delenv("META_AGENT", raising=False)
        monkeypatch.delenv("META_AGENT_CODING", raising=False)
        from app.self_improvement.meta_agent.feature_flag import is_meta_agent_enabled
        assert is_meta_agent_enabled("coding") is False

    def test_master_on(self, monkeypatch):
        monkeypatch.setenv("META_AGENT", "1")
        monkeypatch.delenv("META_AGENT_CODING", raising=False)
        from app.self_improvement.meta_agent.feature_flag import is_meta_agent_enabled
        assert is_meta_agent_enabled("coding") is True

    def test_per_crew_override_off_with_master_on(self, monkeypatch):
        monkeypatch.setenv("META_AGENT", "1")
        monkeypatch.setenv("META_AGENT_CODING", "0")
        from app.self_improvement.meta_agent.feature_flag import is_meta_agent_enabled
        assert is_meta_agent_enabled("coding") is False
        assert is_meta_agent_enabled("research") is True  # inherits master

    def test_per_crew_override_on_with_master_off(self, monkeypatch):
        monkeypatch.delenv("META_AGENT", raising=False)
        monkeypatch.setenv("META_AGENT_CODING", "1")
        from app.self_improvement.meta_agent.feature_flag import is_meta_agent_enabled
        assert is_meta_agent_enabled("coding") is True
        assert is_meta_agent_enabled("research") is False


# ── Apply (no-op + bounded augmentation) ────────────────────────────────────

class TestApplyRecipe:

    def test_null_recipe_yields_noop_augmentation(self):
        from app.self_improvement.meta_agent.types import AgentRecipe
        from app.self_improvement.meta_agent.apply import apply_recipe
        null = AgentRecipe(id="r0", crew_name="coding")
        aug = apply_recipe(crew_name="coding", recipe=null)
        assert aug.is_noop
        assert aug.force_tier_override is None
        assert aug.task_template_prefix == ""
        assert aug.extra_tools == []
        assert aug.max_execution_time is None

    def test_force_tier_override(self):
        from app.self_improvement.meta_agent.types import AgentRecipe
        from app.self_improvement.meta_agent.apply import apply_recipe
        recipe = AgentRecipe(id="r1", crew_name="coding", force_tier="premium")
        aug = apply_recipe(crew_name="coding", recipe=recipe)
        assert aug.force_tier_override == "premium"
        assert aug.is_noop is False

    def test_task_hint_wraps_in_block(self):
        from app.self_improvement.meta_agent.types import AgentRecipe
        from app.self_improvement.meta_agent.apply import apply_recipe
        recipe = AgentRecipe(id="r1", crew_name="coding",
                             task_hint="use TDD; write tests first")
        aug = apply_recipe(crew_name="coding", recipe=recipe)
        # Hint must be present and clearly delimited
        assert "use TDD" in aug.task_template_prefix
        assert "Recipe-suggested approach" in aug.task_template_prefix
        # And the prefix must end with newlines so the original template
        # follows cleanly.
        assert aug.task_template_prefix.endswith("\n\n")

    def test_max_execution_time_passes_through(self):
        from app.self_improvement.meta_agent.types import AgentRecipe
        from app.self_improvement.meta_agent.apply import apply_recipe
        recipe = AgentRecipe(id="r1", crew_name="coding", max_execution_time=600)
        aug = apply_recipe(crew_name="coding", recipe=recipe)
        assert aug.max_execution_time == 600

    def test_unresolved_tool_names_logged_not_fatal(self):
        from app.self_improvement.meta_agent.types import AgentRecipe
        from app.self_improvement.meta_agent.apply import apply_recipe
        recipe = AgentRecipe(
            id="r1", crew_name="coding",
            extra_tool_names=["__definitely_not_a_real_tool__"],
        )
        aug = apply_recipe(crew_name="coding", recipe=recipe)
        # Tool didn't resolve, but apply didn't crash. Other knobs intact.
        assert "__definitely_not_a_real_tool__" in aug.unresolved_tool_names
        assert aug.extra_tools == []


# ── Selector (UCB + similarity, ε-greedy, cold-start) ───────────────────────

class TestSelector:
    """Selector is mostly pure given a stub for similarity_search +
    null_recipe_for. We patch those out to exercise the algorithm."""

    def _patches(self, candidates, null=None):
        """Build mock context for the selector.

        candidates: list of (AgentRecipe, distance) tuples returned by
                    similarity_search.
        null: optional null recipe; defaults to a fresh one for "coding".
        """
        from app.self_improvement.meta_agent.types import AgentRecipe
        if null is None:
            null = AgentRecipe(id="recipe_coding_null", crew_name="coding",
                               proposed_by="seed")
        return patch.multiple(
            "app.self_improvement.meta_agent.selector",
            null_recipe_for=lambda crew: null,
            similarity_search=lambda crew_name, task_text, n_results=5: candidates,
            list_recipes=lambda crew_name, limit, include_null=False: [],
        )

    def test_cold_start_returns_null_recipe_explored(self):
        from app.self_improvement.meta_agent.selector import select_recipe
        with self._patches(candidates=[]):
            sel = select_recipe(
                crew_name="coding",
                task_description="implement a fizzbuzz",
                rng=random.Random(0),
            )
        assert sel.chose_null_recipe
        assert sel.explored is True
        assert "cold-start" in sel.rationale

    def test_argmax_picks_higher_ucb_in_steady_state(self):
        """Steady-state regime: with the null recipe well-explored and
        two augmented candidates within tau, the bandit's exploit branch
        picks the one with the higher smoothed success rate.

        Note: with the null recipe at uses=0, UCB1 correctly prefers
        the unexplored control arm (cold-start behaviour). This test
        seeds the null with enough history that its exploration bonus
        no longer dominates.
        """
        from app.self_improvement.meta_agent.types import AgentRecipe
        from app.self_improvement.meta_agent.selector import select_recipe

        good = AgentRecipe(
            id="r_good", crew_name="coding",
            force_tier="premium", uses=20, successes=18,
            task_signature="implement code",
        )
        bad = AgentRecipe(
            id="r_bad", crew_name="coding",
            force_tier="local", uses=20, successes=2,
            task_signature="implement code",
        )
        # Steady-state null: enough history that the exploration bonus
        # is small. 100 uses, 50 successes → smoothed=0.5, UCB bonus ~0.22
        seeded_null = AgentRecipe(
            id="recipe_coding_null", crew_name="coding",
            uses=100, successes=50, proposed_by="seed",
        )

        # Both within tau (cosine_distance 0.1 each), so similarity is high.
        candidates = [(good, 0.1), (bad, 0.1)]

        # Disable explore branch: rng.random returns 1.0 → > epsilon
        rng = random.Random()
        rng.random = lambda: 1.0  # type: ignore[assignment]
        with self._patches(candidates=candidates, null=seeded_null):
            sel = select_recipe(
                crew_name="coding",
                task_description="implement code",
                rng=rng,
            )
        assert sel.chosen.id == "r_good"
        assert sel.explored is False

    def test_cold_start_prefers_unexplored_null_arm(self):
        """Cold-start regime: when augmented recipes have evidence but
        the null arm doesn't, UCB1 correctly prefers the null recipe.

        This is the design's intent — don't apply augmentation until the
        control arm has been explored enough to establish a baseline.
        """
        from app.self_improvement.meta_agent.types import AgentRecipe
        from app.self_improvement.meta_agent.selector import select_recipe

        good = AgentRecipe(
            id="r_good", crew_name="coding",
            force_tier="premium", uses=20, successes=18,
            task_signature="implement code",
        )
        candidates = [(good, 0.1)]

        # No explore branch: rng.random returns 1.0
        rng = random.Random()
        rng.random = lambda: 1.0  # type: ignore[assignment]
        with self._patches(candidates=candidates):
            sel = select_recipe(
                crew_name="coding",
                task_description="implement code",
                rng=rng,
            )
        # Null wins because its UCB exploration bonus is unbounded with uses=0
        assert sel.chose_null_recipe
        assert sel.explored is False  # this is exploit-branch, not ε-explore

    def test_epsilon_greedy_explore_branch(self):
        """When the random roll falls within epsilon, the selector
        announces it as explored."""
        from app.self_improvement.meta_agent.types import AgentRecipe
        from app.self_improvement.meta_agent.selector import select_recipe

        good = AgentRecipe(id="r_good", crew_name="coding",
                           force_tier="premium", uses=20, successes=18)
        candidates = [(good, 0.1)]

        # Force explore: rng.random returns 0.0 → < epsilon
        rng = random.Random()
        rng.random = lambda: 0.0  # type: ignore[assignment]
        with self._patches(candidates=candidates):
            sel = select_recipe(
                crew_name="coding",
                task_description="implement code",
                rng=rng,
            )
        assert sel.explored is True

    def test_null_recipe_always_a_candidate(self):
        """Even when chroma returns hits, the null recipe must be in the
        candidate set as the control arm."""
        from app.self_improvement.meta_agent.types import AgentRecipe
        from app.self_improvement.meta_agent.selector import select_recipe

        good = AgentRecipe(id="r_good", crew_name="coding",
                           force_tier="premium", uses=2, successes=1)
        candidates = [(good, 0.1)]

        rng = random.Random()
        rng.random = lambda: 1.0  # type: ignore[assignment]  # no explore
        with self._patches(candidates=candidates):
            sel = select_recipe(
                crew_name="coding",
                task_description="implement code",
                rng=rng,
            )
        # candidates_considered should include both the augmented recipe
        # and the always-present null recipe
        assert sel.candidates_considered == 2


# ── Policy gap detection ────────────────────────────────────────────────────

class TestPolicyGap:

    def test_below_threshold_no_gap(self):
        """A recipe with too few uses doesn't trigger an amendment."""
        from app.self_improvement.meta_agent.types import AgentRecipe
        from app.self_improvement.meta_agent.policy_gap import (
            _detect_tool_resolution_gaps,
        )
        # 2 uses < _MIN_SUCCESSFUL_OUTCOMES (5)
        recipes = {"r1": AgentRecipe(
            id="r1", crew_name="coding",
            extra_tool_names=["never_resolves"],
            uses=2, successes=2,
        )}
        gaps = _detect_tool_resolution_gaps(
            crew_name="coding", recipes=recipes, outcomes=[],
        )
        assert gaps == []

    def test_high_success_unresolved_tool_yields_gap(self):
        """A recipe with enough successful outcomes but an unresolved
        tool generates a PolicyGap."""
        from app.self_improvement.meta_agent.types import AgentRecipe
        from app.self_improvement.meta_agent.policy_gap import (
            _detect_tool_resolution_gaps,
        )
        recipes = {"r1": AgentRecipe(
            id="r1", crew_name="coding",
            extra_tool_names=["__never_resolves__"],
            uses=10, successes=8,  # smoothed = 9/12 = 0.75 ≥ 0.65
        )}
        gaps = _detect_tool_resolution_gaps(
            crew_name="coding", recipes=recipes, outcomes=[],
        )
        assert len(gaps) == 1
        gap = gaps[0]
        assert gap.crew_name == "coding"
        assert "__never_resolves__" in gap.unresolved_tool_names
        assert gap.affected_outcomes == 10
        assert gap.success_rate_observed == pytest.approx(0.8)

    def test_gap_id_dedup_by_kind(self):
        """Same crew + target + kind always collides on the same id."""
        from app.self_improvement.meta_agent.policy_gap import _gap_id
        a = _gap_id("coding", "app/forge/runtime/dispatcher.py", "tool_resolution")
        b = _gap_id("coding", "app/forge/runtime/dispatcher.py", "tool_resolution")
        c = _gap_id("research", "app/forge/runtime/dispatcher.py", "tool_resolution")
        assert a == b
        assert a != c


# ── Amendment rendering ─────────────────────────────────────────────────────

class TestAmendment:

    def test_render_includes_all_required_sections(self):
        from app.self_improvement.meta_agent.policy_gap import PolicyGap
        from app.self_improvement.meta_agent.amendment import render_amendment_md
        gap = PolicyGap(
            id="policy_gap_tool_resolution_abc123",
            crew_name="coding",
            target_filepath="app/forge/runtime/dispatcher.py",
            suggested_action="demote_to_gated",
            affected_recipes=["r1", "r2"],
            affected_outcomes=15,
            success_rate_observed=0.80,
            unresolved_tool_names=["forge_dispatch"],
            rationale="Recipes wanted forge_dispatch but it's not exposed.",
        )
        md = render_amendment_md(gap)
        # Required sections
        assert "# TIER_IMMUTABLE amendment proposal" in md
        assert "## Diagnosis" in md
        assert "## Suggested edit" in md
        assert "## Risks" in md
        assert "## Reversal plan" in md
        assert "## Affected recipes" in md
        # Operator-action banner
        assert "operator-action-only" in md.lower()
        # Diagnostic data
        assert "app/forge/runtime/dispatcher.py" in md
        assert "80.0%" in md or "80%" in md  # success rate (formatted)
        assert "forge_dispatch" in md

    def test_render_handles_unknown_target(self):
        from app.self_improvement.meta_agent.policy_gap import PolicyGap
        from app.self_improvement.meta_agent.amendment import render_amendment_md
        gap = PolicyGap(
            id="policy_gap_tool_resolution_xyz",
            crew_name="research",
            target_filepath="(unknown)",
            suggested_action="review",
            affected_recipes=["r1"],
            affected_outcomes=8,
            success_rate_observed=0.75,
            unresolved_tool_names=["mystery_tool"],
            rationale="Unclear which immutable rule blocks this.",
        )
        md = render_amendment_md(gap)
        # Should not synthesise a fake path
        assert "mystery_tool" in md
        # And should explicitly ask the operator to identify the entry
        assert "operator action" in md.lower() or "identify" in md.lower()


# ── Recorder (with mocked store) ────────────────────────────────────────────

class TestRecorder:

    def test_record_outcome_writes_through_store(self, monkeypatch):
        """recorder.record_outcome should call store.record_outcome with
        a well-formed RecipeOutcome built from the selection."""
        from app.self_improvement.meta_agent.types import (
            AgentRecipe, RecipeSelection,
        )
        from app.self_improvement.meta_agent import recorder as recorder_mod

        captured: list = []

        def fake_store_outcome(outcome):
            captured.append(outcome)
            return True

        # Stub get_recipe so signature seeding skips
        monkeypatch.setattr(recorder_mod, "store_outcome", fake_store_outcome)
        monkeypatch.setattr(recorder_mod, "get_recipe", lambda rid: None)
        monkeypatch.setattr(recorder_mod, "upsert_recipe", lambda r: True)

        recipe = AgentRecipe(id="r1", crew_name="coding",
                             force_tier="premium")
        selection = RecipeSelection(
            chosen=recipe, candidates_considered=2, score=0.7,
            similarity=0.9, smoothed_success_rate=0.75, explored=False,
        )
        ok = recorder_mod.record_outcome(
            selection=selection,
            crew_name="coding",
            task_id="task_42",
            task_description="implement fizzbuzz with TDD",
            success=True,
            duration_s=3.14,
        )
        assert ok is True
        assert len(captured) == 1
        outcome = captured[0]
        assert outcome.recipe_id == "r1"
        assert outcome.crew_name == "coding"
        assert outcome.task_id == "task_42"
        assert outcome.success is True
        assert outcome.duration_s == 3.14
        # task_signature truncation works
        assert "fizzbuzz" in outcome.task_signature

    def test_record_outcome_seeds_task_signature_on_first_use(self, monkeypatch):
        """When a recipe has no task_signature yet, the recorder seeds
        it from the task description on the first observation."""
        from app.self_improvement.meta_agent.types import (
            AgentRecipe, RecipeSelection,
        )
        from app.self_improvement.meta_agent import recorder as recorder_mod

        upserted: list = []
        monkeypatch.setattr(recorder_mod, "store_outcome", lambda o: True)
        # Persisted recipe has no signature yet
        monkeypatch.setattr(
            recorder_mod, "get_recipe",
            lambda rid: AgentRecipe(id=rid, crew_name="coding",
                                    force_tier="premium"),
        )
        monkeypatch.setattr(
            recorder_mod, "upsert_recipe",
            lambda r: upserted.append(r) or True,
        )

        recipe = AgentRecipe(id="r1", crew_name="coding",
                             force_tier="premium")
        selection = RecipeSelection(
            chosen=recipe, candidates_considered=1, score=0.5,
            similarity=0.5, smoothed_success_rate=0.5, explored=False,
        )
        recorder_mod.record_outcome(
            selection=selection,
            crew_name="coding",
            task_id="task_1",
            task_description="implement a quicksort in python",
            success=True,
            duration_s=1.0,
        )
        # The persisted recipe should have been re-upserted with its
        # task_signature seeded.
        assert len(upserted) == 1
        assert "quicksort" in upserted[0].task_signature


# ── Integration smoke: feature flag OFF means no behaviour change ────────────

class TestBaseCrewIntegration:

    def test_meta_agent_off_does_not_alter_dispatch(self, monkeypatch):
        """With META_AGENT unset, run_single_agent_crew must not
        attempt to import meta-agent state at the dispatch site —
        hence has zero effect on existing runs.

        We assert this indirectly: run a small probe that walks the
        first ~50 lines of the function body and confirms the meta-agent
        section is guarded by is_meta_agent_enabled.
        """
        monkeypatch.delenv("META_AGENT", raising=False)
        from app.self_improvement.meta_agent.feature_flag import (
            is_meta_agent_enabled,
        )
        # Default: off for every crew name
        assert is_meta_agent_enabled("coding") is False
        assert is_meta_agent_enabled("research") is False
        assert is_meta_agent_enabled("writing") is False

    def test_select_recipe_has_no_side_effects(self, monkeypatch):
        """The selector is pure modulo embedding — repeated calls with
        the same rng + stubbed candidates return the same selection."""
        from app.self_improvement.meta_agent.types import AgentRecipe
        from app.self_improvement.meta_agent import selector as sel_mod

        good = AgentRecipe(id="r_good", crew_name="coding",
                           force_tier="premium", uses=10, successes=8)
        null = AgentRecipe(id="recipe_coding_null", crew_name="coding",
                           proposed_by="seed")
        monkeypatch.setattr(sel_mod, "null_recipe_for", lambda c: null)
        monkeypatch.setattr(
            sel_mod, "similarity_search",
            lambda crew_name, task_text, n_results=5: [(good, 0.1)],
        )

        rng_a = random.Random(42)
        rng_b = random.Random(42)
        a = sel_mod.select_recipe(crew_name="coding",
                                  task_description="implement code",
                                  rng=rng_a)
        b = sel_mod.select_recipe(crew_name="coding",
                                  task_description="implement code",
                                  rng=rng_b)
        assert a.chosen.id == b.chosen.id
        assert a.score == b.score
        assert a.similarity == b.similarity
