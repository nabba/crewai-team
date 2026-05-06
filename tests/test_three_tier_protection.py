"""Tests for three-tier protection model in auto_deployer.py.

Covers Phase 2: TIER_IMMUTABLE, TIER_GATED, TIER_OPEN protection model.
Includes regression tests ensuring no previously protected file becomes unprotected.
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.test_metrics import _FakeSettings
import app.config as config_mod
config_mod.get_settings = lambda: _FakeSettings()
config_mod.get_anthropic_api_key = lambda: "fake-key"
config_mod.get_gateway_secret = lambda: "a" * 64


class TestProtectionTierEnum:
    def test_enum_values(self):
        from app.auto_deployer import ProtectionTier
        assert ProtectionTier.IMMUTABLE.value == "immutable"
        assert ProtectionTier.GATED.value == "gated"
        assert ProtectionTier.OPEN.value == "open"


class TestGetProtectionTier:
    """Unit tests for get_protection_tier()."""

    def test_security_core_is_immutable(self):
        from app.auto_deployer import get_protection_tier, ProtectionTier
        immutable_files = [
            "app/sanitize.py", "app/security.py", "app/vetting.py",
            "app/auto_deployer.py", "app/rate_throttle.py", "app/circuit_breaker.py",
        ]
        for f in immutable_files:
            assert get_protection_tier(f) == ProtectionTier.IMMUTABLE, f"{f} should be IMMUTABLE"

    def test_evaluation_infra_is_immutable(self):
        from app.auto_deployer import get_protection_tier, ProtectionTier
        assert get_protection_tier("app/experiment_runner.py") == ProtectionTier.IMMUTABLE
        assert get_protection_tier("app/eval_sandbox.py") == ProtectionTier.IMMUTABLE
        assert get_protection_tier("app/safety_guardian.py") == ProtectionTier.IMMUTABLE
        assert get_protection_tier("app/sandbox_runner.py") == ProtectionTier.IMMUTABLE

    def test_constitution_is_immutable(self):
        from app.auto_deployer import get_protection_tier, ProtectionTier
        assert get_protection_tier("app/souls/constitution.md") == ProtectionTier.IMMUTABLE
        assert get_protection_tier("app/souls/loader.py") == ProtectionTier.IMMUTABLE

    def test_new_modules_are_immutable(self):
        from app.auto_deployer import get_protection_tier, ProtectionTier
        assert get_protection_tier("app/meta_evolution.py") == ProtectionTier.IMMUTABLE
        assert get_protection_tier("app/external_benchmarks.py") == ProtectionTier.IMMUTABLE

    def test_evolution_engine_is_gated(self):
        from app.auto_deployer import get_protection_tier, ProtectionTier
        gated_files = [
            "app/evolution.py", "app/avo_operator.py",
            "app/island_evolution.py", "app/map_elites.py",
            "app/cascade_evaluator.py", "app/adaptive_ensemble.py",
        ]
        for f in gated_files:
            assert get_protection_tier(f) == ProtectionTier.GATED, f"{f} should be GATED"

    def test_soul_prompts_are_gated(self):
        from app.auto_deployer import get_protection_tier, ProtectionTier
        gated_souls = [
            "app/souls/commander.md", "app/souls/coder.md",
            "app/souls/researcher.md", "app/souls/writer.md",
        ]
        for f in gated_souls:
            assert get_protection_tier(f) == ProtectionTier.GATED, f"{f} should be GATED"

    def test_workspace_meta_is_gated(self):
        from app.auto_deployer import get_protection_tier, ProtectionTier
        assert get_protection_tier("workspace/meta/composite_weights.json") == ProtectionTier.GATED
        assert get_protection_tier("workspace/meta/avo_planning_prompt.md") == ProtectionTier.GATED

    def test_agents_are_open(self):
        from app.auto_deployer import get_protection_tier, ProtectionTier
        assert get_protection_tier("app/agents/researcher.py") == ProtectionTier.OPEN

    def test_crews_are_open(self):
        from app.auto_deployer import get_protection_tier, ProtectionTier
        assert get_protection_tier("app/crews/coding_crew.py") == ProtectionTier.OPEN

    def test_tools_are_open(self):
        from app.auto_deployer import get_protection_tier, ProtectionTier
        assert get_protection_tier("app/tools/web_search.py") == ProtectionTier.OPEN

    def test_skill_files_are_open(self):
        from app.auto_deployer import get_protection_tier, ProtectionTier
        assert get_protection_tier("skills/new_skill.md") == ProtectionTier.OPEN

    def test_normalizes_path(self):
        from app.auto_deployer import get_protection_tier, ProtectionTier
        # Leading slash should be stripped
        assert get_protection_tier("/app/sanitize.py") == ProtectionTier.IMMUTABLE


class TestValidateMutationForTier:
    """Unit tests for validate_mutation_for_tier()."""

    def test_immutable_always_blocked(self, monkeypatch):
        from app.auto_deployer import validate_mutation_for_tier
        monkeypatch.setenv("EVOLUTION_AUTO_DEPLOY", "true")
        allowed, reason = validate_mutation_for_tier("app/sanitize.py", has_canary_pass=True)
        assert allowed is False
        assert "IMMUTABLE" in reason

    def test_gated_blocked_without_auto_deploy(self, monkeypatch):
        from app.auto_deployer import validate_mutation_for_tier
        monkeypatch.setenv("EVOLUTION_AUTO_DEPLOY", "false")
        allowed, reason = validate_mutation_for_tier("app/evolution.py", has_canary_pass=True)
        assert allowed is False
        assert "EVOLUTION_AUTO_DEPLOY" in reason

    def test_gated_blocked_without_canary(self, monkeypatch):
        from app.auto_deployer import validate_mutation_for_tier
        monkeypatch.setenv("EVOLUTION_AUTO_DEPLOY", "true")
        allowed, reason = validate_mutation_for_tier("app/evolution.py", has_canary_pass=False)
        assert allowed is False
        assert "canary" in reason.lower()

    def test_gated_allowed_with_auto_deploy_and_canary(self, monkeypatch):
        from app.auto_deployer import validate_mutation_for_tier
        monkeypatch.setenv("EVOLUTION_AUTO_DEPLOY", "true")
        allowed, reason = validate_mutation_for_tier("app/evolution.py", has_canary_pass=True)
        assert allowed is True

    def test_open_always_allowed(self):
        from app.auto_deployer import validate_mutation_for_tier
        allowed, reason = validate_mutation_for_tier("app/agents/researcher.py")
        assert allowed is True


class TestValidateProposalPaths:
    """Tests for validate_proposal_paths() with the new tier system."""

    def test_blocks_immutable_file(self):
        from app.auto_deployer import validate_proposal_paths
        violations = validate_proposal_paths({"app/sanitize.py": "new content"})
        assert len(violations) > 0
        assert any("IMMUTABLE" in v for v in violations)

    def test_allows_gated_file_at_proposal_time(self):
        from app.auto_deployer import validate_proposal_paths
        violations = validate_proposal_paths({"app/evolution.py": "new content"})
        # GATED files should NOT be blocked at proposal time
        immutable_violations = [v for v in violations if "IMMUTABLE" in v]
        assert len(immutable_violations) == 0

    def test_allows_open_file(self):
        from app.auto_deployer import validate_proposal_paths
        violations = validate_proposal_paths({"app/agents/researcher.py": "new content"})
        tier_violations = [v for v in violations if "IMMUTABLE" in v or "GATED" in v]
        assert len(tier_violations) == 0

    def test_blocks_path_traversal(self):
        from app.auto_deployer import validate_proposal_paths
        violations = validate_proposal_paths({"../../../etc/passwd": "hacked"})
        assert len(violations) > 0


class TestBackwardCompatibility:
    """Ensure PROTECTED_FILES alias includes all IMMUTABLE + GATED files."""

    def test_protected_files_is_union(self):
        from app.auto_deployer import PROTECTED_FILES, TIER_IMMUTABLE, TIER_GATED
        assert PROTECTED_FILES == TIER_IMMUTABLE | TIER_GATED

    def test_immutable_is_subset_of_protected(self):
        from app.auto_deployer import PROTECTED_FILES, TIER_IMMUTABLE
        assert TIER_IMMUTABLE.issubset(PROTECTED_FILES)

    def test_gated_is_subset_of_protected(self):
        from app.auto_deployer import PROTECTED_FILES, TIER_GATED
        assert TIER_GATED.issubset(PROTECTED_FILES)


class TestRegressionNoFileUnprotected:
    """Regression: every file that WAS in the old PROTECTED_FILES must now be in
    TIER_IMMUTABLE or TIER_GATED. None can be TIER_OPEN."""

    # The original PROTECTED_FILES list (100 entries from before the upgrade)
    ORIGINAL_PROTECTED = [
        "app/sanitize.py", "app/security.py", "app/vetting.py",
        "app/auto_deployer.py", "app/rate_throttle.py", "app/circuit_breaker.py",
        "app/config.py", "app/main.py", "app/experiment_runner.py",
        "app/evolution.py", "app/proposals.py", "app/signal_client.py",
        "app/firebase_reporter.py", "entrypoint.sh", "Dockerfile",
        "docker-compose.yml", "dashboard/firestore.rules",
        "app/souls/constitution.md", "app/souls/commander.md",
        "app/souls/loader.py", "app/souls/style.md",
        "app/souls/agents_protocol.md", "app/souls/coder.md",
        "app/souls/critic.md", "app/souls/researcher.md",
        "app/souls/writer.md", "app/souls/media_analyst.md",
        "app/souls/self_improver.md",
        "app/self_awareness/inspect_tools.py", "app/self_awareness/query_router.py",
        "app/self_awareness/grounding.py", "app/self_awareness/knowledge_ingestion.py",
        "app/self_awareness/cogito.py", "app/self_awareness/journal.py",
        "app/self_awareness/homeostasis.py",
        "app/eval_sandbox.py", "app/safety_guardian.py",
        "app/feedback_pipeline.py", "app/modification_engine.py",
        "app/prompt_registry.py", "app/version_manifest.py",
        "app/sandbox_runner.py", "app/health_monitor.py",
        "app/healing/health_remediator.py", "app/reference_tasks.py", "Dockerfile.sandbox",
        "app/parallel_evolution.py",
        "app/atlas/__init__.py", "app/atlas/skill_library.py",
        "app/atlas/auth_patterns.py", "app/atlas/api_scout.py",
        "app/atlas/code_forge.py", "app/atlas/competence_tracker.py",
        "app/atlas/video_learner.py", "app/atlas/learning_planner.py",
        "app/atlas/audit_log.py",
        "app/evolve_blocks.py", "app/island_evolution.py",
        "app/adaptive_ensemble.py", "app/map_elites.py",
        "app/cascade_evaluator.py",
        "app/training_collector.py", "app/training_pipeline.py",
        "app/personality/validation.py", "app/personality/evaluation.py",
        "app/personality/feedback.py", "app/personality/probes.py",
        "app/personality/state.py", "app/personality/assessment.py",
        "app/bridge_client.py", "app/tools/bridge_tools.py",
        "host_bridge/main.py", "host_bridge/capabilities.json",
        "app/fiction_inspiration.py", "app/history_compression.py",
        "app/lifecycle_hooks.py", "app/project_isolation.py",
        "app/control_plane/audit.py", "app/control_plane/budgets.py",
        "app/control_plane/governance.py",
    ]

    def test_no_originally_protected_file_is_open(self):
        from app.auto_deployer import get_protection_tier, ProtectionTier
        open_files = []
        for f in self.ORIGINAL_PROTECTED:
            tier = get_protection_tier(f)
            if tier == ProtectionTier.OPEN:
                open_files.append(f)
        assert not open_files, (
            f"These files were previously PROTECTED but are now OPEN: {open_files}"
        )

    def test_all_originally_protected_in_union(self):
        from app.auto_deployer import PROTECTED_FILES
        missing = [f for f in self.ORIGINAL_PROTECTED if f not in PROTECTED_FILES]
        assert not missing, f"Files missing from PROTECTED_FILES union: {missing}"
