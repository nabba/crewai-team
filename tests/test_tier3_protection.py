"""
Tests for Tier-3 protection coverage.

These tests enforce the CLAUDE.md invariant: "Evaluation functions and
safety constraints live at INFRASTRUCTURE level — must NEVER be in
agent-modifiable code paths."

If a consciousness evaluator, belief store, or homeostatic config is
added to the codebase, it should also be added to TIER3_FILES. The
tests here fail loudly when that invariant is violated.
"""

from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _repo_root_str() -> str:
    """Return the repo root the way safety_guardian expects as app_root."""
    # In production the app lives at /app; here we pass the real repo root
    # so the test works on the host and in the container.
    return str(REPO_ROOT)


class TestTier3Coverage:
    def test_original_infrastructure_still_listed(self):
        """Regression guard: the original Tier-3 list must not shrink."""
        from app.safety_guardian import TIER3_FILES
        for path in [
            "app/eval_sandbox.py",
            "app/safety_guardian.py",
            "app/feedback_pipeline.py",
            "app/security.py",
            "app/sanitize.py",
            "app/vetting.py",
            "app/version_manifest.py",
            "app/sandbox_runner.py",
            "app/health_monitor.py",
            "app/healing/health_remediator.py",
            "app/reference_tasks.py",
        ]:
            assert path in TIER3_FILES, f"regressed: {path} removed from Tier 3"

    def test_consciousness_evaluators_protected(self):
        """New Tier-3 coverage: consciousness evaluators + homeostatic config."""
        from app.safety_guardian import TIER3_FILES
        required = [
            "app/self_awareness/consciousness_probe.py",
            "app/self_awareness/behavioral_assessment.py",
            "app/self_awareness/certainty_vector.py",
            "app/self_awareness/somatic_marker.py",
            "app/self_awareness/meta_cognitive.py",
            "app/self_awareness/sentience_config.py",
            "app/consciousness/adversarial_probes.py",
            "app/consciousness/belief_store.py",
            "app/consciousness/metacognitive_monitor.py",
            "app/consciousness/config.py",
            "app/consciousness/workspace_buffer.py",
            "app/consciousness/attention_schema.py",
            "app/consciousness/global_broadcast.py",
            "app/consciousness/prediction_hierarchy.py",
            "app/consciousness/predictive_layer.py",
        ]
        missing = [p for p in required if p not in TIER3_FILES]
        assert not missing, f"Consciousness evaluators NOT protected: {missing}"

    def test_governance_protected(self):
        """governance.py enforces promotion gates — it must be immutable."""
        from app.safety_guardian import TIER3_FILES
        assert "app/governance.py" in TIER3_FILES

    def test_subia_infrastructure_protected(self):
        """SubIA config + kernel are infrastructure per PROGRAM.md Phase 3."""
        from app.safety_guardian import TIER3_FILES
        assert "app/subia/config.py" in TIER3_FILES
        assert "app/subia/kernel.py" in TIER3_FILES

    def test_phase2_pp1_closure_protected(self):
        """Phase 2 PP-1 half-circuit closure: surprise_routing must be
        Tier-3 because it is the only path by which computed prediction
        errors actually reach the scene. If it were agent-modifiable,
        the Self-Improver could silently re-open the half-circuit.
        """
        from app.safety_guardian import TIER3_FILES
        assert "app/subia/prediction/surprise_routing.py" in TIER3_FILES

    def test_phase2_hot3_closure_protected(self):
        """Phase 2 HOT-3 half-circuit closure: dispatch_gate must be
        Tier-3. It is the policy module that decides whether consulted
        beliefs ALLOW, ESCALATE, or BLOCK crew dispatch. Agent-modifiable
        gating policy would let the Self-Improver weaken its own
        refusal criteria.
        """
        from app.safety_guardian import TIER3_FILES
        assert "app/subia/belief/dispatch_gate.py" in TIER3_FILES

    def test_phase2_certainty_hedging_protected(self):
        """Phase 2 certainty → response-hedging closure: the
        post-processor that applies [Inferred]/[Uncertain] tags must be
        Tier-3. An agent-modifiable hedger could be rewritten to skip
        hedging, which would silently re-open the half-circuit.
        """
        from app.safety_guardian import TIER3_FILES
        assert "app/subia/belief/response_hedging.py" in TIER3_FILES

    def test_phase2_ast1_intervention_guard_protected(self):
        """Phase 2 AST-1 DGM runtime verifier: the module that audits
        every intervention for DGM compliance must be Tier-3. If
        agent-modifiable, the Self-Improver could neutralize the
        verifier to let violating interventions pass silently.
        """
        from app.safety_guardian import TIER3_FILES
        assert "app/subia/scene/intervention_guard.py" in TIER3_FILES

    def test_phase2_injection_harness_protected(self):
        """Phase 2 prediction-hierarchy injection A/B harness: the
        measurement module that certifies injection actually shifts
        output must be Tier-3. If agent-modifiable, the Self-Improver
        could patch the threshold to always PASS and hide a dead
        injection path.
        """
        from app.safety_guardian import TIER3_FILES
        assert "app/subia/prediction/injection_harness.py" in TIER3_FILES

    def test_phase3_integrity_module_protected(self):
        """Phase 3 integrity verifier must itself be Tier-3.
        If the verifier were agent-modifiable, it could be rewritten
        to always return ok=True, defeating the entire integrity
        mechanism.
        """
        from app.safety_guardian import TIER3_FILES
        assert "app/subia/integrity.py" in TIER3_FILES

    def test_phase4_cil_loop_and_hooks_protected(self):
        """Phase 4 CIL sequencer + lifecycle hook integration must be
        Tier-3. The loop orchestrates every consciousness signal that
        gates behavior; an agent-modifiable loop could silently skip
        a gate or inject its own injection context.
        """
        from app.safety_guardian import TIER3_FILES
        assert "app/subia/loop.py" in TIER3_FILES
        assert "app/subia/hooks.py" in TIER3_FILES

    def test_phase4_deferred_safety_protected(self):
        """SubIA Part I §0.4 invariants #2 (setpoint immutability) and
        #3 (audit immutability) must be Tier-3. These are the single
        write paths for their respective invariants; if agent-
        modifiable, an agent could widen the allow-list (invariant #2)
        or add a delete function (invariant #3).
        """
        from app.safety_guardian import TIER3_FILES
        assert "app/subia/safety/setpoint_guard.py" in TIER3_FILES
        assert "app/subia/safety/narrative_audit.py" in TIER3_FILES

    def test_phase4_finish_modules_protected(self):
        """Phase 4 finish: persistence, prediction cache, LLM predict,
        homeostasis engine, and live_integration must all be Tier-3.
        An agent-modifiable persistence layer could silently drop
        kernel fields; an agent-modifiable LLM predict could always
        return high-confidence fallback; an agent-modifiable
        live_integration could skip registration without logging.
        """
        from app.safety_guardian import TIER3_FILES
        for path in (
            "app/subia/persistence.py",
            "app/subia/prediction/cache.py",
            "app/subia/prediction/llm_predict.py",
            "app/subia/homeostasis/engine.py",
            "app/subia/live_integration.py",
        ):
            assert path in TIER3_FILES, f"not protected: {path}"

    def test_phase5_scene_upgrades_protected(self):
        """Phase 5 scene upgrades: tier builder, commitment-orphan
        protection, strategic scan, compact context all Tier-3. An
        agent-modifiable tier builder could silently downgrade an
        orphaned commitment; an agent-modifiable scan could hide
        items it doesn't want surfaced.
        """
        from app.safety_guardian import TIER3_FILES
        for path in (
            "app/subia/scene/tiers.py",
            "app/subia/scene/strategic_scan.py",
            "app/subia/scene/compact_context.py",
        ):
            assert path in TIER3_FILES, f"not protected: {path}"

    def test_phase6_prediction_refinements_protected(self):
        """Phase 6 prediction refinements: accuracy_tracker and cascade
        policy must be Tier-3. An agent-modifiable accuracy_tracker
        could silently report false high accuracy and prevent cascade
        escalation on genuinely failing domains; an agent-modifiable
        cascade policy could be rewritten to never escalate.
        """
        from app.safety_guardian import TIER3_FILES
        for path in (
            "app/subia/prediction/accuracy_tracker.py",
            "app/subia/prediction/cascade.py",
        ):
            assert path in TIER3_FILES, f"not protected: {path}"

    def test_phase7_dual_tier_memory_protected(self):
        """Phase 7 dual-tier memory: consolidator, dual_tier access,
        spontaneous surfacing, and retrospective promotion must all
        be Tier-3. An agent-modifiable consolidator could silently
        drop significant records; an agent-modifiable dual-tier
        access could route `recall()` to the full tier (leaking
        noise into the conscious channel); an agent-modifiable
        retrospective could promote junk to curated or hide genuinely
        significant records.
        """
        from app.safety_guardian import TIER3_FILES
        for path in (
            "app/subia/memory/consolidator.py",
            "app/subia/memory/dual_tier.py",
            "app/subia/memory/spontaneous.py",
            "app/subia/memory/retrospective.py",
        ):
            assert path in TIER3_FILES, f"not protected: {path}"

    def test_phase8_social_and_strange_loop_protected(self):
        """Phase 8 social + strange loop: ToM manager, salience boost,
        consciousness-state page, drift detection must all be Tier-3.
        An agent-modifiable ToM manager could fabricate positive trust
        on its own; agent-modifiable drift detection could suppress
        capability-claim mismatches; agent-modifiable strange-loop
        generator could write a self-flattering page.
        """
        from app.safety_guardian import TIER3_FILES
        for path in (
            "app/subia/social/model.py",
            "app/subia/social/salience_boost.py",
            "app/subia/wiki_surface/consciousness_state.py",
            "app/subia/wiki_surface/drift_detection.py",
        ):
            assert path in TIER3_FILES, f"not protected: {path}"

    def test_phase9_scorecard_protected(self):
        """Phase 9 scorecard evaluators must all be Tier-3.

        An agent-modifiable scorecard is an evaluation function in
        the CLAUDE.md sense. If the Self-Improver could rewrite
        butlin.eval_pp1() to always return STRONG, the scorecard's
        entire integrity guarantee collapses.
        """
        from app.safety_guardian import TIER3_FILES
        for path in (
            "app/subia/probes/indicator_result.py",
            "app/subia/probes/butlin.py",
            "app/subia/probes/rsm.py",
            "app/subia/probes/sk.py",
            "app/subia/probes/scorecard.py",
        ):
            assert path in TIER3_FILES, f"not protected: {path}"

    def test_phase10_connections_protected(self):
        """Phase 10 inter-system bridges must all be Tier-3.

        Agent-modifiable PDS bridge could widen per-loop/per-week
        caps; agent-modifiable phronesis bridge could zero out
        normative penalties; agent-modifiable DGM felt-constraint
        could suppress integrity-drift signals. Each bridge is a
        single write path to a felt-safety variable and must be
        infrastructure-level.
        """
        from app.safety_guardian import TIER3_FILES
        for path in (
            "app/subia/connections/pds_bridge.py",
            "app/subia/connections/phronesis_bridge.py",
            "app/subia/connections/training_signal.py",
            "app/subia/connections/firecrawl_predictor.py",
            "app/subia/connections/dgm_felt_constraint.py",
            "app/subia/connections/service_health.py",
        ):
            assert path in TIER3_FILES, f"not protected: {path}"

    def test_phase1_migrations_protected(self):
        """Migrated modules (Phase 1) are protected at the NEW canonical path.
        Old shim paths remain in TIER3_FILES to protect the redirection.
        """
        from app.safety_guardian import TIER3_FILES
        phase1_pairs = [
            ("app/consciousness/workspace_buffer.py",      "app/subia/scene/buffer.py"),
            ("app/consciousness/attention_schema.py",      "app/subia/scene/attention_schema.py"),
            ("app/consciousness/global_broadcast.py",      "app/subia/scene/broadcast.py"),
            ("app/consciousness/meta_workspace.py",        "app/subia/scene/meta_workspace.py"),
            ("app/consciousness/personality_workspace.py", "app/subia/scene/personality_workspace.py"),
            ("app/consciousness/belief_store.py",          "app/subia/belief/store.py"),
            ("app/consciousness/metacognitive_monitor.py", "app/subia/belief/metacognition.py"),
            ("app/consciousness/prediction_hierarchy.py",  "app/subia/prediction/hierarchy.py"),
            ("app/consciousness/predictive_layer.py",      "app/subia/prediction/layer.py"),
            ("app/consciousness/adversarial_probes.py",    "app/subia/probes/adversarial.py"),
            # self_awareness batch
            ("app/self_awareness/self_model.py",            "app/subia/self/model.py"),
            ("app/self_awareness/hyper_model.py",           "app/subia/self/hyper_model.py"),
            ("app/self_awareness/temporal_identity.py",     "app/subia/self/temporal_identity.py"),
            ("app/self_awareness/agent_state.py",           "app/subia/self/agent_state.py"),
            ("app/self_awareness/loop_closure.py",          "app/subia/self/loop_closure.py"),
            ("app/self_awareness/homeostasis.py",           "app/subia/homeostasis/state.py"),
            ("app/self_awareness/somatic_marker.py",        "app/subia/homeostasis/somatic_marker.py"),
            ("app/self_awareness/somatic_bias.py",          "app/subia/homeostasis/somatic_bias.py"),
            ("app/self_awareness/certainty_vector.py",      "app/subia/belief/certainty.py"),
            ("app/self_awareness/consciousness_probe.py",   "app/subia/probes/consciousness_probe.py"),
            ("app/self_awareness/behavioral_assessment.py", "app/subia/probes/behavioral_assessment.py"),
            # batch 4 (triage-pass migrations)
            ("app/self_awareness/cogito.py",                  "app/subia/belief/cogito.py"),
            ("app/self_awareness/dual_channel.py",            "app/subia/belief/dual_channel.py"),
            ("app/self_awareness/global_workspace.py",        "app/subia/scene/global_workspace.py"),
            ("app/self_awareness/grounding.py",               "app/subia/self/grounding.py"),
            ("app/self_awareness/inferential_competition.py", "app/subia/prediction/inferential_competition.py"),
            ("app/self_awareness/internal_state.py",          "app/subia/belief/internal_state.py"),
            ("app/self_awareness/meta_cognitive.py",          "app/subia/belief/meta_cognitive_layer.py"),
            ("app/self_awareness/precision_weighting.py",     "app/subia/prediction/precision_weighting.py"),
            ("app/self_awareness/query_router.py",            "app/subia/self/query_router.py"),
            ("app/self_awareness/reality_model.py",           "app/subia/prediction/reality_model.py"),
            ("app/self_awareness/sentience_config.py",        "app/subia/sentience_config.py"),
            ("app/self_awareness/state_logger.py",            "app/subia/belief/state_logger.py"),
            ("app/self_awareness/world_model.py",             "app/subia/belief/world_model.py"),
        ]
        for old, new in phase1_pairs:
            assert old in TIER3_FILES, f"shim not protected: {old}"
            assert new in TIER3_FILES, f"target not protected: {new}"

    def test_all_listed_files_exist_on_disk(self):
        """Every declared Tier-3 file must actually exist. Otherwise the
        checksum machinery silently tracks a non-existent path.
        """
        from app.safety_guardian import TIER3_FILES
        missing = []
        for path in TIER3_FILES:
            if not (REPO_ROOT / path).exists():
                missing.append(path)
        assert not missing, f"Declared but missing: {missing}"


class TestTier3Status:
    def test_status_reports_all_present(self):
        from app.safety_guardian import tier3_status, TIER3_FILES
        status = tier3_status(_repo_root_str())
        assert status["total"] == len(TIER3_FILES)
        assert status["missing"] == []
        assert len(status["present"]) == len(TIER3_FILES)

    def test_status_checksums_are_sha256_hex(self):
        from app.safety_guardian import tier3_status
        status = tier3_status(_repo_root_str())
        for path, digest in status["checksums"].items():
            assert len(digest) == 64, f"{path}: not 64-char hex ({digest})"
            int(digest, 16)  # raises if not valid hex

    def test_status_detects_missing(self, tmp_path):
        """Pointing at an empty directory yields all missing, no checksums."""
        from app.safety_guardian import tier3_status, TIER3_FILES
        status = tier3_status(str(tmp_path))
        assert status["present"] == []
        assert set(status["missing"]) == set(TIER3_FILES)
        assert status["checksums"] == {}

    def test_checksum_changes_on_content_change(self, tmp_path):
        """Modifying a tracked file changes its SHA-256 digest."""
        from app.safety_guardian import tier3_status, TIER3_FILES
        # Build a fake app root with the first tier-3 file.
        target = TIER3_FILES[0]
        (tmp_path / target).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / target).write_text("content-A")

        status_a = tier3_status(str(tmp_path))
        digest_a = status_a["checksums"][target]

        (tmp_path / target).write_text("content-B")
        status_b = tier3_status(str(tmp_path))
        digest_b = status_b["checksums"][target]

        assert digest_a != digest_b, "SHA-256 failed to detect content change"
