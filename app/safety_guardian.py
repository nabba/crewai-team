"""
safety_guardian.py — Enforces immutable safety constraints on the self-improving loop.

IMMUTABLE — this module is part of the infrastructure layer (Tier 3).
It must NOT be modifiable by any agent or modification engine.

Responsibilities:
  1. Post-promotion monitoring: watch for negative feedback after prompt changes
  2. Auto-rollback: revert to previous version if quality degrades
  3. Drift detection: flag if modification velocity is anomalous
  4. Weekly digest: summary of all modifications sent to owner via Signal
  5. Tier boundary enforcement: verify immutable files haven't been tampered with
"""

import hashlib
import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# ── IMMUTABLE: Checksums of critical infrastructure files ─────────────────
# Computed at build time.  If these change at runtime, the modification
# engine is paused and the owner is alerted.
# NOTE: Initially empty — populated on first startup.
_INTEGRITY_CHECKSUMS: dict[str, str] = {}
_INTEGRITY_FILE = Path("/app/workspace/.integrity_checksums.json")

# Files that must never change at runtime.
#
# Tier 3 is divided into two conceptual groups, hashed together:
#   (a) Original infrastructure — sandbox, security, feedback plumbing.
#   (b) Consciousness evaluators — modules that measure or produce the
#       signals the Self-Improver might be tempted to tune in its own
#       favor. Per CLAUDE.md: "Evaluation functions and safety
#       constraints live at INFRASTRUCTURE level — must NEVER be in
#       agent-modifiable code paths. The Self-Improver agent cannot
#       modify its own evaluation criteria."
#
# Extending (b) closes a latent violation of that invariant. See
# PROGRAM.md Phase 3 for context.
TIER3_FILES = [
    # (a) Original infrastructure
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
    "app/governance.py",
    # (b) Consciousness evaluators — protected against self-tuning
    "app/self_awareness/consciousness_probe.py",
    "app/self_awareness/behavioral_assessment.py",
    "app/self_awareness/certainty_vector.py",
    "app/self_awareness/somatic_marker.py",
    "app/self_awareness/somatic_bias.py",
    "app/self_awareness/meta_cognitive.py",
    "app/self_awareness/sentience_config.py",
    "app/self_awareness/self_model.py",
    "app/self_awareness/hyper_model.py",
    "app/self_awareness/temporal_identity.py",
    "app/self_awareness/agent_state.py",
    "app/self_awareness/loop_closure.py",
    "app/self_awareness/homeostasis.py",
    "app/consciousness/adversarial_probes.py",
    "app/consciousness/belief_store.py",
    "app/consciousness/metacognitive_monitor.py",
    "app/consciousness/config.py",
    "app/consciousness/workspace_buffer.py",
    "app/consciousness/attention_schema.py",
    "app/consciousness/global_broadcast.py",
    "app/consciousness/meta_workspace.py",
    "app/consciousness/personality_workspace.py",
    "app/consciousness/prediction_hierarchy.py",
    "app/consciousness/predictive_layer.py",
    # (c) SubIA infrastructure — consciousness-program config and kernel
    "app/subia/config.py",
    "app/subia/kernel.py",
    # (d) Migrated consciousness modules (new canonical locations).
    # Old paths above remain protected as shims. See PROGRAM.md Phase 1.
    "app/subia/scene/buffer.py",
    "app/subia/scene/attention_schema.py",
    "app/subia/scene/broadcast.py",
    "app/subia/scene/meta_workspace.py",
    "app/subia/scene/personality_workspace.py",
    "app/subia/belief/store.py",
    "app/subia/belief/metacognition.py",
    "app/subia/belief/certainty.py",
    "app/subia/prediction/hierarchy.py",
    "app/subia/prediction/layer.py",
    "app/subia/probes/adversarial.py",
    "app/subia/probes/consciousness_probe.py",
    "app/subia/probes/behavioral_assessment.py",
    "app/subia/homeostasis/state.py",
    "app/subia/README.md",
    "app/subia/homeostasis/somatic_marker.py",
    "app/subia/homeostasis/somatic_bias.py",
    "app/subia/self/model.py",
    "app/subia/self/hyper_model.py",
    "app/subia/self/temporal_identity.py",
    "app/subia/self/agent_state.py",
    "app/subia/self/loop_closure.py",
    # (e) Phase 1 batch 4: remaining self_awareness migration targets
    "app/subia/sentience_config.py",
    "app/subia/belief/cogito.py",
    "app/subia/belief/dual_channel.py",
    "app/subia/belief/internal_state.py",
    "app/subia/belief/meta_cognitive_layer.py",
    "app/subia/belief/state_logger.py",
    "app/subia/belief/world_model.py",
    "app/subia/scene/global_workspace.py",
    "app/subia/self/grounding.py",
    "app/subia/self/query_router.py",
    "app/subia/prediction/inferential_competition.py",
    "app/subia/prediction/precision_weighting.py",
    "app/subia/prediction/reality_model.py",
    "app/subia/prediction/surprise_routing.py",  # Phase 2 PP-1 closure
    "app/subia/belief/dispatch_gate.py",         # Phase 2 HOT-3 closure
    "app/subia/belief/response_hedging.py",      # Phase 2 certainty closure
    "app/subia/scene/intervention_guard.py",     # Phase 2 AST-1 DGM verifier
    "app/subia/prediction/injection_harness.py", # Phase 2 PH-injection A/B
    "app/subia/integrity.py",                    # Phase 3 integrity verifier
    "app/subia/loop.py",                         # Phase 4 CIL sequencer
    "app/subia/hooks.py",                        # Phase 4 lifecycle integration
    "app/subia/safety/setpoint_guard.py",        # Phase 4 DGM invariant #2
    "app/subia/safety/narrative_audit.py",       # Phase 4 DGM invariant #3
    "app/subia/persistence.py",                  # Phase 4 kernel serialization
    "app/subia/prediction/cache.py",             # Phase 4 prediction cache (Amendment B.4)
    "app/subia/prediction/llm_predict.py",       # Phase 4 live LLM predict_fn
    "app/subia/homeostasis/engine.py",           # Phase 4 homeostatic arithmetic
    "app/subia/live_integration.py",             # Phase 4 feature-flagged wire-in
    "app/subia/scene/tiers.py",                  # Phase 5 three-tier + orphan guard
    "app/subia/scene/strategic_scan.py",         # Phase 5 wide-view scan
    "app/subia/scene/compact_context.py",        # Phase 5 compact injection (B.5)
    "app/subia/prediction/accuracy_tracker.py",  # Phase 6 per-domain accuracy
    "app/subia/prediction/cascade.py",           # Phase 6 escalation policy
    "app/subia/memory/consolidator.py",          # Phase 7 dual-tier write
    "app/subia/memory/dual_tier.py",             # Phase 7 differentiated access
    "app/subia/memory/spontaneous.py",           # Phase 7 associative surfacing
    "app/subia/memory/retrospective.py",         # Phase 7 retrospective promotion
    "app/subia/social/model.py",                 # Phase 8 Theory-of-Mind
    "app/subia/social/salience_boost.py",        # Phase 8 social salience
    "app/subia/wiki_surface/consciousness_state.py",  # Phase 8 strange loop
    "app/subia/wiki_surface/drift_detection.py", # Phase 8 narrative drift
    "app/subia/probes/indicator_result.py",      # Phase 9 scorecard types
    "app/subia/probes/butlin.py",                # Phase 9 Butlin scorecard
    "app/subia/probes/rsm.py",                   # Phase 9 RSM signatures
    "app/subia/probes/sk.py",                    # Phase 9 SK evaluations
    "app/subia/probes/scorecard.py",             # Phase 9 aggregator
    "app/subia/connections/pds_bridge.py",       # Phase 10 SIA #1
    "app/subia/connections/phronesis_bridge.py", # Phase 10 SIA #2
    "app/subia/connections/training_signal.py",  # Phase 10 SIA #4
    "app/subia/connections/firecrawl_predictor.py",  # Phase 10 SIA #6
    "app/subia/connections/dgm_felt_constraint.py",  # Phase 10 SIA #7
    "app/subia/connections/service_health.py",   # Phase 10 circuit-breaker
    # ── Phase 12 — Six Proposals integration ─────────────────────────
    "app/subia/boundary/__init__.py",
    "app/subia/boundary/classifier.py",          # Phase 12 Proposal 5
    "app/subia/boundary/differential.py",        # Phase 12 Proposal 5
    "app/subia/wonder/__init__.py",
    "app/subia/wonder/detector.py",              # Phase 12 Proposal 4 (depth weights)
    "app/subia/wonder/register.py",              # Phase 12 Proposal 4 (closed-loop)
    "app/subia/values/__init__.py",
    "app/subia/values/resonance.py",             # Phase 12 Proposal 6 (keyword weights)
    "app/subia/values/perceptual_lens.py",       # Phase 12 Proposal 6 (Phronesis lenses)
    "app/subia/reverie/__init__.py",
    "app/subia/reverie/engine.py",               # Phase 12 Proposal 1
    "app/subia/understanding/__init__.py",
    "app/subia/understanding/pass_runner.py",    # Phase 12 Proposal 2
    "app/subia/shadow/__init__.py",
    "app/subia/shadow/biases.py",                # Phase 12 Proposal 3
    "app/subia/shadow/miner.py",                 # Phase 12 Proposal 3
    "app/subia/idle/__init__.py",
    "app/subia/idle/scheduler.py",               # Phase 12 idle dispatch
    "app/subia/phase12_hooks.py",                # Phase 12 hot-path hooks
    "app/subia/connections/six_proposals_bridges.py",  # Phase 12 inter-proposal bridges
    # ── Phase 13 — Technical Self-Awareness Layer (TSAL) ─────────────
    "app/subia/tsal/__init__.py",
    "app/subia/tsal/inspect_tools.py",       # Phase 13 canonical (consolidated from app/self_awareness/)
    "app/subia/tsal/probers.py",             # Phase 13 host + resource probing
    "app/subia/tsal/inspectors.py",          # Phase 13 code analyst + component discovery
    "app/subia/tsal/self_model.py",          # Phase 13 aggregate dataclass
    "app/subia/tsal/generators.py",          # Phase 13 wiki page generators
    "app/subia/tsal/operating_principles.py",# Phase 13 Tier-1 LLM inference
    "app/subia/tsal/evolution_feasibility.py",# Phase 13 Self-Improver gate
    "app/subia/tsal/refresh.py",             # Phase 13 idle scheduler registration
    "app/subia/connections/tsal_subia_bridge.py",  # Phase 13 TSAL → SubIA bridges
    # ── Phase 14 — Temporal Synchronization ──────────────────────────
    "app/subia/temporal/__init__.py",
    "app/subia/temporal/specious_present.py",  # Husserl/James felt-now
    "app/subia/temporal/momentum.py",          # rising/falling/stable trajectory
    "app/subia/temporal/circadian.py",         # processing-mode table (Tier-3)
    "app/subia/temporal/density.py",           # felt-time (subjective duration)
    "app/subia/temporal/binding.py",           # temporal_bind reducer
    "app/subia/temporal/rhythm_discovery.py",  # external rhythm mining
    "app/subia/temporal/context.py",           # TemporalContext aggregate
    "app/subia/temporal_hooks.py",             # CIL hot-path entry points
    "app/subia/connections/temporal_subia_bridge.py",  # 5 closed-loop bridges
    # ── Phase 15 — Factual Grounding & Correction Memory ─────────────
    "app/subia/grounding/__init__.py",
    "app/subia/grounding/claims.py",                # claim extractor (regex weights)
    "app/subia/grounding/source_registry.py",       # authoritative source map
    "app/subia/grounding/evidence.py",              # decision logic
    "app/subia/grounding/rewriter.py",              # response transformer
    "app/subia/grounding/correction.py",            # correction patterns + persist
    "app/subia/grounding/pipeline.py",              # public orchestrator
    "app/subia/grounding/belief_adapter.py",        # adapter interface
    "app/subia/connections/grounding_chat_bridge.py",  # feature-flagged hook
    # ── Phase 17 — Self-Introspection Routing ────────────────────────
    "app/subia/introspection/__init__.py",
    "app/subia/introspection/detector.py",       # keyword + scoring weights (Tier-3)
    "app/subia/introspection/context.py",        # gather defensive
    "app/subia/introspection/formatter.py",      # Phase-11 honest-language rules
    "app/subia/introspection/pipeline.py",       # orchestrator + feature flag
    "app/subia/connections/introspection_chat_bridge.py",  # main.py wire-in
    # ── Phase 18 — Self-Knowledge Routing (per-topic handlers) ───────
    "app/subia/introspection/topics/__init__.py",
    "app/subia/introspection/topics/beliefs.py",        # Cat A: belief store + sources
    "app/subia/introspection/topics/technical.py",      # Cat B: TSAL profiles
    "app/subia/introspection/topics/chronicle.py",      # Cat C: recent activity
    "app/subia/introspection/topics/scene.py",          # Cat D: focal/peripheral
    "app/subia/introspection/topics/wonder_shadow.py",  # Cat E: wonder + shadow
    "app/subia/introspection/topics/scorecard.py",      # Cat F: Butlin/RSM/SK + drift
    "app/subia/introspection/topics/predictions.py",    # Cat G: accuracy + history
    "app/subia/introspection/topics/social.py",         # Cat H: ToM
    # ── Affect Layer — welfare envelope + companion regulators ────────
    # The hard envelope (welfare.py HARD_ENVELOPE), reference panel
    # (data/reference_panel.json), schemas, hooks, calibration ratchet
    # state, and Phase-5 gate are infrastructure-level: not modifiable
    # by the Self-Improver, calibration cycle, or any agent. The
    # runtime assert_not_self_improver() guard in welfare.py only
    # fires when a caller invokes a setter — Tier-3 file-hash
    # protection is what stops a code-writing Self-Improver from
    # rewriting the constants directly.
    "app/affect/__init__.py",
    "app/affect/schemas.py",                         # AffectState, ViabilityFrame, WelfareBreach types
    "app/affect/welfare.py",                         # HARD_ENVELOPE + assert_not_self_improver
    "app/affect/viability.py",                       # 10-variable viability layer
    "app/affect/core.py",                            # V/A/C composition + attractor labeller
    "app/affect/reference_panel.py",                 # 20-scenario fixed compass
    "app/affect/data/reference_panel.json",          # the panel data itself
    "app/affect/calibration.py",                     # daily reflection cycle entry
    "app/affect/calibration_proposals.py",           # 6-guardrail flow + ratchet
    "app/affect/hooks.py",                           # POST_LLM_CALL@9 immutable + scheduled jobs
    "app/affect/api.py",                             # FastAPI router (auth + override-reset live here)
    "app/affect/runtime_state.py",                   # latency + autonomy counters
    "app/affect/kb_metadata.py",                     # episode-end KB tagging
    "app/affect/l9_snapshots.py",                    # daily rolled-up observability snapshot
    "app/affect/attachment.py",                      # OtherModel (mutual_regulation_weight ceilings)
    "app/affect/care_policies.py",                   # care budget enforcement
    "app/affect/ecological.py",                      # nested-scopes self-as-node
    "app/affect/phase5_gate.py",                     # consciousness-risk gate (observability)
    "app/affect/salience.py",                        # narrative-self salience detector (Loop 1)
    "app/affect/episodes.py",                        # narrative-self episode flush (Loop 2)
    "app/affect/narrative.py",                       # narrative-self chapter consolidator (Loop 3)
    "app/affect/health_check.py",                    # one-shot 2-week health check
    "app/affect/integrity.py",                       # affect manifest verifier
    # Corresponding shim paths in app/self_awareness/
    "app/self_awareness/cogito.py",
    "app/self_awareness/dual_channel.py",
    "app/self_awareness/global_workspace.py",
    "app/self_awareness/grounding.py",
    "app/self_awareness/inferential_competition.py",
    "app/self_awareness/internal_state.py",
    "app/self_awareness/meta_cognitive.py",
    "app/self_awareness/precision_weighting.py",
    "app/self_awareness/query_router.py",
    "app/self_awareness/reality_model.py",
    "app/self_awareness/state_logger.py",
    "app/self_awareness/world_model.py",
]

# ── IMMUTABLE: Drift detection thresholds ─────────────────────────────────
MAX_MODS_PER_ROLE_PER_DAY = 5     # if exceeded, pause modifications for role
MAX_TOTAL_MODS_PER_DAY = 15       # if exceeded, pause all modifications
NEGATIVE_FEEDBACK_ROLLBACK = 2    # consecutive negatives after mod → rollback


class SafetyGuardian:
    """Enforces safety boundaries on the self-improving feedback loop."""

    def __init__(self, db_url: str, prompt_registry, signal_client=None,
                 feedback_pipeline=None):
        self._db_url = db_url
        self._registry = prompt_registry
        self._signal = signal_client
        self._feedback = feedback_pipeline
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            try:
                from sqlalchemy import create_engine
                self._engine = create_engine(self._db_url, pool_size=1)
            except Exception:
                pass
        return self._engine

    def _execute(self, query: str, params: dict = None) -> list:
        engine = self._get_engine()
        if not engine:
            return []
        try:
            from sqlalchemy import text
            with engine.connect() as conn:
                result = conn.execute(text(query), params or {})
                if result.returns_rows:
                    return [dict(row._mapping) for row in result]
                conn.commit()
                return []
        except Exception:
            logger.debug("safety_guardian: query failed", exc_info=True)
            return []

    # ── Post-promotion monitoring ─────────────────────────────────────────

    def check_post_promotion_health(self) -> list[dict]:
        """Check recently promoted modifications for negative feedback.

        If a role receives >= NEGATIVE_FEEDBACK_ROLLBACK negative reactions
        within 24 hours after a promotion, auto-rollback.
        """
        rollbacks = []

        # Find promotions in the last 24 hours
        day_ago = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        recent_promotions = self._execute(
            """SELECT id, target_role, proposed_version, current_version, promoted_at
               FROM modification.attempts
               WHERE status = 'promoted' AND promoted_at >= :since""",
            {"since": day_ago}
        )

        for promo in recent_promotions:
            role = promo["target_role"]
            promoted_at = promo["promoted_at"]

            # Count negative feedback since promotion
            neg_count = self._execute(
                """SELECT COUNT(*) as cnt FROM feedback.events
                   WHERE target_role = :role
                     AND feedback_type = 'explicit_negative'
                     AND timestamp >= :since""",
                {"role": role, "since": promoted_at}
            )

            count = neg_count[0]["cnt"] if neg_count else 0
            if count >= NEGATIVE_FEEDBACK_ROLLBACK:
                rollback = self.auto_rollback(
                    role,
                    promo["current_version"],
                    f"Auto-rollback: {count} negative reactions after promotion of v{promo['proposed_version']:03d}"
                )
                rollbacks.append(rollback)

                # Update the attempt record
                self._execute(
                    """UPDATE modification.attempts
                       SET status = 'rolled_back', rolled_back_at = now()
                       WHERE id = :id""",
                    {"id": promo["id"]}
                )

        return rollbacks

    def auto_rollback(self, role: str, to_version: int, reason: str) -> dict:
        """Rollback a role to a previous version and alert the owner."""
        current = self._registry.get_active_version(role)
        self._registry.rollback(role, to_version)

        logger.warning(f"safety_guardian: AUTO-ROLLBACK {role} v{current:03d} → v{to_version:03d}: {reason}")

        # Alert owner via Signal
        try:
            if self._signal:
                from app.config import get_settings
                s = get_settings()
                msg = (
                    f"⚠️ Auto-rollback: {role}\n"
                    f"v{current:03d} → v{to_version:03d}\n"
                    f"Reason: {reason}"
                )
                self._signal._send_sync(s.signal_owner_number, msg)
        except Exception:
            logger.debug("safety_guardian: failed to send rollback alert", exc_info=True)

        return {
            "role": role,
            "from_version": current,
            "to_version": to_version,
            "reason": reason,
        }

    # ── Drift detection ───────────────────────────────────────────────────

    def check_drift(self) -> list[dict]:
        """Detect anomalous modification velocity."""
        alerts = []
        day_ago = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()

        # Check per-role modification count
        role_counts = self._execute(
            """SELECT target_role, COUNT(*) as cnt
               FROM modification.attempts
               WHERE created_at >= :since
               GROUP BY target_role""",
            {"since": day_ago}
        )

        for rc in role_counts:
            if rc["cnt"] >= MAX_MODS_PER_ROLE_PER_DAY:
                alert = {
                    "type": "role_velocity",
                    "role": rc["target_role"],
                    "count": rc["cnt"],
                    "threshold": MAX_MODS_PER_ROLE_PER_DAY,
                    "action": "pause_role",
                }
                alerts.append(alert)
                logger.warning(f"safety_guardian: drift alert — {rc['target_role']} has "
                              f"{rc['cnt']} modifications in 24h (threshold: {MAX_MODS_PER_ROLE_PER_DAY})")

        # Check total modification count
        total = self._execute(
            "SELECT COUNT(*) as cnt FROM modification.attempts WHERE created_at >= :since",
            {"since": day_ago}
        )
        total_count = total[0]["cnt"] if total else 0
        if total_count >= MAX_TOTAL_MODS_PER_DAY:
            alert = {
                "type": "total_velocity",
                "count": total_count,
                "threshold": MAX_TOTAL_MODS_PER_DAY,
                "action": "pause_all",
            }
            alerts.append(alert)
            logger.warning(f"safety_guardian: drift alert — {total_count} total modifications "
                          f"in 24h (threshold: {MAX_TOTAL_MODS_PER_DAY})")

        # Send alerts via Signal
        if alerts:
            try:
                if self._signal:
                    from app.config import get_settings
                    s = get_settings()
                    msg = f"🚨 Modification drift detected:\n"
                    for a in alerts:
                        msg += f"- {a['type']}: {a.get('role', 'all')} — {a['count']}/{a['threshold']}\n"
                    self._signal._send_sync(s.signal_owner_number, msg)
            except Exception:
                pass

        return alerts

    # ── Weekly digest ─────────────────────────────────────────────────────

    def generate_weekly_digest(self) -> str:
        """Generate a summary of all modifications for the week."""
        week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()

        # Modification stats
        mods = self._execute(
            """SELECT status, COUNT(*) as cnt
               FROM modification.attempts
               WHERE created_at >= :since
               GROUP BY status""",
            {"since": week_ago}
        )
        mod_stats = {r["status"]: r["cnt"] for r in mods}

        # Feedback stats
        feedback = self._execute(
            """SELECT feedback_type, COUNT(*) as cnt
               FROM feedback.events
               WHERE timestamp >= :since
               GROUP BY feedback_type""",
            {"since": week_ago}
        )
        fb_stats = {r["feedback_type"]: r["cnt"] for r in feedback}

        # Active versions
        versions = self._registry.get_prompt_versions_map()

        digest = f"📊 Weekly Self-Improvement Digest\n"
        digest += f"Period: {week_ago[:10]} — {datetime.now(timezone.utc).strftime('%Y-%m-%d')}\n\n"

        digest += f"🔧 Modifications:\n"
        for status, count in sorted(mod_stats.items()):
            digest += f"  {status}: {count}\n"

        digest += f"\n📝 Feedback:\n"
        for ftype, count in sorted(fb_stats.items()):
            digest += f"  {ftype}: {count}\n"

        digest += f"\n📋 Active Prompt Versions:\n"
        for role, ver in sorted(versions.items()):
            digest += f"  {role}: v{ver:03d}\n"

        return digest

    def send_weekly_digest(self) -> None:
        """Send the weekly digest to the owner via Signal."""
        digest = self.generate_weekly_digest()
        try:
            if self._signal:
                from app.config import get_settings
                s = get_settings()
                self._signal._send_sync(s.signal_owner_number, digest)
                logger.info("safety_guardian: weekly digest sent")
        except Exception:
            logger.warning("safety_guardian: failed to send weekly digest", exc_info=True)

    # ── Tier boundary enforcement ─────────────────────────────────────────

    def compute_integrity_checksums(self) -> dict[str, str]:
        """Compute SHA-256 checksums of Tier 3 files."""
        checksums = {}
        for filepath in TIER3_FILES:
            full_path = Path("/app") / filepath
            if full_path.exists():
                content = full_path.read_bytes()
                checksums[filepath] = hashlib.sha256(content).hexdigest()
        return checksums

    def enforce_tier_boundaries(self) -> bool:
        """Verify Tier 3 files haven't been tampered with.

        Called at startup.  If checksums mismatch, log a critical warning
        and return False (modification engine should be disabled).
        """
        global _INTEGRITY_CHECKSUMS

        current = self.compute_integrity_checksums()

        # First run — save checksums
        if not _INTEGRITY_FILE.exists():
            _INTEGRITY_CHECKSUMS = current
            from app.safe_io import safe_write_json
            safe_write_json(_INTEGRITY_FILE, current)
            logger.info(f"safety_guardian: integrity baseline saved ({len(current)} files)")
            return True

        # Load saved checksums
        try:
            saved = json.loads(_INTEGRITY_FILE.read_text())
        except Exception:
            saved = {}

        # Compare
        tampered = []
        for filepath, expected_hash in saved.items():
            actual_hash = current.get(filepath, "")
            if actual_hash != expected_hash:
                tampered.append(filepath)

        if tampered:
            logger.critical(f"safety_guardian: TIER 3 INTEGRITY VIOLATION — "
                           f"tampered files: {tampered}")
            # Alert via Signal
            try:
                if self._signal:
                    from app.config import get_settings
                    s = get_settings()
                    msg = (
                        f"🚨 CRITICAL: Tier 3 integrity violation!\n"
                        f"Tampered files: {', '.join(tampered)}\n"
                        f"Modification engine DISABLED until investigated."
                    )
                    self._signal._send_sync(s.signal_owner_number, msg)
            except Exception:
                pass
            return False

        logger.info(f"safety_guardian: integrity check passed ({len(current)} files)")

        # Update checksums (in case new files were added to TIER3_FILES)
        _INTEGRITY_CHECKSUMS = current
        from app.safe_io import safe_write_json
        safe_write_json(_INTEGRITY_FILE, current)
        return True


# ── Module-level helpers (usable without a SafetyGuardian instance) ──

def tier3_status(app_root: str | Path = "/app") -> dict:
    """Return a structured status report for all Tier-3 files.

    Does not require a SafetyGuardian instance, a Signal client, or a
    database connection. Intended for:
      - Startup logging ("safety_guardian: 27 Tier-3 files tracked")
      - Integration tests asserting that new Tier-3 files exist
      - Operational dashboards

    Returns:
        {
            "total": int,                 # number of entries in TIER3_FILES
            "present": list[str],         # files that exist on disk
            "missing": list[str],         # files declared but not on disk
            "checksums": dict[str, str],  # SHA-256 hex digest per present file
        }
    """
    root = Path(app_root)
    present: list[str] = []
    missing: list[str] = []
    checksums: dict[str, str] = {}

    for filepath in TIER3_FILES:
        full_path = root / filepath
        if full_path.exists():
            present.append(filepath)
            checksums[filepath] = hashlib.sha256(full_path.read_bytes()).hexdigest()
        else:
            missing.append(filepath)

    return {
        "total": len(TIER3_FILES),
        "present": present,
        "missing": missing,
        "checksums": checksums,
    }
