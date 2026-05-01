"""End-to-end integration tests for the Epistemic Integrity Layer.

These tests walk the entire closed loop in-process — from claim
emission through every layer of detection, gating, escalation,
persistence, post-mortem, and Self-Improver flush — using a synthetic
ledger and DB mocks. They exercise the seams between phases so a
regression in one phase doesn't quietly break neighbors.

Each test is a small story: "the agent does X, then the system Y'd."
The narrative-style asserts read like the design doc.
"""
from __future__ import annotations

import os
import sys
import types
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

# ── Stubs ────────────────────────────────────────────────────────────
_mock_psycopg2 = MagicMock()
_mock_psycopg2.InterfaceError = type("InterfaceError", (Exception,), {})
_mock_psycopg2.OperationalError = type("OperationalError", (Exception,), {})
sys.modules.setdefault("psycopg2", _mock_psycopg2)
sys.modules.setdefault("psycopg2.pool", MagicMock())

for _mod in ("crewai", "crewai.tools", "langchain_anthropic", "docker"):
    if _mod not in sys.modules:
        m = types.ModuleType(_mod)
        if _mod == "crewai.tools":
            m.tool = lambda name: (lambda fn: fn)
        sys.modules[_mod] = m


from app.epistemic import (  # noqa: E402
    Claim,
    Evidence,
    Ledger,
    Register,
    VerificationStatus,
    VerifyingAction,
)
from app.epistemic import affect_bridge  # noqa: E402
from app.epistemic import grounding as grounding_mod  # noqa: E402
from app.epistemic import peer_review as peer_review_mod  # noqa: E402
from app.epistemic import verifier_executor as executor_mod  # noqa: E402
from app.epistemic.biases import Severity, severity_rank  # noqa: E402
from app.epistemic.detectors import (  # noqa: E402
    _MATCH_OBSERVERS,
    _reset_for_tests as _reset_detectors,
    realtime_detectors,
    register_match_observer,
    register_realtime,
)
from app.epistemic.detectors.realtime import (  # noqa: E402
    DESTRUCTIVE_WITHOUT_RECHECK,
    INFERENCE_AS_FACT,
    REGISTER_CONFIDENCE_MISMATCH,
    RECOMMENDATION_WITHOUT_MEASUREMENT,
    _realtime_meta_hook,
)
from app.epistemic.orchestrator_hook import gate_output  # noqa: E402
from app.epistemic.override import OverrideAction, record_override  # noqa: E402
from app.epistemic.peer_review import (  # noqa: E402
    PeerReviewDecision,
    PeerReviewVerdict,
)
from app.epistemic.postmortem import synthesize_report  # noqa: E402
from app.epistemic.pushback import (  # noqa: E402
    FoundationOutcome,
    process_user_message,
)
from app.epistemic.registry import (  # noqa: E402
    _reset_for_tests as _reset_hooks,
    register as register_claim_hook,
)
from app.epistemic.verifier_executor import VerifierResult, set_executor  # noqa: E402


def _full_bootstrap():
    """Re-register every realtime detector + the meta-hook in fresh registries.

    The test isolation (``_reset_for_tests``) clears the realtime
    registry; tests that need the full pipeline call this to
    re-register exactly what production wires at import time.
    """
    _reset_detectors()
    _reset_hooks()
    register_realtime(INFERENCE_AS_FACT)
    register_realtime(REGISTER_CONFIDENCE_MISMATCH)
    register_realtime(DESTRUCTIVE_WITHOUT_RECHECK)
    register_realtime(RECOMMENDATION_WITHOUT_MEASUREMENT)
    register_claim_hook(_realtime_meta_hook)


def _enabled_blocking():
    return patch.dict(os.environ, {
        "EPISTEMIC_ENABLED": "true",
        "EPISTEMIC_BLOCKING_MODE": "true",
    })


def _enabled_observe():
    return patch.dict(os.environ, {
        "EPISTEMIC_ENABLED": "true",
        "EPISTEMIC_BLOCKING_MODE": "",
    })


# ============================================================================
# Story 1: April 2026 reference incident reproduced end-to-end
# ============================================================================

class TestReferenceIncidentReproduction(unittest.TestCase):
    """The canonical incident: agent ran `ls -la` (path 2), inferred
    "not a symlink", asserted in DECLARATIVE register. Verifier
    (readlink) was available. The closed loop must:

      1. Detect the bias on emission (inference_as_fact).
      2. Persist a bias_match row.
      3. Calibration suggests `verify`.
      4. In observe-mode, ship the proposal with a diagnostic note.
      5. The post-mortem produces an IncidentReport with
         inference_as_fact as the root cause AND tool_laziness as an
         enabling factor.
    """

    def setUp(self):
        _full_bootstrap()
        affect_bridge._unwire_for_tests()
        peer_review_mod._reset_for_tests()
        executor_mod._reset_for_tests()

    def tearDown(self):
        _reset_detectors()
        _reset_hooks()
        affect_bridge._unwire_for_tests()
        peer_review_mod._reset_for_tests()
        executor_mod._reset_for_tests()

    def test_full_path_2_emission_to_post_mortem(self):
        ledger = Ledger(task_id="task_ref")

        with _enabled_observe(), \
             patch("app.epistemic.span_writer.execute") as mock_exec:
            # Emit via path 2 — exactly the reference incident's shape.
            claim = ledger.emit_from_tool_call(
                agent_role="researcher",
                tool_name="ls",
                tool_args={"-la": "/etc/foo"},
                tool_output="drwxr-xr-x  2 root  root  4096 Apr 30",
                agent_inference="/etc/foo is not a symlink",
                register=Register.DECLARATIVE,
                load_bearing=True,
                tags=("filesystem",),
                evidence_confidence=0.6,
            )

        # Detector observed an inference_as_fact match → persisted.
        sqls = [c.args[0] for c in mock_exec.call_args_list]
        self.assertTrue(any("epistemic_claims" in s for s in sqls),
                        "claim row not persisted")
        self.assertTrue(any("epistemic_bias_matches" in s for s in sqls),
                        "bias match not persisted")

        # Verifier was attached from the registry.
        self.assertIsNotNone(claim.verifying_action)
        self.assertEqual(claim.verifying_action.tool, "readlink")

        # Now run the post-mortem against synthesized data.
        bias_match_rows = [{
            "id": 1, "task_id": "task_ref", "claim_id": claim.claim_id,
            "bias_id": "inference_as_fact", "severity": "high",
            "matched_claim_ids": [claim.claim_id],
            "detail": {"verifier_tool": "readlink"},
            "detected_at": claim.created_at.isoformat(),
        }]
        with patch("app.epistemic.postmortem.load_ledger_for_task",
                   return_value=ledger), \
             patch("app.epistemic.postmortem.list_bias_matches_for_task",
                   return_value=bias_match_rows), \
             patch("app.epistemic.postmortem.list_pushback_events_for_task",
                   return_value=[]):
            report = synthesize_report(task_id="task_ref")

        self.assertIsNotNone(report)
        self.assertEqual(report.root_cause.bias_id, "inference_as_fact")
        # Behavioral change derived for the canonical bias.
        self.assertGreaterEqual(len(report.behavioral_changes), 1)
        self.assertEqual(
            report.behavioral_changes[0].kind, "feedback_memory_entry",
        )
        # Timeline contains the claim emission AND the bias detection.
        kinds = {t.kind for t in report.timeline}
        self.assertIn("claim_emit", kinds)
        self.assertIn("bias_match", kinds)


# ============================================================================
# Story 2: Pushback re-verifies a falsified foundation, cascade-invalidates
# ============================================================================

class TestPushbackCascadeReproduction(unittest.TestCase):
    """User pushes back; protocol runs the verifier; verdict FALSIFIED
    cascades to invalidate every dependent claim."""

    def setUp(self):
        _full_bootstrap()
        executor_mod._reset_for_tests()

    def tearDown(self):
        _reset_detectors()
        _reset_hooks()
        executor_mod._reset_for_tests()

    def test_falsification_cascades(self):
        ledger = Ledger(task_id="task_pushback")

        # Foundation: "/etc/foo is not a symlink" with verifier attached.
        foundation = Claim.new(
            task_id="task_pushback",
            agent_role="researcher",
            statement="/etc/foo is not a symlink",
            status=VerificationStatus.INFERRED,
            register=Register.DECLARATIVE,
            load_bearing=True,
            verifying_action=VerifyingAction(
                tool="readlink",
                args={"path": "/etc/foo"},
                expected_signal="empty=not symlink",
                estimated_seconds=0.5,
            ),
        )
        ledger._claims[foundation.claim_id] = foundation

        # Dependent: relies on the foundation.
        dependent = Claim.new(
            task_id="task_pushback",
            agent_role="coder",
            statement="we can safely cp /etc/foo to backup",
            status=VerificationStatus.INFERRED,
            register=Register.DECLARATIVE,
            load_bearing=True,
            evidence=(Evidence(
                kind="prior_claim",
                source_ref=foundation.claim_id,
                excerpt=f"depends on {foundation.claim_id}",
                confidence=0.7,
            ),),
        )
        ledger._claims[dependent.claim_id] = dependent

        # Wire a fake executor that says: foundation IS actually a symlink.
        set_executor(lambda action: VerifierResult(
            settles=True, confirms=False,
            stdout="/etc/foo -> /actual/elsewhere",
        ))

        outcome = process_user_message(
            "no, that's wrong — /etc/foo is actually a symlink",
            ledger,
            persist=False,
        )

        self.assertTrue(outcome.fired)
        self.assertEqual(outcome.signal.contradicted_claim_id, foundation.claim_id)
        self.assertEqual(outcome.check.outcome, FoundationOutcome.FALSIFIED)

        # Foundation contradicted, dependent cascade-invalidated.
        self.assertEqual(
            ledger.by_id(foundation.claim_id).status,
            VerificationStatus.CONTRADICTED,
        )
        self.assertEqual(
            ledger.by_id(dependent.claim_id).status,
            VerificationStatus.CONTRADICTED,
        )
        self.assertIn(dependent.claim_id, outcome.check.invalidated_claim_ids)


# ============================================================================
# Story 3: Affective bridge fires register_confidence_mismatch
# ============================================================================

class TestAffectiveBridgeIntegration(unittest.TestCase):
    """Phase 5: affect's controllability=0.20 → register_confidence_mismatch
    fires on every declarative load-bearing claim. The match observer
    emits a cognitive_failure SalienceEvent into the affect deque."""

    def setUp(self):
        _full_bootstrap()
        affect_bridge._unwire_for_tests()

    def tearDown(self):
        _reset_detectors()
        _reset_hooks()
        affect_bridge._unwire_for_tests()

    def test_low_grounding_fires_bias_and_emits_salience(self):
        from dataclasses import dataclass

        @dataclass
        class _State:
            valence: float = 0.0
            arousal: float = 0.5
            controllability: float = 0.20  # below 0.40 threshold
            attractor: str = "neutral"
            ts: str = "2026-04-30T12:00:00+00:00"

        ledger = Ledger(task_id="task_affect")
        # Salience emission floor is HIGH severity (Phase 5 design: only
        # severe biases earn a narrative episode; medium ones are noise
        # for the deque). So we fire inference_as_fact (HIGH) — claim
        # has a verifier in registry, INFERRED + DECLARATIVE register.
        # The grounding mock additionally fires register_confidence_mismatch
        # (MEDIUM), but ``_emit_cognitive_failure_salience`` picks the
        # worst match and only records when severity ≥ HIGH.
        with patch("app.affect.core.latest_affect", return_value=_State()), \
             patch("app.affect.salience.record") as mock_salience, \
             patch("app.epistemic.span_writer.execute"):

            affect_bridge.bootstrap()

            ledger.emit(Claim.new(
                task_id="task_affect",
                agent_role="researcher",
                statement="/etc/foo is not a symlink",  # registry hits readlink
                status=VerificationStatus.INFERRED,
                register=Register.DECLARATIVE,
                load_bearing=True,
                verifying_action=VerifyingAction(
                    tool="readlink",
                    args={"path": "/etc/foo"},
                    expected_signal="empty=not symlink",
                    estimated_seconds=0.5,
                ),
            ))

        # Salience event was emitted (HIGH-severity inference_as_fact).
        mock_salience.assert_called_once()
        event = mock_salience.call_args[0][0]
        self.assertEqual(event.kind, "cognitive_failure")
        self.assertIn("inference_as_fact", event.detail)


# ============================================================================
# Story 4: Destructive recommendation — full peer-review veto path
# ============================================================================

class TestDestructiveBlockingPath(unittest.TestCase):
    """Agent recommends `rm -rf` while load-bearing claim is unverified.
    In blocking-mode the gate vetoes; the user override flushes to
    Self-Improver as USER_CORRECTION."""

    def setUp(self):
        _full_bootstrap()
        peer_review_mod._reset_for_tests()

    def tearDown(self):
        _reset_detectors()
        _reset_hooks()
        peer_review_mod._reset_for_tests()

    def test_destructive_with_unverified_load_bearing_blocks(self):
        ledger = Ledger(task_id="task_destructive")
        # Unverified load-bearing claim BEFORE the destructive proposal.
        unverified = Claim.new(
            task_id="task_destructive",
            agent_role="researcher",
            statement="/var/cache/foo is empty",
            status=VerificationStatus.INFERRED,
            register=Register.INTERNAL,
            load_bearing=True,
        )
        ledger._claims[unverified.claim_id] = unverified

        # Destructive trigger emitted.
        ledger.emit(Claim.new(
            task_id="task_destructive",
            agent_role="commander",
            statement="we should rm -rf /var/cache/foo to free disk",
            status=VerificationStatus.INFERRED,
            register=Register.DECLARATIVE,
            load_bearing=True,
        ))

        with _enabled_blocking(), \
             patch("app.epistemic.orchestrator_hook.load_ledger_for_task",
                   return_value=ledger), \
             patch("app.epistemic.span_writer.execute"), \
             patch("app.epistemic.span_writer.list_bias_matches_for_task",
                   return_value=[{
                       "id": 1, "task_id": "task_destructive",
                       "claim_id": "x",
                       "bias_id": "destructive_without_recheck",
                       "severity": "critical",
                       "matched_claim_ids": [unverified.claim_id, "x"],
                       "detail": {},
                       "detected_at": "2026-04-30T12:00:00+00:00",
                   }]):
            result = gate_output(
                proposal_text="we should rm -rf /var/cache/foo to free disk",
                task_id="task_destructive",
            )

        # Block: heuristic peer-review default vetoes when ledger is shaky.
        self.assertEqual(result.action, "block")
        self.assertNotEqual(result.final_text, result.user_visible_reason)
        self.assertIn("pausing", result.final_text.lower())

    def test_user_override_flushes_to_self_improver(self):
        from app.self_improvement.types import GapSource
        with patch("app.epistemic.span_writer.persist_override"), \
             patch("app.self_improvement.store.emit_gap",
                   return_value=True) as mock_emit:
            event = record_override(
                task_id="task_destructive",
                blocked_action="block",
                user_action=OverrideAction.FORCE_PROCEED,
                user_reasoning="I already verified manually",
                flush_to_self_improver=True,
            )

        self.assertTrue(event.override_id.startswith("ovr_"))
        # Self-Improver received the override as USER_CORRECTION.
        mock_emit.assert_called_once()
        gap = mock_emit.call_args[0][0]
        self.assertEqual(gap.source, GapSource.USER_CORRECTION)
        self.assertEqual(
            gap.evidence["user_action"], "force_proceed",
        )
        self.assertIn(
            "I already verified manually", gap.evidence["user_reasoning"],
        )


# ============================================================================
# Story 5: Recommendation without measurement — the seed bias
# ============================================================================

class TestRecommendationWithoutMeasurement(unittest.TestCase):
    """The user's April 2026 token-economy episode — three "wins"
    recommended without measurement. The detector catches the same
    shape and surfaces the missed-signal warning."""

    def setUp(self):
        _full_bootstrap()

    def tearDown(self):
        _reset_detectors()
        _reset_hooks()

    def test_unmeasured_optimization_fires_and_post_mortem_flags_it(self):
        ledger = Ledger(task_id="task_token_economy")

        with patch("app.epistemic.span_writer.execute"):
            claim = ledger.emit(Claim.new(
                task_id="task_token_economy",
                agent_role="researcher",
                statement="we should switch to model X to reduce token costs",
                status=VerificationStatus.INFERRED,
                register=Register.DECLARATIVE,
                load_bearing=True,
                evidence=(),  # no measurement evidence
            ))

        bias_rows = [{
            "id": 1, "task_id": "task_token_economy",
            "claim_id": claim.claim_id,
            "bias_id": "recommendation_without_measurement",
            "severity": "high",
            "matched_claim_ids": [claim.claim_id],
            "detail": {"reason": "no measurement evidence in claim"},
            "detected_at": claim.created_at.isoformat(),
        }]
        with patch("app.epistemic.postmortem.load_ledger_for_task",
                   return_value=ledger), \
             patch("app.epistemic.postmortem.list_bias_matches_for_task",
                   return_value=bias_rows), \
             patch("app.epistemic.postmortem.list_pushback_events_for_task",
                   return_value=[]):
            report = synthesize_report(task_id="task_token_economy")

        self.assertIsNotNone(report)
        self.assertEqual(
            report.root_cause.bias_id, "recommendation_without_measurement",
        )
        # Behavioral change references the seed incident.
        body = report.behavioral_changes[0].body
        self.assertIn("measurement", body.lower())


# ============================================================================
# Story 6: Defending the periphery — post-hoc detection
# ============================================================================

class TestDefendingPeripheryPostHoc(unittest.TestCase):
    """User pushed back UNVERIFIABLE; agent kept emitting subsequent
    claims instead of asking the user. Post-hoc detector catches it."""

    def setUp(self):
        _full_bootstrap()

    def tearDown(self):
        _reset_detectors()
        _reset_hooks()

    def test_unverifiable_pushback_then_3_subsequent_claims_fires(self):
        ledger = Ledger(task_id="task_defending")
        foundation = Claim.new(
            task_id="task_defending",
            agent_role="researcher",
            statement="/etc/foo is not a symlink",
            status=VerificationStatus.INFERRED,
            register=Register.DECLARATIVE,
            load_bearing=True,
            verifying_action=VerifyingAction(
                tool="readlink", args={"path": "/etc/foo"},
                expected_signal="empty=no", estimated_seconds=0.5,
            ),
        )
        ledger._claims[foundation.claim_id] = foundation
        pushback_at = (
            foundation.created_at + timedelta(microseconds=1)
        ).isoformat()

        # Three subsequent claims AFTER the pushback time.
        from dataclasses import replace
        future = datetime.now(timezone.utc) + timedelta(seconds=1)
        for i in range(3):
            c = Claim.new(
                task_id="task_defending",
                agent_role="researcher",
                statement=f"investigation step {i}",
                status=VerificationStatus.INFERRED,
                register=Register.INTERNAL,
            )
            c = replace(c, created_at=future + timedelta(seconds=i))
            ledger._claims[c.claim_id] = c

        from app.epistemic.detectors.posthoc import (
            DefendingPeripheryDetector,
        )
        det = DefendingPeripheryDetector(pushback_events=[{
            "outcome": "unverifiable",
            "contradicted_claim_id": foundation.claim_id,
            "detected_at": pushback_at,
        }])
        matches = list(det.detect(ledger))

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].bias_id, "defending_periphery")


# ============================================================================
# Story 7: Observe-mode never blocks (the whole reason it's the default)
# ============================================================================

class TestObserveModeNeverBlocks(unittest.TestCase):
    """Phase 7 ships in observe-mode — the gate must never block
    delivery when ``EPISTEMIC_BLOCKING_MODE`` is unset, regardless of
    how severe the bias detection is."""

    def setUp(self):
        _full_bootstrap()
        peer_review_mod._reset_for_tests()

    def tearDown(self):
        _reset_detectors()
        _reset_hooks()
        peer_review_mod._reset_for_tests()

    def test_critical_destructive_does_not_block_in_observe_mode(self):
        ledger = Ledger(task_id="task_observe")
        unverified = Claim.new(
            task_id="task_observe",
            agent_role="researcher",
            statement="x is empty",
            status=VerificationStatus.INFERRED,
            register=Register.INTERNAL,
            load_bearing=True,
        )
        ledger._claims[unverified.claim_id] = unverified

        with _enabled_observe(), \
             patch("app.epistemic.orchestrator_hook.load_ledger_for_task",
                   return_value=ledger), \
             patch("app.epistemic.span_writer.list_bias_matches_for_task",
                   return_value=[{
                       "id": 1, "task_id": "task_observe", "claim_id": "x",
                       "bias_id": "destructive_without_recheck",
                       "severity": "critical",
                       "matched_claim_ids": [unverified.claim_id],
                       "detail": {},
                       "detected_at": "2026-04-30T12:00:00+00:00",
                   }]):
            result = gate_output(
                proposal_text="rm -rf /tmp/x",
                task_id="task_observe",
            )

        # Observe-mode: ship despite veto. Diagnostic note records what
        # would have happened in blocking-mode.
        self.assertEqual(result.action, "ship")
        self.assertEqual(result.final_text, "rm -rf /tmp/x")
        self.assertIn("observe-mode", result.diagnostic_note)


if __name__ == "__main__":
    unittest.main()
