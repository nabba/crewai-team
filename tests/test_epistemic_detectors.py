"""Tests for app.epistemic.detectors and app.epistemic.calibration."""
from __future__ import annotations

import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# ── Stub heavy/optional deps (must precede app imports) ──────────────
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
from app.epistemic.biases import (  # noqa: E402
    BIAS_LIBRARY,
    BiasMatch,
    DetectorPhase,
    Severity,
    severity_rank,
)
from app.epistemic.calibration import (  # noqa: E402
    CalibrationVerdict,
    calibration_check,
)
from app.epistemic.detectors import (  # noqa: E402
    Detector,
    _reset_for_tests as _reset_detectors,
    realtime_detectors,
    register_realtime,
)
from app.epistemic.detectors.realtime import InferenceAsFactDetector  # noqa: E402
from app.epistemic.registry import (  # noqa: E402
    _reset_for_tests as _reset_hooks,
    register as register_claim_hook,
)


def _claim(**overrides) -> Claim:
    defaults = dict(
        task_id="task_abc",
        agent_role="researcher",
        statement="the path is not a symlink",
        status=VerificationStatus.INFERRED,
        register=Register.DECLARATIVE,
        load_bearing=True,
    )
    defaults.update(overrides)
    return Claim.new(**defaults)


# ============================================================================
# BiasLibrary
# ============================================================================

class TestBiasLibrary(unittest.TestCase):

    def test_inference_as_fact_present_with_correct_severity(self):
        d = BIAS_LIBRARY.get("inference_as_fact")
        self.assertEqual(d.severity, Severity.HIGH)
        self.assertEqual(d.phase, DetectorPhase.REALTIME)

    def test_unknown_id_raises(self):
        with self.assertRaises(KeyError):
            BIAS_LIBRARY.get("not_a_real_bias")

    def test_severity_total_order(self):
        self.assertLess(severity_rank(Severity.LOW), severity_rank(Severity.MEDIUM))
        self.assertLess(severity_rank(Severity.MEDIUM), severity_rank(Severity.HIGH))
        self.assertLess(severity_rank(Severity.HIGH), severity_rank(Severity.CRITICAL))


# ============================================================================
# InferenceAsFactDetector
# ============================================================================

class TestInferenceAsFactDetector(unittest.TestCase):

    def setUp(self):
        self.detector = InferenceAsFactDetector()
        self.ledger = Ledger(task_id="task_abc")

    def test_fires_on_inferred_declarative_with_verifier(self):
        c = _claim(
            verifying_action=VerifyingAction(
                tool="readlink", args={"path": "/x"},
                expected_signal="empty=no", estimated_seconds=0.5,
            ),
        )
        matches = list(self.detector.detect(self.ledger, claim=c))
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].bias_id, "inference_as_fact")
        self.assertEqual(matches[0].matched_claim_ids, (c.claim_id,))
        self.assertEqual(matches[0].severity, Severity.HIGH)

    def test_does_not_fire_when_status_verified(self):
        c = _claim(
            status=VerificationStatus.VERIFIED,
            verifying_action=VerifyingAction(
                tool="readlink", args={}, expected_signal="x", estimated_seconds=0.5,
            ),
        )
        self.assertEqual(list(self.detector.detect(self.ledger, claim=c)), [])

    def test_does_not_fire_when_register_hedged(self):
        c = _claim(
            register=Register.HEDGED,
            verifying_action=VerifyingAction(
                tool="readlink", args={}, expected_signal="x", estimated_seconds=0.5,
            ),
        )
        self.assertEqual(list(self.detector.detect(self.ledger, claim=c)), [])

    def test_does_not_fire_without_verifier(self):
        c = _claim(verifying_action=None)  # no cheap verifier available
        self.assertEqual(list(self.detector.detect(self.ledger, claim=c)), [])

    def test_does_not_fire_on_full_scan_call(self):
        # Phase 1 contract: realtime detectors only return for specific
        # claim emission. A None claim => empty.
        self.assertEqual(list(self.detector.detect(self.ledger, claim=None)), [])


# ============================================================================
# Realtime meta-hook: registration + isolation
# ============================================================================

class TestRealtimeMetaHook(unittest.TestCase):

    def setUp(self):
        # Tests expect a clean detector + hook registry. The realtime
        # module re-registers its hook on import; our reset clears
        # everything, then we re-register what we need.
        _reset_detectors()
        _reset_hooks()
        # Re-register the meta-hook + the inference_as_fact detector.
        from app.epistemic.detectors.realtime import (
            INFERENCE_AS_FACT, _realtime_meta_hook,
        )
        register_realtime(INFERENCE_AS_FACT)
        register_claim_hook(_realtime_meta_hook)

    def test_meta_hook_persists_matches(self):
        with patch.dict(os.environ, {"EPISTEMIC_ENABLED": "true"}), \
             patch("app.epistemic.span_writer.execute") as mock_exec:
            ledger = Ledger(task_id="task_abc")
            ledger.emit(_claim(
                verifying_action=VerifyingAction(
                    tool="readlink", args={"path": "/x"},
                    expected_signal="empty=no", estimated_seconds=0.5,
                ),
            ))
            # Two writes: one to epistemic_claims, one to epistemic_bias_matches.
            self.assertEqual(mock_exec.call_count, 2)
            sqls = [call.args[0] for call in mock_exec.call_args_list]
            self.assertTrue(any("INSERT INTO control_plane.epistemic_claims" in s
                                for s in sqls))
            self.assertTrue(any("INSERT INTO control_plane.epistemic_bias_matches" in s
                                for s in sqls))

    def test_buggy_detector_does_not_break_emission(self):
        class BoomDetector(Detector):
            bias_id = "boom"
            def detect(self, ledger, *, claim=None):
                raise RuntimeError("intentional explosion")

        register_realtime(BoomDetector())

        # A normal emission should still succeed.
        ledger = Ledger(task_id="task_abc")
        c = _claim(verifying_action=None)  # no inference_as_fact firing
        result = ledger.emit(c)  # must not raise
        self.assertIs(result, c)

    def test_no_matches_no_persist_call_for_bias(self):
        with patch.dict(os.environ, {"EPISTEMIC_ENABLED": "true"}), \
             patch("app.epistemic.span_writer.execute") as mock_exec:
            ledger = Ledger(task_id="task_abc")
            # Hedged register → no inference_as_fact match.
            ledger.emit(_claim(
                register=Register.HEDGED,
                verifying_action=VerifyingAction(
                    tool="readlink", args={"path": "/x"},
                    expected_signal="empty=no", estimated_seconds=0.5,
                ),
            ))
            # Only the claims write fires; bias_matches table is untouched.
            self.assertEqual(mock_exec.call_count, 1)
            self.assertIn(
                "INSERT INTO control_plane.epistemic_claims",
                mock_exec.call_args_list[0].args[0],
            )


# ============================================================================
# Calibration check
# ============================================================================

class TestCalibrationCheck(unittest.TestCase):

    def setUp(self):
        _reset_hooks()
        self.ledger = Ledger(task_id="task_abc")

    def test_no_matches_proceeds(self):
        verdict = calibration_check(ledger=self.ledger, matches=[])
        self.assertTrue(verdict.proceed)
        self.assertEqual(verdict.suggested_action, "ship")
        self.assertEqual(verdict.biases_detected, ())

    def test_warn_mode_default_inference_as_fact_does_not_block(self):
        # Without EPISTEMIC_CALIBRATION_BLOCKS_OUTPUT, even high-severity
        # matches result in proceed=True (Phase 1 ships in warn-mode).
        c = _claim(
            verifying_action=VerifyingAction(
                tool="readlink", args={"path": "/x"},
                expected_signal="empty=no", estimated_seconds=0.5,
            ),
        )
        self.ledger.emit(c)
        match = BiasMatch(
            bias_id="inference_as_fact",
            matched_claim_ids=(c.claim_id,),
            severity=Severity.HIGH,
        )
        verdict = calibration_check(ledger=self.ledger, matches=[match])
        self.assertTrue(verdict.proceed)
        self.assertEqual(verdict.suggested_action, "verify")  # has unverified load-bearing
        self.assertEqual(verdict.forced_verifier_claim_ids, (c.claim_id,))

    def test_verify_action_when_unverified_load_bearing_with_verifier(self):
        c = _claim(
            verifying_action=VerifyingAction(
                tool="readlink", args={"path": "/x"},
                expected_signal="empty=no", estimated_seconds=0.5,
            ),
        )
        self.ledger.emit(c)
        match = BiasMatch(
            bias_id="inference_as_fact",
            matched_claim_ids=(c.claim_id,),
            severity=Severity.HIGH,
        )
        verdict = calibration_check(ledger=self.ledger, matches=[match])
        self.assertEqual(verdict.suggested_action, "verify")

    def test_hedge_action_when_no_verifier_available(self):
        # No verifier on the claim → can't run a verifier; suggest hedging.
        c = _claim(verifying_action=None)
        self.ledger.emit(c)
        match = BiasMatch(
            bias_id="inference_as_fact",
            matched_claim_ids=(c.claim_id,),
            severity=Severity.HIGH,
        )
        verdict = calibration_check(ledger=self.ledger, matches=[match])
        self.assertEqual(verdict.suggested_action, "hedge")

    def test_critical_severity_suggests_peer_review(self):
        c = _claim()
        self.ledger.emit(c)
        match = BiasMatch(
            bias_id="inference_as_fact",
            matched_claim_ids=(c.claim_id,),
            severity=Severity.CRITICAL,
        )
        verdict = calibration_check(ledger=self.ledger, matches=[match])
        self.assertEqual(verdict.suggested_action, "peer_review")

    def test_blocking_mode_blocks_critical_severity(self):
        c = _claim()
        self.ledger.emit(c)
        match = BiasMatch(
            bias_id="inference_as_fact",
            matched_claim_ids=(c.claim_id,),
            severity=Severity.CRITICAL,
        )
        with patch.dict(os.environ, {"EPISTEMIC_CALIBRATION_BLOCKS_OUTPUT": "true"}):
            verdict = calibration_check(ledger=self.ledger, matches=[match])
        self.assertFalse(verdict.proceed)

    def test_summary_note_aggregates_counts(self):
        c1, c2 = _claim(), _claim()
        self.ledger.emit(c1); self.ledger.emit(c2)
        matches = [
            BiasMatch(bias_id="inference_as_fact",
                      matched_claim_ids=(c1.claim_id,), severity=Severity.HIGH),
            BiasMatch(bias_id="inference_as_fact",
                      matched_claim_ids=(c2.claim_id,), severity=Severity.HIGH),
        ]
        verdict = calibration_check(ledger=self.ledger, matches=matches)
        self.assertEqual(verdict.note_for_post_mortem, "inference_as_fact×2")

    def test_verdict_jsonable(self):
        c = _claim()
        self.ledger.emit(c)
        match = BiasMatch(
            bias_id="inference_as_fact",
            matched_claim_ids=(c.claim_id,),
            severity=Severity.HIGH,
        )
        verdict = calibration_check(ledger=self.ledger, matches=[match])
        as_json = verdict.as_jsonable()
        self.assertIn("proceed", as_json)
        self.assertIn("biases_detected", as_json)
        self.assertEqual(len(as_json["biases_detected"]), 1)
        self.assertEqual(as_json["biases_detected"][0]["bias_id"], "inference_as_fact")


if __name__ == "__main__":
    unittest.main()
