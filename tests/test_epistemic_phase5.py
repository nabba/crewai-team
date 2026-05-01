"""Tests for Phase 5:

* compute_factual_grounding (pure derivation from AffectState)
* live_factual_grounding (degradable read of latest_affect)
* bootstrap (idempotent wiring of grounding provider + match observer)
* _emit_cognitive_failure_salience (high-severity → SalienceEvent)
* /epistemic/now calibration block (factual_grounding wired through)
"""
from __future__ import annotations

import os
import sys
import types
import unittest
from dataclasses import dataclass, field
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


from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.epistemic import (  # noqa: E402
    Claim,
    Ledger,
    Register,
    VerificationStatus,
    VerifyingAction,
)
from app.epistemic import affect_bridge  # noqa: E402
from app.epistemic import grounding as grounding_mod  # noqa: E402
from app.epistemic.affect_bridge import (  # noqa: E402
    _emit_cognitive_failure_salience,
    bootstrap,
    compute_factual_grounding,
    live_factual_grounding,
)
from app.epistemic.api import router  # noqa: E402
from app.epistemic.biases import BiasMatch, Severity  # noqa: E402
from app.epistemic.detectors import (  # noqa: E402
    _MATCH_OBSERVERS,
    register_match_observer,
)
from app.epistemic.detectors.realtime import (  # noqa: E402
    RegisterConfidenceMismatchDetector,
)
from app.epistemic.grounding import (  # noqa: E402
    factual_grounding,
    set_grounding_provider,
)


# ── Fake AffectState (matches the public attribute surface) ─────────

@dataclass
class _FakeAffectState:
    valence: float = 0.0
    arousal: float = 0.0
    controllability: float = 0.7
    attractor: str = "neutral"
    ts: str = "2026-04-30T12:00:00+00:00"


def _build_client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# ============================================================================
# compute_factual_grounding
# ============================================================================

class TestComputeFactualGrounding(unittest.TestCase):

    def test_returns_controllability_for_neutral_attractor(self):
        state = _FakeAffectState(controllability=0.85, attractor="peace")
        self.assertAlmostEqual(compute_factual_grounding(state), 0.85)

    def test_clamps_negative_to_zero(self):
        state = _FakeAffectState(controllability=-0.5, attractor="neutral")
        self.assertEqual(compute_factual_grounding(state), 0.0)

    def test_clamps_above_one(self):
        state = _FakeAffectState(controllability=1.5, attractor="neutral")
        self.assertEqual(compute_factual_grounding(state), 1.0)

    def test_distress_attractor_caps_at_half(self):
        # Even with high controllability, distress means felt context
        # is unreliable.
        state = _FakeAffectState(controllability=0.9, attractor="distress")
        self.assertEqual(compute_factual_grounding(state), 0.5)

    def test_frozen_attractor_caps_at_half(self):
        state = _FakeAffectState(controllability=0.95, attractor="frozen")
        self.assertEqual(compute_factual_grounding(state), 0.5)

    def test_low_controllability_in_low_attractor_passes_through(self):
        # If controllability is already below 0.5, the cap is moot.
        state = _FakeAffectState(controllability=0.3, attractor="distress")
        self.assertEqual(compute_factual_grounding(state), 0.3)


# ============================================================================
# live_factual_grounding
# ============================================================================

class TestLiveFactualGrounding(unittest.TestCase):

    def test_returns_none_when_no_state(self):
        with patch("app.affect.core.latest_affect", return_value=None):
            self.assertIsNone(live_factual_grounding())

    def test_returns_none_when_affect_raises(self):
        with patch(
            "app.affect.core.latest_affect",
            side_effect=RuntimeError("affect down"),
        ):
            self.assertIsNone(live_factual_grounding())

    def test_returns_grounding_when_state_present(self):
        with patch(
            "app.affect.core.latest_affect",
            return_value=_FakeAffectState(controllability=0.6, attractor="exploring"),
        ):
            self.assertAlmostEqual(live_factual_grounding(), 0.6)


# ============================================================================
# bootstrap
# ============================================================================

class TestBootstrap(unittest.TestCase):

    def setUp(self):
        affect_bridge._unwire_for_tests()

    def tearDown(self):
        affect_bridge._unwire_for_tests()

    def test_wires_grounding_provider(self):
        result = bootstrap()
        self.assertTrue(result["grounding_wired"])
        # The grounding module's provider is now live_factual_grounding.
        self.assertIs(grounding_mod._provider, live_factual_grounding)

    def test_wires_match_observer(self):
        result = bootstrap()
        self.assertTrue(result["salience_wired"])
        self.assertIn(_emit_cognitive_failure_salience, _MATCH_OBSERVERS)

    def test_idempotent(self):
        bootstrap()
        before = len(_MATCH_OBSERVERS)
        bootstrap()
        # Calling twice does not double-register.
        self.assertEqual(len(_MATCH_OBSERVERS), before)


# ============================================================================
# Cognitive-failure salience emission
# ============================================================================

class TestCognitiveFailureSalience(unittest.TestCase):

    def setUp(self):
        affect_bridge._unwire_for_tests()
        self.ledger = Ledger(task_id="task_abc")
        self.claim = Claim.new(
            task_id="task_abc", agent_role="researcher",
            statement="x", status=VerificationStatus.INFERRED,
            register=Register.DECLARATIVE, load_bearing=True,
        )

    def tearDown(self):
        affect_bridge._unwire_for_tests()

    def test_high_severity_emits_event(self):
        match = BiasMatch(
            bias_id="inference_as_fact",
            matched_claim_ids=(self.claim.claim_id,),
            severity=Severity.HIGH,
            detail={},
        )
        with patch("app.affect.salience.record") as mock_record:
            _emit_cognitive_failure_salience([match], self.claim, self.ledger)
        mock_record.assert_called_once()
        event = mock_record.call_args[0][0]
        self.assertEqual(event.kind, "cognitive_failure")
        self.assertEqual(event.severity, "warn")
        self.assertIn("inference_as_fact", event.detail)

    def test_critical_severity_emits_critical_event(self):
        match = BiasMatch(
            bias_id="destructive_without_recheck",
            matched_claim_ids=(self.claim.claim_id,),
            severity=Severity.CRITICAL,
            detail={},
        )
        with patch("app.affect.salience.record") as mock_record:
            _emit_cognitive_failure_salience([match], self.claim, self.ledger)
        event = mock_record.call_args[0][0]
        self.assertEqual(event.severity, "critical")

    def test_medium_severity_does_not_emit(self):
        match = BiasMatch(
            bias_id="register_confidence_mismatch",
            matched_claim_ids=(self.claim.claim_id,),
            severity=Severity.MEDIUM,
            detail={},
        )
        with patch("app.affect.salience.record") as mock_record:
            _emit_cognitive_failure_salience([match], self.claim, self.ledger)
        mock_record.assert_not_called()

    def test_empty_matches_no_op(self):
        with patch("app.affect.salience.record") as mock_record:
            _emit_cognitive_failure_salience([], self.claim, self.ledger)
        mock_record.assert_not_called()

    def test_picks_worst_severity_from_batch(self):
        # If multiple matches fire on one claim, the salience event
        # records the worst — the rest are still in epistemic_bias_matches.
        matches = [
            BiasMatch(bias_id="a", matched_claim_ids=(self.claim.claim_id,),
                      severity=Severity.MEDIUM, detail={}),
            BiasMatch(bias_id="b", matched_claim_ids=(self.claim.claim_id,),
                      severity=Severity.CRITICAL, detail={}),
            BiasMatch(bias_id="c", matched_claim_ids=(self.claim.claim_id,),
                      severity=Severity.HIGH, detail={}),
        ]
        with patch("app.affect.salience.record") as mock_record:
            _emit_cognitive_failure_salience(matches, self.claim, self.ledger)
        event = mock_record.call_args[0][0]
        self.assertEqual(event.severity, "critical")
        self.assertIn("b", event.detail)


# ============================================================================
# Realtime detector with bridge wired
# ============================================================================

class TestRegisterConfidenceMismatchWithBridge(unittest.TestCase):

    def setUp(self):
        affect_bridge._unwire_for_tests()
        self.detector = RegisterConfidenceMismatchDetector()
        self.ledger = Ledger(task_id="task_abc")

    def tearDown(self):
        affect_bridge._unwire_for_tests()

    def test_low_grounding_via_bridge_fires_detector(self):
        # Bootstrap wires live_factual_grounding as the provider.
        # Mock latest_affect to return a low-controllability state.
        with patch(
            "app.affect.core.latest_affect",
            return_value=_FakeAffectState(controllability=0.20, attractor="neutral"),
        ):
            bootstrap()
            c = Claim.new(
                task_id="task_abc", agent_role="researcher",
                statement="the deploy will succeed",
                status=VerificationStatus.INFERRED,
                register=Register.DECLARATIVE,
                load_bearing=True,
            )
            matches = list(self.detector.detect(self.ledger, claim=c))
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].bias_id, "register_confidence_mismatch")
        self.assertAlmostEqual(matches[0].detail["factual_grounding"], 0.20)


# ============================================================================
# /epistemic/now calibration block
# ============================================================================

class TestNowCalibrationBlock(unittest.TestCase):

    def test_returns_grounding_when_affect_wired(self):
        client = _build_client()
        with patch(
            "app.affect.core.latest_affect",
            return_value=_FakeAffectState(
                valence=0.3, arousal=0.4,
                controllability=0.75, attractor="exploring",
            ),
        ):
            resp = client.get("/epistemic/now")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("calibration", body)
        cal = body["calibration"]
        self.assertAlmostEqual(cal["factual_grounding"], 0.75)
        self.assertAlmostEqual(cal["valence"], 0.3)
        self.assertAlmostEqual(cal["arousal"], 0.4)
        self.assertEqual(cal["attractor"], "exploring")

    def test_returns_null_grounding_when_no_state(self):
        client = _build_client()
        with patch("app.affect.core.latest_affect", return_value=None):
            resp = client.get("/epistemic/now")
        body = resp.json()
        self.assertIsNone(body["calibration"]["factual_grounding"])
        self.assertIsNone(body["calibration"]["attractor"])


if __name__ == "__main__":
    unittest.main()
