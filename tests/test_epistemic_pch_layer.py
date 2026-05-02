"""Tests for the Pearl Causal Hierarchy layer tagging on claims.

Covers:
* round-trip serialization of ``pch_layer`` + ``causal_evidence_kinds``
  through ``Claim.as_jsonable`` / ``Claim.from_jsonable``
* the ``CausalLayerOverreachDetector`` (L1 silent / L2 unbacked fires /
  L2 with controlled-experiment evidence silent / explicit pch_layer
  overrides the heuristic / L3 fires too)
* ``app.improvement_narrative._emit_l2_narrative_claim`` produces a
  Claim that does NOT trigger the detector (well-formed L2 emission),
  and creates+closes a synthetic ``crew_tasks`` row so the FK on
  ``epistemic_claims.task_id`` resolves and the BiasFeed dashboard
  can surface the emission.
"""
from __future__ import annotations

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


from app.epistemic.ledger import (  # noqa: E402
    CAUSAL_EVIDENCE_KINDS_L2,
    Claim,
    Evidence,
    Ledger,
    Register,
    VerificationStatus,
)
from app.epistemic.detectors.realtime import (  # noqa: E402
    CausalLayerOverreachDetector,
    _infer_pch_layer,
)


def _claim(**overrides) -> Claim:
    defaults = dict(
        task_id="task_abc",
        agent_role="self_improver",
        statement="Yesterday's experiments produced 4 meaningful improvements",
        status=VerificationStatus.VERIFIED,
        register=Register.DECLARATIVE,
        load_bearing=True,
    )
    defaults.update(overrides)
    return Claim.new(**defaults)


# ============================================================================
# Round-trip serialization
# ============================================================================

class TestClaimPchSerialization(unittest.TestCase):

    def test_default_is_none_and_empty(self):
        c = _claim()
        self.assertIsNone(c.pch_layer)
        self.assertEqual(c.causal_evidence_kinds, ())

    def test_round_trip_preserves_pch_layer(self):
        c = _claim(pch_layer="L2",
                   causal_evidence_kinds=("ab_test", "controlled_experiment"))
        roundtripped = Claim.from_jsonable(c.as_jsonable())
        self.assertEqual(roundtripped.pch_layer, "L2")
        self.assertEqual(
            roundtripped.causal_evidence_kinds,
            ("ab_test", "controlled_experiment"),
        )

    def test_round_trip_preserves_none(self):
        c = _claim()
        roundtripped = Claim.from_jsonable(c.as_jsonable())
        self.assertIsNone(roundtripped.pch_layer)
        self.assertEqual(roundtripped.causal_evidence_kinds, ())

    def test_invalid_pch_layer_rejected(self):
        with self.assertRaises(ValueError):
            _claim(pch_layer="L4")  # type: ignore[arg-type]

    def test_legacy_jsonable_without_pch_fields_loads(self):
        # A row written before migration 035 has neither key. Loader
        # must default to None / empty tuple, not raise KeyError.
        legacy = _claim().as_jsonable()
        legacy.pop("pch_layer")
        legacy.pop("causal_evidence_kinds")
        loaded = Claim.from_jsonable(legacy)
        self.assertIsNone(loaded.pch_layer)
        self.assertEqual(loaded.causal_evidence_kinds, ())


# ============================================================================
# Layer-inference heuristic
# ============================================================================

class TestInferPchLayer(unittest.TestCase):

    def test_observational_default_l1(self):
        self.assertEqual(_infer_pch_layer("the file is missing"), "L1")
        self.assertEqual(_infer_pch_layer("X correlates with Y"), "L1")

    def test_interventional_keywords_yield_l2(self):
        for stmt in (
            "the swap improved latency by 30%",
            "switching backends reduced cost",
            "the change made the build faster",
            "yesterday's experiments produced meaningful improvements",
        ):
            self.assertEqual(_infer_pch_layer(stmt), "L2", f"failed on {stmt!r}")

    def test_counterfactual_keywords_yield_l3(self):
        for stmt in (
            "if we had used the cached path Y would have been lower",
            "had we deployed the patch earlier the incident would have been avoided",
            "counterfactually the win disappears",
        ):
            self.assertEqual(_infer_pch_layer(stmt), "L3", f"failed on {stmt!r}")

    def test_l3_wins_over_l2_when_both_match(self):
        # "would have improved" matches both — L3 is strictly more
        # expressive, so it should win.
        self.assertEqual(
            _infer_pch_layer("the patch would have improved p95"), "L3",
        )


# ============================================================================
# CausalLayerOverreachDetector
# ============================================================================

class TestCausalLayerOverreachDetector(unittest.TestCase):

    def setUp(self):
        self.detector = CausalLayerOverreachDetector()
        self.ledger = Ledger(task_id="task_abc")

    def test_l1_observation_silent(self):
        c = _claim(statement="X correlates with Y in the dataset")
        self.assertEqual(list(self.detector.detect(self.ledger, claim=c)), [])

    def test_l2_inferred_without_evidence_fires(self):
        c = _claim(statement="the swap improved latency by 30%")
        matches = list(self.detector.detect(self.ledger, claim=c))
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].bias_id, "causal_layer_overreach")
        self.assertEqual(matches[0].detail["inferred_layer"], "L2")
        self.assertFalse(matches[0].detail["explicit_layer"])

    def test_l2_with_controlled_experiment_evidence_silent(self):
        c = _claim(
            statement="the swap improved latency by 30%",
            causal_evidence_kinds=("controlled_experiment",),
        )
        self.assertEqual(list(self.detector.detect(self.ledger, claim=c)), [])

    def test_explicit_l2_layer_with_no_evidence_fires(self):
        # An agent that explicitly tags L2 without backing it up gets
        # caught the same way as the heuristic-inferred path.
        c = _claim(
            statement="the run finished",  # would otherwise be L1
            pch_layer="L2",
        )
        matches = list(self.detector.detect(self.ledger, claim=c))
        self.assertEqual(len(matches), 1)
        self.assertTrue(matches[0].detail["explicit_layer"])

    def test_l3_counterfactual_without_evidence_fires(self):
        c = _claim(statement="had we used the cached path latency would have been lower")
        matches = list(self.detector.detect(self.ledger, claim=c))
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].detail["inferred_layer"], "L3")

    def test_only_l2_grade_kinds_count_as_evidence(self):
        # An arbitrary tag in causal_evidence_kinds doesn't grant the
        # exemption — only kinds in CAUSAL_EVIDENCE_KINDS_L2 do.
        c = _claim(
            statement="the swap improved latency",
            causal_evidence_kinds=("vibes",),
        )
        matches = list(self.detector.detect(self.ledger, claim=c))
        self.assertEqual(len(matches), 1)

    def test_all_documented_kinds_grant_exemption(self):
        for kind in CAUSAL_EVIDENCE_KINDS_L2:
            c = _claim(
                statement="the swap improved latency",
                causal_evidence_kinds=(kind,),
            )
            self.assertEqual(
                list(self.detector.detect(self.ledger, claim=c)), [],
                f"kind {kind!r} should grant exemption but didn't",
            )


# ============================================================================
# Narrative emission integration
# ============================================================================

class TestNarrativeEmission(unittest.TestCase):
    """The narrative path ships an explicit L2 claim with controlled-
    experiment evidence — the detector must NOT fire on its output.

    We avoid touching the real ledger by capturing the Claim that gets
    constructed inside ``_emit_l2_narrative_claim`` and running the
    detector on it directly. That is enough to assert the integration
    is well-formed; the detector wiring itself is covered above.
    """

    def setUp(self):
        self.detector = CausalLayerOverreachDetector()
        self.ledger = Ledger(task_id="task_abc")

    def test_emit_with_kept_meaningful_does_not_fire(self):
        # Capture the Claim built by _emit_l2_narrative_claim by
        # patching Ledger.emit. We verify two things on the captured
        # Claim: (1) it is L2-tagged with controlled_experiment
        # evidence, (2) the detector is silent on it.
        from app import improvement_narrative as narr

        data = {
            "experiments": [
                {"status": "keep", "delta": 0.05, "change_type": "prompt",
                 "hypothesis": "shorter system prompt reduces latency"},
                {"status": "keep", "delta": 0.02, "change_type": "model",
                 "hypothesis": "switch to faster model on simple tasks"},
                {"status": "discard", "delta": 0.0001, "change_type": "noop",
                 "hypothesis": "no-op test"},  # ignored
            ],
        }

        captured: list[Claim] = []

        def _capture_emit(self, claim):  # type: ignore[no-redef]
            captured.append(claim)
            return claim

        original = Ledger.emit
        Ledger.emit = _capture_emit  # type: ignore[assignment]
        try:
            narr._emit_l2_narrative_claim(data, "2026-05-02")
        finally:
            Ledger.emit = original  # type: ignore[assignment]

        self.assertEqual(len(captured), 1)
        c = captured[0]
        self.assertEqual(c.pch_layer, "L2")
        self.assertIn("controlled_experiment", c.causal_evidence_kinds)
        self.assertEqual(c.agent_role, "self_improver")
        self.assertEqual(
            list(self.detector.detect(self.ledger, claim=c)), [],
            "well-formed narrative claim should not trigger overreach detector",
        )

    def test_emit_without_kept_meaningful_is_noop(self):
        from app import improvement_narrative as narr

        data = {"experiments": [
            {"status": "discard", "delta": 0.0, "change_type": "x"},
        ]}

        captured: list[Claim] = []
        original = Ledger.emit
        Ledger.emit = lambda self, claim: captured.append(claim) or claim  # type: ignore[assignment]
        try:
            narr._emit_l2_narrative_claim(data, "2026-05-02")
        finally:
            Ledger.emit = original  # type: ignore[assignment]

        self.assertEqual(captured, [])

    def test_emit_creates_and_closes_synthetic_task_row(self):
        # The narrative path uses start_task / complete_task to make
        # the FK target exist before the Claim insert and to stop the
        # row from sitting in 'running' state forever.
        from app import improvement_narrative as narr

        data = {
            "experiments": [
                {"status": "keep", "delta": 0.05, "change_type": "prompt",
                 "hypothesis": "h"},
            ],
        }

        with patch("app.control_plane.crew_tasks.start_task") as start_mock, \
             patch("app.control_plane.crew_tasks.complete_task") as complete_mock:
            original = Ledger.emit
            Ledger.emit = lambda self, claim: claim  # type: ignore[assignment]
            try:
                narr._emit_l2_narrative_claim(data, "2026-05-02")
            finally:
                Ledger.emit = original  # type: ignore[assignment]

            start_mock.assert_called_once()
            kwargs = start_mock.call_args.kwargs
            self.assertEqual(kwargs["task_id"], "narrative_2026-05-02")
            self.assertEqual(kwargs["crew"], "self_improver")
            self.assertIsNone(kwargs["project_id"])

            complete_mock.assert_called_once()
            self.assertEqual(
                complete_mock.call_args.kwargs["task_id"],
                "narrative_2026-05-02",
            )


if __name__ == "__main__":
    unittest.main()
