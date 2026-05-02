"""Tests for Phase 2 additions:

* Bias library YAML loader (with safe in-code fallback)
* RegisterConfidenceMismatchDetector (degradable grounding integration)
* DestructiveWithoutRecheckDetector
* RecommendationWithoutMeasurementDetector
* Path 3: Ledger.emit_from_output_text + extraction.regex_extractor
* Reference panel: load_panel + replay_panel
"""
from __future__ import annotations

import os
import sys
import textwrap
import types
import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile
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
from app.epistemic import biases as biases_mod  # noqa: E402
from app.epistemic import grounding as grounding_mod  # noqa: E402
from app.epistemic.biases import (  # noqa: E402
    BIAS_LIBRARY,
    BiasLibrary,
    BiasLibraryLoadError,
    DetectorPhase,
    Severity,
    _load_yaml,
    _reload_for_tests,
)
from app.epistemic.detectors.realtime import (  # noqa: E402
    DestructiveWithoutRecheckDetector,
    InferenceAsFactDetector,
    RecommendationWithoutMeasurementDetector,
    RegisterConfidenceMismatchDetector,
)
from app.epistemic.extraction import (  # noqa: E402
    CAP_PER_OUTPUT,
    ExtractedClaim,
    extract_claims,
    regex_extractor,
)
from app.epistemic.grounding import (  # noqa: E402
    factual_grounding,
    set_grounding_provider,
)
from app.epistemic.reference_panel import (  # noqa: E402
    ReferencePanelLoadError,
    load_panel,
    replay_one,
    replay_panel,
)
from app.epistemic.registry import _reset_for_tests as _reset_hooks  # noqa: E402


def _yaml_path(content: str) -> Path:
    f = NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    f.write(textwrap.dedent(content))
    f.close()
    return Path(f.name)


def _claim(**overrides) -> Claim:
    defaults = dict(
        task_id="task_abc",
        agent_role="researcher",
        statement="example claim",
        status=VerificationStatus.INFERRED,
        register=Register.DECLARATIVE,
        load_bearing=True,
    )
    defaults.update(overrides)
    return Claim.new(**defaults)


# ============================================================================
# BiasLibrary YAML loader
# ============================================================================

class TestBiasLibraryYAMLLoader(unittest.TestCase):

    def test_default_library_has_all_phase_2_biases(self):
        # The shipped YAML must define all biases the detectors consume.
        ids = {d.id for d in BIAS_LIBRARY.all()}
        for required in (
            "inference_as_fact",
            "register_confidence_mismatch",
            "destructive_without_recheck",
            "recommendation_without_measurement",
            "causal_layer_overreach",
            "defending_periphery",
            "coherence_bias",
            "tool_laziness",
            "anomaly_dismissal",
        ):
            self.assertIn(required, ids, f"bias {required!r} missing from library")

    def test_realtime_phase_filter(self):
        realtime_ids = {d.id for d in BIAS_LIBRARY.all(phase=DetectorPhase.REALTIME)}
        self.assertEqual(realtime_ids, {
            "inference_as_fact",
            "register_confidence_mismatch",
            "destructive_without_recheck",
            "recommendation_without_measurement",
            "causal_layer_overreach",
        })

    def test_critical_severity_blocks_by_default(self):
        # destructive_without_recheck is the one critical bias; YAML
        # marks it blocking=true (Phase 7 default), but the detector
        # only fires the bias — the calibration gate decides what to
        # do based on EPISTEMIC_CALIBRATION_BLOCKS_OUTPUT.
        d = BIAS_LIBRARY.get("destructive_without_recheck")
        self.assertEqual(d.severity, Severity.CRITICAL)
        self.assertTrue(d.blocking)

    def test_yaml_structural_errors_rejected(self):
        path = _yaml_path("not_biases: []")
        with self.assertRaises(BiasLibraryLoadError):
            _load_yaml(path)

    def test_invalid_severity_rejected(self):
        path = _yaml_path("""
            biases:
              - id: bad
                name: B
                description: x
                severity: WAT
                detector: realtime
        """)
        with self.assertRaises(BiasLibraryLoadError):
            _load_yaml(path)

    def test_invalid_phase_rejected(self):
        path = _yaml_path("""
            biases:
              - id: bad
                name: B
                description: x
                severity: medium
                detector: yesterday
        """)
        with self.assertRaises(BiasLibraryLoadError):
            _load_yaml(path)

    def test_duplicate_id_rejected(self):
        path = _yaml_path("""
            biases:
              - id: dup
                name: A
                description: x
                severity: low
                detector: realtime
              - id: dup
                name: B
                description: y
                severity: high
                detector: posthoc
        """)
        with self.assertRaises(BiasLibraryLoadError):
            _load_yaml(path)

    def test_fallback_to_in_code_starter_on_yaml_error(self):
        """If the YAML load fails, the library still has the
        canonical inference_as_fact entry (in-code fallback)."""
        # Patch the path to a non-existent file and reload.
        with patch.object(biases_mod, "_DEFAULT_PATH",
                          Path("/nonexistent/biases.yaml")):
            old = BIAS_LIBRARY
            try:
                _reload_for_tests()  # uses the patched path internally
                # Even with the missing file, inference_as_fact is present.
                self.assertIn("inference_as_fact", BIAS_LIBRARY)
            finally:
                # Restore the real library so subsequent tests aren't broken.
                biases_mod.BIAS_LIBRARY = old


# ============================================================================
# Grounding provider
# ============================================================================

class TestGroundingProvider(unittest.TestCase):

    def setUp(self):
        grounding_mod._reset_for_tests()

    def tearDown(self):
        grounding_mod._reset_for_tests()

    def test_default_returns_none(self):
        self.assertIsNone(factual_grounding())

    def test_set_provider_overrides(self):
        set_grounding_provider(lambda: 0.42)
        self.assertAlmostEqual(factual_grounding(), 0.42)

    def test_provider_exception_swallowed_returns_none(self):
        def boom() -> float:
            raise RuntimeError("provider broke")
        set_grounding_provider(boom)
        # Must not propagate — returning None means "unavailable".
        self.assertIsNone(factual_grounding())


# ============================================================================
# RegisterConfidenceMismatchDetector
# ============================================================================

class TestRegisterConfidenceMismatchDetector(unittest.TestCase):

    def setUp(self):
        grounding_mod._reset_for_tests()
        self.detector = RegisterConfidenceMismatchDetector()
        self.ledger = Ledger(task_id="task_abc")

    def tearDown(self):
        grounding_mod._reset_for_tests()

    def test_fires_when_grounding_below_threshold(self):
        set_grounding_provider(lambda: 0.20)
        c = _claim()
        matches = list(self.detector.detect(self.ledger, claim=c))
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].bias_id, "register_confidence_mismatch")

    def test_does_not_fire_when_grounding_high(self):
        set_grounding_provider(lambda: 0.90)
        c = _claim()
        self.assertEqual(list(self.detector.detect(self.ledger, claim=c)), [])

    def test_does_not_fire_when_grounding_unavailable(self):
        # Default provider returns None — bias must not fire.
        c = _claim()
        self.assertEqual(list(self.detector.detect(self.ledger, claim=c)), [])

    def test_does_not_fire_for_non_load_bearing(self):
        set_grounding_provider(lambda: 0.10)
        c = _claim(load_bearing=False)
        self.assertEqual(list(self.detector.detect(self.ledger, claim=c)), [])

    def test_does_not_fire_for_hedged_register(self):
        set_grounding_provider(lambda: 0.10)
        c = _claim(register=Register.HEDGED)
        self.assertEqual(list(self.detector.detect(self.ledger, claim=c)), [])


# ============================================================================
# DestructiveWithoutRecheckDetector
# ============================================================================

class TestDestructiveWithoutRecheckDetector(unittest.TestCase):

    def setUp(self):
        self.detector = DestructiveWithoutRecheckDetector()
        self.ledger = Ledger(task_id="task_abc")

    def _seed_unverified_load_bearing(self) -> Claim:
        c = _claim(
            statement="some load-bearing assumption",
            status=VerificationStatus.INFERRED,
            register=Register.INTERNAL,
            load_bearing=True,
        )
        self.ledger._claims[c.claim_id] = c
        return c

    def test_fires_on_rm_rf_with_unverified(self):
        seed = self._seed_unverified_load_bearing()
        trigger = _claim(statement="we should rm -rf /var/cache to free disk")
        matches = list(self.detector.detect(self.ledger, claim=trigger))
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].bias_id, "destructive_without_recheck")
        # Match-id list ends with the destructive trigger.
        self.assertEqual(matches[0].matched_claim_ids[-1], trigger.claim_id)
        self.assertIn(seed.claim_id, matches[0].matched_claim_ids)

    def test_fires_on_drop_table_with_unverified(self):
        self._seed_unverified_load_bearing()
        trigger = _claim(statement="DROP TABLE legacy_users to clean up")
        matches = list(self.detector.detect(self.ledger, claim=trigger))
        self.assertEqual(len(matches), 1)

    def test_fires_on_force_push_with_unverified(self):
        self._seed_unverified_load_bearing()
        trigger = _claim(statement="we'll force-push to overwrite the bad commit")
        matches = list(self.detector.detect(self.ledger, claim=trigger))
        self.assertEqual(len(matches), 1)

    def test_does_not_fire_when_no_unverified(self):
        # All siblings VERIFIED — agent did diligence.
        verified = _claim(status=VerificationStatus.VERIFIED, register=Register.INTERNAL)
        self.ledger._claims[verified.claim_id] = verified
        trigger = _claim(statement="rm -rf /var/cache")
        self.assertEqual(list(self.detector.detect(self.ledger, claim=trigger)), [])

    def test_does_not_fire_for_non_destructive(self):
        self._seed_unverified_load_bearing()
        trigger = _claim(statement="we could clean up the cache directory someday")
        self.assertEqual(list(self.detector.detect(self.ledger, claim=trigger)), [])

    def test_explicit_tag_overrides_pattern(self):
        # Custom delete tool with an unusual statement; the agent
        # opts in via the tag.
        self._seed_unverified_load_bearing()
        trigger = _claim(
            statement="invoke the migrate-and-vacuum tool",
            tags=("destructive_recommendation",),
        )
        matches = list(self.detector.detect(self.ledger, claim=trigger))
        self.assertEqual(len(matches), 1)


# ============================================================================
# RecommendationWithoutMeasurementDetector
# ============================================================================

class TestRecommendationWithoutMeasurementDetector(unittest.TestCase):

    def setUp(self):
        self.detector = RecommendationWithoutMeasurementDetector()
        self.ledger = Ledger(task_id="task_abc")

    def test_fires_on_unmeasured_optimization(self):
        c = _claim(statement="we should switch to model X to reduce latency")
        matches = list(self.detector.detect(self.ledger, claim=c))
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].bias_id, "recommendation_without_measurement")

    def test_does_not_fire_with_benchmark_evidence(self):
        c = _claim(
            statement="we should switch to model X to reduce latency",
            evidence=(Evidence(
                kind="tool_call", source_ref="span:42",
                excerpt="$ benchmark model_x\np50=120ms p95=200ms",
                confidence=0.9,
            ),),
        )
        self.assertEqual(list(self.detector.detect(self.ledger, claim=c)), [])

    def test_does_not_fire_with_psql_evidence(self):
        c = _claim(
            statement="we should index this column to speed up queries",
            evidence=(Evidence(
                kind="tool_call", source_ref="span:42",
                excerpt="$ psql -c\nSELECT count(*) -- 4M rows",
                confidence=0.9,
            ),),
        )
        self.assertEqual(list(self.detector.detect(self.ledger, claim=c)), [])

    def test_does_not_fire_for_non_recommendation(self):
        c = _claim(statement="the system handles this case correctly")
        self.assertEqual(list(self.detector.detect(self.ledger, claim=c)), [])

    def test_explicit_tag_triggers_check(self):
        # Even without a recognized recommendation pattern, the
        # explicit tag opts the claim in.
        c = _claim(
            statement="propose moving to a different scheduler",
            tags=("optimization_recommendation",),
        )
        matches = list(self.detector.detect(self.ledger, claim=c))
        self.assertEqual(len(matches), 1)


# ============================================================================
# Path 3: extraction
# ============================================================================

class TestRegexExtractor(unittest.TestCase):

    def test_simple_is_pattern(self):
        text = "The deploy is complete. The cache is warm."
        out = regex_extractor(text)
        statements = {c.statement for c in out}
        # Both sentences should be captured (substantive subjects).
        self.assertTrue(any("deploy is complete" in s for s in statements))
        self.assertTrue(any("cache is warm" in s for s in statements))

    def test_empty_text_returns_empty(self):
        self.assertEqual(regex_extractor(""), [])
        self.assertEqual(regex_extractor("   "), [])

    def test_caps_at_max_per_output(self):
        # 12 distinct claims — should be capped to CAP_PER_OUTPUT.
        text = ". ".join(f"Item{i} is correct" for i in range(12)) + "."
        out = regex_extractor(text)
        self.assertEqual(len(out), CAP_PER_OUTPUT)

    def test_skips_pronoun_subjects(self):
        # "It is X" / "This is Y" lack a substantive subject.
        text = "It is true. This is fine. The build is green."
        out = regex_extractor(text)
        statements = {c.statement for c in out}
        for s in statements:
            self.assertNotIn(s.lower().split()[0], {"it", "this", "that", "there"})

    def test_status_is_inferred(self):
        text = "The path is missing."
        out = regex_extractor(text)
        for c in out:
            self.assertEqual(c.status, VerificationStatus.INFERRED)

    def test_dispatcher_uses_regex_by_default(self):
        with patch.dict(os.environ, {"EPISTEMIC_PATH3_LLM_EXTRACTION": ""}):
            out = extract_claims("The cache is warm.")
        self.assertTrue(any("cache is warm" in c.statement for c in out))


class TestEmitFromOutputText(unittest.TestCase):

    def setUp(self):
        _reset_hooks()
        self.ledger = Ledger(task_id="task_abc")

    def test_emits_one_claim_per_extracted(self):
        text = "The deploy is complete. The /etc/foo is not a symlink."
        emitted = self.ledger.emit_from_output_text(
            agent_role="researcher",
            output_text=text,
            register=Register.DECLARATIVE,
            load_bearing=True,
        )
        self.assertGreaterEqual(len(emitted), 2)
        for c in emitted:
            self.assertEqual(c.task_id, "task_abc")
            self.assertEqual(c.agent_role, "researcher")
            self.assertEqual(c.register, Register.DECLARATIVE)
            self.assertTrue(c.load_bearing)

    def test_attaches_verifier_when_registry_matches(self):
        from app.epistemic.verification import _reset_for_tests as _reset_registry
        _reset_registry()
        text = "The /etc/foo is not a symlink."
        emitted = self.ledger.emit_from_output_text(
            agent_role="researcher",
            output_text=text,
            register=Register.DECLARATIVE,
            load_bearing=True,
        )
        self.assertEqual(len(emitted), 1)
        self.assertIsNotNone(emitted[0].verifying_action)
        self.assertEqual(emitted[0].verifying_action.tool, "readlink")

    def test_empty_text_emits_nothing(self):
        emitted = self.ledger.emit_from_output_text(
            agent_role="researcher", output_text="",
        )
        self.assertEqual(emitted, [])

    def test_evidence_kind_is_model_inference(self):
        emitted = self.ledger.emit_from_output_text(
            agent_role="researcher",
            output_text="The cache is warm.",
        )
        if emitted:
            self.assertEqual(emitted[0].evidence[0].kind, "model_inference")


# ============================================================================
# Reference panel
# ============================================================================

class TestReferencePanel(unittest.TestCase):

    def setUp(self):
        # Make sure the realtime detectors and meta-hook are registered.
        # (In Phase 1's test for the same, we re-register manually.)
        from app.epistemic.detectors import _reset_for_tests as _reset_detectors
        _reset_detectors()
        _reset_hooks()
        from app.epistemic.detectors.realtime import (
            CAUSAL_LAYER_OVERREACH,
            DESTRUCTIVE_WITHOUT_RECHECK,
            INFERENCE_AS_FACT,
            RECOMMENDATION_WITHOUT_MEASUREMENT,
            REGISTER_CONFIDENCE_MISMATCH,
            _realtime_meta_hook,
        )
        from app.epistemic.detectors import register_realtime
        from app.epistemic.registry import register as register_claim_hook
        register_realtime(INFERENCE_AS_FACT)
        register_realtime(REGISTER_CONFIDENCE_MISMATCH)
        register_realtime(DESTRUCTIVE_WITHOUT_RECHECK)
        register_realtime(RECOMMENDATION_WITHOUT_MEASUREMENT)
        register_realtime(CAUSAL_LAYER_OVERREACH)
        register_claim_hook(_realtime_meta_hook)

    def test_load_panel_shipped_yaml_has_scenarios(self):
        scenarios = load_panel()
        self.assertGreater(len(scenarios), 8)  # Phase 2 ships 12.
        for s in scenarios:
            self.assertIn("id", s)
            self.assertIn("trigger", s)

    def test_load_panel_rejects_missing_top_level_key(self):
        path = _yaml_path("not_scenarios: []")
        with self.assertRaises(ReferencePanelLoadError):
            load_panel(path)

    def test_load_panel_rejects_duplicate_id(self):
        path = _yaml_path("""
            scenarios:
              - id: dup
                trigger:
                  statement: x
                  status: inferred
                  register: declarative
              - id: dup
                trigger:
                  statement: y
                  status: inferred
                  register: declarative
        """)
        with self.assertRaises(ReferencePanelLoadError):
            load_panel(path)

    def test_replay_panel_all_pass(self):
        """The shipped reference panel must be 100% passing.

        A regression here means the bias library has drifted from its
        canonical scenarios — block promotion until reconciled.
        """
        report = replay_panel()
        self.assertTrue(
            report.all_passed,
            f"reference panel regressions: {[(r.scenario_id, r.diff) for r in report.results if not r.passed]}",
        )
        self.assertGreater(report.total, 0)

    def test_replay_one_canonical_inference_as_fact(self):
        scenarios = load_panel()
        canonical = next(s for s in scenarios if s["id"] == "inference_as_fact_canonical")
        result = replay_one(canonical)
        self.assertTrue(result.passed, result.diff)
        self.assertIn("inference_as_fact", result.actual_bias_ids)


if __name__ == "__main__":
    unittest.main()
