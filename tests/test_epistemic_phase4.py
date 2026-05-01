"""Tests for Phase 4:

* Post-hoc detectors (defending_periphery, coherence_bias,
  tool_laziness, anomaly_dismissal)
* Post-mortem synthesis (timeline, root-cause classification,
  behavioral-change derivation)
* Self-Improver integration (LearningGap emission with bias_id evidence)
* Persistence (persist_incident, list_recent_incidents, load_incident)
* API endpoints (/epistemic/incidents, /epistemic/incidents/{id})
"""
from __future__ import annotations

import os
import sys
import types
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

# ── Stub heavy/optional deps ─────────────────────────────────────────
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
    Evidence,
    Ledger,
    Register,
    VerificationStatus,
    VerifyingAction,
)
from app.epistemic.api import router  # noqa: E402
from app.epistemic.biases import BiasMatch, Severity  # noqa: E402
from app.epistemic.detectors.posthoc import (  # noqa: E402
    AnomalyDismissalDetector,
    CoherenceBiasDetector,
    DefendingPeripheryDetector,
    ToolLazinessDetector,
)
from app.epistemic.postmortem import (  # noqa: E402
    BehavioralChange,
    IncidentReport,
    TimelineEntry,
    emit_to_self_improver,
    synthesize_report,
)
from app.epistemic import span_writer  # noqa: E402


def _build_client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


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


def _verifier(seconds: float = 0.5) -> VerifyingAction:
    return VerifyingAction(
        tool="readlink",
        args={"path": "/etc/foo"},
        expected_signal="empty=not symlink",
        estimated_seconds=seconds,
    )


# ============================================================================
# DefendingPeripheryDetector
# ============================================================================

class TestDefendingPeripheryDetector(unittest.TestCase):

    def setUp(self):
        self.ledger = Ledger(task_id="task_abc")
        # Seed the foundational claim. Pushback fires AFTER the
        # foundation is emitted, so the foundation itself is NOT in
        # the post-pushback claim list — only subsequent claims are.
        foundation = _claim(
            statement="/etc/foo is not a symlink",
            verifying_action=_verifier(),
        )
        self.ledger._claims[foundation.claim_id] = foundation
        self.foundation = foundation
        self.pushback_at = (
            foundation.created_at + timedelta(microseconds=1)
        ).isoformat()

    def _add_claims_after_pushback(self, n: int) -> list[Claim]:
        # Use distinct future timestamps via Claim direct construction.
        added: list[Claim] = []
        future = datetime.now(timezone.utc) + timedelta(seconds=10)
        for i in range(n):
            from dataclasses import replace
            c = _claim(statement=f"investigation step {i}")
            c = replace(c, created_at=future + timedelta(seconds=i))
            self.ledger._claims[c.claim_id] = c
            added.append(c)
        return added

    def test_fires_on_unverifiable_with_3_subsequent(self):
        self._add_claims_after_pushback(3)
        det = DefendingPeripheryDetector(pushback_events=[{
            "outcome": "unverifiable",
            "contradicted_claim_id": self.foundation.claim_id,
            "detected_at": self.pushback_at,
        }])
        matches = list(det.detect(self.ledger))
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].bias_id, "defending_periphery")
        self.assertEqual(matches[0].severity, Severity.HIGH)
        self.assertEqual(matches[0].detail["pushback_outcome"], "unverifiable")
        self.assertEqual(matches[0].detail["post_pushback_claim_count"], 3)

    def test_does_not_fire_on_reverified_outcome(self):
        self._add_claims_after_pushback(5)
        det = DefendingPeripheryDetector(pushback_events=[{
            "outcome": "reverified",  # different outcome
            "contradicted_claim_id": self.foundation.claim_id,
            "detected_at": self.pushback_at,
        }])
        self.assertEqual(list(det.detect(self.ledger)), [])

    def test_does_not_fire_when_fewer_than_threshold_subsequent(self):
        self._add_claims_after_pushback(2)  # only 2 — below the 3-claim threshold
        det = DefendingPeripheryDetector(pushback_events=[{
            "outcome": "unverifiable",
            "contradicted_claim_id": self.foundation.claim_id,
            "detected_at": self.pushback_at,
        }])
        self.assertEqual(list(det.detect(self.ledger)), [])

    def test_with_events_returns_fresh_detector(self):
        det1 = DefendingPeripheryDetector()
        det2 = det1.with_events([{
            "outcome": "unverifiable",
            "contradicted_claim_id": "x",
            "detected_at": "2026-01-01T00:00:00+00:00",
        }])
        self.assertIsNot(det1, det2)
        self.assertEqual(det1._pushback_events, [])
        self.assertEqual(len(det2._pushback_events), 1)

    def test_realtime_path_no_op(self):
        # When called with a specific claim (realtime path), the
        # post-hoc detector must yield nothing.
        det = DefendingPeripheryDetector(pushback_events=[{
            "outcome": "unverifiable",
            "contradicted_claim_id": "x",
            "detected_at": "2026-01-01T00:00:00+00:00",
        }])
        self.assertEqual(list(det.detect(self.ledger, claim=self.foundation)), [])


# ============================================================================
# CoherenceBiasDetector
# ============================================================================

class TestCoherenceBiasDetector(unittest.TestCase):

    def setUp(self):
        self.detector = CoherenceBiasDetector()
        self.ledger = Ledger(task_id="task_abc")

    def _seed_chain(self, length: int, terminal_register: Register, terminal_load_bearing: bool):
        """Build a chain c0 ← c1 ← c2 ← ... where each cN's evidence
        references c(N-1) as a prior_claim.

        Returns the list of claims in chain order (c0 first, terminal last).
        """
        chain: list[Claim] = []
        for i in range(length):
            evidence: tuple[Evidence, ...] = ()
            if i > 0:
                prev = chain[-1]
                evidence = (Evidence(
                    kind="prior_claim",
                    source_ref=prev.claim_id,
                    excerpt=f"depends on {prev.claim_id}",
                    confidence=0.6,
                ),)
            is_terminal = i == length - 1
            c = _claim(
                statement=f"chain step {i}",
                status=VerificationStatus.INFERRED,
                register=terminal_register if is_terminal else Register.INTERNAL,
                load_bearing=terminal_load_bearing if is_terminal else False,
                evidence=evidence,
            )
            self.ledger._claims[c.claim_id] = c
            chain.append(c)
        return chain

    def test_fires_on_chain_of_3_inferred_terminating_at_declarative_load_bearing(self):
        self._seed_chain(3, Register.DECLARATIVE, True)
        matches = list(self.detector.detect(self.ledger))
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].bias_id, "coherence_bias")
        self.assertEqual(matches[0].detail["chain_length"], 3)

    def test_does_not_fire_on_chain_below_threshold(self):
        self._seed_chain(2, Register.DECLARATIVE, True)
        self.assertEqual(list(self.detector.detect(self.ledger)), [])

    def test_does_not_fire_when_terminal_not_load_bearing(self):
        self._seed_chain(3, Register.DECLARATIVE, False)
        self.assertEqual(list(self.detector.detect(self.ledger)), [])

    def test_does_not_fire_when_terminal_hedged(self):
        self._seed_chain(3, Register.HEDGED, True)
        self.assertEqual(list(self.detector.detect(self.ledger)), [])

    def test_does_not_fire_when_chain_has_verified_member(self):
        # Chain of 3 INFERRED → DECLARATIVE+load_bearing terminal.
        # But change one mid-chain claim to VERIFIED to break the all-inferred condition.
        chain = self._seed_chain(3, Register.DECLARATIVE, True)
        from dataclasses import replace
        mid = chain[1]
        verified = replace(mid, status=VerificationStatus.VERIFIED)
        self.ledger._claims[mid.claim_id] = verified
        self.assertEqual(list(self.detector.detect(self.ledger)), [])


# ============================================================================
# ToolLazinessDetector
# ============================================================================

class TestToolLazinessDetector(unittest.TestCase):

    def setUp(self):
        self.detector = ToolLazinessDetector()
        self.ledger = Ledger(task_id="task_abc")

    def _seed_with_evidence(self, n_evidence: int, **overrides) -> Claim:
        evidence = tuple(
            Evidence(
                kind="tool_call",
                source_ref=f"span:{i}",
                excerpt=f"$ ls\nstep {i}",
                confidence=0.6,
            )
            for i in range(n_evidence)
        )
        defaults = dict(verifying_action=_verifier(0.5), evidence=evidence)
        defaults.update(overrides)
        c = _claim(**defaults)
        self.ledger._claims[c.claim_id] = c
        return c

    def test_fires_on_inferred_load_bearing_with_cheap_verifier_and_3_evidence(self):
        c = self._seed_with_evidence(3)
        matches = list(self.detector.detect(self.ledger))
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].bias_id, "tool_laziness")
        self.assertEqual(matches[0].matched_claim_ids, (c.claim_id,))
        self.assertEqual(matches[0].detail["evidence_count"], 3)
        self.assertAlmostEqual(matches[0].detail["verifier_seconds"], 0.5)

    def test_does_not_fire_when_verifier_expensive(self):
        self._seed_with_evidence(3, verifying_action=_verifier(seconds=10.0))
        self.assertEqual(list(self.detector.detect(self.ledger)), [])

    def test_does_not_fire_when_evidence_below_threshold(self):
        self._seed_with_evidence(2)
        self.assertEqual(list(self.detector.detect(self.ledger)), [])

    def test_does_not_fire_when_no_verifier(self):
        self._seed_with_evidence(3, verifying_action=None)
        self.assertEqual(list(self.detector.detect(self.ledger)), [])

    def test_does_not_fire_when_not_load_bearing(self):
        self._seed_with_evidence(3, load_bearing=False)
        self.assertEqual(list(self.detector.detect(self.ledger)), [])

    def test_does_not_fire_when_already_verified(self):
        self._seed_with_evidence(3, status=VerificationStatus.VERIFIED)
        self.assertEqual(list(self.detector.detect(self.ledger)), [])


# ============================================================================
# AnomalyDismissalDetector
# ============================================================================

class TestAnomalyDismissalDetector(unittest.TestCase):

    def setUp(self):
        self.detector = AnomalyDismissalDetector()
        self.ledger = Ledger(task_id="task_abc")

    def test_fires_when_low_confidence_evidence_present(self):
        c = _claim(
            statement="the deploy is fine",
            evidence=(
                Evidence(
                    kind="tool_call", source_ref="span:1",
                    excerpt="$ check\nlooks ok", confidence=0.9,
                ),
                Evidence(
                    kind="tool_call", source_ref="span:2",
                    excerpt="$ check\nweird outlier", confidence=0.15,  # < 0.30
                ),
            ),
        )
        self.ledger._claims[c.claim_id] = c
        matches = list(self.detector.detect(self.ledger))
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].bias_id, "anomaly_dismissal")
        self.assertEqual(matches[0].detail["contradicting_evidence_count"], 1)

    def test_does_not_fire_for_contradicted_claim(self):
        # The agent ALREADY noted the contradiction (status flipped).
        c = _claim(
            statement="x", status=VerificationStatus.CONTRADICTED,
            evidence=(Evidence(
                kind="tool_call", source_ref="span:1",
                excerpt="$ check\noutlier", confidence=0.1,
            ),),
        )
        self.ledger._claims[c.claim_id] = c
        self.assertEqual(list(self.detector.detect(self.ledger)), [])

    def test_does_not_fire_when_all_evidence_above_threshold(self):
        c = _claim(
            statement="y",
            evidence=(Evidence(
                kind="tool_call", source_ref="span:1",
                excerpt="$ check\nfine", confidence=0.7,
            ),),
        )
        self.ledger._claims[c.claim_id] = c
        self.assertEqual(list(self.detector.detect(self.ledger)), [])


# ============================================================================
# Post-mortem synthesis
# ============================================================================

class TestSynthesizeReport(unittest.TestCase):

    def test_returns_none_when_no_biases_fire(self):
        ledger = Ledger.from_claims(task_id="task_abc", claims=[])
        with patch("app.epistemic.postmortem.load_ledger_for_task", return_value=ledger), \
             patch("app.epistemic.postmortem.list_bias_matches_for_task", return_value=[]), \
             patch("app.epistemic.postmortem.list_pushback_events_for_task", return_value=[]):
            self.assertIsNone(synthesize_report(task_id="task_abc"))

    def test_realtime_match_only_produces_report(self):
        c = _claim(
            statement="/etc/foo is not a symlink",
            verifying_action=_verifier(),
        )
        ledger = Ledger.from_claims(task_id="task_abc", claims=[c])
        realtime_rows = [{
            "id": 1, "task_id": "task_abc", "claim_id": c.claim_id,
            "bias_id": "inference_as_fact", "severity": "high",
            "matched_claim_ids": [c.claim_id],
            "detail": {"verifier_tool": "readlink"},
            "detected_at": datetime.now(timezone.utc).isoformat(),
        }]
        with patch("app.epistemic.postmortem.load_ledger_for_task", return_value=ledger), \
             patch("app.epistemic.postmortem.list_bias_matches_for_task",
                   return_value=realtime_rows), \
             patch("app.epistemic.postmortem.list_pushback_events_for_task",
                   return_value=[]):
            report = synthesize_report(task_id="task_abc")

        self.assertIsNotNone(report)
        self.assertEqual(report.task_id, "task_abc")
        self.assertEqual(report.root_cause.bias_id, "inference_as_fact")
        self.assertEqual(report.severity, Severity.HIGH)
        # Behavioral change derived for inference_as_fact.
        self.assertGreater(len(report.behavioral_changes), 0)
        self.assertEqual(
            report.behavioral_changes[0].kind, "feedback_memory_entry",
        )
        # Timeline includes claim_emit + bias_match.
        kinds = {t.kind for t in report.timeline}
        self.assertIn("claim_emit", kinds)
        self.assertIn("bias_match", kinds)

    def test_critical_severity_is_root_cause(self):
        c = _claim(statement="x")
        ledger = Ledger.from_claims(task_id="task_abc", claims=[c])
        # Two matches: one HIGH, one CRITICAL. CRITICAL must be root.
        now = datetime.now(timezone.utc).isoformat()
        rows = [
            {
                "id": 1, "task_id": "task_abc", "claim_id": c.claim_id,
                "bias_id": "inference_as_fact", "severity": "high",
                "matched_claim_ids": [c.claim_id], "detail": {},
                "detected_at": now,
            },
            {
                "id": 2, "task_id": "task_abc", "claim_id": c.claim_id,
                "bias_id": "destructive_without_recheck", "severity": "critical",
                "matched_claim_ids": [c.claim_id], "detail": {},
                "detected_at": now,
            },
        ]
        with patch("app.epistemic.postmortem.load_ledger_for_task", return_value=ledger), \
             patch("app.epistemic.postmortem.list_bias_matches_for_task",
                   return_value=rows), \
             patch("app.epistemic.postmortem.list_pushback_events_for_task",
                   return_value=[]):
            report = synthesize_report(task_id="task_abc")

        self.assertIsNotNone(report)
        self.assertEqual(report.root_cause.bias_id, "destructive_without_recheck")
        self.assertEqual(report.severity, Severity.CRITICAL)
        self.assertEqual(len(report.enabling_factors), 1)
        self.assertEqual(report.enabling_factors[0].bias_id, "inference_as_fact")

    def test_jsonable_round_trip(self):
        match = BiasMatch(
            bias_id="inference_as_fact",
            matched_claim_ids=("clm_aa",),
            severity=Severity.HIGH,
            detail={},
        )
        report = IncidentReport(
            incident_id="inc_test_001",
            task_id="task_abc",
            timeline=(TimelineEntry(
                at=datetime(2026, 4, 30, 12, tzinfo=timezone.utc),
                kind="claim_emit",
                summary="x",
                claim_id="clm_aa",
            ),),
            root_cause=match,
            behavioral_changes=(BehavioralChange(
                kind="feedback_memory_entry",
                target="agent_register_discipline",
                body="hedge or verify",
            ),),
        )
        as_json = report.as_jsonable()
        self.assertEqual(as_json["incident_id"], "inc_test_001")
        self.assertEqual(as_json["root_cause"]["bias_id"], "inference_as_fact")
        self.assertEqual(as_json["severity"], "high")
        self.assertEqual(len(as_json["behavioral_changes"]), 1)


# ============================================================================
# Self-Improver integration
# ============================================================================

class TestEmitToSelfImprover(unittest.TestCase):

    def _report(self, severity: Severity = Severity.HIGH) -> IncidentReport:
        return IncidentReport(
            incident_id="inc_test_001",
            task_id="task_abc",
            timeline=(),
            root_cause=BiasMatch(
                bias_id="inference_as_fact",
                matched_claim_ids=("clm_aa",),
                severity=severity,
                detail={},
            ),
            behavioral_changes=(BehavioralChange(
                kind="feedback_memory_entry",
                target="agent_register_discipline",
                body="hedge or verify",
            ),),
        )

    def test_calls_emit_gap_with_correct_shape(self):
        from app.self_improvement.types import GapSource
        with patch("app.self_improvement.store.emit_gap", return_value=True) as mock_emit:
            ok = emit_to_self_improver(self._report())

        self.assertTrue(ok)
        mock_emit.assert_called_once()
        gap = mock_emit.call_args[0][0]
        self.assertEqual(gap.source, GapSource.LOW_CONFIDENCE)
        self.assertIn("inference_as_fact", gap.description.lower()
                      .replace(" ", "_")
                      + "/inference_as_fact")  # tolerate name → id formatting
        self.assertEqual(gap.evidence["bias_id"], "inference_as_fact")
        self.assertEqual(gap.evidence["incident_id"], "inc_test_001")
        self.assertEqual(gap.evidence["severity"], "high")
        # Signal strength scales with severity.
        self.assertAlmostEqual(gap.signal_strength, 0.70)

    def test_critical_severity_higher_strength(self):
        with patch("app.self_improvement.store.emit_gap", return_value=True) as mock_emit:
            emit_to_self_improver(self._report(severity=Severity.CRITICAL))
        gap = mock_emit.call_args[0][0]
        self.assertAlmostEqual(gap.signal_strength, 0.90)

    def test_returns_false_on_emit_gap_failure(self):
        with patch("app.self_improvement.store.emit_gap",
                   side_effect=RuntimeError("DB down")):
            ok = emit_to_self_improver(self._report())
        self.assertFalse(ok)

    def test_returns_false_when_emit_gap_returns_falsy(self):
        with patch("app.self_improvement.store.emit_gap", return_value=False):
            ok = emit_to_self_improver(self._report())
        self.assertFalse(ok)


# ============================================================================
# Persistence
# ============================================================================

class TestPersistIncident(unittest.TestCase):

    def _report(self) -> IncidentReport:
        return IncidentReport(
            incident_id="inc_aa_001",
            task_id="task_abc",
            timeline=(),
            root_cause=BiasMatch(
                bias_id="inference_as_fact",
                matched_claim_ids=("clm_aa",),
                severity=Severity.HIGH,
                detail={},
            ),
        )

    def test_no_op_when_disabled(self):
        with patch.dict(os.environ, {"EPISTEMIC_ENABLED": ""}), \
             patch.object(span_writer, "execute") as mock_exec:
            ok = span_writer.persist_incident(self._report())
            self.assertFalse(ok)
            mock_exec.assert_not_called()

    def test_insert_when_enabled(self):
        with patch.dict(os.environ, {"EPISTEMIC_ENABLED": "true"}), \
             patch.object(span_writer, "execute") as mock_exec:
            ok = span_writer.persist_incident(self._report())
            self.assertTrue(ok)
            mock_exec.assert_called_once()
            sql, params = mock_exec.call_args[0]
            self.assertIn("INSERT INTO control_plane.epistemic_incidents", sql)
            self.assertIn("ON CONFLICT (incident_id) DO NOTHING", sql)
            (incident_id, task_id, root_cause, severity,
             report_json, emitted, created_at) = params
            self.assertEqual(incident_id, "inc_aa_001")
            self.assertEqual(task_id, "task_abc")
            self.assertEqual(root_cause, "inference_as_fact")
            self.assertEqual(severity, "high")
            self.assertFalse(emitted)

    def test_db_error_returns_false(self):
        with patch.dict(os.environ, {"EPISTEMIC_ENABLED": "true"}), \
             patch.object(
                 span_writer, "execute",
                 side_effect=RuntimeError("connection refused"),
             ):
            ok = span_writer.persist_incident(self._report())
            self.assertFalse(ok)


class TestListAndLoadIncidents(unittest.TestCase):

    def test_list_empty_when_disabled(self):
        with patch.dict(os.environ, {"EPISTEMIC_ENABLED": ""}):
            self.assertEqual(span_writer.list_recent_incidents(), [])

    def test_list_returns_summaries(self):
        rows = [
            {
                "incident_id": "inc_aa", "task_id": "task_abc",
                "root_cause_bias_id": "inference_as_fact",
                "severity": "high", "self_improver_emitted": True,
                "created_at": datetime(2026, 4, 30, 12, tzinfo=timezone.utc),
            },
        ]
        with patch.dict(os.environ, {"EPISTEMIC_ENABLED": "true"}), \
             patch.object(span_writer, "execute", return_value=rows):
            out = span_writer.list_recent_incidents()
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["incident_id"], "inc_aa")
        self.assertTrue(out[0]["self_improver_emitted"])

    def test_load_returns_full_report_with_emitted_flag(self):
        row = {
            "incident_id": "inc_aa", "task_id": "task_abc",
            "root_cause_bias_id": "inference_as_fact",
            "severity": "high", "self_improver_emitted": True,
            "created_at": datetime(2026, 4, 30, 12, tzinfo=timezone.utc),
            "report": {
                "incident_id": "inc_aa",
                "timeline": [],
                "root_cause": {"bias_id": "inference_as_fact"},
            },
        }
        with patch.dict(os.environ, {"EPISTEMIC_ENABLED": "true"}), \
             patch.object(span_writer, "execute_one", return_value=row):
            out = span_writer.load_incident("inc_aa")
        self.assertIsNotNone(out)
        self.assertEqual(out["incident_id"], "inc_aa")
        # emitted flag merged into the JSON for the React panel.
        self.assertTrue(out["self_improver_emitted"])

    def test_load_missing_returns_none(self):
        with patch.dict(os.environ, {"EPISTEMIC_ENABLED": "true"}), \
             patch.object(span_writer, "execute_one", return_value=None):
            self.assertIsNone(span_writer.load_incident("inc_missing"))


# ============================================================================
# API endpoints
# ============================================================================

class TestIncidentsAPI(unittest.TestCase):

    def test_incidents_list(self):
        client = _build_client()
        with patch(
            "app.epistemic.api.list_recent_incidents",
            return_value=[{
                "incident_id": "inc_aa", "task_id": "task_abc",
                "root_cause_bias_id": "inference_as_fact",
                "severity": "high", "self_improver_emitted": False,
                "created_at": "2026-04-30T12:00:00+00:00",
            }],
        ):
            resp = client.get("/epistemic/incidents")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["count"], 1)
        self.assertEqual(body["incidents"][0]["incident_id"], "inc_aa")

    def test_incident_detail(self):
        client = _build_client()
        with patch(
            "app.epistemic.api.load_incident",
            return_value={
                "incident_id": "inc_aa",
                "timeline": [],
                "root_cause": {"bias_id": "inference_as_fact"},
                "self_improver_emitted": True,
            },
        ):
            resp = client.get("/epistemic/incidents/inc_aa")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["incident_id"], "inc_aa")
        self.assertTrue(body["self_improver_emitted"])

    def test_incident_detail_404(self):
        client = _build_client()
        with patch("app.epistemic.api.load_incident", return_value=None):
            resp = client.get("/epistemic/incidents/inc_missing")
        self.assertEqual(resp.status_code, 404)


if __name__ == "__main__":
    unittest.main()
