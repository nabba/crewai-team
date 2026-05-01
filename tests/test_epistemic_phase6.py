"""Tests for Phase 6:

* Pluggable peer-review executor (heuristic default + set_executor)
* request_peer_review (persistence + dispatcher)
* escalate_if_destructive coordinator (no-op when not needed,
  runs when suggested_action == "peer_review")
* calibration.escalate (the calibration → peer_review wiring point)
* Persistence (persist_peer_review, list_recent, aggregates)
* API endpoints (/peer-reviews/stats, /peer-reviews/recent)
"""
from __future__ import annotations

import os
import sys
import types
import unittest
from datetime import datetime, timezone
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
from app.epistemic import peer_review as peer_review_mod  # noqa: E402
from app.epistemic import span_writer  # noqa: E402
from app.epistemic.api import router  # noqa: E402
from app.epistemic.biases import BiasMatch, Severity  # noqa: E402
from app.epistemic.calibration import CalibrationVerdict, escalate  # noqa: E402
from app.epistemic.peer_review import (  # noqa: E402
    EscalationOutcome,
    PeerReviewDecision,
    PeerReviewVerdict,
    escalate_if_destructive,
    heuristic_executor,
    request_peer_review,
    set_executor,
)


def _build_client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _claim(**overrides) -> Claim:
    defaults = dict(
        task_id="task_abc",
        agent_role="researcher",
        statement="example",
        status=VerificationStatus.INFERRED,
        register=Register.DECLARATIVE,
        load_bearing=True,
    )
    defaults.update(overrides)
    return Claim.new(**defaults)


# ============================================================================
# Heuristic executor (default)
# ============================================================================

class TestHeuristicExecutor(unittest.TestCase):

    def test_allows_when_no_unverified_load_bearing(self):
        ledger = Ledger(task_id="task_abc")
        verified = _claim(status=VerificationStatus.VERIFIED)
        ledger._claims[verified.claim_id] = verified
        verdict = heuristic_executor("rm -rf /tmp/foo", ledger)
        self.assertEqual(verdict.decision, PeerReviewDecision.ALLOW)
        self.assertIn("verified", verdict.rationale.lower())
        self.assertEqual(verdict.reviewers, ("heuristic",))

    def test_vetoes_when_unverified_load_bearing_present(self):
        ledger = Ledger(task_id="task_abc")
        unverified = _claim(status=VerificationStatus.INFERRED)
        ledger._claims[unverified.claim_id] = unverified
        verdict = heuristic_executor("rm -rf /tmp/foo", ledger)
        self.assertEqual(verdict.decision, PeerReviewDecision.VETO)
        self.assertIn("unverified", verdict.rationale.lower())

    def test_empty_ledger_allows(self):
        # No claims at all → nothing to be unverified about.
        ledger = Ledger(task_id="task_abc")
        verdict = heuristic_executor("rm -rf /tmp/foo", ledger)
        self.assertEqual(verdict.decision, PeerReviewDecision.ALLOW)

    def test_count_in_rationale_singular(self):
        ledger = Ledger(task_id="task_abc")
        unverified = _claim(status=VerificationStatus.INFERRED)
        ledger._claims[unverified.claim_id] = unverified
        verdict = heuristic_executor("rm -rf /tmp/foo", ledger)
        # 1 claim → "1 load-bearing claim" (no plural s).
        self.assertIn("1 load-bearing claim", verdict.rationale)
        self.assertNotIn("1 load-bearing claims", verdict.rationale)

    def test_count_in_rationale_plural(self):
        ledger = Ledger(task_id="task_abc")
        for _ in range(3):
            c = _claim(status=VerificationStatus.INFERRED)
            ledger._claims[c.claim_id] = c
        verdict = heuristic_executor("rm -rf /tmp/foo", ledger)
        self.assertIn("3 load-bearing claims", verdict.rationale)


# ============================================================================
# Set-executor + dispatcher
# ============================================================================

class TestSetExecutor(unittest.TestCase):

    def setUp(self):
        peer_review_mod._reset_for_tests()

    def tearDown(self):
        peer_review_mod._reset_for_tests()

    def test_default_is_heuristic(self):
        ledger = Ledger(task_id="task_abc")
        with patch.dict(os.environ, {"EPISTEMIC_PEER_REVIEW_LLM": ""}):
            verdict = request_peer_review(
                proposal_text="rm -rf /tmp", ledger=ledger, persist=False,
            )
        self.assertEqual(verdict.reviewers, ("heuristic",))

    def test_set_executor_overrides(self):
        def custom(_proposal: str, _ledger: Ledger) -> PeerReviewVerdict:
            return PeerReviewVerdict(
                decision=PeerReviewDecision.REVISE,
                rationale="custom executor saw this proposal",
                suggested_revision="add a hedge",
                reviewers=("custom",),
                duration_seconds=0.0,
            )

        set_executor(custom)
        ledger = Ledger(task_id="task_abc")
        with patch.dict(os.environ, {"EPISTEMIC_PEER_REVIEW_LLM": ""}):
            verdict = request_peer_review(
                proposal_text="DROP TABLE users",
                ledger=ledger,
                persist=False,
            )
        self.assertEqual(verdict.decision, PeerReviewDecision.REVISE)
        self.assertEqual(verdict.reviewers, ("custom",))


# ============================================================================
# request_peer_review (persistence path)
# ============================================================================

class TestRequestPeerReviewPersistence(unittest.TestCase):

    def setUp(self):
        peer_review_mod._reset_for_tests()

    def tearDown(self):
        peer_review_mod._reset_for_tests()

    def test_persist_calls_span_writer(self):
        ledger = Ledger(task_id="task_abc")
        unverified = _claim(status=VerificationStatus.INFERRED)
        ledger._claims[unverified.claim_id] = unverified
        with patch.dict(os.environ, {"EPISTEMIC_ENABLED": "true"}), \
             patch.object(span_writer, "execute") as mock_exec:
            verdict = request_peer_review(
                proposal_text="rm -rf /var/cache",
                ledger=ledger,
                triggering_claim_id="clm_aa",
                persist=True,
            )
        self.assertEqual(verdict.decision, PeerReviewDecision.VETO)
        mock_exec.assert_called_once()
        sql, params = mock_exec.call_args[0]
        self.assertIn(
            "INSERT INTO control_plane.epistemic_peer_reviews", sql,
        )
        (
            task_id, claim_id, excerpt, decision, rationale,
            suggested, reviewers_json, duration, requested_at,
        ) = params
        self.assertEqual(task_id, "task_abc")
        self.assertEqual(claim_id, "clm_aa")
        self.assertEqual(decision, "veto")
        self.assertIn("rm -rf", excerpt)

    def test_persist_false_skips_db(self):
        ledger = Ledger(task_id="task_abc")
        with patch.dict(os.environ, {"EPISTEMIC_ENABLED": "true"}), \
             patch.object(span_writer, "execute") as mock_exec:
            request_peer_review(
                proposal_text="rm -rf /var", ledger=ledger, persist=False,
            )
        mock_exec.assert_not_called()


# ============================================================================
# escalate_if_destructive coordinator
# ============================================================================

class TestEscalateIfDestructive(unittest.TestCase):

    def setUp(self):
        peer_review_mod._reset_for_tests()

    def tearDown(self):
        peer_review_mod._reset_for_tests()

    def test_no_op_when_action_not_peer_review(self):
        ledger = Ledger(task_id="task_abc")
        outcome = escalate_if_destructive(
            proposal_text="just a thought",
            ledger=ledger,
            suggested_action="hedge",
        )
        self.assertFalse(outcome.escalated)
        self.assertIsNone(outcome.verdict)

    def test_runs_review_when_action_is_peer_review(self):
        ledger = Ledger(task_id="task_abc")
        unverified = _claim(status=VerificationStatus.INFERRED)
        ledger._claims[unverified.claim_id] = unverified
        with patch("app.epistemic.peer_review.persist_peer_review",
                   create=True) as _mock_persist:
            outcome = escalate_if_destructive(
                proposal_text="rm -rf /var",
                ledger=ledger,
                suggested_action="peer_review",
            )
        self.assertTrue(outcome.escalated)
        self.assertEqual(outcome.verdict.decision, PeerReviewDecision.VETO)


# ============================================================================
# calibration.escalate (Phase 6 wiring point)
# ============================================================================

class TestCalibrationEscalate(unittest.TestCase):

    def test_no_escalation_when_action_is_ship(self):
        ledger = Ledger(task_id="task_abc")
        verdict = CalibrationVerdict(proceed=True, suggested_action="ship")
        outcome = escalate(
            proposal_text="all good", ledger=ledger, verdict=verdict,
        )
        self.assertFalse(outcome.escalated)

    def test_escalation_when_action_is_peer_review(self):
        ledger = Ledger(task_id="task_abc")
        unverified = _claim(status=VerificationStatus.INFERRED)
        ledger._claims[unverified.claim_id] = unverified
        verdict = CalibrationVerdict(
            proceed=False,
            suggested_action="peer_review",
            biases_detected=(BiasMatch(
                bias_id="destructive_without_recheck",
                matched_claim_ids=(unverified.claim_id,),
                severity=Severity.CRITICAL,
            ),),
        )
        with patch("app.epistemic.peer_review.persist_peer_review",
                   create=True):
            outcome = escalate(
                proposal_text="rm -rf /var",
                ledger=ledger,
                verdict=verdict,
                triggering_claim_id=unverified.claim_id,
            )
        self.assertTrue(outcome.escalated)
        self.assertEqual(outcome.verdict.decision, PeerReviewDecision.VETO)


# ============================================================================
# Persistence: list_recent_peer_reviews + aggregates
# ============================================================================

class TestPeerReviewPersistence(unittest.TestCase):

    def test_list_empty_when_disabled(self):
        with patch.dict(os.environ, {"EPISTEMIC_ENABLED": ""}):
            self.assertEqual(span_writer.list_recent_peer_reviews(), [])

    def test_list_returns_rows(self):
        rows = [
            {
                "id": 1, "task_id": "task_abc",
                "triggering_claim_id": "clm_aa",
                "proposal_excerpt": "rm -rf /var",
                "decision": "veto",
                "rationale": "diagnosis shaky",
                "suggested_revision": None,
                "reviewers": ["heuristic"],
                "duration_seconds": 0.001,
                "requested_at": datetime(2026, 4, 30, 12, tzinfo=timezone.utc),
            },
        ]
        with patch.dict(os.environ, {"EPISTEMIC_ENABLED": "true"}), \
             patch.object(span_writer, "execute", return_value=rows):
            out = span_writer.list_recent_peer_reviews()
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["decision"], "veto")
        self.assertEqual(out[0]["reviewers"], ["heuristic"])

    def test_aggregates_groups_by_decision(self):
        rows = [
            {"decision": "allow", "n": 4, "mean_seconds": 0.1},
            {"decision": "revise", "n": 2, "mean_seconds": 0.2},
            {"decision": "veto", "n": 3, "mean_seconds": 0.05},
        ]
        with patch.dict(os.environ, {"EPISTEMIC_ENABLED": "true"}), \
             patch.object(span_writer, "execute", return_value=rows):
            agg = span_writer.peer_review_aggregates(window_minutes=60)
        self.assertEqual(agg["total"], 9)
        self.assertEqual(agg["allow"], 4)
        self.assertEqual(agg["revise"], 2)
        self.assertEqual(agg["veto"], 3)
        # Weighted mean: (4*0.1 + 2*0.2 + 3*0.05)/9 = 0.95/9 ≈ 0.106
        self.assertAlmostEqual(agg["mean_seconds"], 0.106, places=2)

    def test_aggregates_empty_when_disabled(self):
        with patch.dict(os.environ, {"EPISTEMIC_ENABLED": ""}):
            agg = span_writer.peer_review_aggregates()
        self.assertEqual(agg["total"], 0)


# ============================================================================
# API endpoints
# ============================================================================

class TestPeerReviewAPI(unittest.TestCase):

    def test_stats_endpoint(self):
        client = _build_client()
        with patch(
            "app.epistemic.api.peer_review_aggregates",
            return_value={
                "window_minutes": 60, "total": 5,
                "allow": 2, "revise": 1, "veto": 2,
                "mean_seconds": 0.15,
            },
        ):
            resp = client.get("/epistemic/peer-reviews/stats?window_min=60")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["total"], 5)
        self.assertEqual(body["veto"], 2)

    def test_recent_endpoint(self):
        client = _build_client()
        reviews = [{
            "id": 1, "task_id": "task_abc",
            "triggering_claim_id": "clm_aa",
            "proposal_excerpt": "rm -rf /var",
            "decision": "veto",
            "rationale": "shaky",
            "suggested_revision": None,
            "reviewers": ["heuristic"],
            "duration_seconds": 0.001,
            "requested_at": "2026-04-30T12:00:00+00:00",
        }]
        with patch(
            "app.epistemic.api.list_recent_peer_reviews",
            return_value=reviews,
        ):
            resp = client.get("/epistemic/peer-reviews/recent?window_min=60")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["count"], 1)
        self.assertEqual(body["reviews"][0]["decision"], "veto")

    def test_window_validation(self):
        client = _build_client()
        resp = client.get("/epistemic/peer-reviews/stats?window_min=0")
        self.assertEqual(resp.status_code, 422)


if __name__ == "__main__":
    unittest.main()
