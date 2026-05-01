"""Tests for Phase 3:

* Pluggable verifier executor (app.epistemic.verifier_executor)
* Contradiction detection (regex classifier)
* Foundation re-check protocol (REVERIFIED / FALSIFIED / UNVERIFIABLE)
* Cascade-invalidation of dependent claims on FALSIFIED
* Persistence (persist_pushback_event, list_recent, aggregates)
* API endpoints (/epistemic/pushback/stats, /pushback/recent)
"""
from __future__ import annotations

import os
import sys
import types
import unittest
from datetime import datetime, timezone
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
from app.epistemic import verifier_executor as executor_mod  # noqa: E402
from app.epistemic.api import router  # noqa: E402
from app.epistemic.pushback import (  # noqa: E402
    ContradictionSignal,
    FoundationOutcome,
    detect_contradiction,
    handle_foundation_check,
    process_user_message,
    regex_detect_contradiction,
)
from app.epistemic.registry import _reset_for_tests as _reset_hooks  # noqa: E402
from app.epistemic.verifier_executor import (  # noqa: E402
    VerifierResult,
    execute,
    set_executor,
)
from app.epistemic import span_writer  # noqa: E402


def _build_client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _verifier(tool: str = "readlink") -> VerifyingAction:
    return VerifyingAction(
        tool=tool,
        args={"path": "/etc/foo"},
        expected_signal="empty=not symlink",
        estimated_seconds=0.5,
    )


def _make_claim(
    *,
    statement: str,
    status: VerificationStatus = VerificationStatus.INFERRED,
    register: Register = Register.DECLARATIVE,
    load_bearing: bool = True,
    verifier: VerifyingAction | None = None,
    evidence: tuple[Evidence, ...] = (),
) -> Claim:
    return Claim.new(
        task_id="task_abc",
        agent_role="researcher",
        statement=statement,
        status=status,
        register=register,
        load_bearing=load_bearing,
        verifying_action=verifier,
        evidence=evidence,
    )


# ============================================================================
# Verifier executor
# ============================================================================

class TestVerifierExecutor(unittest.TestCase):

    def setUp(self):
        executor_mod._reset_for_tests()

    def tearDown(self):
        executor_mod._reset_for_tests()

    def test_default_returns_settles_false(self):
        result = execute(_verifier())
        self.assertFalse(result.settles)
        self.assertFalse(result.confirms)

    def test_set_executor_overrides(self):
        set_executor(lambda action: VerifierResult(
            settles=True, confirms=True, stdout="readlink ran",
        ))
        result = execute(_verifier())
        self.assertTrue(result.settles)
        self.assertTrue(result.confirms)
        self.assertEqual(result.stdout, "readlink ran")

    def test_executor_exception_swallowed(self):
        def boom(_action: VerifyingAction) -> VerifierResult:
            raise RuntimeError("subprocess died")
        set_executor(boom)
        result = execute(_verifier())
        # Must not propagate.
        self.assertFalse(result.settles)
        self.assertIn("executor raised", result.stderr)


# ============================================================================
# Regex contradiction detector
# ============================================================================

class TestRegexDetectContradiction(unittest.TestCase):

    def setUp(self):
        _reset_hooks()
        self.ledger = Ledger(task_id="task_abc")

    def test_no_load_bearing_claims_no_signal(self):
        # Even with explicit pushback phrasing, no candidate claims
        # means we have nothing to re-check.
        sig = regex_detect_contradiction("no, that's wrong", self.ledger)
        self.assertIsNone(sig)

    def test_explicit_phrase_with_overlap_fires(self):
        c = _make_claim(
            statement="/etc/foo is not a symlink",
            verifier=_verifier(),
        )
        self.ledger.emit(c)
        sig = regex_detect_contradiction(
            "no, that's wrong — /etc/foo is actually a symlink",
            self.ledger,
        )
        self.assertIsNotNone(sig)
        self.assertEqual(sig.contradicted_claim_id, c.claim_id)
        self.assertEqual(sig.detector, "regex")
        self.assertGreaterEqual(sig.confidence, 0.6)

    def test_no_explicit_phrase_no_signal(self):
        # Same claim, but the user message lacks any contradiction
        # phrase — looks like a normal follow-up question.
        c = _make_claim(
            statement="/etc/foo is not a symlink",
            verifier=_verifier(),
        )
        self.ledger.emit(c)
        sig = regex_detect_contradiction(
            "what about /etc/foo?",
            self.ledger,
        )
        self.assertIsNone(sig)

    def test_low_overlap_no_signal(self):
        # Pushback phrase present but the user is talking about a
        # different subject — detector must not target a random claim.
        c = _make_claim(
            statement="/etc/foo is not a symlink",
            verifier=_verifier(),
        )
        self.ledger.emit(c)
        sig = regex_detect_contradiction(
            "no, you're wrong about the database backup schedule",
            self.ledger,
        )
        self.assertIsNone(sig)

    def test_dispatcher_uses_regex_by_default(self):
        c = _make_claim(
            statement="/etc/foo is not a symlink",
            verifier=_verifier(),
        )
        self.ledger.emit(c)
        with patch.dict(os.environ, {"EPISTEMIC_PUSHBACK_LLM_DETECTOR": ""}):
            sig = detect_contradiction(
                "actually /etc/foo is a symlink", self.ledger,
            )
        self.assertIsNotNone(sig)


# ============================================================================
# Foundation re-check protocol
# ============================================================================

class TestFoundationCheckProtocol(unittest.TestCase):

    def setUp(self):
        _reset_hooks()
        executor_mod._reset_for_tests()
        self.ledger = Ledger(task_id="task_abc")

    def tearDown(self):
        executor_mod._reset_for_tests()

    def _signal(self, claim_id: str) -> ContradictionSignal:
        return ContradictionSignal(
            contradicted_claim_id=claim_id,
            user_evidence="actually /etc/foo is a symlink",
            confidence=0.8,
            detected_at=datetime.now(timezone.utc),
            detector="regex",
        )

    def test_unknown_claim_id_unverifiable(self):
        result = handle_foundation_check(
            self._signal("clm_doesnotexist"), self.ledger,
        )
        self.assertEqual(result.outcome, FoundationOutcome.UNVERIFIABLE)
        self.assertIn("not in ledger", result.new_evidence_excerpt)

    def test_no_verifier_unverifiable(self):
        c = _make_claim(statement="/etc/foo is not a symlink", verifier=None)
        self.ledger.emit(c)
        result = handle_foundation_check(self._signal(c.claim_id), self.ledger)
        self.assertEqual(result.outcome, FoundationOutcome.UNVERIFIABLE)
        self.assertIn("no exact-answer verifier", result.new_evidence_excerpt)

    def test_default_executor_unverifiable(self):
        # Verifier present but no executor wired → settles=False → UNVERIFIABLE.
        c = _make_claim(
            statement="/etc/foo is not a symlink",
            verifier=_verifier(),
        )
        self.ledger.emit(c)
        result = handle_foundation_check(self._signal(c.claim_id), self.ledger)
        self.assertEqual(result.outcome, FoundationOutcome.UNVERIFIABLE)

    def test_reverified_when_executor_confirms(self):
        c = _make_claim(
            statement="/etc/foo is not a symlink",
            verifier=_verifier(),
        )
        self.ledger.emit(c)
        set_executor(lambda a: VerifierResult(
            settles=True, confirms=True, stdout="(empty)",
        ))
        result = handle_foundation_check(self._signal(c.claim_id), self.ledger)
        self.assertEqual(result.outcome, FoundationOutcome.REVERIFIED)
        # Original claim not superseded.
        self.assertEqual(
            self.ledger.by_id(c.claim_id).status, VerificationStatus.INFERRED,
        )

    def test_falsified_supersedes_target(self):
        c = _make_claim(
            statement="/etc/foo is not a symlink",
            verifier=_verifier(),
        )
        self.ledger.emit(c)
        set_executor(lambda a: VerifierResult(
            settles=True, confirms=False, stdout="/etc/foo -> /actual/path",
        ))
        result = handle_foundation_check(self._signal(c.claim_id), self.ledger)
        self.assertEqual(result.outcome, FoundationOutcome.FALSIFIED)
        # Original claim now CONTRADICTED with superseded_by set.
        contradicted = self.ledger.by_id(c.claim_id)
        self.assertEqual(contradicted.status, VerificationStatus.CONTRADICTED)
        self.assertIsNotNone(contradicted.superseded_by)
        # The replacement is a new VERIFIED claim in the ledger.
        replacement = self.ledger.by_id(contradicted.superseded_by)
        self.assertIsNotNone(replacement)
        self.assertEqual(replacement.status, VerificationStatus.VERIFIED)
        self.assertIn("/etc/foo -> /actual/path",
                      result.new_evidence_excerpt)

    def test_falsified_cascade_invalidates_dependents(self):
        # Foundation: /etc/foo is not a symlink.
        # Dependent: relies on the foundation via prior_claim evidence.
        foundation = _make_claim(
            statement="/etc/foo is not a symlink",
            verifier=_verifier(),
        )
        self.ledger.emit(foundation)

        dependent = _make_claim(
            statement="we can safely cp /etc/foo to backup",
            evidence=(Evidence(
                kind="prior_claim",
                source_ref=foundation.claim_id,
                excerpt="depends on the symlink check above",
                confidence=0.7,
            ),),
        )
        self.ledger.emit(dependent)

        set_executor(lambda a: VerifierResult(
            settles=True, confirms=False,
            stdout="/etc/foo -> /elsewhere",
        ))
        result = handle_foundation_check(
            self._signal(foundation.claim_id), self.ledger,
        )

        self.assertEqual(result.outcome, FoundationOutcome.FALSIFIED)
        self.assertIn(dependent.claim_id, result.invalidated_claim_ids)
        # Dependent now CONTRADICTED.
        self.assertEqual(
            self.ledger.by_id(dependent.claim_id).status,
            VerificationStatus.CONTRADICTED,
        )

    def test_executor_returning_unsettled_unverifiable(self):
        c = _make_claim(
            statement="/etc/foo is not a symlink",
            verifier=_verifier(),
        )
        self.ledger.emit(c)
        set_executor(lambda a: VerifierResult(
            settles=False, confirms=False, stderr="permission denied",
        ))
        result = handle_foundation_check(self._signal(c.claim_id), self.ledger)
        self.assertEqual(result.outcome, FoundationOutcome.UNVERIFIABLE)
        self.assertIn("permission denied", result.new_evidence_excerpt)


# ============================================================================
# process_user_message coordinator
# ============================================================================

class TestProcessUserMessage(unittest.TestCase):

    def setUp(self):
        _reset_hooks()
        executor_mod._reset_for_tests()
        self.ledger = Ledger(task_id="task_abc")

    def tearDown(self):
        executor_mod._reset_for_tests()

    def test_no_contradiction_no_check(self):
        c = _make_claim(
            statement="/etc/foo is not a symlink",
            verifier=_verifier(),
        )
        self.ledger.emit(c)
        outcome = process_user_message(
            "what about the next step?", self.ledger, persist=False,
        )
        self.assertFalse(outcome.fired)
        self.assertIsNone(outcome.signal)
        self.assertIsNone(outcome.check)

    def test_contradiction_runs_check(self):
        c = _make_claim(
            statement="/etc/foo is not a symlink",
            verifier=_verifier(),
        )
        self.ledger.emit(c)
        set_executor(lambda a: VerifierResult(
            settles=True, confirms=True, stdout="(empty)",
        ))
        outcome = process_user_message(
            "actually /etc/foo is a symlink — you're wrong",
            self.ledger,
            persist=False,
        )
        self.assertTrue(outcome.fired)
        self.assertEqual(outcome.signal.contradicted_claim_id, c.claim_id)
        self.assertEqual(outcome.check.outcome, FoundationOutcome.REVERIFIED)


# ============================================================================
# Persistence
# ============================================================================

class TestPersistPushbackEvent(unittest.TestCase):

    def _signal_check_pair(self):
        signal = ContradictionSignal(
            contradicted_claim_id="clm_aa",
            user_evidence="actually wrong",
            confidence=0.8,
            detected_at=datetime(2026, 4, 30, 12, 0, tzinfo=timezone.utc),
            detector="regex",
        )
        from app.epistemic.pushback import FoundationCheckResult
        check = FoundationCheckResult(
            outcome=FoundationOutcome.FALSIFIED,
            contradicted_claim_id="clm_aa",
            new_evidence_excerpt="readlink: /elsewhere",
            invalidated_claim_ids=("clm_bb",),
            duration_seconds=0.42,
        )
        return signal, check

    def test_no_op_when_disabled(self):
        signal, check = self._signal_check_pair()
        with patch.dict(os.environ, {"EPISTEMIC_ENABLED": ""}), \
             patch.object(span_writer, "execute") as mock_exec:
            span_writer.persist_pushback_event(
                task_id="task_abc", signal=signal, check=check,
            )
            mock_exec.assert_not_called()

    def test_insert_when_enabled(self):
        signal, check = self._signal_check_pair()
        with patch.dict(os.environ, {"EPISTEMIC_ENABLED": "true"}), \
             patch.object(span_writer, "execute") as mock_exec:
            span_writer.persist_pushback_event(
                task_id="task_abc", signal=signal, check=check,
            )
            mock_exec.assert_called_once()
            sql, params = mock_exec.call_args[0]
            self.assertIn(
                "INSERT INTO control_plane.epistemic_pushback_events", sql,
            )
            (
                task_id, claim_id, user_evidence, confidence, detector,
                outcome, excerpt, invalidated_json, duration, detected,
            ) = params
            self.assertEqual(task_id, "task_abc")
            self.assertEqual(claim_id, "clm_aa")
            self.assertEqual(user_evidence, "actually wrong")
            self.assertAlmostEqual(confidence, 0.8)
            self.assertEqual(detector, "regex")
            self.assertEqual(outcome, "falsified")
            self.assertIn("/elsewhere", excerpt)
            import json
            self.assertEqual(json.loads(invalidated_json), ["clm_bb"])

    def test_db_error_swallowed(self):
        signal, check = self._signal_check_pair()
        with patch.dict(os.environ, {"EPISTEMIC_ENABLED": "true"}), \
             patch.object(
                 span_writer, "execute",
                 side_effect=RuntimeError("connection refused"),
             ):
            # Must not raise.
            span_writer.persist_pushback_event(
                task_id="task_abc", signal=signal, check=check,
            )


class TestPushbackAggregates(unittest.TestCase):

    def test_empty_when_disabled(self):
        with patch.dict(os.environ, {"EPISTEMIC_ENABLED": ""}):
            agg = span_writer.pushback_aggregates(window_minutes=60)
        self.assertEqual(agg["total"], 0)
        self.assertEqual(agg["reverified"], 0)

    def test_groups_by_outcome(self):
        rows = [
            {"outcome": "reverified", "n": 3, "mean_seconds": 0.5},
            {"outcome": "falsified", "n": 1, "mean_seconds": 0.8},
            {"outcome": "unverifiable", "n": 2, "mean_seconds": 0.3},
        ]
        with patch.dict(os.environ, {"EPISTEMIC_ENABLED": "true"}), \
             patch.object(span_writer, "execute", return_value=rows):
            agg = span_writer.pushback_aggregates(window_minutes=60)
        self.assertEqual(agg["total"], 6)
        self.assertEqual(agg["reverified"], 3)
        self.assertEqual(agg["falsified"], 1)
        self.assertEqual(agg["unverifiable"], 2)
        # Weighted mean: (3*0.5 + 1*0.8 + 2*0.3) / 6 = (1.5+0.8+0.6)/6 ≈ 0.483
        self.assertAlmostEqual(agg["mean_seconds_to_recheck"], 0.483, places=2)

    def test_db_error_returns_empty(self):
        with patch.dict(os.environ, {"EPISTEMIC_ENABLED": "true"}), \
             patch.object(span_writer, "execute",
                          side_effect=RuntimeError("boom")):
            agg = span_writer.pushback_aggregates(window_minutes=60)
        self.assertEqual(agg["total"], 0)


# ============================================================================
# API endpoints
# ============================================================================

class TestPushbackAPIEndpoints(unittest.TestCase):

    def test_stats_returns_aggregate_shape(self):
        client = _build_client()
        with patch("app.epistemic.api.pushback_aggregates", return_value={
            "window_minutes": 60, "total": 5,
            "reverified": 3, "falsified": 1, "unverifiable": 1,
            "mean_seconds_to_recheck": 0.5,
        }):
            resp = client.get("/epistemic/pushback/stats?window_min=60")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["total"], 5)
        self.assertEqual(body["reverified"], 3)
        self.assertEqual(body["mean_seconds_to_recheck"], 0.5)

    def test_recent_returns_event_list(self):
        client = _build_client()
        events = [
            {
                "id": 1, "task_id": "task_abc",
                "contradicted_claim_id": "clm_aa",
                "user_evidence": "actually wrong",
                "confidence": 0.8, "detector": "regex",
                "outcome": "falsified",
                "new_evidence_excerpt": "readlink: /elsewhere",
                "invalidated_claim_ids": ["clm_bb"],
                "duration_seconds": 0.42,
                "detected_at": "2026-04-30T12:00:00+00:00",
            },
        ]
        with patch("app.epistemic.api.list_recent_pushback_events",
                   return_value=events):
            resp = client.get("/epistemic/pushback/recent?window_min=60")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["count"], 1)
        self.assertEqual(body["events"][0]["outcome"], "falsified")

    def test_window_validation(self):
        client = _build_client()
        # window_min=0 is below ge=1 — FastAPI returns 422.
        resp = client.get("/epistemic/pushback/stats?window_min=0")
        self.assertEqual(resp.status_code, 422)


if __name__ == "__main__":
    unittest.main()
