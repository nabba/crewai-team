"""Tests for Phase 7:

* gate_output (the orchestrator entry point)
* is_blocking_mode_enabled (env-var gate)
* OverrideEvent + record_override (persistence + Self-Improver flush)
* OverrideAction enum
* /epistemic/overrides/{stats,recent} GET endpoints
* POST /epistemic/overrides
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
from app.epistemic.calibration import CalibrationVerdict  # noqa: E402
from app.epistemic.orchestrator_hook import (  # noqa: E402
    GateResult,
    gate_output,
    is_blocking_mode_enabled,
)
from app.epistemic.override import (  # noqa: E402
    OverrideAction,
    OverrideEvent,
    record_override,
)
from app.epistemic.peer_review import (  # noqa: E402
    PeerReviewDecision,
    PeerReviewVerdict,
)


def _build_client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _enabled():
    return patch.dict(os.environ, {"EPISTEMIC_ENABLED": "true"})


def _disabled():
    return patch.dict(os.environ, {"EPISTEMIC_ENABLED": ""})


def _blocking():
    return patch.dict(os.environ, {
        "EPISTEMIC_ENABLED": "true",
        "EPISTEMIC_BLOCKING_MODE": "true",
    })


def _observe():
    return patch.dict(os.environ, {
        "EPISTEMIC_ENABLED": "true",
        "EPISTEMIC_BLOCKING_MODE": "",
    })


# ============================================================================
# is_blocking_mode_enabled
# ============================================================================

class TestBlockingModeEnabled(unittest.TestCase):

    def test_default_off(self):
        with patch.dict(os.environ, {"EPISTEMIC_BLOCKING_MODE": ""}):
            self.assertFalse(is_blocking_mode_enabled())

    def test_true_value(self):
        with patch.dict(os.environ, {"EPISTEMIC_BLOCKING_MODE": "true"}):
            self.assertTrue(is_blocking_mode_enabled())

    def test_alternate_true_values(self):
        for val in ("1", "yes", "on", "TRUE", "True"):
            with patch.dict(os.environ, {"EPISTEMIC_BLOCKING_MODE": val}):
                self.assertTrue(is_blocking_mode_enabled(), f"failed for {val!r}")

    def test_garbage_is_false(self):
        for val in ("maybe", "0", "false", "no"):
            with patch.dict(os.environ, {"EPISTEMIC_BLOCKING_MODE": val}):
                self.assertFalse(is_blocking_mode_enabled(), f"failed for {val!r}")


# ============================================================================
# gate_output: master kill switch
# ============================================================================

class TestGateOutputDisabled(unittest.TestCase):

    def test_disabled_layer_passes_through(self):
        with _disabled():
            result = gate_output(
                proposal_text="hello world",
                task_id="task_abc",
            )
        self.assertEqual(result.action, "ship")
        self.assertEqual(result.final_text, "hello world")
        self.assertIn("disabled", result.diagnostic_note.lower())
        self.assertFalse(result.blocking_mode)


# ============================================================================
# gate_output: no-bias path
# ============================================================================

class TestGateOutputClean(unittest.TestCase):

    def test_no_matches_ships_unchanged(self):
        ledger = Ledger.from_claims(task_id="task_abc", claims=[])
        with _enabled(), \
             patch("app.epistemic.orchestrator_hook.load_ledger_for_task",
                   return_value=ledger), \
             patch("app.epistemic.orchestrator_hook.calibration_check",
                   return_value=CalibrationVerdict(
                       proceed=True, suggested_action="ship",
                   )):
            result = gate_output(
                proposal_text="all good",
                task_id="task_abc",
            )
        self.assertEqual(result.action, "ship")
        self.assertEqual(result.final_text, "all good")
        self.assertIsNotNone(result.verdict)
        self.assertEqual(result.verdict.suggested_action, "ship")


# ============================================================================
# gate_output: non-critical (verify/hedge) — observe vs blocking
# ============================================================================

class TestGateOutputNonCritical(unittest.TestCase):

    def _verdict(self) -> CalibrationVerdict:
        return CalibrationVerdict(
            proceed=True,  # warn-mode default
            suggested_action="verify",
            biases_detected=(BiasMatch(
                bias_id="inference_as_fact",
                matched_claim_ids=("clm_aa",),
                severity=Severity.HIGH,
            ),),
            note_for_post_mortem="inference_as_fact×1",
        )

    def test_observe_mode_ships_with_diagnostic(self):
        ledger = Ledger.from_claims(task_id="task_abc", claims=[])
        with _observe(), \
             patch("app.epistemic.orchestrator_hook.load_ledger_for_task",
                   return_value=ledger), \
             patch("app.epistemic.orchestrator_hook.calibration_check",
                   return_value=self._verdict()):
            result = gate_output(
                proposal_text="X is Y", task_id="task_abc",
            )
        self.assertEqual(result.action, "ship")
        self.assertEqual(result.final_text, "X is Y")
        self.assertIn("observe-mode", result.diagnostic_note)
        self.assertFalse(result.blocking_mode)

    def test_blocking_mode_appends_hedge(self):
        ledger = Ledger.from_claims(task_id="task_abc", claims=[])
        with _blocking(), \
             patch("app.epistemic.orchestrator_hook.load_ledger_for_task",
                   return_value=ledger), \
             patch("app.epistemic.orchestrator_hook.calibration_check",
                   return_value=self._verdict()):
            result = gate_output(
                proposal_text="X is Y", task_id="task_abc",
            )
        self.assertEqual(result.action, "revise")
        self.assertTrue(result.revised)
        self.assertIn("X is Y", result.final_text)
        self.assertIn("low confidence", result.final_text.lower())
        self.assertTrue(result.blocking_mode)


# ============================================================================
# gate_output: peer-review escalation paths
# ============================================================================

class TestGateOutputPeerReview(unittest.TestCase):

    def _critical_verdict(self) -> CalibrationVerdict:
        return CalibrationVerdict(
            proceed=False,
            suggested_action="peer_review",
            biases_detected=(BiasMatch(
                bias_id="destructive_without_recheck",
                matched_claim_ids=("clm_aa",),
                severity=Severity.CRITICAL,
            ),),
        )

    def test_observe_mode_ships_despite_veto(self):
        ledger = Ledger.from_claims(task_id="task_abc", claims=[])
        veto = PeerReviewVerdict(
            decision=PeerReviewDecision.VETO,
            rationale="diagnosis shaky",
            reviewers=("heuristic",),
        )
        with _observe(), \
             patch("app.epistemic.orchestrator_hook.load_ledger_for_task",
                   return_value=ledger), \
             patch("app.epistemic.orchestrator_hook.calibration_check",
                   return_value=self._critical_verdict()), \
             patch("app.epistemic.orchestrator_hook.escalate") as mock_escalate:
            from app.epistemic.peer_review import EscalationOutcome
            mock_escalate.return_value = EscalationOutcome(
                escalated=True, verdict=veto,
            )
            result = gate_output(
                proposal_text="rm -rf /var", task_id="task_abc",
            )
        # In observe mode, veto is recorded but proposal still ships.
        self.assertEqual(result.action, "ship")
        self.assertEqual(result.final_text, "rm -rf /var")
        self.assertIn("veto", result.diagnostic_note.lower())
        self.assertIsNotNone(result.escalation)

    def test_blocking_mode_blocks_on_veto(self):
        ledger = Ledger.from_claims(task_id="task_abc", claims=[])
        veto = PeerReviewVerdict(
            decision=PeerReviewDecision.VETO,
            rationale="unverified load-bearing claims",
            reviewers=("heuristic",),
        )
        with _blocking(), \
             patch("app.epistemic.orchestrator_hook.load_ledger_for_task",
                   return_value=ledger), \
             patch("app.epistemic.orchestrator_hook.calibration_check",
                   return_value=self._critical_verdict()), \
             patch("app.epistemic.orchestrator_hook.escalate") as mock_escalate:
            from app.epistemic.peer_review import EscalationOutcome
            mock_escalate.return_value = EscalationOutcome(
                escalated=True, verdict=veto,
            )
            result = gate_output(
                proposal_text="rm -rf /var", task_id="task_abc",
            )
        self.assertEqual(result.action, "block")
        self.assertNotEqual(result.final_text, "rm -rf /var")
        self.assertIn("unverified load-bearing", result.user_visible_reason)
        self.assertIn("pausing", result.final_text.lower())

    def test_blocking_mode_revises_with_suggested_revision(self):
        ledger = Ledger.from_claims(task_id="task_abc", claims=[])
        revise = PeerReviewVerdict(
            decision=PeerReviewDecision.REVISE,
            rationale="add hedge",
            suggested_revision="Maybe rm -rf /var (verify first)",
            reviewers=("heuristic",),
        )
        with _blocking(), \
             patch("app.epistemic.orchestrator_hook.load_ledger_for_task",
                   return_value=ledger), \
             patch("app.epistemic.orchestrator_hook.calibration_check",
                   return_value=self._critical_verdict()), \
             patch("app.epistemic.orchestrator_hook.escalate") as mock_escalate:
            from app.epistemic.peer_review import EscalationOutcome
            mock_escalate.return_value = EscalationOutcome(
                escalated=True, verdict=revise,
            )
            result = gate_output(
                proposal_text="rm -rf /var", task_id="task_abc",
            )
        self.assertEqual(result.action, "revise")
        self.assertEqual(result.final_text, "Maybe rm -rf /var (verify first)")
        self.assertTrue(result.revised)

    def test_blocking_mode_ships_on_allow(self):
        ledger = Ledger.from_claims(task_id="task_abc", claims=[])
        allow = PeerReviewVerdict(
            decision=PeerReviewDecision.ALLOW,
            rationale="all verified",
            reviewers=("heuristic",),
        )
        with _blocking(), \
             patch("app.epistemic.orchestrator_hook.load_ledger_for_task",
                   return_value=ledger), \
             patch("app.epistemic.orchestrator_hook.calibration_check",
                   return_value=self._critical_verdict()), \
             patch("app.epistemic.orchestrator_hook.escalate") as mock_escalate:
            from app.epistemic.peer_review import EscalationOutcome
            mock_escalate.return_value = EscalationOutcome(
                escalated=True, verdict=allow,
            )
            result = gate_output(
                proposal_text="rm -rf /var", task_id="task_abc",
            )
        self.assertEqual(result.action, "ship")
        self.assertEqual(result.final_text, "rm -rf /var")


# ============================================================================
# gate_output: defensive paths (failures must never raise)
# ============================================================================

class TestGateOutputDefensive(unittest.TestCase):

    def test_ledger_load_failure_ships(self):
        with _enabled(), \
             patch("app.epistemic.orchestrator_hook.load_ledger_for_task",
                   side_effect=RuntimeError("DB down")):
            result = gate_output(
                proposal_text="foo", task_id="task_abc",
            )
        self.assertEqual(result.action, "ship")
        self.assertEqual(result.final_text, "foo")
        self.assertIn("ledger unavailable", result.diagnostic_note.lower())

    def test_calibration_failure_ships(self):
        ledger = Ledger.from_claims(task_id="task_abc", claims=[])
        with _enabled(), \
             patch("app.epistemic.orchestrator_hook.load_ledger_for_task",
                   return_value=ledger), \
             patch("app.epistemic.orchestrator_hook.calibration_check",
                   side_effect=RuntimeError("classifier broken")):
            result = gate_output(
                proposal_text="foo", task_id="task_abc",
            )
        self.assertEqual(result.action, "ship")
        self.assertEqual(result.final_text, "foo")
        self.assertIn("calibration_check failed", result.diagnostic_note)

    def test_escalate_failure_ships(self):
        ledger = Ledger.from_claims(task_id="task_abc", claims=[])
        verdict = CalibrationVerdict(
            proceed=False, suggested_action="peer_review",
        )
        with _enabled(), \
             patch("app.epistemic.orchestrator_hook.load_ledger_for_task",
                   return_value=ledger), \
             patch("app.epistemic.orchestrator_hook.calibration_check",
                   return_value=verdict), \
             patch("app.epistemic.orchestrator_hook.escalate",
                   side_effect=RuntimeError("peer review broke")):
            result = gate_output(
                proposal_text="rm -rf /var", task_id="task_abc",
            )
        # Even with peer review broken, the user must not see a 500.
        self.assertEqual(result.action, "ship")


# ============================================================================
# Override: persistence + Self-Improver flush
# ============================================================================

class TestRecordOverride(unittest.TestCase):

    def test_returns_event_with_unique_id(self):
        with patch("app.epistemic.override._flush_to_self_improver",
                   return_value=False), \
             patch("app.epistemic.span_writer.persist_override"):
            event = record_override(
                task_id="task_abc",
                blocked_action="block",
                user_action=OverrideAction.FORCE_PROCEED,
                user_reasoning="I know better, this is fine",
                flush_to_self_improver=False,
            )
        self.assertTrue(event.override_id.startswith("ovr_"))
        self.assertEqual(event.task_id, "task_abc")
        self.assertEqual(event.user_action, OverrideAction.FORCE_PROCEED)
        self.assertEqual(event.user_reasoning, "I know better, this is fine")
        self.assertEqual(event.blocked_action, "block")

    def test_persistence_called(self):
        with patch.object(span_writer, "persist_override") as mock_persist, \
             patch("app.epistemic.override._flush_to_self_improver",
                   return_value=False):
            record_override(
                task_id="task_abc",
                blocked_action="revise",
                user_action=OverrideAction.USE_REVISION,
                user_reasoning="ok",
                flush_to_self_improver=False,
            )
        mock_persist.assert_called_once()

    def test_persistence_failure_swallowed(self):
        # Override capture must not break the user's force-proceed path.
        with patch.object(span_writer, "persist_override",
                          side_effect=RuntimeError("DB down")), \
             patch("app.epistemic.override._flush_to_self_improver",
                   return_value=False):
            event = record_override(
                task_id="task_abc",
                blocked_action="block",
                user_action=OverrideAction.FORCE_PROCEED,
                user_reasoning="proceed anyway",
                flush_to_self_improver=False,
            )
        # Returns the event regardless of persistence outcome.
        self.assertIsNotNone(event.override_id)

    def test_self_improver_flush_uses_user_correction(self):
        from app.self_improvement.types import GapSource
        with patch.object(span_writer, "persist_override"), \
             patch("app.self_improvement.store.emit_gap",
                   return_value=True) as mock_emit:
            record_override(
                task_id="task_abc",
                blocked_action="block",
                user_action=OverrideAction.FORCE_PROCEED,
                user_reasoning="I know my system",
                flush_to_self_improver=True,
            )
        mock_emit.assert_called_once()
        gap = mock_emit.call_args[0][0]
        self.assertEqual(gap.source, GapSource.USER_CORRECTION)
        self.assertAlmostEqual(gap.signal_strength, 0.9)

    def test_self_improver_unavailable_does_not_raise(self):
        # If Self-Improver isn't importable, override still records
        # (the row is in the DB; humans can review later).
        with patch.object(span_writer, "persist_override"), \
             patch("app.self_improvement.store.emit_gap",
                   side_effect=ImportError("not available")):
            event = record_override(
                task_id="task_abc",
                blocked_action="block",
                user_action=OverrideAction.FORCE_PROCEED,
                user_reasoning="proceed",
                flush_to_self_improver=True,
            )
        self.assertIsNotNone(event)


# ============================================================================
# Override: persistence layer functions
# ============================================================================

class TestOverridePersistence(unittest.TestCase):

    def _event(self) -> OverrideEvent:
        return OverrideEvent(
            override_id="ovr_aa",
            task_id="task_abc",
            peer_review_id=42,
            blocked_action="block",
            user_action=OverrideAction.FORCE_PROCEED,
            user_reasoning="I know better",
            overridden_at=datetime(2026, 4, 30, 12, tzinfo=timezone.utc),
        )

    def test_persist_override_no_op_when_disabled(self):
        with _disabled(), patch.object(span_writer, "execute") as mock_exec:
            span_writer.persist_override(self._event())
            mock_exec.assert_not_called()

    def test_persist_override_inserts_when_enabled(self):
        with _enabled(), patch.object(span_writer, "execute") as mock_exec:
            span_writer.persist_override(self._event())
            mock_exec.assert_called_once()
            sql, params = mock_exec.call_args[0]
            self.assertIn(
                "INSERT INTO control_plane.epistemic_overrides", sql,
            )
            (
                override_id, task_id, peer_review_id, blocked_action,
                user_action, user_reasoning, overridden_at,
            ) = params
            self.assertEqual(override_id, "ovr_aa")
            self.assertEqual(task_id, "task_abc")
            self.assertEqual(peer_review_id, 42)
            self.assertEqual(user_action, "force_proceed")

    def test_list_recent_overrides(self):
        rows = [
            {
                "override_id": "ovr_aa", "task_id": "task_abc",
                "peer_review_id": None,
                "blocked_action": "block",
                "user_action": "force_proceed",
                "user_reasoning": "proceed",
                "overridden_at": datetime(2026, 4, 30, 12, tzinfo=timezone.utc),
            },
        ]
        with _enabled(), patch.object(span_writer, "execute", return_value=rows):
            out = span_writer.list_recent_overrides()
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["override_id"], "ovr_aa")

    def test_override_aggregates_groups(self):
        rows = [
            {"user_action": "force_proceed", "n": 3},
            {"user_action": "use_revision", "n": 2},
            {"user_action": "abandon", "n": 1},
        ]
        with _enabled(), patch.object(span_writer, "execute", return_value=rows):
            agg = span_writer.override_aggregates(window_minutes=60)
        self.assertEqual(agg["total"], 6)
        self.assertEqual(agg["force_proceed"], 3)
        self.assertEqual(agg["use_revision"], 2)
        self.assertEqual(agg["abandon"], 1)


# ============================================================================
# API endpoints: overrides
# ============================================================================

class TestOverrideAPI(unittest.TestCase):

    def test_stats_endpoint(self):
        client = _build_client()
        with patch(
            "app.epistemic.api.override_aggregates",
            return_value={
                "window_minutes": 60, "total": 5,
                "force_proceed": 3, "use_revision": 1, "abandon": 1,
            },
        ):
            resp = client.get("/epistemic/overrides/stats?window_min=60")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["total"], 5)
        self.assertEqual(body["force_proceed"], 3)

    def test_recent_endpoint(self):
        client = _build_client()
        items = [{
            "override_id": "ovr_aa", "task_id": "task_abc",
            "peer_review_id": None, "blocked_action": "block",
            "user_action": "force_proceed",
            "user_reasoning": "I know my codebase",
            "overridden_at": "2026-04-30T12:00:00+00:00",
        }]
        with patch(
            "app.epistemic.api.list_recent_overrides", return_value=items,
        ):
            resp = client.get("/epistemic/overrides/recent?window_min=60")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["count"], 1)
        self.assertEqual(body["overrides"][0]["override_id"], "ovr_aa")

    def test_post_override_persists(self):
        client = _build_client()
        with patch(
            "app.epistemic.override._flush_to_self_improver",
            return_value=False,
        ), patch.object(span_writer, "persist_override"):
            resp = client.post(
                "/epistemic/overrides",
                json={
                    "task_id": "task_abc",
                    "blocked_action": "block",
                    "user_action": "force_proceed",
                    "user_reasoning": "context the gate can't see",
                    "flush_to_self_improver": False,
                },
            )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["override_id"].startswith("ovr_"))
        self.assertEqual(body["user_action"], "force_proceed")

    def test_post_override_validates_user_action(self):
        client = _build_client()
        resp = client.post(
            "/epistemic/overrides",
            json={
                "task_id": "task_abc",
                "blocked_action": "block",
                "user_action": "invalid",
                "user_reasoning": "x",
            },
        )
        self.assertEqual(resp.status_code, 400)

    def test_post_override_validates_blocked_action(self):
        client = _build_client()
        resp = client.post(
            "/epistemic/overrides",
            json={
                "task_id": "task_abc",
                "blocked_action": "wat",
                "user_action": "force_proceed",
                "user_reasoning": "x",
            },
        )
        self.assertEqual(resp.status_code, 400)

    def test_post_override_validates_task_id(self):
        client = _build_client()
        resp = client.post(
            "/epistemic/overrides",
            json={
                "blocked_action": "block",
                "user_action": "force_proceed",
                "user_reasoning": "x",
            },
        )
        self.assertEqual(resp.status_code, 400)


if __name__ == "__main__":
    unittest.main()
