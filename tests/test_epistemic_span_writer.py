"""Tests for app.epistemic.span_writer.

The DB layer is mocked at ``app.control_plane.db.execute`` /
``execute_one``. We exercise the SQL parameter shapes and the
gating semantics, not real PostgreSQL.
"""
from __future__ import annotations

import json
import os
import sys
import types
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

# ── Stub psycopg2 + heavy deps before importing app modules ──────────
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
    Register,
    VerificationStatus,
    VerifyingAction,
)
from app.epistemic import span_writer  # noqa: E402


# ============================================================================
# Helpers
# ============================================================================

def _enabled():
    """Patch context manager: turn the layer on for the duration of a test."""
    return patch.dict(os.environ, {"EPISTEMIC_ENABLED": "true"})


def _disabled():
    return patch.dict(os.environ, {"EPISTEMIC_ENABLED": ""})


def _claim(**overrides) -> Claim:
    defaults = dict(
        task_id="task_abc",
        agent_role="researcher",
        statement="path is not a symlink",
        status=VerificationStatus.INFERRED,
        register=Register.DECLARATIVE,
        load_bearing=True,
        span_id=42,
    )
    defaults.update(overrides)
    return Claim.new(**defaults)


# ============================================================================
# persist_claim
# ============================================================================

class TestPersistClaim(unittest.TestCase):

    def test_no_op_when_disabled(self):
        with _disabled(), patch.object(span_writer, "execute") as mock_exec:
            span_writer.persist_claim(_claim())
            mock_exec.assert_not_called()

    def test_basic_insert_when_enabled(self):
        c = _claim(
            evidence=(Evidence(kind="tool_call", source_ref="42",
                               excerpt="drwxr-xr-x", confidence=0.6),),
            verifying_action=VerifyingAction(
                tool="readlink", args={"path": "/foo"},
                expected_signal="empty=not symlink", estimated_seconds=0.5,
            ),
            tags=("filesystem",),
        )
        with _enabled(), patch.object(span_writer, "execute") as mock_exec:
            span_writer.persist_claim(c)
            mock_exec.assert_called_once()
            sql, params = mock_exec.call_args[0]
            self.assertIn("INSERT INTO control_plane.epistemic_claims", sql)
            self.assertIn("ON CONFLICT (claim_id) DO UPDATE", sql)

            # Positional param order must match the column list. Spot-check
            # the load-bearing fields rather than every column.
            (claim_id, task_id, span_id, agent_role, statement,
             status, register, evidence_json, verifier_json,
             load_bearing, tags_json, superseded_by, created_at) = params

            self.assertEqual(claim_id, c.claim_id)
            self.assertEqual(task_id, "task_abc")
            self.assertEqual(span_id, 42)
            self.assertEqual(agent_role, "researcher")
            self.assertEqual(statement, "path is not a symlink")
            self.assertEqual(status, "inferred")
            self.assertEqual(register, "declarative")
            self.assertTrue(load_bearing)
            self.assertIsNone(superseded_by)
            self.assertIsInstance(created_at, datetime)

            evidence = json.loads(evidence_json)
            self.assertEqual(len(evidence), 1)
            self.assertEqual(evidence[0]["source_ref"], "42")
            self.assertAlmostEqual(evidence[0]["confidence"], 0.6)

            verifier = json.loads(verifier_json)
            self.assertEqual(verifier["tool"], "readlink")
            self.assertEqual(verifier["args"], {"path": "/foo"})
            self.assertEqual(verifier["safety"], "read_only")

            tags = json.loads(tags_json)
            self.assertEqual(tags, ["filesystem"])

    def test_null_verifier_when_absent(self):
        c = _claim(verifying_action=None)
        with _enabled(), patch.object(span_writer, "execute") as mock_exec:
            span_writer.persist_claim(c)
            params = mock_exec.call_args[0][1]
            verifier_param = params[8]  # 9th positional param
            self.assertIsNone(verifier_param)

    def test_db_error_swallowed(self):
        with _enabled(), patch.object(
            span_writer, "execute", side_effect=RuntimeError("connection refused"),
        ):
            # Must not raise — the contract is fire-and-forget.
            span_writer.persist_claim(_claim())


# ============================================================================
# load_ledger_for_task
# ============================================================================

class TestLoadLedgerForTask(unittest.TestCase):

    def test_returns_empty_ledger_when_disabled(self):
        with _disabled(), patch.object(span_writer, "execute") as mock_exec:
            ledger = span_writer.load_ledger_for_task("task_abc")
            mock_exec.assert_not_called()
            self.assertEqual(ledger.task_id, "task_abc")
            self.assertEqual(len(ledger), 0)

    def test_rehydrates_from_rows(self):
        # Simulate JSONB columns returned as already-parsed Python values
        # (psycopg2's standard behavior with the default JSONB adapter).
        rows = [
            {
                "claim_id": "clm_aaaa11112222",
                "task_id": "task_abc",
                "span_id": 42,
                "agent_role": "researcher",
                "statement": "path is not a symlink",
                "status": "inferred",
                "register": "declarative",
                "evidence": [{
                    "kind": "tool_call", "source_ref": "42",
                    "excerpt": "drwxr-xr-x", "confidence": 0.6,
                }],
                "verifying_action": {
                    "tool": "readlink", "args": {"path": "/foo"},
                    "expected_signal": "empty=not symlink",
                    "estimated_seconds": 0.5, "safety": "read_only",
                },
                "load_bearing": True,
                "tags": ["filesystem"],
                "superseded_by": None,
                "created_at": datetime(2026, 4, 30, 12, 0, tzinfo=timezone.utc),
            },
            {
                "claim_id": "clm_bbbb33334444",
                "task_id": "task_abc",
                "span_id": None,
                "agent_role": "coder",
                "statement": "needs cp to fix",
                "status": "inferred",
                "register": "declarative",
                "evidence": [],
                "verifying_action": None,
                "load_bearing": False,
                "tags": [],
                "superseded_by": None,
                "created_at": datetime(2026, 4, 30, 12, 1, tzinfo=timezone.utc),
            },
        ]
        with _enabled(), patch.object(span_writer, "execute", return_value=rows):
            ledger = span_writer.load_ledger_for_task("task_abc")

        self.assertEqual(len(ledger), 2)
        first = ledger.by_id("clm_aaaa11112222")
        self.assertIsNotNone(first)
        self.assertEqual(first.statement, "path is not a symlink")
        self.assertEqual(first.status, VerificationStatus.INFERRED)
        self.assertTrue(first.load_bearing)
        self.assertEqual(first.tags, ("filesystem",))
        self.assertIsNotNone(first.verifying_action)
        self.assertEqual(first.verifying_action.tool, "readlink")
        self.assertEqual(len(first.evidence), 1)
        self.assertEqual(first.evidence[0].excerpt, "drwxr-xr-x")

        second = ledger.by_id("clm_bbbb33334444")
        self.assertEqual(second.statement, "needs cp to fix")
        self.assertIsNone(second.verifying_action)
        self.assertEqual(second.evidence, ())

    def test_empty_result_returns_empty_ledger(self):
        with _enabled(), patch.object(span_writer, "execute", return_value=[]):
            ledger = span_writer.load_ledger_for_task("task_abc")
            self.assertEqual(len(ledger), 0)

    def test_db_error_returns_empty_ledger(self):
        with _enabled(), patch.object(
            span_writer, "execute", side_effect=RuntimeError("boom"),
        ):
            ledger = span_writer.load_ledger_for_task("task_abc")
            # Must not raise; returns a usable empty Ledger.
            self.assertEqual(len(ledger), 0)


# ============================================================================
# lookup_claim
# ============================================================================

class TestLookupClaim(unittest.TestCase):

    def test_returns_none_when_disabled(self):
        with _disabled(), patch.object(span_writer, "execute_one") as mock_exec:
            self.assertIsNone(span_writer.lookup_claim("clm_anything"))
            mock_exec.assert_not_called()

    def test_returns_none_when_not_found(self):
        with _enabled(), patch.object(span_writer, "execute_one", return_value=None):
            self.assertIsNone(span_writer.lookup_claim("clm_missing"))

    def test_returns_claim_on_hit(self):
        row = {
            "claim_id": "clm_aaaa11112222",
            "task_id": "task_abc",
            "span_id": 42,
            "agent_role": "researcher",
            "statement": "path is not a symlink",
            "status": "inferred",
            "register": "declarative",
            "evidence": [],
            "verifying_action": None,
            "load_bearing": True,
            "tags": [],
            "superseded_by": None,
            "created_at": datetime(2026, 4, 30, 12, 0, tzinfo=timezone.utc),
        }
        with _enabled(), patch.object(span_writer, "execute_one", return_value=row):
            claim = span_writer.lookup_claim("clm_aaaa11112222")

        self.assertIsNotNone(claim)
        self.assertEqual(claim.claim_id, "clm_aaaa11112222")
        self.assertEqual(claim.statement, "path is not a symlink")

    def test_db_error_returns_none(self):
        with _enabled(), patch.object(
            span_writer, "execute_one", side_effect=RuntimeError("boom"),
        ):
            self.assertIsNone(span_writer.lookup_claim("clm_anything"))


# ============================================================================
# Integration: Ledger.emit triggers persist_claim through the lazy import path
# ============================================================================

class TestLedgerEmitPersistsThroughLazyImport(unittest.TestCase):
    """Phase 0's invariant: a real Ledger.emit() call goes through the
    lazy import in ledger._persist and reaches span_writer.persist_claim
    with the layer enabled. No mock of the lazy path; we patch only the
    underlying ``execute`` so no DB is touched."""

    def test_emit_reaches_execute(self):
        from app.epistemic import Ledger
        ledger = Ledger(task_id="task_abc")
        c = _claim()
        with _enabled(), patch.object(span_writer, "execute") as mock_exec:
            ledger.emit(c)
            mock_exec.assert_called_once()
            params = mock_exec.call_args[0][1]
            self.assertEqual(params[0], c.claim_id)
            self.assertEqual(params[1], "task_abc")


if __name__ == "__main__":
    unittest.main()
