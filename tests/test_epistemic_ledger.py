"""Tests for app.epistemic.ledger and app.epistemic.registry.

Pure-logic tests — the persist path is gated by EPISTEMIC_ENABLED (off
here), so Ledger.emit no-ops the DB write without needing a mock. We do
stub psycopg2 at module level so the lazy import inside the Ledger
doesn't blow up in environments without the driver installed.
"""
from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock

# ── Stub psycopg2 before importing app modules (matches test_control_plane.py) ──
_mock_psycopg2 = MagicMock()
_mock_psycopg2.InterfaceError = type("InterfaceError", (Exception,), {})
_mock_psycopg2.OperationalError = type("OperationalError", (Exception,), {})
sys.modules.setdefault("psycopg2", _mock_psycopg2)
sys.modules.setdefault("psycopg2.pool", MagicMock())

# ── Stub heavy/optional deps (CrewAI, etc.) so config.py imports cleanly ──
for _mod in ("crewai", "crewai.tools", "langchain_anthropic", "docker"):
    if _mod not in sys.modules:
        m = types.ModuleType(_mod)
        if _mod == "crewai.tools":
            m.tool = lambda name: (lambda fn: fn)
        sys.modules[_mod] = m


from app.epistemic import (  # noqa: E402
    LEDGER_MAX_CLAIMS_PER_TASK,
    Claim,
    Evidence,
    Ledger,
    Register,
    VerificationStatus,
    VerifyingAction,
    register_claim_hook,
)
from app.epistemic.ledger import (  # noqa: E402
    DuplicateClaimError,
    LedgerError,
    LedgerFullError,
)
from app.epistemic.registry import _reset_for_tests  # noqa: E402


# ============================================================================
# Evidence
# ============================================================================

class TestEvidence(unittest.TestCase):

    def test_confidence_must_be_in_unit_interval(self):
        with self.assertRaises(ValueError):
            Evidence(kind="tool_call", source_ref="123", excerpt="x", confidence=-0.1)
        with self.assertRaises(ValueError):
            Evidence(kind="tool_call", source_ref="123", excerpt="x", confidence=1.01)

    def test_confidence_at_bounds_ok(self):
        Evidence(kind="tool_call", source_ref="1", excerpt="x", confidence=0.0)
        Evidence(kind="tool_call", source_ref="1", excerpt="x", confidence=1.0)

    def test_long_excerpt_is_truncated(self):
        long_excerpt = "x" * 5000
        e = Evidence(kind="tool_call", source_ref="1", excerpt=long_excerpt, confidence=0.5)
        self.assertLess(len(e.excerpt), 5000)
        self.assertTrue(e.excerpt.endswith("…(truncated)"))

    def test_short_excerpt_unchanged(self):
        e = Evidence(kind="tool_call", source_ref="1", excerpt="hello", confidence=0.5)
        self.assertEqual(e.excerpt, "hello")

    def test_frozen(self):
        e = Evidence(kind="tool_call", source_ref="1", excerpt="x", confidence=0.5)
        with self.assertRaises(Exception):  # FrozenInstanceError
            e.confidence = 0.9  # type: ignore[misc]


# ============================================================================
# VerifyingAction
# ============================================================================

class TestVerifyingAction(unittest.TestCase):

    def test_basic_construction(self):
        va = VerifyingAction(
            tool="readlink",
            args={"path": "/foo"},
            expected_signal="empty=not symlink",
            estimated_seconds=0.5,
        )
        self.assertEqual(va.tool, "readlink")
        self.assertEqual(va.safety, "read_only")

    def test_negative_seconds_rejected(self):
        with self.assertRaises(ValueError):
            VerifyingAction(
                tool="readlink", args={}, expected_signal="x", estimated_seconds=-1.0,
            )

    def test_args_must_be_mapping(self):
        with self.assertRaises(TypeError):
            VerifyingAction(
                tool="readlink",
                args=["not", "a", "mapping"],  # type: ignore[arg-type]
                expected_signal="x",
                estimated_seconds=0.5,
            )

    def test_non_read_only_safety_rejected(self):
        with self.assertRaises(ValueError):
            VerifyingAction(
                tool="rm",
                args={},
                expected_signal="x",
                estimated_seconds=0.5,
                safety="destructive",  # type: ignore[arg-type]
            )


# ============================================================================
# Claim
# ============================================================================

class TestClaim(unittest.TestCase):

    def _new(self, **overrides) -> Claim:
        defaults = dict(
            task_id="task_abc",
            agent_role="researcher",
            statement="the working tree is clean",
            status=VerificationStatus.INFERRED,
            register=Register.DECLARATIVE,
        )
        defaults.update(overrides)
        return Claim.new(**defaults)

    def test_factory_generates_unique_ids(self):
        a, b, c = self._new(), self._new(), self._new()
        self.assertEqual(len({a.claim_id, b.claim_id, c.claim_id}), 3)
        for cid in (a.claim_id, b.claim_id, c.claim_id):
            self.assertTrue(cid.startswith("clm_"))
            self.assertEqual(len(cid), len("clm_") + 12)

    def test_factory_sets_created_at_in_utc(self):
        c = self._new()
        self.assertIsNotNone(c.created_at.tzinfo)
        self.assertEqual(c.created_at.utcoffset().total_seconds(), 0)

    def test_empty_statement_rejected(self):
        with self.assertRaises(ValueError):
            self._new(statement="")

    def test_long_statement_is_truncated(self):
        c = self._new(statement="x" * 8000)
        self.assertLess(len(c.statement), 8000)
        self.assertTrue(c.statement.endswith("…(truncated)"))

    def test_jsonable_roundtrip(self):
        original = self._new(
            evidence=(Evidence(kind="tool_call", source_ref="42",
                               excerpt="output preview", confidence=0.9),),
            verifying_action=VerifyingAction(
                tool="readlink", args={"path": "/x"},
                expected_signal="empty=no", estimated_seconds=0.5,
            ),
            load_bearing=True,
            tags=("filesystem", "config"),
            span_id=42,
        )
        rebuilt = Claim.from_jsonable(original.as_jsonable())
        self.assertEqual(rebuilt.claim_id, original.claim_id)
        self.assertEqual(rebuilt.task_id, original.task_id)
        self.assertEqual(rebuilt.statement, original.statement)
        self.assertEqual(rebuilt.status, original.status)
        self.assertEqual(rebuilt.register, original.register)
        self.assertEqual(len(rebuilt.evidence), 1)
        self.assertEqual(rebuilt.evidence[0].excerpt, "output preview")
        self.assertIsNotNone(rebuilt.verifying_action)
        self.assertEqual(rebuilt.verifying_action.tool, "readlink")
        self.assertEqual(rebuilt.verifying_action.args, {"path": "/x"})
        self.assertTrue(rebuilt.load_bearing)
        self.assertEqual(rebuilt.tags, ("filesystem", "config"))
        self.assertEqual(rebuilt.span_id, 42)

    def test_jsonable_handles_none_verifier(self):
        c = self._new()
        as_json = c.as_jsonable()
        self.assertIsNone(as_json["verifying_action"])
        self.assertIsNone(Claim.from_jsonable(as_json).verifying_action)

    def test_frozen(self):
        c = self._new()
        with self.assertRaises(Exception):  # FrozenInstanceError
            c.statement = "different"  # type: ignore[misc]


# ============================================================================
# Ledger — emission, supersession, queries, hooks
# ============================================================================

class TestLedgerEmission(unittest.TestCase):

    def setUp(self):
        _reset_for_tests()
        self.ledger = Ledger(task_id="task_abc")

    def _claim(self, **overrides) -> Claim:
        defaults = dict(
            task_id="task_abc",
            agent_role="researcher",
            statement="example claim",
            status=VerificationStatus.INFERRED,
            register=Register.DECLARATIVE,
        )
        defaults.update(overrides)
        return Claim.new(**defaults)

    def test_emit_stores_claim(self):
        c = self._claim()
        result = self.ledger.emit(c)
        self.assertIs(result, c)
        self.assertEqual(len(self.ledger), 1)
        self.assertIn(c.claim_id, self.ledger)
        self.assertIs(self.ledger.by_id(c.claim_id), c)

    def test_emit_rejects_empty_task_id(self):
        with self.assertRaises(ValueError):
            Ledger(task_id="")

    def test_emit_rejects_mismatched_task_id(self):
        wrong = Claim.new(
            task_id="task_xyz",  # ledger is task_abc
            agent_role="researcher",
            statement="x",
            status=VerificationStatus.INFERRED,
        )
        with self.assertRaises(ValueError):
            self.ledger.emit(wrong)

    def test_emit_rejects_duplicate_claim_id(self):
        c = self._claim()
        self.ledger.emit(c)
        with self.assertRaises(DuplicateClaimError):
            self.ledger.emit(c)

    def test_emit_enforces_per_task_cap(self):
        # Burn down to one slot remaining via direct dict insertion (avoids
        # 499 emissions and the hook dispatch they'd trigger).
        for i in range(LEDGER_MAX_CLAIMS_PER_TASK):
            placeholder = self._claim(statement=f"placeholder {i}")
            self.ledger._claims[placeholder.claim_id] = placeholder
        self.assertEqual(len(self.ledger), LEDGER_MAX_CLAIMS_PER_TASK)
        with self.assertRaises(LedgerFullError):
            self.ledger.emit(self._claim(statement="overflow"))

    # Path 2 (emit_from_tool_call) shipped in Phase 1 — coverage in
    # test_epistemic_verification.py::TestEmitFromToolCall.
    # Path 3 (emit_from_output_text) shipped in Phase 2 — coverage in
    # test_epistemic_phase2.py::TestEmitFromOutputText.


class TestLedgerSupersession(unittest.TestCase):

    def setUp(self):
        _reset_for_tests()
        self.ledger = Ledger(task_id="task_abc")
        self.original = Claim.new(
            task_id="task_abc",
            agent_role="researcher",
            statement="path is not a symlink",
            status=VerificationStatus.INFERRED,
            register=Register.DECLARATIVE,
            load_bearing=True,
        )
        self.ledger.emit(self.original)

    def test_supersede_flips_status_and_links(self):
        replacement = Claim.new(
            task_id="task_abc",
            agent_role="researcher",
            statement="path IS a symlink (readlink confirmed)",
            status=VerificationStatus.VERIFIED,
            register=Register.DECLARATIVE,
            load_bearing=True,
        )
        contradicted = self.ledger.supersede(
            claim_id=self.original.claim_id, replacement=replacement,
        )
        self.assertEqual(contradicted.status, VerificationStatus.CONTRADICTED)
        self.assertEqual(contradicted.superseded_by, replacement.claim_id)
        self.assertEqual(self.ledger.by_id(self.original.claim_id), contradicted)
        self.assertEqual(self.ledger.by_id(replacement.claim_id), replacement)
        self.assertEqual(len(self.ledger), 2)

    def test_supersede_unknown_id_raises(self):
        replacement = Claim.new(
            task_id="task_abc", agent_role="researcher",
            statement="x", status=VerificationStatus.VERIFIED,
        )
        with self.assertRaises(KeyError):
            self.ledger.supersede(claim_id="clm_doesnotexist", replacement=replacement)

    def test_double_supersession_raises(self):
        replacement = Claim.new(
            task_id="task_abc", agent_role="researcher",
            statement="r1", status=VerificationStatus.VERIFIED,
        )
        self.ledger.supersede(claim_id=self.original.claim_id, replacement=replacement)
        replacement2 = Claim.new(
            task_id="task_abc", agent_role="researcher",
            statement="r2", status=VerificationStatus.VERIFIED,
        )
        with self.assertRaises(LedgerError):
            self.ledger.supersede(claim_id=self.original.claim_id, replacement=replacement2)


class TestLedgerQueries(unittest.TestCase):

    def setUp(self):
        _reset_for_tests()
        self.ledger = Ledger(task_id="task_abc")

        self.verified_load_bearing = Claim.new(
            task_id="task_abc", agent_role="researcher",
            statement="repo is on main", status=VerificationStatus.VERIFIED,
            load_bearing=True,
        )
        self.inferred_load_bearing = Claim.new(
            task_id="task_abc", agent_role="researcher",
            statement="path is not a symlink", status=VerificationStatus.INFERRED,
            load_bearing=True,
        )
        self.assumed_load_bearing = Claim.new(
            task_id="task_abc", agent_role="researcher",
            statement="user wants a fix", status=VerificationStatus.ASSUMED,
            load_bearing=True,
        )
        self.inferred_not_load_bearing = Claim.new(
            task_id="task_abc", agent_role="researcher",
            statement="cosmetic note", status=VerificationStatus.INFERRED,
            load_bearing=False,
        )
        for c in (
            self.verified_load_bearing,
            self.inferred_load_bearing,
            self.assumed_load_bearing,
            self.inferred_not_load_bearing,
        ):
            self.ledger.emit(c)

    def test_load_bearing_filters_correctly(self):
        ids = {c.claim_id for c in self.ledger.load_bearing()}
        self.assertEqual(ids, {
            self.verified_load_bearing.claim_id,
            self.inferred_load_bearing.claim_id,
            self.assumed_load_bearing.claim_id,
        })

    def test_unverified_load_bearing_excludes_verified_and_non_load_bearing(self):
        ids = {c.claim_id for c in self.ledger.unverified_load_bearing()}
        self.assertEqual(ids, {
            self.inferred_load_bearing.claim_id,
            self.assumed_load_bearing.claim_id,
        })

    def test_all_returns_in_emission_order(self):
        ordered = self.ledger.all()
        self.assertEqual([c.claim_id for c in ordered], [
            self.verified_load_bearing.claim_id,
            self.inferred_load_bearing.claim_id,
            self.assumed_load_bearing.claim_id,
            self.inferred_not_load_bearing.claim_id,
        ])


class TestLedgerHooks(unittest.TestCase):

    def setUp(self):
        _reset_for_tests()
        self.ledger = Ledger(task_id="task_abc")
        self.calls: list = []

    def _claim(self) -> Claim:
        return Claim.new(
            task_id="task_abc", agent_role="researcher",
            statement="x", status=VerificationStatus.INFERRED,
        )

    def test_registered_hook_runs_on_emit(self):
        @register_claim_hook
        def hook(claim, ledger):
            self.calls.append((claim.claim_id, ledger.task_id))

        c = self._claim()
        self.ledger.emit(c)
        self.assertEqual(self.calls, [(c.claim_id, "task_abc")])

    def test_misbehaving_hook_does_not_break_emission(self):
        @register_claim_hook
        def bad_hook(claim, ledger):
            raise RuntimeError("oops")

        @register_claim_hook
        def good_hook(claim, ledger):
            self.calls.append(claim.claim_id)

        c = self._claim()
        result = self.ledger.emit(c)  # must not raise
        self.assertIs(result, c)
        self.assertEqual(self.calls, [c.claim_id])

    def test_unregister_removes_hook(self):
        @register_claim_hook
        def hook(claim, ledger):
            self.calls.append(claim.claim_id)

        from app.epistemic.registry import unregister
        unregister(hook)

        self.ledger.emit(self._claim())
        self.assertEqual(self.calls, [])


class TestLedgerFromClaims(unittest.TestCase):

    def setUp(self):
        _reset_for_tests()

    def test_from_claims_does_not_dispatch_hooks(self):
        calls: list = []

        @register_claim_hook
        def hook(claim, ledger):
            calls.append(claim.claim_id)

        c = Claim.new(
            task_id="task_abc", agent_role="researcher",
            statement="x", status=VerificationStatus.VERIFIED,
        )
        ledger = Ledger.from_claims(task_id="task_abc", claims=[c])

        self.assertEqual(len(ledger), 1)
        self.assertIs(ledger.by_id(c.claim_id), c)
        self.assertEqual(calls, [])  # hooks NOT fired

    def test_from_claims_rejects_mismatched_task(self):
        c = Claim.new(
            task_id="task_xyz", agent_role="researcher",
            statement="x", status=VerificationStatus.VERIFIED,
        )
        with self.assertRaises(ValueError):
            Ledger.from_claims(task_id="task_abc", claims=[c])


if __name__ == "__main__":
    unittest.main()
