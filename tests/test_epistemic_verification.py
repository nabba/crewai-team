"""Tests for app.epistemic.verification (Phase 1) and Ledger.emit_from_tool_call."""
from __future__ import annotations

import sys
import textwrap
import types
import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import MagicMock

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
    Ledger,
    Register,
    VerificationStatus,
)
from app.epistemic.registry import _reset_for_tests  # noqa: E402
from app.epistemic.verification import (  # noqa: E402
    DESTRUCTIVE_TOOL_NAMES,
    VerifierRegistryLoadError,
    _VerifierRegistry,
    _reset_for_tests as _reset_registry,
)


def _yaml_path(content: str) -> Path:
    f = NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    f.write(textwrap.dedent(content))
    f.close()
    return Path(f.name)


# ============================================================================
# Loader: structural validation
# ============================================================================

class TestRegistryLoader(unittest.TestCase):

    def test_default_registry_loads(self):
        """The shipped YAML must load cleanly and contain all 10 starters."""
        _reset_registry()
        from app.epistemic.verification import VERIFIER_REGISTRY
        registry = VERIFIER_REGISTRY()
        self.assertGreaterEqual(len(registry), 10)
        ids = {s.id for s in registry}
        # Spot-check the most important ones from the reference incident.
        self.assertIn("filesystem.is_symlink", ids)
        self.assertIn("git.is_clean_tree", ids)
        self.assertIn("git.commit_exists", ids)

    def test_missing_top_level_key_rejected(self):
        path = _yaml_path("not_verifiers: []")
        with self.assertRaises(VerifierRegistryLoadError):
            _VerifierRegistry.load_from(path)

    def test_duplicate_id_rejected(self):
        path = _yaml_path("""
            verifiers:
              - id: foo
                tool: stat
                matches: { claim_pattern: "x" }
              - id: foo
                tool: stat
                matches: { claim_pattern: "y" }
        """)
        with self.assertRaises(VerifierRegistryLoadError) as ctx:
            _VerifierRegistry.load_from(path)
        self.assertIn("duplicate", str(ctx.exception).lower())

    def test_destructive_tool_rejected(self):
        # "rm" is in DESTRUCTIVE_TOOL_NAMES.
        path = _yaml_path("""
            verifiers:
              - id: bad
                tool: rm -rf
                matches: { claim_pattern: "delete (.+)" }
        """)
        with self.assertRaises(VerifierRegistryLoadError) as ctx:
            _VerifierRegistry.load_from(path)
        self.assertIn("destructive", str(ctx.exception).lower())

    def test_destructive_tool_set_is_frozen(self):
        # The list of destructive tools is part of the safety boundary.
        # If this test fails, someone tried to widen the gate at runtime.
        with self.assertRaises(AttributeError):
            DESTRUCTIVE_TOOL_NAMES.add("readlink")  # type: ignore[attr-defined]

    def test_invalid_regex_rejected(self):
        path = _yaml_path("""
            verifiers:
              - id: bad
                tool: stat
                matches: { claim_pattern: "[unclosed" }
        """)
        with self.assertRaises(VerifierRegistryLoadError):
            _VerifierRegistry.load_from(path)

    def test_template_extractor_requires_template(self):
        path = _yaml_path("""
            verifiers:
              - id: bad
                tool: psql
                matches: { claim_pattern: "table (\\w+)" }
                arg_extractor:
                  kind: template
                  groups: { table: 1 }
        """)
        with self.assertRaises(VerifierRegistryLoadError):
            _VerifierRegistry.load_from(path)

    def test_negative_estimated_seconds_rejected(self):
        path = _yaml_path("""
            verifiers:
              - id: bad
                tool: stat
                matches: { claim_pattern: "x" }
                estimated_seconds: -1.0
        """)
        with self.assertRaises(VerifierRegistryLoadError):
            _VerifierRegistry.load_from(path)


# ============================================================================
# Matcher: claim shape → verifier
# ============================================================================

class TestRegistryMatch(unittest.TestCase):

    def setUp(self):
        _reset_registry()
        from app.epistemic.verification import VERIFIER_REGISTRY
        self.registry = VERIFIER_REGISTRY()

    def test_symlink_claim_matches(self):
        va = self.registry.match("/etc/foo is not a symlink")
        self.assertIsNotNone(va)
        self.assertEqual(va.tool, "readlink")
        self.assertIn("path", va.args)

    def test_clean_tree_claim_matches(self):
        va = self.registry.match("the working tree is clean")
        self.assertIsNotNone(va)
        self.assertEqual(va.tool, "git status --porcelain")

    def test_commit_exists_extracts_sha(self):
        va = self.registry.match("commit deadbeef0123 exists")
        self.assertIsNotNone(va)
        self.assertEqual(va.args.get("rev"), "deadbeef0123")

    def test_template_extractor_renders_sql(self):
        va = self.registry.match("table tickets has 4200 rows")
        self.assertIsNotNone(va)
        self.assertEqual(va.tool, "psql -c")
        self.assertIn("FROM tickets", va.args["sql"])

    def test_no_match_returns_none(self):
        va = self.registry.match("the agent is being thoughtful today")
        self.assertIsNone(va)

    def test_cheapest_verifier_wins_when_multiple_match(self):
        # Build a tiny test registry where two shapes match the same claim
        # at different costs. The cheaper one must win.
        path = _yaml_path("""
            verifiers:
              - id: slow
                tool: stat
                matches: { claim_pattern: "(.+) exists" }
                arg_extractor: { kind: regex_capture, groups: { path: 1 } }
                estimated_seconds: 5.0
              - id: fast
                tool: ls
                matches: { claim_pattern: "(.+) exists" }
                arg_extractor: { kind: regex_capture, groups: { path: 1 } }
                estimated_seconds: 0.1
        """)
        registry = _VerifierRegistry.load_from(path)
        va = registry.match("/tmp/foo exists")
        self.assertIsNotNone(va)
        self.assertEqual(va.tool, "ls")  # fast won

    def test_tags_filter_applies(self):
        # Build registry with one tagged shape; matching by claim alone
        # works, but if tags are supplied that don't include the shape's
        # tag, the shape must NOT match.
        path = _yaml_path("""
            verifiers:
              - id: tagged
                tool: stat
                matches:
                  claim_pattern: "(.+) is a symlink"
                  tags_any: [filesystem]
                arg_extractor: { kind: regex_capture, groups: { path: 1 } }
                estimated_seconds: 0.5
        """)
        registry = _VerifierRegistry.load_from(path)
        # Caller didn't tag → still matches (no filter).
        self.assertIsNotNone(registry.match("/x is a symlink"))
        # Caller tagged with relevant tag → matches.
        self.assertIsNotNone(registry.match("/x is a symlink", tags=("filesystem",)))
        # Caller tagged with irrelevant tag → does NOT match.
        self.assertIsNone(registry.match("/x is a symlink", tags=("git",)))


# ============================================================================
# Path 2: Ledger.emit_from_tool_call
# ============================================================================

class TestEmitFromToolCall(unittest.TestCase):

    def setUp(self):
        _reset_for_tests()  # clear claim hooks
        _reset_registry()
        self.ledger = Ledger(task_id="task_abc")

    def test_basic_path_2_emission(self):
        c = self.ledger.emit_from_tool_call(
            agent_role="researcher",
            tool_name="ls",
            tool_args={"-la": "/etc"},
            tool_output="drwxr-xr-x  2 root  root  4096 Apr 30",
            agent_inference="/etc is not a symlink",
            register=Register.DECLARATIVE,
            load_bearing=True,
            tags=("filesystem",),
        )
        self.assertEqual(c.statement, "/etc is not a symlink")
        # Path 2 always defaults to INFERRED — adjacent observation, not exact.
        self.assertEqual(c.status, VerificationStatus.INFERRED)
        self.assertEqual(c.register, Register.DECLARATIVE)
        self.assertTrue(c.load_bearing)
        self.assertEqual(c.tags, ("filesystem",))

    def test_path_2_attaches_verifier_when_registry_matches(self):
        c = self.ledger.emit_from_tool_call(
            agent_role="researcher",
            tool_name="ls",
            tool_args={"path": "/etc"},
            tool_output="drwxr-xr-x …",
            agent_inference="/etc is not a symlink",
        )
        self.assertIsNotNone(c.verifying_action)
        self.assertEqual(c.verifying_action.tool, "readlink")
        self.assertEqual(c.verifying_action.args.get("path"), "/etc")

    def test_path_2_no_verifier_when_no_registry_match(self):
        c = self.ledger.emit_from_tool_call(
            agent_role="researcher",
            tool_name="ls",
            tool_args={},
            tool_output="…",
            agent_inference="this codebase has good vibes",
        )
        self.assertIsNone(c.verifying_action)

    def test_path_2_evidence_records_invocation_and_output(self):
        c = self.ledger.emit_from_tool_call(
            agent_role="coder",
            tool_name="git",
            tool_args={"cmd": "status --porcelain"},
            tool_output="M file.py",
            agent_inference="the working tree is clean",
        )
        self.assertEqual(len(c.evidence), 1)
        ev = c.evidence[0]
        self.assertEqual(ev.kind, "tool_call")
        self.assertIn("git", ev.excerpt)
        self.assertIn("M file.py", ev.excerpt)

    def test_path_2_evidence_confidence_default(self):
        c = self.ledger.emit_from_tool_call(
            agent_role="researcher",
            tool_name="readlink",
            tool_args={"path": "/etc"},
            tool_output="",
            agent_inference="/etc is not a symlink",
        )
        # Default 0.6 — adjacent observation, not exact-answer.
        self.assertAlmostEqual(c.evidence[0].confidence, 0.6)

    def test_path_2_with_explicit_high_confidence(self):
        c = self.ledger.emit_from_tool_call(
            agent_role="researcher",
            tool_name="readlink",
            tool_args={"path": "/etc"},
            tool_output="",
            agent_inference="/etc is not a symlink",
            evidence_confidence=1.0,
        )
        self.assertAlmostEqual(c.evidence[0].confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
