"""
Tests for security audit hardening fixes.

Covers:
  1. Forwarder buffer limit
  2. web_fetch iterative chunk download + HTTPS-only
  3. auto_deployer dangerous import blocking
  4. conversation_store HMAC — no weak fallback
  5. self_heal input sanitization + verbose=False
  6. signal_client sequential chunk delivery
  7. evolution SHA256 (not MD5)
"""

import ast
import hashlib
import importlib
import inspect
import textwrap
import unittest
from unittest.mock import MagicMock, patch


class TestForwarderBufferLimit(unittest.TestCase):
    """#1 — forwarder.py must cap its buffer to prevent OOM."""

    def test_max_buffer_constant_exists(self):
        source = open("signal/forwarder.py").read()
        self.assertIn("_MAX_BUFFER_BYTES", source)

    def test_buffer_overflow_handled(self):
        source = open("signal/forwarder.py").read()
        self.assertIn("_MAX_BUFFER_BYTES", source)
        self.assertIn("discarding", source.lower())


class TestWebFetchChunkedDownload(unittest.TestCase):
    """#2 — web_fetch must use iterative chunk reading, not response.content."""

    def test_uses_iter_content(self):
        source = open("app/tools/web_fetch.py").read()
        self.assertIn("iter_content", source,
                       "web_fetch should use iter_content() for size-limited streaming")

    def test_does_not_use_response_content(self):
        source = open("app/tools/web_fetch.py").read()
        # The old pattern was response.content[:_MAX_RESPONSE_BYTES]
        self.assertNotIn("response.content[:", source,
                         "Should not use response.content slice (downloads everything into RAM)")

    def test_https_only(self):
        source = open("app/tools/web_fetch.py").read()
        # Should only allow HTTPS
        self.assertIn('"https"', source)
        # Should NOT have "http" in the allowed schemes set (but may appear in comments)
        from app.tools.web_fetch import _ALLOWED_SCHEMES
        self.assertEqual(_ALLOWED_SCHEMES, {"https"},
                         "External fetch should only allow HTTPS")


class TestAutoDeployerImportCheck(unittest.TestCase):
    """#3 — auto_deployer must block dangerous imports in LLM-generated code."""

    def test_blocked_imports_set_exists(self):
        from app.auto_deployer import _BLOCKED_IMPORTS
        self.assertIn("subprocess", _BLOCKED_IMPORTS)
        self.assertIn("pickle", _BLOCKED_IMPORTS)
        self.assertIn("socket", _BLOCKED_IMPORTS)
        self.assertIn("ctypes", _BLOCKED_IMPORTS)

    def test_blocks_subprocess_import(self):
        from app.auto_deployer import _check_dangerous_imports
        code = "import subprocess\nsubprocess.run(['ls'])"
        tree = ast.parse(code)
        violations = _check_dangerous_imports(tree)
        self.assertTrue(any("subprocess" in v for v in violations))

    def test_blocks_eval_call(self):
        from app.auto_deployer import _check_dangerous_imports
        code = "x = eval('1+1')"
        tree = ast.parse(code)
        violations = _check_dangerous_imports(tree)
        self.assertTrue(any("eval" in v for v in violations))

    def test_blocks_exec_call(self):
        from app.auto_deployer import _check_dangerous_imports
        code = "exec('import os')"
        tree = ast.parse(code)
        violations = _check_dangerous_imports(tree)
        self.assertTrue(any("exec" in v for v in violations))

    def test_blocks_from_import(self):
        from app.auto_deployer import _check_dangerous_imports
        code = "from pickle import loads"
        tree = ast.parse(code)
        violations = _check_dangerous_imports(tree)
        self.assertTrue(any("pickle" in v for v in violations))

    def test_allows_safe_imports(self):
        from app.auto_deployer import _check_dangerous_imports
        code = textwrap.dedent("""\
            import json
            import logging
            from pathlib import Path
            from datetime import datetime
        """)
        tree = ast.parse(code)
        violations = _check_dangerous_imports(tree)
        self.assertEqual(violations, [])

    def test_blocks_dunder_import(self):
        from app.auto_deployer import _check_dangerous_imports
        code = "__import__('os')"
        tree = ast.parse(code)
        violations = _check_dangerous_imports(tree)
        self.assertTrue(any("__import__" in v for v in violations))


class TestConversationStoreHMAC(unittest.TestCase):
    """#4 — sender ID HMAC must not use a weak fallback key."""

    def test_no_hardcoded_fallback_key(self):
        source = open("app/conversation_store.py").read()
        # The old weak pattern was: key = b"fallback"
        self.assertNotIn('key = b"fallback"', source,
                         "Must not use hardcoded fallback HMAC key")

    def test_ephemeral_key_on_failure(self):
        source = open("app/conversation_store.py").read()
        self.assertIn("secrets.token_bytes", source,
                       "Should use cryptographic random key as ephemeral fallback")


class TestSelfHealSanitization(unittest.TestCase):
    """#5 — self_heal must sanitize user input in diagnosis tasks."""

    def test_imports_sanitize_input(self):
        source = open("app/self_heal.py").read()
        self.assertIn("from app.sanitize import sanitize_input", source)

    def test_uses_sanitize_input(self):
        source = open("app/self_heal.py").read()
        self.assertIn("sanitize_input(", source,
                       "Diagnosis task must sanitize user input")

    def test_verbose_false_on_agent(self):
        source = open("app/self_heal.py").read()
        # All verbose= should be False
        self.assertNotIn("verbose=True", source,
                         "Diagnosis agent/crew must use verbose=False to avoid leaking data")


class TestSignalClientOrdering(unittest.TestCase):
    """#6 — signal_client must send chunks sequentially."""

    def test_no_gather_for_chunks(self):
        source = open("app/signal_client.py").read()
        self.assertNotIn("asyncio.gather", source,
                         "Chunks must not be sent with gather (no ordering guarantee)")


class TestEvolutionHash(unittest.TestCase):
    """#7 — evolution.py must use SHA256, not MD5."""

    def test_uses_sha256(self):
        source = open("app/evolution.py").read()
        self.assertIn("sha256", source)
        self.assertNotIn("md5", source.lower().replace("memory.md", ""),
                         "Should not use MD5 for hashing")


if __name__ == "__main__":
    unittest.main()
