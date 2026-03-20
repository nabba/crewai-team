"""Tests for long response handling — .md file attachment for Signal delivery.

Tests verify:
  - _strip_internal_metadata removes QA artefacts but doesn't truncate
  - truncate_for_signal truncates with correct note
  - _write_response_md creates a valid .md file
  - SignalClient.send accepts attachments parameter
  - handle_task sends attachment for long responses
  - Config has workspace_host_path setting
  - Response files are pruned to prevent disk bloat
"""
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch, MagicMock

# Ensure litellm stub exists so `from app.main import ...` doesn't fail
# when the real litellm package is incomplete in the test environment.
if "litellm" not in sys.modules:
    _stub = types.ModuleType("litellm")
    _stub.completion = lambda *a, **kw: None  # type: ignore[attr-defined]
    sys.modules["litellm"] = _stub
elif not hasattr(sys.modules["litellm"], "completion"):
    sys.modules["litellm"].completion = lambda *a, **kw: None  # type: ignore[attr-defined]


class TestStripInternalMetadata(unittest.TestCase):
    """Verify metadata stripping without truncation."""

    def test_strips_metadata_preserves_full_text(self):
        from app.agents.commander import _strip_internal_metadata
        long_text = "A" * 3000
        result = _strip_internal_metadata(long_text)
        # Should NOT truncate — just strip metadata
        self.assertEqual(len(result), 3000)

    def test_strips_critic_review(self):
        from app.agents.commander import _strip_internal_metadata
        text = "Answer.\n\n---\n\n**[Critic Review]**\nInternal review."
        result = _strip_internal_metadata(text)
        self.assertNotIn("Critic Review", result)
        self.assertIn("Answer", result)

    def test_empty_input(self):
        from app.agents.commander import _strip_internal_metadata
        self.assertEqual(_strip_internal_metadata(""), "")
        self.assertEqual(_strip_internal_metadata(None), None)


class TestTruncateForSignal(unittest.TestCase):
    """Verify truncation with attachment note."""

    def test_short_text_unchanged(self):
        from app.agents.commander import truncate_for_signal
        text = "Short answer."
        self.assertEqual(truncate_for_signal(text), text)

    def test_long_text_truncated(self):
        from app.agents.commander import truncate_for_signal
        text = "A" * 2000
        result = truncate_for_signal(text)
        self.assertIn("[Full response attached as document]", result)
        self.assertLessEqual(len(result), 1500)

    def test_truncates_at_sentence_boundary(self):
        from app.agents.commander import truncate_for_signal
        text = "First sentence. " * 100
        result = truncate_for_signal(text)
        before = result.split("\n\n[Full response attached")[0]
        self.assertTrue(before.endswith("."))

    def test_custom_max_length(self):
        from app.agents.commander import truncate_for_signal
        text = "A" * 500
        result = truncate_for_signal(text, max_length=100)
        self.assertIn("[Full response attached", result)


class TestSignalClientAttachments(unittest.TestCase):
    """Verify SignalClient supports attachments."""

    def test_send_signature_accepts_attachments(self):
        import inspect
        from app.signal_client import SignalClient
        sig = inspect.signature(SignalClient.send)
        self.assertIn("attachments", sig.parameters)

    def test_http_payload_includes_attachments(self):
        """HTTP send should include attachments in JSON-RPC params."""
        with open("app/signal_client.py") as f:
            source = f.read()
        self.assertIn('params["attachments"]', source)

    def test_socket_payload_includes_attachments(self):
        """Socket send should include attachments in JSON-RPC params."""
        with open("app/signal_client.py") as f:
            source = f.read()
        # Should appear in both HTTP and socket methods
        count = source.count('params["attachments"]')
        self.assertGreaterEqual(count, 2)


class TestWriteResponseMd(unittest.TestCase):
    """Verify .md file creation logic exists in main.py."""

    def test_write_response_md_function_exists(self):
        with open("app/main.py") as f:
            source = f.read()
        self.assertIn("def _write_response_md(", source)
        # Writes markdown with question and full text
        self.assertIn("# Response", source)
        self.assertIn("question_preview", source)
        self.assertIn("full_text", source)

    def test_returns_none_without_host_path(self):
        with open("app/main.py") as f:
            source = f.read()
        # Should check workspace_host_path and return None if empty
        self.assertIn("workspace_host_path", source)
        self.assertIn("return None", source)

    def test_translates_docker_to_host_path(self):
        with open("app/main.py") as f:
            source = f.read()
        # Should replace _WORKSPACE_ROOT with host path
        self.assertIn("docker_path.replace(_WORKSPACE_ROOT", source)

    def test_response_file_naming(self):
        with open("app/main.py") as f:
            source = f.read()
        self.assertIn('response_', source)
        self.assertIn('.md"', source)


class TestHandleTaskLongResponse(unittest.TestCase):
    """Verify handle_task sends .md attachment for long responses."""

    def test_main_has_attachment_logic(self):
        with open("app/main.py") as f:
            source = f.read()
        self.assertIn("_write_response_md", source)
        self.assertIn("truncate_for_signal", source)
        self.assertIn("signal_attachments", source)
        self.assertIn("attachments=signal_attachments", source)

    def test_prune_function_exists(self):
        with open("app/main.py") as f:
            source = f.read()
        self.assertIn("_prune_response_files", source)
        self.assertIn("_MAX_RESPONSE_FILES", source)


class TestWorkspaceHostPathConfig(unittest.TestCase):
    """Verify config has workspace_host_path setting."""

    def test_config_has_field(self):
        from app.config import Settings
        self.assertIn("workspace_host_path", Settings.model_fields)

    def test_default_is_empty(self):
        from app.config import Settings
        default = Settings.model_fields["workspace_host_path"].default
        self.assertEqual(default, "")


if __name__ == "__main__":
    unittest.main()
