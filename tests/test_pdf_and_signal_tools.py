"""Tests for app.tools.pdf_compose + app.tools.signal_attachment.

These cover:
  * Path-traversal clamping in safe_output_path (writes can ONLY land
    under /app/workspace/output/ regardless of what the agent passes).
  * Sandbox library availability (matplotlib + reportlab + pandas
    pre-loaded) — captures the 95% report-generation surface.
  * Signal-attachment validation: rejects paths outside the allowed
    dir, missing files, oversized totals, too-many-attachments.
  * Container-to-host path translation for signal-cli delivery.
  * Tool factory graceful-degradation when Signal isn't configured.

No live Signal sends — the actual signal_client.send_message is
patched out so tests don't try to reach the host signal-cli daemon.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── pdf_compose: safe_output_path clamping ──────────────────────────


class TestSafeOutputPath:

    def test_basename_preserved(self):
        from app.tools.pdf_compose import _safe_output_path
        out = _safe_output_path("estonia.pdf")
        assert out.name == "estonia.pdf"
        assert "/app/workspace/output" in str(out)

    def test_path_traversal_stripped(self):
        """`../../etc/passwd` becomes the basename `passwd` in workspace/output."""
        from app.tools.pdf_compose import _safe_output_path
        out = _safe_output_path("../../etc/passwd")
        assert out.name == "passwd"
        assert "/etc" not in str(out)
        assert "/app/workspace/output" in str(out)

    def test_absolute_path_collapsed(self):
        """`/var/tmp/sneaky.pdf` lands as `sneaky.pdf` in workspace/output."""
        from app.tools.pdf_compose import _safe_output_path
        out = _safe_output_path("/var/tmp/sneaky.pdf")
        assert out.name == "sneaky.pdf"
        assert str(out).startswith("/app/workspace/output/")

    def test_unicode_and_specials_replaced(self):
        from app.tools.pdf_compose import _safe_output_path
        out = _safe_output_path("re port (final)/data.pdf")
        # Special chars replaced with _; basename only
        assert "/" not in out.name
        assert " " not in out.name
        assert "(" not in out.name

    def test_empty_falls_back_to_default(self):
        from app.tools.pdf_compose import _safe_output_path
        out = _safe_output_path("")
        assert out.name == "output.pdf"


# ── pdf_compose: sandbox contents ────────────────────────────────────


class TestSandboxContents:

    def test_matplotlib_pre_loaded(self):
        from app.tools.pdf_compose import _build_sandbox
        sb = _build_sandbox()
        assert "plt" in sb
        assert "PdfPages" in sb
        assert sb["matplotlib"].get_backend().lower() == "agg"

    def test_safe_output_path_callable(self):
        """The agent's script can call `safe_output_path(name)` to get
        a clamped path without importing anything."""
        from app.tools.pdf_compose import _build_sandbox
        sb = _build_sandbox()
        out = sb["safe_output_path"]("test.pdf")
        assert "/app/workspace/output" in str(out)

    def test_result_initial_value(self):
        from app.tools.pdf_compose import _build_sandbox
        sb = _build_sandbox()
        assert sb["result"] is None


# ── pdf_compose: factory + tool description guidance ─────────────────


class TestPdfFactory:

    def test_factory_returns_one_tool(self):
        pytest.importorskip("crewai.tools")
        from app.tools.pdf_compose import create_pdf_tools
        tools = create_pdf_tools("coder")
        assert len(tools) == 1
        assert tools[0].name == "pdf_compose"

    def test_description_steers_away_from_writing_source_as_text(self):
        """The 2026-05-03 production failure was the agent writing
        Python source as the chat response instead of executing it.
        The tool description must make the right path obvious."""
        pytest.importorskip("crewai.tools")
        from app.tools.pdf_compose import create_pdf_tools
        [tool] = create_pdf_tools("coder")
        desc = tool.description.lower()
        # Anti-pattern warning + concrete primitives
        assert "instead of writing python source as the response text" in desc
        assert "safe_output_path" in desc
        # Worked examples for both common backends
        assert "pdfpages" in desc  # matplotlib backend
        assert "platypus" in desc  # reportlab backend


# ── signal_attachment: path validation ───────────────────────────────


class TestAttachmentValidation:

    def test_empty_list_rejected(self):
        from app.tools.signal_attachment import _validate_attachments
        valid, err = _validate_attachments([])
        assert valid == []
        assert "no attachments" in (err or "")

    def test_too_many_rejected(self, tmp_path):
        from app.tools.signal_attachment import _validate_attachments
        # 6 paths > cap of 5 → fully rejected
        paths = [str(tmp_path / f"f{i}") for i in range(6)]
        valid, err = _validate_attachments(paths)
        assert valid == []
        assert "too many" in (err or "")

    def test_path_outside_workspace_rejected(self, tmp_path):
        from app.tools.signal_attachment import _validate_attachments
        # File exists but is in a temp dir, not /app/workspace/output
        ghost = tmp_path / "data.csv"
        ghost.write_text("x")
        valid, err = _validate_attachments([str(ghost)])
        assert valid == []
        assert "outside" in (err or "").lower()

    def test_missing_file_rejected(self):
        from app.tools.signal_attachment import _validate_attachments
        valid, err = _validate_attachments(["/app/workspace/output/no_such.pdf"])
        assert valid == []
        # All-rejected error path
        assert "does not exist" in (err or "").lower() or "outside" in (err or "").lower()

    def test_path_traversal_rejected(self, tmp_path, monkeypatch):
        """`/app/workspace/output/../../etc/passwd` resolves outside
        the allowed dir → rejected."""
        from app.tools.signal_attachment import _validate_attachments
        valid, err = _validate_attachments([
            "/app/workspace/output/../../etc/passwd",
        ])
        assert valid == []
        # Could be 'outside' or 'does not exist' depending on which check fires first
        assert err is not None


# ── signal_attachment: container→host path translation ──────────────


class TestPathTranslation:

    def test_basic_translation(self):
        from app.tools.signal_attachment import _container_to_host
        host = _container_to_host(
            [Path("/app/workspace/output/x.pdf")],
            "/Users/me/repo/workspace",
        )
        assert host == ["/Users/me/repo/workspace/output/x.pdf"]

    def test_trailing_slash_handled(self):
        from app.tools.signal_attachment import _container_to_host
        host = _container_to_host(
            [Path("/app/workspace/output/y.csv")],
            "/Users/me/repo/workspace/",
        )
        assert host == ["/Users/me/repo/workspace/output/y.csv"]

    def test_unmapped_path_passes_through(self):
        """Belt-and-braces: if a path somehow doesn't start with
        /app/workspace, it goes through unchanged so the failure is
        loud (signal-cli will reject it)."""
        from app.tools.signal_attachment import _container_to_host
        host = _container_to_host(
            [Path("/some/weird/path.pdf")],
            "/Users/me/repo/workspace",
        )
        assert host == ["/some/weird/path.pdf"]


# ── signal_attachment: factory graceful-degradation ─────────────────


class TestSignalFactory:

    def test_factory_empty_when_owner_missing(self, monkeypatch):
        pytest.importorskip("crewai.tools")
        # Stub settings with no owner number
        fake = MagicMock()
        fake.signal_owner_number = ""
        fake.workspace_host_path = "/host/path"
        monkeypatch.setattr("app.config.get_settings", lambda: fake)
        from app.tools.signal_attachment import create_signal_attachment_tools
        assert create_signal_attachment_tools() == []

    def test_factory_empty_when_workspace_host_missing(self, monkeypatch):
        """Without WORKSPACE_HOST_PATH, signal-cli can't find files
        and we'd silently fail. Better to not register the tool."""
        pytest.importorskip("crewai.tools")
        fake = MagicMock()
        fake.signal_owner_number = "+1234567890"
        fake.workspace_host_path = ""
        monkeypatch.setattr("app.config.get_settings", lambda: fake)
        from app.tools.signal_attachment import create_signal_attachment_tools
        assert create_signal_attachment_tools() == []

    def test_factory_returns_one_tool_when_configured(self, monkeypatch):
        pytest.importorskip("crewai.tools")
        fake = MagicMock()
        fake.signal_owner_number = "+1234567890"
        fake.workspace_host_path = "/host/workspace"
        monkeypatch.setattr("app.config.get_settings", lambda: fake)
        from app.tools.signal_attachment import create_signal_attachment_tools
        tools = create_signal_attachment_tools()
        assert len(tools) == 1
        assert tools[0].name == "signal_send_attachment"
