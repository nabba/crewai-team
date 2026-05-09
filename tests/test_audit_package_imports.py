"""Regression: ``app.audit`` must export every legacy symbol from
the pre-package ``app/audit.py`` module — same class of bug as T3.5
in ``app/utils``.

Pre-fix shape (the operator-reported gateway crashloop, 2026-05-10):

  ImportError: cannot import name 'log_tool_blocked' from 'app.audit'
  (/app/app/audit/__init__.py)

  Two things existed at the same import path:
    - app/audit.py             (legacy module — log_request_received,
                                 log_response_sent, log_tool_call,
                                 log_tool_blocked, log_security_event,
                                 log_crew_dispatch, audit_logger)
    - app/audit/__init__.py    (new package — RolledLogStore primitive)

  Python prefers packages over modules with the same name, so
  ``from app.audit import log_tool_blocked`` failed at module-load
  time. The chain that surfaced it:

    main.py:95 → from app.agents.commander import Commander
    commander/__init__.py → from .orchestrator import Commander
    orchestrator.py:12 → from app.tools.attachment_reader import …
    attachment_reader.py:14 → from app.audit import log_tool_blocked
    ImportError → uvicorn fails → container crashloop

  All inbound traffic was blocked.

Post-fix:
  app/audit.py is deleted; its contents are merged into
  app/audit/__init__.py. Both surfaces (legacy event-log helpers AND
  the rolled-log primitive) are now exported from the package.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent


# ── Layout contracts ────────────────────────────────────────────────


class TestNoShadowedAuditModule:
    """The legacy module file MUST NOT come back. If a future PR
    re-adds app/audit.py, Python will silently shadow the package
    again — same gateway-crashloop symptom as 2026-05-10."""

    def test_app_audit_py_does_not_exist(self) -> None:
        legacy = _REPO_ROOT / "app" / "audit.py"
        assert not legacy.exists(), (
            f"{legacy} re-introduced — Python would shadow the "
            f"app/audit/ package and the gateway would crashloop on "
            f"`from app.audit import log_*`."
        )

    def test_app_audit_package_exists(self) -> None:
        pkg_init = _REPO_ROOT / "app" / "audit" / "__init__.py"
        assert pkg_init.exists()


# ── Legacy event-log helpers ───────────────────────────────────────


class TestLegacyEventLogHelpers:
    """All six event-log helpers from the legacy audit.py must be
    importable from the package."""

    def test_log_request_received_importable(self) -> None:
        from app.audit import log_request_received
        # Accepts (sender_redacted, message_length).
        assert callable(log_request_received)
        sig = inspect.signature(log_request_received)
        assert list(sig.parameters) == ["sender_redacted", "message_length"]

    def test_log_response_sent_importable(self) -> None:
        from app.audit import log_response_sent
        assert callable(log_response_sent)

    def test_log_crew_dispatch_importable(self) -> None:
        from app.audit import log_crew_dispatch
        assert callable(log_crew_dispatch)

    def test_log_tool_call_importable(self) -> None:
        from app.audit import log_tool_call
        assert callable(log_tool_call)

    def test_log_tool_blocked_importable(self) -> None:
        """The exact symbol whose ImportError triggered the
        2026-05-10 gateway crashloop."""
        from app.audit import log_tool_blocked
        assert callable(log_tool_blocked)

    def test_log_security_event_importable(self) -> None:
        from app.audit import log_security_event
        assert callable(log_security_event)

    def test_audit_logger_importable(self) -> None:
        """The shared logger instance (originally a module-level
        global in app/audit.py) must remain accessible — it's the
        sink configured in main.py."""
        from app.audit import audit_logger
        assert audit_logger.name == "crewai.audit"


# ── Rolled-log primitive (original package surface) ───────────────


class TestRolledLogPrimitivePreserved:
    """The hash-chain audit storage that was the package's reason to
    exist must still be re-exported."""

    def test_rolled_log_store_importable(self) -> None:
        from app.audit import RolledLogStore
        assert RolledLogStore is not None

    def test_rolled_log_reader_importable(self) -> None:
        from app.audit import RolledLogReader
        assert RolledLogReader is not None

    def test_rolled_log_verifier_importable(self) -> None:
        from app.audit import RolledLogVerifier
        assert RolledLogVerifier is not None

    def test_genesis_constant_importable(self) -> None:
        from app.audit import GENESIS
        assert GENESIS is not None


# ── Critical top-level import sites ────────────────────────────────


class TestTopLevelImportSitesLoad:
    """The crash chain was main.py → commander → orchestrator →
    attachment_reader → app.audit. Each link must import cleanly."""

    def test_attachment_reader_imports(self) -> None:
        # tools/attachment_reader.py:14 had:
        #   from app.audit import log_tool_blocked
        # at module top-level → broke uvicorn at startup.
        import app.tools.attachment_reader  # noqa: F401

    def test_web_fetch_imports(self) -> None:
        # tools/web_fetch.py:8 has the same top-level import.
        import app.tools.web_fetch  # noqa: F401

    def test_file_manager_imports(self) -> None:
        # tools/file_manager.py:3 has the same top-level import.
        import app.tools.file_manager  # noqa: F401


# ── Functional smoke: the helpers run without raising ─────────────


class TestHelpersRunWithoutRaising:
    """The handlers do real I/O (logging.info). Smoke-test that they
    don't blow up when called with reasonable inputs — guards
    against silent breakage of the dispatch path."""

    def test_log_tool_blocked_runs(self, caplog) -> None:
        import logging
        from app.audit import log_tool_blocked
        with caplog.at_level(logging.INFO, logger="crewai.audit"):
            log_tool_blocked("file_manager", "researcher", "path traversal")
        # Find the one record with our event.
        records = [r for r in caplog.records if r.name == "crewai.audit"]
        assert len(records) == 1
        assert "tool_blocked" in records[0].message
        assert "file_manager" in records[0].message
