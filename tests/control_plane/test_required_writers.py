"""Tests pinning the required-write semantic for the 4 high-value writer modules.

PR 3 (2026-05-16). Before this PR, every state-transition write
silently swallowed DB errors and returned ``None`` / ``False`` /
``{}`` / no exception. Callers couldn't distinguish "no matching row"
from "DB unreachable". This batch converted the load-bearing writes
to ``execute_required`` / ``execute_one_required``.

The contract these tests pin:

* ``AuditTrail.log`` still doesn't propagate to its caller (fire-and-
  forget by contract) BUT the inner ``execute_required`` raises,
  the catch fires, and the prominent ``AUDIT WRITE FAILED`` log
  finally surfaces in operator-visible logs.
* ``GovernanceGate.request_approval`` / ``approve`` / ``reject``
  propagate ``DBUnavailable`` and ``psycopg2.Error`` to callers
  (was previously returning empty dict / False).
* ``TicketManager`` writes (INSERTs + UPDATEs for state transitions)
  propagate DB errors.
* ``error_monitor._record_anomaly`` logs at WARNING (was DEBUG/silent);
  ``acknowledge`` returns False on DB failure (was previously True).
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import psycopg2
import pytest


# ── Helpers ──────────────────────────────────────────────────────────


def _patch_db(monkeypatch, module_path: str, *,
              required_raises: Exception | None = None,
              required_returns: list | None = None):
    """Patch ``execute_required`` on a target module.

    ``required_raises``: exception instance to raise on call.
    ``required_returns``: list to return on call.
    """
    def _fake(query, params=(), fetch=False):
        if required_raises is not None:
            raise required_raises
        return required_returns or []
    monkeypatch.setattr(f"{module_path}.execute_required", _fake)


# ── AuditTrail ───────────────────────────────────────────────────────


class TestAuditLogRequired:
    """AuditTrail.log: fire-and-forget contract preserved, but the
    prominent ERROR log now actually fires on DB failure."""

    def test_log_no_longer_silently_loses_audit_rows(self, monkeypatch, caplog):
        """A DB failure during audit.log MUST hit the AUDIT WRITE FAILED
        log line — the whole point of the conversion."""
        from app.control_plane.db import DBUnavailable
        from app.control_plane.audit import AuditTrail

        _patch_db(
            monkeypatch, "app.control_plane.audit",
            required_raises=DBUnavailable("pool down"),
        )

        with caplog.at_level(logging.ERROR, logger="app.control_plane.audit"):
            AuditTrail().log(actor="user", action="test.action")

        assert any(
            "AUDIT WRITE FAILED" in record.message
            for record in caplog.records
        )

    def test_log_does_not_propagate_to_caller(self, monkeypatch):
        """Fire-and-forget contract: log() must not raise to the caller."""
        from app.control_plane.audit import AuditTrail

        _patch_db(
            monkeypatch, "app.control_plane.audit",
            required_raises=psycopg2.Error("table missing"),
        )
        # No exception escapes
        AuditTrail().log(actor="user", action="test.action")

    def test_log_success_path_calls_execute_required(self, monkeypatch):
        """Happy path: execute_required is called with INSERT SQL."""
        from app.control_plane.audit import AuditTrail

        calls = []
        def _fake(query, params=(), fetch=False):
            calls.append(query)
            return []
        monkeypatch.setattr(
            "app.control_plane.audit.execute_required", _fake,
        )
        AuditTrail().log(actor="user", action="x")
        assert len(calls) == 1
        assert "INSERT INTO control_plane.audit_log" in calls[0]


# ── GovernanceGate ───────────────────────────────────────────────────


class TestGovernanceRequired:
    """GovernanceGate.{request_approval, approve, reject}: DB errors
    propagate to caller (were previously silently returning {}, False, False)."""

    def test_request_approval_propagates_dbunavailable(self, monkeypatch):
        from app.control_plane.db import DBUnavailable
        from app.control_plane.governance import GovernanceGate

        # governance imports execute_one_required from db; patch the
        # bound name in the consuming module so call-site dispatch
        # uses our stub. No need to also patch execute_required —
        # request_approval uses execute_one_required directly.
        def _fake_one(query, params=()):
            raise DBUnavailable("pool down")
        monkeypatch.setattr(
            "app.control_plane.governance.execute_one_required", _fake_one,
        )

        with pytest.raises(DBUnavailable):
            GovernanceGate().request_approval(
                "p1", "code_change", "user", "test",
            )

    def test_approve_propagates_dbunavailable(self, monkeypatch):
        from app.control_plane.db import DBUnavailable
        from app.control_plane.governance import GovernanceGate

        def _fake_one(query, params=()):
            raise DBUnavailable("pool down")
        monkeypatch.setattr(
            "app.control_plane.governance.execute_one_required", _fake_one,
        )

        with pytest.raises(DBUnavailable):
            GovernanceGate().approve("req-id")

    def test_approve_returns_false_for_legitimate_no_match(self, monkeypatch):
        """When the UPDATE matches no rows (request not pending), still
        returns False — NOT DBUnavailable. Only DB errors raise."""
        from app.control_plane.governance import GovernanceGate

        def _fake_one(query, params=()):
            return None  # No matching row
        monkeypatch.setattr(
            "app.control_plane.governance.execute_one_required", _fake_one,
        )

        result = GovernanceGate().approve("nonexistent-id")
        assert result is False

    def test_reject_propagates_dbunavailable(self, monkeypatch):
        from app.control_plane.db import DBUnavailable
        from app.control_plane.governance import GovernanceGate

        def _fake_one(query, params=()):
            raise DBUnavailable("pool down")
        monkeypatch.setattr(
            "app.control_plane.governance.execute_one_required", _fake_one,
        )

        with pytest.raises(DBUnavailable):
            GovernanceGate().reject("req-id")


# ── TicketManager ────────────────────────────────────────────────────


class TestTicketsRequired:
    """TicketManager state-transition writes propagate DB errors."""

    def test_create_from_signal_propagates_dbunavailable(self, monkeypatch):
        from app.control_plane.db import DBUnavailable
        from app.control_plane.tickets import TicketManager

        # The dedup-query (execute_one) returns None (no existing match);
        # then INSERT via execute_one_required raises.
        monkeypatch.setattr(
            "app.control_plane.tickets.execute_one",
            lambda q, p=(): None,
        )
        def _fake_required(query, params=()):
            raise DBUnavailable("pool down")
        monkeypatch.setattr(
            "app.control_plane.tickets.execute_one_required", _fake_required,
        )

        with pytest.raises(DBUnavailable):
            TicketManager().create_from_signal(
                "hello", sender="u", project_id="p",
            )

    def test_create_manual_propagates_sql_error(self, monkeypatch):
        from app.control_plane.tickets import TicketManager

        def _fake(query, params=()):
            raise psycopg2.Error("table missing")
        monkeypatch.setattr(
            "app.control_plane.tickets.execute_one_required", _fake,
        )

        with pytest.raises(psycopg2.Error):
            TicketManager().create_manual("title", "p1")

    def test_assign_to_crew_propagates_dbunavailable(self, monkeypatch):
        from app.control_plane.db import DBUnavailable
        from app.control_plane.tickets import TicketManager

        def _fake(query, params=(), fetch=False):
            raise DBUnavailable("pool down")
        monkeypatch.setattr(
            "app.control_plane.tickets.execute_required", _fake,
        )

        with pytest.raises(DBUnavailable):
            TicketManager().assign_to_crew("t1", "researcher", "researcher")

    def test_complete_propagates_dbunavailable(self, monkeypatch):
        from app.control_plane.db import DBUnavailable
        from app.control_plane.tickets import TicketManager

        # complete() does a separate SELECT to fetch project_id wrapped
        # in try/except — that's fine.
        monkeypatch.setattr(
            "app.control_plane.tickets.execute_one",
            lambda q, p=(): {"project_id": "p1"},
        )
        def _fake(query, params=(), fetch=False):
            raise DBUnavailable("pool down")
        monkeypatch.setattr(
            "app.control_plane.tickets.execute_required", _fake,
        )

        with pytest.raises(DBUnavailable):
            TicketManager().complete("t1", "result")

    def test_fail_propagates_dbunavailable(self, monkeypatch):
        from app.control_plane.db import DBUnavailable
        from app.control_plane.tickets import TicketManager

        def _fake(query, params=(), fetch=False):
            raise DBUnavailable("pool down")
        monkeypatch.setattr(
            "app.control_plane.tickets.execute_required", _fake,
        )

        with pytest.raises(DBUnavailable):
            TicketManager().fail("t1", "boom")

    def test_close_propagates_dbunavailable(self, monkeypatch):
        from app.control_plane.db import DBUnavailable
        from app.control_plane.tickets import TicketManager

        def _fake(query, params=(), fetch=False):
            raise DBUnavailable("pool down")
        monkeypatch.setattr(
            "app.control_plane.tickets.execute_required", _fake,
        )

        with pytest.raises(DBUnavailable):
            TicketManager().close("t1")


# ── error_monitor ────────────────────────────────────────────────────


class TestErrorMonitorRequired:
    """_record_anomaly logs at WARNING on DB failure (was DEBUG);
    acknowledge returns False on DB failure (was True)."""

    def test_record_anomaly_logs_at_warning_on_db_failure(self, monkeypatch, caplog):
        from app.control_plane.db import DBUnavailable

        # Inject the failure at app.control_plane.db so the in-function
        # import sees the patched symbol.
        from app.control_plane import db as _db
        def _boom(*a, **kw):
            raise DBUnavailable("pool down")
        monkeypatch.setattr(_db, "execute_required", _boom)

        with caplog.at_level(logging.WARNING, logger="app.observability.error_monitor"):
            from app.observability.error_monitor import _record_anomaly
            _record_anomaly(
                "sig", "sample", "spike",
                hourly_rate=100.0, baseline=10.0, severity="warning",
            )

        # The failure must surface at WARNING level — pre-PR-3 this was DEBUG.
        warning_records = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING
            and "anomaly insert failed" in r.message
        ]
        assert len(warning_records) >= 1

    def test_acknowledge_returns_false_on_db_failure(self, monkeypatch):
        from app.control_plane.db import DBUnavailable
        from app.control_plane import db as _db

        def _boom(*a, **kw):
            raise DBUnavailable("pool down")
        monkeypatch.setattr(_db, "execute_required", _boom)

        from app.observability.error_monitor import acknowledge
        # Pre-PR-3 this returned True because execute() silently
        # swallowed the failure; post-PR-3 it returns False as documented.
        assert acknowledge("some-anomaly-id") is False
